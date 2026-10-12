"""Step 4: store chunk embeddings in Qdrant and search them *with access control*.

Qdrant runs in one of three modes (``QDRANT_MODE``): ``memory`` (tests),
``local`` (embedded, persisted under ``data/runtime/qdrant`` - no server, the
default) or ``cloud`` (the original Qdrant Cloud setup).

Security: the access filter (tenant, role, document status) is applied by the
vector database itself, **before** any text can reach the answer generator.
Nothing here relies on the LLM to hide documents.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from hr_assistant.config import Settings
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger
from hr_assistant.textutils import term_set

logger = get_logger(__name__)

# langchain-qdrant stores document metadata under this payload key.
_META = "metadata"


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved passage with its vector similarity score."""

    text: str
    score: float
    metadata: dict

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))


def _make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_mode == "memory":
        return QdrantClient(location=":memory:")
    if settings.qdrant_mode == "cloud":
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def access_filter(*, role: str, tenant_id: str, statuses: tuple[str, ...] = ("active",)) -> models.Filter:
    """Qdrant filter restricting results to what ``role`` may read in ``tenant_id``."""
    return models.Filter(
        must=[
            models.FieldCondition(key=f"{_META}.tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key=f"{_META}.allowed_roles", match=models.MatchValue(value=role)),
            models.FieldCondition(key=f"{_META}.status", match=models.MatchAny(any=list(statuses))),
        ]
    )


@dataclass(frozen=True)
class TermStats:
    """Document frequencies of stemmed terms over the passages a role may read.

    Used to weight a question's terms by rarity: a missing rare term ("jumeaux") should weigh more
    than a matched common one ("jour").
    """

    n_chunks: int
    df: dict

    def weight(self, term: str) -> float:
        # Smoothed IDF in [1, ~ln(2N)+1]; terms never seen in the corpus get the maximum weight.
        return math.log((self.n_chunks + 1) / (self.df.get(term, 0) + 0.5)) + 1.0


class KnowledgeIndex:
    """Vector index over the HR documents (one Qdrant collection per embedding configuration)."""

    def __init__(self, client: QdrantClient, collection: str, embeddings: Embeddings):
        self._client = client
        self.collection = collection
        self._embeddings = embeddings
        self._store: QdrantVectorStore | None = None
        self._version = 0  # bumped on every write; invalidates the term-statistics cache
        self._stats_cache: dict[tuple, tuple[int, TermStats]] = {}

    @classmethod
    def from_settings(cls, settings: Settings, embeddings: Embeddings | None = None) -> "KnowledgeIndex":
        return cls(
            _make_client(settings),
            settings.collection_name,
            embeddings or get_embeddings_model(settings),
        )

    def _ensure_store(self) -> QdrantVectorStore:
        if self._store is None:
            if not self._client.collection_exists(self.collection):
                size = len(self._embeddings.embed_query("dimension probe"))
                self._client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", self.collection, size)
            self._store = QdrantVectorStore(
                client=self._client, collection_name=self.collection, embedding=self._embeddings
            )
        return self._store

    # ----- write -----
    def add_chunks(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        self._ensure_store().add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        self._version += 1
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        self._ensure_store()
        self._version += 1
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key=f"{_META}.document_id", match=models.MatchValue(value=document_id))]
                )
            ),
        )

    # ----- read -----
    def search(
        self,
        query: str,
        *,
        role: str,
        tenant_id: str,
        k: int = 8,
        statuses: tuple[str, ...] = ("active",),
    ) -> list[ScoredChunk]:
        store = self._ensure_store()
        flt = access_filter(role=role, tenant_id=tenant_id, statuses=statuses)
        results = store.similarity_search_with_score(query, k=k, filter=flt)
        return [ScoredChunk(text=doc.page_content, score=float(score), metadata=dict(doc.metadata)) for doc, score in results]

    def term_stats(self, *, role: str, tenant_id: str, statuses: tuple[str, ...] = ("active",)) -> TermStats:
        """Document frequencies over the passages ``role`` may read (cached until the next write)."""
        key = (role, tenant_id, statuses)
        cached = self._stats_cache.get(key)
        if cached and cached[0] == self._version:
            return cached[1]
        self._ensure_store()
        flt = access_filter(role=role, tenant_id=tenant_id, statuses=statuses)
        df: Counter = Counter()
        total, offset = 0, None
        while True:
            records, offset = self._client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset,
                with_payload=["page_content"], with_vectors=False,
            )
            for record in records:
                total += 1
                df.update(term_set((record.payload or {}).get("page_content", "")))
            if offset is None:
                break
        stats = TermStats(total, dict(df))
        self._stats_cache[key] = (self._version, stats)
        return stats

    def count(self, document_id: str | None = None) -> int:
        self._ensure_store()
        flt = None
        if document_id:
            flt = models.Filter(
                must=[models.FieldCondition(key=f"{_META}.document_id", match=models.MatchValue(value=document_id))]
            )
        return self._client.count(self.collection, count_filter=flt, exact=True).count

    def close(self) -> None:
        self._client.close()
