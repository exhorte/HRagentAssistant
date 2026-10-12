"""Step 3: turn text into numbers (vectors).

Two interchangeable backends behind LangChain's ``Embeddings`` interface:

* ``hashing`` (default): deterministic feature-hashing of stemmed terms. No
  network, no API key, no cost, same vectors on every run. It captures
  *lexical* similarity only (no synonyms), which is enough for the offline
  prototype and for deterministic tests.
* ``jina``: Jina AI embeddings (semantic, needs ``JINA_API_KEY``).

Vectors from different backends are not comparable, so each backend gets its
own Qdrant collection (see ``Settings.collection_name``).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings

from hr_assistant.config import Settings
from hr_assistant.logger import get_logger
from hr_assistant.textutils import content_terms

logger = get_logger(__name__)

HASHING_DIM = 1024


class HashingEmbeddings(Embeddings):
    """Bag-of-stems + bigrams, sub-linear TF, hashed into ``dim`` buckets, L2-normalised."""

    def __init__(self, dim: int = HASHING_DIM):
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        # The last dimension is reserved for "no content terms" so no vector is ever all-zero.
        return int.from_bytes(digest, "big") % (self.dim - 1)

    def _embed(self, text: str) -> list[float]:
        terms = content_terms(text)
        vector = [0.0] * self.dim
        if not terms:
            vector[self.dim - 1] = 1.0
            return vector
        for term, count in Counter(terms).items():
            vector[self._bucket(term)] += 1.0 + math.log(count)
        for left, right in zip(terms, terms[1:]):
            vector[self._bucket(f"{left}_{right}")] += 0.5
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class JinaEmbeddings(Embeddings):
    """Minimal Jina AI embeddings client (``langchain-community``'s one is being sunset).

    ``jina-embeddings-v3`` is multilingual and takes a ``task`` hint (``retrieval.passage`` for documents,
    ``retrieval.query`` for questions); older models ignore it.
    """

    URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: str, model_name: str, *, timeout: float = 30.0, batch_size: int = 64, client=None):
        self._api_key = api_key
        self.model_name = model_name
        self._timeout = timeout
        self._batch_size = batch_size
        self._client = client  # an ``httpx.Client`` (injected in tests)

    def _post(self, texts: list[str], task: str | None) -> list[list[float]]:
        import httpx  # imported lazily: only needed for this backend

        payload: dict = {"model": self.model_name, "input": texts}
        if task and self.model_name.startswith("jina-embeddings-v3"):
            payload["task"] = task
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        post = self._client.post if self._client is not None else httpx.post
        response = post(self.URL, headers=headers, json=payload, timeout=self._timeout)
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._post(texts[start : start + self._batch_size], "retrieval.passage"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "retrieval.query")[0]


def get_embeddings_model(settings: Settings) -> Embeddings:
    """Return the embeddings backend selected in the settings."""
    if settings.embedding_backend == "jina":
        logger.info("Embeddings backend: jina (model=%s)", settings.embedding_model)
        return JinaEmbeddings(settings.jina_api_key or "", settings.embedding_model)
    logger.info("Embeddings backend: hashing (offline, deterministic)")
    return HashingEmbeddings()
