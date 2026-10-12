"""Document registry and ingestion: validate → extract → split → index, with explicit status.

The SQLite ``documents`` table is the source of truth (name, type, version,
language, status, indexing outcome); Qdrant only holds the searchable passages
derived from it. Importing/removing/re-indexing requires the HR role.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hr_assistant import auth
from hr_assistant.auth import User
from hr_assistant.config import Settings
from hr_assistant.db import Database
from hr_assistant.document_loader import (
    DocumentError,
    extract_text_from_bytes,
    sanitize_filename,
    validate_upload,
)
from hr_assistant.logger import get_logger
from hr_assistant.splitter import split_into_chunks
from hr_assistant.textutils import detect_language
from hr_assistant.vector_store import KnowledgeIndex

logger = get_logger(__name__)

STATUSES = ("active", "obsolete", "draft")


class DuplicateDocumentError(DocumentError):
    """The same file content is already registered."""

    def __init__(self, existing: "DocumentRecord"):
        super().__init__(f"Ce document est déjà importé sous le nom « {existing.name} ».")
        self.existing = existing


class DocumentNotFound(DocumentError):
    pass


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    tenant_id: str
    name: str
    doc_type: str
    sha256: str
    size_bytes: int
    language: str | None
    version: str | None
    family: str | None
    status: str
    access_level: str
    is_demo: bool
    imported_at: str
    imported_by: str | None
    indexing_status: str
    indexing_error: str | None
    chunk_count: int
    page_count: int
    collection: str | None


def _record(row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        tenant_id=row["tenant_id"],
        name=row["name"],
        doc_type=row["doc_type"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        language=row["language"],
        version=row["version"],
        family=row["family"],
        status=row["status"],
        access_level=row["access_level"],
        is_demo=bool(row["is_demo"]),
        imported_at=row["imported_at"],
        imported_by=row["imported_by"],
        indexing_status=row["indexing_status"],
        indexing_error=row["indexing_error"],
        chunk_count=row["chunk_count"],
        page_count=row["page_count"],
        collection=row["collection"],
    )


class DocumentService:
    def __init__(self, db: Database, index: KnowledgeIndex, settings: Settings):
        self._db = db
        self._index = index
        self._settings = settings

    # ----- read -----
    def get(self, user: User, document_id: str) -> DocumentRecord:
        auth.require(user, auth.DOCUMENTS_LIST)
        row = self._db.query_one(
            "SELECT * FROM documents WHERE id = ? AND tenant_id = ?", (document_id, user.tenant_id)
        )
        if row is None:
            raise DocumentNotFound("Document introuvable.")
        record = _record(row)
        if record.access_level == auth.ACCESS_HR_ONLY and user.role != auth.HR_MANAGER:
            # Same answer as for a missing document: do not reveal that it exists.
            raise DocumentNotFound("Document introuvable.")
        return record

    def list_documents(self, user: User) -> list[DocumentRecord]:
        """Documents visible to ``user`` (HR-only documents are invisible to employees)."""
        auth.require(user, auth.DOCUMENTS_LIST)
        rows = self._db.query(
            "SELECT * FROM documents WHERE tenant_id = ? ORDER BY imported_at DESC, name", (user.tenant_id,)
        )
        records = [_record(r) for r in rows]
        if user.role != auth.HR_MANAGER:
            records = [r for r in records if r.access_level == auth.ACCESS_ALL]
        return records

    # ----- write -----
    def import_bytes(
        self,
        actor: User,
        filename: str,
        data: bytes,
        *,
        version: str | None = None,
        family: str | None = None,
        status: str = "active",
        access_level: str = auth.ACCESS_ALL,
        is_demo: bool = False,
    ) -> DocumentRecord:
        """Register and index a document. The returned record carries the indexing outcome.

        Raises ``DocumentError`` subclasses (user-safe messages) for invalid or duplicate files;
        extraction/indexing failures are *recorded* on the document (``indexing_status='failed'``)
        and also raised so the caller can show the reason.
        """
        auth.require(actor, auth.DOCUMENTS_MANAGE)
        if status not in STATUSES:
            raise DocumentError(f"Statut inconnu : {status}.")
        if access_level not in (auth.ACCESS_ALL, auth.ACCESS_HR_ONLY):
            raise DocumentError(f"Niveau d'accès inconnu : {access_level}.")
        name = sanitize_filename(filename)
        # Type, size and signature are checked before anything is written to disk.
        doc_type = validate_upload(name, data, self._settings.max_upload_bytes)
        sha = hashlib.sha256(data).hexdigest()

        existing = self._db.query_one(
            "SELECT * FROM documents WHERE tenant_id = ? AND sha256 = ?", (actor.tenant_id, sha)
        )
        if existing is not None:
            if existing["indexing_status"] != "failed":
                raise DuplicateDocumentError(_record(existing))
            self._remove_row_and_file(existing["id"], existing["stored_path"])  # retry a failed import

        document_id = uuid.uuid4().hex
        self._settings.upload_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self._settings.upload_dir / f"{document_id}{Path(name).suffix.lower()}"
        stored_path.write_bytes(data)
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO documents (id, tenant_id, name, doc_type, sha256, size_bytes, version, family, status, "
                "access_level, is_demo, imported_at, imported_by, stored_path, indexing_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    document_id, actor.tenant_id, name, doc_type, sha, len(data), version or None, family or None,
                    status, access_level, int(is_demo), now, actor.id, str(stored_path),
                ),
            )
        self._index_document(document_id, name, data)
        return self.get(actor, document_id)

    def _index_document(self, document_id: str, name: str, data: bytes) -> None:
        """Extract + split + embed + upsert. Always leaves a definite ``indexing_status`` behind."""
        row = self._db.query_one("SELECT * FROM documents WHERE id = ?", (document_id,))
        try:
            extracted = extract_text_from_bytes(name, data, self._settings.max_upload_bytes)
            language = extracted.language
            chunks = split_into_chunks(
                extracted,
                self._chunk_metadata(row, language),
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            if not chunks:
                raise DocumentError("Aucun passage exploitable n'a pu être extrait du document.")
            self._index.delete_document(document_id)  # idempotent re-index
            self._index.add_chunks(chunks)
        except DocumentError as exc:
            self._mark_failed(document_id, str(exc))
            raise
        except Exception as exc:  # embedding/vector store/network failure: never leak details to the UI
            logger.exception("Indexing failed for document %s", document_id[:8])
            try:
                self._index.delete_document(document_id)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            message = f"Échec de l'indexation ({type(exc).__name__}). Consultez les journaux techniques."
            self._mark_failed(document_id, message)
            raise DocumentError(message) from exc
        page_count = len([p for p in extracted.pages if p.text])
        self._db.execute(
            "UPDATE documents SET indexing_status = 'indexed', indexing_error = NULL, chunk_count = ?, "
            "page_count = ?, language = ?, collection = ? WHERE id = ?",
            (len(chunks), page_count, language, self._index.collection, document_id),
        )
        logger.info("Indexed document %s: %d chunk(s)", document_id[:8], len(chunks))

    @staticmethod
    def _chunk_metadata(row, language: str) -> dict:
        return {
            "document_id": row["id"],
            "document_name": row["name"],
            "document_type": row["doc_type"],
            "source": row["name"],
            "language": language,
            "version": row["version"],
            "family": row["family"],
            "status": row["status"],
            "access_level": row["access_level"],
            "allowed_roles": auth.roles_for_access_level(row["access_level"]),
            "tenant_id": row["tenant_id"],
            "is_demo": bool(row["is_demo"]),
        }

    def _mark_failed(self, document_id: str, message: str) -> None:
        self._db.execute(
            "UPDATE documents SET indexing_status = 'failed', indexing_error = ?, chunk_count = 0, collection = NULL "
            "WHERE id = ?",
            (message[:500], document_id),
        )

    def _remove_row_and_file(self, document_id: str, stored_path: str | None) -> None:
        try:
            self._index.delete_document(document_id)
        except Exception:  # pragma: no cover - index may not exist yet
            logger.warning("Could not delete vectors of %s", document_id[:8])
        self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        if stored_path:
            Path(stored_path).unlink(missing_ok=True)

    def remove(self, actor: User, document_id: str) -> None:
        auth.require(actor, auth.DOCUMENTS_MANAGE)
        row = self._db.query_one(
            "SELECT * FROM documents WHERE id = ? AND tenant_id = ?", (document_id, actor.tenant_id)
        )
        if row is None:
            raise DocumentNotFound("Document introuvable.")
        self._remove_row_and_file(document_id, row["stored_path"])
        logger.info("Removed document %s", document_id[:8])

    def reindex(self, actor: User, document_id: str) -> DocumentRecord:
        auth.require(actor, auth.DOCUMENTS_MANAGE)
        row = self._db.query_one(
            "SELECT * FROM documents WHERE id = ? AND tenant_id = ?", (document_id, actor.tenant_id)
        )
        if row is None:
            raise DocumentNotFound("Document introuvable.")
        path = Path(row["stored_path"] or "")
        if not path.is_file():
            self._mark_failed(document_id, "Le fichier source n'est plus disponible : réimportez le document.")
            raise DocumentError("Le fichier source n'est plus disponible : réimportez le document.")
        self._db.execute("UPDATE documents SET indexing_status = 'pending' WHERE id = ?", (document_id,))
        self._index_document(document_id, row["name"], path.read_bytes())
        return self.get(actor, document_id)

    def update_metadata(
        self,
        actor: User,
        document_id: str,
        *,
        status: str | None = None,
        access_level: str | None = None,
        version: str | None = None,
        family: str | None = None,
    ) -> DocumentRecord:
        """Change status/access/version/family, then re-index so the payloads match the registry."""
        auth.require(actor, auth.DOCUMENTS_MANAGE)
        if status is not None and status not in STATUSES:
            raise DocumentError(f"Statut inconnu : {status}.")
        if access_level is not None and access_level not in (auth.ACCESS_ALL, auth.ACCESS_HR_ONLY):
            raise DocumentError(f"Niveau d'accès inconnu : {access_level}.")
        current = self.get(actor, document_id)
        self._db.execute(
            "UPDATE documents SET status = ?, access_level = ?, version = ?, family = ? WHERE id = ?",
            (
                status or current.status,
                access_level or current.access_level,
                version if version is not None else current.version,
                family if family is not None else current.family,
                document_id,
            ),
        )
        return self.reindex(actor, document_id)

    def reindex_all(self, actor: User) -> tuple[int, int]:
        """Re-index every document (e.g. after changing the embedding backend). Returns (ok, failed)."""
        auth.require(actor, auth.DOCUMENTS_MANAGE)
        ok = failed = 0
        for record in self.list_documents(actor):
            try:
                self.reindex(actor, record.id)
                ok += 1
            except DocumentError:
                failed += 1
        return ok, failed
