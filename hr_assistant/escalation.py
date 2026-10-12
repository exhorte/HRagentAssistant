"""Hand-off to HR: prepare, confirm and track a follow-up request.

In this prototype a hand-off is *recorded and visible* to HR managers in the
application. **No e-mail or notification is sent** (no sending service is
configured) and the UI says so explicitly. The summary is built by
deterministic code from the question and the retrieval outcome - no model is
involved, so it cannot invent context.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from hr_assistant import auth
from hr_assistant.auth import PermissionDenied, User
from hr_assistant.db import Database
from hr_assistant.logger import get_logger

if TYPE_CHECKING:  # avoid a runtime import cycle
    from hr_assistant.answering import Answer

logger = get_logger(__name__)

OPEN, IN_PROGRESS, RESOLVED = "open", "in_progress", "resolved"
STATUS_LABELS = {OPEN: "Ouverte", IN_PROGRESS: "En cours", RESOLVED: "Résolue"}
MAX_QUESTION_CHARS = 1000

NO_NOTIFICATION_NOTICE = (
    "Aucune notification (e-mail, message) n'est envoyée par ce prototype : la demande est enregistrée "
    "et visible dans la file des responsables RH de l'application."
)

_REASONS = {
    "insufficient_evidence": "Les documents disponibles ne contiennent pas d'information suffisante pour répondre de manière fiable.",
    "conflicting_sources": "Plusieurs documents actifs donnent des informations différentes ; l'arbitrage relève des RH.",
    "partial_coverage": "Les documents ne couvrent qu'une partie de la question.",
    "model_insufficient_evidence": "Les extraits retrouvés ne suffisent pas pour répondre de manière fiable.",
    "user_request": "L'utilisateur souhaite l'avis d'un responsable RH.",
}


class EscalationNotFound(Exception):
    pass


@dataclass(frozen=True)
class EscalationDraft:
    question: str
    summary: str
    reason_code: str
    reason: str
    context: dict = field(default_factory=dict)

    def with_note(self, note: str | None) -> "EscalationDraft":
        """Same draft plus a free-text clarification from the employee (no-op if empty)."""
        note = (note or "").strip()[:500]
        if not note:
            return self
        return EscalationDraft(
            self.question,
            f"{self.summary}\nPrécision de l'employé : {note}",
            self.reason_code,
            self.reason,
            {**self.context, "user_note": note},
        )


@dataclass(frozen=True)
class Escalation:
    id: str
    user_id: str
    question: str
    summary: str
    reason: str
    context: dict
    status: str
    created_at: str
    updated_at: str
    handled_by: str | None
    resolution_note: str | None

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


def _row(row) -> Escalation:
    return Escalation(
        id=row["id"], user_id=row["user_id"], question=row["question"], summary=row["summary"],
        reason=row["reason"], context=json.loads(row["context_json"]), status=row["status"],
        created_at=row["created_at"], updated_at=row["updated_at"], handled_by=row["handled_by"],
        resolution_note=row["resolution_note"],
    )


class EscalationService:
    def __init__(self, db: Database):
        self._db = db

    def prepare(self, user: User, question: str, answer: "Answer | None" = None, note: str | None = None) -> EscalationDraft:
        """Build the hand-off summary. Nothing is stored until ``submit``."""
        auth.require(user, auth.ESCALATION_CREATE)
        question = (question or "").strip()
        if not question:
            raise ValueError("Une question est nécessaire pour préparer un transfert.")
        question = question[:MAX_QUESTION_CHARS]
        reason_code = (answer.reason if answer and answer.reason in _REASONS else None) or "user_request"
        context: dict = {"reason_code": reason_code}
        lines = [f"Question de l'employé : {question}", f"Pourquoi une intervention humaine : {_REASONS[reason_code]}"]
        if answer is not None:
            names = sorted({c.document_name for c in answer.citations} | {r.document_name for r in answer.retrieved if r.document_name})
            context["documents_consulted"] = names
            context["answer_status"] = answer.status
            if answer.missing_terms:
                context["missing_terms"] = answer.missing_terms
                lines.append("Termes de la question non retrouvés dans les documents : " + ", ".join(answer.missing_terms))
            if names:
                lines.append("Documents consultés : " + ", ".join(names))
            else:
                lines.append("Aucun document pertinent n'a été retrouvé.")
        note = (note or "").strip()
        if note:
            lines.append(f"Précision de l'employé : {note[:500]}")
            context["user_note"] = note[:500]
        return EscalationDraft(question, "\n".join(lines), reason_code, _REASONS[reason_code], context)

    def submit(self, user: User, draft: EscalationDraft, *, confirmed: bool) -> Escalation:
        """Persist the hand-off (status ``open``). Requires explicit confirmation; sends nothing."""
        auth.require(user, auth.ESCALATION_CREATE)
        if not confirmed:
            raise ValueError("Le transfert doit être confirmé explicitement.")
        now = datetime.now(timezone.utc).isoformat()
        escalation_id = uuid.uuid4().hex
        self._db.execute(
            "INSERT INTO escalations (id, tenant_id, user_id, question, summary, reason, context_json, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (escalation_id, user.tenant_id, user.id, draft.question, draft.summary, draft.reason,
             json.dumps(draft.context, ensure_ascii=False), now, now),
        )
        logger.info("Escalation %s recorded for user %s (%s)", escalation_id[:8], user.id[:8], draft.reason_code)
        return self.get(user, escalation_id)

    def get(self, user: User, escalation_id: str) -> Escalation:
        row = self._db.query_one(
            "SELECT * FROM escalations WHERE id = ? AND tenant_id = ?", (escalation_id, user.tenant_id)
        )
        if row is None:
            raise EscalationNotFound("Transfert introuvable.")
        item = _row(row)
        if item.user_id != user.id and not user.can(auth.ESCALATION_VIEW_ALL):
            raise EscalationNotFound("Transfert introuvable.")
        return item

    def list_for_user(self, user: User) -> list[Escalation]:
        rows = self._db.query(
            "SELECT * FROM escalations WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC",
            (user.tenant_id, user.id),
        )
        return [_row(r) for r in rows]

    def list_all(self, user: User, status: str | None = None) -> list[Escalation]:
        auth.require(user, auth.ESCALATION_VIEW_ALL)
        sql, params = "SELECT * FROM escalations WHERE tenant_id = ?", [user.tenant_id]
        if status:
            sql, params = sql + " AND status = ?", [*params, status]
        return [_row(r) for r in self._db.query(sql + " ORDER BY created_at DESC", tuple(params))]

    def take(self, hr_user: User, escalation_id: str) -> Escalation:
        auth.require(hr_user, auth.ESCALATION_HANDLE)
        current = self.get(hr_user, escalation_id)
        if current.status != OPEN:
            raise ValueError("Seul un transfert ouvert peut être pris en charge.")
        self._db.execute(
            "UPDATE escalations SET status = 'in_progress', handled_by = ?, updated_at = ? WHERE id = ?",
            (hr_user.id, datetime.now(timezone.utc).isoformat(), escalation_id),
        )
        return self.get(hr_user, escalation_id)

    def resolve(self, hr_user: User, escalation_id: str, note: str) -> Escalation:
        auth.require(hr_user, auth.ESCALATION_HANDLE)
        note = (note or "").strip()
        if not note:
            raise ValueError("Une note de résolution est obligatoire.")
        current = self.get(hr_user, escalation_id)
        if current.status == RESOLVED:
            raise ValueError("Ce transfert est déjà résolu.")
        self._db.execute(
            "UPDATE escalations SET status = 'resolved', handled_by = ?, resolution_note = ?, updated_at = ? WHERE id = ?",
            (hr_user.id, note[:1000], datetime.now(timezone.utc).isoformat(), escalation_id),
        )
        return self.get(hr_user, escalation_id)
