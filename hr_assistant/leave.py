"""Leave-request workflow: a small, explicit state machine persisted in SQLite.

The model never decides anything here. Every transition is a deliberate
backend call by an authenticated user with the right role:

    draft ──submit (owner, explicit confirmation)──▶ submitted
    submitted ──start_review (HR)──▶ under_review
    under_review ──decide (HR, not the requester)──▶ approved | rejected
    draft | submitted | under_review | approved ──cancel (owner)──▶ cancelled

Anything else is refused. No leave balance is computed: there is no reliable
data source (no HRIS connected), so the recap says so instead of guessing.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from hr_assistant import auth
from hr_assistant.auth import PermissionDenied, User
from hr_assistant.db import Database
from hr_assistant.logger import get_logger
from hr_assistant.textutils import strip_accents

logger = get_logger(__name__)

REQUEST_TYPES = {
    "annual_leave": "Congé annuel",
    "sick_leave": "Congé maladie",
    "unpaid_leave": "Congé sans solde",
    "family_event": "Événement familial",
    "other": "Autre absence",
}
REQUIRED_FIELDS = ("request_type", "start_date", "end_date")
FIELD_LABELS = {
    "request_type": "type de demande",
    "start_date": "date de début",
    "end_date": "date de fin",
    "reason": "motif",
}

DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, CANCELLED = (
    "draft", "submitted", "under_review", "approved", "rejected", "cancelled",
)
STATUSES = (DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, CANCELLED)
STATUS_LABELS = {
    DRAFT: "Brouillon",
    SUBMITTED: "Soumise",
    UNDER_REVIEW: "En cours d'examen",
    APPROVED: "Approuvée",
    REJECTED: "Refusée",
    CANCELLED: "Annulée",
}
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    DRAFT: frozenset({SUBMITTED, CANCELLED}),
    SUBMITTED: frozenset({UNDER_REVIEW, CANCELLED}),
    UNDER_REVIEW: frozenset({APPROVED, REJECTED, CANCELLED}),
    APPROVED: frozenset({CANCELLED}),
    REJECTED: frozenset(),
    CANCELLED: frozenset(),
}
# Statuses that "occupy" the calendar (used for the overlap check).
_BLOCKING = (SUBMITTED, UNDER_REVIEW, APPROVED)
MAX_REASON_CHARS = 500
MAX_DURATION_DAYS = 366

BALANCE_NOTICE = "Solde de congés : non disponible (aucun système RH connecté à ce prototype)."


class LeaveError(Exception):
    """Base class; messages are user-safe."""


class LeaveValidationError(LeaveError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{FIELD_LABELS.get(k, k)} : {v}" for k, v in errors.items()))
        self.errors = errors


class InvalidTransition(LeaveError):
    pass


class ConfirmationRequired(LeaveError):
    pass


class LeaveNotFound(LeaveError):
    pass


@dataclass(frozen=True)
class LeaveRequest:
    id: str
    tenant_id: str
    user_id: str
    request_type: str
    start_date: date
    end_date: date
    reason: str | None
    status: str
    created_at: str
    updated_at: str
    submitted_at: str | None
    decided_by: str | None
    decided_at: str | None
    decision_comment: str | None

    @property
    def calendar_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def type_label(self) -> str:
        return REQUEST_TYPES.get(self.request_type, self.request_type)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


@dataclass(frozen=True)
class LeaveEvent:
    at: str
    actor_id: str
    from_status: str | None
    to_status: str
    comment: str | None


@dataclass(frozen=True)
class LeavePreparation:
    """Result of checking a (possibly incomplete) request before anything is saved."""

    missing: list[str]
    errors: dict[str, str]
    recap: str | None

    @property
    def ready(self) -> bool:
        return not self.missing and not self.errors


def parse_date(value) -> date | None:
    """Accept ``date``, ``YYYY-MM-DD`` or ``DD/MM/YYYY``; ``None`` for empty input."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"date invalide « {text} » (formats acceptés : AAAA-MM-JJ ou JJ/MM/AAAA)")


_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_FR_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_TYPE_KEYWORDS = (
    ("sick_leave", ("maladie", "arret de travail", "arret maladie")),
    ("unpaid_leave", ("sans solde",)),
    ("family_event", ("mariage", "pacs", "naissance", "adoption", "deces", "evenement familial")),
    ("annual_leave", ("conge", "vacances", "rtt", "repos")),
)


def extract_leave_fields(text: str) -> dict:
    """Best-effort extraction of leave fields (dates, type) from a chat message.

    Only unambiguous values are returned (ISO dates or DD/MM/YYYY, in order of appearance: first =
    start, second = end; and a type keyword). Anything else stays missing and is asked for explicitly.
    """
    fields: dict = {}
    dates: list[tuple[int, date]] = []
    for regex in (_ISO_DATE_RE, _FR_DATE_RE):
        for match in regex.finditer(text or ""):
            try:
                dates.append((match.start(), parse_date(match.group(1))))
            except ValueError:
                continue
    dates.sort(key=lambda item: item[0])
    if dates:
        fields["start_date"] = dates[0][1]
    if len(dates) > 1:
        fields["end_date"] = dates[1][1]
    folded = strip_accents(text or "").lower()
    for request_type, keywords in _TYPE_KEYWORDS:
        if any(k in folded for k in keywords):
            fields["request_type"] = request_type
            break
    return fields


def _row(row) -> LeaveRequest:
    return LeaveRequest(
        id=row["id"], tenant_id=row["tenant_id"], user_id=row["user_id"], request_type=row["request_type"],
        start_date=date.fromisoformat(row["start_date"]), end_date=date.fromisoformat(row["end_date"]),
        reason=row["reason"], status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
        submitted_at=row["submitted_at"], decided_by=row["decided_by"], decided_at=row["decided_at"],
        decision_comment=row["decision_comment"],
    )


class LeaveService:
    def __init__(self, db: Database, today: Callable[[], date] | None = None):
        self._db = db
        self._today = today or date.today

    # ------------------------------------------------------------------ validation
    def _validate(self, fields: dict) -> tuple[dict, list[str], dict[str, str]]:
        """Return (cleaned values, missing field names, per-field errors)."""
        cleaned: dict = {}
        missing: list[str] = []
        errors: dict[str, str] = {}

        request_type = (fields.get("request_type") or "").strip()
        if not request_type:
            missing.append("request_type")
        elif request_type not in REQUEST_TYPES:
            errors["request_type"] = "type inconnu"
        else:
            cleaned["request_type"] = request_type

        for name in ("start_date", "end_date"):
            raw = fields.get(name)
            if raw in (None, ""):
                missing.append(name)
                continue
            try:
                cleaned[name] = parse_date(raw)
            except ValueError as exc:
                errors[name] = str(exc)

        reason = (fields.get("reason") or "").strip()
        if len(reason) > MAX_REASON_CHARS:
            errors["reason"] = f"{MAX_REASON_CHARS} caractères maximum"
        cleaned["reason"] = reason or None
        if cleaned.get("request_type") == "other" and not reason:
            missing.append("reason")

        start, end = cleaned.get("start_date"), cleaned.get("end_date")
        if start and end:
            if end < start:
                errors["end_date"] = "la date de fin précède la date de début"
            elif (end - start).days + 1 > MAX_DURATION_DAYS:
                errors["end_date"] = f"durée supérieure à {MAX_DURATION_DAYS} jours"
        if start and "start_date" not in errors and cleaned.get("request_type") not in (None, "sick_leave"):
            if start < self._today():
                errors["start_date"] = "la date de début est dans le passé"
        return cleaned, missing, errors

    def prepare(self, user: User, fields: dict) -> LeavePreparation:
        """Check an incomplete/complete request and build the recap shown before confirmation."""
        auth.require(user, auth.LEAVE_CREATE)
        cleaned, missing, errors = self._validate(fields)
        recap = None
        if not missing and not errors:
            recap = self.recap_text(cleaned)
        return LeavePreparation(missing=missing, errors=errors, recap=recap)

    @staticmethod
    def recap_text(values: dict | LeaveRequest) -> str:
        get = values.__getitem__ if isinstance(values, dict) else (lambda k: getattr(values, k))
        days = (get("end_date") - get("start_date")).days + 1
        lines = [
            f"Type : {REQUEST_TYPES.get(get('request_type'), get('request_type'))}",
            f"Du {get('start_date').strftime('%d/%m/%Y')} au {get('end_date').strftime('%d/%m/%Y')} "
            f"({days} jour{'s' if days > 1 else ''} calendaire{'s' if days > 1 else ''}, jours ouvrés non vérifiés)",
        ]
        reason = get("reason")
        if reason:
            lines.append(f"Motif : {reason}")
        lines.append(BALANCE_NOTICE)
        return "\n".join(lines)

    def _overlaps(self, user_id: str, tenant_id: str, start: date, end: date, exclude_id: str | None = None) -> bool:
        marks = ",".join("?" for _ in _BLOCKING)
        row = self._db.query_one(
            f"SELECT 1 FROM leave_requests WHERE tenant_id = ? AND user_id = ? AND status IN ({marks}) "
            "AND start_date <= ? AND end_date >= ? AND id != ? LIMIT 1",
            (tenant_id, user_id, *_BLOCKING, end.isoformat(), start.isoformat(), exclude_id or ""),
        )
        return row is not None

    # ------------------------------------------------------------------ create / edit
    def create_draft(self, user: User, *, request_type=None, start_date=None, end_date=None, reason=None) -> LeaveRequest:
        auth.require(user, auth.LEAVE_CREATE)
        cleaned, missing, errors = self._validate(
            {"request_type": request_type, "start_date": start_date, "end_date": end_date, "reason": reason}
        )
        for name in missing:
            errors.setdefault(name, "champ obligatoire")
        if errors:
            raise LeaveValidationError(errors)
        now = datetime.now(timezone.utc).isoformat()
        request_id = uuid.uuid4().hex
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO leave_requests (id, tenant_id, user_id, request_type, start_date, end_date, reason, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, user.tenant_id, user.id, cleaned["request_type"], cleaned["start_date"].isoformat(),
                 cleaned["end_date"].isoformat(), cleaned["reason"], DRAFT, now, now),
            )
            conn.execute(
                "INSERT INTO leave_events (request_id, at, actor_id, from_status, to_status, comment) VALUES (?, ?, ?, NULL, ?, NULL)",
                (request_id, now, user.id, DRAFT),
            )
        logger.info("Leave draft %s created by %s", request_id[:8], user.id[:8])
        return self.get(user, request_id)

    def update_draft(self, user: User, request_id: str, **changes) -> LeaveRequest:
        current = self._owned(user, request_id)
        if current.status != DRAFT:
            raise InvalidTransition("Seul un brouillon peut être modifié.")
        merged = {
            "request_type": current.request_type, "start_date": current.start_date,
            "end_date": current.end_date, "reason": current.reason,
        }
        merged.update({k: v for k, v in changes.items() if k in merged})
        cleaned, missing, errors = self._validate(merged)
        for name in missing:
            errors.setdefault(name, "champ obligatoire")
        if errors:
            raise LeaveValidationError(errors)
        self._db.execute(
            "UPDATE leave_requests SET request_type = ?, start_date = ?, end_date = ?, reason = ?, updated_at = ? WHERE id = ?",
            (cleaned["request_type"], cleaned["start_date"].isoformat(), cleaned["end_date"].isoformat(),
             cleaned["reason"], datetime.now(timezone.utc).isoformat(), request_id),
        )
        return self.get(user, request_id)

    # ------------------------------------------------------------------ read
    def get(self, user: User, request_id: str) -> LeaveRequest:
        row = self._db.query_one(
            "SELECT * FROM leave_requests WHERE id = ? AND tenant_id = ?", (request_id, user.tenant_id)
        )
        if row is None:
            raise LeaveNotFound("Demande introuvable.")
        request = _row(row)
        if request.user_id != user.id and not user.can(auth.LEAVE_VIEW_ALL):
            raise LeaveNotFound("Demande introuvable.")  # do not reveal other users' request ids
        return request

    def list_for_user(self, user: User) -> list[LeaveRequest]:
        rows = self._db.query(
            "SELECT * FROM leave_requests WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC",
            (user.tenant_id, user.id),
        )
        return [_row(r) for r in rows]

    def list_all(self, user: User, status: str | None = None) -> list[LeaveRequest]:
        auth.require(user, auth.LEAVE_VIEW_ALL)
        sql, params = "SELECT * FROM leave_requests WHERE tenant_id = ?", [user.tenant_id]
        if status:
            sql, params = sql + " AND status = ?", [*params, status]
        rows = self._db.query(sql + " ORDER BY created_at DESC", tuple(params))
        return [_row(r) for r in rows]

    def history(self, user: User, request_id: str) -> list[LeaveEvent]:
        self.get(user, request_id)  # access check
        rows = self._db.query("SELECT * FROM leave_events WHERE request_id = ? ORDER BY id", (request_id,))
        return [LeaveEvent(r["at"], r["actor_id"], r["from_status"], r["to_status"], r["comment"]) for r in rows]

    # ------------------------------------------------------------------ transitions
    def _owned(self, user: User, request_id: str) -> LeaveRequest:
        request = self.get(user, request_id)
        if request.user_id != user.id:
            raise PermissionDenied("Seul l'auteur de la demande peut effectuer cette action.")
        return request

    def _transition(self, request: LeaveRequest, actor: User, to_status: str, comment: str | None = None, **extra) -> LeaveRequest:
        if to_status not in ALLOWED_TRANSITIONS[request.status]:
            raise InvalidTransition(
                f"Transition interdite : {STATUS_LABELS[request.status]} → {STATUS_LABELS[to_status]}."
            )
        now = datetime.now(timezone.utc).isoformat()
        columns = {"status": to_status, "updated_at": now, **extra}
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with self._db.transaction() as conn:
            # The WHERE on the previous status makes the transition atomic against concurrent changes.
            cursor = conn.execute(
                f"UPDATE leave_requests SET {assignments} WHERE id = ? AND status = ?",
                (*columns.values(), request.id, request.status),
            )
            if cursor.rowcount != 1:
                raise InvalidTransition("La demande a été modifiée entre-temps ; rechargez la page.")
            conn.execute(
                "INSERT INTO leave_events (request_id, at, actor_id, from_status, to_status, comment) VALUES (?, ?, ?, ?, ?, ?)",
                (request.id, now, actor.id, request.status, to_status, comment),
            )
        logger.info("Leave %s: %s -> %s by %s", request.id[:8], request.status, to_status, actor.id[:8])
        return self.get(actor, request.id)

    def submit(self, user: User, request_id: str, *, confirmed: bool) -> LeaveRequest:
        """Owner submits a draft. ``confirmed`` must be True: the UI shows the recap first."""
        request = self._owned(user, request_id)
        auth.require(user, auth.LEAVE_CREATE)
        if not confirmed:
            raise ConfirmationRequired("La soumission doit être confirmée explicitement.")
        if request.status != DRAFT:
            raise InvalidTransition(
                f"Transition interdite : {STATUS_LABELS[request.status]} → {STATUS_LABELS[SUBMITTED]}."
            )
        if self._overlaps(user.id, user.tenant_id, request.start_date, request.end_date, exclude_id=request.id):
            raise LeaveValidationError({"start_date": "cette période chevauche une autre demande en cours ou approuvée"})
        now = datetime.now(timezone.utc).isoformat()
        return self._transition(request, user, SUBMITTED, submitted_at=now)

    def start_review(self, hr_user: User, request_id: str) -> LeaveRequest:
        auth.require(hr_user, auth.LEAVE_DECIDE)
        request = self.get(hr_user, request_id)
        if request.user_id == hr_user.id:
            raise PermissionDenied("Vous ne pouvez pas traiter votre propre demande.")
        return self._transition(request, hr_user, UNDER_REVIEW)

    def decide(self, hr_user: User, request_id: str, *, approve: bool, comment: str | None = None) -> LeaveRequest:
        """Only a HR manager, only on a request under review, never on their own request."""
        auth.require(hr_user, auth.LEAVE_DECIDE)
        request = self.get(hr_user, request_id)
        if request.user_id == hr_user.id:
            raise PermissionDenied("Vous ne pouvez pas décider de votre propre demande.")
        comment = (comment or "").strip() or None
        if not approve and not comment:
            raise LeaveValidationError({"decision_comment": "un motif est obligatoire pour refuser une demande"})
        now = datetime.now(timezone.utc).isoformat()
        return self._transition(
            request, hr_user, APPROVED if approve else REJECTED, comment,
            decided_by=hr_user.id, decided_at=now, decision_comment=comment,
        )

    def cancel(self, user: User, request_id: str, comment: str | None = None) -> LeaveRequest:
        request = self._owned(user, request_id)
        return self._transition(request, user, CANCELLED, (comment or "").strip() or None)
