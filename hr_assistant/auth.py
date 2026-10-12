"""Authentication and role-based access control.

This is a *prototype-grade* local authentication: users live in the SQLite
database, passwords are salted + hashed with PBKDF2-HMAC-SHA256 and compared in
constant time. It is NOT SSO, has no lockout/rate limiting and no TLS - see
docs/SECURITY.md before putting any real HR data behind it.

Authorisation is enforced here, in the backend, never by the LLM and never by
the UI alone: every service function takes an authenticated ``User`` and calls
``require(user, permission)``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from hr_assistant.db import Database

EMPLOYEE = "employee"
HR_MANAGER = "hr_manager"
ROLES = (EMPLOYEE, HR_MANAGER)

# Permissions
ASK = "ask"
LEAVE_CREATE = "leave:create"
LEAVE_VIEW_ALL = "leave:view_all"
LEAVE_DECIDE = "leave:decide"
ESCALATION_CREATE = "escalation:create"
ESCALATION_VIEW_ALL = "escalation:view_all"
ESCALATION_HANDLE = "escalation:handle"
DOCUMENTS_MANAGE = "documents:manage"
DOCUMENTS_LIST = "documents:list"

_EMPLOYEE_PERMISSIONS = frozenset({ASK, LEAVE_CREATE, ESCALATION_CREATE, DOCUMENTS_LIST})
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    EMPLOYEE: _EMPLOYEE_PERMISSIONS,
    HR_MANAGER: _EMPLOYEE_PERMISSIONS
    | {LEAVE_VIEW_ALL, LEAVE_DECIDE, ESCALATION_VIEW_ALL, ESCALATION_HANDLE, DOCUMENTS_MANAGE},
}

ACCESS_ALL = "all"
ACCESS_HR_ONLY = "hr_only"
_ACCESS_TO_ROLES = {
    ACCESS_ALL: [EMPLOYEE, HR_MANAGER],
    ACCESS_HR_ONLY: [HR_MANAGER],
}

_PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8


class PermissionDenied(Exception):
    """The authenticated user is not allowed to perform this action."""


@dataclass(frozen=True)
class User:
    id: str
    username: str
    display_name: str
    role: str
    tenant_id: str

    def can(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, frozenset())


def require(user: User | None, permission: str) -> None:
    """Raise ``PermissionDenied`` unless ``user`` holds ``permission``."""
    if user is None or not user.can(permission):
        raise PermissionDenied(f"Permission '{permission}' required.")


def roles_for_access_level(access_level: str) -> list[str]:
    """Roles allowed to read a document with the given access level."""
    try:
        return list(_ACCESS_TO_ROLES[access_level])
    except KeyError as exc:
        raise ValueError(f"Unknown access level '{access_level}'.") from exc


# ----- passwords -----

def hash_password(password: str, *, iterations: int | None = None) -> str:
    iterations = iterations or _PBKDF2_ITERATIONS  # read at call time (tests lower it for speed)
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


# ----- user store -----

def _row_to_user(row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        role=row["role"],
        tenant_id=row["tenant_id"],
    )


def create_user(
    db: Database,
    *,
    username: str,
    password: str,
    role: str,
    display_name: str | None = None,
    tenant_id: str = "default",
) -> User:
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'.")
    username = username.strip().lower()
    if not username:
        raise ValueError("Username is required.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    user_id = uuid.uuid4().hex
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO users (id, tenant_id, username, display_name, role, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                tenant_id,
                username,
                display_name or username,
                role,
                hash_password(password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return User(user_id, username, display_name or username, role, tenant_id)


def get_user(db: Database, user_id: str) -> User | None:
    row = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
    return _row_to_user(row) if row else None


def get_user_by_username(db: Database, username: str) -> User | None:
    row = db.query_one("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    return _row_to_user(row) if row else None


# A fixed dummy hash so unknown usernames cost the same time as a wrong password.
_DUMMY_HASH = hash_password("not-a-real-password", iterations=_PBKDF2_ITERATIONS)


def authenticate(db: Database, username: str, password: str) -> User | None:
    """Return the user if the credentials are valid, else ``None`` (no hint on which part failed)."""
    row = db.query_one("SELECT * FROM users WHERE username = ?", ((username or "").strip().lower(),))
    stored = row["password_hash"] if row else _DUMMY_HASH
    ok = verify_password(password or "", stored)
    return _row_to_user(row) if (row and ok) else None
