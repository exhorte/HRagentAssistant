"""All settings for the app live here, in one place.

Settings are an immutable dataclass built from environment variables
(``Settings.from_env()``, which also loads a local ``.env`` file). Everything
has a safe, offline, zero-cost default so the prototype runs without any API
key: extractive answers, local hashing embeddings, local Qdrant.

Secrets (API keys) are only ever read from the environment, never hard-coded,
and are excluded from ``repr`` so they cannot leak into logs by accident.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_PROVIDERS = ("none", "groq", "portkey")
EMBEDDING_BACKENDS = ("hashing", "jina")
QDRANT_MODES = ("memory", "local", "cloud")

# Kept for backward compatibility with the original single-document demo / notebook.
DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.getenv(name) or default).strip().lower()
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got '{raw}'.")
    return raw


@dataclass(frozen=True)
class Settings:
    # --- tenancy (schema is multi-tenant ready; only one tenant is implemented) ---
    tenant_id: str = "default"

    # --- paths ---
    data_dir: Path = BASE_DIR / "data" / "runtime"
    samples_dir: Path = BASE_DIR / "data" / "samples"
    special_samples_dir: Path = BASE_DIR / "data" / "samples_special"

    # --- LLM (optional: "none" = deterministic extractive answers) ---
    llm_provider: str = "none"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str | None = field(default=None, repr=False)
    portkey_api_key: str | None = field(default=None, repr=False)
    portkey_provider: str = "@hrpolicy"
    guard_enabled: bool = False
    guard_model: str = "openai/gpt-oss-safeguard-20b"

    # --- embeddings ---
    embedding_backend: str = "hashing"
    embedding_model: str = "jina-embeddings-v2-base-en"
    jina_api_key: str | None = field(default=None, repr=False)

    # --- vector store (Qdrant) ---
    qdrant_mode: str = "local"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = "hr_documents"

    # --- chunking / retrieval ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    candidate_k: int = 8
    min_score: float = 0.12
    min_coverage: float = 0.5

    # --- limits ---
    max_upload_mb: int = 10
    max_question_chars: int = 1000

    # ----- derived -----
    @property
    def db_path(self) -> Path:
        return self.data_dir / "hr_assistant.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def collection_name(self) -> str:
        """One collection per embedding configuration (vector sizes/spaces differ)."""
        if self.embedding_backend == "jina":
            slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.lower()).strip("-")
        else:
            slug = "hashing-v1"
        return f"{self.qdrant_collection}__{slug}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **changes)

    # ----- construction -----
    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = os.getenv("HR_DATA_DIR")
        defaults = cls()
        return cls(
            tenant_id=os.getenv("TENANT_ID", defaults.tenant_id),
            data_dir=Path(data_dir) if data_dir else defaults.data_dir,
            llm_provider=_env_choice("LLM_PROVIDER", "none", LLM_PROVIDERS),
            llm_model=os.getenv("LLM_MODEL_NAME", defaults.llm_model),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            portkey_api_key=os.getenv("PORTKEY_API_KEY") or None,
            portkey_provider=os.getenv("PORTKEY_PROVIDER", defaults.portkey_provider),
            guard_enabled=_env_bool("GUARD_LLM_ENABLED", False),
            guard_model=os.getenv("GUARD_MODEL_NAME", defaults.guard_model),
            embedding_backend=_env_choice("EMBEDDING_BACKEND", "hashing", EMBEDDING_BACKENDS),
            embedding_model=os.getenv("EMBEDDING_MODEL_NAME", defaults.embedding_model),
            jina_api_key=os.getenv("JINA_API_KEY") or None,
            qdrant_mode=_env_choice("QDRANT_MODE", "local", QDRANT_MODES),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION_NAME", defaults.qdrant_collection),
            chunk_size=_env_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            top_k=_env_int("TOP_K_RESULTS", defaults.top_k),
            candidate_k=_env_int("CANDIDATE_K", defaults.candidate_k),
            min_score=_env_float("MIN_SCORE", defaults.min_score),
            min_coverage=_env_float("MIN_COVERAGE", defaults.min_coverage),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", defaults.max_upload_mb),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", defaults.max_question_chars),
        )

    # ----- validation -----
    def validate(self) -> None:
        """Fail early, with an actionable message, if the chosen providers are not usable.

        Only the providers that were *selected* need their credentials, so the
        default (offline) configuration always validates.
        """
        problems: list[str] = []
        if self.llm_provider == "groq" and not self.groq_api_key:
            problems.append("LLM_PROVIDER=groq requires GROQ_API_KEY.")
        if self.llm_provider == "portkey" and not self.portkey_api_key:
            problems.append("LLM_PROVIDER=portkey requires PORTKEY_API_KEY.")
        if self.embedding_backend == "jina" and not self.jina_api_key:
            problems.append("EMBEDDING_BACKEND=jina requires JINA_API_KEY.")
        if self.qdrant_mode == "cloud" and not (self.qdrant_url and self.qdrant_api_key):
            problems.append("QDRANT_MODE=cloud requires QDRANT_URL and QDRANT_API_KEY.")
        if self.guard_enabled and not self.groq_api_key:
            problems.append("GUARD_LLM_ENABLED=true requires GROQ_API_KEY.")
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if problems:
            raise ValueError(" ".join(problems))


def get_settings() -> Settings:
    """Settings built from the current environment (re-read on every call)."""
    return Settings.from_env()
