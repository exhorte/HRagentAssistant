import ast
import sys
import tomllib
from pathlib import Path

import pytest

from hr_assistant import auth
from hr_assistant.config import Settings
from hr_assistant.db import Database

ROOT = Path(__file__).resolve().parents[2]


# ----- authentication
def test_password_hashing_is_salted_and_verifiable():
    h1, h2 = auth.hash_password("correct horse battery"), auth.hash_password("correct horse battery")
    assert h1 != h2 and h1.startswith("pbkdf2_sha256$")
    assert auth.verify_password("correct horse battery", h1)
    assert not auth.verify_password("wrong password", h1)
    assert not auth.verify_password("x", "garbage-hash")


def test_authenticate_gives_no_hint_on_failure(app):
    assert app.login("alice", "wrong-password-1") is None
    assert app.login("nobody", "whatever-password") is None
    assert app.login("", "") is None


def test_roles_and_permissions(app, alice, claire):
    assert alice.role == auth.EMPLOYEE and claire.role == auth.HR_MANAGER
    assert alice.can(auth.ASK) and not alice.can(auth.LEAVE_DECIDE) and not alice.can(auth.DOCUMENTS_MANAGE)
    assert claire.can(auth.LEAVE_DECIDE) and claire.can(auth.DOCUMENTS_MANAGE)
    with pytest.raises(auth.PermissionDenied):
        auth.require(alice, auth.LEAVE_DECIDE)
    with pytest.raises(auth.PermissionDenied):
        auth.require(None, auth.ASK)


def test_create_user_validates_input():
    db = Database()
    with pytest.raises(ValueError):
        auth.create_user(db, username="x", password="short", role="employee")
    with pytest.raises(ValueError):
        auth.create_user(db, username="x", password="long-enough-pw", role="superuser")
    db.close()


# ----- configuration
def test_defaults_are_offline_and_need_no_key():
    s = Settings()
    s.validate()
    assert s.llm_provider == "none" and s.embedding_backend == "hashing" and s.qdrant_mode == "local"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"llm_provider": "groq"}, "GROQ_API_KEY"),
        ({"llm_provider": "portkey"}, "PORTKEY_API_KEY"),
        ({"embedding_backend": "jina"}, "JINA_API_KEY"),
        ({"qdrant_mode": "cloud"}, "QDRANT_URL"),
        ({"chunk_overlap": 900}, "CHUNK_OVERLAP"),
    ],
)
def test_validation_names_the_missing_setting(overrides, message):
    with pytest.raises(ValueError, match=message):
        Settings(**overrides).validate()


def test_secrets_never_appear_in_repr():
    s = Settings(groq_api_key="gsk_SECRETVALUE123", jina_api_key="jina_SECRET", qdrant_api_key="qd-SECRET", portkey_api_key="pk-SECRET")
    text = repr(s)
    assert "SECRET" not in text


def test_from_env_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("HR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOP_K_RESULTS", "7")
    s = Settings.from_env()
    assert s.data_dir == tmp_path and s.top_k == 7
    monkeypatch.setenv("LLM_PROVIDER", "mystery")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_collection_name_depends_on_embedding_configuration():
    assert Settings().collection_name != Settings(embedding_backend="jina", jina_api_key="k").collection_name


# ----- dependencies: one source of truth, and every third-party import is declared
_IMPORT_TO_DISTRIBUTION = {
    "dotenv": "python-dotenv",
    "langchain_core": "langchain-core",
    "langchain_text_splitters": "langchain-text-splitters",
    "httpx": "httpx",
    "langchain_qdrant": "langchain-qdrant",
    "qdrant_client": "qdrant-client",
    "langchain_groq": "langchain-groq",
    "langchain_openai": "langchain-openai",
    "portkey_ai": "portkey-ai",
    "streamlit": "streamlit",
    "pypdf": "pypdf",
    "docx": "python-docx",
    "numpy": "numpy",
    "langsmith": "langsmith",
    "openevals": "openevals",
}


def _declared() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    specs = list(data["dependencies"])
    for extra in data.get("optional-dependencies", {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        for sep in (">", "<", "=", "[", ";", "~", "!"):
            spec = spec.split(sep)[0]
        names.add(spec.strip().lower())
    return names


def test_requirements_txt_only_points_to_pyproject():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    effective = [l for l in lines if l and not l.startswith("#")]
    assert effective == ["-e .[dev]"]


def test_every_third_party_import_is_declared_in_pyproject():
    declared = _declared()
    stdlib = set(sys.stdlib_module_names)
    missing: set[str] = set()
    sources = list((ROOT / "hr_assistant").glob("*.py")) + [ROOT / "app.py", ROOT / "main.py"]
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name in {"hr_assistant", "app", "main", "__future__"}:
                    continue
                distribution = _IMPORT_TO_DISTRIBUTION.get(name)
                if distribution is None or distribution not in declared:
                    missing.add(f"{name} ({path.name})")
    assert not missing, f"imports not covered by pyproject.toml: {sorted(missing)}"
