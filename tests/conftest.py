"""Shared fixtures. Everything runs offline: in-memory Qdrant, hashing embeddings, no LLM, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `import app` / `import main` from tests without installing the project
    sys.path.insert(0, str(ROOT))

from hr_assistant import auth as _auth  # noqa: E402

# PBKDF2 at its production cost (200k iterations) makes every fixture seconds-slow; the hash format stores
# the iteration count, so verification stays correct. test_password_hashing covers the format.
_auth._PBKDF2_ITERATIONS = 1_000

from hr_assistant.config import Settings  # noqa: E402
from hr_assistant.demo import import_sample_dir, seed_users  # noqa: E402
from hr_assistant.pipeline import build_app  # noqa: E402

PASSWORDS = {"alice": "alice-test-pw-1", "bob": "bob-test-pw-123", "claire": "claire-test-pw-1"}


class RecordingLLM:
    """Scripted chat model that also records every prompt it receives."""

    class _Reply:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, reply: str | dict | Exception):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        content = self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)
        return self._Reply(content)

    @property
    def prompts(self) -> str:
        return "\n".join(m["content"] for call in self.calls for m in call)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        qdrant_mode="memory",
        llm_provider="none",
        embedding_backend="hashing",
    )


@pytest.fixture
def app(settings):
    application = build_app(settings, llm=None)
    seed_users(application, PASSWORDS)
    yield application
    application.close()


@pytest.fixture
def alice(app):
    return app.login("alice", PASSWORDS["alice"])


@pytest.fixture
def bob(app):
    return app.login("bob", PASSWORDS["bob"])


@pytest.fixture
def claire(app):
    return app.login("claire", PASSWORDS["claire"])


@pytest.fixture
def loaded_app(app, claire, settings):
    """App with the main synthetic corpus imported (6 documents)."""
    report = import_sample_dir(app, claire, settings.samples_dir)
    assert not report.failed, report.failed
    return app


@pytest.fixture
def full_app(loaded_app, claire, settings):
    """Main corpus + special corpus (obsolete version, conflicting note, injection bait)."""
    report = import_sample_dir(loaded_app, claire, settings.special_samples_dir)
    assert not report.failed, report.failed
    return loaded_app
