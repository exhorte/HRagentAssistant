"""Shared logging setup.

Every module asks this file for a logger instead of configuring its own.

Rules enforced by convention across the code base:
  * importing this module has NO side effect (no directory, no file);
  * logging is configured explicitly by entry points via ``configure_logging()``;
  * user text (questions, answers, document content) is NEVER logged - only
    ids, counts, statuses and durations. See ``fingerprint()`` for a safe
    correlation handle.
"""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import os
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def configure_logging(log_dir: str | os.PathLike | None = None, level: int = logging.INFO) -> None:
    """Configure console logging (and a rotating file if ``log_dir`` is given). Idempotent."""
    global _configured
    if _configured:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path / "hr_assistant.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
        )
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def fingerprint(text: str) -> str:
    """Short, non-reversible handle to correlate log lines without logging the text itself."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
