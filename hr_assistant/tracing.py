"""Step 0b: LangSmith tracing (optional).

LangSmith tracing needs no wiring in our own code - LangChain looks for
LANGSMITH_TRACING / LANGSMITH_ENDPOINT / LANGSMITH_API_KEY / LANGSMITH_PROJECT
directly in the environment (loaded from .env by ``config``) and, if tracing
is turned on, sends a trace of every LLM call to your LangSmith project.

Privacy note: traces contain prompts and retrieved HR passages. Keep tracing
OFF (the default) for anything but synthetic demo data.

This module doesn't turn tracing on - the env vars already do that. All it
does is log, once per run, whether tracing is active.
"""

from __future__ import annotations

import os

from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def check_langsmith_tracing() -> None:
    """Log whether LangSmith tracing is enabled for this run."""
    if tracing_enabled():
        logger.warning(
            "LangSmith tracing ENABLED (project '%s'): prompts and HR passages are sent to LangSmith.",
            os.getenv("LANGSMITH_PROJECT", "default"),
        )
    else:
        logger.info("LangSmith tracing is OFF (set LANGSMITH_TRACING=true in .env to enable)")
