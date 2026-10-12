"""Step 6: connect to the LLM (the "brain" of the assistant) - optional.

``LLM_PROVIDER`` selects the provider:

* ``none`` (default): no model. The assistant returns *extractive* answers
  (verbatim sentences from the retrieved documents, with their sources).
* ``groq``: Groq directly (``GROQ_API_KEY``).
* ``portkey``: Groq through the Portkey gateway (``PORTKEY_API_KEY``).

The model only *phrases* an answer from passages the backend already
authorised; it decides neither access rights nor workflow status.
"""

from __future__ import annotations

from hr_assistant.config import Settings
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def get_llm(settings: Settings):
    """Return a chat model for the configured provider, or ``None`` for extractive mode."""
    if settings.llm_provider == "none":
        logger.info("LLM provider: none (extractive answers)")
        return None
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq  # lazy: only when selected

        logger.info("LLM provider: groq (model=%s)", settings.llm_model)
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    if settings.llm_provider == "portkey":
        from hr_assistant.gateway import get_gateway_llm

        return get_gateway_llm(settings)
    raise ValueError(f"Unknown LLM provider '{settings.llm_provider}'.")
