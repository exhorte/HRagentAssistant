"""Step 6b: route the LLM through the Portkey gateway (optional provider).

Instead of calling Groq directly, the LLM call goes through Portkey. Portkey
stores the real Groq credentials behind a "slug" (set up once in the Portkey
dashboard) - our code never sees the raw Groq key.

NOTE on fallback: a Portkey "config" (strategy: fallback + targets) is not
sent. The original workspace has "block_inline_config" enabled, which rejects
any ``x-portkey-config`` header, so we route straight to one provider via
``x-portkey-provider``. The trade-off: no automatic Portkey-side fallback to a
second slug.

Selected with ``LLM_PROVIDER=portkey`` (needs ``PORTKEY_API_KEY``; the slug is
``PORTKEY_PROVIDER``, default ``@hrpolicy``).
"""

from __future__ import annotations

from hr_assistant.config import Settings
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# Default slug of the Groq integration configured in the Portkey dashboard.
PRIMARY_PROVIDER = "@hrpolicy"


def get_gateway_llm(settings: Settings):
    """Return a chat model routed through Portkey (no config/fallback - see module docstring)."""
    # Lazy imports: portkey-ai / langchain-openai are only needed for this provider.
    from langchain_openai import ChatOpenAI
    from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

    logger.info("Routing LLM calls through Portkey (provider=%s)", settings.portkey_provider)
    headers = createHeaders(api_key=settings.portkey_api_key, provider=settings.portkey_provider)
    return ChatOpenAI(
        api_key="portkey",  # dummy value - the real auth is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        model=settings.llm_model,
        default_headers=headers,
        temperature=0,
        timeout=30,
        max_retries=1,
    )
