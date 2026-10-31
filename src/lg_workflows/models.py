"""Chat models used by graph nodes.

- local_llm: Qwen on the LAN box. Unmetered but single-slot, so keep concurrency low.
- brain_llm: metered models behind the LiteLLM gateway, rate-limited in-process.
"""

import os

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

load_dotenv()

_gateway_limiter = InMemoryRateLimiter(requests_per_second=0.5, max_bucket_size=3)


def local_llm(thinking: bool = True, **kwargs) -> ChatOpenAI:
    """`thinking=False` skips Qwen's reasoning phase: ~4x faster, fine for labeling/extraction."""
    if not thinking:
        # "low" is not safe here: it made the model loop and emit 30x the expected labels.
        kwargs.setdefault("reasoning_effort", "none")
    kwargs.setdefault("timeout", 180)
    # The server keeps working on abandoned requests, so retries pile up in its queue.
    kwargs.setdefault("max_retries", 1)
    kwargs.setdefault("temperature", 0.2)
    return ChatOpenAI(
        base_url=os.environ["LOCAL_BASE_URL"],
        api_key=os.environ["LOCAL_API_KEY"],
        model=os.environ.get("LOCAL_MODEL", "Qwen3.8-Flash-Next"),
        **kwargs,
    )


def brain_llm(model: str | None = None, **kwargs) -> ChatOpenAI:
    model = model or os.environ.get("BRAIN_MODEL", "gpt-6.1-sol")
    return ChatOpenAI(
        base_url=os.environ["GATEWAY_BASE_URL"],
        api_key=os.environ["GATEWAY_API_KEY"],
        model=model,
        # gpt-6.x on this gateway only supports function tools via /v1/responses.
        use_responses_api=model.startswith("gpt-"),
        rate_limiter=_gateway_limiter,
        timeout=180,
        max_retries=2,
        **kwargs,
    )
