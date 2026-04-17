from __future__ import annotations

from fanllm.providers._base import (
    get_api_key,
    openai_compatible_call,
    run_with_result,
)
from fanllm.types import LLMResult

NAME = "perplexity"
DEFAULT_MODEL = "sonar"
API_KEY_ENV_VAR = "PERPLEXITY_API_KEY"
BASE_URL = "https://api.perplexity.ai/chat/completions"


async def call(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    timeout: float = 90.0,
) -> LLMResult:
    resolved_model = model or DEFAULT_MODEL

    async def _do() -> tuple[str, int | None, int | None]:
        api_key = get_api_key(API_KEY_ENV_VAR, NAME)
        return await openai_compatible_call(
            base_url=BASE_URL,
            api_key=api_key,
            model=resolved_model,
            prompt=prompt,
            system_prompt=system_prompt,
            timeout=timeout,
            provider_label=NAME,
        )

    return await run_with_result(provider=NAME, model=resolved_model, fn=_do)
