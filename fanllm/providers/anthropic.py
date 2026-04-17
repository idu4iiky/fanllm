from __future__ import annotations

import httpx

from fanllm.providers._base import get_api_key, raise_for_status, run_with_result
from fanllm.types import LLMResult

NAME = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-5"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MAX_TOKENS = 4096


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
        payload: dict = {
            "model": resolved_model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(BASE_URL, headers=headers, json=payload)
            raise_for_status(response, NAME)
            data = response.json()

        text = ""
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text") or ""

        usage = data.get("usage") or {}
        return text, usage.get("input_tokens"), usage.get("output_tokens")

    return await run_with_result(provider=NAME, model=resolved_model, fn=_do)
