from __future__ import annotations

import httpx

from fanllm.providers._base import get_api_key, raise_for_status, run_with_result
from fanllm.types import LLMResult

NAME = "google"
DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
BASE_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


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
        url = BASE_URL_TEMPLATE.format(model=resolved_model)
        payload: dict = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
            ],
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_prompt}],
            }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            raise_for_status(response, NAME)
            data = response.json()

        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text") or "" for p in parts)

        usage = data.get("usageMetadata") or {}
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")

        return text, input_tokens, output_tokens

    return await run_with_result(provider=NAME, model=resolved_model, fn=_do)
