from __future__ import annotations

import os
import time
from typing import Awaitable, Callable

import httpx

from fanllm._retry import with_retry
from fanllm.errors import FanllmError, ProviderAuthError, ProviderRateLimitError
from fanllm.types import LLMResult


def get_api_key(env_var: str, provider_label: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise ProviderAuthError(f"{provider_label} API key missing ({env_var})")
    return key


async def run_with_result(
    *,
    provider: str,
    model: str,
    fn: Callable[[], Awaitable[tuple[str, int | None, int | None]]],
) -> LLMResult:
    start = time.perf_counter()
    try:
        text, input_tokens, output_tokens = await with_retry(fn)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResult(
            provider=provider,
            model=model,
            response=text,
            error=None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except ProviderAuthError as e:
        err = str(e)
    except ProviderRateLimitError as e:
        err = f"rate limited: {e}"
    except httpx.HTTPStatusError as e:
        err = f"HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        err = "timeout"
    except httpx.HTTPError as e:
        err = f"HTTP error: {e.__class__.__name__}"
    except FanllmError as e:
        err = str(e)
    except Exception as e:
        err = f"{e.__class__.__name__}: {e}"

    latency_ms = int((time.perf_counter() - start) * 1000)
    return LLMResult(
        provider=provider,
        model=model,
        response=None,
        error=err,
        latency_ms=latency_ms,
        input_tokens=None,
        output_tokens=None,
    )


def bearer_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def raise_for_status(response: httpx.Response, provider_label: str) -> None:
    status = response.status_code
    if status in (401, 403):
        raise ProviderAuthError(f"{provider_label} API key invalid or missing")
    if status == 429:
        raise ProviderRateLimitError(f"{provider_label} rate limited")
    response.raise_for_status()


async def openai_compatible_call(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str | None,
    timeout: float,
    provider_label: str,
) -> tuple[str, int | None, int | None]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            base_url,
            headers=bearer_headers(api_key),
            json=payload,
        )
        raise_for_status(response, provider_label)
        data = response.json()

    text = ""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""

    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    return text, input_tokens, output_tokens
