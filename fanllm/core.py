from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from fanllm.providers import REGISTRY, available_providers
from fanllm.types import LLMResult

load_dotenv()


async def run(
    prompt: str,
    *,
    providers: list[str] | None = None,
    system_prompt: str | None = None,
    models: dict[str, str] | None = None,
    timeout: float = 90.0,
    max_concurrency: int = 10,
) -> list[LLMResult]:
    selected = providers if providers is not None else available_providers()
    models = models or {}

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(name: str) -> LLMResult:
        module = REGISTRY.get(name)
        if module is None:
            return LLMResult(
                provider=name,
                model="",
                response=None,
                error=f"unknown provider: {name}",
                latency_ms=0,
                input_tokens=None,
                output_tokens=None,
            )
        async with semaphore:
            return await module.call(
                prompt,
                system_prompt=system_prompt,
                model=models.get(name),
                timeout=timeout,
            )

    tasks = [_run_one(name) for name in selected]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[LLMResult] = []
    for name, item in zip(selected, raw):
        if isinstance(item, LLMResult):
            results.append(item)
        elif isinstance(item, BaseException):
            model = REGISTRY[name].DEFAULT_MODEL if name in REGISTRY else ""
            results.append(
                LLMResult(
                    provider=name,
                    model=models.get(name, model),
                    response=None,
                    error=f"{item.__class__.__name__}: {item}",
                    latency_ms=0,
                    input_tokens=None,
                    output_tokens=None,
                )
            )

    results.sort(key=lambda r: r.provider)
    return results


__all__ = ["run"]
