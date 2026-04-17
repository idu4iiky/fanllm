from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    response: str | None
    error: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
