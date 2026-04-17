from __future__ import annotations

import os
from types import ModuleType

from fanllm.providers import (
    anthropic,
    deepseek,
    google,
    mistral,
    openai,
    perplexity,
    xai,
)

REGISTRY: dict[str, ModuleType] = {
    anthropic.NAME: anthropic,
    deepseek.NAME: deepseek,
    google.NAME: google,
    mistral.NAME: mistral,
    openai.NAME: openai,
    perplexity.NAME: perplexity,
    xai.NAME: xai,
}


def available_providers() -> list[str]:
    return sorted(
        name for name, mod in REGISTRY.items() if os.environ.get(mod.API_KEY_ENV_VAR)
    )
