from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx

from fanllm.errors import ProviderAuthError, ProviderRateLimitError


async def with_retry(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except ProviderAuthError:
            raise
        except ProviderRateLimitError as e:
            last_exc = e
        except httpx.HTTPStatusError as e:
            if 500 <= e.response.status_code < 600:
                last_exc = e
            else:
                raise
        if attempt < max_attempts - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
    assert last_exc is not None
    raise last_exc
