from __future__ import annotations


class FanllmError(Exception):
    pass


class ProviderAuthError(FanllmError):
    pass


class ProviderRateLimitError(FanllmError):
    pass
