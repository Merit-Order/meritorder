"""Retrying calls that fail with transient errors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")
TRANSIENT = (TimeoutError, ConnectionError)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to retry a failed call, and how long to wait before each retry.

    ``max_retries`` counts the retries after the first attempt. The wait before retry n (n = 1, 2, ...) is
    ``base_delay * factor ** (n - 1)`` seconds, but never more than ``max_delay``.
    """

    max_retries: int = 2
    base_delay: float = 1.0
    factor: float = 3.0
    max_delay: float = 60.0

    def delays(self) -> list[float]:
        """The wait before each retry, in order."""
        return [min(self.base_delay * self.factor ** n, self.max_delay) for n in range(self.max_retries)]


def with_retries(fn: Callable[..., T], policy: RetryPolicy, sleep: Callable[[float], None] = time.sleep) -> Callable[..., T]:
    """Wrap ``fn`` so that transient errors are retried as ``policy`` says; the last error is raised."""

    def wrapper(*args, **kwargs) -> T:
        for delay in policy.delays():
            try:
                return fn(*args, **kwargs)
            except TRANSIENT:
                sleep(delay)
        return fn(*args, **kwargs)

    return wrapper
