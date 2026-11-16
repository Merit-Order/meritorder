"""Backoff for the v1 importer."""


def backoff_delays(retries: int = 3, first: float = 1.0) -> list[float]:
    """1 s, 2 s, 4 s, ...: the wait doubles before each retry."""
    return [first * 2**i for i in range(retries)]
