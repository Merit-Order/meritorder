"""Client for the bank's statement feed."""

from __future__ import annotations

import json
import socket
import urllib.request
from datetime import date


class BankFeed:
    """Fetches statement lines from the feed.

    Each call makes exactly one HTTP request; retrying is up to the caller (the jobs wrap ``fetch_statement`` with a
    RetryPolicy). A request that takes longer than ``timeout`` seconds raises TimeoutError.
    """

    def __init__(self, url: str, timeout: float):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def fetch_statement(self, day: date) -> list[dict]:
        try:
            with urllib.request.urlopen(f"{self.url}/statements/{day.isoformat()}", timeout=self.timeout) as r:
                return json.load(r)["lines"]
        except socket.timeout as e:
            raise TimeoutError(f"bank feed timed out after {self.timeout}s") from e
