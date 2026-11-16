"""Profiles: named sets of settings (production, staging, dev)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .profiles import PROFILES
from .retry import RetryPolicy


@dataclass
class Profile:
    name: str
    bank_feed_url: str
    ledger_dsn: str
    http_timeout: float = 10.0
    retries: dict = field(default_factory=dict)

    def retry_policy(self, service: str) -> RetryPolicy:
        """The retry policy for calls to ``service``.

        Uses the service's own entry under ``retries`` when there is one, otherwise the ``default`` entry. Entries are
        not combined: a setting the chosen entry leaves out gets RetryPolicy's default value.
        """
        settings = self.retries.get(service, self.retries.get("default", {}))
        return RetryPolicy(**settings)


def load_profile(name: str) -> Profile:
    """The ``base`` settings with the named profile's settings on top.

    Each top-level key the profile sets replaces the base value as a whole.
    """
    if name == "base" or name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}")
    settings = {**PROFILES["base"], **PROFILES[name]}
    return Profile(name=name, **settings)
