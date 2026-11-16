"""The reconciliation jobs."""

from __future__ import annotations

from datetime import date, timedelta

from .bankfeed import BankFeed
from .config import load_profile
from .ledger import Ledger
from .matching import Report, match_lines
from .retry import with_retries


def nightly(profile_name: str = "production") -> Report:
    """Reconcile yesterday's statement against the ledger and store the report."""
    return _run(date.today() - timedelta(days=1), profile_name, store=True)


def check(day: str, profile_name: str = "dev") -> Report:
    """Reconcile one day without storing anything (for looking into a report)."""
    return _run(date.fromisoformat(day), profile_name, store=False)


def _run(day: date, profile_name: str, store: bool) -> Report:
    profile = load_profile(profile_name)
    feed = BankFeed(profile.bank_feed_url, timeout=profile.http_timeout)
    fetch = with_retries(feed.fetch_statement, profile.retry_policy("bankfeed"))
    ledger = Ledger(profile.ledger_dsn, retry=profile.retry_policy("ledger"))
    report = match_lines(fetch(day), ledger.entries(day))
    if store:
        ledger.store_report(day, report)
    return report
