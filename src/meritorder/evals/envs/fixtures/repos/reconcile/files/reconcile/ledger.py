"""Our side of the books."""

from __future__ import annotations

from datetime import date

from .retry import RetryPolicy, with_retries


class Ledger:
    def __init__(self, dsn: str, retry: RetryPolicy | None = None):
        self.dsn = dsn
        self.retry = retry or RetryPolicy(max_retries=0)

    def entries(self, day: date) -> list[dict]:
        """Ledger entries booked on ``day``."""
        return with_retries(self._query, self.retry)("select * from entries where booked_on = %s", day)

    def store_report(self, day: date, report) -> None:
        with_retries(self._execute, self.retry)("insert into reports (day, body) values (%s, %s)", day, report.summary())

    def _query(self, sql: str, *params):
        raise NotImplementedError("needs a database driver")

    def _execute(self, sql: str, *params):
        raise NotImplementedError("needs a database driver")
