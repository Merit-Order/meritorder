"""v1 importer: reads the bank's old CSV export."""

from __future__ import annotations

import csv
import io
import time

from .backoff import backoff_delays


def import_csv(read, retries: int = 3) -> list[dict]:
    """``read()`` returns the CSV text; it is retried on TimeoutError with backoff_delays()."""
    for delay in backoff_delays(retries):
        try:
            text = read()
            break
        except TimeoutError:
            time.sleep(delay)
    else:
        text = read()
    return [{"reference": row["ref"], "amount_cents": int(row["amount"])} for row in csv.DictReader(io.StringIO(text))]
