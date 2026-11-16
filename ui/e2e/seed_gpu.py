"""An hour of GPU time for the browser tests, as the capacity controller accrues it while replicas run: gpu-static up for
the current hour at its $4.00, serving none of the seeded requests. The console then counts it in spend (decision 64).

Refuses any database whose name doesn't contain "e2e" or "test", as reset_db.py does.
"""

import asyncio
import os
import sys
from datetime import UTC, datetime
from urllib.parse import urlparse

from meritorder.store import PostgresStore


async def main() -> None:
    url = os.environ.get("MERITORDER_DATABASE_URL", "")
    name = urlparse(url).path.lstrip("/")
    if not url or not any(s in name for s in ("e2e", "test")):
        sys.exit(f"refusing to write to database {name!r}: its name must contain 'e2e' or 'test'")
    store = PostgresStore(url)
    await store.init()
    try:
        await store.add_gpu_usage("gpu-static", datetime.now(UTC), 3600, 4.0)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
