"""Empty the browser tests' database before the gateway starts, so every run begins from the YAML file.

Refuses any database whose name doesn't contain "e2e" or "test": this drops every table.
"""

import asyncio
import os
import sys
from urllib.parse import urlparse

import asyncpg


async def main() -> None:
    url = os.environ.get("MERITORDER_DATABASE_URL", "")
    name = urlparse(url).path.lstrip("/")
    if not url or not any(s in name for s in ("e2e", "test")):
        sys.exit(f"refusing to reset database {name!r}: its name must contain 'e2e' or 'test'")
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    finally:
        await conn.close()
    print(f"reset database {name}")


if __name__ == "__main__":
    asyncio.run(main())
