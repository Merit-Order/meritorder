"""Data analysis over a read-only SQLite database, graded on one exact number.

A dataset is fixtures/sql/<name>.sql (schema with documenting comments, then the rows) plus <name>.yaml (the answer and
tolerance). The model gets list_tables, describe_table (the CREATE statement, comments included) and run_sql. The
database lives in memory, opened read-only (``PRAGMA query_only`` and an authorizer that allows nothing but reading);
a progress handler aborts queries that run too long, and results are capped in rows and characters.

The final answer must end with ``ANSWER: <number>``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

import yaml

from . import Env, tool

DATASETS = Path(__file__).with_name("fixtures") / "sql"
MAX_ROWS = 100
MAX_CELL = 200
MAX_RESULT = 12_000
MAX_STEPS = 20_000_000   # SQLite VM instructions per query
MAX_SECONDS = 3.0
PROGRESS_EVERY = 1_000
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
_DENIED_FUNCTIONS = {"load_extension"}
_READ_PRAGMAS = {"table_info", "table_xinfo", "table_list", "index_list", "index_info", "index_xinfo", "foreign_key_list"}
_ANSWER = re.compile(r"ANSWER\s*[:=]\s*(?:EUR|€|\$|USD)?\s*(-?[\d][\d.,' \u00a0]*)", re.IGNORECASE)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    t = tool(name, description, properties)
    t["function"]["parameters"]["required"] = list(required)
    return t


def _err(message: str) -> str:
    return json.dumps({"error": message})


def parse_amount(text: str) -> float | None:
    """'12,535.90', '12535.9', '12 535,90', '12.535,90' -> 12535.9. None if it isn't a number."""
    s = re.sub(r"[\s'\u00a0]", "", text).rstrip(".,")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", "") if len(tail) == 3 and head else head.replace(",", "") + "." + tail
    try:
        return float(s)
    except ValueError:
        return None


class SqlEnv(Env):
    grading = "The final answer's 'ANSWER: <number>' must match the figure computed from the data"  # shown on the console's Tasks tab
    tools = [
        _tool("list_tables", "The database's tables and how many rows each has.", {}, []),
        _tool("describe_table", "A table's columns and its full CREATE TABLE statement, including the comments that document it.",
              {"table": {"type": "string"}}, ["table"]),
        _tool("run_sql", f"Run one read-only SQL query (SQLite dialect) and return the result: at most {MAX_ROWS} rows, so "
                         "aggregate in SQL rather than fetching raw rows.",
              {"query": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement"}}, ["query"]),
    ]

    def __init__(self, sandbox=None, dataset: str = "billing", **args):
        super().__init__(sandbox, dataset=dataset, **args)
        if not re.fullmatch(r"[a-z0-9_-]+", str(dataset)) or not (DATASETS / f"{dataset}.sql").exists():
            raise ValueError(f"unknown dataset {dataset!r}")
        meta = yaml.safe_load((DATASETS / f"{dataset}.yaml").read_text(encoding="utf-8"))
        self.answer, self.tolerance = float(meta["answer"]), float(meta.get("tolerance", 0.005))
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.executescript((DATASETS / f"{dataset}.sql").read_text(encoding="utf-8"))
        self.tables = {}
        for name, sql in self.db.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"):
            columns = [{"name": c[1], "type": c[2], "not_null": bool(c[3]), "primary_key": bool(c[5])}
                       for c in self.db.execute(f"PRAGMA table_info({name})")]
            rows = self.db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            self.tables[name] = {"columns": columns, "sql": sql, "rows": rows}
        self.db.execute("PRAGMA query_only = ON")
        self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        self.db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50_000)
        self.db.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        self.db.set_authorizer(self._authorize)
        self.db.set_progress_handler(self._progress, PROGRESS_EVERY)
        self._steps, self._deadline = 0, 0.0

    @staticmethod
    def _authorize(action: int, arg1, arg2, db_name, source) -> int:
        """Reading only: SELECTs, reads, functions (not load_extension) and the schema-inspecting pragmas."""
        if action == sqlite3.SQLITE_PRAGMA:
            return sqlite3.SQLITE_OK if str(arg1).lower() in _READ_PRAGMAS else sqlite3.SQLITE_DENY
        if action not in _ALLOWED or (action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in _DENIED_FUNCTIONS):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _progress(self) -> int:
        self._steps += PROGRESS_EVERY
        return int(self._steps > MAX_STEPS or time.monotonic() > self._deadline)

    def call(self, name: str, arguments: dict) -> str:
        if name == "list_tables":
            return json.dumps({"tables": [{"name": n, "rows": t["rows"]} for n, t in self.tables.items()]})
        if name == "describe_table":
            table = str(arguments.get("table") or "").strip().strip('"`[]').lower()
            if table not in self.tables:
                return _err(f"no table {arguments.get('table')!r}; tables: {', '.join(self.tables)}")
            t = self.tables[table]
            return json.dumps({"table": table, "rows": t["rows"], "columns": t["columns"], "definition": t["sql"]})
        return self._run_sql(arguments.get("query"))

    def _run_sql(self, query: object) -> str:
        if not isinstance(query, str) or not query.strip():
            return _err("query must be a non-empty SQL string")
        self._steps, self._deadline = 0, time.monotonic() + MAX_SECONDS
        try:
            cur = self.db.execute(query)
            rows = cur.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in cur.description or []]
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                return _err(f"query aborted: it ran longer than allowed ({MAX_SECONDS:.0f}s / {MAX_STEPS:,} steps). "
                            "Filter earlier, avoid cross joins, or break it into smaller queries.")
            if "not authorized" in str(e) or "readonly" in str(e):
                return _err(f"{e}: the database is read-only and only SELECT queries are allowed")
            return _err(f"SQL error: {e}")
        except (sqlite3.Error, ValueError, OverflowError) as e:
            return _err(f"SQL error: {e}")
        truncated = len(rows) > MAX_ROWS
        rows = [[c[:MAX_CELL] + "..." if isinstance(c, str) and len(c) > MAX_CELL else
                 c.hex()[:MAX_CELL] if isinstance(c, bytes) else c for c in row] for row in rows[:MAX_ROWS]]
        while True:
            out = json.dumps({"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}, default=str)
            if len(out) <= MAX_RESULT or not rows:
                return out
            rows, truncated = rows[: len(rows) // 2], True

    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        found = _ANSWER.findall(final_answer or "")
        if not found:
            return False, ["no 'ANSWER: <number>' line in the final answer"]
        value = parse_amount(found[-1])
        if value is None:
            return False, [f"couldn't read a number from ANSWER: {found[-1]!r}"]
        if abs(value - self.answer) > self.tolerance:
            return False, [f"expected {self.answer:.2f}, got {value:.2f}"]
        return True, []
