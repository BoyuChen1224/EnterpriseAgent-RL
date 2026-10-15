"""Single-query subprocess: stdlib only; invoked by SQLTool, never accepts file SQL."""

from __future__ import annotations

import json
import math
import resource
import sqlite3
import sys
import time
from pathlib import Path

TABLES = ("regions", "customers", "products", "orders", "order_items")
FUNCTIONS = {
    "sum",
    "count",
    "avg",
    "min",
    "max",
    "coalesce",
    "nullif",
    "abs",
    "round",
    "lower",
    "upper",
    "length",
    "substr",
    "substring",
    "trim",
    "like",
    "typeof",
}


def execute(request: dict) -> dict:
    memory = request["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = math.ceil(request["timeout_s"]) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    operation = request.get("operation", "sql")
    if operation not in ("sql", "schema"):
        return {"ok": False, "error": "unknown_operation"}
    sql = request.get("sql", "")
    if operation == "sql" and (not isinstance(sql, str) or not sql.strip() or len(sql) > 16000):
        return {"ok": False, "error": "invalid_sql_length"}
    deadline = time.monotonic() + request["timeout_s"]
    db = sqlite3.connect(
        Path(request["database"]).resolve().as_uri() + "?mode=ro", timeout=0.1, uri=True
    )
    try:
        db.enable_load_extension(False)
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA cache_size=-1024")
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, request["max_bytes"])
        db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16000)
        db.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 64)
        db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        db.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 10)

        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 500)
        if operation == "schema":
            schema = {
                table: {
                    "columns": [
                        {"name": r[1], "type": r[2], "primary_key": bool(r[5])}
                        for r in db.execute(f'PRAGMA table_info("{table}")')
                    ],
                    "foreign_keys": [
                        {"table": r[2], "from": r[3], "to": r[4]}
                        for r in db.execute(f'PRAGMA foreign_key_list("{table}")')
                    ],
                }
                for table in TABLES
            }
            return {"ok": True, "schema": schema}

        def authorize(code, arg1, arg2, database, source):
            if code in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if (
                code == sqlite3.SQLITE_READ
                and arg1 in TABLES
                and (database == "main" or (database is None and arg2 == ""))
            ):
                return sqlite3.SQLITE_OK
            if code == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 500)
        cursor = db.execute(sql)
        if cursor.description is None:
            return {"ok": False, "error": "query_has_no_result"}
        columns = [col[0] for col in cursor.description]
        if len(set(columns)) != len(columns):
            return {"ok": False, "error": "duplicate_columns_require_aliases"}
        rows = cursor.fetchmany(request["max_rows"] + 1)
        result = {
            "ok": True,
            "columns": columns,
            "rows": [list(r) for r in rows[: request["max_rows"]]],
            "truncated": len(rows) > request["max_rows"],
        }
        if len(json.dumps(result).encode()) > request["max_bytes"]:
            return {"ok": False, "error": "result_byte_limit"}
        return result
    except (sqlite3.Error, ValueError, MemoryError) as exc:
        return {"ok": False, "error": str(exc)[:300] or type(exc).__name__}
    finally:
        db.close()


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin))))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
