"""Single-query subprocess: stdlib only; invoked by SQLTool, never accepts file SQL."""

# 单次查询的隔离执行器，仅依赖 Python 标准库，可直接作为子进程启动。
# 防护分为三层：操作系统资源限制、SQLite 只读/authorizer、结果行数与字节数限制。
# 这是一套受限 SQLite 执行环境，不能把它等同于完整的容器或跨租户安全沙箱。

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


# 输入来自 SQLTool 的 JSON 请求，输出 ok/columns/rows 或可诊断的 error。
# schema 是固定表集合的内部操作；用户 SQL 不能借此自由执行任意 PRAGMA。
def execute(request: dict) -> dict:
    memory = request["memory_mb"] * 1024 * 1024
    # 限制子进程地址空间、CPU 和文件写入大小。父进程还会独立控制墙钟时间。
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
    # 使用 mode=ro 打开库；后续 query_only 与 authorizer 再收紧执行权限。
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
        # Schema 工具仅读取项目固定表的字段和外键。
        # 普通 SQL 的 authorizer 不开放 sqlite_master 扫描。
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

        # 按 SQLite 实际解析出的操作授权，而不是靠 SQL 是否以 SELECT 开头来猜安全性。
        # 只允许主库的白名单业务表及函数，其他操作默认拒绝，包括写入与 ATTACH。
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
        # execute 只接受单条语句；模型给出多语句时由 SQLite 拒绝，不替它拆开执行。
        cursor = db.execute(sql)
        if cursor.description is None:
            return {"ok": False, "error": "query_has_no_result"}
        columns = [col[0] for col in cursor.description]
        if len(set(columns)) != len(columns):
            return {"ok": False, "error": "duplicate_columns_require_aliases"}
        # 多取一行是为了检测截断。返回前再检查总字节，避免少量巨大字段绕过行数限制。
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
