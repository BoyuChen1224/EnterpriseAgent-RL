"""Read-only schema/dictionary/SQL tools; the query process has a hard wall deadline."""

# 父进程的工具适配层。真正的 SQLite 连接在 sql_worker.py 子进程中创建。
# execute 返回统一字典；schema 复用子进程隔离；dictionary 读取项目固定的业务词条。
# SQL audit 保存查询哈希、耗时和上下文，用于区分逻辑动作与实际 worker 执行。

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import DICTIONARY


class SQLTool:
    def __init__(self, database: Path, settings: Settings, audit_path: Path | None = None):
        self.database = database.resolve(strict=True)
        self.settings = settings
        self.audit_path = audit_path
        self._audit_lock = threading.Lock()

    # 单次 SQL 请求携带只读库路径和资源上限。时限不能超过外层任务传入的剩余时间。
    def execute(
        self, sql: str, *, context: dict | None = None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        started = time.monotonic()
        timeout = min(self.settings.sql_timeout_s, timeout_s or self.settings.sql_timeout_s)
        request = {
            "database": str(self.database),
            "sql": sql,
            "timeout_s": timeout,
            "max_rows": self.settings.sql_max_rows,
            "max_bytes": self.settings.sql_max_bytes,
            "memory_mb": self.settings.sql_memory_mb,
        }
        result = self._invoke(request, timeout)
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                "ok": result["ok"],
                "error": result.get("error"),
                "elapsed_s": time.monotonic() - started,
                **(context or {}),
            }
            with self._audit_lock, self.audit_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return result

    # 用当前 Python 解释器启动标准库 worker，并以 JSON 传参。
    # 父进程的 subprocess timeout 是墙钟硬边界，补充 worker 内的 SQLite 进度检查。
    @staticmethod
    def _invoke(request: dict, timeout: float) -> dict:
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("sql_worker.py"))],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            result = (
                json.loads(completed.stdout)
                if completed.returncode == 0
                else {"ok": False, "error": "query_worker_resource_limit_or_failure"}
            )
        except subprocess.TimeoutExpired:
            result = {"ok": False, "error": "query_timeout"}
        except (ValueError, OSError):
            result = {"ok": False, "error": "query_worker_failure"}
        return result

    # 表结构读取也可能阻塞，因此同样走有超时的 worker，而不在主进程直接连接 SQLite。
    def schema(self, *, timeout_s: float | None = None) -> dict:
        timeout = min(self.settings.sql_timeout_s, timeout_s or self.settings.sql_timeout_s)
        result = self._invoke(
            {
                "operation": "schema",
                "database": str(self.database),
                "timeout_s": timeout,
                "memory_mb": self.settings.sql_memory_mb,
                "max_bytes": self.settings.sql_max_bytes,
            },
            timeout,
        )
        if not result["ok"]:
            raise ValueError("schema_" + result["error"])
        return result["schema"]

    # 公开业务定义帮助模型理解收入口径、状态值等；它不是任务的参考答案。
    @staticmethod
    def dictionary(term: str | None = None) -> dict:
        if term is not None and term not in DICTIONARY:
            raise ValueError("unknown_dictionary_term")
        return {term: DICTIONARY[term]} if term else dict(DICTIONARY)
