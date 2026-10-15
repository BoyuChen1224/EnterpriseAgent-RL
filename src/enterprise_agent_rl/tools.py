"""Read-only schema/dictionary/SQL tools; the query process has a hard wall deadline."""

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

    @staticmethod
    def dictionary(term: str | None = None) -> dict:
        if term is not None and term not in DICTIONARY:
            raise ValueError("unknown_dictionary_term")
        return {term: DICTIONARY[term]} if term else dict(DICTIONARY)
