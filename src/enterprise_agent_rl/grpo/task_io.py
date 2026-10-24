"""Load explicit public task descriptors without inferring benchmark paths."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .contracts import TaskInput


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        if isinstance(value, dict):
            values = value.get("tasks", value.get("records", value))
        else:
            values = value
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError(f"task_index_must_be_list_of_objects:{path}")
    return values


def _public_task(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("public_task")
    if value is None and isinstance(row.get("task"), dict):
        value = row["task"].get("public", row["task"])
    if not isinstance(value, dict):
        raise ValueError(f"public_task_missing:{row.get('task_id')}")
    forbidden = {"reference_sql", "expected_answer", "hidden", "verifier_only"}
    if forbidden & set(value):
        raise ValueError(f"hidden_task_fields_in_public_descriptor:{row.get('task_id')}")
    return dict(value)


def _task_id(row: dict[str, Any]) -> str:
    public = _public_task(row)
    return str(row.get("task_id") or public.get("task_id") or "")


def _panel_tasks(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        if not isinstance(task_id, str) or not isinstance(prompt, str) or not prompt:
            raise ValueError(f"public_panel_row_invalid:{path}")
        if task_id in result:
            raise ValueError(f"public_panel_duplicate_task:{task_id}")
        result[task_id] = {"task_id": task_id, "prompt": prompt}
    return result


def load_task_inputs(
    path: Path,
    *,
    repo_root: Path,
    task_ids: Iterable[str] | None = None,
    public_panel: Path | None = None,
    database_root: Path | None = None,
    default_database: Path | None = None,
) -> list[TaskInput]:
    rows = _rows(path)
    by_id = {_task_id(row): row for row in rows}
    requested = list(task_ids) if task_ids is not None else list(by_id)
    if len(by_id) != len(rows):
        raise ValueError(f"task_index_duplicate_task_id:{path}")
    missing = [task_id for task_id in requested if task_id not in by_id]
    if missing:
        raise ValueError(f"task_index_requested_ids_missing:{missing}")
    panel = _panel_tasks(public_panel)
    result: list[TaskInput] = []
    for index, task_id in enumerate(requested):
        row = by_id[task_id]
        if {"reference_sql", "expected_answer", "hidden", "verifier_only"} & set(row):
            raise ValueError(f"hidden_task_fields_in_descriptor:{row.get('task_id')}")
        public = panel.get(task_id, _public_task(row))
        group_id = str(row.get("group_id") or row.get("group") or f"g{index}")
        database = row.get("database_path") or row.get("database")
        if database is None and row.get("database_id") and database_root is not None:
            database = database_root / f"{row['database_id']}.sqlite3"
        if database is None and default_database is not None:
            database = default_database
        if database is not None:
            database_path = Path(str(database))
            if not database_path.is_absolute():
                database_path = repo_root / database_path
            database = str(database_path.resolve())
        verifier_payload = row.get("verifier_payload")
        if verifier_payload is None and isinstance(row.get("task"), dict):
            verifier_payload = row["task"]
        result.append(
            TaskInput(
                task_id=task_id,
                group_id=group_id,
                public_task=public,
                database_path=str(database) if database is not None else None,
                verifier_payload=dict(verifier_payload)
                if isinstance(verifier_payload, dict)
                else None,
                metadata={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "task",
                        "public_task",
                        "verifier_payload",
                        "reference_sql",
                        "expected_answer",
                    }
                },
            )
        )
    if not result:
        raise ValueError(f"task_index_empty:{path}")
    return result
