#!/usr/bin/env python3
"""Prepare and run the isolated eight-case Qwen3 train Base panel.

The worker path deliberately uses the production LocalModelPolicy ->
run_episode -> SQLTool -> rewards.verify chain.  This module never loads a
model during ``--prepare`` and never exposes reference SQL to the policy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from enterprise_agent_rl.agent import LocalModelPolicy, run_episode
from enterprise_agent_rl.benchmark import (
    SPLITS,
    create_instance,
    database_content_fingerprint,
    python_oracle,
    task_specs,
)
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask, PublicTask
from enterprise_agent_rl.evaluation import summarize
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import TrajectoryStore, database_fingerprint

RUN_DEFAULT = Path("artifacts/qwen3-base-lora-pilot/t031-v1")
MODEL_DEFAULT = Path("configs/model/qwen3-8b.toml")
AGENT_DEFAULT = Path("configs/agent/qwen3-base-panel.toml")
CASE_TEMPLATES = ("schema", "region", "product", "recovery")
INFRA_CODES = {
    "oom",
    "non_finite_logits",
    "model_transport_error",
    "model_http_error",
    "model_busy",
    "generation_error",
    "inference_timeout",
    "model_response_too_large",
    "context_length_exceeded",
    "query_timeout",
    "query_worker_failure",
    "query_worker_resource_limit_or_failure",
    "database_unavailable",
}
INFRA_TEXT = (
    "cuda",
    "out_of_memory",
    "non_finite",
    "model_load",
    "tokenizer",
    "server_crash",
    "shared_library",
    "database is locked",
    "unable to open",
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _task_for(instance: int, template: str) -> BenchmarkTask:
    for candidate, prompt, reference_sql, constraints in task_specs("train", instance):
        if candidate == template:
            return BenchmarkTask(
                public=PublicTask(task_id=f"train-{instance}-{template}", prompt=prompt),
                split="train",
                reference_sql=reference_sql,
                ordered=True,
                **constraints,
            )
    raise ValueError(f"unknown_train_template:{template}")


def _selected_task_rows() -> list[tuple[int, str]]:
    return [(instance, template) for instance in range(2) for template in CASE_TEMPLATES]


def prepare_run(output: Path) -> dict[str, Any]:
    """Materialize only train instance 0/1 and validate eight labels offline."""

    database_root = output / "base" / "databases"
    database_root.mkdir(parents=True, exist_ok=True)
    settings = Settings.load(AGENT_DEFAULT)
    databases = []
    for instance in range(2):
        database_id = f"train-{instance}"
        database = database_root / f"{database_id}.sqlite3"
        if not database.exists():
            create_instance(database, "train", instance)
        if not database.is_file():
            raise FileNotFoundError(database)
        databases.append(
            {
                "id": database_id,
                "split": "train",
                "seed": SPLITS["train"][1] + instance,
                "year": SPLITS["train"][0],
                "path": str(database.resolve()),
                "physical_sha256": database_fingerprint(database),
                "logical_fingerprint": database_content_fingerprint(database),
            }
        )

    task_rows = []
    oracle_checks = []
    for instance, template in _selected_task_rows():
        task = _task_for(instance, template)
        database = database_root / f"train-{instance}.sqlite3"
        result = SQLTool(database, settings).execute(task.reference_sql)
        expected_rows = python_oracle(database, template, "train", instance)
        if not result["ok"] or result["truncated"] or result["rows"] != expected_rows:
            raise RuntimeError(f"reference_oracle_mismatch:{task.public.task_id}")
        oracle = {"columns": result["columns"], "rows": expected_rows}
        oracle_checks.append(
            {
                "task_id": task.public.task_id,
                "database_id": f"train-{instance}",
                "template": template,
                "oracle_sha256": _sha(oracle),
                "columns": result["columns"],
            }
        )
        task_rows.append(
            {
                "task_id": task.public.task_id,
                "database_id": f"train-{instance}",
                "instance": instance,
                "template": template,
                "public_task": task.public.model_dump(),
                "reference_sql_sha256": hashlib.sha256(task.reference_sql.encode()).hexdigest(),
                "constraints": {
                    "ordered": task.ordered,
                    "min_sql_calls": task.min_sql_calls,
                    "required_probe_sql": task.required_probe_sql is not None,
                    "required_schema": task.required_schema,
                    "required_sql_error": task.required_sql_error,
                },
            }
        )

    manifest = {
        "schema_version": "1.0",
        "experiment": "T-031/E-028",
        "run_id": output.name,
        "created_at_utc": _now(),
        "split": "train",
        "year": 2021,
        "seeds": {"train-0": 101, "train-1": 102},
        "planned": len(task_rows),
        "task_templates": list(CASE_TEMPLATES),
        "tasks": task_rows,
        "databases": databases,
        "oracle_checks": oracle_checks,
        "database_fingerprint_kind": "physical_sha256_and_logical_canonical_rows_v1",
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "reference_sql_policy": "verifier_and_cpu_oracle_only",
        "model_messages_contain_reference_sql": False,
        "cpu_sql_oracle_calls": len(task_rows),
        "cpu_sql_oracle_call_budget": 32,
    }
    _write_json(output / "panel-manifest.json", manifest)
    _write_json(
        output / "cpu-preflight.json",
        {
            "status": "passed",
            "python": sys.version,
            "cwd": str(Path.cwd()),
            "entrypoint": str(Path(__file__).resolve()),
            "settings": settings.model_dump(),
            "planned": len(task_rows),
            "database_root": str(database_root.resolve()),
            "validation_split_accessed": False,
            "test_split_accessed": False,
        },
    )
    return manifest


def _explicit_infrastructure_errors(trajectory) -> list[str]:
    errors: list[str] = []
    for attempt in trajectory.model_attempts:
        values = []
        for key in ("error", "engine_error"):
            if attempt.get(key):
                values.append(attempt[key])
        server_error = attempt.get("server_error")
        if server_error:
            values.append(server_error)
        telemetry = attempt.get("telemetry", {})
        if telemetry.get("non_finite_logits_detected"):
            values.append("non_finite_logits")
        for value in values:
            code = str(value.get("code", "")) if isinstance(value, dict) else str(value)
            message = str(value.get("message", "")) if isinstance(value, dict) else str(value)
            text = f"{code} {message}".lower()
            if code in INFRA_CODES or any(token in text for token in INFRA_TEXT):
                errors.append(code or message[:200])
    for item in trajectory.runtime_errors:
        text = f"{item.get('type', '')} {item.get('message', '')}".lower()
        if any(token in text for token in INFRA_TEXT) or any(
            code in text for code in INFRA_CODES
        ):
            errors.append(text[:200])
    return sorted(set(errors))


def _business_category(trajectory, reward) -> str:
    if reward is not None and reward.success:
        return "Successful"
    reason = str(reward.reason if reward is not None else trajectory.terminal_reason)
    lower = reason.lower()
    if "budget_exhausted" in lower or "tool_budget" in lower:
        return "Action Budget Exhausted"
    if any(
        attempt.get("error") in {"invalid_json", "invalid_action", "invalid_action_arguments"}
        for attempt in trajectory.model_attempts
    ):
        return "Invalid Action"
    if any(
        step.tool_name == "sql" and not step.tool_result.ok for step in trajectory.steps
    ):
        return "SQL Error or Recovery Failure"
    if trajectory.terminal_reason == "completed":
        return "Wrong Final Answer"
    return "Other Business Failure"


def _persist_engine_events(engine_dir: Path, response_dir: Path) -> int:
    events_path = engine_dir / "generations.jsonl"
    if not events_path.is_file():
        return 0
    response_dir.mkdir(parents=True, exist_ok=True)
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    _write_json(response_dir / "engine-events.json", events)
    with (response_dir / "engine-events.jsonl").open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    for event in events:
        event_id = str(event.get("id", "unknown"))
        _write_json(response_dir / f"request-{event_id}.json", event.get("request", {}))
        _write_json(
            response_dir / f"response-{event_id}.json",
            {
                "id": event_id,
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "error": event.get("error"),
                "telemetry": event.get("telemetry", {}),
            },
        )
    return len(events)


def run_worker(
    output: Path,
    endpoint: str,
    engine_dir: Path,
    worker_index: int,
    worker_count: int,
) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("qwen3_panel_requires_slurm_job")
    manifest = json.loads((output / "panel-manifest.json").read_text())
    settings = Settings.load(AGENT_DEFAULT)
    model_config = ModelSettings.load(MODEL_DEFAULT)
    if model_config.generation.enable_thinking is not False:
        raise RuntimeError("qwen3_panel_requires_hard_thinking_off")
    worker_root = output / "base" / f"worker-{worker_index}"
    trajectory_store = TrajectoryStore(worker_root / "trajectories")
    response_dir = worker_root / "response"
    results_path = worker_root / "case-results.jsonl"
    worker_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        worker_root / "worker-meta.json",
        {
            "worker_index": worker_index,
            "worker_count": worker_count,
            "endpoint": endpoint,
            "engine_dir": str(engine_dir.resolve()),
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model": model_config.model_dump(),
        },
    )
    existing = {}
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["task_id"]] = record
    policy = LocalModelPolicy(
        endpoint,
        model_config.model_id,
        "qwen3-base-panel-v1",
        model_config.generation,
    )
    selected = [
        item
        for index, item in enumerate(manifest["tasks"])
        if index % worker_count == worker_index
    ]
    for item in selected:
        task_id = item["task_id"]
        if task_id in existing and existing[task_id].get("completed"):
            continue
        task = _task_for(int(item["instance"]), str(item["template"]))
        database = output / "base" / "databases" / f"{item['database_id']}.sqlite3"
        started = time.monotonic()
        trajectory = None
        reward = None
        verifier_error = None
        unknown_reasons: list[str] = []
        try:
            trajectory = run_episode(
                task.public,
                policy,
                database,
                settings,
                worker_root / "sql-audit.jsonl",
            )
            trajectory_path = trajectory_store.save(trajectory)
            unknown_reasons = _explicit_infrastructure_errors(trajectory)
            if not unknown_reasons:
                try:
                    reward = verify(task, trajectory, database)
                except Exception as exc:
                    verifier_error = {"type": type(exc).__name__, "message": str(exc)}
                    unknown_reasons = ["verifier:" + type(exc).__name__]
            if reward is not None:
                trajectory = trajectory.model_copy(update={"reward": reward})
                trajectory_path = trajectory_store.save(trajectory)
            case = {
                "schema_version": "1.0",
                "task_id": task_id,
                "worker_index": worker_index,
                "attempted": True,
                "unknown": bool(unknown_reasons),
                "unknown_reasons": unknown_reasons,
                "qualified": not unknown_reasons,
                "scored": reward is not None,
                "success": bool(reward and reward.success),
                "category": (
                    "Unknown / Infrastructure"
                    if unknown_reasons
                    else _business_category(trajectory, reward)
                ),
                "reward": reward.model_dump() if reward is not None else None,
                "verifier_status": "scored" if reward is not None else "unknown",
                "verifier_error": verifier_error,
                "terminal_reason": trajectory.terminal_reason,
                "database_sha256": trajectory.database_sha256,
                "trajectory_path": str(trajectory_path.resolve()),
                "model_attempts": len(trajectory.model_attempts),
                "tool_attempts": len([step for step in trajectory.steps if step.tool_name]),
                "elapsed_s": time.monotonic() - started,
                "engine_event_ids": [
                    attempt.get("engine_event_id")
                    for attempt in trajectory.model_attempts
                    if attempt.get("engine_event_id")
                ],
                "completed": True,
            }
        except Exception as exc:
            case = {
                "schema_version": "1.0",
                "task_id": task_id,
                "worker_index": worker_index,
                "attempted": True,
                "unknown": True,
                "unknown_reasons": [f"runner:{type(exc).__name__}:{str(exc)[:300]}"],
                "qualified": False,
                "scored": False,
                "success": False,
                "category": "Unknown / Infrastructure",
                "reward": None,
                "verifier_status": "not_run",
                "verifier_error": None,
                "terminal_reason": "runner_exception",
                "database_sha256": database_fingerprint(database),
                "trajectory_path": None,
                "model_attempts": 0,
                "tool_attempts": 0,
                "elapsed_s": time.monotonic() - started,
                "engine_event_ids": [],
                "runner_error": {"type": type(exc).__name__, "message": str(exc)},
                "completed": True,
            }
        _append_jsonl(results_path, case)
        existing[task_id] = case
        _persist_engine_events(engine_dir, response_dir)

    _persist_engine_events(engine_dir, response_dir)
    summary = {
        "worker_index": worker_index,
        "planned": len(selected),
        "completed": sum(bool(item.get("completed")) for item in existing.values()),
        "unknown": sum(bool(item.get("unknown")) for item in existing.values()),
        "scored": sum(bool(item.get("scored")) for item in existing.values()),
        "engine_events": len(
            json.loads((response_dir / "engine-events.json").read_text())
            if (response_dir / "engine-events.json").is_file()
            else []
        ),
    }
    _write_json(worker_root / "summary.json", summary)
    return 0 if summary["completed"] == summary["planned"] and summary["unknown"] == 0 else 2


def merge_workers(output: Path) -> int:
    manifest = json.loads((output / "panel-manifest.json").read_text())
    settings = Settings.load(AGENT_DEFAULT)
    records_by_id = {}
    for worker_path in sorted((output / "base").glob("worker-*/case-results.jsonl")):
        for line in worker_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records_by_id[record["task_id"]] = record
    ordered = [records_by_id.get(item["task_id"]) for item in manifest["tasks"]]
    if any(record is None for record in ordered):
        raise RuntimeError("base_panel_missing_case_result")
    case_path = output / "base" / "case-results.jsonl"
    with case_path.open("w") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    trajectories = []
    for record in ordered:
        if record["scored"] and record.get("trajectory_path"):
            trajectories.append(TrajectoryStore.load(Path(record["trajectory_path"])))
    aggregate = (
        summarize(trajectories, settings, allow_multiple_databases=True)
        if trajectories
        else None
    )
    all_steps = [step for trajectory in trajectories for step in trajectory.steps]
    sql_steps = [step for step in all_steps if step.tool_name == "sql"]
    sql_fingerprints = [
        _sha(step.action.model_dump()) for step in sql_steps if step.tool_result.ok
    ]
    event_ids = []
    peak_allocated = []
    peak_reserved = []
    for worker_dir in sorted((output / "base").glob("worker-*")):
        events_file = worker_dir / "response" / "engine-events.json"
        if events_file.is_file():
            for event in json.loads(events_file.read_text()):
                event_ids.append(event.get("id"))
                telemetry = event.get("telemetry", {})
                if telemetry.get("peak_allocated_bytes") is not None:
                    peak_allocated.append(telemetry["peak_allocated_bytes"])
                if telemetry.get("peak_reserved_bytes") is not None:
                    peak_reserved.append(telemetry["peak_reserved_bytes"])
    counts = {
        "planned": len(ordered),
        "attempted": sum(bool(record["attempted"]) for record in ordered),
        "qualified": sum(bool(record["qualified"]) for record in ordered),
        "scored": sum(bool(record["scored"]) for record in ordered),
        "unknown": sum(bool(record["unknown"]) for record in ordered),
        "successes": sum(bool(record["success"]) for record in ordered),
    }
    metrics = {
        "schema_version": "1.0",
        "status": "BASE_PANEL_COMPLETE"
        if counts["scored"] == counts["planned"] and counts["unknown"] == 0
        else "BASE_PANEL_INCOMPLETE",
        "counts": counts,
        "planned_denominator": len(ordered),
        "business_success_rate": (
            counts["successes"] / counts["planned"]
            if counts["scored"] == counts["planned"] and counts["unknown"] == 0
            else None
        ),
        "scored_subset_success_rate": counts["successes"] / counts["scored"]
        if counts["scored"]
        else None,
        "unknown_reasons": sorted(
            {
                reason
                for record in ordered
                for reason in record.get("unknown_reasons", [])
            }
        ),
        "sql_executable_rate": sum(step.tool_result.ok for step in sql_steps) / len(sql_steps)
        if sql_steps
        else None,
        "sql_calls": len(sql_steps),
        "executable_sql_calls": sum(step.tool_result.ok for step in sql_steps),
        "tool_calls": len(all_steps),
        "tool_errors": sum(not step.tool_result.ok for step in all_steps),
        "repeated_sql": len(sql_fingerprints) - len(set(sql_fingerprints)),
        "behavior_counts": dict(
            Counter(
                step.action.kind
                for trajectory in trajectories
                for step in trajectory.steps
            )
        ),
        "engine_events": len(set(event_ids)),
        "engine_event_ids_unique": len(event_ids) == len(set(event_ids)),
        "peak_allocated_bytes": max(peak_allocated, default=None),
        "peak_reserved_bytes": max(peak_reserved, default=None),
        "aggregate_scored_metrics": aggregate,
        "model_requests_limit": 340,
        "runtime_tool_attempt_limit": 112,
        "verifier_limit": 24,
        "verifier_sql_limit": 48,
        "cpu_oracle_sql_limit": 32,
        "old_validation_comparison_allowed": False,
    }
    _write_json(output / "base" / "metrics.json", metrics)
    _write_json(
        output / "base" / "verifier-results.json",
        {
            "scored": counts["scored"],
            "unknown": counts["unknown"],
            "verifier_calls": counts["scored"],
            "verifier_sql_calls": counts["scored"],
            "reference_sql_content_exposed_to_model": False,
        },
    )
    _write_json(
        output / "base" / "summary.json",
        {
            "status": metrics["status"],
            "counts": counts,
            "task_ids": [record["task_id"] for record in ordered],
            "finished_at_utc": _now(),
        },
    )
    return 0 if metrics["status"] == "BASE_PANEL_COMPLETE" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--endpoint")
    parser.add_argument("--engine-dir", type=Path)
    args = parser.parse_args()
    if sum(bool(value) for value in (args.prepare, args.merge, args.worker_index is not None)) != 1:
        raise SystemExit("choose exactly one of --prepare, --merge, --worker-index")
    if args.prepare:
        prepare_run(args.output)
        return 0
    if args.merge:
        return merge_workers(args.output)
    if args.worker_index is None or not args.endpoint or args.engine_dir is None:
        raise SystemExit("worker requires --worker-index --endpoint --engine-dir")
    if not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker-index-out-of-range")
    return run_worker(
        args.output,
        args.endpoint,
        args.engine_dir,
        args.worker_index,
        args.worker_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
