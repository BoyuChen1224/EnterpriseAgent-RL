#!/usr/bin/env python3
"""Prepare and score the frozen 20-task Qwen3 Base/SFT comparison.

Only ``PublicTask`` objects and actual tool observations enter
``LocalModelPolicy``.  Reference SQL, expected rows, and verifier-only
constraints live in ``evaluation/validation/private`` and are read only by
the scoring worker after the episode has been collected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
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
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.tools import SQLTool
from enterprise_agent_rl.trajectories import TrajectoryStore, database_fingerprint

RUN_DEFAULT = Path("artifacts/qwen3-runtime-sft-comparison/t032-v1")
MODEL_DEFAULT = Path("configs/model/qwen3-8b.toml")
AGENT_DEFAULT = Path("configs/agent/qwen3-base-panel.toml")
VALIDATION_TEMPLATES = (
    "schema",
    "region",
    "top",
    "customer",
    "product",
    "empty",
    "constraints",
    "recovery",
    "status",
    "monthly",
)
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


class PanelError(RuntimeError):
    """Raised when a frozen-panel or persistence contract is violated."""


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


SOURCE_PATHS = (
    "configs/training/qwen3-runtime-sft.toml",
    "scripts/evaluation/record_t032_allocation.py",
    "scripts/evaluation/run_qwen3_runtime_panel.py",
    "scripts/slurm/qwen3_runtime_sft_comparison.sbatch",
    "scripts/training/run_qwen3_sft.py",
    "src/enterprise_agent_rl/lora.py",
    "src/enterprise_agent_rl/qwen3_sft.py",
    "tests/test_qwen3_runtime_sft.py",
)


def write_run_manifest(output: Path) -> dict[str, Any]:
    """Snapshot only this task's source paths; never sweep the worktree."""

    snapshot_root = output / "source-snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    files = []
    tracked_diff_parts = []
    for relative in SOURCE_PATHS:
        source = Path(relative)
        if not source.is_file():
            raise PanelError(f"source_snapshot_missing:{relative}")
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "snapshot_path": str(target.relative_to(output)),
            }
        )
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--", relative],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.stdout:
            tracked_diff_parts.append(diff.stdout)
        if (
            not subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            untracked_diff = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", relative],
                capture_output=True,
                text=True,
                check=False,
            )
            if untracked_diff.stdout:
                tracked_diff_parts.append(untracked_diff.stdout)
    dirty_patch = output / "source-snapshot" / "dirty.patch"
    dirty_patch.write_text("\n".join(tracked_diff_parts))
    status = subprocess.run(
        ["git", "status", "--short", "--", *SOURCE_PATHS],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    manifest = {
        "schema_version": "1.0",
        "experiment": "T-032/E-029",
        "code_commit_at_snapshot": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "git_dirty_for_task_paths": bool(status),
        "task_path_status": status,
        "source_paths": files,
        "dirty_patch": str(dirty_patch.resolve()),
        "dirty_patch_sha256": hashlib.sha256(dirty_patch.read_bytes()).hexdigest(),
        "protected_paths_included": [],
        "created_at_utc": _now(),
    }
    _write_json(output / "manifest.json", manifest)
    commands = {
        "training_prepare": (
            "python scripts/training/run_qwen3_sft.py --stage prepare "
            f"--output {output} --cache artifacts/model-cache-qwen3-8b "
            "--model-config configs/model/qwen3-8b.toml "
            "--training-config configs/training/qwen3-runtime-sft.toml"
        ),
        "tiny_cpu_gate": (
            f"python scripts/training/run_qwen3_sft.py --stage tiny --output {output}"
        ),
        "validation_prepare": (
            f"python scripts/evaluation/run_qwen3_runtime_panel.py --prepare --output {output}"
        ),
        "training": "T032_STAGE=train scripts/slurm/qwen3_runtime_sft_comparison.sbatch",
        "reload": "T032_STAGE=reload scripts/slurm/qwen3_runtime_sft_comparison.sbatch",
        "comparison": "T032_STAGE=compare scripts/slurm/qwen3_runtime_sft_comparison.sbatch",
    }
    _write_json(output / "commands.json", commands)
    return manifest


def _write_immutable_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise PanelError(f"immutable_artifact_conflict:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _append_unique_jsonl(path: Path, value: dict[str, Any], *, key: str) -> None:
    existing = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(item.get(key) == value.get(key) for item in existing):
        raise PanelError(f"duplicate_persisted_event:{key}:{value.get(key)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _task_from_spec(instance: int, template: str) -> BenchmarkTask:
    for candidate, prompt, reference_sql, constraints in task_specs("validation", instance):
        if candidate == template:
            return BenchmarkTask(
                public=PublicTask(task_id=f"validation-{instance}-{template}", prompt=prompt),
                split="validation",
                reference_sql=reference_sql,
                ordered=True,
                **constraints,
            )
    raise PanelError(f"unknown_validation_template:{template}")


def _task_rows() -> list[tuple[int, str]]:
    return [(instance, template) for instance in range(2) for template in VALIDATION_TEMPLATES]


def prepare_validation(output: Path, *, agent_config: Path = AGENT_DEFAULT) -> dict[str, Any]:
    """Materialize only validation-0/1, without calling build_manifest."""

    root = output / "evaluation" / "validation"
    database_root = root / "databases"
    private_root = root / "private"
    database_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    settings = Settings.load(agent_config)
    databases = []
    for instance in range(2):
        database_id = f"validation-{instance}"
        database = database_root / f"{database_id}.sqlite3"
        if not database.exists():
            create_instance(database, "validation", instance)
        databases.append(
            {
                "id": database_id,
                "split": "validation",
                "year": SPLITS["validation"][0],
                "seed": SPLITS["validation"][1] + instance,
                "path": str(database.resolve()),
                "physical_sha256": database_fingerprint(database),
                "logical_fingerprint": database_content_fingerprint(database),
            }
        )

    public_rows = []
    private_rows = []
    oracle_rows = []
    for instance, template in _task_rows():
        task = _task_from_spec(instance, template)
        database = database_root / f"validation-{instance}.sqlite3"
        result = SQLTool(database, settings).execute(task.reference_sql)
        expected_rows = python_oracle(database, template, "validation", instance)
        if not result["ok"] or result["truncated"] or result["rows"] != expected_rows:
            raise PanelError(f"reference_oracle_mismatch:{task.public.task_id}")
        public_rows.append(
            {
                "task_id": task.public.task_id,
                "database_id": f"validation-{instance}",
                "instance": instance,
                "template": template,
                "public_task": task.public.model_dump(),
                "public_task_sha256": _sha(task.public.model_dump()),
            }
        )
        private_rows.append(
            {
                "task_id": task.public.task_id,
                "database_id": f"validation-{instance}",
                "task": task.model_dump(),
                "reference_sql": task.reference_sql,
                "expected": {"columns": result["columns"], "rows": expected_rows},
                "expected_sha256": _sha({"columns": result["columns"], "rows": expected_rows}),
            }
        )
        oracle_rows.append(
            {
                "task_id": task.public.task_id,
                "database_id": f"validation-{instance}",
                "template": template,
                "oracle_sha256": _sha({"columns": result["columns"], "rows": expected_rows}),
                "columns": result["columns"],
            }
        )

    public_path = root / "public-panel.jsonl"
    private_path = private_root / "verifier-inputs.jsonl"
    public_path.write_text("\n".join(_canonical(row) for row in public_rows) + "\n")
    private_path.write_text("\n".join(_canonical(row) for row in private_rows) + "\n")
    manifest = {
        "schema_version": "1.0",
        "experiment": "T-032/E-029",
        "created_at_utc": _now(),
        "split": "validation",
        "year": 2023,
        "seeds": {"validation-0": 201, "validation-1": 202},
        "task_templates": list(VALIDATION_TEMPLATES),
        "planned": len(public_rows),
        "task_ids": [row["task_id"] for row in public_rows],
        "tasks": public_rows,
        "databases": databases,
        "oracle_checks": oracle_rows,
        "public_panel_path": str(public_path.resolve()),
        "public_panel_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "private_verifier_path": str(private_path.resolve()),
        "private_verifier_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "reference_sql_policy": "private_verifier_only",
        "model_messages_contain_reference_sql": False,
        "validation_payload_read_by_training": False,
        "test_payload_accessed": False,
        "cpu_sql_oracle_calls": len(oracle_rows),
        "cpu_sql_oracle_call_budget": 32,
        "database_fingerprint_kind": "physical_sha256_and_logical_canonical_rows_v1",
    }
    _write_json(root / "validation-manifest.json", manifest)
    _write_json(
        root / "frozen-config.json",
        {
            "split": "validation",
            "year": 2023,
            "seeds": {"validation-0": 201, "validation-1": 202},
            "task_order": list(VALIDATION_TEMPLATES),
            "settings": settings.model_dump(),
            "model_messages_contain_reference_sql": False,
            "private_verifier_path": str(private_path.resolve()),
            "test_payload_accessed": False,
        },
    )
    _write_json(
        output / "commands.json",
        {
            "prepare_validation": (
                f"python scripts/evaluation/run_qwen3_runtime_panel.py --output {output} --prepare"
            ),
            "source": "run_qwen3_runtime_panel.py",
            "created_at_utc": _now(),
        },
    )
    write_run_manifest(output)
    return manifest


def _explicit_infrastructure_errors(trajectory: Any) -> list[str]:
    errors = []
    for attempt in trajectory.model_attempts:
        values = [attempt.get(key) for key in ("error", "engine_error") if attempt.get(key)]
        if attempt.get("server_error"):
            values.append(attempt["server_error"])
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
        if any(token in text for token in INFRA_TEXT) or any(code in text for code in INFRA_CODES):
            errors.append(text[:200])
    return sorted(set(errors))


def _business_category(trajectory: Any, reward: Any) -> str:
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
    if any(step.tool_name == "sql" and not step.tool_result.ok for step in trajectory.steps):
        return "SQL Error or Recovery Failure"
    if trajectory.terminal_reason == "completed":
        return "Wrong Final Answer"
    return "Other Business Failure"


def persist_engine_events(engine_dir: Path, response_dir: Path) -> list[dict[str, Any]]:
    events_path = engine_dir / "generations.jsonl"
    if not events_path.is_file():
        return []
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    ids = [event.get("id") for event in events]
    if any(not event_id for event_id in ids) or len(ids) != len(set(ids)):
        raise PanelError("engine_event_ids_not_unique")
    response_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        event_id = str(event["id"])
        _write_immutable_json(response_dir / f"event-{event_id}.json", event)
        _write_immutable_json(response_dir / f"request-{event_id}.json", event.get("request", {}))
        _write_immutable_json(
            response_dir / f"response-{event_id}.json",
            {
                "id": event_id,
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "error": event.get("error"),
                "telemetry": event.get("telemetry", {}),
            },
        )
    aggregate_path = response_dir / "engine-events.json"
    if aggregate_path.exists():
        previous = json.loads(aggregate_path.read_text())
        if events[: len(previous)] != previous:
            raise PanelError("engine_events_aggregate_conflict")
    _write_json(response_dir / "engine-events.json", events)
    jsonl = "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n"
    jsonl_path = response_dir / "engine-events.jsonl"
    if jsonl_path.exists():
        previous_jsonl = jsonl_path.read_text()
        if not jsonl.startswith(previous_jsonl):
            raise PanelError("engine_events_jsonl_conflict")
    jsonl_path.write_text(jsonl)
    return events


def _load_private(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if item["task_id"] in result:
                raise PanelError("duplicate_private_task")
            result[item["task_id"]] = item
    return result


def run_worker(
    output: Path,
    *,
    arm: str,
    endpoint: str,
    engine_dir: Path,
    adapter: Path | None,
    parent_adapter: Path | None,
    worker_index: int,
    worker_count: int,
    model_config: Path = MODEL_DEFAULT,
    agent_config: Path = AGENT_DEFAULT,
) -> int:
    if arm not in {"base", "sft", "rl"}:
        raise PanelError("invalid_arm")
    if not os.environ.get("SLURM_JOB_ID"):
        raise PanelError("qwen3_panel_requires_slurm_job")
    manifest = json.loads(
        (output / "evaluation" / "validation" / "validation-manifest.json").read_text()
    )
    local_private = output / "evaluation" / "validation" / "private" / "verifier-inputs.jsonl"
    declared_private = Path(manifest["private_verifier_path"])
    local_private_sha = (
        hashlib.sha256(local_private.read_bytes()).hexdigest() if local_private.is_file() else None
    )
    if local_private_sha == manifest.get("private_verifier_sha256"):
        private_path = local_private
    else:
        private_path = declared_private
    private = _load_private(private_path)
    settings = Settings.load(agent_config)
    model = ModelSettings.load(model_config)
    if model.generation.enable_thinking is not False:
        raise PanelError("qwen3_panel_requires_hard_thinking_off")
    if arm == "base" and adapter is not None:
        raise PanelError("base_arm_must_not_load_adapter")
    if arm == "base" and parent_adapter is not None:
        raise PanelError("base_arm_must_not_load_parent_adapter")
    if arm == "sft" and (adapter is None or parent_adapter is not None):
        raise PanelError("sft_arm_requires_final_adapter")
    if arm == "rl" and (adapter is None or parent_adapter is None):
        raise PanelError("rl_arm_requires_parent_and_delta_adapters")
    model_version = {
        "base": "qwen3-base-frozen-v1",
        "sft": "qwen3-sft-frozen-v1",
        "rl": "qwen3-sft-plus-t034-delta-step2-v1",
    }[arm]
    arm_root = output / "evaluation" / arm
    worker_root = arm_root / f"worker-{worker_index}"
    trajectory_store = TrajectoryStore(worker_root / "trajectories")
    response_dir = worker_root / "response"
    results_path = worker_root / "case-results.jsonl"
    worker_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        worker_root / "worker-meta.json",
        {
            "schema_version": "1.0",
            "arm": arm,
            "model_version": model_version,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "endpoint": endpoint,
            "engine_dir": str(engine_dir.resolve()),
            "adapter": str(adapter.resolve()) if adapter else None,
            "parent_adapter": str(parent_adapter.resolve()) if parent_adapter else None,
            "validation_private_path": str(private_path.resolve()),
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model": model.model_dump(),
            "settings": settings.model_dump(),
        },
    )
    existing = {}
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                if item["task_id"] in existing:
                    raise PanelError("duplicate_case_result")
                existing[item["task_id"]] = item
    policy = LocalModelPolicy(
        endpoint,
        model.model_id,
        model_version,
        model.generation,
    )
    selected = [
        item for index, item in enumerate(manifest["tasks"]) if index % worker_count == worker_index
    ]
    for public_item in selected:
        task_id = public_item["task_id"]
        if task_id in existing and existing[task_id].get("completed"):
            continue
        private_item = private.get(task_id)
        if private_item is None:
            raise PanelError(f"private_task_missing:{task_id}")
        task = BenchmarkTask.model_validate(private_item["task"])
        if task.public.model_dump() != public_item["public_task"]:
            raise PanelError(f"public_private_task_mismatch:{task_id}")
        database = (
            output
            / "evaluation"
            / "validation"
            / "databases"
            / (f"{public_item['database_id']}.sqlite3")
        )
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
                arm_root / "sql-audit.jsonl",
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
            events = persist_engine_events(engine_dir, response_dir)
            event_ids = [attempt.get("engine_event_id") for attempt in trajectory.model_attempts]
            if any(not event_id for event_id in event_ids) or not set(event_ids).issubset(
                {event.get("id") for event in events}
            ):
                raise PanelError(f"trajectory_event_binding_failed:{task_id}")
            case = {
                "schema_version": "1.0",
                "task_id": task_id,
                "arm": arm,
                "model_version": policy.version,
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
                "accepted_actions": len(trajectory.steps),
                "tool_attempts": len([step for step in trajectory.steps if step.tool_name]),
                "elapsed_s": time.monotonic() - started,
                "engine_event_ids": event_ids,
                "completed": True,
            }
        except PanelError:
            _write_json(
                worker_root / "persistence-failure.json",
                {
                    "status": "STOP_NEW_REQUESTS",
                    "task_id": task_id,
                    "message": "event or trajectory persistence/binding failed",
                },
            )
            raise
        except Exception as exc:
            case = {
                "schema_version": "1.0",
                "task_id": task_id,
                "arm": arm,
                "model_version": policy.version,
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
                "accepted_actions": 0,
                "tool_attempts": 0,
                "elapsed_s": time.monotonic() - started,
                "engine_event_ids": [],
                "runner_error": {"type": type(exc).__name__, "message": str(exc)},
                "completed": True,
            }
        _append_unique_jsonl(results_path, case, key="task_id")
        existing[task_id] = case
        persist_engine_events(engine_dir, response_dir)
    persist_engine_events(engine_dir, response_dir)
    summary = {
        "schema_version": "1.0",
        "arm": arm,
        "worker_index": worker_index,
        "planned": len(selected),
        "completed": sum(bool(item.get("completed")) for item in existing.values()),
        "unknown": sum(bool(item.get("unknown")) for item in existing.values()),
        "scored": sum(bool(item.get("scored")) for item in existing.values()),
        "engine_events": len(persist_engine_events(engine_dir, response_dir)),
    }
    _write_json(worker_root / "summary.json", summary)
    return 0 if summary["completed"] == summary["planned"] and summary["unknown"] == 0 else 2


def _load_case_records(output: Path, arm: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_id = {}
    for worker_path in sorted((output / "evaluation" / arm).glob("worker-*/case-results.jsonl")):
        for line in worker_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record["task_id"] in records_by_id:
                    raise PanelError(f"duplicate_case_across_workers:{record['task_id']}")
                records_by_id[record["task_id"]] = record
    ordered = [records_by_id.get(task["task_id"]) for task in manifest["tasks"]]
    if any(record is None for record in ordered):
        raise PanelError(f"{arm}_panel_missing_case_result")
    return ordered


def _step_budget_usage(trajectory: Any) -> tuple[int, int, int]:
    consumed = 0
    refusals = 0
    previous = 0
    for step in trajectory.steps:
        current = int(step.tool_result.metadata.get("tool_calls_used", previous))
        if step.tool_name:
            if current > previous:
                consumed += 1
            else:
                refusals += 1
        previous = current
    return consumed, refusals, previous


def _cycle_metrics(trajectories: list[Any]) -> dict[str, Any]:
    historical = adjacent = error_next = 0
    sql_attempts = sql_workers = sql_ok = sql_rejected = sql_refused = 0
    error_sql_with_later_success = 0
    strict_error_next_success = 0
    for trajectory in trajectories:
        sql_steps = [step for step in trajectory.steps if step.tool_name == "sql"]
        seen: set[str] = set()
        sql_attempts += len(sql_steps)
        previous_tool_calls = 0
        for step in trajectory.steps:
            current_tool_calls = int(
                step.tool_result.metadata.get("tool_calls_used", previous_tool_calls)
            )
            if step.tool_name == "sql":
                query = str(step.action.arguments.get("query", "")).strip()
                if query in seen:
                    historical += 1
                seen.add(query)
                if current_tool_calls > previous_tool_calls:
                    sql_workers += 1
                    if step.tool_result.ok:
                        sql_ok += 1
                    else:
                        sql_rejected += 1
                else:
                    sql_refused += 1
            previous_tool_calls = current_tool_calls
        for previous, current in zip(sql_steps, sql_steps[1:], strict=False):
            previous_query = str(previous.action.arguments.get("query", "")).strip()
            current_query = str(current.action.arguments.get("query", "")).strip()
            if previous_query == current_query:
                adjacent += 1
                if not previous.tool_result.ok:
                    error_next += 1
            if not previous.tool_result.ok and current.tool_result.ok:
                strict_error_next_success += 1
        if any(
            not step.tool_result.ok
            and any(later.tool_result.ok for later in sql_steps[index + 1 :])
            for index, step in enumerate(sql_steps)
        ):
            error_sql_with_later_success += 1
    return {
        "sql_attempts_denominator": sql_attempts,
        "sql_worker_calls": sql_workers,
        "sql_ok": sql_ok,
        "sql_rejected": sql_rejected,
        "sql_budget_refused": sql_refused,
        "sql_execution_rate": sql_ok / sql_workers if sql_workers else None,
        "history_reuse": {
            "count": historical,
            "denominator": sql_attempts,
            "rate": historical / sql_attempts if sql_attempts else None,
            "definition": "same task query.strip() appeared in an earlier SQL attempt",
        },
        "adjacent_sql_repeat": {
            "count": adjacent,
            "denominator": sum(
                max(0, len([step for step in trajectory.steps if step.tool_name == "sql"]) - 1)
                for trajectory in trajectories
            ),
            "rate": adjacent
            / sum(
                max(0, len([step for step in trajectory.steps if step.tool_name == "sql"]) - 1)
                for trajectory in trajectories
            )
            if sum(
                max(0, len([step for step in trajectory.steps if step.tool_name == "sql"]) - 1)
                for trajectory in trajectories
            )
            else None,
            "definition": "neighboring SQL attempts within one task have equal query.strip()",
        },
        "error_then_next_sql_repeat": {
            "count": error_next,
            "denominator": sql_rejected,
            "rate": error_next / sql_rejected if sql_rejected else None,
            "definition": (
                "a rejected SQL attempt is followed by the next same-task SQL "
                "with equal query.strip()"
            ),
        },
        "tasks_with_error_then_later_sql_success": error_sql_with_later_success,
        "strict_error_then_next_sql_success": strict_error_next_success,
        "cross_task_sql_comparison": False,
    }


def merge_arm(output: Path, *, arm: str, agent_config: Path = AGENT_DEFAULT) -> dict[str, Any]:
    manifest = json.loads(
        (output / "evaluation" / "validation" / "validation-manifest.json").read_text()
    )
    settings = Settings.load(agent_config)
    records = _load_case_records(output, arm, manifest)
    case_path = output / "evaluation" / arm / "case-results.jsonl"
    rendered = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    if case_path.exists() and case_path.read_text() != rendered:
        raise PanelError(f"immutable_case_results_conflict:{arm}")
    case_path.write_text(rendered)
    trajectories = []
    for record in records:
        if record.get("trajectory_path"):
            trajectories.append(TrajectoryStore.load(Path(record["trajectory_path"])))
    event_records = []
    for worker_dir in sorted((output / "evaluation" / arm).glob("worker-*")):
        events_file = worker_dir / "response" / "engine-events.json"
        if events_file.is_file():
            event_records.extend(json.loads(events_file.read_text()))
    event_ids = [event.get("id") for event in event_records]
    if any(not item for item in event_ids) or len(event_ids) != len(set(event_ids)):
        raise PanelError(f"{arm}_engine_event_ids_not_unique")
    counts = {
        "planned": len(records),
        "attempted": sum(bool(record["attempted"]) for record in records),
        "qualified": sum(bool(record["qualified"]) for record in records),
        "scored": sum(bool(record["scored"]) for record in records),
        "unknown": sum(bool(record["unknown"]) for record in records),
        "successes": sum(bool(record["success"]) for record in records),
    }
    all_steps = [step for trajectory in trajectories for step in trajectory.steps]
    tool_steps = [step for step in all_steps if step.tool_name]
    behavior_counts = Counter(step.action.kind for step in all_steps)
    behavior_success_counts = Counter(step.action.kind for step in all_steps if step.tool_result.ok)
    consumed = 0
    refusals = 0
    for trajectory in trajectories:
        used, rejected, _ = _step_budget_usage(trajectory)
        consumed += used
        refusals += rejected
    attempts = [attempt for trajectory in trajectories for attempt in trajectory.model_attempts]
    format_errors = Counter(
        attempt.get("error")
        for attempt in attempts
        if attempt.get("error") and not attempt.get("format_valid")
    )
    generation_latencies = [
        float(event.get("telemetry", {}).get("generation_latency_s") or event.get("elapsed_s", 0))
        for event in event_records
    ]
    output_tokens = sum(
        int(event.get("usage", {}).get("completion_tokens", 0)) for event in event_records
    )
    input_tokens = sum(
        int(event.get("usage", {}).get("prompt_tokens", 0)) for event in event_records
    )
    total_generation_s = sum(generation_latencies)
    cycle = _cycle_metrics(trajectories)
    status = (
        "FROZEN_COMPARISON_ARM_COMPLETE"
        if counts["planned"] == 20
        and counts["attempted"] == 20
        and counts["qualified"] == 20
        and counts["scored"] == 20
        and counts["unknown"] == 0
        else "FROZEN_COMPARISON_ARM_INCOMPLETE"
    )
    metrics = {
        "schema_version": "1.0",
        "status": status,
        "arm": arm,
        "model_version": {
            "base": "qwen3-base-frozen-v1",
            "sft": "qwen3-sft-frozen-v1",
            "rl": "qwen3-sft-plus-t034-delta-step2-v1",
        }[arm],
        "counts": counts,
        "planned_denominator": 20,
        "business_success_rate": counts["successes"] / 20
        if status == "FROZEN_COMPARISON_ARM_COMPLETE"
        else None,
        "scored_subset_success_rate": counts["successes"] / counts["scored"]
        if counts["scored"]
        else None,
        "unknown_reasons": sorted(
            {reason for record in records for reason in record.get("unknown_reasons", [])}
        ),
        "format_errors": dict(format_errors),
        "accepted_actions": len(all_steps),
        "accepted_action_kind_counts": dict(sorted(behavior_counts.items())),
        "successful_observation_kind_counts": dict(sorted(behavior_success_counts.items())),
        "model_attempts": len(attempts),
        "model_physical_requests": len(event_records),
        "model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "model_token_usage_known": all(
            "prompt_tokens" in event.get("usage", {})
            and "completion_tokens" in event.get("usage", {})
            for event in event_records
        ),
        "model_generation_latency_s_total": total_generation_s,
        "model_generation_latency_s_mean": total_generation_s / len(event_records)
        if event_records
        else None,
        "model_generation_throughput_tokens_per_s": output_tokens / total_generation_s
        if total_generation_s
        else None,
        "episode_wall_time_s_total": sum(trajectory.elapsed_s for trajectory in trajectories),
        "episode_wall_time_s_mean": (
            sum(trajectory.elapsed_s for trajectory in trajectories) / len(trajectories)
            if trajectories
            else None
        ),
        "peak_allocated_bytes": max(
            (event.get("peak_allocated_bytes") for event in event_records), default=None
        ),
        "peak_reserved_bytes": max(
            (event.get("peak_reserved_bytes") for event in event_records), default=None
        ),
        "tool_attempts": len(tool_steps),
        "tool_budget_consumed": consumed,
        "tool_budget_refused": refusals,
        "runtime_tool_attempt_limit": 600,
        "sql": cycle,
        "termination_reasons": dict(
            sorted(Counter(trajectory.terminal_reason for trajectory in trajectories).items())
        ),
        "step_count": {
            "total": len(all_steps),
            "mean": len(all_steps) / len(trajectories) if trajectories else None,
        },
        "engine_event_ids_unique": len(event_ids) == len(set(event_ids)),
        "engine_events": len(event_records),
        "verifier_calls": counts["scored"],
        "verifier_sql_calls": counts["scored"],
        "reference_sql_content_exposed_to_model": False,
        "settings": settings.model_dump(),
        "task_ids": [record["task_id"] for record in records],
        "database_sha256": [database["physical_sha256"] for database in manifest["databases"]],
        "finished_at_utc": _now(),
    }
    _write_json(output / "evaluation" / arm / "metrics.json", metrics)
    _write_json(output / "evaluation" / arm / "summary.json", metrics)
    return metrics


def compare(output: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output / "evaluation" / "validation" / "validation-manifest.json").read_text()
    )
    base = json.loads((output / "evaluation" / "base" / "metrics.json").read_text())
    sft = json.loads((output / "evaluation" / "sft" / "metrics.json").read_text())
    resource_accounting = {
        "compare_job_id": None,
        "partition": None,
        "node": None,
        "gpu_count": None,
        "wrapper_wall_time_s": None,
        "allocation_elapsed_s": None,
        "allocation_gpu_minutes_shared": None,
        "slurm_accounting_evidence_path": None,
        "currency_cost": None,
        "currency_cost_source": None,
        "service_wall_definition": "shared concurrent Base/SFT compare wrapper wall time",
    }
    ledger_path = output / "allocation-ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
        completed_compare = [
            record
            for record in ledger.get("records", [])
            if record.get("stage") == "compare" and record.get("status") == "COMPLETED"
        ]
        if completed_compare:
            record = completed_compare[-1]
            resource_accounting.update(
                {
                    "compare_job_id": record.get("job_id"),
                    "partition": record.get("partition"),
                    "node": record.get("node"),
                    "gpu_count": record.get("gpu_count"),
                    "wrapper_wall_time_s": record.get("wrapper_elapsed_s"),
                    "allocation_gpu_minutes_shared": record.get("wrapper_gpu_minutes"),
                    "slurm_accounting_evidence_path": record.get("slurm_accounting_evidence_path"),
                }
            )
            evidence_path = record.get("slurm_accounting_evidence_path")
            if evidence_path and Path(evidence_path).is_file():
                lines = [line for line in Path(evidence_path).read_text().splitlines() if line]
                if len(lines) >= 2:
                    fields = lines[1].split("|")
                    if len(fields) >= 3:
                        resource_accounting["allocation_elapsed_s"] = int(fields[2])
    complete = (
        base["status"] == "FROZEN_COMPARISON_ARM_COMPLETE"
        and sft["status"] == "FROZEN_COMPARISON_ARM_COMPLETE"
    )
    if base["task_ids"] != sft["task_ids"] or base["task_ids"] != manifest["task_ids"]:
        raise PanelError("comparison_task_set_mismatch")
    if base["database_sha256"] != sft["database_sha256"]:
        raise PanelError("comparison_database_mismatch")
    base_records = {item["task_id"]: item for item in _load_case_records(output, "base", manifest)}
    sft_records = {item["task_id"]: item for item in _load_case_records(output, "sft", manifest)}
    transitions = [
        {
            "task_id": task_id,
            "base_success": bool(base_records[task_id]["success"]),
            "sft_success": bool(sft_records[task_id]["success"]),
            "base_category": base_records[task_id]["category"],
            "sft_category": sft_records[task_id]["category"],
            "base_terminal_reason": base_records[task_id]["terminal_reason"],
            "sft_terminal_reason": sft_records[task_id]["terminal_reason"],
        }
        for task_id in manifest["task_ids"]
    ]
    comparison = {
        "schema_version": "1.0",
        "status": "FROZEN_COMPARISON_COMPLETE" if complete else "FROZEN_COMPARISON_INCOMPLETE",
        "experiment": "T-032/E-029",
        "task_ids": manifest["task_ids"],
        "planned_denominator": 20,
        "base": {
            "status": base["status"],
            "successes": base["counts"]["successes"],
            "success_rate": base["business_success_rate"],
            "unknown": base["counts"]["unknown"],
        },
        "sft": {
            "status": sft["status"],
            "successes": sft["counts"]["successes"],
            "success_rate": sft["business_success_rate"],
            "unknown": sft["counts"]["unknown"],
        },
        "primary_endpoint": {
            "sft_successes_minus_base_successes": (
                sft["counts"]["successes"] - base["counts"]["successes"] if complete else None
            ),
            "sft_success_rate_minus_base_success_rate": (
                sft["business_success_rate"] - base["business_success_rate"] if complete else None
            ),
            "fixed_denominator": 20,
        },
        "transitions": transitions,
        "model_identity": {
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "base_adapter": "NONE",
            "sft_adapter": str((output / "training" / "final-adapter").resolve()),
            "checkpoint_selection": "step27_fixed_before_scoring",
        },
        "resource_accounting": resource_accounting,
        "validation_manifest_sha256": hashlib.sha256(
            (output / "evaluation" / "validation" / "validation-manifest.json").read_bytes()
        ).hexdigest(),
        "finished_at_utc": _now(),
    }
    _write_json(output / "comparison.json", comparison)
    for arm_name, arm_metrics in (("base", base), ("sft", sft)):
        arm_metrics["resource_accounting"] = {
            **resource_accounting,
            "arm_share": "shared_concurrent_compare_job",
        }
        _write_json(output / "evaluation" / arm_name / "metrics.json", arm_metrics)
        _write_json(output / "evaluation" / arm_name / "summary.json", arm_metrics)
    _write_json(
        output / "metrics.json",
        {
            "schema_version": "1.0",
            "status": comparison["status"],
            "primary_endpoint": comparison["primary_endpoint"],
            "arms": {"base": base, "sft": sft},
            "comparison": str((output / "comparison.json").resolve()),
        },
    )
    training_metrics_path = output / "training" / "training-metrics.json"
    reload_check_path = output / "training" / "reload-check.json"
    training_metrics = (
        json.loads(training_metrics_path.read_text()) if training_metrics_path.is_file() else {}
    )
    reload_check = json.loads(reload_check_path.read_text()) if reload_check_path.is_file() else {}
    base_frozen = training_metrics.get(
        "base_parameters_changed"
    ) is False and not training_metrics.get("base_gradients_present")
    report_lines = [
        "# T-032 / E-029 — Qwen3 Runtime SFT与冻结Base/SFT对照",
        "",
        f"- 总状态：`{comparison['status']}`",
        f"- 代码快照：`{_git_head()}`",
        "- 主终点：SFT successes − Base successes = "
        f"`{comparison['primary_endpoint']['sft_successes_minus_base_successes']}`；"
        "固定分母率差 = "
        f"`{comparison['primary_endpoint']['sft_success_rate_minus_base_success_rate']}`",
        "",
        "## 技术资格",
        "",
        "- 训练状态："
        f"`{training_metrics.get('status', 'NOT_REPORTED')}`；54条、108完成微批、"
        f"27次有效更新；backward={training_metrics.get('backward_count')}",
        "- Base冻结："
        f"`{base_frozen}`；LoRA变化参数="
        f"{training_metrics.get('lora_changed_parameter_count')}",
        "- 独立重载："
        f"`{reload_check.get('status', 'NOT_REPORTED')}`；"
        f"finite logits=`{reload_check.get('finite_logits')}`；"
        f"Adapter SHA=`{reload_check.get('adapter_sha256')}`",
        "",
        "## 冻结比较",
        "",
        "- Base："
        f"{base['counts']['successes']}/20，scored={base['counts']['scored']}，"
        f"unknown={base['counts']['unknown']}，状态=`{base['status']}`",
        "- SFT step27："
        f"{sft['counts']['successes']}/20，scored={sft['counts']['scored']}，"
        f"unknown={sft['counts']['unknown']}，状态=`{sft['status']}`",
        "",
        "逐任务变化见 [comparison.json](comparison.json) 的 `transitions`，"
        "两臂完整时主终点才从 null 提升为数值。请求、response、engine event、"
        "SQL audit、trajectory 与 verifier 结果分别保存在 `evaluation/base/` "
        "和 `evaluation/sft/`。",
        "",
        "## 复现与边界",
        "",
        "- train-only 源为固定54条 Runtime decision；validation 仅使用本run冻结的 "
        "validation-0/1，test 未打开；私有 verifier 输入不进入 policy messages "
        "或训练进程。",
        "- checkpoint 规则在评分前固定为 step27；不按验证结果选 checkpoint，"
        "不扩训、不扫参、不启动 GRPO。",
        "- 资源与 Job 证据见 [allocation-ledger.json](allocation-ledger.json)、"
        "[job-chain.json](job-chain.json)；wrapper GPU分钟与 Slurm allocation evidence 分列。",
        f"- 并发 compare wrapper wall=`{resource_accounting['wrapper_wall_time_s']}`秒，"
        f"共享 allocation GPU分钟=`{resource_accounting['allocation_gpu_minutes_shared']}`；"
        "货币成本=`null`（无计费来源）。",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n")
    write_run_manifest(output)
    return comparison


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--arm", choices=("base", "sft", "rl"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--endpoint")
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--parent-adapter", type=Path)
    parser.add_argument("--model-config", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--agent-config", type=Path, default=AGENT_DEFAULT)
    args = parser.parse_args()
    selected = sum(
        bool(value)
        for value in (args.prepare, args.merge, args.compare, args.worker_index is not None)
    )
    if selected != 1:
        raise SystemExit("choose exactly one of --prepare, --merge, --compare, --worker-index")
    if args.prepare:
        prepare_validation(args.output, agent_config=args.agent_config)
        return 0
    if args.compare:
        result = compare(args.output)
        return 0 if result["status"] == "FROZEN_COMPARISON_COMPLETE" else 2
    if args.merge:
        if args.arm is None:
            raise SystemExit("--merge requires --arm")
        metrics = merge_arm(args.output, arm=args.arm, agent_config=args.agent_config)
        return 0 if metrics["status"] == "FROZEN_COMPARISON_ARM_COMPLETE" else 2
    if (
        args.arm is None
        or args.worker_index is None
        or not args.endpoint
        or args.engine_dir is None
    ):
        raise SystemExit("worker requires --arm --worker-index --endpoint --engine-dir")
    if not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker-index-out-of-range")
    return run_worker(
        args.output,
        arm=args.arm,
        endpoint=args.endpoint,
        engine_dir=args.engine_dir,
        adapter=args.adapter,
        parent_adapter=args.parent_adapter,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        model_config=args.model_config,
        agent_config=args.agent_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
