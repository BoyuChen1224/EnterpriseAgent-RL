"""CPU-only scoring and closeout for one immutable saved trajectory.

This module deliberately has no model, tokenizer, torch, Slurm, or training
imports.  It validates the persisted model evidence, calls the production
``rewards.verify`` exactly for the explicitly selected case, and builds a
read-only merged view over the historical T-024 indexes.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.rewards import verify as production_verify
from enterprise_agent_rl.trajectories import RewardResult, Trajectory

from .aggregation import aggregate_source, three_arm_comparison
from .contracts import EpisodeSample, TokenRecord
from .evidence import qualify_episode_evidence
from .io import (
    atomic_write_json,
    canonical_hash,
    read_json,
    read_jsonl,
    safe_relative_path,
    sha256_file,
)

SOURCE_MANIFEST_SHA256 = "df79597e6ec1f7c8ef287627367c807480a26bc65d5f5497b9eb25797b271825"
TARGET_TRAJECTORY_SHA256 = "6fb1ef6aa0a411655c8d42a01b512c7cdaa16d9a4add523728fe992a7faa4607"
TARGET_DATABASE_SHA256 = "7c9501257be13290c97b33cc59c72f0e496d1918b2d5453fccd18078a2040c75"
TARGET_LOGICAL_DATABASE_SHA256 = "5fd46ef8f2c41d02a88cb826e306c30f29ee6636cdd48697dd2eb949ebe92c32"
TARGET_POLICY_VERSION = "frozen-99e471b896886783-RL"
TARGET_ARM = "RL"
TARGET_TASK_ID = "validation-1-monthly"
TARGET_EPISODE_ID = "RL-validation-1-monthly"
HISTORICAL_VERIFIER_CALLS = 60
HISTORICAL_VERIFIER_SQL = 66
HISTORICAL_MODEL_CALLS = 519
HISTORICAL_GPU_ALLOCATION_SECONDS = 2614.0
MAX_VERIFIER_ATTEMPTS = 2
MAX_VERIFIER_SQL_PER_ATTEMPT = 2
MAX_VERIFIER_SQL_TOTAL = 4


class SavedScoringError(RuntimeError):
    """The selected immutable saved case is not safe to score or merge."""


def _json_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SavedScoringError(f"json_object_required:{path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an index atomically and accept only a canonical exact replay."""

    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    if path.exists():
        try:
            existing = read_jsonl(path)
        except (OSError, ValueError) as exc:
            raise SavedScoringError(f"immutable_jsonl_unreadable:{path}") from exc
        if canonical_hash(existing) != canonical_hash(rows):
            raise SavedScoringError(f"immutable_jsonl_conflict:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _choose_output_dir(requested: Path) -> tuple[Path, bool]:
    """Keep a recognized closeout idempotent; suffix unrelated directories."""

    requested = requested.expanduser().resolve()
    if not requested.exists() or not any(requested.iterdir()):
        return requested, False
    if (requested / "supplemental-score.json").is_file() or (
        requested / "call-ledger.json"
    ).is_file():
        return requested, True
    index = 2
    while True:
        candidate = requested.parent / f"{requested.name}-r{index}"
        if not candidate.exists():
            return candidate, False
        index += 1


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Capture branch, HEAD, and raw dirty paths without staging anything."""

    root = repo_root.resolve()

    def run_git(*arguments: str) -> dict[str, Any]:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return {
            "argv": ["git", *arguments],
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    branch = run_git("symbolic-ref", "--short", "HEAD")
    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short", "--branch", "--untracked-files=all")
    status_lines = status["stdout"].splitlines()
    return {
        "schema_version": "t025-git-state-v1",
        "captured_at_epoch": time.time(),
        "repo_root": str(root),
        "branch": branch["stdout"].strip(),
        "head": head["stdout"].strip(),
        "dirty": bool(status_lines[1:] if status_lines[:1] else status_lines),
        "status_porcelain_v1": status["stdout"],
        "status_command": status,
        "allowed_task_edit_paths": [
            "src/enterprise_agent_rl/grpo/aggregation.py",
            "src/enterprise_agent_rl/grpo/cli.py",
            "src/enterprise_agent_rl/grpo/evidence.py",
            "src/enterprise_agent_rl/grpo/saved_scoring.py",
            "tests/test_grpo_aggregation.py",
            "tests/test_grpo_saved_scoring.py",
            "docs/grpo-training.md",
        ],
        "no_commit_push_branch_switch": True,
    }


def _source_manifest_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise SavedScoringError("source_manifest_files_missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise SavedScoringError("source_manifest_row_invalid")
        path = row["path"]
        if path in result:
            raise SavedScoringError(f"source_manifest_duplicate:{path}")
        result[path] = row
    return result


def _target_private_task(private_plan: Path, task_id: str) -> tuple[BenchmarkTask, dict[str, Any]]:
    value = _json_object(private_plan)
    records = value.get("records", value.get("tasks"))
    if not isinstance(records, list):
        raise SavedScoringError("private_task_plan_records_missing")
    selected: dict[str, Any] | None = None
    for record in records:
        if not isinstance(record, dict):
            raise SavedScoringError("private_task_plan_record_invalid")
        task_value = record.get("task") if isinstance(record.get("task"), dict) else record
        public = task_value.get("public") if isinstance(task_value, dict) else None
        record_task_id = (
            task_value.get("task_id")
            if isinstance(task_value, dict)
            else None
        ) or (public.get("task_id") if isinstance(public, dict) else None)
        if record_task_id == task_id:
            if selected is not None:
                raise SavedScoringError(f"private_task_duplicate:{task_id}")
            selected = task_value
    if selected is None:
        raise SavedScoringError(f"private_task_missing:{task_id}")
    try:
        benchmark = BenchmarkTask.model_validate(selected)
    except Exception as exc:
        raise SavedScoringError(f"private_task_invalid:{task_id}:{exc}") from exc
    return benchmark, selected


def _load_saved_episode(
    *,
    source_root: Path,
    trajectory_path: Path,
    trajectory_value: dict[str, Any],
    arm: str,
) -> tuple[Trajectory, dict[str, Any], EpisodeSample]:
    try:
        trajectory = Trajectory.model_validate(trajectory_value)
    except Exception as exc:
        raise SavedScoringError(f"trajectory_contract_invalid:{exc}") from exc
    attempts = trajectory_value.get("model_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise SavedScoringError("trajectory_model_attempts_missing")
    records: list[TokenRecord] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise SavedScoringError("trajectory_attempt_not_object")
        response_path_value = attempt.get("response_path")
        if not isinstance(response_path_value, str) or not response_path_value:
            raise SavedScoringError("trajectory_response_path_missing")
        response_path = safe_relative_path(source_root, response_path_value)
        response = _json_object(response_path)
        token_record = response.get("token_record")
        if not isinstance(token_record, dict):
            raise SavedScoringError(f"response_token_record_missing:{attempt.get('request_id')}")
        try:
            records.append(TokenRecord.from_dict(token_record))
        except Exception as exc:
            raise SavedScoringError(
                f"response_token_record_invalid:{attempt.get('request_id')}:{exc}"
            ) from exc
    if len(records) != len(attempts) or len({record.request_id for record in records}) != len(
        records
    ):
        raise SavedScoringError("trajectory_model_attempt_cardinality_invalid")
    first = records[0]
    if any(record.episode_id != first.episode_id for record in records):
        raise SavedScoringError("trajectory_token_episode_mismatch")
    episode = EpisodeSample(
        episode_id=first.episode_id,
        task_id=first.task_id,
        group_id=first.group_id,
        sample_index=first.sample_index,
        seed=first.seed,
        policy_snapshot_id=first.policy_snapshot_id,
        model_ref_id=first.model_ref_id,
        token_records=records,
        metadata={"trajectory": trajectory_value},
    )
    try:
        qualification = qualify_episode_evidence(
            episode,
            run_dir=source_root,
            expected_stage="evaluation",
            expected_arm=arm,
            trajectory_path=_relative(trajectory_path, source_root),
        )
    except Exception as exc:
        raise SavedScoringError(f"saved_evidence_unqualified:{exc}") from exc
    if qualification.get("status") != "QUALIFIED":
        raise SavedScoringError("saved_evidence_not_qualified")
    return trajectory, qualification, episode


def _task_ids_from_frozen_plan(plan: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 20:
        raise SavedScoringError("frozen_plan_requires_20_tasks")
    task_ids: list[str] = []
    selected: dict[str, Any] | None = None
    for row in tasks:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise SavedScoringError("frozen_plan_task_invalid")
        task_id = row["task_id"]
        if task_id in task_ids:
            raise SavedScoringError(f"frozen_plan_duplicate_task:{task_id}")
        task_ids.append(task_id)
        if task_id == TARGET_TASK_ID:
            selected = row
    if selected is None:
        raise SavedScoringError(f"frozen_plan_target_missing:{TARGET_TASK_ID}")
    return task_ids, selected


def validate_saved_target(
    *,
    source_run: Path,
    private_task_plan: Path,
    arm: str = TARGET_ARM,
    task_id: str = TARGET_TASK_ID,
    strict_identity: bool = True,
) -> dict[str, Any]:
    """Validate source identity, public task, DB, settings, and all 20 calls."""

    source_root = source_run.expanduser().resolve()
    if not source_root.is_dir():
        raise SavedScoringError(f"source_run_missing:{source_root}")
    if arm != TARGET_ARM or task_id != TARGET_TASK_ID:
        raise SavedScoringError("saved_scoring_scope_is_rl_validation_1_monthly")
    manifest_path = source_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    if strict_identity and manifest_sha != SOURCE_MANIFEST_SHA256:
        raise SavedScoringError(f"source_manifest_sha256_mismatch:{manifest_sha}")
    manifest = _json_object(manifest_path)
    manifest_rows = _source_manifest_rows(manifest)
    if strict_identity and len(manifest_rows) != 2046:
        raise SavedScoringError(f"source_manifest_file_count_mismatch:{len(manifest_rows)}")
    plan_path = source_root / "inputs" / "frozen-plan.json"
    protocol_path = source_root / "protocol.json"
    fingerprint_path = source_root / "inputs" / "fingerprint-index.json"
    target_trajectory_path = (
        source_root / "evaluation" / arm / "trajectories" / f"{arm}-{task_id}.json"
    )
    database_path = source_root / "inputs" / "databases" / "validation-1.sqlite3"
    for path in (plan_path, protocol_path, fingerprint_path, target_trajectory_path, database_path):
        if not path.is_file():
            raise SavedScoringError(f"source_file_missing:{path}")
    trajectory_sha = sha256_file(target_trajectory_path)
    database_sha = sha256_file(database_path)
    if strict_identity and trajectory_sha != TARGET_TRAJECTORY_SHA256:
        raise SavedScoringError(f"target_trajectory_sha256_mismatch:{trajectory_sha}")
    if strict_identity and database_sha != TARGET_DATABASE_SHA256:
        raise SavedScoringError(f"target_database_sha256_mismatch:{database_sha}")
    for relative, expected in (
        (_relative(target_trajectory_path, source_root), trajectory_sha),
        (_relative(database_path, source_root), database_sha),
    ):
        row = manifest_rows.get(relative)
        if not isinstance(row, dict) or row.get("sha256") != expected:
            raise SavedScoringError(f"source_manifest_entry_mismatch:{relative}")

    plan = _json_object(plan_path)
    task_ids, public_row = _task_ids_from_frozen_plan(plan)
    protocol = _json_object(protocol_path)
    runtime = protocol.get("runtime")
    if not isinstance(runtime, dict):
        raise SavedScoringError("protocol_runtime_missing")
    try:
        protocol_settings = Settings.model_validate(runtime)
    except Exception as exc:
        raise SavedScoringError(f"protocol_settings_invalid:{exc}") from exc
    trajectory_value = _json_object(target_trajectory_path)
    try:
        trajectory_settings = Settings.model_validate(trajectory_value.get("settings"))
    except Exception as exc:
        raise SavedScoringError(f"trajectory_settings_invalid:{exc}") from exc
    if protocol_settings.model_dump() != trajectory_settings.model_dump():
        raise SavedScoringError("trajectory_settings_protocol_mismatch")
    if trajectory_value.get("database_sha256") != database_sha:
        raise SavedScoringError("trajectory_database_identity_mismatch")
    if trajectory_value.get("policy_version") != TARGET_POLICY_VERSION:
        raise SavedScoringError("trajectory_policy_version_mismatch")
    if trajectory_value.get("task") != public_row.get("public_task"):
        raise SavedScoringError("trajectory_public_task_mismatch")

    benchmark, private_task = _target_private_task(private_task_plan, task_id)
    public_value = benchmark.public.model_dump()
    if public_value != public_row.get("public_task"):
        raise SavedScoringError("private_public_task_mismatch")
    if benchmark.split != "validation":
        raise SavedScoringError("private_task_split_mismatch")
    private_database_sha = None
    records = _json_object(private_task_plan).get("records", [])
    for record in records:
        if not isinstance(record, dict):
            continue
        task_value = record.get("task") if isinstance(record.get("task"), dict) else record
        task_public = task_value.get("public") if isinstance(task_value, dict) else None
        if isinstance(task_public, dict) and task_public.get("task_id") == task_id:
            private_database_sha = record.get("database_sha256")
            break
    if private_database_sha != TARGET_LOGICAL_DATABASE_SHA256:
        raise SavedScoringError(f"logical_database_fingerprint_mismatch:{private_database_sha}")

    fingerprint_index = _json_object(fingerprint_path)
    fingerprint_rows = fingerprint_index.get("rows")
    fingerprint_row = next(
        (
            row
            for row in fingerprint_rows or []
            if isinstance(row, dict) and row.get("task_id") == task_id
        ),
        None,
    )
    if not isinstance(fingerprint_row, dict):
        raise SavedScoringError("target_fingerprint_index_row_missing")
    source_metadata = fingerprint_row.get("source_metadata")
    if (
        not isinstance(source_metadata, dict)
        or source_metadata.get("database_sha256") != TARGET_LOGICAL_DATABASE_SHA256
    ):
        raise SavedScoringError("target_logical_fingerprint_index_mismatch")

    verifier_index_path = source_root / "evaluation" / arm / "verifier-results.jsonl"
    verifier_rows = [
        row
        for row in read_jsonl(verifier_index_path)
        if row.get("record_type", "RESULT") == "RESULT"
    ]
    target_rows = [row for row in verifier_rows if row.get("task_id") == task_id]
    if len(target_rows) != 1:
        raise SavedScoringError(f"target_source_result_cardinality:{len(target_rows)}")
    original_unknown = target_rows[0]
    if (
        original_unknown.get("status") != "unknown"
        or original_unknown.get("verifier_called") is not False
        or original_unknown.get("trajectory_sha256") != trajectory_sha
        or original_unknown.get("qualification", {}).get("status") != "QUALIFIED"
    ):
        raise SavedScoringError("target_source_unknown_identity_mismatch")
    if original_unknown.get("episode_id") != TARGET_EPISODE_ID:
        raise SavedScoringError("target_source_episode_id_mismatch")
    arm_manifest = _json_object(source_root / "evaluation" / arm / "arm-manifest.json")
    snapshot = arm_manifest.get("arm_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("policy_snapshot_id") != TARGET_POLICY_VERSION
    ):
        raise SavedScoringError("target_arm_policy_snapshot_mismatch")
    trajectory, qualification, episode = _load_saved_episode(
        source_root=source_root,
        trajectory_path=target_trajectory_path,
        trajectory_value=trajectory_value,
        arm=arm,
    )
    if (
        qualification.get("attempted_model_calls") != 20
        or qualification.get("completed_model_calls") != 20
    ):
        raise SavedScoringError("target_model_call_count_mismatch")
    if qualification.get("trajectory_sha256") != trajectory_sha:
        raise SavedScoringError("qualified_trajectory_hash_mismatch")
    source_allocation = _json_object(source_root / "allocation-ledger.json")
    budget_state = _json_object(source_root / "budget-state.json")
    source_metrics = _json_object(source_root / "metrics.json")
    source_model_calls = sum(
        int(source_metrics.get(arm_name, {}).get("attempted_model_calls", 0))
        for arm_name in ("Base", "N", "RL")
    )
    if strict_identity and source_model_calls != HISTORICAL_MODEL_CALLS:
        raise SavedScoringError(f"source_model_call_count_mismatch:{source_model_calls}")
    return {
        "source_root": source_root,
        "source_manifest_path": manifest_path,
        "source_manifest_sha256": manifest_sha,
        "source_manifest_file_count": len(manifest_rows),
        "manifest_rows": manifest_rows,
        "plan_path": plan_path,
        "plan_sha256": sha256_file(plan_path),
        "task_ids": task_ids,
        "public_task": copy.deepcopy(public_row.get("public_task")),
        "public_task_row": public_row,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_runtime": protocol_settings.model_dump(mode="json"),
        "protocol_fingerprint": canonical_hash(protocol_settings.model_dump(mode="json")),
        "private_task_plan": private_task_plan.expanduser().resolve(),
        "private_task_plan_sha256": sha256_file(private_task_plan),
        "benchmark_task": benchmark,
        "private_task": private_task,
        "trajectory_path": target_trajectory_path,
        "trajectory_sha256": trajectory_sha,
        "trajectory_value": trajectory_value,
        "trajectory": trajectory,
        "episode": episode,
        "qualification": qualification,
        "database_path": database_path,
        "database_sha256": database_sha,
        "logical_database_sha256": TARGET_LOGICAL_DATABASE_SHA256,
        "fingerprint_index_path": fingerprint_path,
        "fingerprint_row": fingerprint_row,
        "verifier_index_path": verifier_index_path,
        "verifier_index_sha256": sha256_file(verifier_index_path),
        "original_unknown": copy.deepcopy(original_unknown),
        "policy_version": TARGET_POLICY_VERSION,
        "source_allocation": source_allocation,
        "source_budget_state": budget_state,
        "source_model_calls": source_model_calls,
        "historical_verifier_calls": int(
            budget_state.get("call_counters", {}).get(
                "verifier_calls", HISTORICAL_VERIFIER_CALLS
            )
        ),
        "historical_verifier_sql": int(
            budget_state.get("call_counters", {}).get("verifier_sql", HISTORICAL_VERIFIER_SQL)
        ),
    }


class _CapturedSQL:
    """Patch only SQLTool's subprocess boundary while production verify runs."""

    def __init__(self, log_dir: Path, *, max_queries: int = MAX_VERIFIER_SQL_PER_ATTEMPT):
        self.log_dir = log_dir
        self.max_queries = max_queries
        self.calls: list[dict[str, Any]] = []
        self._original: Any = None

    def __enter__(self) -> _CapturedSQL:
        from enterprise_agent_rl.tools import SQLTool

        self._original = SQLTool._invoke

        def invoke(request: dict[str, Any], timeout: float) -> dict[str, Any]:
            call_index = len(self.calls) + 1
            if call_index > self.max_queries:
                raise SavedScoringError("verifier_sql_per_attempt_limit_exceeded")
            started = time.monotonic()
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "sql_worker.py")]
            stdout = ""
            stderr = ""
            return_code: int | None = None
            result: dict[str, Any]
            try:
                completed = subprocess.run(
                    command,
                    input=json.dumps(request),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                return_code = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                if completed.returncode == 0:
                    try:
                        parsed = json.loads(stdout)
                        result = (
                            parsed
                            if isinstance(parsed, dict)
                            else {"ok": False, "error": "query_worker_failure"}
                        )
                    except (TypeError, ValueError):
                        result = {"ok": False, "error": "query_worker_failure"}
                else:
                    result = {"ok": False, "error": "query_worker_resource_limit_or_failure"}
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    exc.stdout.decode()
                    if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")
                )
                stderr = (
                    exc.stderr.decode()
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")
                )
                result = {"ok": False, "error": "query_timeout"}
            except (ValueError, OSError):
                result = {"ok": False, "error": "query_worker_failure"}
            record = {
                "schema_version": "t025-verifier-sql-call-v1",
                "call_index": call_index,
                "request": request,
                "command": command,
                "worker_script": command[-1],
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": return_code,
                "elapsed_s": time.monotonic() - started,
                "result": result,
            }
            self.calls.append(record)
            atomic_write_json(self.log_dir / f"sql-call-{call_index:02d}.json", record)
            return result

        SQLTool._invoke = staticmethod(invoke)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        from enterprise_agent_rl.tools import SQLTool

        SQLTool._invoke = staticmethod(self._original)


def _trajectory_runtime_counts(trajectory: Trajectory) -> dict[str, Any]:
    sql_steps = [step for step in trajectory.steps if step.tool_name == "sql"]
    non_sql_steps = [
        step
        for step in trajectory.steps
        if step.tool_name in {"schema", "dictionary", "check"}
    ]

    def is_budget_error(step: Any) -> bool:
        return "budget" in str(step.tool_result.error or "").lower()

    errors = [step.tool_result.error for step in sql_steps if not step.tool_result.ok]
    return {
        "sql_action_count": len(sql_steps),
        "sql_ok_count": sum(bool(step.tool_result.ok) for step in sql_steps),
        "runtime_worker_sql_count": sum(not is_budget_error(step) for step in sql_steps),
        "runtime_worker_non_sql_count": sum(not is_budget_error(step) for step in non_sql_steps),
        "worker_count": sum(not is_budget_error(step) for step in sql_steps + non_sql_steps),
        "budget_rejection_count": sum("budget" in str(error or "").lower() for error in errors),
        "error": errors,
    }


def _score_row(
    *,
    source: dict[str, Any],
    reward: RewardResult,
    verifier_sql_count: int,
) -> dict[str, Any]:
    counts = _trajectory_runtime_counts(source["trajectory"])
    reward_value = reward.model_dump(mode="json")
    return {
        "schema_version": "grpo-score-v1",
        "record_type": "RESULT",
        "episode_id": TARGET_EPISODE_ID,
        "task_id": TARGET_TASK_ID,
        "status": "scored",
        "reward_value": float(reward.value),
        "success": bool(reward.success),
        "reason": reward.reason,
        "sql_action_count": counts["sql_action_count"],
        "sql_ok_count": counts["sql_ok_count"],
        "worker_count": counts["worker_count"],
        "runtime_worker_sql_count": counts["runtime_worker_sql_count"],
        "runtime_worker_non_sql_count": counts["runtime_worker_non_sql_count"],
        "verifier_sql_count": verifier_sql_count,
        "budget_rejection_count": counts["budget_rejection_count"],
        "infra_error": None,
        "error": counts["error"],
        "verifier_record_id": TARGET_EPISODE_ID,
        "source_kind": "t025-supplemental-verifier",
        "metadata": {
            "reward": reward_value,
            "supplemental_task": "T-025/E-026",
            "model_identity_source": "T-024-frozen-trajectory",
        },
        "policy_snapshot_id": source["policy_version"],
        "raw_saved": True,
        "verifier_called": True,
        "trajectory_path": _relative(source["trajectory_path"], source["source_root"]),
        "trajectory_sha256": source["trajectory_sha256"],
        "qualification": source["qualification"],
        "trajectory_source_root": str(source["source_root"]),
        "source_manifest_sha256": source["source_manifest_sha256"],
    }


def _write_source_index(output: Path, source: dict[str, Any]) -> None:
    atomic_write_json(
        output / "source-index.json",
        {
            "schema_version": "t025-source-index-v1",
            "source_root": str(source["source_root"]),
            "source_manifest": {
                "path": str(source["source_manifest_path"]),
                "sha256": source["source_manifest_sha256"],
                "file_count": source["source_manifest_file_count"],
            },
            "source_manifest_is_read_only": True,
            "frozen_plan": {
                "path": str(source["plan_path"]),
                "sha256": source["plan_sha256"],
                "task_ids": source["task_ids"],
                "target_task_id": TARGET_TASK_ID,
            },
            "protocol": {
                "path": str(source["protocol_path"]),
                "sha256": source["protocol_sha256"],
                "runtime": source["protocol_runtime"],
                "fingerprint": source["protocol_fingerprint"],
            },
            "private_task_source": {
                "path": str(source["private_task_plan"]),
                "sha256": source["private_task_plan_sha256"],
                "used_only_for_production_verifier": True,
            },
            "target": {
                "arm": TARGET_ARM,
                "task_id": TARGET_TASK_ID,
                "episode_id": TARGET_EPISODE_ID,
                "trajectory_path": _relative(source["trajectory_path"], source["source_root"]),
                "trajectory_sha256": source["trajectory_sha256"],
                "policy_version": source["policy_version"],
                "qualification": source["qualification"],
                "original_unknown": {
                    "status": source["original_unknown"].get("status"),
                    "unknown_reason": source["original_unknown"].get("unknown_reason"),
                    "verifier_called": source["original_unknown"].get("verifier_called"),
                    "trajectory_sha256": source["original_unknown"].get("trajectory_sha256"),
                    "index_path": str(source["verifier_index_path"]),
                    "index_sha256": source["verifier_index_sha256"],
                },
            },
            "database": {
                "source_path": str(source["database_path"]),
                "physical_sha256": source["database_sha256"],
                "logical_fingerprint": source["logical_database_sha256"],
                "source_fingerprint_index": source["fingerprint_row"],
            },
            "historical_accounting": {
                "verifier_calls": source["historical_verifier_calls"],
                "verifier_sql": source["historical_verifier_sql"],
                "model_calls": source["source_model_calls"],
                "gpu_allocation_seconds": source["source_allocation"].get(
                    "allocation_elapsed_seconds_total"
                ),
                "old_budget_status": source["source_budget_state"].get("status"),
            },
        },
    )


def _copy_read_only_database(output: Path, source: dict[str, Any]) -> Path:
    target = output / "inputs" / "databases" / "validation-1.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != source["database_sha256"]:
            raise SavedScoringError("copied_database_hash_conflict")
    else:
        shutil.copy2(source["database_path"], target)
    if sha256_file(source["database_path"]) != source["database_sha256"]:
        raise SavedScoringError("source_database_changed_before_scoring")
    if sha256_file(target) != source["database_sha256"]:
        raise SavedScoringError("copied_database_hash_mismatch")
    target.chmod(0o444)
    if sha256_file(target) != source["database_sha256"]:
        raise SavedScoringError("copied_database_changed_after_readonly")
    return target


def _load_existing_supplement(output: Path, source: dict[str, Any]) -> dict[str, Any] | None:
    path = output / "supplemental-score.json"
    if not path.is_file():
        return None
    value = _json_object(path)
    if value.get("source_manifest_sha256") != source["source_manifest_sha256"]:
        raise SavedScoringError("supplement_source_manifest_mismatch")
    if value.get("trajectory_sha256") != source["trajectory_sha256"]:
        raise SavedScoringError("supplement_trajectory_mismatch")
    if value.get("task_id") != TARGET_TASK_ID or value.get("arm") != TARGET_ARM:
        raise SavedScoringError("supplement_identity_mismatch")
    if value.get("status") not in {"scored", "unknown"}:
        raise SavedScoringError("supplement_status_invalid")
    score_row = value.get("score_row")
    if not isinstance(score_row, dict) or score_row.get("status") != value.get("status"):
        raise SavedScoringError("supplement_score_row_missing")
    return value


def _write_call_ledger(
    output: Path, source: dict[str, Any], attempts: list[dict[str, Any]]
) -> None:
    atomic_write_json(
        output / "call-ledger.json",
        {
            "schema_version": "t025-call-ledger-v1",
            "scope": "RL/validation-1-monthly only",
            "historical": {
                "verifier_calls": source["historical_verifier_calls"],
                "verifier_sql": source["historical_verifier_sql"],
                "verifier_limit": 60,
                "budget_status": source["source_budget_state"].get("status"),
            },
            "independent_task_budget": {
                "planned_verifier_calls": 1,
                "max_verifier_attempts": MAX_VERIFIER_ATTEMPTS,
                "max_sql_per_attempt": MAX_VERIFIER_SQL_PER_ATTEMPT,
                "max_sql_total": MAX_VERIFIER_SQL_TOTAL,
            },
            "attempts": attempts,
            "new_verifier_calls": sum(bool(row.get("verifier_invoked")) for row in attempts),
            "new_verifier_sql": sum(int(row.get("verifier_sql_count", 0)) for row in attempts),
            "combined_verifier_calls": source["historical_verifier_calls"]
            + sum(bool(row.get("verifier_invoked")) for row in attempts),
            "combined_verifier_sql": source["historical_verifier_sql"]
            + sum(int(row.get("verifier_sql_count", 0)) for row in attempts),
            "gpu_seconds": 0,
            "model_calls": 0,
            "training_calls": 0,
            "slurm_calls": 0,
            "old_budget_state_unchanged": True,
        },
    )


def _run_verifier_attempt(
    *,
    output: Path,
    source: dict[str, Any],
    database: Path,
    attempt_index: int,
    verifier: Callable[..., Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempt_dir = output / "verifier" / f"attempt-{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": "t025-verifier-request-v1",
        "arm": TARGET_ARM,
        "task_id": TARGET_TASK_ID,
        "episode_id": TARGET_EPISODE_ID,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "trajectory_sha256": source["trajectory_sha256"],
        "database_source_sha256": source["database_sha256"],
        "database_copy": str(database),
        "settings": source["protocol_runtime"],
        "benchmark_task_sha256": canonical_hash(source["private_task"]),
        "reference_sql_sha256": canonical_hash(source["benchmark_task"].reference_sql),
        "policy_version": source["policy_version"],
    }
    atomic_write_json(attempt_dir / "verifier-request.json", request)
    capture = _CapturedSQL(attempt_dir / "sql")
    invoked = True
    started = time.monotonic()
    try:
        with capture:
            reward_value = verifier(
                source["benchmark_task"],
                source["trajectory"],
                database,
                Settings.model_validate(source["protocol_runtime"]),
            )
        reward = (
            reward_value
            if isinstance(reward_value, RewardResult)
            else RewardResult.model_validate(reward_value)
        )
        if len(capture.calls) > MAX_VERIFIER_SQL_PER_ATTEMPT:
            raise SavedScoringError("verifier_sql_per_attempt_limit_exceeded")
        reward_json = reward.model_dump(mode="json")
        output_record = {
            "schema_version": "t025-verifier-output-v1",
            "status": "returned",
            "exit_code": 0,
            "elapsed_s": time.monotonic() - started,
            "reward_result": reward_json,
            "sql_calls": [
                {
                    "call_index": call["call_index"],
                    "exit_code": call["exit_code"],
                    "result": call["result"],
                    "sql": call["request"].get("sql"),
                }
                for call in capture.calls
            ],
        }
        atomic_write_json(attempt_dir / "verifier-output.json", output_record)
        row = _score_row(
            source=source,
            reward=reward,
            verifier_sql_count=len(capture.calls),
        )
        attempt_record = {
            "attempt": attempt_index,
            "status": "scored",
            "verifier_invoked": invoked,
            "verifier_sql_count": len(capture.calls),
            "reward_persisted": True,
            "output": str((attempt_dir / "verifier-output.json").relative_to(output)),
        }
        return row, attempt_record
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        atomic_write_json(
            attempt_dir / "verifier-output.json",
            {
                "schema_version": "t025-verifier-output-v1",
                "status": "technical_failure",
                "exit_code": 1,
                "elapsed_s": time.monotonic() - started,
                "error": error,
                "sql_calls": capture.calls,
            },
        )
        return None, {
            "attempt": attempt_index,
            "status": "technical_failure",
            "verifier_invoked": invoked,
            "verifier_sql_count": len(capture.calls),
            "reward_persisted": False,
            "error": error,
            "output": str((attempt_dir / "verifier-output.json").relative_to(output)),
        }


def _unknown_row(source: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": "grpo-score-v1",
        "record_type": "RESULT",
        "episode_id": TARGET_EPISODE_ID,
        "task_id": TARGET_TASK_ID,
        "status": "unknown",
        "success": None,
        "reward_value": None,
        "reason": reason,
        "unknown_reason": reason,
        "technical_unknown": True,
        "verifier_called": True,
        "verifier_sql_count": None,
        "source_kind": "t025-supplemental-verifier",
        "policy_snapshot_id": source["policy_version"],
        "raw_saved": True,
        "trajectory_path": _relative(source["trajectory_path"], source["source_root"]),
        "trajectory_sha256": source["trajectory_sha256"],
        "qualification": source["qualification"],
        "trajectory_source_root": str(source["source_root"]),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "metadata": {"supplemental_task": "T-025/E-026"},
    }


def _score_saved_case(
    *,
    output: Path,
    source: dict[str, Any],
    database: Path,
    verifier: Callable[..., Any],
) -> tuple[dict[str, Any], bool]:
    existing = _load_existing_supplement(output, source)
    if existing is not None:
        return existing, True
    ledger_value = read_json(output / "call-ledger.json", {}) or {}
    attempts = list(ledger_value.get("attempts", [])) if isinstance(ledger_value, dict) else []
    prior_sql = sum(int(row.get("verifier_sql_count", 0)) for row in attempts)
    if prior_sql > MAX_VERIFIER_SQL_TOTAL:
        raise SavedScoringError("verifier_sql_total_limit_exceeded")
    if len(attempts) >= MAX_VERIFIER_ATTEMPTS:
        last_error = attempts[-1].get("error", {}) if attempts else {}
        reason = "technical_verifier_failure_after_max_attempts"
        supplement = {
            "schema_version": "t025-supplemental-score-v1",
            "status": "unknown",
            "arm": TARGET_ARM,
            "task_id": TARGET_TASK_ID,
            "episode_id": TARGET_EPISODE_ID,
            "source_manifest_sha256": source["source_manifest_sha256"],
            "trajectory_sha256": source["trajectory_sha256"],
            "policy_version": source["policy_version"],
            "status_reason": reason,
            "technical_error": last_error,
            "score_row": _unknown_row(source, reason),
            "source_relation": {
                "from": "original_unknown",
                "to": "t025_supplemental_unknown",
                "original_unknown_reason": source["original_unknown"].get("unknown_reason"),
            },
        }
        atomic_write_json(output / "supplemental-score.json", supplement)
        return supplement, False
    while len(attempts) < MAX_VERIFIER_ATTEMPTS:
        attempt_index = len(attempts) + 1
        row, attempt = _run_verifier_attempt(
            output=output,
            source=source,
            database=database,
            attempt_index=attempt_index,
            verifier=verifier,
        )
        attempts.append(attempt)
        _write_call_ledger(output, source, attempts)
        if row is not None:
            supplement = {
                "schema_version": "t025-supplemental-score-v1",
                "status": "scored",
                "arm": TARGET_ARM,
                "task_id": TARGET_TASK_ID,
                "episode_id": TARGET_EPISODE_ID,
                "source_manifest_sha256": source["source_manifest_sha256"],
                "trajectory_sha256": source["trajectory_sha256"],
                "policy_version": source["policy_version"],
                "source_relation": {
                    "from": "original_unknown",
                    "to": "t025_supplemental_scored",
                    "original_status": source["original_unknown"].get("status"),
                    "original_unknown_reason": source["original_unknown"].get("unknown_reason"),
                    "original_verifier_called": source["original_unknown"].get("verifier_called"),
                    "original_trajectory_sha256": source["original_unknown"].get(
                        "trajectory_sha256"
                    ),
                },
                "qualification": source["qualification"],
                "verifier_attempts": attempts,
                "score_row": row,
                "reward_result": row["metadata"]["reward"],
            }
            atomic_write_json(output / "supplemental-score.json", supplement)
            return supplement, False
    reason = "technical_verifier_failure_after_max_attempts"
    supplement = {
        "schema_version": "t025-supplemental-score-v1",
        "status": "unknown",
        "arm": TARGET_ARM,
        "task_id": TARGET_TASK_ID,
        "episode_id": TARGET_EPISODE_ID,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "trajectory_sha256": source["trajectory_sha256"],
        "policy_version": source["policy_version"],
        "source_relation": {
            "from": "original_unknown",
            "to": "t025_supplemental_unknown",
            "original_unknown_reason": source["original_unknown"].get("unknown_reason"),
        },
        "qualification": source["qualification"],
        "verifier_attempts": attempts,
        "status_reason": reason,
        "score_row": _unknown_row(source, reason),
    }
    atomic_write_json(output / "supplemental-score.json", supplement)
    return supplement, False


def _build_merged_view(
    *,
    output: Path,
    source: dict[str, Any],
    supplement: dict[str, Any],
) -> dict[str, Any]:
    directory = output / "merged-verifier-index"
    directory.mkdir(parents=True, exist_ok=True)
    supplement_row = supplement.get("score_row")
    if not isinstance(supplement_row, dict):
        raise SavedScoringError("supplement_score_row_missing_for_merge")
    row_manifest: dict[str, Any] = {}
    for arm in ("Base", "N", "RL"):
        index_path = source["source_root"] / "evaluation" / arm / "verifier-results.jsonl"
        original_rows = [
            row for row in read_jsonl(index_path) if row.get("record_type", "RESULT") == "RESULT"
        ]
        by_task: dict[str, dict[str, Any]] = {}
        for row in original_rows:
            task_value = row.get("task_id")
            if not isinstance(task_value, str) or task_value in by_task:
                raise SavedScoringError(f"source_result_duplicate_or_invalid:{arm}:{task_value}")
            by_task[task_value] = row
        if set(by_task) != set(source["task_ids"]):
            raise SavedScoringError(f"source_task_set_mismatch:{arm}")
        target_row = by_task[TARGET_TASK_ID]
        if arm == TARGET_ARM:
            if (
                target_row.get("status") != "unknown"
                or target_row.get("verifier_called") is not False
            ):
                raise SavedScoringError("source_target_not_original_unknown")
            if target_row.get("trajectory_sha256") != source["trajectory_sha256"]:
                raise SavedScoringError("source_target_trajectory_conflict")
            by_task[TARGET_TASK_ID] = copy.deepcopy(supplement_row)
        elif target_row.get("status") != "scored":
            raise SavedScoringError(f"complete_source_arm_contains_unknown:{arm}")
        merged_rows: list[dict[str, Any]] = []
        for task_value in source["task_ids"]:
            row = copy.deepcopy(by_task[task_value])
            if not isinstance(row.get("trajectory_path"), str):
                raise SavedScoringError(f"merged_trajectory_path_missing:{arm}:{task_value}")
            row["trajectory_source_root"] = str(source["source_root"])
            row["source_manifest_sha256"] = source["source_manifest_sha256"]
            row["source_run"] = str(source["source_root"])
            merged_rows.append(row)
        merged_path = directory / f"{arm}.jsonl"
        _write_jsonl(merged_path, merged_rows)
        row_manifest[arm] = {
            "path": str(merged_path.relative_to(output)),
            "sha256": sha256_file(merged_path),
            "rows": len(merged_rows),
            "source_root": str(source["source_root"]),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "supplement_replaced_task": TARGET_TASK_ID if arm == TARGET_ARM else None,
        }
    atomic_write_json(
        directory / "index.json",
        {
            "schema_version": "t025-merged-verifier-index-v1",
            "source_root": str(source["source_root"]),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "task_ids": source["task_ids"],
            "arms": row_manifest,
            "source_rows_are_read_only_references": True,
            "supplement_status": supplement.get("status"),
        },
    )
    return row_manifest


def _aggregate_merged(
    *,
    output: Path,
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    for arm in ("Base", "N", "RL"):
        metrics[arm] = aggregate_source(
            source_root=output,
            verifier_index=output / "merged-verifier-index" / f"{arm}.jsonl",
            identity_field="task_id",
            expected_count=20,
            source_kind=f"T024-E026-source-plus-T025-{arm}",
            planned_task_ids=source["task_ids"],
            allowed_trajectory_roots={source["source_root"]},
        )
        metrics[arm].update(
            {
                "protocol_fingerprint": source["protocol_fingerprint"],
                "source_root": str(source["source_root"]),
                "source_manifest_sha256": source["source_manifest_sha256"],
                "comparison_task_ids": source["task_ids"],
            }
        )
    comparison = three_arm_comparison(
        metrics["Base"], metrics["N"], metrics["RL"], expected_count=20
    )
    return metrics, comparison


def _snapshot_sources(output: Path, repo_root: Path) -> list[dict[str, Any]]:
    relative_paths = [
        Path("src/enterprise_agent_rl/grpo/aggregation.py"),
        Path("src/enterprise_agent_rl/grpo/cli.py"),
        Path("src/enterprise_agent_rl/grpo/evidence.py"),
        Path("src/enterprise_agent_rl/grpo/saved_scoring.py"),
        Path("tests/test_grpo_aggregation.py"),
        Path("tests/test_grpo_saved_scoring.py"),
        Path("docs/grpo-training.md"),
    ]
    rows: list[dict[str, Any]] = []
    for relative in relative_paths:
        source_path = repo_root / relative
        if not source_path.is_file():
            continue
        target = output / "source-snapshots" / "current" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        rows.append(
            {
                "path": str(relative),
                "snapshot": str(target.relative_to(output)),
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }
        )
    atomic_write_json(
        output / "source-snapshots" / "index.json",
        {"schema_version": "t025-source-snapshot-v1", "files": rows},
    )
    return rows


def _write_report(
    *,
    output: Path,
    source: dict[str, Any],
    supplement: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    reused: bool,
) -> None:
    ledger = _json_object(output / "call-ledger.json")
    lines = [
        "# T-025 / E-026 saved Verifier closeout",
        "",
        (
            f"Status: **{comparison['status']}**; "
            f"saved target supplement status: **{supplement['status']}**."
        ),
        "",
        "## Scope and provenance",
        "",
        (
            f"- Source run: `{source['source_root']}`; "
            f"manifest SHA `{source['source_manifest_sha256']}` (read-only)."
        ),
        (
            f"- Target: `{TARGET_ARM}/{TARGET_TASK_ID}`; "
            f"trajectory SHA `{source['trajectory_sha256']}`; "
            f"canonical policy `{source['policy_version']}`."
        ),
        (
            f"- Relation: original `unknown` "
            f"(`{source['original_unknown'].get('unknown_reason')}`) → "
            f"T-025 `{supplement['status']}`; the original source row was not changed."
        ),
        (
            f"- Source model evidence remains `{source['source_model_calls']}` calls; "
            "new model/training/GPU/Slurm calls: `0`; Job: `N/A`."
        ),
        (
            f"- Source allocation remains "
            f"`{source['source_allocation'].get('allocation_elapsed_seconds_total')}` "
            "GPU seconds; this CPU run records `0` GPU seconds."
        ),
        "",
        "## Scores and comparisons",
        "",
        "| arm | planned | qualified | scored | unknown | successes | success rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("Base", "N", "RL"):
        value = metrics[arm]
        lines.append(
            f"| {arm} | {value.get('planned')} | {value.get('qualified')} | "
            f"{value.get('scored')} | {value.get('unknown')} | "
            f"{value.get('successes')} | {value.get('success_rate')} |"
        )
    for key in ("N_minus_Base", "RL_minus_N", "RL_minus_Base"):
        delta = comparison[key]
        lines.append(
            f"- `{key}`: status `{delta.get('status')}`, "
            f"delta `{delta.get('successes')}`, rate `{delta.get('success_rate')}`, "
            f"denominator `{delta.get('denominator')}`, "
            f"planned denominator `{delta.get('planned_denominator')}`."
        )
    lines.extend(
        [
            (
                f"- Global three-arm status: `{comparison['status']}`; "
                f"global denominator: `{comparison.get('denominator')}`; "
                "all pair planned denominators remain 20."
            ),
            "",
            "## Calls and failure handling",
            "",
            (
                f"- Independent Verifier attempts: `{ledger.get('new_verifier_calls')}`; "
                f"SQL calls: `{ledger.get('new_verifier_sql')}`; "
                f"combined historical accounting: `{ledger.get('combined_verifier_calls')}` "
                f"Verifier / `{ledger.get('combined_verifier_sql')}` SQL."
            ),
            (
                f"- Supplemental RewardResult: success="
                f"`{supplement.get('reward_result', {}).get('success')}`, "
                f"value=`{supplement.get('reward_result', {}).get('value')}`, "
                f"reason=`{supplement.get('reward_result', {}).get('reason')}`."
            ),
            (
                f"- Reused persisted supplement on this invocation: `{reused}`. "
                "A business `false` is a scored result and is not retried; only a "
                "technical failure without a persisted RewardResult can use the second attempt."
            ),
            (
                "- The copied database is `inputs/databases/validation-1.sqlite3`, "
                "hash-checked against the source and made read-only. No validation-0, "
                "train, or test database was opened."
            ),
            (
                "- `supplemental-score.json`, per-attempt request/output, SQL worker "
                "stdout/stderr/exit records, `call-ledger.json`, and "
                "`merged-verifier-index/` are the durable scoring evidence."
            ),
            "",
            "## Reproducibility boundary",
            "",
            (
                "- The 59 historical scores are immutable T-024 references; only the "
                "exact RL unknown is replaced in the new view after source/trajectory "
                "hash matching."
            ),
            (
                "- Hidden task fields are supplied only to the isolated production "
                "Verifier; they are not passed to a model or training path."
            ),
            (
                "- See `metrics.json`, `comparison.json`, `cpu/checks.json`, "
                "`commands.json`, `source-index.json`, and `manifest.json` for "
                "machine-readable evidence."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": "t025-manifest-v1",
            "run_dir": str(output),
            "source_root": None,
            "files": rows,
        },
    )


def score_saved_trajectory(
    *,
    source_run: Path,
    private_task_plan: Path,
    output: Path,
    repo_root: Path,
    arm: str = TARGET_ARM,
    task_id: str = TARGET_TASK_ID,
    verifier: Callable[..., Any] = production_verify,
    strict_identity: bool = True,
) -> dict[str, Any]:
    """Score one saved case and construct the complete merged comparison view."""

    source = validate_saved_target(
        source_run=source_run,
        private_task_plan=private_task_plan,
        arm=arm,
        task_id=task_id,
        strict_identity=strict_identity,
    )
    output, output_preexisting = _choose_output_dir(output)
    try:
        output.relative_to(source["source_root"])
    except ValueError:
        pass
    else:
        raise SavedScoringError("output_must_not_be_inside_source_run")
    output.mkdir(parents=True, exist_ok=True)
    git_baseline = output / "cpu" / "git-baseline.json"
    if not git_baseline.is_file():
        baseline = _git_state(repo_root)
        baseline["capture_point"] = "before_closeout_artifacts"
        baseline["output_preexisting"] = output_preexisting
        atomic_write_json(git_baseline, baseline)
    _write_source_index(output, source)
    database = _copy_read_only_database(output, source)
    supplement, reused = _score_saved_case(
        output=output,
        source=source,
        database=database,
        verifier=verifier,
    )
    _build_merged_view(output=output, source=source, supplement=supplement)
    metrics, comparison = _aggregate_merged(output=output, source=source)
    atomic_write_json(output / "supplemental-score.json", supplement)
    atomic_write_json(output / "metrics.json", metrics)
    atomic_write_json(output / "comparison.json", comparison)
    source_manifest_after = sha256_file(source["source_manifest_path"])
    source_trajectory_after = sha256_file(source["trajectory_path"])
    source_database_after = sha256_file(source["database_path"])
    atomic_write_json(
        output / "cpu" / "checks.json",
        {
            "schema_version": "t025-cpu-checks-v1",
            "status": (
                "PASS"
                if source_manifest_after == source["source_manifest_sha256"]
                and source_database_after == source["database_sha256"]
                else "FAIL"
            ),
            "source_manifest_sha256_before": source["source_manifest_sha256"],
            "source_manifest_sha256_after": source_manifest_after,
            "source_trajectory_sha256_before": source["trajectory_sha256"],
            "source_trajectory_sha256_after": source_trajectory_after,
            "source_database_sha256_before": source["database_sha256"],
            "source_database_sha256_after": source_database_after,
            "copied_database_sha256": sha256_file(database),
            "model_load_calls": 0,
            "tokenizer_construct_calls": 0,
            "training_calls": 0,
            "backward_calls": 0,
            "optimizer_update_calls": 0,
            "gpu_seconds": 0,
            "slurm_calls": 0,
            "job_id": "N/A",
            "reused_persisted_score": reused,
            "real_verifier_scope": f"{TARGET_ARM}/{TARGET_TASK_ID}",
            "fixture_business_sql": False,
        },
    )
    atomic_write_json(
        output / "commands.json",
        {
            "schema_version": "t025-commands-v1",
            "argv": list(sys.argv),
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "repo_root": str(repo_root.resolve()),
            "source_run": str(source["source_root"]),
            "output": str(output),
            "job_id": "N/A",
            "gpu_seconds": 0,
            "model_calls": 0,
            "training_calls": 0,
            "slurm_calls": 0,
            "exit_code": 0,
        },
    )
    _snapshot_sources(output, repo_root.resolve())
    _write_report(
        output=output,
        source=source,
        supplement=supplement,
        metrics=metrics,
        comparison=comparison,
        reused=reused,
    )
    after = _git_state(repo_root)
    after["capture_point"] = "after_closeout_artifacts"
    atomic_write_json(output / "cpu" / "git-after.json", after)
    _write_manifest(output)
    manifest_value = _json_object(output / "manifest.json")
    manifest_value["source_root"] = str(source["source_root"])
    manifest_value["source_manifest_sha256"] = source["source_manifest_sha256"]
    manifest_value["target_trajectory_sha256"] = source["trajectory_sha256"]
    atomic_write_json(output / "manifest.json", manifest_value)
    return {
        "status": comparison["status"],
        "run_dir": str(output),
        "supplement_status": supplement["status"],
        "reused": reused,
        "comparison": comparison,
        "metrics": metrics,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "target_trajectory_sha256": source["trajectory_sha256"],
        "new_verifier_calls": _json_object(output / "call-ledger.json").get("new_verifier_calls"),
        "new_verifier_sql": _json_object(output / "call-ledger.json").get("new_verifier_sql"),
        "output_preexisting": output_preexisting,
    }
