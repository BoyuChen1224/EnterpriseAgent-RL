"""Formal Base Agent benchmark orchestration and artifact reporting.

This module deliberately reuses the production Agent Runtime and model policy.
It owns only benchmark scoping, checkpointed artifacts, and post-run analysis.
The benchmark is validation-only: validation database files are opened, while
train/test database files are never opened by this runner.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from enterprise_agent_rl.agent import Policy, run_episode
from enterprise_agent_rl.benchmark import canonical_hash, database_content_fingerprint
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import Trajectory, TrajectoryStore

EXPECTED_SPLIT_COUNTS = {"train": 20, "validation": 20, "test": 20}
INFRASTRUCTURE_CODE_TOKENS = (
    "cuda",
    "oom",
    "out_of_memory",
    "non_finite",
    "nan",
    "inf_logits",
    "tokenizer",
    "model_load",
    "model_transport",
    "model_http",
    "model_busy",
    "generation_error",
    "inference_timeout",
    "context_length",
    "shared_library",
    "gpu_unavailable",
    "server_crash",
    "database_unavailable",
    "unable to open",
    "database is locked",
    "database disk image",
    "i/o error",
)
BUSINESS_ERROR_CODES = {
    "empty_output",
    "invalid_json",
    "invalid_action",
    "invalid_action_arguments",
}
SQL_ERROR_PATTERNS = (
    "no such column",
    "no such table",
    "syntax error",
    "ambiguous",
    "incomplete input",
    "misuse of",
    "near ",
)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _safe_task_id(task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    return value[:180] or "unknown-task"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git_state(output: Path) -> None:
    commands = [
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("status", ["git", "status", "--short", "--branch"]),
        ("diff_stat", ["git", "diff", "--stat"]),
    ]
    lines = []
    for label, argv in commands:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        lines.append(f"[{label}]\n{result.stdout}{result.stderr}")
    (output / "git-state.txt").write_text("\n".join(lines))


def _scheduler_fields() -> dict[str, Any]:
    fields = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
        "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True, check=False
        )
        fields["scontrol_output"] = result.stdout + result.stderr
        for key, pattern in {
            "account": r"Account=([^\s]+)",
            "partition": r"Partition=([^\s]+)",
            "qos": r"QOS=([^\s]+)",
            "node": r"NodeList=([^\s]+)",
            "state": r"JobState=([^\s]+)",
            "exit_code": r"ExitCode=([^\s]+)",
            "elapsed": r"RunTime=([^\s]+)",
        }.items():
            match = re.search(pattern, fields["scontrol_output"])
            if match:
                fields[key] = match.group(1)
    return fields


def _gpu_fields(runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "gpu_name": runtime_metadata.get("gpu_name"),
        "compute_capability": runtime_metadata.get("compute_capability"),
        "gpu_memory_total": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        fields["nvidia_smi_query"] = f"unavailable:{exc}"
        return fields
    fields["nvidia_smi_query"] = result.stdout.strip() or result.stderr.strip()
    first = result.stdout.strip().splitlines()
    if first:
        columns = [part.strip() for part in first[0].split(",")]
        if columns and not fields["gpu_name"]:
            fields["gpu_name"] = columns[0]
        if len(columns) > 1:
            fields["gpu_memory_total"] = columns[1]
        if len(columns) > 2:
            fields["driver_version"] = columns[2]
    return fields


def write_environment(
    output: Path,
    model_config: ModelSettings,
    settings: Settings,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_metadata = runtime_metadata or {}
    environment = {
        "schema_version": "1.0",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": _scheduler_fields(),
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "tokenizer_revision": model_config.revision,
        "dtype": model_config.dtype,
        "effective_dtype": runtime_metadata.get("model_dtype"),
        "device": runtime_metadata.get("device", "cuda"),
        "gpu_name": runtime_metadata.get("gpu_name"),
        "compute_capability": runtime_metadata.get("compute_capability"),
        "attention_backend": model_config.attention_backend,
        "non_finite_logits_detected": False,
        "runtime_versions": {
            "python": sys.version,
            "torch": runtime_metadata.get("torch"),
            "transformers": runtime_metadata.get("transformers"),
            "cuda_runtime": runtime_metadata.get("cuda_runtime"),
        },
        "generation": model_config.generation.model_dump(),
        "agent_settings": settings.model_dump(),
        "runtime_metadata": runtime_metadata,
        "gpu": _gpu_fields(runtime_metadata),
        "test_split_accessed": False,
        "created_at_utc": _now(),
    }
    _write_json(output / "environment.json", environment)
    return environment


def _check_record_fingerprint(record: dict[str, Any]) -> None:
    fields = {key: value for key, value in record.items() if key != "data_fingerprint"}
    if canonical_hash(fields) != record.get("data_fingerprint"):
        raise ValueError("task_fingerprint_mismatch")


def prepare_validation_plan(manifest_path: Path, database_root: Path) -> dict[str, Any]:
    """Validate the frozen manifest while opening validation databases only."""
    raw = json.loads(manifest_path.read_text())
    declared = raw.get("manifest_sha256")
    if not declared:
        raise ValueError("manifest_hash_missing")
    unsigned = dict(raw)
    unsigned.pop("manifest_sha256", None)
    if canonical_hash(unsigned) != declared:
        raise ValueError("manifest_hash_mismatch")
    lock_path = manifest_path.with_suffix(".sha256")
    if not lock_path.is_file() or lock_path.read_text().strip() != declared:
        raise ValueError("frozen_manifest_changed")
    split_counts = dict(raw.get("split_counts", {}))
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"split_counts_mismatch:{split_counts}")

    records = [
        record
        for record in raw.get("records", [])
        if record.get("task", {}).get("split") == "validation"
    ]
    if len(records) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(f"validation_task_count_mismatch:{len(records)}")
    task_ids = [record["task"]["public"]["task_id"] for record in records]
    if len(task_ids) != len(set(task_ids)) or any(
        not task_id.startswith("validation-") for task_id in task_ids
    ):
        raise ValueError("validation_task_ids_mismatch")
    for record in records:
        _check_record_fingerprint(record)

    database_entries = {
        entry["id"]: entry
        for entry in raw.get("databases", [])
        if entry.get("split") == "validation"
    }
    database_ids = sorted({record["database_id"] for record in records})
    if set(database_ids) != set(database_entries):
        raise ValueError("validation_database_selection_mismatch")
    database_fingerprints = {}
    for database_id in database_ids:
        database = database_root / f"{database_id}.sqlite3"
        if not database.is_file():
            raise FileNotFoundError(database)
        fingerprint = database_content_fingerprint(database)
        if fingerprint != database_entries[database_id]["sha256"]:
            raise ValueError(f"benchmark_database_changed:{database_id}")
        database_fingerprints[database_id] = fingerprint
    return {
        "manifest_sha256": declared,
        "split_counts": split_counts,
        "records": records,
        "task_ids": task_ids,
        "database_ids": database_ids,
        "database_fingerprints": database_fingerprints,
        "test_split_accessed": False,
        "train_split_accessed": False,
    }


def _error_strings(trajectory: Trajectory) -> list[str]:
    values: list[str] = []
    for attempt in trajectory.model_attempts:
        for key in ("error", "server_error"):
            value = attempt.get(key)
            if isinstance(value, dict):
                values.extend(str(value.get(field, "")) for field in ("code", "message", "type"))
            elif value:
                values.append(str(value))
        telemetry = attempt.get("telemetry", {})
        if telemetry.get("non_finite_logits_detected"):
            values.append("non_finite_logits_detected")
    for item in trajectory.runtime_errors:
        values.extend(str(item.get(field, "")) for field in ("type", "message"))
    for step in trajectory.steps:
        if not step.tool_result.ok and step.tool_result.error:
            values.append(str(step.tool_result.error))
    return [value for value in values if value]


def _is_infrastructure_text(value: str) -> bool:
    text = value.lower()
    if any(code in text for code in BUSINESS_ERROR_CODES):
        return False
    return any(token in text for token in INFRASTRUCTURE_CODE_TOKENS)


def infrastructure_reasons(trajectory: Trajectory) -> list[dict[str, str]]:
    reasons = []
    if trajectory.runtime_errors:
        reasons.extend(
            {
                "type": "runtime_error",
                "code": str(item.get("type", "runtime_error")),
                "message": str(item.get("message", "")),
            }
            for item in trajectory.runtime_errors
        )
    for attempt in trajectory.model_attempts:
        telemetry = attempt.get("telemetry", {})
        if telemetry.get("non_finite_logits_detected"):
            reasons.append(
                {
                    "type": "non_finite_logits",
                    "code": "non_finite_logits_detected",
                    "message": "finite-logit guard reported non-finite logits",
                }
            )
        for key in ("error", "server_error"):
            value = attempt.get(key)
            if isinstance(value, dict):
                code = str(value.get("code", ""))
                message = str(value.get("message", ""))
            elif value:
                code, message = str(value), str(value)
            else:
                continue
            if _is_infrastructure_text(code) or _is_infrastructure_text(message):
                reasons.append({"type": "model_runtime", "code": code, "message": message})
    for step in trajectory.steps:
        if step.tool_result.ok or not step.tool_result.error:
            continue
        error = str(step.tool_result.error)
        if _is_infrastructure_text(error):
            reasons.append({"type": "tool_infrastructure", "code": "tool_error", "message": error})
    if trajectory.terminal_reason.startswith("policy_or_runtime_error"):
        reasons.append(
            {
                "type": "policy_runtime_error",
                "code": trajectory.terminal_reason,
                "message": trajectory.terminal_reason,
            }
        )
    unique = []
    seen = set()
    for reason in reasons:
        key = (reason["type"], reason["code"], reason["message"])
        if key not in seen:
            seen.add(key)
            unique.append(reason)
    return unique


def _sql_error_type(error: str) -> str:
    text = error.lower()
    if "no such column" in text:
        return "no_such_column"
    if "no such table" in text:
        return "no_such_table"
    if "syntax error" in text or "near " in text or "incomplete input" in text:
        return "syntax_error"
    if "ambiguous" in text:
        return "ambiguous_column"
    if "wrong aggregation" in text:
        return "wrong_aggregation"
    if "wrong filter" in text:
        return "wrong_filter"
    return "other_sql_error"


def _recovery(trajectory: Trajectory) -> dict[str, Any]:
    sql_steps = [step for step in trajectory.steps if step.tool_name == "sql"]
    failed = [step for step in sql_steps if not step.tool_result.ok]
    recovery_count = sum(
        any(next_step.tool_result.ok for next_step in sql_steps[index + 1 :])
        for index, step in enumerate(sql_steps)
        if not step.tool_result.ok
    )
    repeated_loop = False
    for index, step in enumerate(sql_steps):
        if step.tool_result.ok:
            continue
        later_failures = [
            later
            for later in sql_steps[index + 1 :]
            if not later.tool_result.ok and later.tool_arguments == step.tool_arguments
        ]
        if later_failures:
            repeated_loop = True
            break
    if not failed:
        status = "no_recovery_required"
    elif repeated_loop and recovery_count == 0:
        status = "repeated_recovery_loop"
    elif recovery_count:
        status = "successful_recovery"
    elif len(sql_steps) > len(failed):
        status = "unsuccessful_recovery"
    else:
        status = "no_recovery"
    return {
        "recovery_count": recovery_count,
        "recovery_status": status,
        "sql_error_count": len(failed),
        "agent_observed_error_feedback": any(
            any(not step.tool_result.ok for step in trajectory.steps[:index])
            for index in range(1, len(trajectory.steps))
        ),
        "repeated_recovery_loop": repeated_loop,
    }


def _task_result(
    trajectory: Trajectory | None,
    task: BenchmarkTask,
    database_id: str,
    output: Path,
    model_config: ModelSettings,
    runtime_metadata: dict[str, Any],
    reward: Any | None,
    verification_status: str,
    infrastructure: list[dict[str, str]],
    trajectory_path: str | None,
    raw_trajectory_path: str | None,
) -> dict[str, Any]:
    if trajectory is None:
        tools = []
        sql_steps = []
        attempts = []
        input_tokens = output_tokens = 0
        input_known = output_known = False
        recovery = {"recovery_count": 0, "recovery_status": "not_run"}
        latency = 0.0
        terminal_reason = "infrastructure_failure"
        policy_kind = "local_llm"
        policy_version = None
        runtime_errors = []
        non_finite = False
    else:
        tools = [step for step in trajectory.steps if step.tool_name]
        sql_steps = [step for step in tools if step.tool_name == "sql"]
        attempts = trajectory.model_attempts
        input_known = all("prompt_tokens" in attempt.get("usage", {}) for attempt in attempts)
        output_known = all("completion_tokens" in attempt.get("usage", {}) for attempt in attempts)
        input_tokens = sum(attempt.get("usage", {}).get("prompt_tokens", 0) for attempt in attempts)
        output_tokens = sum(
            attempt.get("usage", {}).get("completion_tokens", 0) for attempt in attempts
        )
        recovery = _recovery(trajectory)
        latency = trajectory.elapsed_s
        terminal_reason = trajectory.terminal_reason
        policy_kind = trajectory.policy_kind
        policy_version = trajectory.policy_version
        runtime_errors = trajectory.runtime_errors
        non_finite = any(
            attempt.get("telemetry", {}).get("non_finite_logits_detected", False)
            for attempt in attempts
        )
    sql_errors = [str(step.tool_result.error) for step in sql_steps if not step.tool_result.ok]
    sql_error_types = Counter(_sql_error_type(error) for error in sql_errors)
    invalid_outputs = sum(
        str(attempt.get("error", "")) in BUSINESS_ERROR_CODES for attempt in attempts
    )
    if infrastructure:
        category = "Infrastructure Failure"
        failure_reason = infrastructure[0].get("code") or "infrastructure_failure"
    elif reward is not None and reward.success:
        category = "Successful"
        failure_reason = None
    else:
        failure_reason = (
            reward.reason if reward is not None else terminal_reason
        ) or terminal_reason
        lower_reason = str(failure_reason).lower()
        if "invalid_action" in lower_reason or invalid_outputs:
            category = "Invalid Action"
        elif "budget_exhausted" in lower_reason or "tool_budget" in lower_reason:
            category = "Action Budget Exhausted"
        elif sql_errors and recovery["recovery_status"] in {
            "unsuccessful_recovery",
            "repeated_recovery_loop",
        }:
            category = "SQL Recovery Failed"
        elif sql_errors:
            category = "SQL Error"
        elif "incorrect_result" in lower_reason or (
            trajectory is not None and trajectory.terminal_reason == "completed"
        ):
            category = "Wrong Final Answer"
        else:
            category = "Other Business Failure"

    telemetry = next(
        (attempt.get("telemetry", {}) for attempt in attempts if attempt.get("telemetry")),
        {},
    )
    peak_allocated = max(
        (
            attempt.get("telemetry", {}).get("peak_allocated_bytes")
            for attempt in attempts
            if attempt.get("telemetry", {}).get("peak_allocated_bytes") is not None
        ),
        default=None,
    )
    peak_reserved = max(
        (
            attempt.get("telemetry", {}).get("peak_reserved_bytes")
            for attempt in attempts
            if attempt.get("telemetry", {}).get("peak_reserved_bytes") is not None
        ),
        default=None,
    )
    tool_selection = Counter(
        "result_inspection" if step.tool_name == "check" else step.tool_name for step in tools
    )
    result = {
        "task_id": task.public.task_id,
        "database": database_id,
        "success": bool(reward and reward.success and not infrastructure),
        "failure_reason": failure_reason,
        "failure_category": category,
        "reward": _dump(reward),
        "tool_calls": len(tools),
        "invalid_tool_calls": sum(not step.tool_result.ok for step in tools),
        "invalid_action_count": invalid_outputs,
        "SQL_attempts": len(sql_steps),
        "sql_attempts": len(sql_steps),
        "successful_sql_executions": sum(step.tool_result.ok for step in sql_steps),
        "first_sql_success": sql_steps[0].tool_result.ok if sql_steps else None,
        "first_sql_failure": bool(sql_steps and not sql_steps[0].tool_result.ok),
        "sql_error_types": dict(sorted(sql_error_types.items())),
        "sql_error_messages": sql_errors,
        "tool_selection": dict(sorted(tool_selection.items())),
        "recovery_count": recovery["recovery_count"],
        "recovery_status": recovery["recovery_status"],
        "agent_observed_error_feedback": recovery.get("agent_observed_error_feedback", False),
        "repeated_recovery_loop": recovery.get("repeated_recovery_loop", False),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_usage_known": input_known,
        "output_token_usage_known": output_known,
        "latency": latency,
        "trajectory_path": trajectory_path,
        "raw_trajectory_path": raw_trajectory_path,
        "model_revision": model_config.revision,
        "tokenizer_revision": model_config.revision,
        "dtype": model_config.dtype,
        "effective_dtype": runtime_metadata.get("model_dtype"),
        "device": runtime_metadata.get("device", "cuda"),
        "gpu_name": runtime_metadata.get("gpu_name"),
        "compute_capability": runtime_metadata.get("compute_capability"),
        "attention_backend": model_config.attention_backend,
        "non_finite_logits_detected": non_finite,
        "generation_telemetry": {
            "input_tokens": telemetry.get("input_tokens"),
            "output_tokens": telemetry.get("output_tokens"),
            "generation_latency_s": telemetry.get("generation_latency_s"),
            "generation_tokens_per_s": telemetry.get("generation_tokens_per_s"),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "model_attempt_count": len(attempts),
        "policy_kind": policy_kind,
        "policy_version": policy_version,
        "terminal_reason": terminal_reason,
        "verification_status": verification_status,
        "runtime_errors": runtime_errors,
        "infrastructure_errors": infrastructure,
    }
    return result


def _failure_analysis(
    results: list[dict[str, Any]], infrastructure_failures: list[dict[str, Any]]
) -> dict[str, Any]:
    categories = [
        "Successful",
        "Invalid Action",
        "SQL Error",
        "SQL Recovery Failed",
        "Wrong Final Answer",
        "Action Budget Exhausted",
        "Other Business Failure",
        "Infrastructure Failure",
    ]
    counts = Counter(item.get("failure_category", "Other Business Failure") for item in results)
    category_counts = {category: counts.get(category, 0) for category in categories}
    business_failures = sum(
        value
        for category, value in category_counts.items()
        if category not in {"Successful", "Infrastructure Failure"}
    )
    top = sorted(
        (
            {"category": category, "count": count}
            for category, count in category_counts.items()
            if category not in {"Successful", "Infrastructure Failure"} and count
        ),
        key=lambda item: (-item["count"], item["category"]),
    )
    sql_error_types = Counter()
    tool_selection = Counter()
    recovery = Counter()
    for item in results:
        sql_error_types.update(item.get("sql_error_types", {}))
        tool_selection.update(item.get("tool_selection", {}))
        recovery.update([item.get("recovery_status", "unknown")])
    return {
        "category_counts": category_counts,
        "business_failures": business_failures,
        "infrastructure_failures": len(infrastructure_failures),
        "top_failure_modes": top,
        "sql_error_type_distribution": dict(sorted(sql_error_types.items())),
        "tool_selection_distribution": dict(sorted(tool_selection.items())),
        "recovery_outcomes": dict(sorted(recovery.items())),
        "infrastructure_failure_details": infrastructure_failures,
    }


def _metrics(
    results: list[dict[str, Any]], scheduled: int, model_settings: Settings
) -> dict[str, Any]:
    denominator = len(results)
    successes = sum(bool(item.get("success")) for item in results)
    sql_calls = sum(item.get("SQL_attempts", 0) for item in results)
    executable_sql = sum(item.get("successful_sql_executions", 0) for item in results)
    tool_calls = sum(item.get("tool_calls", 0) for item in results)
    tool_errors = sum(item.get("invalid_tool_calls", 0) for item in results)
    input_known = (
        all(item.get("input_token_usage_known", False) for item in results) if results else False
    )
    output_known = (
        all(item.get("output_token_usage_known", False) for item in results) if results else False
    )
    rewards = [item["reward"]["value"] for item in results if item.get("reward") is not None]
    generation_latencies = [
        item["generation_telemetry"]["generation_latency_s"]
        for item in results
        if item.get("generation_telemetry", {}).get("generation_latency_s") is not None
    ]
    attempts = sum(item.get("model_attempt_count", 0) for item in results)
    invalid_actions = sum(item.get("invalid_action_count", 0) for item in results)
    peak_allocated = max(
        (
            item.get("generation_telemetry", {}).get("peak_allocated_bytes")
            for item in results
            if item.get("generation_telemetry", {}).get("peak_allocated_bytes") is not None
        ),
        default=None,
    )
    peak_reserved = max(
        (
            item.get("generation_telemetry", {}).get("peak_reserved_bytes")
            for item in results
            if item.get("generation_telemetry", {}).get("peak_reserved_bytes") is not None
        ),
        default=None,
    )

    def average(field: str) -> float | None:
        return statistics.mean(item.get(field, 0) for item in results) if results else None

    average_input = (
        statistics.mean(item["input_tokens"] for item in results) if input_known else None
    )
    average_output = (
        statistics.mean(item["output_tokens"] for item in results) if output_known else None
    )
    average_total = (
        statistics.mean(item["total_tokens"] for item in results)
        if input_known and output_known
        else None
    )
    return {
        "scheduled_episodes": scheduled,
        "completed_episodes": denominator,
        "scored_episodes": sum(item.get("reward") is not None for item in results),
        "task_success_rate": successes / scheduled if scheduled else None,
        "task_success_rate_completed": successes / denominator if denominator else None,
        "verifier_success_rate": (
            sum(item.get("verification_status") == "scored" for item in results) / scheduled
            if scheduled
            else None
        ),
        "executable_sql_rate": executable_sql / sql_calls if sql_calls else None,
        "sql_executable_rate": executable_sql / sql_calls if sql_calls else None,
        "sql_calls": sql_calls,
        "executable_sql_calls": executable_sql,
        "tool_error_rate": tool_errors / tool_calls if tool_calls else None,
        "invalid_tool_call_count": tool_errors,
        "average_tool_calls": average("tool_calls"),
        "average_invalid_tool_calls": average("invalid_tool_calls"),
        "average_sql_attempts": average("SQL_attempts"),
        "average_recovery_count": average("recovery_count"),
        "average_input_tokens": average_input,
        "average_output_tokens": average_output,
        "average_total_tokens": average_total,
        "average_latency": average("latency"),
        "average_latency_s": average("latency"),
        "average_inference_latency_s": statistics.mean(generation_latencies)
        if generation_latencies
        else None,
        "average_reward": statistics.mean(rewards) if rewards else None,
        "median_reward": statistics.median(rewards) if rewards else None,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "reward_count": len(rewards),
        "model_requests": attempts,
        "model_outputs": attempts - invalid_actions,
        "invalid_output_count": invalid_actions,
        "token_usage_known": input_known and output_known,
        "input_token_usage_known": input_known,
        "output_token_usage_known": output_known,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "non_finite_logits_detected": any(
            item.get("non_finite_logits_detected", False) for item in results
        ),
        "policy_kind": results[0].get("policy_kind") if results else "local_llm",
        "policy_version": results[0].get("policy_version") if results else None,
        "llm_capability_result": True,
        "settings": model_settings.model_dump(),
    }


def _write_progress(output: Path, **updates: Any) -> None:
    path = output / "progress.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    existing.update(updates, updated_at_utc=_now())
    _write_json(path, existing)


def _write_config(
    output: Path,
    plan: dict[str, Any],
    model_config: ModelSettings,
    settings: Settings,
) -> None:
    _write_json(
        output / "config.json",
        {
            "benchmark": "Base Agent baseline",
            "split": "validation",
            "task_count": len(plan["records"]),
            "task_ids": plan["task_ids"],
            "database_ids": plan["database_ids"],
            "manifest_sha256": plan["manifest_sha256"],
            "model_config": model_config.model_dump(),
            "agent_settings": settings.model_dump(),
            "policy_kind": "local_llm",
            "deterministic_fixture_expert": False,
            "test_split_accessed": False,
            "train_split_accessed": False,
        },
    )


def _build_report(
    output: Path,
    plan: dict[str, Any],
    model_config: ModelSettings,
    settings: Settings,
    results: list[dict[str, Any]],
    verification_errors: list[dict[str, Any]],
    infrastructure_failures: list[dict[str, Any]],
    status: str,
    extra_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis = _failure_analysis(results, infrastructure_failures)
    metrics = _metrics(results, len(plan["records"]), settings)
    error_text = " ".join(
        str(value).lower()
        for item in results
        for value in item.get("sql_error_messages", []) + item.get("runtime_errors", [])
    )
    error_text += " " + " ".join(str(value).lower() for value in infrastructure_failures)
    runtime_stability = {
        "non_finite_logits_detected": metrics["non_finite_logits_detected"],
        "oom": "oom" in error_text or "out_of_memory" in error_text,
        "cuda_errors": "cuda" in error_text,
        "tokenizer_errors": "tokenizer" in error_text,
        "verifier_errors": len(verification_errors),
        "runtime_errors": sum(len(item.get("runtime_errors", [])) for item in results),
        "infrastructure_failure_count": len(infrastructure_failures),
    }
    report = {
        "schema_version": "2.0",
        "benchmark": "Base Agent baseline",
        "status": status,
        "complete": status == "READY_FOR_SFT" and len(results) == len(plan["records"]),
        "split": "validation",
        "task_count": len(plan["records"]),
        "scheduled_task_ids": plan["task_ids"],
        "completed_task_ids": [item["task_id"] for item in results],
        "manifest_sha256": plan["manifest_sha256"],
        "database_ids": plan["database_ids"],
        "policy_kind": "local_llm",
        "deterministic_fixture_expert": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "model": model_config.model_dump(),
        "settings": settings.model_dump(),
        "metrics": metrics,
        "failure_analysis": analysis,
        "runtime_stability": runtime_stability,
        "business_failures": analysis["business_failures"],
        "infrastructure_failures": analysis["infrastructure_failures"],
        "verification_errors": verification_errors,
        "episodes": results,
        "failure_trajectories": [
            item for item in results if item.get("failure_category") != "Successful"
        ],
        "error": extra_error,
        "interpretation_boundary": (
            "20 validation tasks measure this Base Agent baseline; they do not represent final "
            "generalization performance on the unseen test split."
        ),
        "created_at_utc": _now(),
    }
    _write_json(
        output / "summary.json",
        {
            "status": status,
            "task_count": len(plan["records"]),
            "completed_task_count": len(results),
            "business_failures": analysis["business_failures"],
            "infrastructure_failures": analysis["infrastructure_failures"],
            "top_failure_modes": analysis["top_failure_modes"],
            "metrics": metrics,
        },
    )
    _write_json(output / "benchmark-report.json", report)
    return report


def run_base_benchmark(
    policy: Policy,
    manifest_path: Path,
    database_root: Path,
    settings: Settings,
    model_config: ModelSettings,
    output: Path,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every validation task with the existing multi-turn runtime."""
    if policy.kind != "local_llm":
        raise ValueError("base_benchmark_requires_local_llm")
    plan = prepare_validation_plan(manifest_path, database_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_config(output, plan, model_config, settings)
    _git_state(output)
    write_environment(output, model_config, settings, runtime_metadata)
    _write_progress(
        output,
        schema_version="1.0",
        status="started",
        split="validation",
        task_count=len(plan["records"]),
        task_ids=plan["task_ids"],
        test_split_accessed=False,
        train_split_accessed=False,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )

    store = TrajectoryStore(output / "trajectories")
    results: list[dict[str, Any]] = []
    verification_errors: list[dict[str, Any]] = []
    infrastructure_failures: list[dict[str, Any]] = []
    runtime_metadata = runtime_metadata or {}
    for _index, record in enumerate(plan["records"]):
        task = BenchmarkTask.model_validate(record["task"])
        database_id = record["database_id"]
        database = database_root / f"{database_id}.sqlite3"
        task_slug = _safe_task_id(task.public.task_id)
        trajectory: Trajectory | None = None
        reward = None
        raw_trajectory_path: str | None = None
        task_trajectory_path: str | None = None
        task_infrastructure: list[dict[str, str]] = []
        verification_status = "not_run"
        try:
            trajectory = run_episode(
                task.public,
                policy,
                database,
                settings,
                output / "sql-audit.jsonl",
            )
            raw_path = store.save(trajectory)
            raw_trajectory_path = str(raw_path.relative_to(output))
            task_infrastructure = infrastructure_reasons(trajectory)
            if not task_infrastructure:
                try:
                    reward = verify(task, trajectory, database)
                    verification_status = "scored"
                    trajectory = trajectory.model_copy(update={"reward": reward})
                    raw_path = store.save(trajectory)
                    raw_trajectory_path = str(raw_path.relative_to(output))
                except Exception as exc:
                    verification_status = "infrastructure_error"
                    error = {
                        "task_id": task.public.task_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "run_id": trajectory.run_id,
                    }
                    verification_errors.append(error)
                    task_infrastructure.append(
                        {
                            "type": "verifier_infrastructure",
                            "code": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            task_trajectory_path = f"task-{task_slug}-trajectory.json"
            (output / task_trajectory_path).write_text(trajectory.model_dump_json(indent=2) + "\n")
        except Exception as exc:
            task_infrastructure.append(
                {
                    "type": "benchmark_infrastructure",
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
            verification_status = "infrastructure_error"
            verification_errors.append(
                {
                    "task_id": task.public.task_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if task_infrastructure:
            infrastructure_failures.extend(
                {"task_id": task.public.task_id, **reason} for reason in task_infrastructure
            )
        result = _task_result(
            trajectory,
            task,
            database_id,
            output,
            model_config,
            runtime_metadata,
            reward,
            verification_status,
            task_infrastructure,
            task_trajectory_path,
            raw_trajectory_path,
        )
        result_path = output / f"task-{task_slug}-result.json"
        _write_json(result_path, result)
        result["result_path"] = str(result_path.relative_to(output))
        # Rewrite once with the self-relative result path included.
        _write_json(result_path, result)
        results.append(result)
        _write_progress(
            output,
            status="blocked" if infrastructure_failures else "running",
            current_task=task.public.task_id,
            completed_task_count=len(results),
            infrastructure_failure_count=len(infrastructure_failures),
        )
        if infrastructure_failures:
            break

        # Read-only tasks must not mutate their validation database.
        after = database_content_fingerprint(database)
        if after != plan["database_fingerprints"][database_id]:
            reason = {
                "task_id": task.public.task_id,
                "type": "database_changed",
                "code": "validation_database_changed",
                "message": database_id,
            }
            infrastructure_failures.append(reason)
            result["failure_category"] = "Infrastructure Failure"
            result["success"] = False
            result["failure_reason"] = "validation_database_changed"
            result["infrastructure_errors"] = [reason]
            _write_json(result_path, result)
            break

    status = (
        "READY_FOR_SFT"
        if not infrastructure_failures and len(results) == len(plan["records"])
        else "BLOCKED"
    )
    _write_progress(
        output,
        status=status,
        completed_task_count=len(results),
        infrastructure_failure_count=len(infrastructure_failures),
        finished_at_utc=_now(),
    )
    return _build_report(
        output,
        plan,
        model_config,
        settings,
        results,
        verification_errors,
        infrastructure_failures,
        status,
    )


def write_incomplete_report(
    output: Path,
    manifest_path: Path,
    database_root: Path,
    model_config: ModelSettings,
    settings: Settings,
    error: dict[str, Any],
) -> dict[str, Any]:
    """Preserve a BLOCKED report if model startup or the supervisor fails."""
    output.mkdir(parents=True, exist_ok=True)
    try:
        plan = prepare_validation_plan(manifest_path, database_root)
    except Exception as plan_error:
        plan = {
            "manifest_sha256": None,
            "split_counts": {},
            "records": [{} for _ in range(20)],
            "task_ids": [],
            "database_ids": [],
            "database_fingerprints": {},
            "test_split_accessed": False,
            "train_split_accessed": False,
        }
        error = {
            **error,
            "plan_error": {"type": type(plan_error).__name__, "message": str(plan_error)},
        }
    _write_config(output, plan, model_config, settings)
    _git_state(output)
    write_environment(output, model_config, settings, {})
    results = []
    for path in sorted(output.glob("task-*-result.json")):
        try:
            results.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    infrastructure = [{"type": "supervisor", **error}]
    report = _build_report(
        output,
        plan,
        model_config,
        settings,
        results,
        [],
        infrastructure,
        "BLOCKED",
        extra_error=error,
    )
    _write_progress(
        output,
        status="BLOCKED",
        completed_task_count=len(results),
        infrastructure_failure_count=len(infrastructure),
        finished_at_utc=_now(),
    )
    return report
