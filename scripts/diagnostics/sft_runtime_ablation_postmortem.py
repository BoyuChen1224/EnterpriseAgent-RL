"""CPU-only postmortem for the completed SFT Runtime protocol ablation.

This module reads only saved train-only artifacts.  It deliberately does not
import the model, Environment, SQLTool, verifier, or task loader.  The SQL
strings included in the output are the already-recorded Action arguments and
are never executed.
"""

# The generated Markdown contains intentionally long evidence sentences.
# Keep code/style checks active for logic while excluding only line length.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

GROUPS = (
    "baseline",
    "initial-observation",
    "canonical-history",
    "no-format-retry",
    "full-alignment",
)
TASK_RESULT_SUFFIX = "-result.json"
TASK_TRAJECTORY_SUFFIX = "-trajectory.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _sql_preview(query: str | None, limit: int = 180) -> str | None:
    if query is None:
        return None
    compact = re.sub(r"\s+", " ", query).strip()
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _trajectory_for(root: Path, group: str, task_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((root / group).glob(f"*{task_id}{TASK_TRAJECTORY_SUFFIX}"))
    if len(candidates) != 1:
        raise ValueError(f"expected one trajectory for {group}/{task_id}, found {candidates}")
    path = candidates[0]
    return path, _read_json(path)


def _next_step(steps: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    return next((step for step in steps if step["step_index"] > index), None)


def _task_record(root: Path, group: str, result_path: Path) -> dict[str, Any]:
    result = _read_json(result_path)
    task_id = result["task_id"]
    trajectory_path, trajectory = _trajectory_for(root, group, task_id)
    steps = trajectory["steps"]
    sql_steps = [step for step in steps if step["action"]["kind"] == "sql"]
    sql_errors = [step for step in sql_steps if not step["tool_result"]["ok"]]
    first_sql_index = min((step["step_index"] for step in sql_steps), default=None)
    schema_dictionary_steps = [
        step for step in steps if step["action"]["kind"] in {"schema", "dictionary"}
    ]
    pre_sql_schema_dictionary = [
        step
        for step in schema_dictionary_steps
        if first_sql_index is None or step["step_index"] < first_sql_index
    ]

    sql_events: list[dict[str, Any]] = []
    for step in sql_steps:
        query = step["action"].get("arguments", {}).get("query")
        following = _next_step(steps, step["step_index"])
        sql_events.append(
            {
                "step_index": step["step_index"],
                "ok": bool(step["tool_result"]["ok"]),
                "error": step["tool_result"].get("error"),
                "query": query,
                "query_preview": _sql_preview(query),
                "query_sha256": hashlib.sha256(str(query).encode()).hexdigest(),
                "next_action": following["action"]["kind"] if following else None,
            }
        )

    error_feedback_next_actions = [
        (
            following["action"]["kind"]
            if (following := _next_step(steps, error_step["step_index"]))
            else None
        )
        for error_step in sql_errors
    ]
    actual_sql_recovery_count = sum(
        any(
            later["action"]["kind"] == "sql"
            and later["tool_result"]["ok"]
            and later["step_index"] > error_step["step_index"]
            for later in steps
        )
        for error_step in sql_errors
    )
    actual_any_tool_recovery_count = sum(
        any(
            later["tool_result"]["ok"] and later["step_index"] > error_step["step_index"]
            for later in steps
        )
        for error_step in sql_errors
    )

    check_steps = [step for step in steps if step["action"]["kind"] == "check"]
    finish_steps = [step for step in steps if step["action"]["kind"] == "finish"]
    parser_attempts = trajectory.get("model_attempts", [])
    parser_acceptances = sum(bool(attempt.get("format_valid")) for attempt in parser_attempts)
    action_kind_counts = Counter(step["action"]["kind"] for step in steps)
    first_action = steps[0]["action"] if steps else None
    terminal_reason = result["terminal_reason"]
    has_sql_success = any(event["ok"] for event in sql_events)
    has_sql_error = bool(sql_errors)

    failure_stages: list[str] = []
    if not steps or terminal_reason in {
        "invalid_json",
        "invalid_action",
        "invalid_action_arguments",
    }:
        failure_stages.append("parser_or_action_validation")
    if has_sql_error:
        failure_stages.append("sql_execution_error")
    if has_sql_success:
        failure_stages.append("sql_execution_reached")
    if has_sql_error and actual_sql_recovery_count == 0:
        failure_stages.append("no_successful_sql_after_error")
    if any(not step["tool_result"]["ok"] for step in check_steps + finish_steps):
        failure_stages.append("check_or_finish_evidence_error")
    if result["budget_exhausted"]:
        failure_stages.append("budget_exhaustion")

    if not steps or terminal_reason in {
        "invalid_json",
        "invalid_action",
        "invalid_action_arguments",
    }:
        primary_failure_stage = "parser_or_action_validation"
    elif has_sql_success and (has_sql_error or terminal_reason != "completed"):
        primary_failure_stage = "sql_success_not_completed"
    elif has_sql_error:
        primary_failure_stage = "sql_error_recovery"
    elif result["budget_exhausted"]:
        primary_failure_stage = "budget_or_termination"
    else:
        primary_failure_stage = "unconfirmed"

    verifier_result = result.get("verifier_result") or {}
    return {
        "task_id": task_id,
        "database_id": result["database_id"],
        "template_id": result["template_id"],
        "teacher_provenance": result.get("teacher_provenance"),
        "success": bool(result["success"]),
        "reward": result.get("reward"),
        "terminal_reason": terminal_reason,
        "failure_reason": result.get("failure_reason"),
        "verifier_reason": verifier_result.get("reason"),
        "verifier_result": verifier_result,
        "primary_failure_stage": primary_failure_stage,
        "failure_stages": failure_stages,
        "first_action": first_action["kind"] if first_action else None,
        "first_action_arguments": first_action.get("arguments") if first_action else None,
        "first_action_legal": bool(result["first_action_legal"]),
        "accepted_action_count": len(steps),
        "parser_attempts": len(parser_attempts),
        "parser_acceptances": parser_acceptances,
        "action_kind_counts": dict(sorted(action_kind_counts.items())),
        "invalid_json": result["invalid_json"],
        "invalid_envelope": result["invalid_envelope"],
        "invalid_arguments": result["invalid_arguments"],
        "format_retry_count": result["format_retry_count"],
        "schema_dictionary_steps": [
            {
                "step_index": step["step_index"],
                "kind": step["action"]["kind"],
                "arguments": step["action"].get("arguments", {}),
                "ok": bool(step["tool_result"]["ok"]),
                "error": step["tool_result"].get("error"),
            }
            for step in schema_dictionary_steps
        ],
        "pre_sql_schema_dictionary": [step["action"]["kind"] for step in pre_sql_schema_dictionary],
        "first_sql_step": first_sql_index,
        "sql_attempts": result["sql_attempts"],
        "executable_sql_count": result["executable_sql_count"],
        "sql_error_count": result["sql_error_count"],
        "sql_error_types": result["sql_error_types"],
        "sql_events": sql_events,
        "recovery_count": result["recovery_count"],
        "reported_recovery_success_count": result["recovery_success_count"],
        "reported_recovery_next_actions": error_feedback_next_actions,
        "actual_sql_recovery_count": actual_sql_recovery_count,
        "actual_any_tool_recovery_count": actual_any_tool_recovery_count,
        "repeated_sql": result["repeated_sql"],
        "check_present": bool(result["check_present"]),
        "check_action_count": len(check_steps),
        "check_ok_count": sum(bool(step["tool_result"]["ok"]) for step in check_steps),
        "finish_present": bool(result["finish_present"]),
        "finish_action_count": len(finish_steps),
        "finish_ok_count": sum(bool(step["tool_result"]["ok"]) for step in finish_steps),
        "budget_exhausted": bool(result["budget_exhausted"]),
        "tool_calls": result["tool_calls"],
        "action_count": result["action_count"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_s": result["latency_s"],
        "source_files": {
            "result": str(result_path.relative_to(root)),
            "trajectory": str(trajectory_path.relative_to(root)),
            "model_requests": f"{group}/model-requests.jsonl",
            "sql_audit": f"{group}/sql-audit.jsonl",
            "raw_generations": f"{group}/raw-generations.jsonl",
        },
    }


def load_records(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for group in GROUPS:
        result_paths = sorted((root / group).glob(f"*{TASK_RESULT_SUFFIX}"))
        if len(result_paths) != 20:
            raise ValueError(f"{group}: expected 20 result files, found {len(result_paths)}")
        group_records = {
            (record := _task_record(root, group, path))["task_id"]: record for path in result_paths
        }
        if len(group_records) != 20:
            raise ValueError(f"{group}: duplicate task ids")
        records[group] = group_records
    task_ids = set(records[GROUPS[0]])
    if any(set(records[group]) != task_ids for group in GROUPS[1:]):
        raise ValueError("groups do not contain the same task ids")
    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    parser_attempts = sum(record["parser_attempts"] for record in records)
    parser_acceptances = sum(record["parser_acceptances"] for record in records)
    sql_attempts = sum(record["sql_attempts"] for record in records)
    executable_sql = sum(record["executable_sql_count"] for record in records)
    sql_errors = sum(record["sql_error_count"] for record in records)
    recovery_count = sum(record["recovery_count"] for record in records)
    recovery_success = sum(record["reported_recovery_success_count"] for record in records)
    sql_error_types: Counter[str] = Counter()
    for record in records:
        sql_error_types.update(record["sql_error_types"])

    action_distribution: Counter[str] = Counter()
    for record in records:
        action_distribution.update(record["action_kind_counts"])

    def average(key: str) -> float:
        return sum(float(record[key] or 0) for record in records) / count if count else 0.0

    # This recomputation is for all requested scalar definitions.
    return {
        "task_count": count,
        "task_success_count": sum(record["success"] for record in records),
        "task_success_rate": _ratio(sum(record["success"] for record in records), count),
        "parser_acceptances": parser_acceptances,
        "parser_attempts": parser_attempts,
        "parser_acceptance_rate": _ratio(parser_acceptances, parser_attempts),
        "first_action_legal_rate": _ratio(
            sum(record["first_action_legal"] for record in records), count
        ),
        "schema_dictionary_before_first_sql_count": sum(
            bool(record["pre_sql_schema_dictionary"]) for record in records
        ),
        "schema_dictionary_before_first_sql_coverage": _ratio(
            sum(bool(record["pre_sql_schema_dictionary"]) for record in records), count
        ),
        "sql_attempts": sql_attempts,
        "executable_sql_count": executable_sql,
        "sql_executable_rate": _ratio(executable_sql, sql_attempts),
        "sql_error_count": sql_errors,
        "sql_error_rate": _ratio(sql_errors, sql_attempts),
        "recovery_count": recovery_count,
        "reported_recovery_success_count": recovery_success,
        "recovery_success_rate": _ratio(recovery_success, recovery_count),
        "actual_sql_recovery_count": sum(record["actual_sql_recovery_count"] for record in records),
        "sql_repeat_count": sum(record["repeated_sql"] for record in records),
        "sql_repeat_rate": _ratio(sum(record["repeated_sql"] for record in records), sql_attempts),
        "check_task_count": sum(record["check_present"] for record in records),
        "check_rate": _ratio(sum(record["check_present"] for record in records), count),
        "finish_task_count": sum(record["finish_present"] for record in records),
        "finish_rate": _ratio(sum(record["finish_present"] for record in records), count),
        "budget_exhausted_task_count": sum(record["budget_exhausted"] for record in records),
        "budget_exhaustion_rate": _ratio(
            sum(record["budget_exhausted"] for record in records), count
        ),
        "average_tool_calls": average("tool_calls"),
        "average_latency_s": average("latency_s"),
        "average_input_tokens": average("input_tokens"),
        "average_output_tokens": average("output_tokens"),
        "average_total_tokens": average("input_tokens") + average("output_tokens"),
        "average_reward": average("reward"),
        "invalid_json_count": sum(record["invalid_json"] for record in records),
        "invalid_envelope_count": sum(record["invalid_envelope"] for record in records),
        "invalid_arguments_count": sum(record["invalid_arguments"] for record in records),
        "format_retry_count": sum(record["format_retry_count"] for record in records),
        "sql_error_types": dict(sorted(sql_error_types.items())),
        "action_kind_distribution_recomputed": dict(sorted(action_distribution.items())),
    }


def _recorded_metrics(root: Path, group: str) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = _read_json(root / "comparison.json")["groups"][group]["metrics"]
    summary = _read_json(root / group / "summary.json")["metrics"]
    return comparison, summary


def compare_metrics(
    computed: dict[str, Any], comparison: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "task_count",
        "task_success_count",
        "task_success_rate",
        "parser_acceptances",
        "parser_attempts",
        "parser_acceptance_rate",
        "first_action_legal_rate",
        "schema_dictionary_before_first_sql_coverage",
        "sql_executable_rate",
        "sql_error_rate",
        "recovery_success_rate",
        "sql_repeat_rate",
        "check_rate",
        "finish_rate",
        "budget_exhaustion_rate",
        "average_tool_calls",
        "average_latency_s",
        "average_input_tokens",
        "average_output_tokens",
        "average_total_tokens",
        "average_reward",
        "invalid_json_count",
        "invalid_envelope_count",
        "invalid_arguments_count",
        "format_retry_count",
    )

    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, float) or isinstance(right, float):
            return abs(float(left) - float(right)) <= 1e-12
        return left == right

    comparison_mismatches = {
        field: {"computed": computed.get(field), "recorded": comparison.get(field)}
        for field in fields
        if not equal(computed.get(field), comparison.get(field))
    }
    summary_mismatches = {
        field: {"computed": computed.get(field), "recorded": summary.get(field)}
        for field in fields
        if not equal(computed.get(field), summary.get(field))
    }
    return {
        "comparison_mismatches": comparison_mismatches,
        "summary_mismatches": summary_mismatches,
        "all_scalar_metrics_match": not comparison_mismatches and not summary_mismatches,
    }


def build_common_patterns(records: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    task_ids = sorted(records[GROUPS[0]])

    def intersection(predicate: Any) -> list[str]:
        return [
            task_id
            for task_id in task_ids
            if all(predicate(records[group][task_id]) for group in GROUPS)
        ]

    patterns = [
        {
            "id": "all_tasks_zero_success",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: not record["success"]),
            "evidence": "All five saved result records have success=false and reward=0 for each task.",
        },
        {
            "id": "no_accepted_action_on_constraints_and_recovery",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: not record["first_action_legal"]),
            "evidence": "The first model attempt is not format-valid in every group; no accepted step is saved for these tasks.",
        },
        {
            "id": "sql_error_in_every_group",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: record["sql_error_count"] > 0),
            "evidence": "Each listed task has at least one recorded SQL tool error in every group.",
        },
        {
            "id": "no_successful_sql_after_error",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(
                lambda record: record["sql_error_count"] > 0
                and record["actual_sql_recovery_count"] == 0
            ),
            "evidence": "No saved group trajectory contains a later successful SQL Action after an earlier SQL error for these tasks.",
        },
        {
            "id": "no_finish_in_every_group",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: not record["finish_present"]),
            "evidence": "Finish presence is an Action-kind count, not proof of successful termination; this intersection is stricter than group-level finish rate.",
        },
        {
            "id": "no_check_in_every_group",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: not record["check_present"]),
            "evidence": "No Check Action is present in any saved trajectory for these task ids across all groups.",
        },
        {
            "id": "exact_sql_repetition_in_every_group",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(lambda record: record["repeated_sql"] > 0),
            "evidence": "Repeated SQL is exact raw query-string duplication as implemented by the ablation aggregator.",
        },
        {
            "id": "error_feedback_stays_sql_or_terminates",
            "classification": "confirmed_common_failure",
            "task_ids": intersection(
                lambda record: bool(record["reported_recovery_next_actions"])
                and all(
                    action in {"sql", None} for action in record["reported_recovery_next_actions"]
                )
            ),
            "evidence": "For these tasks, every group either emits another SQL Action after the error or has no next Action.",
        },
    ]
    return patterns


def build_postmortem(root: Path) -> dict[str, Any]:
    records = load_records(root)
    task_ids = sorted(records[GROUPS[0]])
    group_summaries: dict[str, Any] = {}
    metric_checks: dict[str, Any] = {}
    for group in GROUPS:
        computed = aggregate_records(list(records[group].values()))
        comparison, summary = _recorded_metrics(root, group)
        group_summaries[group] = {
            "computed": computed,
            "comparison_json": comparison,
            "summary_json": summary,
        }
        metric_checks[group] = compare_metrics(computed, comparison, summary)

    paired_tasks = []
    for task_id in task_ids:
        first = records[GROUPS[0]][task_id]
        paired_tasks.append(
            {
                "task_id": task_id,
                "database_id": first["database_id"],
                "template_id": first["template_id"],
                "teacher_provenance": first["teacher_provenance"],
                "groups": {group: records[group][task_id] for group in GROUPS},
            }
        )

    b_records = records["initial-observation"]
    b_schema_tasks = [
        task_id for task_id, record in b_records.items() if record["pre_sql_schema_dictionary"]
    ]
    b_dictionary_tasks = [
        task_id
        for task_id, record in b_records.items()
        if any(kind == "dictionary" for kind in record["pre_sql_schema_dictionary"])
    ]
    b_executable_tasks = [
        task_id for task_id, record in b_records.items() if record["executable_sql_count"] > 0
    ]
    b_task_details = {}
    for task_id in sorted(set(b_schema_tasks) | set(b_executable_tasks)):
        b = b_records[task_id]
        a = records["baseline"][task_id]
        b_task_details[task_id] = {
            "baseline": {
                "accepted_action_count": a["accepted_action_count"],
                "first_action": a["first_action"],
                "sql_attempts": a["sql_attempts"],
                "executable_sql_count": a["executable_sql_count"],
                "terminal_reason": a["terminal_reason"],
                "source_files": a["source_files"],
            },
            "initial_observation": {
                "pre_sql_schema_dictionary": b["pre_sql_schema_dictionary"],
                "schema_dictionary_steps": b["schema_dictionary_steps"],
                "first_sql_step": b["first_sql_step"],
                "sql_events": b["sql_events"],
                "check_present": b["check_present"],
                "finish_present": b["finish_present"],
                "terminal_reason": b["terminal_reason"],
                "source_files": b["source_files"],
            },
            "paired_sql_comparison": {
                "baseline_has_saved_sql": a["sql_attempts"] > 0,
                "initial_observation_has_saved_sql": b["sql_attempts"] > 0,
                "direct_query_diff_confirmable": a["sql_attempts"] > 0 and b["sql_attempts"] > 0,
                "conclusion": (
                    "unconfirmed: baseline has no accepted SQL Action, so a direct A/B SQL-structure comparison is unavailable"
                    if a["sql_attempts"] == 0
                    else "see saved SQL events"
                ),
            },
        }

    e_records = records["full-alignment"]
    e_executable_tasks = [
        task_id for task_id, record in e_records.items() if record["executable_sql_count"] > 0
    ]
    e_sql_recovery_tasks = [
        task_id for task_id, record in e_records.items() if record["actual_sql_recovery_count"] > 0
    ]
    e_check_error_count = sum(
        record["check_action_count"] - record["check_ok_count"] for record in e_records.values()
    )
    e_finish_error_count = sum(
        record["finish_action_count"] - record["finish_ok_count"] for record in e_records.values()
    )

    return {
        "schema_version": "1.0",
        "status": "SFT_RUNTIME_POSTMORTEM_COMPLETE",
        "scope": {
            "artifact_root": str(root),
            "split": "train",
            "task_count": len(task_ids),
            "groups": list(GROUPS),
            "model_or_sql_execution_performed": False,
            "verifier_called": False,
            "validation_test_accessed": False,
        },
        "metric_semantics": {
            "task_success_rate": {
                "numerator": "saved result.success=true count",
                "denominator": "task count",
                "source": "scripts/diagnostics/sft_runtime_protocol_ablation.py:_aggregate",
            },
            "parser_acceptance": {
                "numerator": "saved model attempt format_valid=true count",
                "denominator": "all saved model attempts, including format retries",
                "unit": "model output attempt, not accepted Action step",
                "source": "scripts/diagnostics/sft_runtime_protocol_ablation.py:_task_result and _aggregate",
            },
            "first_action_legal": {
                "definition": "format_valid on the first attempt of the first accepted step",
                "caveat": "name is misleading: it is parser validity, not semantic Action legality",
            },
            "schema_dictionary_before_first_sql": {
                "numerator": "tasks with at least one schema/dictionary Action before first SQL Action",
                "denominator": "all tasks, including tasks with no SQL Action",
                "source": "scripts/diagnostics/sft_runtime_protocol_ablation.py:_task_result and _aggregate",
            },
            "sql_executable_rate": {
                "numerator": "SQL Actions with tool_result.ok=true",
                "denominator": "all SQL Actions",
                "source": "scripts/diagnostics/sft_runtime_protocol_ablation.py:_task_result and _aggregate",
            },
            "sql_error_rate": {
                "numerator": "SQL Actions with tool_result.ok=false",
                "denominator": "all SQL Actions",
                "caveat": "includes recorded budget/action terminal sentinels when attached to an SQL step",
            },
            "recovery": {
                "recovery_count": "number of SQL errors; not a separate recovery-action count",
                "reported_success": "an SQL error has any later step with tool_result.ok=true, including Check",
                "denominator": "SQL error count; zero-error groups receive 0.0 through the aggregator's zero-denominator rule",
                "actual_sql_recovery": "postmortem-only stricter count: a later successful SQL Action after an earlier SQL error",
                "source": "scripts/diagnostics/sft_runtime_protocol_ablation.py:_task_result",
            },
            "sql_repetition_rate": {
                "numerator": "len(raw SQL query strings) - len(set(raw SQL query strings))",
                "denominator": "all SQL Actions",
                "definition": "exact string duplication; not semantic SQL equivalence",
            },
            "check_finish_rate": {
                "numerator": "tasks containing at least one Action kind check/finish",
                "denominator": "task count",
                "caveat": "presence does not require tool_result.ok or completed terminal state",
            },
            "average_tool_calls": {
                "unit": "mean per task of steps with tool_name != null",
                "excludes": ["plan", "finish"],
            },
            "reward": {
                "definition": "mean saved result.reward value from the completed run",
                "caveat": "stored verifier/reward output is read only; verifier is not called in this postmortem",
                "source": "src/enterprise_agent_rl/rewards.py:verify and saved result files",
            },
        },
        "metric_checks": metric_checks,
        "group_summaries": group_summaries,
        "paired_tasks": paired_tasks,
        "b_initial_observation_analysis": {
            "schema_before_sql_tasks": sorted(b_schema_tasks),
            "dictionary_before_sql_tasks": sorted(b_dictionary_tasks),
            "executable_sql_tasks": sorted(b_executable_tasks),
            "task_details": b_task_details,
            "conclusion": "B's 10% coverage is two Schema Actions on the two top tasks; no Dictionary Action is accepted before first SQL.",
        },
        "e_full_alignment_analysis": {
            "executable_sql_tasks": sorted(e_executable_tasks),
            "actual_sql_recovery_tasks": sorted(e_sql_recovery_tasks),
            "actual_sql_recovery_task_count": len(e_sql_recovery_tasks),
            "check_action_task_count": sum(
                record["check_present"] for record in e_records.values()
            ),
            "check_action_error_count": e_check_error_count,
            "finish_action_task_count": sum(
                record["finish_present"] for record in e_records.values()
            ),
            "finish_action_error_count": e_finish_error_count,
            "terminal_reason_counts": dict(
                Counter(record["terminal_reason"] for record in e_records.values())
            ),
            "conclusion": "E's successful SQL Actions are the two top-task probe queries; no task executes a successful SQL after an error, and lower calls do not establish improved completion.",
        },
        "common_failure_patterns": build_common_patterns(records),
        "conclusions": {
            "confirmed": [
                "All five groups fail all 20 tasks and have zero saved reward.",
                "Four task ids (both constraints and recovery tasks) have no accepted Action step in every group.",
                "No group contains a successful SQL Action after an earlier SQL error for the task-level SQL-error set.",
                "The reported recovery metric counts later successful non-SQL tools such as Check; it is not SQL recovery.",
            ],
            "likely": [
                "Malformed SQL arguments/schema assumptions are the dominant downstream bottleneck after the parser/action gate; saved errors repeatedly cite nonexistent tables or columns and the existing SQL argument audit reports the same supervision problem.",
                "E's lower tool-call count is partly early termination and one-shot Check/Finish behavior under format_retries=0, not a demonstrated planning improvement.",
            ],
            "unconfirmed": [
                "Whether retaining the initial observation directly caused the two top-task probe executions, because baseline top tasks terminate before any accepted SQL Action.",
                "Whether any successful SQL result was semantically correct; the saved successful probes are not final task answers and no new verifier call is permitted.",
                "Whether canonical history changes behavior independently; C's saved aggregate and task paths match A on the principal failure signals.",
            ],
        },
        "next_experiment_recommendation": {
            "priority": "B: SQL parameter supervision and Dictionary/Schema branch learning diagnostic",
            "why_first": [
                "B/E are the only groups that reach executable SQL, and both do so only on the two top-task probe queries.",
                "Both then emit the same recorded main-query failure (no such column: o.list_price_cents) and repeat it; no group recovers with a later successful SQL Action.",
                "The existing SQL argument analysis independently reports SQL parameter mismatch and Dictionary-to-Schema/SQL substitutions, giving a direct training-target hypothesis.",
            ],
            "hypothesis_to_separate": "The primary bottleneck is learned Action-kind/argument and schema-grounded SQL parameter selection, rather than only Runtime history serialization or format retry.",
            "minimal_design": "CPU-only audit/replay of the existing train SFT records and saved trajectories: map each assistant target to Action kind/arguments, compare to the corresponding failed runtime Action and error, and quantify branch transitions before first SQL. Do not rerun any task or query.",
            "gpu_required": False,
            "production_code_change_required": False,
            "avoid": [
                "retraining",
                "GRPO",
                "formal validation/test evaluation",
                "repeating the completed five-group Runtime ablation",
            ],
            "secondary_followup": "After the SQL-argument bottleneck is characterized, run a separate offline recovery decision audit; do not interpret reported recovery_success as SQL recovery.",
        },
    }


def _fmt_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _task_cell(record: dict[str, Any]) -> str:
    pre = ",".join(record["pre_sql_schema_dictionary"]) or "-"
    next_actions = (
        ",".join(
            sorted(set(action for action in record["reported_recovery_next_actions"] if action))
        )
        or "-"
    )
    return (
        f"stage={record['primary_failure_stage']}; first={record['first_action'] or '-'}; "
        f"pre={pre}; sql={record['sql_attempts']}/{record['executable_sql_count']}/{record['sql_error_count']}; "
        f"rec={record['reported_recovery_success_count']}/{record['recovery_count']} "
        f"(sql={record['actual_sql_recovery_count']}); next={next_actions}; "
        f"rep={record['repeated_sql']}; C/F={int(record['check_present'])}/{int(record['finish_present'])}; "
        f"budget={int(record['budget_exhausted'])}; reward={record['reward']}; "
        f"term={record['terminal_reason']}"
    )


def render_markdown(postmortem: dict[str, Any]) -> str:
    lines = [
        "# SFT Runtime Ablation Postmortem",
        "",
        "状态：`SFT_RUNTIME_POSTMORTEM_COMPLETE`",
        "",
        "本报告仅读取已完成的 train-only Runtime ablation artifacts、相关诊断代码和训练数据元数据；未加载模型、执行 SQL、调用 Verifier、访问 validation/test、提交 Slurm、训练或修改生产 Runtime。原始 artifacts 未被修改。",
        "",
        "## 1. 证据与指标口径核验",
        "",
        "指标重算来自每组 20 个 `*-result.json` 与对应 `*-trajectory.json`；`comparison.json`、各组 `summary.json` 与本次 scalar 重算的差异见 `postmortem.json.metric_checks`。",
        "",
        """| 指标 | 统计口径 | 关键限制 |
|---|---|---|
| Task Success Rate | `success=true` task / 20 | 读取保存的 reward/verifier 结果，不重新调用 Verifier |
| Parser acceptance | `format_valid=true` model attempt / 全部 model attempts | 单位是 model output attempt，包含 format retry，不是 accepted Action step |
| 首次 SQL 前 Schema/Dictionary | 满足条件的 task / 20 | 条件是任一 Schema/Dictionary Action 严格位于首个 SQL Action 之前 |
| SQL executable/error | `SQL tool_result.ok=true/false` / SQL Action | error 包括保存的 budget/action sentinel |
| Recovery | SQL error 数；reported success 为其后任一 `tool_result.ok=true` | 可能把成功 Check 算作 recovery；不是 SQL recovery |
| SQL repetition | `len(SQL query strings)-len(set(query strings))` / SQL Action | 精确字符串重复，不判断语义等价 |
| Check/Finish | 含对应 Action kind 的 task / 20 | 不要求 Action 成功，也不要求 episode completed |
| Average tool calls | 每 task 的 `tool_name != null` steps 平均值 | 排除 Plan 和 Finish |
| Reward | 保存的 reward value 的 task 平均值 | 本报告不重新验证 |
""",
        "",
        "补充：`first_action_legal` 实际是第一 accepted step 的第一 model attempt 的 `format_valid`；名称不能解释为语义 Action 合法。`recovery_count` 本身就是 SQL error 数；因此 `recovery_count=0` 表示没有 SQL error。`recovery_success_rate=0` 则还可能表示有 error 但其后没有成功 tool，或无 error 时聚合器的零分母默认值。",
        "",
        "## 2. 五组汇总",
        "",
        """| 组 | Success | Parser | 首次 SQL 前 Schema/Dict | SQL executable | SQL error | Reported recovery | 实际 SQL recovery | SQL repeat | Check | Finish | Avg tool calls | Avg latency (s) | Avg reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|""",
    ]
    for group in GROUPS:
        metrics = postmortem["group_summaries"][group]["computed"]
        lines.append(
            "| {group} | {success}/20 | {parser} | {schema} | {exec} | {error} | {recovery} | {actual} | {repeat} | {check} | {finish} | {tools:.2f} | {latency:.2f} | {reward:.2f} |".format(
                group=group,
                success=metrics["task_success_count"],
                parser=_fmt_percent(metrics["parser_acceptance_rate"]),
                schema=_fmt_percent(metrics["schema_dictionary_before_first_sql_coverage"]),
                exec=_fmt_percent(metrics["sql_executable_rate"]),
                error=_fmt_percent(metrics["sql_error_rate"]),
                recovery=_fmt_percent(metrics["recovery_success_rate"]),
                actual=metrics["actual_sql_recovery_count"],
                repeat=_fmt_percent(metrics["sql_repeat_rate"]),
                check=_fmt_percent(metrics["check_rate"]),
                finish=_fmt_percent(metrics["finish_rate"]),
                tools=metrics["average_tool_calls"],
                latency=metrics["average_latency_s"],
                reward=metrics["average_reward"],
            )
        )
    lines += [
        "",
        "Reported recovery is not actual SQL recovery: B reports 9.47% because some SQL errors are followed by successful non-SQL tool results; no group has a later successful SQL Action after an earlier SQL error. E's average calls fall to 6.75, but this accompanies 9 `invalid_action` and 2 `invalid_action_arguments` terminal outcomes, so lower calls are not evidence of better completion.",
        "",
        "## 3. 20-task paired comparison",
        "",
        "单元格格式：`sql=attempts/executable/errors`；`rec=reported-success/error-count (actual SQL recovery count)`；`C/F` 是 Action presence；完整 verifier result、SQL events 与 source file 关联保存在 `postmortem.json.paired_tasks`。",
        "",
        """| task_id | template | provenance | A baseline | B initial-observation | C canonical-history | D no-format-retry | E full-alignment |
|---|---|---|---|---|---|---|---|""",
    ]
    for task in postmortem["paired_tasks"]:
        cells = [_task_cell(task["groups"][group]).replace("|", "\\|") for group in GROUPS]
        lines.append(
            f"| {task['task_id']} | {task['template_id']} | {task['teacher_provenance']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "任务级阶段判断只使用对应 trajectory/result：parser/action gate、SQL execution error、SQL success reached、no successful SQL after error、Check/Finish evidence error、budget exhaustion 等标签不是重新运行得到的推断。",
        "",
        "## 4. B 组：Initial Observation Retention",
        "",
        "B 的 Schema/Dictionary 覆盖任务只有 `train-0-top` 与 `train-1-top`，两者都是 step 1 的 `schema {}`，没有 Dictionary Action；Schema 均发生在首个 SQL step 2 之前。",
        "",
        "两条任务在 step 2 都成功执行已保存的 probe SQL：`SELECT COUNT(*) AS paid_orders FROM orders WHERE status='paid' AND ordered_at LIKE '2021%' LIMIT 1`。随后主 SQL 在 step 3 起报告 `no such column: o.list_price_cents`，后续输出继续重复该错误结构，最终 `tool_budget_exhausted`；没有 Check 或 Finish。",
        "",
        "A 中这两个 top task 没有 accepted SQL Action（terminal `invalid_action_arguments`），因此不存在可核验的 A/B SQL 参数或结构差异；“Initial Observation 直接导致 SQL 改善”仍为 `unconfirmed`。可确认的是 B 把这两条任务推进到了 probe SQL，但没有把 schema 访问转化为正确的主 SQL 参数。",
        "",
        "## 5. E 组：低调用量但未完成",
        "",
        "E 的两个 executable SQL task 仍只有 `train-0-top` 与 `train-1-top`，成功的是同一类 probe SQL；成功 SQL 之后继续执行主查询，遇到 `no such column: o.list_price_cents`，重复并耗尽 tool budget。E 没有任何 task 在 SQL error 后再次成功执行 SQL。",
        "",
        "E 有 11/20 task 出现 Check Action，但不少 Check 的保存结果是 `invalid_evidence_step`；只有 1/20 task 出现 Finish，且 Finish 结果并未让 episode completed。E 的低调用量来自 `format_retries=0` 下更早的 invalid action/argument 终止，以及部分单次 SQL→Check 路径；不能解释为自主规划改善。",
        "",
        "## 6. 五组共有与分组特有失败模式",
        "",
        "### Confirmed common failure",
        "",
    ]
    for pattern in postmortem["common_failure_patterns"]:
        if pattern["classification"] == "confirmed_common_failure":
            lines.append(
                f"- `{pattern['id']}`（{len(pattern['task_ids'])} tasks）：{pattern['evidence']} IDs: {', '.join(pattern['task_ids'])}。"
            )
    lines += [
        "",
        "### Group-specific failure",
        "",
        "- B/E 特有的正向路径是两个 top task 的 Schema→probe SQL；其后的主 SQL 参数错误仍未恢复。",
        "- B 的 reported recovery 上升来自后续成功的非 SQL tool result；D/E reported recovery 为 0，不能据此直接比较 SQL recovery 能力。",
        "- D/E 的无重试路径更容易在 parser/action validation 阶段结束；E 同时有较多 Check Action，但 Check 证据错误仍存在。",
        "",
        "### Likely / Unconfirmed",
        "",
        "- `Likely`：SQL 参数、表/字段假设和 Action-kind/argument 分支学习是端到端失败的主要下游瓶颈；这是由保存的错误文本、重复 query 参数和已有 SQL argument audit 共同支持的训练归因，但本次没有重新评估 SQL 语义。",
        "- `Unconfirmed`：Schema Action 是否真正改变了后续 query 结构；A top task 没有可配对的 SQL。",
        "- `Unconfirmed`：成功 probe 的结果是否满足业务目标；它不是最终答案，且本次禁止再次调用 Verifier。",
        "",
        "## 7. 下一阶段最小实验建议",
        "",
        "优先选择：**B. SQL 参数监督和 Dictionary/Schema 分支学习诊断**。理由是：B/E 是唯一到达 executable SQL 的组，且只到达两个 top-task probe；之后二者都重复 `o.list_price_cents` 错误，没有 SQL recovery。既有 `sft-sql-argument-error-analysis.md` 还直接记录 SQL 参数不匹配以及 Dictionary target 被替换为 Schema/SQL，这比继续重复 Runtime 协议消融更接近已定位的失败阶段。",
        "",
        "建议先做 CPU-only 的离线数据/轨迹审计：逐条对齐 SFT assistant target 的 Action kind/arguments、保存的 runtime Action、SQL error 和下一步 Action，统计首次 SQL 前的 Schema/Dictionary 分支与参数错误。该实验不需要 GPU，不修改生产代码，不重新执行 task/SQL/Verifier，不重复五组 Runtime 实验，也不需要增加训练轮数、扩充数据或启动 GRPO。",
        "",
        "Recovery 行为诊断可作为后续第二步，但必须使用“error 后成功 SQL”作为主指标，不能复用当前 reported `recovery_success_rate` 的含义。",
        "",
        "## 8. 证据路径",
        "",
        "机器可读的逐任务证据见 `artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation/postmortem.json`；每条记录包含 result、trajectory、request、SQL audit、raw generation 的相对路径。指标实现依据为 `scripts/diagnostics/sft_runtime_protocol_ablation.py:_task_result` / `_aggregate`，奖励定义依据为 `src/enterprise_agent_rl/rewards.py:verify`；本报告没有调用这些执行路径。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    postmortem = build_postmortem(args.artifact_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(postmortem, ensure_ascii=False, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(postmortem))
    print(
        json.dumps(
            {
                "status": postmortem["status"],
                "report": str(args.report),
                "json": str(args.json_output),
            }
        )
    )


if __name__ == "__main__":
    main()
