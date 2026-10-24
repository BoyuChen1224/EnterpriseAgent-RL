#!/usr/bin/env python3
"""Attribute train-only SQL failures by action-branch coverage.

This diagnostic deliberately consumes the already materialized
``sql-parameter-supervision-audit.json`` only.  It does not load a dataset
loader, model, tokenizer, runtime, verifier, or database.  The prior audit
contains the source-file pointers and the train-only event records needed for
this second-stage aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = (
    REPO_ROOT
    / "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation"
    / "sql-parameter-supervision-audit.json"
)
REPORT_PATH = REPO_ROOT / "docs/experiments/sft-sql-branch-attribution.md"
OUTPUT_PATH = (
    REPO_ROOT
    / "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation"
    / "sql-branch-attribution.json"
)

GROUPS = (
    "baseline",
    "initial-observation",
    "canonical-history",
    "no-format-retry",
    "full-alignment",
)
GROUP_LABELS = {
    "baseline": "A",
    "initial-observation": "B",
    "canonical-history": "C",
    "no-format-retry": "D",
    "full-alignment": "E",
}
BRANCH_CLASSES = (
    "schema_and_dictionary",
    "schema_only",
    "dictionary_only",
    "neither",
)
SPECIAL_TASKS = (
    "train-0-top",
    "train-1-top",
    "train-0-product",
    "train-1-product",
    "train-0-recovery",
    "train-1-recovery",
)
MISMATCH_CATEGORIES = (
    "identifier_mismatch",
    "relation_join_mismatch",
    "filter_date_dialect_mismatch",
    "aggregation_group_order_mismatch",
    "syntax_or_serialization_error",
    "formatting_only",
    "unclassified",
)


def _json_hash(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None}, key=str)


def branch_class(schema_evidence: bool, dictionary_evidence: bool) -> str:
    """Return a stable four-way branch stratum."""

    if schema_evidence and dictionary_evidence:
        return "schema_and_dictionary"
    if schema_evidence:
        return "schema_only"
    if dictionary_evidence:
        return "dictionary_only"
    return "neither"


def classify_runtime_error_category(event: dict[str, Any]) -> str:
    """Classify only the saved tool error text, never by executing SQL."""

    if event.get("tool_result_ok") is True:
        return "not_error"
    if event.get("error_is_budget_or_terminal_sentinel"):
        return "budget_or_terminal_sentinel"
    error = str(event.get("error") or "").lower()
    if not error:
        return "unclassified"
    if "no such column" in error or "no such table" in error:
        return "identifier_mismatch"
    if "no such function" in error or "unknown function" in error:
        return "filter_date_dialect_mismatch"
    if "syntax error" in error or 'near "' in error or "incomplete input" in error:
        return "syntax_or_serialization_error"
    return "unclassified"


def _identifiers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _gold_mismatch_category(event: dict[str, Any]) -> str:
    comparison = event.get("gold_comparison") or {}
    if comparison.get("exact_gold_sequence_indices"):
        return "exact_or_no_mismatch"
    if comparison.get("format_equal_gold_sequence_indices"):
        return "formatting_only"
    classification = comparison.get("classification")
    if isinstance(classification, dict):
        return classification.get("primary_category") or "unclassified"
    if isinstance(classification, str) and classification:
        return classification
    return "unclassified"


def _dataset_action_for_teacher_record(
    task: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any] | None:
    message_index = record.get("assistant_message_index")
    for action in task.get("actions", []):
        if action.get("kind") == "sql" and action.get("assistant_message_index") == message_index:
            return action
    gold_action = record.get("gold_action")
    for action in task.get("actions", []):
        if action.get("kind") == "sql" and action.get("arguments") == (gold_action or {}).get(
            "arguments"
        ):
            return action
    return None


def _gold_next_action(task: dict[str, Any], action: dict[str, Any] | None) -> str | None:
    if action is None:
        return None
    sequence_index = action.get("sequence_index")
    if not isinstance(sequence_index, int):
        return None
    for candidate in task.get("actions", []):
        if candidate.get("sequence_index") == sequence_index + 1:
            return candidate.get("kind")
    return None


def build_teacher_sql_events(
    dataset_tasks: list[dict[str, Any]], teacher_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    task_map = {task.get("task_id"): task for task in dataset_tasks}
    events: list[dict[str, Any]] = []
    for record in teacher_records:
        task = task_map.get(record.get("task_id"), {})
        action = _dataset_action_for_teacher_record(task, record)
        preceding_kinds = list((action or {}).get("preceding_action_kinds") or [])
        preceding_schema_action = "schema" in preceding_kinds
        preceding_dictionary_action = "dictionary" in preceding_kinds
        visibility = (action or {}).get("sql_visibility") or {}
        schema_visible = (
            preceding_schema_action and visibility.get("status") == "confirmed_visible_schema_check"
        )
        dictionary_visible = preceding_dictionary_action
        gold_query = record.get("gold_query")
        predicted_query = record.get("predicted_query")
        mismatch = record.get("sql_mismatch") or {}
        primary_category = mismatch.get("primary_category")
        if not primary_category:
            primary_category = (
                "exact_or_no_mismatch" if record.get("exact_match") else "unclassified"
            )
        previous_dictionary_arguments = [
            candidate.get("arguments")
            for candidate in task.get("actions", [])
            if candidate.get("sequence_index", -1) < (action or {}).get("sequence_index", -1)
            and candidate.get("kind") == "dictionary"
        ]
        previous_schema_arguments = [
            candidate.get("arguments")
            for candidate in task.get("actions", [])
            if candidate.get("sequence_index", -1) < (action or {}).get("sequence_index", -1)
            and candidate.get("kind") == "schema"
        ]
        events.append(
            {
                "history_source": "gold_teacher_forced_prefix",
                "task_id": record.get("task_id"),
                "group": "teacher_forced",
                "group_label": "E3",
                "database_id": record.get("database_id"),
                "template_id": record.get("template_id"),
                "trajectory_source": record.get("trajectory_source"),
                "prediction_index": record.get("prediction_index"),
                "target_id": record.get("target_id"),
                "assistant_message_index": record.get("assistant_message_index"),
                "sql_event_index": record.get("prediction_index"),
                "query_hash": _json_hash(predicted_query),
                "gold_query_hash": _json_hash(gold_query),
                "preceding_schema_accepted": None,
                "preceding_dictionary_accepted": None,
                "gold_prefix_schema_action": preceding_schema_action,
                "gold_prefix_dictionary_action": preceding_dictionary_action,
                "gold_prefix_schema_visible": schema_visible,
                "gold_prefix_dictionary_visible": dictionary_visible,
                "gold_prefix_branch_class": branch_class(schema_visible, dictionary_visible),
                "gold_prefix_action_kinds": preceding_kinds,
                "gold_prefix_schema_arguments": previous_schema_arguments,
                "gold_prefix_dictionary_arguments": previous_dictionary_arguments,
                "tool_result_ok": None,
                "error_category": "not_executed_teacher_forced",
                "error_identifiers": [],
                "next_action_kind": _gold_next_action(task, action),
                "exact_query_repeat": None,
                "later_successful_sql_after_error": None,
                "gold_kind": record.get("gold_kind"),
                "predicted_kind": record.get("predicted_kind"),
                "parameter_exact": record.get("parameter_exact"),
                "exact_match": record.get("exact_match"),
                "sql_mismatch_category": primary_category,
                "gold_query": gold_query,
                "predicted_query": predicted_query,
                "source_files": {
                    "prediction": record.get("source_file"),
                    "dataset": task.get("source_file"),
                    "source_trajectory": task.get("source_trajectory_path"),
                },
            }
        )
    return events


def _task_index(runtime_tasks: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(task.get("group"), task.get("task_id")): task for task in runtime_tasks}


def _preceding_runtime_branch(
    task: dict[str, Any], event: dict[str, Any]
) -> tuple[bool, bool, list[dict[str, Any]]]:
    event_step = event.get("step_index")
    accepted_schema = False
    accepted_dictionary = False
    accepted_actions: list[dict[str, Any]] = []
    for action in task.get("actions", []):
        step = action.get("step_index")
        if isinstance(event_step, int) and isinstance(step, int) and step >= event_step:
            continue
        if action.get("kind") in {"schema", "dictionary"}:
            accepted_actions.append(action)
            if action.get("kind") == "schema":
                accepted_schema = True
            if action.get("kind") == "dictionary":
                accepted_dictionary = True
    return accepted_schema, accepted_dictionary, accepted_actions


def _next_sql_event(events: list[dict[str, Any]], current_index: int) -> dict[str, Any] | None:
    for candidate in events:
        if candidate.get("sql_event_index") == current_index + 1:
            return candidate
    return None


def _termination_reason(task: dict[str, Any]) -> str:
    if task.get("budget_terminal"):
        return "budget_exhaustion"
    if task.get("parser_or_action_terminal"):
        return "parser_or_action_validation"
    return "other_termination"


def _terminal_reason_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    """Preserve the prior audit's parser/action/budget distinction."""

    counts = Counter()
    for task in tasks:
        reason = task.get("terminal_reason") or task.get("failure_reason")
        if reason == "invalid_json":
            counts["parser_failure"] += 1
        elif reason in {"invalid_action", "invalid_action_arguments"}:
            counts["action_validation_failure"] += 1
        elif reason in {"tool_budget_exhausted", "action_budget_exhausted"}:
            counts["budget_exhaustion"] += 1
        elif task.get("parser_or_action_terminal"):
            counts["parser_or_action_other"] += 1
        else:
            counts["other_termination"] += 1
    return {
        "parser_failure_task_count": counts["parser_failure"],
        "action_validation_failure_task_count": counts["action_validation_failure"],
        "parser_or_action_other_task_count": counts["parser_or_action_other"],
        "parser_or_action_validation_task_count": (
            counts["parser_failure"]
            + counts["action_validation_failure"]
            + counts["parser_or_action_other"]
        ),
        "budget_terminal_task_count": counts["budget_exhaustion"],
        "other_termination_task_count": counts["other_termination"],
    }


def _next_event_class(
    event: dict[str, Any],
    next_event: dict[str, Any] | None,
    task: dict[str, Any],
) -> str:
    next_action = event.get("next_action")
    if next_action == "sql":
        if next_event is not None and next_event.get("query") == event.get("query"):
            return "retry_same_sql"
        if next_event is not None:
            return "retry_modified_sql"
        return "other"
    if next_action == "schema":
        return "switch_to_schema"
    if next_action == "dictionary":
        return "switch_to_dictionary"
    if next_action == "check":
        return "switch_to_check"
    if next_action == "finish":
        return "switch_to_finish"
    if task.get("parser_or_action_terminal"):
        return "invalid_action_or_parser_failure"
    if next_action is None:
        return "terminate_without_next_action"
    return "other"


def build_runtime_sql_events(
    runtime_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _task_map = _task_index(runtime_tasks)
    events: list[dict[str, Any]] = []
    for task in runtime_tasks:
        raw_events = list(task.get("sql_events") or [])
        for raw_event in raw_events:
            schema, dictionary, accepted_actions = _preceding_runtime_branch(task, raw_event)
            next_event = _next_sql_event(raw_events, raw_event.get("sql_event_index", -1))
            error_category = classify_runtime_error_category(raw_event)
            is_execution_error = (
                raw_event.get("tool_result_ok") is False
                and error_category != "budget_or_terminal_sentinel"
            )
            later_success = bool(
                is_execution_error
                and any(
                    later.get("sql_event_index", -1) > raw_event.get("sql_event_index", -1)
                    and later.get("tool_result_ok") is True
                    for later in raw_events
                )
            )
            exact_repeat = bool(
                is_execution_error
                and next_event is not None
                and next_event.get("query") == raw_event.get("query")
            )
            event_record = {
                "history_source": "runtime_accepted_prefix",
                "task_id": raw_event.get("task_id", task.get("task_id")),
                "group": raw_event.get("group", task.get("group")),
                "group_label": raw_event.get("group_label", task.get("group_label")),
                "database_id": raw_event.get("database_id", task.get("database_id")),
                "template_id": raw_event.get("template_id", task.get("template_id")),
                "trajectory_source": raw_event.get(
                    "trajectory_source", task.get("trajectory_source")
                ),
                "sql_event_index": raw_event.get("sql_event_index"),
                "step_index": raw_event.get("step_index"),
                "query": raw_event.get("query"),
                "query_hash": raw_event.get("query_sha256"),
                "preceding_schema_accepted": schema,
                "preceding_dictionary_accepted": dictionary,
                "preceding_accepted_branch_class": branch_class(schema, dictionary),
                "preceding_accepted_schema_dictionary_actions": accepted_actions,
                "tool_result_ok": raw_event.get("tool_result_ok"),
                "tool_result": raw_event.get("tool_result"),
                "error": raw_event.get("error"),
                "error_category": error_category,
                "error_identifiers": _identifiers(raw_event.get("error_identifier")),
                "error_is_budget_or_terminal_sentinel": bool(
                    raw_event.get("error_is_budget_or_terminal_sentinel")
                ),
                "next_action_kind": raw_event.get("next_action"),
                "next_action_class": _next_event_class(raw_event, next_event, task)
                if raw_event.get("tool_result_ok") is False
                else None,
                "exact_query_repeat": exact_repeat if is_execution_error else None,
                "later_successful_sql_after_error": later_success if is_execution_error else None,
                "sql_again_after_error": bool(
                    is_execution_error and raw_event.get("next_action") == "sql"
                ),
                "modified_query_after_error": bool(
                    is_execution_error
                    and next_event is not None
                    and raw_event.get("next_action") == "sql"
                    and next_event.get("query") != raw_event.get("query")
                ),
                "sql_mismatch_category": _gold_mismatch_category(raw_event),
                "gold_comparison": raw_event.get("gold_comparison"),
                "source_files": raw_event.get("source_files") or task.get("source_files"),
            }
            events.append(event_record)
    return events


def _empty_runtime_stratum() -> dict[str, Any]:
    return {
        "task_count": 0,
        "sql_event_count": 0,
        "all_saved_error_event_count": 0,
        "execution_error_event_count": 0,
        "all_saved_error_rate": 0.0,
        "execution_error_rate": 0.0,
        "failed_task_count": 0,
        "all_error_task_count": 0,
        "exact_query_repeat_count": 0,
        "exact_query_repeat_rate_among_execution_errors": 0.0,
        "sql_again_after_error_event_count": 0,
        "modified_query_after_error_event_count": 0,
        "modified_but_failed_query_event_count": 0,
        "successful_sql_after_prior_sql_error_event_count": 0,
        "successful_sql_after_prior_sql_error_task_count": 0,
        "switch_to_check_event_count": 0,
        "switch_to_finish_event_count": 0,
        "invalid_action_or_parser_failure_event_count": 0,
        "terminate_without_next_action_event_count": 0,
        "other_event_count": 0,
        "budget_terminal_task_count": 0,
        "parser_or_action_terminal_task_count": 0,
        "parser_failure_task_count": 0,
        "action_validation_failure_task_count": 0,
        "parser_or_action_other_task_count": 0,
        "parser_or_action_validation_task_count": 0,
        "other_termination_task_count": 0,
        "error_category_counts": {},
        "sql_mismatch_category_counts": {},
        "next_action_class_counts": {},
    }


def aggregate_runtime_strata(
    events: list[dict[str, Any]],
    runtime_tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    task_map = _task_index(runtime_tasks)
    result: dict[str, dict[str, Any]] = {}
    for branch in BRANCH_CLASSES:
        selected = [
            event for event in events if event.get("preceding_accepted_branch_class") == branch
        ]
        stats = _empty_runtime_stratum()
        task_keys = {(event.get("group"), event.get("task_id")) for event in selected}
        stats["task_count"] = len(task_keys)
        stats["sql_event_count"] = len(selected)
        error_events = [event for event in selected if event.get("tool_result_ok") is False]
        execution_errors = [
            event
            for event in error_events
            if event.get("error_category") != "budget_or_terminal_sentinel"
        ]
        stats["all_saved_error_event_count"] = len(error_events)
        stats["execution_error_event_count"] = len(execution_errors)
        if selected:
            stats["all_saved_error_rate"] = len(error_events) / len(selected)
            stats["execution_error_rate"] = len(execution_errors) / len(selected)
        failed_tasks = {(event.get("group"), event.get("task_id")) for event in execution_errors}
        all_error_tasks = {(event.get("group"), event.get("task_id")) for event in error_events}
        stats["failed_task_count"] = len(failed_tasks)
        stats["all_error_task_count"] = len(all_error_tasks)
        repeated = [event for event in execution_errors if event.get("exact_query_repeat")]
        stats["exact_query_repeat_count"] = len(repeated)
        if execution_errors:
            stats["exact_query_repeat_rate_among_execution_errors"] = len(repeated) / len(
                execution_errors
            )
        stats["sql_again_after_error_event_count"] = sum(
            bool(event.get("sql_again_after_error")) for event in execution_errors
        )
        modified = [event for event in execution_errors if event.get("modified_query_after_error")]
        stats["modified_query_after_error_event_count"] = len(modified)
        stats["modified_but_failed_query_event_count"] = sum(
            not event.get("later_successful_sql_after_error") for event in modified
        )
        recovered = [
            event for event in execution_errors if event.get("later_successful_sql_after_error")
        ]
        stats["successful_sql_after_prior_sql_error_event_count"] = len(recovered)
        stats["successful_sql_after_prior_sql_error_task_count"] = len(
            {(event.get("group"), event.get("task_id")) for event in recovered}
        )
        next_classes = Counter(event.get("next_action_class") for event in error_events)
        next_classes.pop(None, None)
        stats["next_action_class_counts"] = _counter_dict(next_classes)
        stats["switch_to_check_event_count"] = next_classes.get("switch_to_check", 0)
        stats["switch_to_finish_event_count"] = next_classes.get("switch_to_finish", 0)
        stats["invalid_action_or_parser_failure_event_count"] = next_classes.get(
            "invalid_action_or_parser_failure", 0
        )
        stats["terminate_without_next_action_event_count"] = next_classes.get(
            "terminate_without_next_action", 0
        )
        stats["other_event_count"] = next_classes.get("other", 0)
        error_categories = Counter(event.get("error_category") for event in error_events)
        stats["error_category_counts"] = _counter_dict(error_categories)
        mismatch_categories = Counter(event.get("sql_mismatch_category") for event in selected)
        stats["sql_mismatch_category_counts"] = _counter_dict(mismatch_categories)
        terminal_tasks = [task_map[key] for key in task_keys if key in task_map]
        stats.update(_terminal_reason_counts(terminal_tasks))
        result[branch] = stats
    return result


def aggregate_teacher_strata(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for branch in BRANCH_CLASSES:
        selected = [event for event in events if event.get("gold_prefix_branch_class") == branch]
        task_ids = {event.get("task_id") for event in selected}
        mismatched = [
            event
            for event in selected
            if event.get("sql_mismatch_category") != "exact_or_no_mismatch"
        ]
        mismatch_tasks = {event.get("task_id") for event in mismatched}
        result[branch] = {
            "task_count": len(task_ids),
            "sql_target_count": len(selected),
            "independent_mismatch_task_count": len(mismatch_tasks),
            "mismatch_event_count": len(mismatched),
            "mismatch_rate": len(mismatched) / len(selected) if selected else 0.0,
            "parameter_exact_count": sum(bool(event.get("parameter_exact")) for event in selected),
            "kind_match_count": sum(
                event.get("gold_kind") == event.get("predicted_kind") for event in selected
            ),
            "error_rate": None,
            "error_rate_definition": "not_applicable_teacher_forced_not_executed",
            "successful_sql_after_prior_sql_error_event_count": None,
            "exact_query_repeat_count": None,
            "sql_mismatch_category_counts": _counter_dict(
                Counter(event.get("sql_mismatch_category") for event in selected)
            ),
        }
    return result


def _action_kind_confusion(teacher_records: list[dict[str, Any]], gold_kind: str) -> dict[str, Any]:
    selected = [record for record in teacher_records if record.get("gold_kind") == gold_kind]
    confusion = Counter(
        f"{record.get('gold_kind')}→{record.get('predicted_kind')}" for record in selected
    )
    return {
        "target_count": len(selected),
        "gold_action_kind": gold_kind,
        "predicted_action_kind_counts": _counter_dict(
            Counter(record.get("predicted_kind") for record in selected)
        ),
        "confusion_matrix": _counter_dict(confusion),
        "kind_match_count": sum(
            record.get("gold_kind") == record.get("predicted_kind") for record in selected
        ),
        "parameter_exact_count": sum(bool(record.get("parameter_exact")) for record in selected),
        "by_trajectory_source": {
            source: {
                "target_count": sum(
                    record.get("trajectory_source") == source for record in selected
                ),
                "kind_match_count": sum(
                    record.get("trajectory_source") == source
                    and record.get("gold_kind") == record.get("predicted_kind")
                    for record in selected
                ),
            }
            for source in sorted({record.get("trajectory_source") for record in selected})
        },
    }


def _runtime_group_summary(
    group: str,
    events: list[dict[str, Any]],
    runtime_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [event for event in events if event.get("group") == group]
    strata = aggregate_runtime_strata(
        selected, [task for task in runtime_tasks if task.get("group") == group]
    )
    errors = [event for event in selected if event.get("tool_result_ok") is False]
    execution_errors = [
        event for event in errors if event.get("error_category") != "budget_or_terminal_sentinel"
    ]
    recovered = [
        event for event in execution_errors if event.get("later_successful_sql_after_error")
    ]
    terminal_tasks = [task for task in runtime_tasks if task.get("group") == group]
    return {
        "group": group,
        "group_label": GROUP_LABELS[group],
        "task_count": len({event.get("task_id") for event in selected}),
        "sql_event_count": len(selected),
        "sql_error_event_count": len(errors),
        "sql_execution_error_event_count": len(execution_errors),
        "successful_sql_after_prior_sql_error": {
            "event_count": len(recovered),
            "task_count": len({(event.get("group"), event.get("task_id")) for event in recovered}),
            "definition": (
                "A later accepted SQL Action in the same saved episode has tool_result.ok=true "
                "after this saved SQL error."
            ),
        },
        "error_to_next_action": _counter_dict(
            Counter(event.get("next_action_class") for event in errors)
        ),
        "same_query_retry_event_count": sum(
            event.get("next_action_class") == "retry_same_sql" for event in errors
        ),
        "modified_query_retry_event_count": sum(
            event.get("next_action_class") == "retry_modified_sql" for event in errors
        ),
        "switch_to_check_event_count": sum(
            event.get("next_action_class") == "switch_to_check" for event in errors
        ),
        "switch_to_finish_event_count": sum(
            event.get("next_action_class") == "switch_to_finish" for event in errors
        ),
        "terminal_task_counts": _terminal_reason_counts(terminal_tasks),
        "by_preceding_accepted_branch": strata,
    }


def _aggregate_runtime_by_dimension(
    events: list[dict[str, Any]],
    runtime_tasks: list[dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    values = sorted({event.get(dimension) for event in events if event.get(dimension) is not None})
    result = {}
    for value in values:
        selected = [event for event in events if event.get(dimension) == value]
        selected_tasks = [task for task in runtime_tasks if task.get(dimension) == value]
        result[str(value)] = {
            "by_preceding_accepted_branch": aggregate_runtime_strata(selected, selected_tasks)
        }
    return result


def _aggregate_teacher_by_dimension(events: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    values = sorted({event.get(dimension) for event in events if event.get(dimension) is not None})
    return {
        str(value): aggregate_teacher_strata(
            [event for event in events if event.get(dimension) == value]
        )
        for value in values
    }


def _dataset_summary(
    dataset_tasks: list[dict[str, Any]], teacher_events: list[dict[str, Any]]
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in teacher_events:
        by_task[event.get("task_id")].append(event)
    task_records = []
    for task in dataset_tasks:
        task_events = by_task.get(task.get("task_id"), [])
        task_records.append(
            {
                "task_id": task.get("task_id"),
                "database_id": task.get("database_id"),
                "template_id": task.get("template_id"),
                "trajectory_source": task.get("trajectory_source"),
                "gold_action_kind_sequence": task.get("action_kind_sequence"),
                "gold_sql_target_count": len(task.get("gold_sql_targets") or []),
                "gold_prefix_branch_class_counts": _counter_dict(
                    Counter(event.get("gold_prefix_branch_class") for event in task_events)
                ),
                "schema_positions": task.get("schema_positions"),
                "dictionary_positions": task.get("dictionary_positions"),
                "sql_positions": task.get("sql_positions"),
                "real_error_feedback": task.get("real_error_feedback"),
                "has_post_error_correction_sql_target": task.get(
                    "has_post_error_correction_sql_target"
                ),
                "anomalies": task.get("anomalies"),
                "source_files": {
                    "dataset": task.get("source_file"),
                    "source_trajectory": task.get("source_trajectory_path"),
                },
            }
        )
    observed = {
        "task_count": len(dataset_tasks),
        "assistant_target_count": sum(task.get("action_count", 0) for task in dataset_tasks),
        "sql_target_count": len(teacher_events),
    }
    expected = {"task_count": 20, "assistant_target_count": 108, "sql_target_count": 24}
    return {
        **observed,
        "expected_counts": expected,
        "count_differences": {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if expected[key] != observed[key]
        },
        "trajectory_source_counts": _counter_dict(
            Counter(task.get("trajectory_source") for task in dataset_tasks)
        ),
        "tasks": task_records,
        "source": "reused_from_prior_sql_parameter_supervision_audit",
    }


def _runtime_task_evidence(
    runtime_tasks: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_task[(event.get("group"), event.get("task_id"))].append(event)
    records = []
    for task in runtime_tasks:
        task_events = by_task.get((task.get("group"), task.get("task_id")), [])
        records.append(
            {
                "task_id": task.get("task_id"),
                "group": task.get("group"),
                "group_label": task.get("group_label"),
                "template_id": task.get("template_id"),
                "trajectory_source": task.get("trajectory_source"),
                "accepted_action_kind_sequence": task.get("action_kind_sequence"),
                "accepted_action_indices": [
                    action.get("step_index")
                    for action in task.get("actions", [])
                    if action.get("step_index") is not None
                ],
                "sql_event_count": len(task_events),
                "sql_event_indices": [event.get("sql_event_index") for event in task_events],
                "preceding_accepted_branch_classes": _sorted_unique(
                    event.get("preceding_accepted_branch_class") for event in task_events
                ),
                "execution_error_event_indices": [
                    event.get("sql_event_index")
                    for event in task_events
                    if event.get("tool_result_ok") is False
                    and event.get("error_category") != "budget_or_terminal_sentinel"
                ],
                "successful_sql_after_prior_sql_error_event_indices": [
                    event.get("sql_event_index")
                    for event in task_events
                    if event.get("later_successful_sql_after_error")
                ],
                "terminal_reason": task.get("terminal_reason"),
                "parser_or_action_terminal": task.get("parser_or_action_terminal"),
                "budget_terminal": task.get("budget_terminal"),
                "source_files": {
                    "result": task.get("result_file"),
                    "trajectory": task.get("trajectory_file"),
                    **(task.get("source_files") or {}),
                },
            }
        )
    return records


def _special_task_evidence(
    teacher_events: list[dict[str, Any]], runtime_events: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id in SPECIAL_TASKS:
        runtime_refs = [
            {
                "group": event.get("group"),
                "group_label": event.get("group_label"),
                "sql_event_index": event.get("sql_event_index"),
                "preceding_accepted_branch_class": event.get("preceding_accepted_branch_class"),
                "query_hash": event.get("query_hash"),
                "tool_result_ok": event.get("tool_result_ok"),
                "tool_result": event.get("tool_result"),
                "error_category": event.get("error_category"),
                "error_identifiers": event.get("error_identifiers"),
                "next_action_kind": event.get("next_action_kind"),
                "next_action_class": event.get("next_action_class"),
                "exact_query_repeat": event.get("exact_query_repeat"),
                "later_successful_sql_after_error": event.get("later_successful_sql_after_error"),
                "query": event.get("query"),
                "gold_comparison": event.get("gold_comparison"),
                "source_files": event.get("source_files"),
            }
            for event in runtime_events
            if event.get("task_id") == task_id
        ]
        result[task_id] = {
            "teacher_forced_sql": [
                event for event in teacher_events if event.get("task_id") == task_id
            ],
            "runtime_sql_event_refs": runtime_refs,
            "saved_probe_success_evidence": [
                ref for ref in runtime_refs if ref.get("tool_result_ok") is True
            ],
            "saved_identifier_error_evidence": [
                ref for ref in runtime_refs if ref.get("error_category") == "identifier_mismatch"
            ],
            "comparison_warning": (
                "Teacher-forced records and Runtime records are observationally aligned by "
                "task_id only; "
                "they are not counterfactual pairings."
            ),
        }
    return result


def _decision_findings(
    teacher_events: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
    runtime_by_group: dict[str, Any],
    teacher_strata: dict[str, Any],
    runtime_overall: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    confirmed: list[dict[str, Any]] = []
    likely: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []

    runtime_branch_task_counts = {
        branch: runtime_overall[branch]["task_count"] for branch in BRANCH_CLASSES
    }
    runtime_branch_event_counts = {
        branch: runtime_overall[branch]["sql_event_count"] for branch in BRANCH_CLASSES
    }
    accepted_dictionary_events = sum(
        event.get("preceding_dictionary_accepted") is True for event in runtime_events
    )
    accepted_schema_events = sum(
        event.get("preceding_schema_accepted") is True for event in runtime_events
    )
    confirmed.append(
        {
            "id": "runtime_branch_coverage_is_sparse",
            "statement": (
                "Across the saved five-group Runtime records, accepted Dictionary never "
                "precedes a SQL event; "
                "accepted Schema precedes SQL only in the schema_only strata of B/E top episodes."
            ),
            "evidence": {
                "accepted_schema_before_sql_event_count": accepted_schema_events,
                "accepted_dictionary_before_sql_event_count": accepted_dictionary_events,
                "branch_task_counts": runtime_branch_task_counts,
                "branch_event_counts": runtime_branch_event_counts,
            },
            "causal_status": "confirmed_observation_not_causal",
        }
    )
    all_recovery_zero = all(
        summary["successful_sql_after_prior_sql_error"]["event_count"] == 0
        for summary in runtime_by_group.values()
    )
    confirmed.append(
        {
            "id": "strict_recovery_is_zero",
            "statement": (
                "The strict successful_sql_after_prior_sql_error metric is zero in every group; "
                "there is no saved episode with a later accepted SQL Action whose "
                "tool_result.ok=true after a SQL error."
            ),
            "evidence": {
                group: summary["successful_sql_after_prior_sql_error"]
                for group, summary in runtime_by_group.items()
            },
            "causal_status": "confirmed_observation_not_business_correctness",
        }
    )
    dictionary_confusion = _action_kind_confusion(
        [
            {
                "gold_kind": event.get("gold_kind"),
                "predicted_kind": event.get("predicted_kind"),
                "parameter_exact": event.get("parameter_exact"),
                "trajectory_source": event.get("trajectory_source"),
            }
            for event in teacher_events
        ],
        "dictionary",
    )
    confirmed.append(
        {
            "id": "teacher_dictionary_branch_confusion",
            "statement": (
                "Teacher-forced Dictionary targets are not learned as a stable branch: "
                "the saved E3 records contain kind substitutions to Schema and SQL, plus "
                "Dictionary parameter loss."
            ),
            "evidence": dictionary_confusion,
            "causal_status": "confirmed_teacher_forced_observation",
        }
    )

    branch_mismatch_presence = {
        branch: bool(
            runtime_overall[branch]["error_category_counts"].get("identifier_mismatch", 0)
            or runtime_overall[branch]["sql_mismatch_category_counts"].get("identifier_mismatch", 0)
        )
        for branch in BRANCH_CLASSES
    }
    if branch_mismatch_presence.get("neither") and branch_mismatch_presence.get("schema_only"):
        likely.append(
            {
                "id": "sql_mapping_signal_across_coverage",
                "statement": (
                    "Identifier-related SQL drift is observed in both the uncovered Runtime "
                    "stratum and the "
                    "schema_only stratum; under the pre-registered rule this prioritizes SQL "
                    "parameter/database-structure "
                    "mapping as a hypothesis, but the branch imbalance prevents a causal claim."
                ),
                "evidence": {
                    "branch_identifier_signal": {
                        branch: {
                            "tool_error_identifier_mismatch": runtime_overall[branch][
                                "error_category_counts"
                            ].get("identifier_mismatch", 0),
                            "gold_comparison_identifier_mismatch": runtime_overall[branch][
                                "sql_mismatch_category_counts"
                            ].get("identifier_mismatch", 0),
                        }
                        for branch in BRANCH_CLASSES
                    },
                    "teacher_sql_mismatch_by_gold_prefix": teacher_strata,
                },
                "causal_status": "likely_pattern_only",
            }
        )
    repeated_any = sum(
        runtime_overall[branch]["exact_query_repeat_count"] for branch in BRANCH_CLASSES
    )
    if repeated_any and all_recovery_zero:
        likely.append(
            {
                "id": "recovery_failure_signal",
                "statement": (
                    "After SQL errors, saved Runtime episodes contain exact query repeats "
                    "while strict later-success "
                    "recovery remains zero; this supports prioritizing error-feedback recovery "
                    "behavior for validation."
                ),
                "evidence": {
                    "exact_query_repeat_event_count": repeated_any,
                    "strict_recovery_zero_all_groups": all_recovery_zero,
                    "by_group": {
                        group: {
                            "same_query_retry_event_count": summary["same_query_retry_event_count"],
                            "strict_recovery": summary["successful_sql_after_prior_sql_error"],
                        }
                        for group, summary in runtime_by_group.items()
                    },
                },
                "causal_status": "likely_pattern_only",
            }
        )
    teacher_branches_with_mismatch = [
        branch for branch, stats in teacher_strata.items() if stats["mismatch_event_count"] > 0
    ]
    if len(teacher_branches_with_mismatch) >= 2:
        likely.append(
            {
                "id": "teacher_parameter_drift_not_confined_to_one_prefix",
                "statement": (
                    "Teacher-forced SQL mismatch is present under multiple gold-prefix "
                    "evidence strata, "
                    "so a single branch-selection explanation is insufficient for the saved "
                    "predictions."
                ),
                "evidence": {
                    "strata_with_mismatch": teacher_branches_with_mismatch,
                    "teacher_strata": teacher_strata,
                },
                "causal_status": "likely_pattern_only",
            }
        )

    unconfirmed.extend(
        [
            {
                "id": "schema_effect_on_sql_correctness",
                "statement": (
                    "The saved data cannot establish that accepting Schema reduces field "
                    "errors: only two tasks in B/E "
                    "have the accepted-Schema path, no Runtime Dictionary path is present, and "
                    "there is no matched "
                    "counterfactual with identical generation conditions."
                ),
                "evidence": {
                    "schema_only_runtime_task_counts": {
                        group: runtime_by_group[group]["by_preceding_accepted_branch"][
                            "schema_only"
                        ]["task_count"]
                        for group in runtime_by_group
                    }
                },
                "causal_status": "unconfirmed",
            },
            {
                "id": "business_answer_correctness",
                "statement": (
                    "A successful SQL tool result is not evidence of a correct business "
                    "answer; the saved records do not "
                    "provide a separate answer-evaluation success for these SQL Actions."
                ),
                "evidence": {"strict_metric_scope": "tool_result.ok only"},
                "causal_status": "unconfirmed",
            },
            {
                "id": "unique_root_cause",
                "statement": (
                    "The branch-selection, SQL-parameter mapping, and recovery mechanisms are "
                    "not causally separable "
                    "from these observational train-only trajectories."
                ),
                "evidence": {
                    "runtime_branch_task_counts": runtime_branch_task_counts,
                    "teacher_prefix_strata": teacher_strata,
                },
                "causal_status": "unconfirmed",
            },
        ]
    )
    return confirmed, likely, unconfirmed


def _recommendation(runtime_overall: dict[str, Any], special: dict[str, Any]) -> dict[str, Any]:
    return {
        "priority": 1,
        "hypothesis": (
            "A saved SQL error is not converted into a new evidence-conditioned SQL decision: "
            "the next action often repeats or modifies SQL without a strict later successful SQL."
        ),
        "why_this_first": (
            "It is directly measurable in the existing episodes, has zero strict recoveries in "
            "all five groups, "
            "and can be distinguished from branch selection without assuming that a successful "
            "SQL execution is a "
            "correct business answer. The top tasks provide a concrete Schema → probe-success "
            "→ main-column-error "
            "trace, followed by repeated main queries."
        ),
        "supporting_task_ids": list(SPECIAL_TASKS),
        "supporting_observations": {
            "strict_recovery_by_branch": {
                branch: runtime_overall[branch]["successful_sql_after_prior_sql_error_event_count"]
                for branch in BRANCH_CLASSES
            },
            "same_query_retry_by_branch": {
                branch: runtime_overall[branch]["exact_query_repeat_count"]
                for branch in BRANCH_CLASSES
            },
            "top_task_trace_scope": "B/E only; A/C/D have no accepted SQL for the top tasks",
        },
        "minimal_experiment": {
            "design": (
                "Offline replay of the already saved event table with a deterministic decision "
                "label after each SQL "
                "error: same query, modified query, Schema, Dictionary, Check, Finish, or "
                "termination. Compare the "
                "strict later-success metric after conditioning on the preceding branch class "
                "and error category; "
                "do not execute SQL. If a new controlled artifact is later authorized, use "
                "matched synthetic records "
                "with identical pre-error history and vary only the saved error-feedback "
                "observation."
            ),
            "cpu_only": True,
            "requires_training": False,
            "requires_gpu": False,
            "requires_production_code_change": False,
            "current_turn_executed": False,
        },
        "competing_hypotheses": {
            "branch_selection": (
                "Need a balanced accepted-Schema/Dictionary comparison; current Runtime data lacks "
                "it."
            ),
            "sql_parameter_mapping": (
                "Use identifier mismatch and gold-prefix strata, but avoid treating branch strata "
                "as randomized."
            ),
            "recovery": (
                "Use strict later-success and exact-repeat events; this is the most directly "
                "testable current signal."
            ),
        },
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(report: dict[str, Any]) -> str:
    dataset = report["dataset_target_audit"]
    teacher = report["action_branch_confusion"]["teacher_forced"]
    _runtime = report["action_branch_confusion"]["runtime"]
    top_probe_summaries = []
    top_error_summaries = []
    for task_id in ("train-0-top", "train-1-top"):
        evidence = report["task_level_evidence"]["special_tasks"][task_id]
        for ref in evidence["saved_probe_success_evidence"]:
            tool_result = ref.get("tool_result") or {}
            if isinstance(tool_result, dict):
                top_probe_summaries.append(
                    f"{task_id}/{ref['group_label']} event {ref['sql_event_index']}: "
                    f"ok=true, columns={tool_result.get('columns')}, "
                    f"row_count={tool_result.get('row_count')}, "
                    f"rows={tool_result.get('rows', tool_result.get('rows_preview'))}"
                )
        for ref in evidence["saved_identifier_error_evidence"][:1]:
            shape = ((ref.get("gold_comparison") or {}).get("classification") or {}).get(
                "shape", {}
            )
            top_error_summaries.append(
                f"{task_id}/{ref['group_label']} event {ref['sql_event_index']}: "
                f"error_identifiers={ref.get('error_identifiers')}, "
                f"gold_removed_qualified_identifiers={shape.get('removed_qualified_identifiers')}, "
                f"gold_tables={shape.get('gold_tables')}, "
                f"predicted_tables={shape.get('predicted_tables')}"
            )
    lines = [
        "# SFT SQL Branch Attribution",
        "",
        "状态：`SFT_SQL_BRANCH_ATTRIBUTION_COMPLETE`",
        "",
        (
            "本报告只重用了上一轮 train-only `sql-parameter-supervision-audit.json` 中已物化的 "
            "dataset、E3 "
            "prediction 和 Runtime event records。未加载模型/tokenizer，未访问 "
            "validation/test，未执行 SQL 或 "
            "Verifier，也未重跑任务。"
        ),
        "",
        "## 结论摘要",
        "",
        (
            "严格恢复指标 `successful_sql_after_prior_sql_error` 在 A/B/C/D/E 均为 0。Runtime "
            "中 accepted "
            "Dictionary 在任何 SQL 之前均未出现；accepted Schema 只出现在 B/E 的两个 top "
            "task，且该路径仍出现主 SQL "
            "字段错误和后续重复。由此可以确认恢复失败信号，并观察到 SQL 参数映射信号；不能把 "
            "Schema/Dictionary 分支与 SQL 参数错误作因果区分。"
        ),
        "",
        (
            "下一步优先验证：SQL error observation 是否被转化为新的、证据条件化的 SQL "
            "决策。该假设不需要 "
            "GPU、训练或生产代码修改，当前只给出实验设计，没有执行新实验。"
        ),
        "",
        "## 数据边界与样本核验",
        "",
        (
            f"dataset 重建摘要：{dataset['task_count']} 个 "
            f"task、{dataset['assistant_target_count']} "
            f"个 "
            f"assistant target、{dataset['sql_target_count']} 个 gold SQL target；数据来源分布为 "
            f"`{dataset['trajectory_source_counts']}`。Teacher-forced SQL 与 Runtime SQL "
            f"分开保存，gold "
            f"prefix evidence 不等同于 Runtime accepted action。"
        ),
        "",
        "## Teacher-forced：按 gold prefix evidence 分层",
        "",
        _markdown_table(
            [
                "gold prefix",
                "task",
                "SQL target",
                "mismatch event",
                "mismatch rate",
                "parameter exact",
                "error rate",
            ],
            [
                [
                    branch,
                    stats["task_count"],
                    stats["sql_target_count"],
                    stats["mismatch_event_count"],
                    f"{stats['mismatch_rate']:.3f}",
                    stats["parameter_exact_count"],
                    "N/A (not executed)",
                ]
                for branch, stats in teacher["by_gold_prefix_branch"].items()
            ],
        ),
        "",
        (
            "Teacher-forced 的 mismatch 类别是预测 SQL 与 gold SQL "
            "的字符串/结构比较，不是工具错误率；因此 error "
            "rate、query "
            "repeat 和后续成功 SQL 在此层级均为 N/A。"
        ),
        "",
        "## Runtime：按 preceding accepted branch 分层",
        "",
        _markdown_table(
            [
                "group",
                "branch",
                "task",
                "SQL event",
                "execution error",
                "error rate",
                "same-query repeat",
                "strict recovery",
            ],
            [
                [
                    summary["group_label"],
                    branch,
                    stats["task_count"],
                    stats["sql_event_count"],
                    stats["execution_error_event_count"],
                    f"{stats['execution_error_rate']:.3f}",
                    stats["exact_query_repeat_count"],
                    stats["successful_sql_after_prior_sql_error_event_count"],
                ]
                for summary in report["error_to_next_action"]["by_group"].values()
                for branch, stats in summary["by_preceding_accepted_branch"].items()
                if stats["sql_event_count"]
            ],
        ),
        "",
        (
            "五组 Runtime 的事件级统一证据表保存在机器可读报告的 "
            "`sql_event_evidence.runtime`。每条记录包含 query "
            "hash、preceding Schema/Dictionary accepted 状态、tool result、error "
            "category/identifier、下一 "
            "Action、精确重复和严格后续成功字段，并保留 "
            "result/trajectory/request/audit/raw-generation 路径。"
        ),
        "",
        "## Error 后决策转移",
        "",
        _markdown_table(
            [
                "group",
                "SQL errors",
                "retry same",
                "retry modified",
                "switch Check",
                "switch Finish",
                "strict recovery",
                "terminal",
            ],
            [
                [
                    summary["group_label"],
                    summary["sql_execution_error_event_count"],
                    summary["same_query_retry_event_count"],
                    summary["modified_query_retry_event_count"],
                    summary["switch_to_check_event_count"],
                    summary["switch_to_finish_event_count"],
                    summary["successful_sql_after_prior_sql_error"]["event_count"],
                    summary["terminal_task_counts"],
                ]
                for summary in report["error_to_next_action"]["by_group"].values()
            ],
        ),
        "",
        (
            "`tool_result.ok=true` "
            "只证明工具执行返回成功，不能直接证明业务答案正确；本审计不把二者合并。不同 query "
            "也不自动算修正成功，只有同一 "
            "episode 后续出现成功 SQL 才计入 strict recovery。"
        ),
        "",
        "## 六个重点 task",
        "",
        _markdown_table(
            [
                "task",
                "teacher prefix strata",
                "Runtime groups with SQL",
                "accepted branch",
                "strict recovery events",
            ],
            [
                [
                    task_id,
                    ", ".join(
                        f"{branch}:"
                        + str(
                            len(
                                [
                                    event
                                    for event in evidence["teacher_forced_sql"]
                                    if event.get("gold_prefix_branch_class") == branch
                                ]
                            )
                        )
                        for branch in BRANCH_CLASSES
                        if any(
                            e.get("gold_prefix_branch_class") == branch
                            for e in evidence["teacher_forced_sql"]
                        )
                    )
                    or "none",
                    ", ".join(
                        sorted({ref["group_label"] for ref in evidence["runtime_sql_event_refs"]})
                    )
                    or "none",
                    ", ".join(
                        sorted(
                            {
                                ref["preceding_accepted_branch_class"]
                                for ref in evidence["runtime_sql_event_refs"]
                            }
                        )
                    )
                    or "none",
                    sum(
                        bool(ref["later_successful_sql_after_error"])
                        for ref in evidence["runtime_sql_event_refs"]
                    ),
                ]
                for task_id, evidence in report["task_level_evidence"]["special_tasks"].items()
            ],
        ),
        "",
        (
            "B/E 的 `train-0-top` 与 `train-1-top` 保存了 Schema → probe SQL 成功 → 主 SQL "
            "`o.list_price_cents` 字段错误 → 后续重复/预算终止的路径；A/C/D 对这些 top task "
            "没有对应 accepted "
            "SQL，因此不能进行 "
            "A/B 或 C/E 的逐条 accepted-SQL 反事实配对。probe 成功和主 SQL 错误均只按保存的 "
            "tool result 报告，不重新执行查询。"
        ),
        "",
        "probe SQL 成功的直接保存证据：" + ("；".join(top_probe_summaries) or "unconfirmed") + "。",
        "主 SQL 与 gold 的可观察字段差异："
        + ("；".join(top_error_summaries) or "unconfirmed")
        + "。这些是保存的字符串比较和工具错误标识，不是重新执行后的业务语义判断。",
        "",
        "## 归因边界",
        "",
    ]
    for finding in report["confirmed_findings"]:
        lines.append(f"- Confirmed `{finding['id']}`：{finding['statement']}")
    lines.append("")
    for finding in report["likely_findings"]:
        lines.append(f"- Likely `{finding['id']}`：{finding['statement']}")
    lines.append("")
    for finding in report["unconfirmed_findings"]:
        lines.append(f"- Unconfirmed `{finding['id']}`：{finding['statement']}")
    lines.extend(
        [
            "",
            (
                "预先定义的判别逻辑已按观察性结果应用：覆盖前后都出现标识符漂移时优先调查 SQL "
                "参数映射；漂移主要集中于无 evidence 时才优先调查分支学习；首条 "
                "SQL "
                "差异较小但 error 后重复时优先调查恢复策略。本数据同时出现后两类信号，且 "
                "Schema/Dictionary "
                "分支严重不平衡，因此不强行选择唯一根因。上述规则只产生优先级，不产生因果证明。"
            ),
            "",
            "## 下一步最小实验",
            "",
            report["next_experiment_recommendation"]["hypothesis"],
            "",
            report["next_experiment_recommendation"]["why_this_first"],
            "",
            (
                "建议的最小设计：只对已保存事件表做 CPU-only 离线重标注/分层，条件化 preceding "
                "branch 与 error category，比较 "
                "same-query、modified-query、Check/Finish/termination 和 strict "
                "later-success；若未来获得新实验授权，再构造只改变 error-feedback observation "
                "的 matched "
                "synthetic "
                "records。当前不需要训练、GPU 或生产代码变更。"
            ),
            "",
            "## 可复核索引",
            "",
            f"机器可读结果：`{OUTPUT_PATH.relative_to(REPO_ROOT)}`。输入审计：`"
            f"{AUDIT_PATH.relative_to(REPO_ROOT)}`。完整 source file 列表位于 JSON 的 event/tas"
            f"k records 中。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("status") != "SFT_SQL_PARAMETER_SUPERVISION_AUDIT_COMPLETE":
        raise ValueError("prior audit is not complete")
    _scope = audit.get("scope") or {}
    dataset_tasks = audit["dataset_target_audit"]["tasks"]
    teacher_records = audit["action_branch_confusion"]["teacher_forced"]["sql"]["records"]
    runtime_tasks = audit["task_level_evidence"]["runtime_tasks"]
    teacher_events = build_teacher_sql_events(dataset_tasks, teacher_records)
    runtime_events = build_runtime_sql_events(runtime_tasks)
    teacher_strata = aggregate_teacher_strata(teacher_events)
    runtime_overall = aggregate_runtime_strata(runtime_events, runtime_tasks)
    runtime_by_group = {
        group: _runtime_group_summary(group, runtime_events, runtime_tasks) for group in GROUPS
    }
    confirmed, likely, unconfirmed = _decision_findings(
        teacher_events,
        runtime_events,
        runtime_by_group,
        teacher_strata,
        runtime_overall,
    )
    special = _special_task_evidence(teacher_events, runtime_events)
    report: dict[str, Any] = {
        "schema_version": "sft-sql-branch-attribution-v1",
        "status": "SFT_SQL_BRANCH_ATTRIBUTION_COMPLETE",
        "scope": {
            "split": "train",
            "cpu_only": True,
            "raw_artifacts_reread": False,
            "source": "prior_train_only_machine_readable_audit",
            "validation_test_payload_accessed": False,
            "database_accessed": False,
            "sql_executed": False,
            "verifier_called": False,
            "model_or_tokenizer_loaded": False,
        },
        "input_manifest": {
            "prior_audit": str(AUDIT_PATH.relative_to(REPO_ROOT)),
            "prior_audit_status": audit.get("status"),
            "embedded_source_files_reused": True,
            "raw_input_files_not_reread": [
                "artifacts/sft-data-train-teacher-v2/dataset.jsonl",
                "artifacts/sft-behavior-attribution/20261017T192604-sft-behavior-attribution/epoch-3/predictions.jsonl",
                "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation/*",
            ],
            "provenance_note": (
                "All event source paths are copied from the prior audit; this script does not "
                "open those raw files."
            ),
        },
        "dataset_target_audit": _dataset_summary(dataset_tasks, teacher_events),
        "action_branch_confusion": {
            "teacher_forced": {
                "history_definition": (
                    "branch strata use visible Schema evidence and preceding Dictionary evidence "
                    "in the gold prefix; they are not Runtime accepted actions"
                ),
                "by_gold_prefix_branch": teacher_strata,
                "by_trajectory_source": _aggregate_teacher_by_dimension(
                    teacher_events, "trajectory_source"
                ),
                "by_template": _aggregate_teacher_by_dimension(teacher_events, "template_id"),
                "dictionary_action_confusion": _action_kind_confusion(
                    teacher_records, "dictionary"
                ),
                "schema_action_confusion": _action_kind_confusion(teacher_records, "schema"),
            },
            "runtime": {
                "history_definition": (
                    "branch strata use accepted Schema/Dictionary actions before the saved SQL "
                    "event in the same Runtime episode"
                ),
                "by_preceding_accepted_branch": runtime_overall,
                "by_group": runtime_by_group,
                "by_trajectory_source": _aggregate_runtime_by_dimension(
                    runtime_events, runtime_tasks, "trajectory_source"
                ),
                "by_template": _aggregate_runtime_by_dimension(
                    runtime_events, runtime_tasks, "template_id"
                ),
            },
        },
        "sql_argument_mismatches": {
            "teacher_forced_by_gold_prefix_branch": {
                branch: stats["sql_mismatch_category_counts"]
                for branch, stats in teacher_strata.items()
            },
            "runtime_tool_error_by_preceding_accepted_branch": {
                branch: stats["error_category_counts"] for branch, stats in runtime_overall.items()
            },
            "runtime_gold_comparison_by_preceding_accepted_branch": {
                branch: stats["sql_mismatch_category_counts"]
                for branch, stats in runtime_overall.items()
            },
            "classification_note": (
                "Runtime tool error categories use saved error text; gold-comparison "
                "categories use saved SQL strings and prior audit metadata."
            ),
        },
        "error_to_next_action": {
            "definition": {
                ("successful_sql_after_prior_sql_error"): (
                    "same episode later SQL event has tool_result.ok=true after a non-sentinel SQL "
                    "error"
                ),
                "retry_same_sql": "next SQL query string exactly equals the errored query",
                "retry_modified_sql": (
                    "next SQL query exists and differs; this alone is not success"
                ),
            },
            "by_group": runtime_by_group,
            "by_preceding_accepted_branch": runtime_overall,
        },
        "sql_event_evidence": {
            "teacher_forced": teacher_events,
            "runtime": runtime_events,
            "field_note": (
                "Teacher records have gold-prefix visibility fields and null Runtime accepted "
                "fields; Runtime records have accepted-prefix fields and strict tool-result "
                "recovery fields."
            ),
        },
        "task_level_evidence": {
            "runtime_tasks": _runtime_task_evidence(runtime_tasks, runtime_events),
            "special_tasks": special,
            "special_task_ids": list(SPECIAL_TASKS),
            "cross_source_pairing_warning": (
                "Teacher-forced and Runtime records are not paired as counterfactuals merely "
                "because task_id matches."
            ),
        },
        "confirmed_findings": confirmed,
        "likely_findings": likely,
        "unconfirmed_findings": unconfirmed,
        "next_experiment_recommendation": _recommendation(runtime_overall, special),
        "verification": {
            "runtime_task_record_count": len(runtime_tasks),
            "runtime_sql_event_record_count": len(runtime_events),
            "teacher_forced_sql_event_record_count": len(teacher_events),
            "expected_groups_present": [group for group in GROUPS if group in runtime_by_group],
            "special_task_ids_present_in_teacher": sorted(
                task_id
                for task_id in SPECIAL_TASKS
                if any(event.get("task_id") == task_id for event in teacher_events)
            ),
            "special_task_ids_present_in_runtime": sorted(
                task_id
                for task_id in SPECIAL_TASKS
                if any(event.get("task_id") == task_id for event in runtime_events)
            ),
            "no_validation_test_access": True,
            "no_sql_execution": True,
            "no_model_execution": True,
        },
    }
    return report


def _write_outputs(report: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=AUDIT_PATH,
        help="Only the existing prior audit JSON is allowed as input.",
    )
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    if audit_path != AUDIT_PATH.resolve():
        raise SystemExit(f"refusing non-whitelisted input: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report = build_report(audit)
    _write_outputs(report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_json": str(OUTPUT_PATH),
                "output_markdown": str(REPORT_PATH),
                "runtime_sql_events": report["verification"]["runtime_sql_event_record_count"],
                "teacher_sql_events": report["verification"][
                    "teacher_forced_sql_event_record_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
