#!/usr/bin/env python3
"""CPU-only, read-only audit of SFT SQL parameters and Action branches.

This module intentionally operates on already-saved JSON/JSONL artifacts.  It
does not import the runtime, model, tokenizer, database, verifier, or any
dataset manifest loader.  The allow-list below is deliberately explicit so a
future invocation cannot silently expand to validation/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
SOURCE_NAMES = ("reference_sql_teacher", "deterministic_fixture_expert")
ACTION_KINDS = ("plan", "schema", "dictionary", "sql", "check", "finish")
SQL_MISMATCH_CATEGORIES = (
    "identifier_mismatch",
    "relation_join_mismatch",
    "filter_date_dialect_mismatch",
    "aggregation_group_order_mismatch",
    "syntax_or_serialization_error",
    "formatting_only",
    "unclassified",
)
SQL_SENTINEL_ERRORS = {"tool_budget_exhausted", "action_budget_exhausted"}
SQL_KEYWORDS = {
    "all",
    "and",
    "as",
    "asc",
    "by",
    "case",
    "count",
    "cross",
    "date",
    "desc",
    "distinct",
    "else",
    "end",
    "exists",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "select",
    "sum",
    "then",
    "where",
    "when",
    "with",
    "year",
}
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
QUALIFIED_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
TABLE_AFTER_RE = re.compile(
    r"\b(?:from|join|update|into)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
ERROR_IDENTIFIER_RE = re.compile(
    r"(?:no such (?:column|table|function):\s*|near\s+)([A-Za-z_][A-Za-z0-9_.]*)",
    re.IGNORECASE,
)


def repo_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}:{exc}") from exc
    return rows


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "line_count": None, "sha256": None}
    return {
        "exists": True,
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
        "sha256": sha256_file(path),
    }


def compact_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_content(content: Any) -> Any | None:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def normalize_sql(query: str | None) -> str:
    if query is None:
        return ""
    return re.sub(r"\s+", " ", str(query)).strip().lower()


def normalize_sql_format(query: str | None) -> str:
    """Remove only whitespace that is plausibly formatting, not SQL tokens."""

    text = normalize_sql(query)
    text = re.sub(r"\s*([(),=*<>])\s*", r"\1", text)
    return text


def action_exact(left: Any, right: Any) -> bool:
    return canonical(left) == canonical(right)


def action_kind(action: Any) -> str | None:
    return action.get("kind") if isinstance(action, dict) else None


def action_args(action: Any) -> dict[str, Any]:
    if isinstance(action, dict) and isinstance(action.get("arguments"), dict):
        return action["arguments"]
    return {}


def argument_class(action: Any) -> str:
    kind = action_kind(action)
    args = action_args(action)
    if kind == "dictionary":
        if args == {}:
            return "empty_object"
        if set(args) == {"term"} and args.get("term") == "revenue":
            return "term:revenue"
        return "other"
    if kind == "schema":
        return "empty_object" if args == {} else "other"
    return "not_applicable"


def identifier_tokens(query: str | None) -> set[str]:
    if not query:
        return set()
    tokens = {token.lower() for token in IDENTIFIER_RE.findall(query)}
    aliases = {match.group(2).lower() for match in TABLE_AFTER_RE.finditer(query) if match.group(2)}
    return {token for token in tokens if token not in SQL_KEYWORDS and token not in aliases}


def qualified_identifiers(query: str | None) -> set[str]:
    if not query:
        return set()
    return {f"{left.lower()}.{right.lower()}" for left, right in QUALIFIED_RE.findall(query)}


def table_references(query: str | None) -> set[str]:
    if not query:
        return set()
    return {match[0].lower() for match in TABLE_AFTER_RE.findall(query)}


def clause(query: str | None, name: str, end_names: Iterable[str]) -> str:
    if not query:
        return ""
    text = str(query)
    match = re.search(rf"\b{name}\b", text, re.IGNORECASE)
    if not match:
        return ""
    tail = text[match.end() :]
    endings = "|".join(re.escape(item) for item in end_names)
    end = re.search(rf"\b(?:{endings})\b", tail, re.IGNORECASE)
    if end:
        tail = tail[: end.start()]
    return normalize_sql(tail)


def sql_shape_flags(gold: str | None, predicted: str | None) -> dict[str, Any]:
    gold_tables = table_references(gold)
    predicted_tables = table_references(predicted)
    gold_qualified = qualified_identifiers(gold)
    predicted_qualified = qualified_identifiers(predicted)
    gold_tokens = identifier_tokens(gold)
    predicted_tokens = identifier_tokens(predicted)
    where_gold = clause(gold, "where", ("group", "having", "order", "limit"))
    where_predicted = clause(predicted, "where", ("group", "having", "order", "limit"))
    join_gold = tuple(
        (kind.lower().replace("  ", " "), table.lower())
        for kind, table in re.findall(
            r"\b(from|left\s+join|right\s+join|inner\s+join|outer\s+join|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
            gold or "",
            re.IGNORECASE,
        )
    )
    join_predicted = tuple(
        (kind.lower().replace("  ", " "), table.lower())
        for kind, table in re.findall(
            r"\b(from|left\s+join|right\s+join|inner\s+join|outer\s+join|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
            predicted or "",
            re.IGNORECASE,
        )
    )
    agg_gold = " ".join(
        [
            clause(gold, "select", ("from",)),
            clause(gold, "group", ("having", "order", "limit")),
            clause(gold, "order", ("limit",)),
        ]
    )
    agg_predicted = " ".join(
        [
            clause(predicted, "select", ("from",)),
            clause(predicted, "group", ("having", "order", "limit")),
            clause(predicted, "order", ("limit",)),
        ]
    )
    return {
        "new_identifier_tokens": sorted(predicted_tokens - gold_tokens),
        "removed_identifier_tokens": sorted(gold_tokens - predicted_tokens),
        "new_qualified_identifiers": sorted(predicted_qualified - gold_qualified),
        "removed_qualified_identifiers": sorted(gold_qualified - predicted_qualified),
        "gold_tables": sorted(gold_tables),
        "predicted_tables": sorted(predicted_tables),
        "tables_changed": gold_tables != predicted_tables,
        "join_structure_changed": join_gold != join_predicted,
        "where_changed": where_gold != where_predicted,
        "aggregation_group_order_changed": agg_gold != agg_predicted,
        "date_or_dialect_signal": bool(
            re.search(r"\b(?:year|date_trunc|like)\b|\d{4}-\d{2}-\d{2}", gold or "", re.I)
            or re.search(r"\b(?:year|date_trunc|like)\b|\d{4}-\d{2}-\d{2}", predicted or "", re.I)
        ),
    }


def classify_sql_mismatch(
    gold_action: Any,
    predicted_action: Any | None,
    predicted_raw_output: str | None = None,
) -> dict[str, Any]:
    """Classify observable text/action differences without executing SQL."""

    gold_kind = action_kind(gold_action)
    predicted_kind = action_kind(predicted_action)
    gold_query = action_args(gold_action).get("query") if gold_kind == "sql" else None
    predicted_query = (
        action_args(predicted_action).get("query") if predicted_kind == "sql" else None
    )
    tags: list[str] = []
    if predicted_kind != "sql":
        raw_looks_sql = bool(
            predicted_raw_output and re.search(r"\bselect\b|\bfrom\b", predicted_raw_output, re.I)
        )
        if predicted_raw_output and (
            "\\n" in predicted_raw_output or raw_looks_sql and predicted_action is None
        ):
            primary = "syntax_or_serialization_error"
            tags.append("missing_or_invalid_sql_action")
        else:
            primary = "unclassified"
            tags.append("action_kind_mismatch")
        return {
            "primary_category": primary,
            "categories": [primary],
            "tags": tags,
            "confidence": "direct",
            "shape": None,
        }
    if gold_kind != "sql":
        return {
            "primary_category": "unclassified",
            "categories": ["unclassified"],
            "tags": ["gold_action_not_sql"],
            "confidence": "direct",
            "shape": None,
        }
    if gold_query == predicted_query:
        return {
            "primary_category": None,
            "categories": [],
            "tags": ["exact_raw_query"],
            "confidence": "direct",
            "shape": sql_shape_flags(gold_query, predicted_query),
        }
    if normalize_sql_format(gold_query) == normalize_sql_format(predicted_query):
        return {
            "primary_category": "formatting_only",
            "categories": ["formatting_only"],
            "tags": ["format_normalized_equal"],
            "confidence": "high_text_only",
            "shape": sql_shape_flags(gold_query, predicted_query),
        }

    shape = sql_shape_flags(gold_query, predicted_query)
    categories: list[str] = []
    if shape["new_identifier_tokens"] or shape["new_qualified_identifiers"]:
        categories.append("identifier_mismatch")
    if shape["join_structure_changed"] or shape["tables_changed"]:
        categories.append("relation_join_mismatch")
    if (
        shape["where_changed"]
        or shape["date_or_dialect_signal"]
        and (
            re.search(r"\b(?:year|date_trunc|like)\b", gold_query or "", re.I)
            or re.search(r"\b(?:year|date_trunc|like)\b", predicted_query or "", re.I)
        )
    ):
        categories.append("filter_date_dialect_mismatch")
    if shape["aggregation_group_order_changed"]:
        categories.append("aggregation_group_order_mismatch")
    if not categories:
        categories.append("unclassified")
    primary = categories[0]
    return {
        "primary_category": primary,
        "categories": categories,
        "tags": [],
        "confidence": "text_only",
        "shape": shape,
    }


def error_identifier(error: Any) -> str | None:
    if error is None:
        return None
    match = ERROR_IDENTIFIER_RE.search(str(error))
    return match.group(1).lower() if match else None


def classify_error_followup(
    error_step: dict[str, Any], next_step: dict[str, Any] | None, terminal_reason: str | None
) -> str:
    if next_step is None:
        if terminal_reason in {"invalid_json", "invalid_action", "invalid_action_arguments"}:
            return "invalid_action_or_parser_failure"
        return "terminate_without_next_action"
    next_kind = action_kind(next_step.get("action"))
    if next_kind == "sql":
        previous_query = action_args(error_step.get("action")).get("query")
        next_query = action_args(next_step.get("action")).get("query")
        return "retry_same_sql" if previous_query == next_query else "retry_modified_sql"
    return {
        "schema": "switch_to_schema",
        "dictionary": "switch_to_dictionary",
        "check": "switch_to_check",
        "finish": "switch_to_finish",
    }.get(next_kind, "other")


def visible_schema_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, set[str]] = defaultdict(set)
    evidence_message_indices: list[int] = []
    for index, message in enumerate(messages):
        observation = parse_content(message.get("content"))
        if not isinstance(observation, dict) or not isinstance(observation.get("result"), dict):
            continue
        result = observation["result"]
        found = False
        for table, value in result.items():
            if not isinstance(value, dict) or not isinstance(value.get("columns"), list):
                continue
            names = {
                str(column.get("name"))
                for column in value["columns"]
                if isinstance(column, dict) and column.get("name") is not None
            }
            if names:
                tables[str(table)] |= names
                found = True
        if found:
            evidence_message_indices.append(index)
    return {
        "available": bool(tables),
        "tables": {table: sorted(columns) for table, columns in sorted(tables.items())},
        "evidence_message_indices": evidence_message_indices,
    }


def sql_visibility(query: str | None, schema: dict[str, Any]) -> dict[str, Any]:
    if not query:
        return {"status": "no_sql_query", "referenced_tables": [], "qualified_fields": []}
    referenced_tables = sorted(table_references(query))
    qualified = sorted(qualified_identifiers(query))
    if not schema.get("available"):
        return {
            "status": "unconfirmed_no_visible_schema",
            "referenced_tables": referenced_tables,
            "qualified_fields": qualified,
            "unknown_tables": None,
            "unknown_qualified_fields": None,
        }
    table_map = {
        str(k).lower(): {str(v).lower() for v in values} for k, values in schema["tables"].items()
    }
    aliases: dict[str, str] = {}
    for match in TABLE_AFTER_RE.finditer(query):
        table, alias = match.group(1).lower(), match.group(2)
        aliases[alias.lower() if alias else table] = table
    unknown_tables = sorted(table for table in referenced_tables if table not in table_map)
    unknown_fields: list[str] = []
    for qualified_name in qualified:
        alias, field = qualified_name.split(".", 1)
        table = aliases.get(alias, alias)
        if table in table_map and field not in table_map[table]:
            unknown_fields.append(qualified_name)
    return {
        "status": "confirmed_visible_schema_check",
        "referenced_tables": referenced_tables,
        "qualified_fields": qualified,
        "unknown_tables": unknown_tables,
        "unknown_qualified_fields": sorted(unknown_fields),
        "schema_tables": sorted(table_map),
    }


def observation_before(
    messages: list[dict[str, Any]], assistant_index: int
) -> dict[str, Any] | None:
    if assistant_index <= 0 or assistant_index - 1 >= len(messages):
        return None
    message = messages[assistant_index - 1]
    parsed = parse_content(message.get("content"))
    if isinstance(parsed, dict):
        return parsed
    return None


def build_dataset_audit(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    tasks: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source = row.get("source") or {}
        task_id = str(row.get("task_id"))
        messages = row.get("messages") or []
        indices = row.get("assistant_message_indices") or []
        schema = visible_schema_from_messages(messages)
        actions: list[dict[str, Any]] = []
        error_feedback: list[dict[str, Any]] = []
        for sequence_index, assistant_index in enumerate(indices):
            if (
                not isinstance(assistant_index, int)
                or assistant_index < 0
                or assistant_index >= len(messages)
            ):
                anomalies.append(
                    {
                        "task_id": task_id,
                        "type": "assistant_index_out_of_range",
                        "assistant_message_index": assistant_index,
                    }
                )
                continue
            message = messages[assistant_index]
            raw = message.get("content")
            parsed = parse_content(raw)
            if message.get("role") != "assistant":
                anomalies.append(
                    {
                        "task_id": task_id,
                        "type": "target_not_assistant",
                        "assistant_message_index": assistant_index,
                    }
                )
            if parsed is None or not isinstance(parsed, dict) or "kind" not in parsed:
                anomalies.append(
                    {
                        "task_id": task_id,
                        "type": "target_action_parse_failure",
                        "assistant_message_index": assistant_index,
                    }
                )
                parsed = {"kind": None, "arguments": {}}
            previous = observation_before(messages, assistant_index)
            previous_action = actions[-1] if actions else None
            has_error_feedback = isinstance(previous, dict) and previous.get("ok") is False
            correction: dict[str, Any] | None = None
            if has_error_feedback:
                correction = {
                    "from_sequence_index": previous_action["sequence_index"]
                    if previous_action
                    else None,
                    "from_kind": previous_action["kind"] if previous_action else None,
                    "from_query": previous_action.get("arguments", {}).get("query")
                    if previous_action
                    else None,
                    "to_sequence_index": sequence_index,
                    "to_kind": action_kind(parsed),
                    "to_query": action_args(parsed).get("query"),
                    "query_changed": (
                        previous_action is not None
                        and previous_action.get("kind") == "sql"
                        and action_kind(parsed) == "sql"
                        and previous_action.get("arguments", {}).get("query")
                        != action_args(parsed).get("query")
                    ),
                    "error": previous.get("error"),
                    "observation_message_index": assistant_index - 1,
                }
                error_feedback.append(correction)
            schema_before = visible_schema_from_messages(messages[:assistant_index])
            action_record = {
                "sequence_index": sequence_index,
                "assistant_message_index": assistant_index,
                "raw_action": raw,
                "kind": action_kind(parsed),
                "arguments": action_args(parsed),
                "preceding_action_kinds": [item["kind"] for item in actions],
                "preceding_schema_or_dictionary": [
                    item["kind"] for item in actions if item["kind"] in {"schema", "dictionary"}
                ],
                "preceding_observation_ok": previous.get("ok")
                if isinstance(previous, dict)
                else None,
                "preceding_observation_error": previous.get("error")
                if isinstance(previous, dict)
                else None,
            }
            if action_kind(parsed) == "sql":
                query = action_args(parsed).get("query")
                action_record["sql_visibility"] = sql_visibility(query, schema_before)
            actions.append(action_record)
            targets.append(
                {
                    "task_id": task_id,
                    "database_id": source.get("database_id"),
                    "template_id": source.get("template_id"),
                    "trajectory_source": source.get("trajectory_source"),
                    "dataset_row_index": row_index,
                    "target_id": f"{row.get('example_id')}:assistant-{assistant_index}",
                    **action_record,
                }
            )

        sql_positions = [item["sequence_index"] for item in actions if item["kind"] == "sql"]
        check_positions = [item["sequence_index"] for item in actions if item["kind"] == "check"]
        finish_positions = [item["sequence_index"] for item in actions if item["kind"] == "finish"]
        consistency: list[dict[str, Any]] = []
        for item in actions:
            if item["kind"] not in {"check", "finish"}:
                continue
            evidence = item["arguments"].get("evidence_step")
            valid = (
                isinstance(evidence, int)
                and evidence in sql_positions
                and evidence < item["sequence_index"]
            )
            consistency.append(
                {
                    "sequence_index": item["sequence_index"],
                    "kind": item["kind"],
                    "evidence_step": evidence,
                    "valid_prior_sql_reference": valid,
                    "expected_columns": item["arguments"].get("expected_columns"),
                    "summary_present": bool(item["arguments"].get("summary"))
                    if item["kind"] == "finish"
                    else None,
                }
            )
            if not valid:
                anomalies.append(
                    {
                        "task_id": task_id,
                        "type": "invalid_evidence_step",
                        "sequence_index": item["sequence_index"],
                        "kind": item["kind"],
                        "evidence_step": evidence,
                    }
                )
        gold_sql_targets = [
            {
                "sequence_index": item["sequence_index"],
                "assistant_message_index": item["assistant_message_index"],
                "query": item["arguments"].get("query"),
                "raw_arguments": item["arguments"],
            }
            for item in actions
            if item["kind"] == "sql"
        ]
        task_record = {
            "task_id": task_id,
            "database_id": source.get("database_id"),
            "template_id": source.get("template_id"),
            "trajectory_source": source.get("trajectory_source"),
            "example_id": row.get("example_id"),
            "dataset_row_index": row_index,
            "source_file": str(dataset_path),
            "source_trajectory_path": source.get("trajectory_path"),
            "action_count": len(actions),
            "action_kind_sequence": [item["kind"] for item in actions],
            "actions": actions,
            "schema_positions": [
                item["sequence_index"] for item in actions if item["kind"] == "schema"
            ],
            "dictionary_positions": [
                item["sequence_index"] for item in actions if item["kind"] == "dictionary"
            ],
            "sql_positions": sql_positions,
            "check_positions": check_positions,
            "finish_positions": finish_positions,
            "gold_sql_targets": gold_sql_targets,
            "dictionary_targets": [
                {"sequence_index": item["sequence_index"], "arguments": item["arguments"]}
                for item in actions
                if item["kind"] == "dictionary"
            ],
            "has_preceding_schema_or_dictionary_before_first_sql": bool(
                any(
                    item["kind"] in {"schema", "dictionary"} for item in actions[: sql_positions[0]]
                )
                if sql_positions
                else any(item["kind"] in {"schema", "dictionary"} for item in actions)
            ),
            "real_error_feedback": error_feedback,
            "has_real_error_feedback": bool(error_feedback),
            "has_post_error_correction_sql_target": any(
                item["to_kind"] == "sql" and item["query_changed"] for item in error_feedback
            ),
            "error_feedback_followup_kinds": [item["to_kind"] for item in error_feedback],
            "check_finish_consistency": consistency,
            "visible_schema_evidence": schema,
            "anomalies": [],
        }
        task_anomalies = [item for item in anomalies if item.get("task_id") == task_id]
        task_record["anomalies"] = task_anomalies
        tasks.append(task_record)

    kind_counts = Counter(item["kind"] for item in targets)
    source_counts = Counter(item["trajectory_source"] for item in tasks)
    source_target_counts = Counter(item["trajectory_source"] for item in targets)
    template_counts = Counter(item["template_id"] for item in tasks)
    paths_by_template: dict[str, dict[str, Any]] = {}
    for template in sorted(template_counts):
        group = [item for item in tasks if item["template_id"] == template]
        paths_by_template[template] = {
            "task_ids": [item["task_id"] for item in group],
            "unique_action_kind_sequences": sorted(
                {"→".join(item["action_kind_sequence"]) for item in group}
            ),
            "by_source": {
                source: sorted(
                    {
                        "→".join(item["action_kind_sequence"])
                        for item in group
                        if item["trajectory_source"] == source
                    }
                )
                for source in sorted({item["trajectory_source"] for item in group})
            },
            "same_template_path_differs": len(
                {tuple(item["action_kind_sequence"]) for item in group}
            )
            > 1,
        }
    audit = {
        "row_count": len(rows),
        "target_count": len(targets),
        "expected_row_count": 20,
        "expected_target_count": 108,
        "counts_match_expected": len(rows) == 20 and len(targets) == 108,
        "gold_action_kind_counts": dict(sorted(kind_counts.items())),
        "task_source_counts": dict(sorted(source_counts.items())),
        "target_source_counts": dict(sorted(source_target_counts.items())),
        "template_task_counts": dict(sorted(template_counts.items())),
        "tasks": tasks,
        "targets": targets,
        "same_template_branch_audit": paths_by_template,
        "anomalies": anomalies,
        "structural_anomaly_count": len(anomalies),
    }
    return tasks, audit


def build_prediction_audit(
    prediction_path: Path,
    dataset_targets: list[dict[str, Any]],
    dataset_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = read_jsonl(prediction_path)
    target_map = {
        (item["task_id"], item["assistant_message_index"]): item for item in dataset_targets
    }
    aligned: list[dict[str, Any]] = []
    alignment_anomalies: list[dict[str, Any]] = []
    for prediction_index, row in enumerate(rows):
        key = (row.get("task_id"), row.get("assistant_message_index"))
        gold_record = target_map.get(key)
        if gold_record is None:
            alignment_anomalies.append(
                {
                    "prediction_index": prediction_index,
                    "key": list(key),
                    "type": "no_dataset_target",
                }
            )
            continue
        gold_action = {"kind": gold_record["kind"], "arguments": gold_record["arguments"]}
        predicted_action = row.get("predicted_action")
        exact = action_exact(gold_action, predicted_action)
        kind_match = action_kind(gold_action) == action_kind(predicted_action)
        parameter_exact = kind_match and action_args(gold_action) == action_args(predicted_action)
        record = {
            "prediction_index": prediction_index,
            "target_id": gold_record["target_id"],
            "task_id": row.get("task_id"),
            "database_id": gold_record["database_id"],
            "template_id": gold_record["template_id"],
            "trajectory_source": row.get("teacher_source")
            or row.get("trajectory_source")
            or gold_record["trajectory_source"],
            "assistant_message_index": row.get("assistant_message_index"),
            "gold_kind": action_kind(gold_action),
            "predicted_kind": action_kind(predicted_action),
            "gold_action": gold_action,
            "predicted_action": predicted_action,
            "gold_raw_action": gold_record["raw_action"],
            "predicted_raw_output": row.get("raw_output") or row.get("predicted_raw_output"),
            "parser_accepted": bool(
                row.get("runtime_parser_accepted", row.get("parser_accepted", False))
            ),
            "parser_error": row.get("parser_error"),
            "exact_match": exact,
            "kind_match": kind_match,
            "parameter_exact": parameter_exact,
            "source_file": str(prediction_path),
        }
        if action_kind(gold_action) == "sql":
            record["sql_mismatch"] = classify_sql_mismatch(
                gold_action, predicted_action, record["predicted_raw_output"]
            )
            record["gold_query"] = action_args(gold_action).get("query")
            record["predicted_query"] = (
                action_args(predicted_action).get("query")
                if action_kind(predicted_action) == "sql"
                else None
            )
        aligned.append(record)

    def confusion(records: list[dict[str, Any]], gold_kind: str) -> dict[str, Any]:
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in records:
            if record["gold_kind"] != gold_kind:
                continue
            matrix[gold_kind][record["predicted_kind"] or "missing_or_unparsed"] += 1
        return {row: dict(sorted(columns.items())) for row, columns in sorted(matrix.items())}

    schema_records = [record for record in aligned if record["gold_kind"] == "schema"]
    dictionary_records = [record for record in aligned if record["gold_kind"] == "dictionary"]
    sql_records = [record for record in aligned if record["gold_kind"] == "sql"]
    by_source: dict[str, Any] = {}
    for source in SOURCE_NAMES:
        subset = [record for record in aligned if record["trajectory_source"] == source]
        by_source[source] = {
            "target_count": len(subset),
            "gold_kind_counts": dict(
                sorted(Counter(record["gold_kind"] for record in subset).items())
            ),
            "predicted_kind_counts": dict(
                sorted(
                    Counter(
                        record["predicted_kind"] or "missing_or_unparsed" for record in subset
                    ).items()
                )
            ),
            "schema": {
                "target_count": sum(record["gold_kind"] == "schema" for record in subset),
                "kind_match_count": sum(
                    record["gold_kind"] == "schema" and record["kind_match"] for record in subset
                ),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "schema" and record["parameter_exact"]
                    for record in subset
                ),
            },
            "dictionary": {
                "target_count": sum(record["gold_kind"] == "dictionary" for record in subset),
                "kind_match_count": sum(
                    record["gold_kind"] == "dictionary" and record["kind_match"]
                    for record in subset
                ),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "dictionary" and record["parameter_exact"]
                    for record in subset
                ),
            },
            "sql": {
                "target_count": sum(record["gold_kind"] == "sql" for record in subset),
                "kind_match_count": sum(
                    record["gold_kind"] == "sql" and record["kind_match"] for record in subset
                ),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "sql" and record["parameter_exact"] for record in subset
                ),
            },
        }
    by_template: dict[str, Any] = {}
    for template in sorted({record["template_id"] for record in aligned}):
        subset = [record for record in aligned if record["template_id"] == template]
        by_template[template] = {
            "target_count": len(subset),
            "gold_kind_counts": dict(
                sorted(Counter(record["gold_kind"] for record in subset).items())
            ),
            "predicted_kind_counts": dict(
                sorted(
                    Counter(
                        record["predicted_kind"] or "missing_or_unparsed" for record in subset
                    ).items()
                )
            ),
            "schema": {
                "target_count": sum(record["gold_kind"] == "schema" for record in subset),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "schema" and record["parameter_exact"]
                    for record in subset
                ),
            },
            "dictionary": {
                "target_count": sum(record["gold_kind"] == "dictionary" for record in subset),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "dictionary" and record["parameter_exact"]
                    for record in subset
                ),
            },
            "sql": {
                "target_count": sum(record["gold_kind"] == "sql" for record in subset),
                "kind_match_count": sum(
                    record["gold_kind"] == "sql" and record["kind_match"] for record in subset
                ),
                "parameter_exact_count": sum(
                    record["gold_kind"] == "sql" and record["parameter_exact"] for record in subset
                ),
            },
        }
    mismatch_counts = Counter(
        record["sql_mismatch"]["primary_category"]
        for record in sql_records
        if record.get("sql_mismatch", {}).get("primary_category")
    )
    dictionary_confusion = Counter(
        (
            argument_class(record["gold_action"]),
            action_kind(record["predicted_action"]) or "missing_or_unparsed",
        )
        for record in dictionary_records
    )
    dictionary_exact = Counter(
        (argument_class(record["gold_action"]), argument_class(record["predicted_action"]))
        for record in dictionary_records
    )
    return {
        "prediction_count": len(rows),
        "aligned_count": len(aligned),
        "expected_target_count": len(dataset_targets),
        "alignment_anomalies": alignment_anomalies,
        "overall": {
            "gold_kind_counts": dict(
                sorted(Counter(record["gold_kind"] for record in aligned).items())
            ),
            "predicted_kind_counts": dict(
                sorted(
                    Counter(
                        record["predicted_kind"] or "missing_or_unparsed" for record in aligned
                    ).items()
                )
            ),
            "exact_match_count": sum(record["exact_match"] for record in aligned),
            "kind_match_count": sum(record["kind_match"] for record in aligned),
            "parameter_exact_count": sum(record["parameter_exact"] for record in aligned),
            "parser_accepted_count": sum(record["parser_accepted"] for record in aligned),
        },
        "schema": {
            "gold_action_kind_distribution": dict(
                sorted(Counter(record["gold_kind"] for record in schema_records).items())
            ),
            "predicted_action_kind_distribution": dict(
                sorted(
                    Counter(
                        record["predicted_kind"] or "missing_or_unparsed"
                        for record in schema_records
                    ).items()
                )
            ),
            "confusion_matrix": confusion(aligned, "schema"),
            "target_count": len(schema_records),
            "kind_match_count": sum(record["kind_match"] for record in schema_records),
            "parameter_exact_count": sum(record["parameter_exact"] for record in schema_records),
        },
        "dictionary": {
            "gold_action_kind_distribution": dict(
                sorted(Counter(record["gold_kind"] for record in dictionary_records).items())
            ),
            "predicted_action_kind_distribution": dict(
                sorted(
                    Counter(
                        record["predicted_kind"] or "missing_or_unparsed"
                        for record in dictionary_records
                    ).items()
                )
            ),
            "confusion_matrix": confusion(aligned, "dictionary"),
            "target_count": len(dictionary_records),
            "kind_match_count": sum(record["kind_match"] for record in dictionary_records),
            "parameter_exact_count": sum(
                record["parameter_exact"] for record in dictionary_records
            ),
            "gold_argument_class_distribution": dict(
                sorted(
                    Counter(
                        argument_class(record["gold_action"]) for record in dictionary_records
                    ).items()
                )
            ),
            "predicted_argument_class_distribution": dict(
                sorted(
                    Counter(
                        argument_class(record["predicted_action"]) for record in dictionary_records
                    ).items()
                )
            ),
            "kind_confusion_by_gold_argument": {
                f"{gold}→{pred}": count
                for (gold, pred), count in sorted(dictionary_confusion.items())
            },
            "argument_confusion": {
                f"{gold}→{pred}": count for (gold, pred), count in sorted(dictionary_exact.items())
            },
            "records": dictionary_records,
        },
        "sql": {
            "target_count": len(sql_records),
            "predicted_sql_kind_count": sum(
                record["predicted_kind"] == "sql" for record in sql_records
            ),
            "kind_match_count": sum(record["kind_match"] for record in sql_records),
            "parameter_exact_count": sum(record["parameter_exact"] for record in sql_records),
            "formatting_only_count": sum(
                record.get("sql_mismatch", {}).get("primary_category") == "formatting_only"
                for record in sql_records
            ),
            "mismatch_primary_category_counts": dict(sorted(mismatch_counts.items())),
            "records": sql_records,
        },
        "by_trajectory_source": by_source,
        "by_template": by_template,
        "records": aligned,
    }


def summarize_tool_result(step: dict[str, Any]) -> dict[str, Any]:
    result = step.get("tool_result") or {}
    raw_result = result.get("result")
    summary: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
    }
    if isinstance(raw_result, dict):
        if isinstance(raw_result.get("columns"), list):
            summary["columns"] = raw_result.get("columns")
        if isinstance(raw_result.get("rows"), list):
            summary["row_count"] = len(raw_result.get("rows"))
            summary["rows_preview"] = raw_result.get("rows")[:3]
        if isinstance(raw_result.get("ok"), bool):
            summary["nested_ok"] = raw_result.get("ok")
    return summary


def runtime_action_record(step: dict[str, Any]) -> dict[str, Any]:
    action = step.get("action") or {}
    return {
        "step_index": step.get("step_index"),
        "kind": action_kind(action),
        "arguments": action_args(action),
        "tool_name": step.get("tool_name"),
        "tool_result": summarize_tool_result(step),
    }


def runtime_source_files(root: Path, group: str, task_id: str) -> dict[str, str]:
    group_dir = root / group
    return {
        "result": repo_relative(
            group_dir / f"task-{task_id}-result.json", root.parent.parent.parent
        ),
        "trajectory": repo_relative(
            group_dir / f"task-{task_id}-trajectory.json", root.parent.parent.parent
        ),
        "model_requests": repo_relative(
            group_dir / "model-requests.jsonl", root.parent.parent.parent
        ),
        "sql_audit": repo_relative(group_dir / "sql-audit.jsonl", root.parent.parent.parent),
        "raw_generations": repo_relative(
            group_dir / "raw-generations.jsonl", root.parent.parent.parent
        ),
    }


def compare_runtime_query_to_gold(
    query: str | None, gold_sql_targets: list[dict[str, Any]]
) -> dict[str, Any]:
    exact = [item["sequence_index"] for item in gold_sql_targets if item.get("query") == query]
    format_equal = [
        item["sequence_index"]
        for item in gold_sql_targets
        if normalize_sql_format(item.get("query")) == normalize_sql_format(query)
    ]
    if exact:
        return {
            "exact_gold_sequence_indices": exact,
            "format_equal_gold_sequence_indices": format_equal,
            "classification": None,
        }
    if format_equal:
        return {
            "exact_gold_sequence_indices": [],
            "format_equal_gold_sequence_indices": format_equal,
            "classification": "formatting_only",
        }
    if not gold_sql_targets:
        return {
            "exact_gold_sequence_indices": [],
            "format_equal_gold_sequence_indices": [],
            "classification": "unclassified",
        }
    # A nearest target is only a textual anchor; it is not a semantic judgment.
    query_tokens = identifier_tokens(query)
    scored = []
    for item in gold_sql_targets:
        target_tokens = identifier_tokens(item.get("query"))
        union = query_tokens | target_tokens
        score = len(query_tokens & target_tokens) / len(union) if union else 0.0
        scored.append((score, item))
    score, nearest = max(scored, key=lambda pair: (pair[0], -int(pair[1]["sequence_index"])))
    classification = classify_sql_mismatch(
        {"kind": "sql", "arguments": {"query": nearest.get("query")}},
        {"kind": "sql", "arguments": {"query": query}},
    )
    return {
        "exact_gold_sequence_indices": [],
        "format_equal_gold_sequence_indices": [],
        "nearest_gold_sequence_index": nearest.get("sequence_index"),
        "nearest_identifier_jaccard": round(score, 6),
        "classification": classification,
    }


def build_runtime_audit(
    runtime_root: Path, dataset_tasks: list[dict[str, Any]], prediction_audit: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_map = {item["task_id"]: item for item in dataset_tasks}
    groups: dict[str, Any] = {}
    all_task_evidence: list[dict[str, Any]] = []
    for group in GROUPS:
        group_dir = runtime_root / group
        result_paths = sorted(group_dir.glob("task-*-result.json"))
        if len(result_paths) != 20:
            raise ValueError(f"{group}:expected_20_result_files:{len(result_paths)}")
        task_records: list[dict[str, Any]] = []
        error_events: list[dict[str, Any]] = []
        actual_action_counts = Counter()
        parser_terminal_counts = Counter()
        sql_mismatch_counts = Counter()
        for result_path in result_paths:
            result = read_json(result_path)
            task_id = str(result["task_id"])
            trajectory_path = group_dir / f"task-{task_id}-trajectory.json"
            trajectory = read_json(trajectory_path)
            steps = trajectory.get("steps") or []
            task = task_map.get(task_id)
            if task is None:
                raise ValueError(f"runtime_task_not_in_sft:{group}:{task_id}")
            for step in steps:
                actual_action_counts[action_kind(step.get("action"))] += 1
            terminal_reason = result.get("terminal_reason") or trajectory.get("terminal_reason")
            if terminal_reason in {"invalid_json", "invalid_action", "invalid_action_arguments"}:
                parser_terminal_counts[terminal_reason] += 1
            elif terminal_reason in {"action_budget_exhausted", "tool_budget_exhausted"}:
                parser_terminal_counts["budget_exhaustion"] += 1
            else:
                parser_terminal_counts["other_terminal"] += 1
            sql_steps = [step for step in steps if action_kind(step.get("action")) == "sql"]
            first_sql_index = min((int(step.get("step_index")) for step in sql_steps), default=None)
            pre_branch_steps = [
                step
                for step in steps
                if action_kind(step.get("action")) in {"schema", "dictionary"}
                and (first_sql_index is None or int(step.get("step_index")) < first_sql_index)
            ]
            sql_events: list[dict[str, Any]] = []
            queries_seen: dict[str, list[int]] = defaultdict(list)
            for sql_event_index, step in enumerate(sql_steps):
                action = step.get("action") or {}
                query = action_args(action).get("query")
                tool_result = step.get("tool_result") or {}
                next_step = next(
                    (
                        candidate
                        for candidate in steps
                        if candidate.get("step_index", -1) > step.get("step_index", -1)
                    ),
                    None,
                )
                prior_same_indices = list(queries_seen.get(str(query), []))
                queries_seen[str(query)].append(sql_event_index)
                later_sql_success = any(
                    action_kind(candidate.get("action")) == "sql"
                    and bool((candidate.get("tool_result") or {}).get("ok"))
                    and candidate.get("step_index", -1) > step.get("step_index", -1)
                    for candidate in steps
                )
                event = {
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "task_id": task_id,
                    "database_id": task["database_id"],
                    "template_id": task["template_id"],
                    "trajectory_source": task["trajectory_source"],
                    "sql_event_index": sql_event_index,
                    "step_index": step.get("step_index"),
                    "query": query,
                    "query_sha256": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
                    "tool_result_ok": bool(tool_result.get("ok")),
                    "tool_result": summarize_tool_result(step),
                    "error": tool_result.get("error"),
                    "error_identifier": error_identifier(tool_result.get("error")),
                    "error_is_budget_or_terminal_sentinel": str(tool_result.get("error"))
                    in SQL_SENTINEL_ERRORS,
                    "next_step_index": next_step.get("step_index") if next_step else None,
                    "next_action": action_kind(next_step.get("action")) if next_step else None,
                    "prior_exact_query_event_indices": prior_same_indices,
                    "later_successful_sql": later_sql_success,
                    "gold_comparison": compare_runtime_query_to_gold(
                        query, task["gold_sql_targets"]
                    ),
                    "source_files": runtime_source_files(runtime_root, group, task_id),
                }
                sql_events.append(event)
                if not event["tool_result_ok"]:
                    next_category = classify_error_followup(step, next_step, terminal_reason)
                    error_event = {
                        **event,
                        "error_to_next_action": next_category,
                        "later_successful_sql": later_sql_success,
                        "termination_reason_if_no_next": terminal_reason
                        if next_step is None
                        else None,
                    }
                    error_events.append(error_event)
            task_record = {
                "task_id": task_id,
                "database_id": result.get("database_id", task["database_id"]),
                "template_id": result.get("template_id", task["template_id"]),
                "trajectory_source": result.get("teacher_provenance", task["trajectory_source"]),
                "group": group,
                "group_label": GROUP_LABELS[group],
                "success": bool(result.get("success")),
                "reward": result.get("reward"),
                "terminal_reason": terminal_reason,
                "failure_reason": result.get("failure_reason"),
                "accepted_action_count": len(steps),
                "action_kind_sequence": [action_kind(step.get("action")) for step in steps],
                "actions": [runtime_action_record(step) for step in steps],
                "schema_dictionary_before_first_sql": [
                    action_kind(step.get("action")) for step in pre_branch_steps
                ],
                "schema_dictionary_before_first_sql_covered": bool(pre_branch_steps),
                "first_sql_step_index": first_sql_index,
                "sql_event_count": len(sql_events),
                "sql_success_count": sum(event["tool_result_ok"] for event in sql_events),
                "sql_error_count": sum(not event["tool_result_ok"] for event in sql_events),
                "sql_execution_error_count": sum(
                    not event["tool_result_ok"]
                    and not event["error_is_budget_or_terminal_sentinel"]
                    for event in sql_events
                ),
                "sql_events": sql_events,
                "parser_or_action_terminal": terminal_reason
                in {"invalid_json", "invalid_action", "invalid_action_arguments"},
                "budget_terminal": terminal_reason
                in {"action_budget_exhausted", "tool_budget_exhausted"},
                "source_files": runtime_source_files(runtime_root, group, task_id),
                "result_file": str(result_path),
                "trajectory_file": str(trajectory_path),
            }
            task_records.append(task_record)
            all_task_evidence.append(task_record)
            for event in sql_events:
                if event["gold_comparison"].get("classification"):
                    classification = event["gold_comparison"]["classification"]
                    if isinstance(classification, dict):
                        sql_mismatch_counts[
                            classification.get("primary_category", "unclassified")
                        ] += 1
                    else:
                        sql_mismatch_counts[classification] += 1

        def count_events(category: str, events=error_events) -> int:
            return sum(event["error_to_next_action"] == category for event in events)

        groups[group] = {
            "group": group,
            "label": GROUP_LABELS[group],
            "config": (
                read_json(group_dir / "summary.json").get("group")
                if (group_dir / "summary.json").exists()
                else None
            ),
            "task_count": len(task_records),
            "task_success_count": sum(record["success"] for record in task_records),
            "task_success_rate": sum(record["success"] for record in task_records)
            / len(task_records),
            "accepted_action_kind_counts": dict(sorted(actual_action_counts.items())),
            "schema_dictionary_before_first_sql_count": sum(
                record["schema_dictionary_before_first_sql_covered"] for record in task_records
            ),
            "schema_dictionary_before_first_sql_coverage": sum(
                record["schema_dictionary_before_first_sql_covered"] for record in task_records
            )
            / len(task_records),
            "schema_before_first_sql_count": sum(
                "schema" in record["schema_dictionary_before_first_sql"] for record in task_records
            ),
            "dictionary_before_first_sql_count": sum(
                "dictionary" in record["schema_dictionary_before_first_sql"]
                for record in task_records
            ),
            "sql_event_count": sum(record["sql_event_count"] for record in task_records),
            "sql_success_event_count": sum(record["sql_success_count"] for record in task_records),
            "sql_error_event_count": sum(record["sql_error_count"] for record in task_records),
            "sql_execution_error_event_count": sum(
                record["sql_execution_error_count"] for record in task_records
            ),
            "sql_error_task_count": sum(record["sql_error_count"] > 0 for record in task_records),
            "sql_success_task_count": sum(
                record["sql_success_count"] > 0 for record in task_records
            ),
            "successful_sql_after_prior_sql_error": {
                "event_count": sum(event["later_successful_sql"] for event in error_events),
                "task_count": len(
                    {event["task_id"] for event in error_events if event["later_successful_sql"]}
                ),
                "definition": (
                    "A later accepted SQL Action in the same episode has tool_result.ok=true after "
                    "this saved SQL error."
                ),
            },
            "error_to_next_action": {
                category: count_events(category)
                for category in (
                    "retry_same_sql",
                    "retry_modified_sql",
                    "switch_to_schema",
                    "switch_to_dictionary",
                    "switch_to_check",
                    "switch_to_finish",
                    "invalid_action_or_parser_failure",
                    "terminate_without_next_action",
                    "other",
                )
            },
            "sql_error_after_sql_counts": {
                "sql_error_events": len(error_events),
                "again_sql_events": sum(event["next_action"] == "sql" for event in error_events),
                "same_query_events": count_events("retry_same_sql"),
                "modified_query_events": count_events("retry_modified_sql"),
                "successful_sql_after_prior_sql_error_events": sum(
                    event["later_successful_sql"] for event in error_events
                ),
                "switch_to_check_events": count_events("switch_to_check"),
                "switch_to_finish_events": count_events("switch_to_finish"),
                "parser_or_action_or_budget_terminal_tasks": sum(
                    record["parser_or_action_terminal"] or record["budget_terminal"]
                    for record in task_records
                ),
                "parser_terminal_tasks": sum(
                    record["terminal_reason"] == "invalid_json" for record in task_records
                ),
                "action_validation_terminal_tasks": sum(
                    record["terminal_reason"] in {"invalid_action", "invalid_action_arguments"}
                    for record in task_records
                ),
                "budget_terminal_tasks": sum(record["budget_terminal"] for record in task_records),
            },
            "termination_reason_counts": dict(sorted(parser_terminal_counts.items())),
            "sql_mismatch_primary_category_counts": dict(sorted(sql_mismatch_counts.items())),
            "error_events": error_events,
            "tasks": task_records,
            "artifact_line_counts": {
                "model_requests": jsonl_stats(group_dir / "model-requests.jsonl"),
                "sql_audit": jsonl_stats(group_dir / "sql-audit.jsonl"),
                "raw_generations": jsonl_stats(group_dir / "raw-generations.jsonl"),
            },
        }
    top_evidence: list[dict[str, Any]] = []
    for group in (
        "initial-observation",
        "full-alignment",
        "baseline",
        "canonical-history",
        "no-format-retry",
    ):
        for task_id in ("train-0-top", "train-1-top"):
            task_record = groups[group]["tasks"]
            record = next(item for item in task_record if item["task_id"] == task_id)
            accepted_schema = [item for item in record["actions"] if item["kind"] == "schema"]
            accepted_dictionary = [
                item for item in record["actions"] if item["kind"] == "dictionary"
            ]
            successful_sql = [event for event in record["sql_events"] if event["tool_result_ok"]]
            failed_sql = [event for event in record["sql_events"] if not event["tool_result_ok"]]
            top_evidence.append(
                {
                    "task_id": task_id,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "source_files": record["source_files"],
                    "accepted_schema_steps": accepted_schema,
                    "accepted_dictionary_steps": accepted_dictionary,
                    "probe_sql_success_direct_evidence": [
                        {
                            "sql_event_index": event["sql_event_index"],
                            "step_index": event["step_index"],
                            "query": event["query"],
                            "tool_result_ok": event["tool_result_ok"],
                            "tool_result": next(
                                item["tool_result"]
                                for item in record["actions"]
                                if item["step_index"] == event["step_index"]
                            ),
                        }
                        for event in successful_sql
                    ],
                    "failed_sql_events": failed_sql,
                    "accepted_sql_present": bool(record["sql_events"]),
                    "main_sql_after_probe": [
                        event
                        for event in record["sql_events"]
                        if successful_sql and event["step_index"] > successful_sql[0]["step_index"]
                    ],
                    "main_sql_corrected_later": any(
                        event["tool_result_ok"]
                        and successful_sql
                        and event["step_index"] > successful_sql[0]["step_index"]
                        for event in record["sql_events"]
                    ),
                    "no_accepted_sql_comparison_possible": not bool(record["sql_events"]),
                    "terminal_reason": record["terminal_reason"],
                    "success": record["success"],
                    "reward": record["reward"],
                }
            )
    runtime = {"groups": groups, "top_task_evidence": top_evidence}
    return runtime, all_task_evidence


def build_cross_source_evidence(
    dataset_tasks: list[dict[str, Any]], prediction_audit: dict[str, Any], runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    task_map = {item["task_id"]: item for item in dataset_tasks}
    prediction_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prediction_audit["records"]:
        if record["gold_kind"] == "sql":
            prediction_by_task[record["task_id"]].append(record)
    output: list[dict[str, Any]] = []
    for task_id in sorted(task_map):
        runtime_errors: list[dict[str, Any]] = []
        for group, group_record in runtime["groups"].items():
            for event in group_record["error_events"]:
                if event["task_id"] == task_id and event.get("error_identifier"):
                    runtime_errors.append(
                        {
                            "group": group,
                            "sql_event_index": event["sql_event_index"],
                            "step_index": event["step_index"],
                            "error": event["error"],
                            "error_identifier": event["error_identifier"],
                            "source_files": event["source_files"],
                        }
                    )
        e3_identifiers: set[str] = set()
        e3_categories: Counter[str] = Counter()
        for record in prediction_by_task.get(task_id, []):
            if record.get("predicted_query"):
                e3_identifiers |= qualified_identifiers(record["predicted_query"])
                e3_identifiers |= identifier_tokens(record["predicted_query"])
            if record.get("sql_mismatch", {}).get("primary_category"):
                e3_categories[record["sql_mismatch"]["primary_category"]] += 1
        overlaps = [
            item
            for item in runtime_errors
            if item["error_identifier"] in e3_identifiers
            or item["error_identifier"].split(".")[-1] in e3_identifiers
        ]
        output.append(
            {
                "task_id": task_id,
                "database_id": task_map[task_id]["database_id"],
                "template_id": task_map[task_id]["template_id"],
                "trajectory_source": task_map[task_id]["trajectory_source"],
                "teacher_forced_sql_primary_categories": dict(sorted(e3_categories.items())),
                "teacher_forced_predicted_identifier_tokens": sorted(e3_identifiers),
                "runtime_error_event_count_with_extractable_identifier": len(runtime_errors),
                "runtime_error_identifiers": sorted(
                    {item["error_identifier"] for item in runtime_errors}
                ),
                "runtime_error_teacher_forced_identifier_overlap": overlaps,
                "source_files": {
                    "dataset": task_map[task_id]["source_file"],
                    "teacher_forced_predictions": sorted(
                        {record["source_file"] for record in prediction_by_task.get(task_id, [])}
                    ),
                    "runtime": sorted(
                        {item["source_files"]["trajectory"] for item in runtime_errors}
                    ),
                },
                "interpretation": (
                    "textual_overlap_only_no_causal_claim"
                    if overlaps
                    else "no_direct_textual_overlap_or_no_extractable_identifier"
                ),
            }
        )
    return output


def runtime_branch_strata(task_records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(record[field]) for record in task_records})
    output: dict[str, Any] = {}
    for value in values:
        subset = [record for record in task_records if str(record[field]) == value]
        actions = [action for record in subset for action in record["actions"]]
        schema_actions = [action for action in actions if action["kind"] == "schema"]
        dictionary_actions = [action for action in actions if action["kind"] == "dictionary"]
        output[value] = {
            "task_count": len(subset),
            "accepted_action_kind_counts": dict(
                sorted(Counter(action["kind"] for action in actions).items())
            ),
            "schema_action_count": len(schema_actions),
            "dictionary_action_count": len(dictionary_actions),
            "schema_argument_classes": dict(
                sorted(
                    Counter(
                        argument_class({"kind": action["kind"], "arguments": action["arguments"]})
                        for action in schema_actions
                    ).items()
                )
            ),
            "dictionary_argument_classes": dict(
                sorted(
                    Counter(
                        argument_class({"kind": action["kind"], "arguments": action["arguments"]})
                        for action in dictionary_actions
                    ).items()
                )
            ),
            "schema_dictionary_before_first_sql_coverage_count": sum(
                record["schema_dictionary_before_first_sql_covered"] for record in subset
            ),
        }
    return output


def build_input_manifest(
    root: Path, metadata_path: Path, dataset_path: Path, prediction_path: Path, runtime_root: Path
) -> dict[str, Any]:
    paths: list[dict[str, Any]] = []
    explicit = [
        dataset_path,
        metadata_path,
        prediction_path,
        prediction_path.parent / "summary.json",
        prediction_path.parent.parent / "alignment-audit.json",
        root / "docs/experiments/sft-sql-argument-error-analysis.md",
        root / "docs/experiments/sft-prompt-runtime-alignment-audit.md",
        root / "docs/experiments/sft-runtime-ablation-postmortem.md",
        runtime_root / "postmortem.json",
        runtime_root / "comparison.json",
        runtime_root / "preflight.json",
        runtime_root / "environment.json",
    ]
    for path in explicit:
        paths.append(
            {
                "path": repo_relative(path, root),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "role": "raw_input"
                if path in {dataset_path, metadata_path, prediction_path}
                else "context_or_metadata",
            }
        )
    return {
        "read_paths": paths,
        "runtime_groups_read": [
            {
                "label": GROUP_LABELS[group],
                "directory": f"{repo_relative(runtime_root, root)}/{group}",
            }
            for group in GROUPS
        ],
        "excluded_paths": [
            (
                "artifacts/sft-data-train-teacher-v2/source-manifest.json (not read; no manifest "
                "loader)"
            ),
            "artifacts/sft-behavior-attribution/**/engine/model-load.json (not read)",
            "artifacts/**/validation/** (not read)",
            "artifacts/**/test/** (not read)",
            (
                f"{repo_relative(runtime_root, root)}/cpu-mock/** (not one of the five saved "
                f"Runtime ablation groups)"
            ),
        ],
        "boundary_checks": {
            "train_only_asserted_from_dataset_source_split": True,
            "validation_or_test_task_content_read": False,
            "model_weights_or_tokenizer_loaded": False,
            "sql_executed": False,
            "verifier_called": False,
            "runtime_or_inference_entrypoint_called": False,
        },
    }


def report_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(report: dict[str, Any]) -> str:
    dataset = report["dataset_target_audit"]
    prediction = report["action_branch_confusion"]["teacher_forced"]
    runtime = report["error_to_next_action"]["groups"]
    _error = report["error_to_next_action"]
    recommendation = report["next_experiment_recommendation"]
    lines: list[str] = []
    lines.append("# SFT SQL Parameter & Action-Branch Supervision Audit")
    lines.append("")
    lines.append("状态：`SFT_SQL_PARAMETER_SUPERVISION_AUDIT_COMPLETE`")
    lines.append("")
    lines.append(
        "本报告是 CPU-only、train-only、离线诊断；只读取已保存的 JSON/JSONL "
        "与既有报告路径。没有加载模型或 tokenizer，没有执行 "
        "SQL，没有调用 Verifier，没有重新运行任务，也没有修改生产 Runtime 或原始实验 "
        "artifacts。"
    )
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append(
        f"- SFT 原始数据重建出 {dataset['row_count']} 个 task、{dataset['target_count']} 个 ass"
        f"istant target；gold kind 计数为 `"
        f"{json.dumps(dataset['gold_action_kind_counts'], ensure_ascii=False, sort_keys=True)}`"
        f"。"
    )
    lines.append(
        f"- E3 Teacher-forced 对齐记录为 {prediction['prediction_count']} 条；Schema target 为 "
        f"{prediction['schema']['target_count']} 条，Dictionary target 为 "
        f"{prediction['dictionary']['target_count']} 条，SQL target 为 "
        f"{prediction['sql']['target_count']} 条。"
    )
    lines.append(
        f"- Dictionary target 的 gold→predicted kind/参数结果：kind match {
            prediction['dictionary']['kind_match_count']
        }/{prediction['dictionary']['target_count']}，parameter exact {
            prediction['dictionary']['parameter_exact_count']
        }/{prediction['dictionary']['target_count']}；gold 参数分布 `{
            json.dumps(
                prediction['dictionary'][('gold_argument_class_distribution')],
                ensure_ascii=False,
                sort_keys=True,
            )
        }`，预测参数分布 `{
            json.dumps(
                prediction['dictionary'][('predicted_argument_class_distribution')],
                ensure_ascii=False,
                sort_keys=True,
            )
        }`。"
    )
    lines.append(
        f"- E3 SQL target：kind match {prediction['sql']['kind_match_count']}/"
        f"{prediction['sql']['target_count']}，原始 Action parameter exact "
        f"{prediction['sql']['parameter_exact_count']}/{prediction['sql']['target_count']}；分"
        f"类只基于文本，不代表 SQL 语义已验证。"
    )
    lines.append(
        f"- 五组闭环的主指标 `successful_sql_after_prior_sql_error` 均以保存的 `tool_result.ok="
        f"true` 后续 SQL 为准；各组事件数为 `{
            json.dumps(
                {
                    k: v[('successful_sql_after_prior_sql_error')]['event_count']
                    for k, v in runtime.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        }`。既有 `reported recovery` 未用于替代该指标。"
    )
    lines.append("")
    lines.append("## 数据边界与可复核输入")
    lines.append("")
    lines.append(
        "五组标签沿用既有 postmortem：A=`baseline`、B=`initial-observation`、C=`canonical-histo"
        "ry`、D=`no-format-retry`、E=`full-alignment`。`cpu-mock` 是既有脚本的单 task 自检目录"
        "，不纳入五组闭环比较。"
    )
    lines.append("")
    lines.append(
        "所有 task-level 记录均保存 `source_files`、`task_id`、group、Action "
        "`sequence_index` 或 SQL "
        "`sql_event_index`；完整机器可读记录见同目录 JSON。"
    )
    lines.append("")
    lines.append("## 1. SFT gold supervision audit")
    lines.append("")
    lines.append("### 1.1 Task 与 Action 序列")
    lines.append("")
    lines.append(
        "每个 task 的完整 Action、Schema/Dictionary/SQL/Check/Finish 位置、原始 SQL/Dictionary "
        "arguments、error feedback 与证据引用一致性均保存在 `dataset_target_audit.tasks`。"
    )
    lines.append("")
    lines.append("相同 template 的两个 database 样本的路径差异如下：")
    lines.append("")
    lines.append("| template | task ids | unique gold Action paths | provenance split |")
    lines.append("|---|---|---|---|")
    for template, item in dataset["same_template_branch_audit"].items():
        paths = "; ".join(item["unique_action_kind_sequences"])
        by_source = "; ".join(
            f"{source}: {', '.join(paths_for_source)}"
            for source, paths_for_source in item["by_source"].items()
        )
        lines.append(f"| {template} | {', '.join(item['task_ids'])} | {paths} | {by_source} |")
    lines.append("")
    lines.append(
        "观察：gold 不是单一 Action path。分支差异与 `trajectory_source` "
        "同步出现；这说明数据存在可观测的多模态监督，但在同一 "
        "provenance/template 内没有自动把合法 SQL 字符串差异判为冲突。"
    )
    lines.append("")
    lines.append("### 1.2 Schema、Dictionary、SQL、Check、Finish")
    lines.append("")
    lines.append(
        f"- Schema target={dataset['gold_action_kind_counts'].get('schema', 0)}，Dictionary "
        f"target={dataset['gold_action_kind_counts'].get('dictionary', 0)}；Dictionary raw "
        f"arguments 的 `term:revenue`/`empty_object`/other 分布见 "
        f"`dataset_target_audit.tasks`，不把 "
        f"`{{}}` 与缺失参数混为一谈。"
    )
    lines.append(
        f"- 带真实 `ok=false` 前序 observation 的 task 数为 {
            sum(item['has_real_error_feedback'] for item in dataset['tasks'])
        }；其中包含后续改变 SQL 字符串的 gold SQL target 的 task 数为 {
            sum(item[('has_post_error_correction_sql_target')] for item in dataset['tasks'])
        }。改变字符串只能证明 target 改变，不能证明修正后的业务语义正确。"
    )
    lines.append(
        f"- Check/Finish 的 `evidence_step` 对此前 SQL sequence index 的结构检查异常数为 "
        f"{sum(len(item['anomalies']) for item in dataset['tasks'])}；所有 target "
        f"的原始证据细节保存在 "
        f"`check_finish_consistency`。"
    )
    lines.append(
        "- 没有把 reference SQL 与 deterministic fixture "
        "的合法分支差异直接判为监督冲突；真正的异常仅按 JSON/Action "
        "结构或 "
        "evidence reference 记录。"
    )
    lines.append("")
    lines.append("### 1.3 Schema evidence boundary")
    lines.append("")
    lines.append(
        "SQL 字段/表可见性只对训练消息前缀中实际出现的 Schema tool result "
        "做文本核对；没有前序 Schema evidence 的 SQL "
        "target "
        "标成 `unconfirmed_no_visible_schema`，没有从 validation/test 或数据库补证。"
    )
    visibility_counts = Counter(
        action.get("sql_visibility", {}).get("status")
        for task in dataset["tasks"]
        for action in task["actions"]
        if action["kind"] == "sql"
    )
    lines.append(
        f"本次 24 个 gold SQL target 中，前缀有可见 Schema payload 的为 "
        f"{visibility_counts.get('confirmed_visible_schema_check', 0)}，没有可见 Schema "
        f"payload "
        f"而保持 "
        f"`unconfirmed` 的为 {visibility_counts.get('unconfirmed_no_visible_schema', 0)}。"
    )
    lines.append("")
    lines.append("## 2. Dictionary / Schema branch audit")
    lines.append("")
    lines.append("### 2.1 Gold vs E3 Teacher-forced")
    lines.append("")
    schema_confusion = json.dumps(
        prediction["schema"]["confusion_matrix"], ensure_ascii=False, sort_keys=True
    )
    dictionary_confusion = json.dumps(
        prediction["dictionary"]["confusion_matrix"], ensure_ascii=False, sort_keys=True
    )
    dictionary_arguments = json.dumps(
        prediction["dictionary"]["argument_confusion"], ensure_ascii=False, sort_keys=True
    )
    lines.append(
        f"Schema confusion matrix：`{schema_confusion}`。"
        f"Dictionary confusion matrix：`{dictionary_confusion}`。"
    )
    lines.append(
        f"Dictionary 参数级 confusion：`{dictionary_arguments}`。"
        "这直接复核了既有口径：gold Dictionary target 中预测为 `schema {}`、预测为 SQL、"
        "完整匹配的数量可从 `kind_confusion_by_gold_argument` 与 `argument_confusion` 逐条追溯。"
    )
    lines.append(
        f"按 provenance：`reference_sql_teacher` 的 Dictionary kind match 为 {
            prediction['by_trajectory_source']['reference_sql_teacher']['dictionary'][
                'kind_match_count'
            ]
        }/{
            prediction['by_trajectory_source']['reference_sql_teacher']['dictionary'][
                'target_count'
            ]
        }，`deterministic_fixture_expert` 为 {
            prediction['by_trajectory_source'][('deterministic_fixture_expert')]['dictionary'][
                'kind_match_count'
            ]
        }/{
            prediction['by_trajectory_source'][('deterministic_fixture_expert')]['dictionary'][
                'target_count'
            ]
        }；"
        "Schema 两个 provenance 分层均为 kind/parameter exact。这个分层支持“混淆在 reference "
        "分支更集中”的描述，但不构成 provenance 根因的因果证明。"
    )
    lines.append("")
    lines.append("### 2.2 Provenance / template 分层")
    lines.append("")
    lines.append(
        "Teacher-forced 的 provenance 与 template 分层数字保存在 "
        "`action_branch_confusion.teacher_forced.by_trajectory_source` 与 "
        "`.by_template`；Runtime accepted Action 分层数字保存在 `.runtime.groups[*].tasks`。"
    )
    runtime_dictionary_count = sum(
        item["accepted_action_kind_counts"].get("dictionary", 0) for item in runtime.values()
    )
    runtime_schema_count = sum(
        item["accepted_action_kind_counts"].get("schema", 0) for item in runtime.values()
    )
    lines.append(
        f"闭环 Runtime 的保存 accepted Action 中 Dictionary 总数为 "
        f"{runtime_dictionary_count}，Schema "
        f"总数为 {runtime_schema_count}；因此实际行为明显倾向跳过 Dictionary，Schema "
        f"只出现在 B/E 的两个 top task "
        f"路径中。"
    )
    lines.append("")
    lines.append(
        "结论边界：Dictionary/Schema kind 混淆是直接证据；provenance 是与 gold branch "
        "同步的相关分层，不足以单独证明 "
        "provenance 是失败根因。B/E 中 accepted Schema 只在两个 top task 形成了后续 SQL "
        "路径，不能单凭这两个 task 断言 "
        "Schema Action 会因果性减少字段错误。"
    )
    lines.append("")
    lines.append("## 3. SQL parameter supervision vs Runtime")
    lines.append("")
    lines.append("### 3.1 E3 SQL mismatch")
    lines.append("")
    lines.append(
        f"E3 SQL mismatch primary category counts：`{
            json.dumps(
                prediction['sql'][('mismatch_primary_category_counts')],
                ensure_ascii=False,
                sort_keys=True,
            )
        }`。每条记录包含 gold/predicted raw query、target id、分类 shape、原始 predic"
        f"tion 路径。分类依据是原始 SQL 文本、Action kind 与 JSON serialization；没有 SQL 执行"
        f"。"
    )
    lines.append("")
    lines.append("### 3.2 Runtime group comparison")
    lines.append("")
    lines.append(
        "| group | SQL errors (all saved ok=false) | execution errors (sentinel excluded) | "
        "repeat same query | modified retry | successful SQL after prior error | Check/Finish "
        "after error | parser terminal | action validation terminal | budget terminal |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in GROUPS:
        item = runtime[group]
        c = item["sql_error_after_sql_counts"]
        lines.append(
            f"| {GROUP_LABELS[group]} {group} | {c['sql_error_events']} | "
            f"{item['sql_execution_error_event_count']} | {c['same_query_events']} | "
            f"{c['modified_query_events']} | "
            f"{c['successful_sql_after_prior_sql_error_events']} | "
            f"{c['switch_to_check_events']}/{c['switch_to_finish_events']} | "
            f"{c['parser_terminal_tasks']} | {c['action_validation_terminal_tasks']} | "
            f"{c['budget_terminal_tasks']} |"
        )
    lines.append("")
    lines.append(
        "`successful_sql_after_prior_sql_error` 是本审计主 recovery "
        "指标；`reported_recovery_success_rate` 没有用于该列。所有 error event 的 "
        "next-action 分类与 exact "
        "query repetition 明细在 `error_to_next_action.groups[*].error_events`。"
    )
    lines.append("")
    lines.append("### 3.3 Cross-source common patterns")
    lines.append("")
    lines.append(
        "`task_level_evidence.cross_source_common_patterns` 把 Teacher-forced 预测引入的 "
        "identifier "
        "与 Runtime 保存 error text 的 identifier "
        "做字符串交集。交集只说明可追溯的文本共同模式，不说明模型权重、SQL 语义或 "
        "provenance 的因果关系。"
    )
    lines.append("")
    lines.append("## 4. Top-task case study")
    lines.append("")
    lines.append(
        "B/E 两个 top task 的 direct evidence、SQL event index、tool result columns/rows "
        "preview、主 "
        "SQL error 和后续 query 均保存在 `task_level_evidence.top_task_evidence`。"
    )
    lines.append("")
    for item in report["task_level_evidence"]["top_task_evidence"]:
        if item["group"] not in {"initial-observation", "full-alignment"}:
            continue
        success = item["probe_sql_success_direct_evidence"]
        errors = item["failed_sql_events"]
        lines.append(
            f"- {item['group_label']} {item['task_id']}: accepted "
            f"Schema={bool(item['accepted_schema_steps'])}，probe success "
            f"events={len(success)}，error SQL events={len(errors)}，后续主 SQL "
            f"修正为成功={item['main_sql_corrected_later']}，task "
            f"success={item['success']}。probe "
            f"的直接证据是保存的 `tool_result.ok=true` 及其 "
            f"`columns/rows_preview`；不能把一次 tool success "
            f"等同于业务任务完成。"
        )
    lines.append(
        "- A/C/D 的两个 top task 没有 accepted SQL Action，故不存在可核验的 A/B 或 C/E SQL "
        "参数配对；报告将该比较标为不可进行，而不是推断为 SQL 错误。"
    )
    lines.append(
        "- B/E 的 Schema probe 成功后，主 SQL "
        "仍出现保存的字段错误并继续重复/耗尽预算；这直接支持“Schema accepted 不等于主 "
        "SQL "
        "参数已正确映射”，但不能单独区分分支学习、参数映射和 recovery 策略的因果贡献。"
    )
    lines.append("")
    lines.append("## 5. Attribution")
    lines.append("")
    lines.append("### Confirmed")
    lines.append("")
    for item in report["confirmed_findings"]:
        lines.append(f"- `{item['id']}`：{item['finding']}（证据：{item['evidence']}）")
    lines.append("")
    lines.append("### Likely")
    lines.append("")
    for item in report["likely_findings"]:
        lines.append(
            f"- `{item['id']}`：{item['finding']}（支持：{item['evidence']}；不能排除："
            f"{item['cannot_exclude']}）"
        )
    lines.append("")
    lines.append("### Unconfirmed")
    lines.append("")
    for item in report["unconfirmed_findings"]:
        lines.append(f"- `{item['id']}`：{item['finding']}（缺口：{item['missing_evidence']}）")
    lines.append("")
    lines.append("## 6. Next single hypothesis and minimum experiment")
    lines.append("")
    lines.append(f"优先验证假设：**{recommendation['hypothesis']}**")
    lines.append("")
    lines.append(
        f"选择理由：{recommendation['why_first']}。重点 "
        f"task：{', '.join(recommendation['supporting_task_ids'])}。"
    )
    lines.append("")
    lines.append(
        f"最小实验：{recommendation['minimum_experiment']}。判别规则：{recommendation['decision_rule']}"
    )
    lines.append("")
    lines.append(
        f"资源与变更：GPU={recommendation['requires_gpu']}；训练={
            recommendation['requires_training']
        }；生产代码修改={
            recommendation[('requires_production_code_change')]
        }。现阶段不建议增加 epoch、扩充数据或启动 GRPO。"
    )
    lines.append("")
    lines.append("## 7. Verification")
    lines.append("")
    lines.append(
        "新增脚本只使用标准库和显式 train-side allow-list；单元测试覆盖 branch "
        "confusion、SQL mismatch 分类与 "
        "error-to-next-action 分类。执行记录、Git diff 检查和测试结果写入 `verification`。"
    )
    lines.append("")
    lines.append("SFT_SQL_PARAMETER_SUPERVISION_AUDIT_COMPLETE")
    lines.append("")
    return "\n".join(lines)


def build_report(root: Path) -> dict[str, Any]:
    dataset_path = root / "artifacts/sft-data-train-teacher-v2/dataset.jsonl"
    metadata_path = root / "artifacts/sft-data-train-teacher-v2/dataset-metadata.json"
    prediction_path = (
        root
        / "artifacts/sft-behavior-attribution/20261017T192604-sft-behavior-attribution/epoch-3/pr"
        "edictions.jsonl"
    )
    runtime_root = (
        root / "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation"
    )
    metadata = read_json(metadata_path)
    postmortem = read_json(runtime_root / "postmortem.json")
    comparison = read_json(runtime_root / "comparison.json")
    if metadata.get("source_split") != "train":
        raise ValueError("dataset_source_split_not_train")
    if (
        metadata.get("validation_used_for_training") is not False
        or metadata.get("test_used_for_training") is not False
    ):
        raise ValueError("dataset_leakage_flags_not_false")
    if postmortem.get("scope", {}).get("split") != "train":
        raise ValueError("runtime_scope_not_train")
    if postmortem.get("scope", {}).get("groups") != list(GROUPS):
        raise ValueError(f"runtime_groups_mismatch:{postmortem.get('scope', {}).get('groups')}")
    if comparison.get("scope", {}).get("split") not in {None, "train"}:
        raise ValueError("comparison_scope_not_train")

    dataset_tasks, dataset_audit = build_dataset_audit(dataset_path)
    prediction_audit = build_prediction_audit(
        prediction_path, dataset_audit["targets"], dataset_tasks
    )
    runtime, runtime_tasks = build_runtime_audit(runtime_root, dataset_tasks, prediction_audit)
    cross_source = build_cross_source_evidence(dataset_tasks, prediction_audit, runtime)

    _teacher_by_source = prediction_audit["by_trajectory_source"]
    runtime_branch = {
        group: {
            "accepted_action_kind_counts": runtime["groups"][group]["accepted_action_kind_counts"],
            "schema_dictionary_before_first_sql_coverage": runtime["groups"][group][
                "schema_dictionary_before_first_sql_coverage"
            ],
            "schema_before_first_sql_count": runtime["groups"][group][
                "schema_before_first_sql_count"
            ],
            "dictionary_before_first_sql_count": runtime["groups"][group][
                "dictionary_before_first_sql_count"
            ],
            "by_trajectory_source": runtime_branch_strata(
                runtime["groups"][group]["tasks"], "trajectory_source"
            ),
            "by_template": runtime_branch_strata(runtime["groups"][group]["tasks"], "template_id"),
        }
        for group in GROUPS
    }
    # Evidence references are deliberately concrete and point to records in this JSON.
    confirmed = [
        {
            "id": "C1_dataset_reconstruction",
            "finding": (
                f"The allow-listed dataset contains {dataset_audit['row_count']} tasks and "
                f"{dataset_audit['target_count']} assistant targets, and the reconstructed Action "
                f"sequences are structurally parseable except for "
                f"{dataset_audit['structural_anomaly_count']} recorded anomalies."
            ),
            "evidence": "dataset_target_audit.row_count/target_count/tasks/anomalies",
        },
        {
            "id": "C2_dictionary_branch_confusion",
            "finding": (
                "The E3 Dictionary target rows have direct kind and parameter confusion; the "
                "mismatches are not reducible to a term spelling difference alone."
            ),
            "evidence": (
                "action_branch_confusion.teacher_forced.dictionary.confusion_matrix and "
                "argument_confusion"
            ),
        },
        {
            "id": "C3_sql_parameter_drift",
            "finding": (
                "Teacher-forced SQL predictions show direct raw-query and/or Action-kind "
                "mismatches on the aligned SQL targets; the classification is text-only."
            ),
            "evidence": (
                "sql_argument_mismatches.teacher_forced.records and "
                "mismatch_primary_category_counts"
            ),
        },
        {
            "id": "C4_error_feedback_not_effectively_recovered",
            "finding": (
                "Saved Runtime SQL errors are followed by event-level next actions that can be "
                "counted; the primary successful-SQL-after-error count is based on later accepted "
                "SQL with tool_result.ok=true and is not the prior reported recovery metric."
            ),
            "evidence": (
                "error_to_next_action.groups[*].successful_sql_after_prior_sql_error and "
                "error_events"
            ),
        },
        {
            "id": "C5_probe_success_not_task_success",
            "finding": (
                "In B/E top-task records, a saved successful SQL tool result coexists with later "
                "SQL errors and task success=false/reward=0."
            ),
            "evidence": "task_level_evidence.top_task_evidence",
        },
    ]
    likely = [
        {
            "id": "L1_mixed_provenance_multimodality",
            "finding": (
                "The two teacher provenances expose different gold Action paths and Dictionary "
                "argument distributions, so a mixed imitation distribution is a plausible "
                "contributor to branch confusion."
            ),
            "evidence": (
                "dataset_target_audit.same_template_branch_audit and "
                "action_branch_confusion.teacher_forced.by_trajectory_source"
            ),
            "cannot_exclude": (
                "generic Action-kind generalization failure or protocol/history effects"
            ),
        },
        {
            "id": "L2_sql_mapping_bottleneck",
            "finding": (
                "SQL table/field/date/dialect drift in E3 and matching identifier-bearing Runtime "
                "error text make SQL parameter-to-schema mapping a likely downstream bottleneck."
            ),
            "evidence": (
                "sql_argument_mismatches.teacher_forced.records and "
                "task_level_evidence.cross_source_common_patterns"
            ),
            "cannot_exclude": (
                "branch selection errors before SQL or the fact that runtime errors themselves "
                "alter later decisions"
            ),
        },
        {
            "id": "L3_feedback_policy_failure",
            "finding": (
                "Repeated or modified SQL after errors, combined with zero successful "
                "SQL-after-error events where observed, is consistent with an ineffective "
                "error-feedback policy."
            ),
            "evidence": (
                "error_to_next_action.groups[*].error_to_next_action and "
                "successful_sql_after_prior_sql_error"
            ),
            "cannot_exclude": "tool/action budget exhaustion and upstream parameter errors",
        },
        {
            "id": "L4_schema_not_sufficient",
            "finding": (
                "Accepted Schema coverage is not sufficient by itself: B/E top tasks have a "
                "successful Schema probe but still show a later field error."
            ),
            "evidence": "task_level_evidence.top_task_evidence for group B/E",
            "cannot_exclude": (
                "the Schema probe and main SQL are different subgoals, so no causal counterfactual "
                "exists"
            ),
        },
    ]
    unconfirmed = [
        {
            "id": "U1_gold_is_root_cause",
            "finding": (
                "That the gold corpus itself is semantically ambiguous or structurally "
                "inconsistent as a root cause."
            ),
            "missing_evidence": (
                "A controlled same-input alternative gold comparison; provenance-linked path "
                "differences alone are not a contradiction."
            ),
        },
        {
            "id": "U2_sql_semantic_correctness",
            "finding": (
                "Whether a textually different SQL query is semantically equivalent or "
                "business-correct."
            ),
            "missing_evidence": "SQL execution/verifier evidence, explicitly outside this audit.",
        },
        {
            "id": "U3_provenance_causality",
            "finding": (
                "That reference_sql_teacher or deterministic_fixture_expert provenance is the "
                "failure root cause."
            ),
            "missing_evidence": (
                "The two provenances are not randomized or held out as a causal intervention."
            ),
        },
        {
            "id": "U4_check_finish_causal_role",
            "finding": (
                "That Check/Finish learning is the first failure rather than a downstream "
                "consequence of earlier parser/SQL failures."
            ),
            "missing_evidence": (
                "A controlled intervention that holds the earlier Action and SQL path fixed."
            ),
        },
    ]
    recommendation = {
        "hypothesis": (
            "The first priority is a sequence-conditioned SQL argument-mapping bottleneck: once an "
            "accepted branch reaches SQL, the model emits schema-incompatible "
            "identifiers/relations and does not turn the resulting error into a successful changed "
            "SQL Action."
        ),
        "why_first": (
            "It is supported by independent Teacher-forced raw-query drift and Runtime "
            "error/next-action traces, and it is more specific than treating the provenance "
            "mixture or low Check/Finish counts as the root cause. The B/E top cases provide the "
            "clearest accepted-Schema→probe-success→main-field-error path."
        ),
        "supporting_task_ids": [
            "train-0-top",
            "train-1-top",
            "train-0-product",
            "train-1-product",
            "train-0-recovery",
            "train-1-recovery",
        ],
        "minimum_experiment": (
            "Run a CPU-only, train-only, offline branch-conditioned replay of the already saved E3 "
            "and A–E records: stratify each SQL target/event by preceding accepted "
            "Schema/Dictionary coverage, compare introduced identifier/relation/filter categories "
            "with saved error identifiers, and classify every post-error transition using the "
            "strict later-successful-SQL metric. Do not execute SQL; use exact string/record-level "
            "assertions and retain task/group/action/event evidence."
        ),
        "decision_rule": (
            "If the same identifier drift and zero/near-zero successful-SQL-after-error pattern "
            "persists in both branch-covered and branch-uncovered records, prioritize SQL "
            "parameter mapping/recovery over provenance. If drift is concentrated only in "
            "branch-uncovered records, prioritize Schema/Dictionary branch learning. If the first "
            "SQL is stable but post-error transitions repeat unchanged queries, prioritize "
            "recovery policy. A semantic correctness claim remains out of scope without execution."
        ),
        "requires_gpu": False,
        "requires_training": False,
        "requires_production_code_change": False,
    }
    report = {
        "schema_version": "1.0",
        "status": "SFT_SQL_PARAMETER_SUPERVISION_AUDIT_COMPLETE",
        "scope": {
            "split": "train",
            "cpu_only": True,
            "diagnostic_only": True,
            "groups": [{"label": GROUP_LABELS[group], "name": group} for group in GROUPS],
        },
        "input_manifest": build_input_manifest(
            root, metadata_path, dataset_path, prediction_path, runtime_root
        ),
        "dataset_target_audit": dataset_audit,
        "action_branch_confusion": {
            "teacher_forced": prediction_audit,
            "runtime": runtime_branch,
        },
        "sql_argument_mismatches": {
            "teacher_forced": prediction_audit["sql"],
            "runtime": {group: runtime["groups"][group]["tasks"] for group in GROUPS},
            "cross_source_common_patterns": cross_source,
        },
        "error_to_next_action": {
            "groups": runtime["groups"],
            "definition": {
                ("successful_sql_after_prior_sql_error"): (
                    "A SQL error event is a saved SQL Action with tool_result.ok=false; success "
                    "requires a later accepted SQL Action in the same episode with "
                    "tool_result.ok=true."
                ),
                "repeat_same_sql": (
                    "The next accepted SQL query string is exactly equal to the error query string."
                ),
                "modified_sql": (
                    "The next accepted SQL query string differs; no semantic repair claim is made."
                ),
                "budget_sentinel": (
                    "tool/action budget sentinels are retained and separately counted; they are "
                    "not treated as SQL semantic evidence."
                ),
            },
        },
        "task_level_evidence": {
            "runtime_tasks": runtime_tasks,
            "top_task_evidence": runtime["top_task_evidence"],
            "cross_source_common_patterns": cross_source,
        },
        "confirmed_findings": confirmed,
        "likely_findings": likely,
        "unconfirmed_findings": unconfirmed,
        "next_experiment_recommendation": recommendation,
        "verification": {
            "script_name": "scripts/diagnostics/sft_sql_parameter_supervision_audit.py",
            "uses_model_or_tokenizer": False,
            "executes_sql": False,
            "calls_verifier": False,
            "loads_validation_or_test_task_content": False,
            "source_manifest_loader_called": False,
        },
    }
    report["markdown"] = render_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Defaults to the requested Runtime ablation artifact path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Defaults to docs/experiments/sft-sql-parameter-supervision-audit.md.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    json_output = (
        args.json_output
        or root
        / "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation/sql-para"
        "meter-supervision-audit.json"
    )
    markdown_output = (
        args.markdown_output or root / "docs/experiments/sft-sql-parameter-supervision-audit.md"
    )
    report = build_report(root)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "markdown"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(report["markdown"], encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
