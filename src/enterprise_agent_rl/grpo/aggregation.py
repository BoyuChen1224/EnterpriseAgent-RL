"""Verifier-indexed evidence aggregation for real and synthetic sources.

The aggregator never discovers trajectories with glob/rglob.  A formal score
row must identify the one trajectory it scored; raw_saved rows are accepted as
the durable predecessor of that scored row but are never counted themselves.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, canonical_hash, read_jsonl, safe_relative_path


class EvidenceError(RuntimeError):
    """Formal score evidence is missing, ambiguous, or inconsistent."""


def _error_name(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "unknown_sql_error")
    if isinstance(error, str):
        return error
    return "unknown_sql_error"


def _error_record(error: Any) -> dict[str, Any]:
    if isinstance(error, dict):
        return {str(key): value for key, value in error.items()}
    if isinstance(error, str):
        return {"message": error}
    return {"value": error}


def _status_key(row: dict[str, Any], identity_field: str) -> str:
    value = row.get(identity_field)
    if value is None and identity_field == "episode_id":
        value = row.get("verifier_record_id") or row.get("task_id")
    if value is None:
        value = row.get("verifier_record_id") or row.get("episode_id") or row.get("task_id")
    if not isinstance(value, str | int) or not str(value):
        raise EvidenceError("formal_row_missing_unique_identity")
    return str(value)


def _trajectory_from_row(
    row: dict[str, Any],
    source_root: Path,
    *,
    allowed_source_roots: set[Path] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    direct = row.get("trajectory")
    if isinstance(direct, dict):
        return direct, None
    path_value = row.get("trajectory_path")
    if path_value is None:
        path_value = row.get("scored_trajectory_path")
    if not isinstance(path_value, str) or not path_value:
        return None, None
    trajectory_root = source_root
    declared_root = row.get("trajectory_source_root")
    if declared_root is not None:
        if not isinstance(declared_root, str) or not declared_root:
            raise EvidenceError("trajectory_source_root_invalid")
        declared_path = Path(declared_root)
        if not declared_path.is_absolute():
            raise EvidenceError("trajectory_source_root_must_be_absolute")
        trajectory_root = declared_path.resolve()
    if allowed_source_roots is not None and trajectory_root not in allowed_source_roots:
        raise EvidenceError(f"trajectory_source_root_not_allowed:{trajectory_root}")
    path = safe_relative_path(trajectory_root, path_value)
    if not path.is_file():
        raise EvidenceError(f"trajectory_missing:{path_value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"trajectory_unreadable:{path_value}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"trajectory_not_object:{path_value}")
    return value, path_value


def _trajectory_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    value = trajectory.get("steps")
    if value is None and isinstance(trajectory.get("trajectory"), dict):
        value = trajectory["trajectory"].get("steps")
    if not isinstance(value, list):
        raise EvidenceError("trajectory_steps_missing")
    if any(not isinstance(step, dict) for step in value):
        raise EvidenceError("trajectory_step_not_object")
    return value


def _step_sql_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    attempts = successes = worker_calls = budget_rejections = 0
    worker_known = False
    errors: Counter[str] = Counter()
    error_values: list[dict[str, Any]] = []
    for step in _trajectory_steps(trajectory):
        action = step.get("action") or {}
        action_kind = step.get("tool_name") or action.get("kind")
        if action_kind != "sql":
            continue
        attempts += 1
        result = step.get("tool_result") or step.get("result") or {}
        if not isinstance(result, dict):
            raise EvidenceError("sql_tool_result_not_object")
        worker_value = result.get(
            "worker_calls", result.get("worker_count", step.get("worker_calls"))
        )
        if worker_value is not None:
            worker_known = True
            worker_calls += int(worker_value)
        elif result.get("worker_invoked") is True or result.get("worker") is not None:
            worker_known = True
            worker_calls += 1
        ok = bool(result.get("ok"))
        if ok:
            successes += 1
            continue
        error = result.get("error") or result.get("stderr") or result.get("message")
        name = _error_name(error)
        errors[name] += 1
        error_values.append(_error_record(error))
        if "budget" in name.lower() or "budget" in json.dumps(error, ensure_ascii=False).lower():
            budget_rejections += 1
    return {
        "sql_action_count": attempts,
        "sql_ok_count": successes,
        "worker_count": worker_calls,
        "worker_count_known": worker_known,
        "budget_rejection_count": budget_rejections,
        "sql_error_types": dict(sorted(errors.items())),
        "sql_errors": error_values,
    }


def _deduplicate_rows(
    rows: list[dict[str, Any]], identity_field: str
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    raw: dict[str, dict[str, Any]] = {}
    scored: dict[str, dict[str, Any]] = {}
    unknown: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type", "RESULT") != "RESULT":
            continue
        status = row.get("status")
        if status not in {"raw_saved", "scored", "unknown"}:
            continue
        key = _status_key(row, identity_field)
        target = {"raw_saved": raw, "scored": scored, "unknown": unknown}[status]
        previous = target.get(key)
        if previous is not None and canonical_hash(previous) != canonical_hash(row):
            raise EvidenceError(f"conflicting_duplicate:{key}:{status}")
        target[key] = row
    return raw, scored, unknown


def aggregate_source(
    *,
    source_root: Path,
    verifier_index: Path,
    identity_field: str,
    expected_count: int,
    source_kind: str,
    fixture: bool = False,
    planned_task_ids: list[str] | None = None,
    allowed_trajectory_roots: set[Path] | None = None,
) -> dict[str, Any]:
    """Aggregate one formal index with missing evidence represented as unknown."""

    source_root = source_root.resolve()
    verifier_index = verifier_index.resolve()
    if not verifier_index.is_file():
        raise EvidenceError(f"verifier_index_missing:{verifier_index}")
    rows = read_jsonl(verifier_index)
    normalized_planned_task_ids: list[str] | None = None
    if planned_task_ids is not None:
        normalized_planned_task_ids = [str(task_id) for task_id in planned_task_ids]
        if len(normalized_planned_task_ids) != len(set(normalized_planned_task_ids)):
            raise EvidenceError("planned_task_ids_duplicate")
        if len(normalized_planned_task_ids) != expected_count:
            raise EvidenceError("planned_task_ids_count_mismatch")
    raw_rows, scored_rows, formal_unknown = _deduplicate_rows(rows, identity_field)
    if set(scored_rows) & set(formal_unknown):
        raise EvidenceError("conflicting_status_scored_and_unknown")
    duplicate_keys = set(raw_rows) & set(scored_rows)
    entries: list[dict[str, Any]] = []
    sql_totals = Counter()
    sql_errors: Counter[str] = Counter()
    sql_error_values: list[dict[str, Any]] = []
    infra_errors: list[dict[str, Any]] = []
    worker_values: list[int] = []
    non_sql_worker_values: list[int] = []
    verifier_sql_values: list[int] = []
    missing: list[str] = []
    for key, row in sorted(scored_rows.items()):
        if row.get("raw_saved") is False and key not in raw_rows:
            raise EvidenceError(f"scored_without_raw_saved:{key}")
        try:
            trajectory, path = _trajectory_from_row(
                row,
                source_root,
                allowed_source_roots=allowed_trajectory_roots,
            )
            if trajectory is None:
                raise EvidenceError(f"scored_trajectory_reference_missing:{key}")
            sql = _step_sql_metrics(trajectory)
            for counter_key in (
                "sql_action_count",
                "sql_ok_count",
                "worker_count",
                "budget_rejection_count",
            ):
                sql_totals[counter_key] += int(sql[counter_key])
            sql_errors.update(sql["sql_error_types"])
            sql_error_values.extend(sql["sql_errors"])
            if row.get("runtime_worker_sql_count") is not None:
                worker_values.append(int(row["runtime_worker_sql_count"]))
            elif sql["worker_count_known"]:
                worker_values.append(int(sql["worker_count"]))
            if row.get("runtime_worker_non_sql_count") is not None:
                non_sql_worker_values.append(int(row["runtime_worker_non_sql_count"]))
            if row.get("verifier_sql_count") is not None:
                verifier_sql_values.append(int(row["verifier_sql_count"]))
            entries.append(
                {
                    "identity": key,
                    "task_id": row.get("task_id"),
                    "episode_id": row.get("episode_id"),
                    "status": "scored",
                    "success": bool(
                        row.get("success", row.get("reward", {}).get("success", False))
                    ),
                    "reward_value": row.get("reward_value", row.get("reward", {}).get("value")),
                    "trajectory_path": path,
                    "sql": sql,
                    "runtime_worker_sql_count": row.get(
                        "runtime_worker_sql_count", sql["worker_count"]
                        if sql["worker_count_known"]
                        else None
                    ),
                    "runtime_worker_non_sql_count": row.get("runtime_worker_non_sql_count"),
                    "verifier_sql_count": row.get("verifier_sql_count"),
                }
            )
        except EvidenceError as exc:
            missing.append(key)
            infra_errors.append({"identity": key, "type": type(exc).__name__, "message": str(exc)})
    for key, row in sorted(formal_unknown.items()):
        infra_errors.append(
            {
                "identity": key,
                "type": "formal_unknown",
                "reason": row.get("unknown_reason"),
                "error": row.get("error"),
            }
        )
    observed = set(scored_rows) | set(formal_unknown)
    if expected_count < 0:
        raise ValueError("expected_count_must_be_nonnegative")
    missing_count = max(0, expected_count - len(observed))
    if missing_count:
        missing.extend([f"<missing-{index}>" for index in range(missing_count)])
    scored_count = len(entries)
    unknown_count = expected_count - scored_count
    if unknown_count < 0:
        raise EvidenceError(f"too_many_formal_records:{expected_count}:{scored_count}")
    successes = sum(int(entry["success"]) for entry in entries)
    worker_count = (
        sum(worker_values)
        if len(worker_values) == scored_count and scored_count
        else None
    )
    runtime_worker_non_sql_count = (
        sum(non_sql_worker_values)
        if len(non_sql_worker_values) == scored_count and scored_count
        else None
    )
    verifier_sql_count = (
        sum(verifier_sql_values)
        if len(verifier_sql_values) == scored_count and scored_count
        else None
    )
    result = {
        "schema_version": "grpo-aggregated-source-v1",
        "source_kind": source_kind,
        "fixture": bool(fixture),
        "verifier_index": str(verifier_index),
        "identity_field": identity_field,
        "planned": expected_count,
        "planned_task_ids": normalized_planned_task_ids,
        "attempted": len(observed),
        "observed_unique": len(observed),
        "scored": scored_count,
        "unknown": unknown_count,
        # Qualification is a model-evidence property, not a synonym for a
        # successful business score.  Keep a qualified-but-unscored technical
        # unknown visible in this count instead of silently dropping it.
        "qualified": sum(
            row.get("qualification", {}).get("status") == "QUALIFIED"
            for row in [*scored_rows.values(), *formal_unknown.values()]
        ),
        "verifier_called": sum(bool(row.get("verifier_called")) for row in scored_rows.values()),
        "attempted_model_calls": sum(
            int(row.get("qualification", {}).get("attempted_model_calls", 0))
            for row in [*scored_rows.values(), *formal_unknown.values()]
        ),
        "completed_model_calls": sum(
            int(row.get("qualification", {}).get("completed_model_calls", 0))
            for row in [*scored_rows.values(), *formal_unknown.values()]
        ),
        "incomplete_model_calls": sum(
            int(row.get("qualification", {}).get("incomplete_model_calls", 0))
            for row in [*scored_rows.values(), *formal_unknown.values()]
        ),
        "successes": successes,
        "success_rate": successes / expected_count
        if expected_count and unknown_count == 0
        else None,
        "sql_action_count": int(sql_totals["sql_action_count"]),
        "sql_ok_count": int(sql_totals["sql_ok_count"]),
        "worker_count": worker_count,
        "runtime_worker_sql_count": worker_count,
        "runtime_worker_non_sql_count": runtime_worker_non_sql_count,
        "verifier_sql_count": verifier_sql_count,
        "budget_rejection_count": int(sql_totals["budget_rejection_count"]),
        "sql_error_types": dict(sorted(sql_errors.items())),
        "sql_errors": sql_error_values,
        "infra_resource_errors": infra_errors,
        "missing_or_invalid_evidence": sorted(set(missing)),
        "raw_saved_keys": sorted(raw_rows),
        "scored_keys": sorted(scored_rows),
        "raw_scored_duplicate_keys_not_counted": sorted(duplicate_keys),
        "entries": entries,
        "evidence_status": "COMPLETE" if unknown_count == 0 else "INCOMPLETE",
    }
    return result


def paired_comparison(
    n: dict[str, Any], rl: dict[str, Any], *, expected_count: int
) -> dict[str, Any]:
    n_rows = {str(row.get("task_id")): row for row in n.get("entries", [])}
    rl_rows = {str(row.get("task_id")): row for row in rl.get("entries", [])}
    task_ids = sorted(set(n_rows) | set(rl_rows))
    pairs = []
    for task_id in task_ids:
        n_row = n_rows.get(task_id)
        rl_row = rl_rows.get(task_id)
        complete = n_row is not None and rl_row is not None
        pairs.append(
            {
                "task_id": task_id,
                "N_success": n_row["success"] if complete else None,
                "RL_success": rl_row["success"] if complete else None,
                "RL_minus_N": int(rl_row["success"]) - int(n_row["success"]) if complete else None,
            }
        )
    complete = n.get("unknown") == 0 and rl.get("unknown") == 0 and len(pairs) == expected_count
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "denominator": expected_count if complete else None,
        "N_successes": n.get("successes") if complete else None,
        "RL_successes": rl.get("successes") if complete else None,
        "RL_successes_minus_N_successes": rl.get("successes", 0) - n.get("successes", 0)
        if complete
        else None,
        "N_success_rate": n.get("success_rate") if complete else None,
        "RL_success_rate": rl.get("success_rate") if complete else None,
        "RL_minus_N_success_rate": (rl.get("success_rate") - n.get("success_rate"))
        if complete
        else None,
        "pairs": pairs,
    }


def three_arm_comparison(
    base: dict[str, Any], n: dict[str, Any], rl: dict[str, Any], *, expected_count: int
) -> dict[str, Any]:
    """Compare Base/N/RL with independent arm and pair completeness gates.

    The three-arm status remains a global status, but a pair only depends on
    its two arms.  In particular, a missing RL row cannot hide a complete
    N-minus-Base result.  Unknown and unqualified rows never become false
    successes and every emitted denominator remains the declared panel size.
    """

    by_arm = {"Base": base, "N": n, "RL": rl}
    rows_by_arm = {
        arm: {str(row.get("task_id")): row for row in metrics.get("entries", [])}
        for arm, metrics in by_arm.items()
    }
    task_ids = sorted(set().union(*(set(rows) for rows in rows_by_arm.values())))

    def _declared_task_ids(metrics: dict[str, Any], arm: str) -> list[str] | None:
        value = metrics.get("planned_task_ids")
        if value is None:
            value = metrics.get("task_ids")
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return [f"<invalid-{arm}-task-ids>"]
        return list(value)

    def _arm_contract(arm: str) -> dict[str, Any]:
        metrics = by_arm[arm]
        entries = metrics.get("entries", [])
        entry_ids = [str(row.get("task_id")) for row in entries if isinstance(row, dict)]
        reasons: list[str] = []
        if len(entry_ids) != len(set(entry_ids)):
            reasons.append("duplicate_scored_task_id")
        declared = _declared_task_ids(metrics, arm)
        if declared is not None:
            if len(declared) != len(set(declared)):
                reasons.append("duplicate_planned_task_id")
            if len(declared) != expected_count:
                reasons.append("planned_task_id_count_mismatch")
            if set(entry_ids) != set(declared):
                reasons.append("scored_task_ids_do_not_match_planned")
            task_set = set(declared)
        else:
            task_set = set(entry_ids)
        if metrics.get("planned") != expected_count:
            reasons.append("planned_count_mismatch")
        if metrics.get("scored") != expected_count:
            reasons.append("scored_count_incomplete")
        if metrics.get("unknown") != 0:
            reasons.append("unknown_rows_present")
        if metrics.get("qualified") != expected_count:
            reasons.append("qualification_incomplete")
        if len(entries) != expected_count:
            reasons.append("entry_count_mismatch")
        return {
            "status": "COMPLETE" if not reasons else "INCOMPLETE",
            "task_ids": sorted(task_set),
            "protocol_fingerprint": metrics.get("protocol_fingerprint"),
            "reasons": reasons,
        }

    contracts = {arm: _arm_contract(arm) for arm in by_arm}

    def _pair_contract(left: str, right: str) -> dict[str, Any]:
        left_contract = contracts[left]
        right_contract = contracts[right]
        reasons = [
            f"{left}:{reason}" for reason in left_contract["reasons"]
        ] + [f"{right}:{reason}" for reason in right_contract["reasons"]]
        if left_contract["task_ids"] != right_contract["task_ids"]:
            reasons.append("task_id_set_mismatch")
        left_protocol = left_contract["protocol_fingerprint"]
        right_protocol = right_contract["protocol_fingerprint"]
        if (left_protocol is None) != (right_protocol is None):
            reasons.append("protocol_fingerprint_missing")
        elif left_protocol is not None and left_protocol != right_protocol:
            reasons.append("protocol_fingerprint_mismatch")
        return {
            "status": "COMPLETE" if not reasons else "INCOMPLETE",
            "planned_denominator": expected_count,
            "reasons": reasons,
        }

    pair_contracts = {
        "N_minus_Base": _pair_contract("N", "Base"),
        "RL_minus_N": _pair_contract("RL", "N"),
        "RL_minus_Base": _pair_contract("RL", "Base"),
    }

    pairs: list[dict[str, Any]] = []
    for task_id in task_ids:
        rows = {arm: rows_by_arm[arm].get(task_id) for arm in by_arm}
        item: dict[str, Any] = {
            "task_id": task_id,
            "Base_success": rows["Base"]["success"] if rows["Base"] else None,
            "N_success": rows["N"]["success"] if rows["N"] else None,
            "RL_success": rows["RL"]["success"] if rows["RL"] else None,
        }
        for key, left, right in (
            ("N_minus_Base", "N", "Base"),
            ("RL_minus_N", "RL", "N"),
            ("RL_minus_Base", "RL", "Base"),
        ):
            item[key] = (
                int(rows[left]["success"]) - int(rows[right]["success"])
                if pair_contracts[key]["status"] == "COMPLETE"
                and rows[left] is not None
                and rows[right] is not None
                else None
            )
        pairs.append(item)

    def _delta(left: str, right: str) -> dict[str, Any]:
        key = f"{left}_minus_{right}"
        contract = pair_contracts[key]
        left_metrics = by_arm[left]
        right_metrics = by_arm[right]
        if contract["status"] != "COMPLETE":
            return {
                "status": "INCOMPLETE",
                "planned_denominator": expected_count,
                "successes": None,
                "success_rate": None,
                "denominator": None,
                "positive_tasks": [],
                "negative_tasks": [],
                "reasons": contract["reasons"],
            }
        transitions = [row for row in pairs if row.get(key) is not None]
        return {
            "status": "COMPLETE",
            "planned_denominator": expected_count,
            "successes": int(left_metrics["successes"]) - int(right_metrics["successes"]),
            "success_rate": float(left_metrics["success_rate"])
            - float(right_metrics["success_rate"]),
            "denominator": expected_count,
            "positive_tasks": [row["task_id"] for row in transitions if row[key] > 0],
            "negative_tasks": [row["task_id"] for row in transitions if row[key] < 0],
            "reasons": [],
        }

    arm_complete = all(value["status"] == "COMPLETE" for value in contracts.values())
    complete = arm_complete and all(
        value["status"] == "COMPLETE" for value in pair_contracts.values()
    )

    return {
        "schema_version": "grpo-three-arm-comparison-v2",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "planned_denominator": expected_count,
        "denominator": expected_count if complete else None,
        "arms": {
            arm: {
                "status": contracts[arm]["status"],
                "scored": metrics.get("scored"),
                "unknown": metrics.get("unknown"),
                "qualified": metrics.get("qualified"),
                "successes": metrics.get("successes")
                if contracts[arm]["status"] == "COMPLETE"
                else None,
                "success_rate": metrics.get("success_rate")
                if contracts[arm]["status"] == "COMPLETE"
                else None,
                "planned_task_ids": contracts[arm]["task_ids"],
                "reasons": contracts[arm]["reasons"],
            }
            for arm, metrics in by_arm.items()
        },
        "N_minus_Base": _delta("N", "Base"),
        "RL_minus_N": _delta("RL", "N"),
        "RL_minus_Base": _delta("RL", "Base"),
        "pair_status": {
            key: value["status"] for key, value in pair_contracts.items()
        },
        "arm_contracts": contracts,
        "pairs": pairs,
    }


def replay_t021(t021_root: Path, output_path: Path) -> dict[str, Any]:
    """Replay only T-021's public formal indexes; never execute old code."""

    root = t021_root.resolve()
    train_rounds = []
    for round_index in range(2):
        train_rounds.append(
            aggregate_source(
                source_root=root,
                verifier_index=root
                / "rounds"
                / f"r{round_index}"
                / "rollouts"
                / "formal-scores.jsonl",
                identity_field="episode_id",
                expected_count=16,
                source_kind="T021-E024-real-train",
            )
        )
    n = aggregate_source(
        source_root=root,
        verifier_index=root / "validation" / "N" / "episode-results.jsonl",
        identity_field="task_id",
        expected_count=20,
        source_kind="T021-E024-real-validation-N",
    )
    rl = aggregate_source(
        source_root=root,
        verifier_index=root / "validation" / "RL" / "episode-results.jsonl",
        identity_field="task_id",
        expected_count=20,
        source_kind="T021-E024-real-validation-RL",
    )
    train_total = {
        "planned": sum(row["planned"] for row in train_rounds),
        "scored": sum(row["scored"] for row in train_rounds),
        "unknown": sum(row["unknown"] for row in train_rounds),
        "sql_action_count": sum(row["sql_action_count"] for row in train_rounds),
        "sql_ok_count": sum(row["sql_ok_count"] for row in train_rounds),
        # T-021's delivered formal rows do not bind every Runtime worker
        # audit. Preserve that uncertainty rather than turning it into zero.
        "worker_count": (
            sum(row["worker_count"] for row in train_rounds)
            if all(row["worker_count"] is not None for row in train_rounds)
            else None
        ),
        "budget_rejection_count": sum(row["budget_rejection_count"] for row in train_rounds),
    }
    result = {
        "schema_version": "grpo-t021-replayed-metrics-v1",
        "status": "PASS"
        if n["unknown"] == 0 and rl["unknown"] == 0 and train_total["unknown"] == 0
        else "INCOMPLETE",
        "source": {
            "root": str(root),
            "executed_historical_python": False,
            "read_databases": False,
            "read_weights": False,
            "fixture_values_used": False,
        },
        "train": {"rounds": train_rounds, "total": train_total},
        "N": n,
        "RL": rl,
        "primary_comparison": paired_comparison(n, rl, expected_count=20),
        "expected": {
            "N": {
                "scored": 20,
                "unknown": 0,
                "successes": 4,
                "sql_action_count": 32,
                "sql_ok_count": 15,
            },
            "RL": {
                "scored": 20,
                "unknown": 0,
                "successes": 4,
                "sql_action_count": 20,
                "sql_ok_count": 16,
            },
        },
        "fixture_exclusion": {
            "fixture": True,
            "reason": (
                "T021 fixtures/aggregation.json is not a real source "
                "and is not read into these metrics"
            ),
        },
    }
    atomic_write_json(output_path, result)
    return result
