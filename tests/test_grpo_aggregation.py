from __future__ import annotations

import pytest

from enterprise_agent_rl.grpo.aggregation import EvidenceError, aggregate_source
from enterprise_agent_rl.grpo.io import append_jsonl


def _trajectory(success: bool) -> dict:
    return {
        "steps": [
            {"action": {"kind": "sql"}, "tool_result": {"ok": True, "worker_calls": 1}},
            {"action": {"kind": "sql"}, "tool_result": {"ok": False, "error": "business_failed"}},
            {
                "action": {"kind": "sql"},
                "tool_result": {"ok": False, "error": {"code": "budget_rejected"}},
            },
        ],
        "success": success,
    }


def _row(task_id: str, *, status: str, success: bool = True, trajectory=None) -> dict:
    value = {
        "record_type": "RESULT",
        "task_id": task_id,
        "episode_id": f"episode-{task_id}",
        "status": status,
        "success": success,
        "reward_value": 1.0 if success else 0.0,
    }
    if status == "scored":
        value["raw_saved"] = True
        if trajectory is not None:
            value["trajectory"] = trajectory
    return value


def test_verifier_index_deduplicates_raw_and_scored_and_preserves_errors(tmp_path):
    index = tmp_path / "formal.jsonl"
    append_jsonl(index, _row("task-a", status="raw_saved"))
    append_jsonl(index, _row("task-a", status="raw_saved"))
    append_jsonl(index, _row("task-a", status="scored", trajectory=_trajectory(True)))
    append_jsonl(index, _row("task-a", status="scored", trajectory=_trajectory(True)))
    append_jsonl(index, _row("task-b", status="raw_saved", success=False))
    append_jsonl(
        index, _row("task-b", status="scored", success=False, trajectory=_trajectory(False))
    )

    result = aggregate_source(
        source_root=tmp_path,
        verifier_index=index,
        identity_field="task_id",
        expected_count=2,
        source_kind="synthetic",
        fixture=True,
    )
    assert result["fixture"] is True
    assert result["scored"] == 2
    assert result["unknown"] == 0
    assert result["successes"] == 1
    assert result["sql_action_count"] == 6
    assert result["sql_ok_count"] == 2
    assert result["raw_scored_duplicate_keys_not_counted"] == ["task-a", "task-b"]
    assert result["sql_error_types"]["business_failed"] == 2
    assert result["sql_error_types"]["budget_rejected"] == 2
    assert any(error == {"code": "budget_rejected"} for error in result["sql_errors"])
    assert any(error == {"message": "business_failed"} for error in result["sql_errors"])


def test_missing_evidence_is_unknown_and_conflicts_are_rejected(tmp_path):
    missing_index = tmp_path / "missing.jsonl"
    append_jsonl(missing_index, _row("task-a", status="raw_saved"))
    append_jsonl(missing_index, _row("task-a", status="scored"))
    missing = aggregate_source(
        source_root=tmp_path,
        verifier_index=missing_index,
        identity_field="task_id",
        expected_count=1,
        source_kind="synthetic",
    )
    assert missing["scored"] == 0
    assert missing["unknown"] == 1
    assert missing["success_rate"] is None
    assert missing["missing_or_invalid_evidence"] == ["task-a"]

    unknown_index = tmp_path / "explicit-unknown.jsonl"
    append_jsonl(unknown_index, _row("task-a", status="raw_saved"))
    unknown = _row("task-a", status="unknown")
    unknown["trajectory"] = _trajectory(True)
    append_jsonl(unknown_index, unknown)
    explicit_unknown = aggregate_source(
        source_root=tmp_path,
        verifier_index=unknown_index,
        identity_field="task_id",
        expected_count=1,
        source_kind="synthetic",
    )
    assert explicit_unknown["scored"] == 0
    assert explicit_unknown["unknown"] == 1

    conflict_index = tmp_path / "conflict.jsonl"
    append_jsonl(conflict_index, _row("task-a", status="raw_saved"))
    append_jsonl(
        conflict_index, _row("task-a", status="scored", success=True, trajectory=_trajectory(True))
    )
    append_jsonl(
        conflict_index,
        _row("task-a", status="scored", success=False, trajectory=_trajectory(False)),
    )
    with pytest.raises(EvidenceError, match="conflicting_duplicate"):
        aggregate_source(
            source_root=tmp_path,
            verifier_index=conflict_index,
            identity_field="task_id",
            expected_count=1,
            source_kind="synthetic",
        )


def test_missing_file_and_path_escape_do_not_become_success(tmp_path):
    missing_index = tmp_path / "missing-file.jsonl"
    row = _row("task-a", status="scored")
    row["trajectory_path"] = "does-not-exist.json"
    append_jsonl(missing_index, _row("task-a", status="raw_saved"))
    append_jsonl(missing_index, row)
    result = aggregate_source(
        source_root=tmp_path,
        verifier_index=missing_index,
        identity_field="task_id",
        expected_count=1,
        source_kind="synthetic",
    )
    assert result["unknown"] == 1
    assert result["successes"] == 0

    escape_index = tmp_path / "escape.jsonl"
    escape = _row("task-a", status="scored")
    escape["trajectory_path"] = "../outside.json"
    append_jsonl(escape_index, _row("task-a", status="raw_saved"))
    append_jsonl(escape_index, escape)
    with pytest.raises(ValueError, match="evidence_path_escape"):
        aggregate_source(
            source_root=tmp_path,
            verifier_index=escape_index,
            identity_field="task_id",
            expected_count=1,
            source_kind="synthetic",
        )
