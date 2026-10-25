from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.grpo.aggregation import three_arm_comparison
from enterprise_agent_rl.grpo.io import read_json
from enterprise_agent_rl.grpo.saved_scoring import (
    SavedScoringError,
    _score_saved_case,
    _write_jsonl,
    validate_saved_target,
)
from enterprise_agent_rl.trajectories import RewardResult

TASK_IDS = [f"task-{index:02d}" for index in range(20)]


def _metrics(
    *,
    successes: set[str],
    task_ids: list[str] | None = None,
    qualified: int = 20,
    unknown: int = 0,
) -> dict:
    declared = list(task_ids or TASK_IDS)
    entries = [
        {"task_id": task_id, "success": task_id in successes}
        for task_id in declared[: 20 - unknown]
    ]
    return {
        "planned": 20,
        "planned_task_ids": declared,
        "scored": len(entries),
        "unknown": unknown,
        "qualified": qualified,
        "successes": sum(int(row["success"]) for row in entries),
        "success_rate": (
            sum(int(row["success"]) for row in entries) / 20 if unknown == 0 else None
        ),
        "protocol_fingerprint": "fixed-protocol",
        "entries": entries,
    }


def test_rl_unknown_does_not_gate_complete_n_minus_base_pair():
    base = _metrics(successes=set())
    n = _metrics(successes=set(TASK_IDS[:8]))
    rl = _metrics(successes=set(TASK_IDS[:8]), unknown=1)

    result = three_arm_comparison(base, n, rl, expected_count=20)

    assert result["status"] == "INCOMPLETE"
    assert result["planned_denominator"] == 20
    assert result["arms"]["Base"]["status"] == "COMPLETE"
    assert result["arms"]["N"]["status"] == "COMPLETE"
    assert result["arms"]["RL"]["status"] == "INCOMPLETE"
    assert result["pair_status"]["N_minus_Base"] == "COMPLETE"
    assert result["N_minus_Base"] == {
        "status": "COMPLETE",
        "planned_denominator": 20,
        "successes": 8,
        "success_rate": 0.4,
        "denominator": 20,
        "positive_tasks": TASK_IDS[:8],
        "negative_tasks": [],
        "reasons": [],
    }
    assert result["pair_status"]["RL_minus_N"] == "INCOMPLETE"
    assert result["RL_minus_N"]["successes"] is None
    assert result["RL_minus_Base"]["planned_denominator"] == 20
    assert result["pairs"][0]["N_minus_Base"] == 1


@pytest.mark.parametrize(
    "mutator, expected_reason",
    [
        (
            lambda value: value["planned_task_ids"].__setitem__(19, "other-task"),
            "scored_task_ids_do_not_match_planned",
        ),
        (lambda value: value.update(qualified=19), "qualification_incomplete"),
    ],
)
def test_pair_is_incomplete_for_task_or_qualification_conflicts(mutator, expected_reason):
    base = _metrics(successes=set())
    n = _metrics(successes=set(TASK_IDS[:8]))
    mutator(n)
    rl = _metrics(successes=set(TASK_IDS[:8]))

    result = three_arm_comparison(base, n, rl, expected_count=20)

    assert result["pair_status"]["N_minus_Base"] == "INCOMPLETE"
    assert any(expected_reason in reason for reason in result["N_minus_Base"]["reasons"])
    assert result["N_minus_Base"]["planned_denominator"] == 20
    assert result["N_minus_Base"]["successes"] is None


def test_saved_score_business_false_is_persisted_and_idempotent(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    trajectory_path = source_root / "trajectory.json"
    trajectory_path.write_text("{}\n", encoding="utf-8")
    source = {
        "source_root": source_root,
        "source_manifest_sha256": "manifest-sha",
        "trajectory_sha256": "trajectory-sha",
        "database_sha256": "database-sha",
        "trajectory_path": trajectory_path,
        "trajectory": SimpleNamespace(steps=[]),
        "policy_version": "frozen-policy-RL",
        "protocol_runtime": {},
        "private_task": {"task_id": "private-target"},
        "benchmark_task": SimpleNamespace(reference_sql="SELECT 1"),
        "qualification": {"status": "QUALIFIED", "attempted_model_calls": 20},
        "original_unknown": {
            "status": "unknown",
            "unknown_reason": "budget_stop",
            "verifier_called": False,
        },
        "historical_verifier_calls": 60,
        "historical_verifier_sql": 66,
        "source_budget_state": {"status": "BUDGET_STOP"},
    }
    database = tmp_path / "validation-1.sqlite3"
    database.write_bytes(b"fake-db-not-opened")
    calls: list[tuple] = []

    def fake_verifier(task, trajectory, database_path, settings):
        calls.append((task, trajectory, database_path, settings))
        return RewardResult(
            value=0.0,
            success=False,
            reason="business_false",
            components={"business": False},
        )

    output = tmp_path / "closeout"
    first, reused_first = _score_saved_case(
        output=output,
        source=source,
        database=database,
        verifier=fake_verifier,
    )
    second, reused_second = _score_saved_case(
        output=output,
        source=source,
        database=database,
        verifier=fake_verifier,
    )

    assert first["status"] == "scored"
    assert first["score_row"]["success"] is False
    assert reused_first is False
    assert reused_second is True
    assert second == first
    assert len(calls) == 1
    ledger = read_json(output / "call-ledger.json")
    assert ledger["new_verifier_calls"] == 1
    assert ledger["new_verifier_sql"] == 0
    assert len(ledger["attempts"]) == 1
    assert (
        json.loads((output / "verifier" / "attempt-01" / "verifier-output.json").read_text())[
            "reward_result"
        ]["success"]
        is False
    )


def test_merged_jsonl_replay_ignores_key_order_but_rejects_data_conflict(tmp_path):
    path = tmp_path / "merged.jsonl"
    _write_jsonl(path, [{"b": 2, "a": 1}])
    _write_jsonl(path, [{"a": 1, "b": 2}])
    with pytest.raises(SavedScoringError, match="immutable_jsonl_conflict"):
        _write_jsonl(path, [{"a": 1, "b": 3}])


def test_saved_scoring_rejects_wrong_scope_before_verifier(tmp_path):
    with pytest.raises(SavedScoringError, match="saved_scoring_scope"):
        validate_saved_target(
            source_run=tmp_path,
            private_task_plan=tmp_path / "not-read.json",
            task_id="different-task",
        )


def test_saved_scoring_rejects_persisted_trajectory_identity_conflict(tmp_path):
    output = tmp_path / "closeout"
    output.mkdir()
    (output / "supplemental-score.json").write_text(
        json.dumps(
            {
                "arm": "RL",
                "task_id": "validation-1-monthly",
                "status": "scored",
                "source_manifest_sha256": "manifest-sha",
                "trajectory_sha256": "wrong-trajectory-sha",
                "score_row": {"status": "scored"},
            }
        ),
        encoding="utf-8",
    )
    source = {
        "source_manifest_sha256": "manifest-sha",
        "trajectory_sha256": "correct-trajectory-sha",
    }

    with pytest.raises(SavedScoringError, match="supplement_trajectory_mismatch"):
        _score_saved_case(
            output=output,
            source=source,
            database=tmp_path / "unused.sqlite3",
            verifier=lambda *_args: pytest.fail("verifier must not be called"),
        )
