"""CPU-only tests for the saved-artifact Runtime ablation postmortem."""

from __future__ import annotations

from pathlib import Path

from scripts.diagnostics.sft_runtime_ablation_postmortem import (
    aggregate_records,
    build_postmortem,
)

ROOT = Path(__file__).resolve().parents[1]


def _record(**overrides):
    record = {
        "success": False,
        "first_action_legal": True,
        "pre_sql_schema_dictionary": [],
        "sql_attempts": 2,
        "executable_sql_count": 0,
        "sql_error_count": 2,
        "recovery_count": 2,
        "reported_recovery_success_count": 1,
        "actual_sql_recovery_count": 0,
        "repeated_sql": 1,
        "check_present": True,
        "finish_present": False,
        "budget_exhausted": False,
        "tool_calls": 2,
        "latency_s": 1.0,
        "input_tokens": 10,
        "output_tokens": 4,
        "reward": 0.0,
        "invalid_json": 0,
        "invalid_envelope": 0,
        "invalid_arguments": 0,
        "format_retry_count": 0,
        "parser_attempts": 2,
        "parser_acceptances": 2,
        "sql_error_types": {"no such column: x": 2},
        "action_kind_counts": {"plan": 1, "sql": 2, "check": 1},
        "sql_events": [
            {"ok": False, "error": "no such column: x"},
            {"ok": False, "error": "no such column: x"},
        ],
    }
    record.update(overrides)
    return record


def test_reported_recovery_is_not_sql_recovery():
    metrics = aggregate_records([_record()])
    assert metrics["recovery_count"] == 2
    assert metrics["reported_recovery_success_count"] == 1
    assert metrics["actual_sql_recovery_count"] == 0
    assert metrics["recovery_success_rate"] == 0.5


def test_sql_rates_use_sql_action_denominator_and_exact_repetition():
    metrics = aggregate_records(
        [
            _record(
                sql_attempts=4,
                executable_sql_count=1,
                sql_error_count=3,
                recovery_count=3,
                repeated_sql=2,
            )
        ]
    )
    assert metrics["sql_executable_rate"] == 0.25
    assert metrics["sql_error_rate"] == 0.75
    assert metrics["sql_repeat_rate"] == 0.5


def test_saved_artifact_postmortem_matches_all_group_summaries():
    root = ROOT / "artifacts/sft-runtime-ablation/20261017T130311Z-sft-runtime-protocol-ablation"
    postmortem = build_postmortem(root)
    assert postmortem["status"] == "SFT_RUNTIME_POSTMORTEM_COMPLETE"
    assert all(
        check["all_scalar_metrics_match"] for check in postmortem["metric_checks"].values()
    )
    assert postmortem["b_initial_observation_analysis"]["schema_before_sql_tasks"] == [
        "train-0-top",
        "train-1-top",
    ]
    assert postmortem["e_full_alignment_analysis"]["actual_sql_recovery_task_count"] == 0
