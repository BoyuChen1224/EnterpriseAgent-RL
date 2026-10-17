#!/usr/bin/env python3
"""Compare the frozen Base validation report with one SFT validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "task_success_rate",
    "executable_sql_rate",
    "verifier_success_rate",
    "average_tool_calls",
    "average_invalid_tool_calls",
    "average_sql_attempts",
    "average_recovery_count",
    "tool_error_rate",
    "average_latency",
    "average_input_tokens",
    "average_output_tokens",
    "average_total_tokens",
    "average_reward",
    "median_reward",
    "min_reward",
    "max_reward",
)


def _load_report(root: Path) -> dict[str, Any]:
    path = root / "benchmark-report.json"
    if not path.is_file():
        raise SystemExit(f"missing_benchmark_report:{path}")
    return json.loads(path.read_text())


def _metric(report: dict[str, Any], name: str) -> Any:
    metrics = report.get("metrics", {})
    if name in metrics:
        return metrics[name]
    if name == "verifier_success_rate":
        return metrics.get("verifier_success_rate")
    return None


def _delta(base: Any, sft: Any) -> Any:
    if isinstance(base, int | float) and isinstance(sft, int | float):
        return sft - base
    return None


def _validate_scope(base: dict[str, Any], sft: dict[str, Any]) -> None:
    for label, report in (("base", base), ("sft", sft)):
        if report.get("split") != "validation":
            raise SystemExit(f"{label}_not_validation")
        if report.get("task_count") != 20:
            raise SystemExit(f"{label}_task_count_not_20")
        if report.get("test_split_accessed") is not False:
            raise SystemExit(f"{label}_test_split_accessed")
        if report.get("train_split_accessed") is not False:
            raise SystemExit(f"{label}_train_split_accessed")
        if report.get("policy_kind") != "local_llm":
            raise SystemExit(f"{label}_not_local_llm")
        if report.get("deterministic_fixture_expert") is not False:
            raise SystemExit(f"{label}_fixture_policy_used")
    if base.get("scheduled_task_ids") != sft.get("scheduled_task_ids"):
        raise SystemExit("validation_task_order_or_membership_changed")
    if base.get("model") != sft.get("model"):
        raise SystemExit("base_and_sft_generation_model_config_changed")
    if base.get("settings") != sft.get("settings"):
        raise SystemExit("base_and_sft_agent_settings_changed")


def build_comparison(base: dict[str, Any], sft: dict[str, Any]) -> dict[str, Any]:
    _validate_scope(base, sft)
    base_metrics = base.get("metrics", {})
    sft_metrics = sft.get("metrics", {})
    metrics = {
        name: {
            "base": _metric(base, name),
            "sft": _metric(sft, name),
            "delta_sft_minus_base": _delta(_metric(base, name), _metric(sft, name)),
        }
        for name in METRICS
    }
    base_analysis = base.get("failure_analysis", {})
    sft_analysis = sft.get("failure_analysis", {})
    return {
        "schema_version": "1.0",
        "status": (
            "READY_FOR_SFT_COMPARISON"
            if base.get("infrastructure_failures", 0) == 0
            and sft.get("infrastructure_failures", 0) == 0
            else "SFT_EVALUATION_BLOCKED"
        ),
        "base_artifact_status": base.get("status"),
        "sft_artifact_status": sft.get("status"),
        "split": "validation",
        "validation_task_count": 20,
        "validation_task_ids_identical_and_ordered": True,
        "test_split_accessed": False,
        "train_split_used_for_training": False,
        "generation_and_runtime_contract_held_constant": True,
        "metrics": metrics,
        "sql_rate_definition": (
            "executable_sql_rate = successful_sql_executions / sql_attempts; "
            "it is not task success rate."
        ),
        "counts": {
            "base": {
                "sql_calls": base_metrics.get("sql_calls"),
                "executable_sql_calls": base_metrics.get("executable_sql_calls"),
                "invalid_tool_call_count": base_metrics.get("invalid_tool_call_count"),
                "model_requests": base_metrics.get("model_requests"),
                "model_outputs": base_metrics.get("model_outputs"),
                "infrastructure_failures": base.get("infrastructure_failures"),
            },
            "sft": {
                "sql_calls": sft_metrics.get("sql_calls"),
                "executable_sql_calls": sft_metrics.get("executable_sql_calls"),
                "invalid_tool_call_count": sft_metrics.get("invalid_tool_call_count"),
                "model_requests": sft_metrics.get("model_requests"),
                "model_outputs": sft_metrics.get("model_outputs"),
                "infrastructure_failures": sft.get("infrastructure_failures"),
            },
        },
        "tool_selection_distribution": {
            "base": base_analysis.get("tool_selection_distribution", {}),
            "sft": sft_analysis.get("tool_selection_distribution", {}),
        },
        "failure_category_counts": {
            "base": base_analysis.get("category_counts", {}),
            "sft": sft_analysis.get("category_counts", {}),
        },
        "sql_error_type_distribution": {
            "base": base_analysis.get("sql_error_type_distribution", {}),
            "sft": sft_analysis.get("sql_error_type_distribution", {}),
        },
        "recovery_outcomes": {
            "base": base_analysis.get("recovery_outcomes", {}),
            "sft": sft_analysis.get("recovery_outcomes", {}),
        },
        "runtime_stability": {
            "base": base.get("runtime_stability", {}),
            "sft": sft.get("runtime_stability", {}),
        },
        "interpretation_boundary": (
            "This is one controlled pilot with 20 validation tasks and 20 train-only teacher "
            "trajectories; it is not evidence of broad generalization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = build_comparison(_load_report(args.base), _load_report(args.sft))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
