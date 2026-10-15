"""Offline execution evaluation. Fixture reports are always labeled as such."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from enterprise_agent_rl.agent import Policy, run_episode
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import Trajectory, TrajectoryStore, database_fingerprint


def summarize(
    trajectories: list[Trajectory], settings: Settings, *, allow_multiple_databases: bool = False
) -> dict:
    if not trajectories or any(t.reward is None for t in trajectories):
        raise ValueError("evaluation_requires_scored_episodes")
    policy_keys = {
        (t.policy_kind, t.policy_version, None if allow_multiple_databases else t.database_sha256)
        for t in trajectories
    }
    if len(policy_keys) != 1:
        raise ValueError("cannot_mix_policies_or_databases")
    sql = [s for t in trajectories for s in t.steps if s.tool_name == "sql"]
    tools = [s for t in trajectories for s in t.steps if s.tool_name]
    tokens_known = all(
        all(s.token_usage_known for s in t.steps)
        and (t.policy_kind == "deterministic_fixture" or t.terminal_reason == "completed")
        for t in trajectories
    )
    token_counts = [
        sum(s.prompt_tokens + s.completion_tokens for s in t.steps) for t in trajectories
    ]
    calls = [sum(s.tool_name is not None for s in t.steps) for t in trajectories]
    avg_tokens = mean(token_counts) if tokens_known else None
    attempts = [a for t in trajectories for a in t.model_attempts]
    outputs = [a for a in attempts if a.get("raw_output") is not None]
    telemetry = [a.get("telemetry", {}) for a in attempts]
    measured = [m for m in telemetry if m.get("generation_latency_s", 0) > 0]
    total_generation_s = sum(m["generation_latency_s"] for m in measured)
    measured_tokens = sum(
        a.get("usage", {}).get("completion_tokens", 0)
        for a in attempts
        if a.get("telemetry", {}).get("generation_latency_s", 0) > 0
    )
    error_codes = [a["error"] for a in attempts if a.get("error")]
    invalid_codes = {"empty_output", "invalid_json", "invalid_action", "invalid_action_arguments"}
    return {
        "action_format_validity_rate": (
            sum(a.get("format_valid", False) for a in outputs) / len(outputs)
        )
        if outputs
        else None,
        "strict_action_format_validity_rate": (
            sum(a.get("strict_format_valid", False) for a in outputs) / len(outputs)
        )
        if outputs
        else None,
        "model_requests": len(attempts),
        "model_outputs": len(outputs),
        "generation_throughput_tokens_per_s": (measured_tokens / total_generation_s)
        if total_generation_s
        else None,
        "gpu_peak_allocated_bytes": max(
            (m["peak_allocated_bytes"] for m in telemetry if "peak_allocated_bytes" in m),
            default=None,
        ),
        "gpu_peak_reserved_bytes": max(
            (m["peak_reserved_bytes"] for m in telemetry if "peak_reserved_bytes" in m),
            default=None,
        ),
        "timeout_episodes": sum("timeout" in t.terminal_reason for t in trajectories),
        "inference_timeout_count": error_codes.count("inference_timeout"),
        "oom_count": error_codes.count("oom"),
        "invalid_output_count": sum(c in invalid_codes for c in error_codes),
        "episodes": len(trajectories),
        "task_success_rate": mean(float(t.reward.success) for t in trajectories),
        "executable_sql_rate": sum(s.tool_result.ok for s in sql) / len(sql) if sql else None,
        "tool_error_rate": sum(not s.tool_result.ok for s in tools) / len(tools) if tools else None,
        "average_tool_calls": mean(calls),
        "average_latency_s": mean(t.elapsed_s for t in trajectories),
        "average_tokens": avg_tokens,
        "token_usage_known": tokens_known,
        "average_end_to_end_cost": (
            avg_tokens * settings.token_price_per_million / 1e6 + mean(calls) * settings.tool_price
        )
        if tokens_known
        else None,
        "cost_basis": "configured token/tool prices; excludes unmeasured GPU/CPU billing",
        "policy_kind": trajectories[0].policy_kind,
        "policy_version": trajectories[0].policy_version,
        "llm_capability_result": trajectories[0].policy_kind == "local_llm",
    }


def evaluate(
    policy: Policy, tasks: list[BenchmarkTask], database: Path, settings: Settings, output: Path
) -> dict:
    store = TrajectoryStore(output / "trajectories")
    trajectories = []
    for task in tasks:
        trajectory = run_episode(
            task.public, policy, database, settings, output / "sql-audit.jsonl"
        )
        store.save(trajectory)
        trajectory = trajectory.model_copy(update={"reward": verify(task, trajectory, database)})
        store.save(trajectory)
        trajectories.append(trajectory)
    report = {
        "schema_version": "1.0",
        "split": sorted({t.split for t in tasks}),
        "task_ids": [t.public.task_id for t in tasks],
        "seed": settings.seed,
        "database_sha256": database_fingerprint(database),
        "settings": settings.model_dump(),
        "metrics": summarize(trajectories, settings),
        "episodes": [
            {
                "task_id": t.task.task_id,
                "run_id": t.run_id,
                "reward": t.reward.model_dump(),
                "terminal_reason": t.terminal_reason,
            }
            for t in trajectories
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def compare_reports(reports: list[dict]) -> dict:
    if len(reports) < 2:
        raise ValueError("comparison_requires_two_reports")
    keys = {
        (
            r["database_sha256"],
            tuple(sorted(r["task_ids"])),
            r["seed"],
            tuple(r["split"]),
            r["metrics"]["policy_kind"],
            json.dumps(r["settings"], sort_keys=True),
        )
        for r in reports
    }
    if len(keys) != 1:
        raise ValueError("incomparable_task_sets_database_seeds_or_policy_kinds")
    return {
        "schema_version": "1.0",
        "task_ids": reports[0]["task_ids"],
        "database_sha256": reports[0]["database_sha256"],
        "results": [r["metrics"] for r in reports],
        "llm_capability_result": reports[0]["metrics"]["llm_capability_result"],
    }
