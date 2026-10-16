"""Offline execution evaluation. Fixture reports are always labeled as such."""

from __future__ import annotations

import json
from collections import Counter
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
    output_token_counts = []
    output_token_usage_known = True
    for trajectory in trajectories:
        total = 0
        for attempt in trajectory.model_attempts:
            usage = attempt.get("usage", {})
            if "completion_tokens" not in usage:
                output_token_usage_known = False
            else:
                total += usage["completion_tokens"]
        output_token_counts.append(total)
    error_distribution = Counter(error_codes)
    for trajectory in trajectories:
        error_distribution.update(
            f"runtime:{item.get('type', 'unknown')}" for item in trajectory.runtime_errors
        )
        for step in trajectory.steps:
            if step.tool_name and not step.tool_result.ok:
                error_distribution[
                    f"tool:{step.tool_name}:{step.tool_result.error or 'error'}"
                ] += 1
    generation_latencies = [
        m["generation_latency_s"] for m in measured if m.get("generation_latency_s", 0) > 0
    ]
    sql_calls = len(sql)
    executable_sql_calls = sum(s.tool_result.ok for s in sql)
    tool_errors = sum(not s.tool_result.ok for s in tools)
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
        "average_inference_latency_s": mean(generation_latencies) if generation_latencies else None,
        "total_inference_latency_s": total_generation_s or None,
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
        "executable_sql_rate": executable_sql_calls / sql_calls if sql_calls else None,
        "sql_executable_rate": executable_sql_calls / sql_calls if sql_calls else None,
        "sql_calls": sql_calls,
        "executable_sql_calls": executable_sql_calls,
        "tool_error_rate": tool_errors / len(tools) if tools else None,
        "invalid_tool_call_count": tool_errors,
        "average_tool_calls": mean(calls),
        "average_latency_s": mean(t.elapsed_s for t in trajectories),
        "average_episode_latency_s": mean(t.elapsed_s for t in trajectories),
        "average_tokens": avg_tokens,
        "token_usage_known": tokens_known,
        "average_output_tokens": mean(output_token_counts) if output_token_usage_known else None,
        "output_token_usage_known": output_token_usage_known,
        "output_tokens_total": sum(output_token_counts) if output_token_usage_known else None,
        "error_type_distribution": dict(sorted(error_distribution.items())),
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
    verification_errors = []
    episode_records = []

    def write_report() -> dict:
        metrics = summarize(trajectories, settings) if trajectories else None
        if metrics is not None:
            scored_successes = sum(bool(t.reward and t.reward.success) for t in trajectories)
            metrics.update(
                task_success_rate_scored=metrics["task_success_rate"],
                task_success_rate=scored_successes / len(tasks) if tasks else None,
                task_success_denominator=len(tasks),
                scored_task_successes=scored_successes,
                verifier_success_rate=len(trajectories) / len(tasks) if tasks else None,
                verifier_error_count=len(verification_errors),
            )
        report = {
            "schema_version": "1.1",
            "split": sorted({t.split for t in tasks}),
            "task_ids": [t.public.task_id for t in tasks],
            "seed": settings.seed,
            "database_sha256": database_fingerprint(database),
            "settings": settings.model_dump(),
            "scheduled_episodes": len(tasks),
            "scored_episodes": len(trajectories),
            "metrics_scope": (
                "task_success_rate uses scheduled episodes as denominator; other aggregate "
                "execution metrics use scored episodes"
            ),
            "verification_errors": verification_errors,
            "metrics": metrics,
            "complete": not verification_errors and len(trajectories) == len(tasks),
            "episodes": episode_records,
            "failure_trajectories": [
                item
                for item in episode_records
                if item.get("verification_status") == "error"
                or not (item.get("reward") or {}).get("success", False)
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report

    for task in tasks:
        trajectory = run_episode(
            task.public, policy, database, settings, output / "sql-audit.jsonl"
        )
        trajectory_path = store.save(trajectory)
        try:
            trajectory = trajectory.model_copy(
                update={"reward": verify(task, trajectory, database)}
            )
        except Exception as exc:
            verification_errors.append(
                {
                    "task_id": task.public.task_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "run_id": trajectory.run_id,
                    "trajectory_path": str(trajectory_path.relative_to(output)),
                }
            )
            episode_records.append(
                {
                    "task_id": trajectory.task.task_id,
                    "run_id": trajectory.run_id,
                    "reward": None,
                    "terminal_reason": trajectory.terminal_reason,
                    "verification_status": "error",
                    "verification_error": verification_errors[-1],
                    "trajectory_path": str(trajectory_path.relative_to(output)),
                }
            )
            write_report()
            continue
        trajectory_path = store.save(trajectory)
        trajectories.append(trajectory)
        episode_records.append(
            {
                "task_id": trajectory.task.task_id,
                "run_id": trajectory.run_id,
                "reward": trajectory.reward.model_dump(),
                "terminal_reason": trajectory.terminal_reason,
                "verification_status": "scored",
                "trajectory_path": str(trajectory_path.relative_to(output)),
            }
        )
        write_report()
    return write_report()


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
