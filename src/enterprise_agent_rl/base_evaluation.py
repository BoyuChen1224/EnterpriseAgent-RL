"""Evaluate only a chosen split, saving trajectories before independent scoring."""

from __future__ import annotations

import json
from pathlib import Path

from enterprise_agent_rl.agent import Policy, run_episode
from enterprise_agent_rl.benchmark import load_frozen_manifest
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.evaluation import summarize
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import TrajectoryStore


def evaluate_suite(
    policy: Policy,
    manifest: Path,
    database_root: Path,
    split: str,
    settings: Settings,
    output: Path,
) -> dict:
    if split not in {"train", "validation", "test"}:
        raise ValueError("choose_one_split")
    frozen = load_frozen_manifest(manifest, database_root)
    records = [r for r in frozen["records"] if r["task"]["split"] == split]
    store = TrajectoryStore(output / "trajectories")
    trajectories, verification_errors = [], []
    episode_records = []

    def _metrics() -> dict | None:
        if not trajectories:
            return None
        metrics = summarize(trajectories, settings, allow_multiple_databases=True)
        scored_successes = sum(bool(t.reward and t.reward.success) for t in trajectories)
        metrics.update(
            task_success_rate_scored=metrics["task_success_rate"],
            task_success_rate=scored_successes / len(records) if records else None,
            task_success_denominator=len(records),
            scored_task_successes=scored_successes,
            verifier_success_rate=len(trajectories) / len(records) if records else None,
            verifier_error_count=len(verification_errors),
        )
        return metrics

    def write_report():
        metrics = _metrics()
        report = {
            "schema_version": "1.2",
            "split": split,
            "manifest_sha256": frozen["manifest_sha256"],
            "database_fingerprint_kind": frozen["database_fingerprint_kind"],
            "policy_kind": policy.kind,
            "policy_version": policy.version,
            "settings": settings.model_dump(),
            "scheduled_episodes": len(records),
            "scored_episodes": len(trajectories),
            "metrics_scope": (
                "task_success_rate uses all scheduled episodes as denominator; other aggregate "
                "execution metrics use scored episodes; incomplete if verifier errors exist"
            ),
            "verification_errors": verification_errors,
            "metrics": metrics,
            "complete": not verification_errors and len(trajectories) == len(records),
            "episodes": episode_records,
            "failure_trajectories": [
                item
                for item in episode_records
                if item.get("verification_status") == "error"
                or not (item.get("reward") or {}).get("success", False)
            ],
        }
        report_path = output / "report.json"
        temporary = output / ".report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(report_path)
        return report

    try:
        for record in records:
            task = BenchmarkTask.model_validate(record["task"])
            database = database_root / (record["database_id"] + ".sqlite3")
            trajectory = run_episode(
                task.public, policy, database, settings, output / "sql-audit.jsonl"
            )
            trajectory_path = store.save(trajectory)
            try:
                reward = verify(task, trajectory, database)
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
                        "database_sha256": trajectory.database_sha256,
                        "terminal_reason": trajectory.terminal_reason,
                        "reward": None,
                        "verification_status": "error",
                        "verification_error": verification_errors[-1],
                        "trajectory_path": str(trajectory_path.relative_to(output)),
                    }
                )
                write_report()
                continue
            trajectory = trajectory.model_copy(update={"reward": reward})
            trajectory_path = store.save(trajectory)
            trajectories.append(trajectory)
            episode_records.append(
                {
                    "task_id": trajectory.task.task_id,
                    "run_id": trajectory.run_id,
                    "database_sha256": trajectory.database_sha256,
                    "terminal_reason": trajectory.terminal_reason,
                    "reward": trajectory.reward.model_dump(),
                    "verification_status": "scored",
                    "trajectory_path": str(trajectory_path.relative_to(output)),
                }
            )
            write_report()
    finally:
        report = write_report()
    return report
