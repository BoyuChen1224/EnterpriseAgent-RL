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

    def write_report():
        report = {
            "schema_version": "1.1",
            "split": split,
            "manifest_sha256": frozen["manifest_sha256"],
            "policy_kind": policy.kind,
            "policy_version": policy.version,
            "settings": settings.model_dump(),
            "scheduled_episodes": len(records),
            "scored_episodes": len(trajectories),
            "metrics_scope": "scored episodes; inspect complete before using as a baseline",
            "verification_errors": verification_errors,
            "metrics": summarize(trajectories, settings, allow_multiple_databases=True)
            if trajectories
            else None,
            "complete": not verification_errors and len(trajectories) == len(records),
            "episodes": [
                {
                    "task_id": t.task.task_id,
                    "run_id": t.run_id,
                    "database_sha256": t.database_sha256,
                    "terminal_reason": t.terminal_reason,
                    "reward": t.reward.model_dump(),
                }
                for t in trajectories
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
            store.save(trajectory)
            try:
                reward = verify(task, trajectory, database)
            except Exception as exc:
                verification_errors.append(
                    {
                        "task_id": task.public.task_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "run_id": trajectory.run_id,
                    }
                )
                write_report()
                continue
            trajectory = trajectory.model_copy(update={"reward": reward})
            store.save(trajectory)
            trajectories.append(trajectory)
            write_report()
    finally:
        report = write_report()
    return report
