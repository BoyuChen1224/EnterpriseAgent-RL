#!/usr/bin/env python3
"""Generate train-only v2 teacher trajectories and verified SFT artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.reference_sql_teacher import (
    TRAJECTORY_SOURCE,
    reference_sql_execution_status,
    run_reference_sql_teacher,
)
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.sft_data import build_sft_dataset, load_train_source
from enterprise_agent_rl.trajectories import TrajectoryStore

DATASET_VERSION = "train-teacher-v2"
ALLOWED_SOURCES = {"deterministic_fixture_expert", TRAJECTORY_SOURCE}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _existing_successes(
    source: Any,
    existing_dir: Path,
    destination_store: TrajectoryStore,
    database_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Reuse only already-successful deterministic traces, preserving provenance."""

    selected: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for path in sorted(existing_dir.glob("*.json")):
        try:
            trajectory = TrajectoryStore.load(path)
        except Exception as exc:
            rejected.append(
                {
                    "path": str(path),
                    "task_id": None,
                    "reason": "existing_trajectory_schema_invalid",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        task_id = trajectory.task.task_id
        record = source.records.get(task_id)
        if record is None or trajectory.policy_kind != "deterministic_fixture":
            continue
        task = BenchmarkTask.model_validate(record["task"])
        database = database_root / f"{record['database_id']}.sqlite3"
        try:
            reward = verify(task, trajectory, database)
        except Exception as exc:
            rejected.append(
                {
                    "path": str(path),
                    "task_id": task_id,
                    "reason": "existing_trajectory_verifier_error",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        if not reward.success:
            continue
        rescored = trajectory.model_copy(update={"reward": reward})
        saved = destination_store.save(rescored)
        selected[task_id] = {
            "trajectory": rescored,
            "path": saved,
            "original_path": path,
            "reward": reward,
        }
    return selected, rejected


def _entry(
    *,
    task: BenchmarkTask,
    record: dict[str, Any],
    trajectory: Any,
    output_dir: Path,
    status: str,
    reward_consistent: bool,
    original_path: Path | None = None,
    verifier_error: str | None = None,
) -> dict[str, Any]:
    reward = trajectory.reward.model_dump() if trajectory.reward else None
    source = (
        "reference_sql_teacher"
        if trajectory.policy_kind == "programmatic_reference_sql_teacher"
        else "deterministic_fixture_expert"
    )
    trajectory_path = output_dir / "trajectories" / f"{trajectory.run_id}.json"
    return {
        "task_id": task.public.task_id,
        "template_id": record.get("template"),
        "source_split": task.split,
        "existing_teacher_success": source == "deterministic_fixture_expert",
        "trajectory_source": source,
        "policy_kind": trajectory.policy_kind,
        "policy_version": trajectory.policy_version,
        "reference_sql_available": bool(task.reference_sql.strip()),
        "reference_sql_execution_status": reference_sql_execution_status(
            trajectory, task.reference_sql
        ),
        "actual_tool_execution": any(step.tool_name for step in trajectory.steps),
        "verifier_status": (
            "error" if verifier_error else "passed" if reward and reward["success"] else "failed"
        ),
        "verifier_error": verifier_error,
        "reward": reward,
        "reward_consistent": reward_consistent,
        "generation_status": status,
        "trajectory_path": _relative(trajectory_path, output_dir),
        "original_trajectory_path": str(original_path) if original_path else None,
    }


def generate(
    manifest_path: Path,
    database_root: Path,
    existing_trajectory_dir: Path,
    output_dir: Path,
    settings: Settings,
) -> dict[str, Any]:
    source = load_train_source(manifest_path, database_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_output:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_store = TrajectoryStore(output_dir / "trajectories")

    reused, existing_rejections = _existing_successes(
        source, existing_trajectory_dir, trajectory_store, database_root
    )
    entries: list[dict[str, Any]] = []
    attempted_new = 0
    for task_id in sorted(source.records):
        record = source.records[task_id]
        task = BenchmarkTask.model_validate(record["task"])
        database = database_root / f"{record['database_id']}.sqlite3"
        if task_id in reused:
            item = reused[task_id]
            entries.append(
                _entry(
                    task=task,
                    record=record,
                    trajectory=item["trajectory"],
                    output_dir=output_dir,
                    status="reused_existing_verified_success",
                    reward_consistent=True,
                    original_path=item["original_path"],
                )
            )
            continue

        attempted_new += 1
        trajectory = run_reference_sql_teacher(
            task, database, settings, output_dir / "sql-audit.jsonl"
        )
        verifier_error = None
        reward_consistent = False
        try:
            reward = verify(task, trajectory, database)
        except Exception as exc:
            verifier_error = f"{type(exc).__name__}:{exc}"
            reward = None
        if reward is not None:
            reward_consistent = True
            trajectory = trajectory.model_copy(update={"reward": reward})
        trajectory_store.save(trajectory)
        status = (
            "reference_sql_teacher_verified_success"
            if reward is not None and reward.success
            else "reference_sql_teacher_generation_failure"
            if verifier_error is None
            else "reference_sql_teacher_verifier_error"
        )
        entries.append(
            _entry(
                task=task,
                record=record,
                trajectory=trajectory,
                output_dir=output_dir,
                status=status,
                reward_consistent=reward_consistent,
                verifier_error=verifier_error,
            )
        )

    metadata = build_sft_dataset(
        manifest_path,
        database_root,
        output_dir / "trajectories",
        output_dir,
        dataset_version=DATASET_VERSION,
        expected_trajectory_source=None,
        allowed_trajectory_sources=ALLOWED_SOURCES,
    )
    selected_task_ids = set(metadata["quality_report"]["accepted_task_ids"])
    for item in entries:
        item["included_in_sft_dataset"] = item["task_id"] in selected_task_ids

    source_distribution = Counter(item["trajectory_source"] for item in entries)
    template_success = Counter(
        item["template_id"] for item in entries if item["included_in_sft_dataset"]
    )
    teacher_report = {
        "schema_version": "1.0",
        "dataset_version": DATASET_VERSION,
        "source_split": "train",
        "created_from_commit": _git_commit(),
        "manifest_sha256": source.manifest_sha256,
        "source_task_count": len(source.records),
        "attempted_new_task_count": attempted_new,
        "reused_existing_verified_success_count": len(reused),
        "trajectory_count_persisted": len(list((output_dir / "trajectories").glob("*.json"))),
        "entries": sorted(entries, key=lambda item: item["task_id"]),
        "existing_candidate_rejections": existing_rejections,
        "summary": {
            "verified_successful_task_count": len(selected_task_ids),
            "verified_successful_template_count": len(template_success),
            "verified_successful_templates": sorted(template_success),
            "trajectory_source_distribution": dict(sorted(source_distribution.items())),
            "generation_status_distribution": dict(
                sorted(Counter(item["generation_status"] for item in entries).items())
            ),
            "reference_sql_execution_status_distribution": dict(
                sorted(
                    Counter(
                        item["reference_sql_execution_status"]["status"] for item in entries
                    ).items()
                )
            ),
            "verifier_status_distribution": dict(
                sorted(Counter(item["verifier_status"] for item in entries).items())
            ),
            "uncovered_task_ids": sorted(set(source.records) - selected_task_ids),
            "uncovered_templates": sorted(
                {
                    source.records[task_id].get("template")
                    for task_id in set(source.records) - selected_task_ids
                }
            ),
        },
        "validation_used_for_training": False,
        "test_used_for_training": False,
        "base_benchmark_included": False,
        "reference_sql_field_in_sft_examples": False,
        "supervision_sql_in_assistant_actions": True,
        "sft_metadata": {
            "trajectory_count": metadata["trajectory_count"],
            "successful_task_count": metadata["successful_task_count"],
            "final_decision": metadata["final_decision"],
        },
    }
    _write_json(output_dir / "teacher-generation-report.json", teacher_report)
    return metadata | {"teacher_generation_report": teacher_report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument(
        "--database-root", type=Path, default=Path("artifacts/m2a-databases-final")
    )
    parser.add_argument(
        "--existing-trajectories",
        type=Path,
        default=Path("artifacts/m2b-train-audit/trajectories"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sft-data-train-teacher-v2")
    )
    args = parser.parse_args()
    result = generate(
        args.manifest,
        args.database_root,
        args.existing_trajectories,
        args.output,
        Settings(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
