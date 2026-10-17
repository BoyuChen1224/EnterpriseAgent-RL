"""Train-only reference SQL teacher regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_agent_rl.agent import DeterministicPolicy, run_episode
from enterprise_agent_rl.benchmark import build_manifest
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.reference_sql_teacher import (
    POLICY_KIND,
    TRAJECTORY_SOURCE,
    ReferenceSQLTeacherPolicy,
    run_reference_sql_teacher,
)
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.sft_data import build_sft_dataset
from enterprise_agent_rl.trajectories import TrajectoryStore


@pytest.fixture
def frozen_benchmark(tmp_path: Path) -> tuple[Path, Path, dict]:
    database_root = tmp_path / "databases"
    manifest = build_manifest(database_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_path.with_suffix(".sha256").write_text(manifest["manifest_sha256"] + "\n")
    return manifest_path, database_root, manifest


def _record(manifest: dict, task_id: str) -> dict:
    return next(r for r in manifest["records"] if r["task"]["public"]["task_id"] == task_id)


def test_reference_sql_teacher_uses_runtime_and_formal_reward(frozen_benchmark):
    _, database_root, manifest = frozen_benchmark
    record = _record(manifest, "train-0-schema")
    task = BenchmarkTask.model_validate(record["task"])
    database = database_root / f"{record['database_id']}.sqlite3"

    trajectory = run_reference_sql_teacher(task, database, Settings())
    reward = verify(task, trajectory, database)
    trajectory = trajectory.model_copy(update={"reward": reward})

    assert reward.success and reward.value > 0
    assert trajectory.policy_kind == POLICY_KIND
    assert trajectory.policy_metadata["trajectory_source"] == TRAJECTORY_SOURCE
    assert trajectory.policy_metadata["source_split"] == "train"
    assert trajectory.policy_metadata["reference_sql_execution"]["status"] == "success"
    assert [step.action.kind for step in trajectory.steps] == [
        "plan",
        "schema",
        "sql",
        "check",
        "finish",
    ]
    sql_steps = [step for step in trajectory.steps if step.action.kind == "sql"]
    assert sql_steps[0].action.arguments["query"] == task.reference_sql
    assert sql_steps[0].tool_result.ok
    assert trajectory.final_answer == trajectory.steps[-1].tool_result.result


def test_reference_sql_teacher_rejects_non_train_task(frozen_benchmark):
    _, _, manifest = frozen_benchmark
    validation_record = _record(manifest, "validation-0-schema")
    validation_task = BenchmarkTask.model_validate(validation_record["task"])
    with pytest.raises(ValueError, match="train_only"):
        ReferenceSQLTeacherPolicy(validation_task)


def test_mixed_teacher_sources_are_allowed_but_local_llm_is_rejected(
    frozen_benchmark, tmp_path
):
    manifest_path, database_root, manifest = frozen_benchmark
    output = tmp_path / "output"
    trajectories = tmp_path / "trajectories"
    store = TrajectoryStore(trajectories)

    deterministic_record = _record(manifest, "train-0-region")
    deterministic_task = BenchmarkTask.model_validate(deterministic_record["task"])
    deterministic_database = database_root / f"{deterministic_record['database_id']}.sqlite3"
    deterministic = run_episode(
        deterministic_task.public,
        DeterministicPolicy(),
        deterministic_database,
        Settings(),
    )
    deterministic = deterministic.model_copy(
        update={"reward": verify(deterministic_task, deterministic, deterministic_database)}
    )
    store.save(deterministic)

    teacher_record = _record(manifest, "train-0-schema")
    teacher_task = BenchmarkTask.model_validate(teacher_record["task"])
    teacher_database = database_root / f"{teacher_record['database_id']}.sqlite3"
    teacher = run_reference_sql_teacher(teacher_task, teacher_database, Settings())
    teacher = teacher.model_copy(
        update={"reward": verify(teacher_task, teacher, teacher_database)}
    )
    store.save(teacher)

    result = build_sft_dataset(
        manifest_path,
        database_root,
        trajectories,
        output,
        dataset_version="train-teacher-v2-test",
        expected_trajectory_source=None,
        allowed_trajectory_sources={"deterministic_fixture_expert", TRAJECTORY_SOURCE},
    )
    assert result["trajectory_count"] == 2
    rows = [json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()]
    assert {row["source"]["trajectory_source"] for row in rows} == {
        "deterministic_fixture_expert",
        TRAJECTORY_SOURCE,
    }
    assert all("reference_sql" not in row for row in rows)

    local_output = tmp_path / "local-output"
    local_trajectory = deterministic.model_copy(
        update={"schema_version": "1.0", "policy_kind": "local_llm"}
    )
    local_dir = tmp_path / "local-trajectories"
    TrajectoryStore(local_dir).save(local_trajectory)
    rejected = build_sft_dataset(
        manifest_path,
        database_root,
        local_dir,
        local_output,
        expected_trajectory_source=None,
        allowed_trajectory_sources={"deterministic_fixture_expert", TRAJECTORY_SOURCE},
    )
    assert rejected["trajectory_count"] == 0
    assert (
        rejected["quality_report"]["rejection_reason_counts"][
            "trajectory_source_not_allowed"
        ]
        == 1
    )
