"""Train-only SFT data construction tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from enterprise_agent_rl.agent import DeterministicPolicy, run_episode
from enterprise_agent_rl.benchmark import build_manifest
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask
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


def _scored_trajectory(database_root: Path, record: dict, *, reward: bool = True):
    task = BenchmarkTask.model_validate(record["task"])
    database = database_root / f"{record['database_id']}.sqlite3"
    trajectory = run_episode(task.public, DeterministicPolicy(), database, Settings())
    if reward:
        trajectory = trajectory.model_copy(update={"reward": verify(task, trajectory, database)})
    return trajectory


def _write_trajectory(directory: Path, trajectory) -> None:
    TrajectoryStore(directory).save(trajectory)


def _record(manifest: dict, task_id: str) -> dict:
    return next(r for r in manifest["records"] if r["task"]["public"]["task_id"] == task_id)


def test_train_split_only(frozen_benchmark, tmp_path, monkeypatch):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "train-0-region")),
    )
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "validation-0-region")),
    )
    import enterprise_agent_rl.sft_data as module

    opened = []
    original = module.database_content_fingerprint

    def track(path):
        opened.append(path.name)
        return original(path)

    monkeypatch.setattr(module, "database_content_fingerprint", track)
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["source_split"] == "train"
    assert result["trajectory_count"] == 1
    assert all(name.startswith("train-") for name in opened)


def test_validation_not_in_sft(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "validation-0-region")),
    )
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["final_decision"] == "BLOCKED_ON_SFT_DATA"
    assert result["trajectory_count"] == 0
    assert (
        "source_split_not_train:validation" in result["quality_report"]["rejection_reason_counts"]
    )


def test_test_not_in_sft(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "test-0-region")),
    )
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["final_decision"] == "BLOCKED_ON_SFT_DATA"
    assert result["trajectory_count"] == 0
    assert "source_split_not_train:test" in result["quality_report"]["rejection_reason_counts"]


def test_success_filter(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "train-0-region")),
    )
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "train-0-constraints")),
    )
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["trajectory_count"] == 1
    assert result["successful_task_count"] == 1
    assert result["quality_report"]["rejection_reason_counts"]["verifier_success_required"] == 1


def test_verifier_required(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    record = _record(manifest, "train-0-region")
    trajectory = _scored_trajectory(database_root, record).model_copy(update={"reward": None})
    trajectories = tmp_path / "trajectories"
    _write_trajectory(trajectories, trajectory)
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["trajectory_count"] == 0
    reasons = result["quality_report"]["rejection_reason_counts"]
    assert reasons["reward_missing"] == 1


def test_provenance(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "train-0-region")),
    )
    result = build_sft_dataset(
        manifest_path,
        database_root,
        trajectories,
        tmp_path / "out",
        expected_trajectory_source="local_llm",
    )
    assert result["trajectory_count"] == 0
    assert result["quality_report"]["rejection_reason_counts"]["trajectory_source_mismatch"] == 1


def test_trajectory_schema(frozen_benchmark, tmp_path):
    manifest_path, database_root, _ = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    (trajectories / "malformed.json").write_text('{"not": "a trajectory"}\n')
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["trajectory_count"] == 0
    assert result["quality_report"]["rejection_reason_counts"]["trajectory_schema_invalid"] == 1


def test_duplicate_detection(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectory = _scored_trajectory(database_root, _record(manifest, "train-0-region"))
    duplicate = trajectory.model_copy(update={"run_id": str(uuid.uuid4())})
    trajectories = tmp_path / "trajectories"
    _write_trajectory(trajectories, trajectory)
    _write_trajectory(trajectories, duplicate)
    result = build_sft_dataset(manifest_path, database_root, trajectories, tmp_path / "out")
    assert result["trajectory_count"] == 1
    assert result["quality_report"]["quality"]["duplicate_trajectory_count"] == 1


def test_sft_format(frozen_benchmark, tmp_path):
    manifest_path, database_root, manifest = frozen_benchmark
    trajectories = tmp_path / "trajectories"
    _write_trajectory(
        trajectories,
        _scored_trajectory(database_root, _record(manifest, "train-0-region")),
    )
    output = tmp_path / "out"
    build_sft_dataset(manifest_path, database_root, trajectories, output)
    rows = [json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    example = rows[0]
    assert example["assistant_only_loss"] is True
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert all(
        example["messages"][i]["role"] == "assistant" for i in example["assistant_message_indices"]
    )
    assert "reference_sql" not in json.dumps(example, ensure_ascii=False)
