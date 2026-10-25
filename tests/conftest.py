import hashlib
import json
from pathlib import Path

import pytest

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import initialize_database


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return initialize_database(tmp_path / "business.sqlite3")


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def synthetic_decision_source(tmp_path_factory):
    """Build the fixed 54-row contract without reading historical training data."""
    from enterprise_agent_rl.qwen3_sft import EXPECTED_ACTION_COUNTS

    root = tmp_path_factory.mktemp("decision-source")
    actions = {
        "plan": {"text": "Inspect the synthetic schema."},
        "schema": {},
        "dictionary": {"key": "paid"},
        "sql": {"query": "SELECT 1 AS value"},
        "check": {"evidence_step": 0},
        "finish": {"evidence_step": 0, "summary": "Synthetic result."},
    }
    rows = []
    for kind, count in EXPECTED_ACTION_COUNTS.items():
        for index in range(count):
            rows.append(
                {
                    "example_id": f"synthetic-{kind}-{index}",
                    "task_id": f"train-0-synthetic-{kind}",
                    "step_index": index,
                    "context_messages": [
                        {"role": "system", "content": "Use the public tools."},
                        {"role": "user", "content": "Synthetic public observation."},
                    ],
                    "target_action": {"kind": kind, "arguments": actions[kind]},
                    "future_result_absent": True,
                    "source_policy_kind": "synthetic_test_fixture",
                }
            )
    source = root / "decisions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def train_protocol_data(tmp_path_factory):
    """Generate isolated train trajectories and metadata for protocol tests."""
    from enterprise_agent_rl.agent import DeterministicPolicy, run_episode
    from enterprise_agent_rl.benchmark import build_manifest
    from enterprise_agent_rl.data import BenchmarkTask
    from enterprise_agent_rl.reference_sql_teacher import (
        TRAJECTORY_SOURCE,
        run_reference_sql_teacher,
    )
    from enterprise_agent_rl.rewards import verify
    from enterprise_agent_rl.sft_data import build_sft_dataset
    from enterprise_agent_rl.trajectories import TrajectoryStore

    root = tmp_path_factory.mktemp("train-protocol")
    databases = root / "databases"
    manifest = build_manifest(databases)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_path.with_suffix(".sha256").write_text(manifest["manifest_sha256"] + "\n")
    trajectories = root / "trajectories"
    store = TrajectoryStore(trajectories)
    for record in manifest["records"]:
        if record["task"]["split"] != "train":
            continue
        task = BenchmarkTask.model_validate(record["task"])
        database = databases / f"{record['database_id']}.sqlite3"
        if record["template"] in {"region", "top", "inactive"}:
            trajectory = run_episode(task.public, DeterministicPolicy(), database, Settings())
        else:
            trajectory = run_reference_sql_teacher(task, database, Settings())
        reward = verify(task, trajectory, database)
        assert reward.success, task.public.task_id
        store.save(trajectory.model_copy(update={"reward": reward}))
    output = root / "dataset"
    result = build_sft_dataset(
        manifest_path,
        databases,
        trajectories,
        output,
        expected_trajectory_source=None,
        allowed_trajectory_sources={"deterministic_fixture_expert", TRAJECTORY_SOURCE},
    )
    assert result["trajectory_count"] == 20
    return {
        "manifest_path": manifest_path,
        "dataset_path": output / "dataset.jsonl",
        "metadata_path": output / "dataset-metadata.json",
        "database_root": databases,
    }
