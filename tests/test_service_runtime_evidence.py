"""CPU-only provenance checks for service runtime counters."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/serving/run_sft_n_api.py"
SPEC = importlib.util.spec_from_file_location("t028_runtime_evidence_launcher", LAUNCHER_PATH)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_trajectory(path: Path, *, attempts: int, sql_steps: int, reward) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_attempts": [{} for _ in range(attempts)],
                "steps": [{"tool_name": "sql"} for _ in range(sql_steps)],
                "reward": reward,
            }
        )
    )


def test_runtime_evidence_uses_one_primary_source_without_double_counting(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "model" / "generations.jsonl",
        [{"id": "event-1"}, {"id": "event-2"}],
    )
    _write_jsonl(
        run_dir / "api" / "sql-audit.jsonl",
        [{"sql_sha256": "sql-1"}, {"sql_sha256": "sql-2"}, {"sql_sha256": "sql-3"}],
    )
    _write_trajectory(
        run_dir / "api" / "trajectories" / "one.json",
        attempts=9,
        sql_steps=7,
        reward={"success": True},
    )
    _write_trajectory(
        run_dir / "api" / "trajectories" / "two.json",
        attempts=4,
        sql_steps=5,
        reward=None,
    )

    evidence = launcher.collect_runtime_evidence(run_dir)

    assert evidence["model_calls"]["value"] == 2
    assert evidence["model_calls"]["source_kind"] == "persisted_run_artifact"
    assert evidence["business_sql_calls"]["value"] == 3
    assert evidence["verifier_calls"]["value"] == 1
    assert evidence["model_calls"]["reason"] is None
    assert evidence["business_sql_calls"]["reason"] is None


def test_runtime_evidence_falls_back_to_trajectories_and_missing_is_null(tmp_path):
    run_dir = tmp_path / "fallback"
    _write_trajectory(
        run_dir / "api" / "trajectories" / "one.json",
        attempts=2,
        sql_steps=1,
        reward=None,
    )

    evidence = launcher.collect_runtime_evidence(run_dir)
    assert evidence["model_calls"]["value"] == 2
    assert evidence["business_sql_calls"]["value"] == 1
    assert evidence["verifier_calls"]["value"] == 0
    assert evidence["model_calls"]["source_kind"] == "persisted_trajectory_fallback"
    assert evidence["business_sql_calls"]["source_kind"] == "persisted_trajectory_fallback"

    missing = launcher.collect_runtime_evidence(tmp_path / "missing")
    assert all(item["value"] is None for item in missing.values())
    assert all(item["reason"] is not None for item in missing.values())


def test_finalize_results_persists_runtime_provenance(tmp_path):
    run_dir = tmp_path / "summary"
    _write_jsonl(run_dir / "model" / "generations.jsonl", [{"id": "event-1"}])
    summary = launcher.finalize_results(
        run_dir,
        ["task-one"],
        mode="smoke",
        status="COMPLETED",
        exit_code=0,
    )
    persisted = json.loads((run_dir / "summary.json").read_text())
    assert summary["runtime_evidence"] == persisted["runtime_evidence"]
    assert persisted["runtime_evidence"]["model_calls"]["value"] == 1
    assert persisted["runtime_evidence"]["business_sql_calls"]["value"] is None
