"""Independent label checks; verifier SQL is used only in this CPU test module."""

import json
import sqlite3
from pathlib import Path

import pytest

from enterprise_agent_rl.benchmark import build_manifest, load_frozen_manifest


def test_sixty_labels_split_isolation_and_frozen_fingerprints(tmp_path):
    manifest = build_manifest(tmp_path / "databases")
    frozen_path = Path(__file__).resolve().parents[1] / "benchmarks/m2a/manifest.json"
    assert manifest == json.loads(frozen_path.read_text())
    assert manifest["split_counts"] == {"train": 20, "validation": 20, "test": 20}
    assert len({r["data_fingerprint"] for r in manifest["records"]}) == 60
    assert len({d["sha256"] for d in manifest["databases"]}) == 6
    names, templates = {}, {}
    for split in manifest["split_counts"]:
        names[split] = set()
        templates[split] = {
            r["template"] for r in manifest["records"] if r["task"]["split"] == split
        }
        for index in range(2):
            with sqlite3.connect(tmp_path / "databases" / f"{split}-{index}.sqlite3") as db:
                names[split].update(n[0] for n in db.execute("SELECT name FROM customers"))
                names[split].update(n[0] for n in db.execute("SELECT name FROM products"))
    assert not names["train"] & names["test"]
    assert not names["validation"] & names["test"]
    assert len(templates["test"] - templates["train"] - templates["validation"]) == 2
    assert load_frozen_manifest(frozen_path, tmp_path / "databases") == manifest
    # VACUUM changes SQLite file metadata while preserving the dataset identity.
    from enterprise_agent_rl.trajectories import database_fingerprint

    db_path = tmp_path / "databases" / "test-0.sqlite3"
    physical_before = database_fingerprint(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("VACUUM")
    assert database_fingerprint(db_path) != physical_before
    assert load_frozen_manifest(frozen_path, tmp_path / "databases") == manifest
    with sqlite3.connect(tmp_path / "databases" / "test-0.sqlite3") as db:
        db.execute("UPDATE customers SET name='tampered' WHERE id=1")
    with pytest.raises(ValueError, match="database_changed"):
        load_frozen_manifest(frozen_path, tmp_path / "databases")
    altered = tmp_path / "manifest.json"
    data = dict(manifest)
    data["records"][0]["task"]["prompt"] = "changed"
    altered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest_hash"):
        load_frozen_manifest(altered, tmp_path / "databases")


def test_suite_reports_fixture_and_retains_verifier_failures(tmp_path, monkeypatch):
    import enterprise_agent_rl.base_evaluation as module
    from enterprise_agent_rl.agent import DeterministicPolicy
    from enterprise_agent_rl.base_evaluation import evaluate_suite
    from enterprise_agent_rl.common import Settings

    manifest = build_manifest(tmp_path / "db")
    # Use validation only; this exercises the report plumbing, not an LLM benchmark.
    records = [r for r in manifest["records"] if r["task"]["split"] == "validation"]
    manifest["records"] = [records[0], records[10]]
    monkeypatch.setattr(module, "load_frozen_manifest", lambda *a: manifest)
    original = module.verify
    count = 0

    def verifier(*args):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("intentional CPU verifier outage")
        return original(*args)

    monkeypatch.setattr(module, "verify", verifier)
    report = evaluate_suite(
        DeterministicPolicy(),
        tmp_path / "unused.json",
        tmp_path / "db",
        "validation",
        Settings(max_actions=2),
        tmp_path / "evaluation",
    )
    assert not report["complete"] and report["scored_episodes"] == 1
    assert report["verification_errors"][0]["type"] == "RuntimeError"
    assert report["metrics"]["llm_capability_result"] is False
    assert len(list((tmp_path / "evaluation/trajectories").glob("*.json"))) == 2
    assert json.loads((tmp_path / "evaluation/report.json").read_text()) == report
