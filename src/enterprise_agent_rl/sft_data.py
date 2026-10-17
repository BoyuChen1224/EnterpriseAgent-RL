"""Train-only, verified trajectory to chat-SFT conversion.

This module deliberately has its own split-scoped manifest loader.  The existing
benchmark loader validates every database, which is useful for benchmark runs but
would make a data-construction command touch validation and test databases.  SFT
construction therefore validates the frozen manifest and opens only train files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_agent_rl.agent import SYSTEM_PROMPT
from enterprise_agent_rl.benchmark import canonical_hash, database_content_fingerprint
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.model_protocol import parse_action
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import (
    Trajectory,
    TrajectoryStore,
    database_fingerprint,
    replay,
)

DATASET_SCHEMA_VERSION = "1.0"
EXAMPLE_SCHEMA_VERSION = "1.0"
DATASET_VERSION = "train-teacher-v1"
EXPECTED_SPLIT_COUNTS = {"train": 20, "validation": 20, "test": 20}
KNOWN_POLICY_SOURCES = {
    "deterministic_fixture": "deterministic_fixture_expert",
    "local_llm": "local_llm",
    "programmatic_reference_sql_teacher": "reference_sql_teacher",
}


@dataclass(frozen=True)
class TrainSource:
    manifest_path: Path
    database_root: Path
    manifest_sha256: str
    benchmark_version: str
    database_fingerprint_kind: str
    split_counts: dict[str, int]
    records: dict[str, dict[str, Any]]
    all_task_splits: dict[str, str]
    database_entries: dict[str, dict[str, Any]]


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def reward_version() -> str:
    """Return a source fingerprint for the formal verifier/reward implementation."""

    path = Path(__file__).with_name("rewards.py")
    return f"verify:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def load_train_source(manifest_path: Path, database_root: Path) -> TrainSource:
    """Validate the frozen manifest while opening only train databases.

    Reading split metadata for validation/test is allowed for leakage checks.  No
    validation/test SQLite file is opened and no validation/test task payload is
    copied into the returned records.
    """

    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("manifest_not_object")
    declared = raw.get("manifest_sha256")
    if not isinstance(declared, str) or not declared:
        raise ValueError("manifest_hash_missing")
    unsigned = dict(raw)
    unsigned.pop("manifest_sha256", None)
    if canonical_hash(unsigned) != declared:
        raise ValueError("manifest_hash_mismatch")
    lock_path = manifest_path.with_suffix(".sha256")
    if not lock_path.is_file() or lock_path.read_text().strip() != declared:
        raise ValueError("frozen_manifest_changed")

    split_counts = raw.get("split_counts")
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"split_counts_mismatch:{split_counts}")

    raw_records = raw.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("manifest_records_not_list")
    all_task_splits: dict[str, str] = {}
    for record in raw_records:
        try:
            task = record["task"]
            task_id = task["public"]["task_id"]
            split = task["split"]
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed_manifest_record") from exc
        if task_id in all_task_splits:
            raise ValueError("duplicate_task_id")
        all_task_splits[task_id] = split

    train_records = [
        record for record in raw_records if record.get("task", {}).get("split") == "train"
    ]
    if len(train_records) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"train_task_count_mismatch:{len(train_records)}")
    records: dict[str, dict[str, Any]] = {}
    for record in train_records:
        fields = {key: value for key, value in record.items() if key != "data_fingerprint"}
        if canonical_hash(fields) != record.get("data_fingerprint"):
            raise ValueError("task_fingerprint_mismatch")
        task_id = record["task"]["public"]["task_id"]
        if not task_id.startswith("train-"):
            raise ValueError("train_task_id_mismatch")
        records[task_id] = record

    entries = {entry["id"]: entry for entry in raw.get("databases", [])}
    train_database_ids = {record["database_id"] for record in train_records}
    if not train_database_ids or not train_database_ids <= set(entries):
        raise ValueError("train_database_selection_mismatch")
    if any(entries[db_id].get("split") != "train" for db_id in train_database_ids):
        raise ValueError("non_train_database_selected")

    # This is the only database-opening loop in the loader, and it is explicitly
    # restricted to database IDs referenced by train records.
    for database_id in sorted(train_database_ids):
        database = database_root / f"{database_id}.sqlite3"
        if not database.is_file():
            raise FileNotFoundError(database)
        if database_content_fingerprint(database) != entries[database_id]["sha256"]:
            raise ValueError(f"benchmark_database_changed:{database_id}")

    for record in train_records:
        if record["database_sha256"] != entries[record["database_id"]]["sha256"]:
            raise ValueError("record_database_fingerprint_mismatch")

    return TrainSource(
        manifest_path=manifest_path,
        database_root=database_root,
        manifest_sha256=declared,
        benchmark_version=raw.get("version", "unknown"),
        database_fingerprint_kind=raw.get("database_fingerprint_kind", "unknown"),
        split_counts=dict(split_counts),
        records=records,
        all_task_splits=all_task_splits,
        database_entries={
            key: value for key, value in entries.items() if value.get("split") == "train"
        },
    )


def _trajectory_content_hash(trajectory: Trajectory) -> str:
    value = trajectory.model_dump()
    value.pop("run_id", None)
    value.pop("elapsed_s", None)
    for step in value.get("steps", []):
        step.pop("elapsed_s", None)
    return _hash_json(value)


def _source_for_policy(policy_kind: str) -> str | None:
    return KNOWN_POLICY_SOURCES.get(policy_kind)


def _assistant_content(step: Any) -> tuple[str, bool]:
    """Return the actual generation when available, otherwise a teacher action JSON."""

    if step.model_attempts:
        raw = step.model_attempts[-1].get("raw_output")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("missing_raw_generation")
        parsed, _ = parse_action(raw)
        if parsed != step.action:
            raise ValueError("generation_action_mismatch")
        return raw, True
    return _json(step.action.model_dump()), False


def trajectory_to_example(
    trajectory: Trajectory,
    record: dict[str, Any],
    trajectory_source: str,
) -> dict[str, Any]:
    """Convert one verified trajectory without adding reference SQL to context."""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": trajectory.task.prompt},
        {"role": "user", "content": _json(trajectory.initial_observation)},
    ]
    assistant_indices: list[int] = []
    raw_generation_available = True
    for step in trajectory.steps:
        content, was_raw = _assistant_content(step)
        raw_generation_available &= was_raw
        assistant_indices.append(len(messages))
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": _json(step.tool_result)})

    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "example_id": f"{trajectory.task.task_id}:{trajectory.run_id}",
        "task_id": trajectory.task.task_id,
        "messages": messages,
        "assistant_message_indices": assistant_indices,
        "assistant_only_loss": True,
        "source": {
            "source_split": "train",
            "trajectory_source": trajectory_source,
            "policy_kind": trajectory.policy_kind,
            "policy_version": trajectory.policy_version,
            "trajectory_run_id": trajectory.run_id,
            "database_id": record["database_id"],
            "database_content_sha256": record["database_sha256"],
            "database_physical_sha256": trajectory.database_sha256,
            "reward_success": True,
            "reward_value": trajectory.reward.value if trajectory.reward else None,
            "raw_generation_available": raw_generation_available,
            "verified_success": bool(trajectory.reward and trajectory.reward.success),
            "actual_tool_execution": any(step.tool_name for step in trajectory.steps),
            "verifier_success": bool(trajectory.reward and trajectory.reward.success),
        },
    }


def _validate_example(example: dict[str, Any]) -> None:
    def contains_reference_sql_field(value: Any) -> bool:
        if isinstance(value, dict):
            return "reference_sql" in value or any(
                contains_reference_sql_field(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_reference_sql_field(item) for item in value)
        return False

    if contains_reference_sql_field(example):
        raise ValueError("reference_sql_leakage")
    messages = example.get("messages")
    indices = example.get("assistant_message_indices")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages_missing")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("invalid_initial_message_roles")
    if not isinstance(indices, list) or not indices:
        raise ValueError("assistant_targets_missing")
    for index in indices:
        if type(index) is not int or index < 0 or index >= len(messages):
            raise ValueError("assistant_target_index_invalid")
        if messages[index].get("role") != "assistant":
            raise ValueError("assistant_target_role_invalid")
        parse_action(messages[index]["content"])
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "user":
            raise ValueError("tool_observation_missing")


def _candidate_reasons(
    trajectory: Trajectory,
    source: TrainSource,
    expected_source: str | None,
    allowed_sources: set[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    task_id = trajectory.task.task_id
    reasons: list[str] = []
    details: dict[str, Any] = {"task_id": task_id}
    split = source.all_task_splits.get(task_id)
    details["source_split"] = split
    if split is None:
        reasons.append("unknown_task")
        return reasons, details
    if split != "train":
        reasons.append(f"source_split_not_train:{split}")
        return reasons, details
    record = source.records[task_id]
    database = source.database_root / f"{record['database_id']}.sqlite3"
    details["database_id"] = record["database_id"]
    if trajectory.task != BenchmarkTask.model_validate(record["task"]).public:
        reasons.append("public_task_mismatch")
    if trajectory.database_sha256 != database_fingerprint(database):
        reasons.append("database_physical_fingerprint_mismatch")
    inferred_source = _source_for_policy(trajectory.policy_kind)
    details["trajectory_source"] = inferred_source
    if inferred_source is None:
        reasons.append("unknown_policy_kind")
    elif expected_source is not None and inferred_source != expected_source:
        reasons.append("trajectory_source_mismatch")
    elif allowed_sources is not None and inferred_source not in allowed_sources:
        reasons.append("trajectory_source_not_allowed")

    try:
        replay_result = replay(trajectory, database)
        details["replay"] = replay_result
        if not replay_result.get("matched"):
            reasons.append("trajectory_replay_mismatch")
    except Exception as exc:
        details["replay_error"] = f"{type(exc).__name__}:{exc}"
        reasons.append("trajectory_replay_error")

    task = BenchmarkTask.model_validate(record["task"])
    try:
        rescored = verify(task, trajectory, database)
        details["rescored_reward"] = rescored.model_dump()
        if trajectory.reward is None:
            reasons.append("reward_missing")
        elif trajectory.reward.model_dump() != rescored.model_dump():
            reasons.append("reward_mismatch")
        if not rescored.success:
            reasons.append("verifier_success_required")
        if rescored.value <= 0:
            reasons.append("reward_not_positive")
        details["reward_consistent"] = trajectory.reward is not None and (
            trajectory.reward.model_dump() == rescored.model_dump()
        )
        details["verifier_success"] = rescored.success
    except Exception as exc:
        details["verifier_error"] = f"{type(exc).__name__}:{exc}"
        reasons.append("verifier_error")

    details["actual_tool_execution"] = any(step.tool_name for step in trajectory.steps)
    if not details["actual_tool_execution"]:
        reasons.append("actual_tool_execution_required")

    if trajectory.terminal_reason != "completed":
        reasons.append("trajectory_not_completed")
    try:
        example = trajectory_to_example(trajectory, record, inferred_source or "unknown")
        _validate_example(example)
        details["format_valid"] = True
    except Exception as exc:
        details["format_error"] = f"{type(exc).__name__}:{exc}"
        reasons.append("format_invalid")
    return reasons, details


def _quality_flags(
    accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]
) -> dict[str, Any]:
    accepted_details = [item["details"] for item in accepted]
    all_train = all(item.get("source_split") == "train" for item in accepted_details)
    return {
        "schema_validation": {
            "passed": all(item.get("schema_valid", False) for item in accepted),
            "accepted_count": len(accepted),
        },
        "verifier_validation": {
            "passed": all("rescored_reward" in item for item in accepted_details)
            and all(item["rescored_reward"].get("success") is True for item in accepted_details),
            "accepted_count": len(accepted),
        },
        "reward_validation": {
            "passed": all(
                item.get("rescored_reward", {}).get("value", 0) > 0
                and "reward_mismatch" not in item.get("reasons", [])
                for item in accepted
            ),
            "accepted_count": len(accepted),
        },
        "provenance_validation": {
            "passed": all_train and all(item.get("trajectory_source") for item in accepted_details),
            "accepted_count": len(accepted),
        },
        "format_validation": {
            "passed": all(item.get("format_valid", False) for item in accepted_details),
            "accepted_count": len(accepted),
        },
        "execution_validation": {
            "passed": all(
                item.get("replay", {}).get("matched") is True
                and item.get("actual_tool_execution") is True
                for item in accepted_details
            ),
            "accepted_count": len(accepted),
        },
        "reward_consistency_validation": {
            "passed": all(item.get("reward_consistent") is True for item in accepted_details),
            "accepted_count": len(accepted),
        },
        "rejected_candidate_count": len(rejected),
    }


def build_sft_dataset(
    manifest_path: Path,
    database_root: Path,
    trajectory_dir: Path,
    output_dir: Path,
    *,
    dataset_version: str = DATASET_VERSION,
    expected_trajectory_source: str | None = "deterministic_fixture_expert",
    allowed_trajectory_sources: set[str] | None = None,
) -> dict[str, Any]:
    """Build JSONL SFT data and audit artifacts from verified train trajectories."""

    source = load_train_source(manifest_path, database_root)
    files = sorted(trajectory_dir.glob("*.json"))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}

    for path in files:
        try:
            trajectory = TrajectoryStore.load(path)
        except Exception as exc:
            rejected.append(
                {
                    "path": str(path),
                    "task_id": None,
                    "reasons": ["trajectory_schema_invalid"],
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        reasons, details = _candidate_reasons(
            trajectory,
            source,
            expected_trajectory_source,
            allowed_trajectory_sources,
        )
        details["path"] = str(path)
        details["trajectory_schema_valid"] = True
        content_hash = _trajectory_content_hash(trajectory)
        details["trajectory_content_sha256"] = content_hash
        if not reasons and content_hash in seen_content:
            reasons.append("duplicate_trajectory")
            details["duplicate_of"] = seen_content[content_hash]
        if reasons:
            rejected.append(
                {
                    "path": str(path),
                    "task_id": trajectory.task.task_id,
                    "reasons": reasons,
                    "details": details,
                }
            )
            continue
        seen_content[content_hash] = trajectory.run_id
        record = source.records[trajectory.task.task_id]
        example = trajectory_to_example(trajectory, record, details["trajectory_source"])
        example["source"].update(
            {
                "trajectory_path": str(path),
                "trajectory_content_sha256": content_hash,
                "actual_tool_execution": details.get("actual_tool_execution", False),
                "verifier_success": details.get("verifier_success", False),
                "reward_consistent": details.get("reward_consistent", False),
                "template_id": record.get("template"),
            }
        )
        _validate_example(example)
        accepted.append(
            {
                "trajectory": trajectory,
                "record": record,
                "example": example,
                "details": details,
                "reasons": reasons,
                "schema_valid": True,
                "trajectory_source": details["trajectory_source"],
                "rescored_reward": details["rescored_reward"],
                "content_hash": content_hash,
            }
        )

    accepted.sort(key=lambda item: (item["trajectory"].task.task_id, item["trajectory"].run_id))
    examples = [item["example"] for item in accepted]
    for example in examples:
        _validate_example(example)

    task_counts = Counter(item["trajectory"].task.task_id for item in accepted)
    template_by_task = {
        record["task"]["public"]["task_id"]: record.get("template")
        for record in source.records.values()
    }
    source_templates = sorted({record.get("template") for record in source.records.values()})
    accepted_templates = sorted(
        {template_by_task[item["trajectory"].task.task_id] for item in accepted}
    )
    exact_duplicate_count = sum(1 for item in rejected if "duplicate_trajectory" in item["reasons"])
    duplicate_task_count = sum(count > 1 for count in task_counts.values())
    quality = _quality_flags(accepted, rejected)
    quality["duplicate_validation"] = {
        "passed": exact_duplicate_count == 0,
        "duplicate_trajectory_count": exact_duplicate_count,
        "tasks_with_multiple_trajectories": duplicate_task_count,
    }
    quality["duplicate_trajectory_count"] = exact_duplicate_count
    quality_passed = all(
        item["passed"] for item in quality.values() if isinstance(item, dict) and "passed" in item
    )
    accepted_task_count = len(task_counts)
    if not accepted:
        final_decision = "BLOCKED_ON_SFT_DATA"
    elif (
        not quality_passed
        or accepted_task_count < len(source.records)
        or set(accepted_templates) != set(source_templates)
    ):
        final_decision = "SFT_DATA_INSUFFICIENT"
    else:
        final_decision = "READY_FOR_SFT_TRAINING"

    validation_leakage = [
        item["trajectory"].task.task_id
        for item in accepted
        if source.all_task_splits.get(item["trajectory"].task.task_id) == "validation"
    ]
    test_leakage = [
        item["trajectory"].task.task_id
        for item in accepted
        if source.all_task_splits.get(item["trajectory"].task.task_id) == "test"
    ]
    policy_sources = sorted({item["trajectory_source"] for item in accepted})
    policy_kinds = sorted({item["trajectory"].policy_kind for item in accepted})
    policy_versions = sorted({item["trajectory"].policy_version for item in accepted})
    trajectory_source_distribution = dict(Counter(item["trajectory_source"] for item in accepted))

    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "source_split": "train",
        "source_task_count": len(source.records),
        "successful_task_count": accepted_task_count,
        "trajectory_count": len(examples),
        "verified_successful_trajectories": len(examples),
        "unique_tasks": accepted_task_count,
        "average_trajectories_per_task": len(examples) / accepted_task_count
        if accepted_task_count
        else 0,
        "source_template_count": len(source_templates),
        "accepted_template_count": len(accepted_templates),
        "source_templates": source_templates,
        "accepted_templates": accepted_templates,
        "duplicates": {
            "exact_trajectory_count": exact_duplicate_count,
            "tasks_with_multiple_trajectories": duplicate_task_count,
            "task_counts": dict(sorted(task_counts.items())),
        },
        "validation_used_for_training": False,
        "test_used_for_training": False,
        "validation_leakage": {
            "count": len(validation_leakage),
            "accepted_task_ids": validation_leakage,
        },
        "test_leakage": {"count": len(test_leakage), "accepted_task_ids": test_leakage},
        "policy_sources": policy_sources,
        "trajectory_source_distribution": trajectory_source_distribution,
        "policy_kinds": policy_kinds,
        "policy_versions": policy_versions,
        "teacher": {
            "trajectory_source": policy_sources,
            "policy_kind": policy_kinds,
            "reference_answers_in_model_context": False,
            "raw_model_generation_available": any(
                item["example"]["source"]["raw_generation_available"] for item in accepted
            ),
        },
        "verifier": "enterprise_agent_rl.rewards.verify",
        "verifier_version": "enterprise_agent_rl.rewards.verify",
        "reward_version": reward_version(),
        "manifest_sha256": source.manifest_sha256,
        "database_fingerprint_kind": source.database_fingerprint_kind,
        "created_from_commit": _git_commit(),
        "quality": quality,
        "final_decision": final_decision,
    }

    source_manifest = {
        "schema_version": "1.0",
        "source_split": "train",
        "manifest_path": str(manifest_path),
        "manifest_sha256": source.manifest_sha256,
        "benchmark_version": source.benchmark_version,
        "database_fingerprint_kind": source.database_fingerprint_kind,
        "split_counts": source.split_counts,
        "train_task_ids": sorted(source.records),
        "train_database_ids": sorted(source.database_entries),
        "selected_task_ids": sorted(task_counts),
        "rejected_candidates": rejected,
        "forbidden_split_task_counts": {
            "validation": sum(split == "validation" for split in source.all_task_splits.values()),
            "test": sum(split == "test" for split in source.all_task_splits.values()),
        },
        "validation_used_for_training": False,
        "test_used_for_training": False,
    }
    quality_report = {
        "schema_version": "1.0",
        "dataset_version": dataset_version,
        "source_split": "train",
        "candidate_files": len(files),
        "accepted_trajectories": len(accepted),
        "accepted_unique_tasks": accepted_task_count,
        "rejected_candidates": len(rejected),
        "accepted_task_ids": sorted(task_counts),
        "accepted_templates": accepted_templates,
        "trajectory_source_distribution": trajectory_source_distribution,
        "quality": quality,
        "rejection_reason_counts": dict(
            sorted(Counter(reason for item in rejected for reason in item["reasons"]).items())
        ),
        "validation_leakage_count": len(validation_leakage),
        "test_leakage_count": len(test_leakage),
        "reference_sql_in_examples": False,
        "final_decision": final_decision,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    dataset_path.write_text("".join(_json(example) + "\n" for example in examples))
    (output_dir / "dataset-metadata.json").write_text(_json(metadata) + "\n")
    (output_dir / "quality-report.json").write_text(_json(quality_report) + "\n")
    (output_dir / "source-manifest.json").write_text(_json(source_manifest) + "\n")
    return metadata | {"output_dir": str(output_dir), "quality_report": quality_report}
