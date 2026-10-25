"""Frozen evaluation through the shared engine, verifier, and evidence guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .budget import BudgetError, BudgetGuard
from .contracts import EpisodeSample, ScoreResult, TaskInput
from .evidence import EvidenceError, QualificationError, qualify_episode_evidence
from .io import (
    ImmutableWriteError,
    append_jsonl,
    atomic_write_json,
    canonical_hash,
    immutable_write_json,
    read_json,
    read_jsonl,
    sha256_file,
)


def _error_value(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:1000]}


def _technical_unknown(
    *,
    task: TaskInput,
    episode_id: str,
    policy_snapshot_id: str,
    reason: str,
    error: Any,
    trajectory_path: str | None = None,
    qualification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "record_type": "RESULT",
        "task_id": task.task_id,
        "episode_id": episode_id,
        "status": "unknown",
        "unknown_reason": reason,
        "technical_unknown": True,
        "error": error,
        "policy_snapshot_id": policy_snapshot_id,
        "raw_saved": True,
        "verifier_called": False,
    }
    if trajectory_path is not None:
        row["trajectory_path"] = trajectory_path
    if qualification is not None:
        row["qualification"] = qualification
    return row


def _task_fingerprint(tasks: list[TaskInput]) -> str:
    return canonical_hash(
        [
            {
                "task_id": task.task_id,
                "group_id": task.group_id,
                "public_task": task.public_task,
                "database_path": task.database_path,
                "metadata": task.metadata,
            }
            for task in tasks
        ]
    )


def evaluate_frozen(
    *,
    run_dir: Path,
    arm: str,
    tasks: list[TaskInput],
    engine: Any,
    worker: Any,
    guard: BudgetGuard,
    policy_snapshot: dict[str, Any],
    arm_metadata: dict[str, Any] | None = None,
    evaluation_seed: int = 17,
    require_qualified_evidence: bool = False,
) -> dict[str, Any]:
    """Evaluate one frozen arm and never score an unqualified model episode.

    ``require_qualified_evidence`` is enabled by the formal eval-only command.
    Historical CPU integration tests may continue to exercise the injected
    engine/worker boundary without pretending their synthetic rows are real
    model evidence.
    """

    if arm not in {"Base", "N", "RL"}:
        raise ValueError("evaluation_arm_must_be_Base_N_or_RL")
    output = run_dir / "evaluation" / arm
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "eval-status.json"
    arm_snapshot = {
        **policy_snapshot,
        "evaluation_arm": arm,
        "evaluation_mode": "greedy",
        "evaluation_seed": evaluation_seed,
        **(arm_metadata or {}),
    }
    arm_snapshot["policy_snapshot_id"] = (
        f"{policy_snapshot.get('policy_snapshot_id', 'final')}-{arm}"
    )
    fingerprint = {
        "arm": arm,
        "policy_snapshot_id": arm_snapshot.get("policy_snapshot_id"),
        "task_fingerprint": _task_fingerprint(tasks),
        "tasks": [task.task_id for task in tasks],
        "evaluation_seed": evaluation_seed,
        "evaluation_mode": "greedy",
        "require_qualified_evidence": bool(require_qualified_evidence),
    }
    prior = read_json(marker_path, {}) or {}
    if prior.get("input") is not None and prior.get("input") != fingerprint:
        raise RuntimeError(f"evaluation_input_changed:{arm}")
    if prior.get("status") == "COMPLETED" and not require_qualified_evidence:
        return prior

    results_path = output / "verifier-results.jsonl"
    prior_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(results_path):
        if row.get("record_type", "RESULT") != "RESULT":
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("evaluation_result_missing_task_id")
        if row.get("status") not in {"scored", "unknown"}:
            raise RuntimeError(f"evaluation_result_invalid_status:{task_id}")
        previous = prior_rows.get(task_id)
        if previous is not None and canonical_hash(previous) != canonical_hash(row):
            raise RuntimeError(f"evaluation_conflicting_duplicate:{task_id}")
        if require_qualified_evidence and row.get("status") == "scored":
            if row.get("qualification", {}).get("status") != "QUALIFIED":
                raise QualificationError(f"historical_score_not_qualified:{arm}:{task_id}")
        prior_rows[task_id] = row

    if prior.get("status") == "COMPLETED":
        if set(prior_rows) != {task.task_id for task in tasks}:
            raise QualificationError(f"completed_marker_rows_incomplete:{arm}")
        if any(row.get("status") != "scored" for row in prior_rows.values()):
            raise QualificationError(f"completed_marker_contains_unknown:{arm}")
        return prior

    scored = sum(row.get("status") == "scored" for row in prior_rows.values())
    unknown = sum(row.get("status") == "unknown" for row in prior_rows.values())
    successes = sum(
        bool(row.get("success")) for row in prior_rows.values() if row.get("status") == "scored"
    )
    stopped = bool(prior.get("stopped_before_verifier"))
    stop_reason = prior.get("stop_reason")

    for index, task in enumerate(tasks):
        existing = prior_rows.get(task.task_id)
        if existing is not None:
            if existing.get("status") == "unknown":
                stopped = bool(existing.get("technical_unknown", False)) or stopped
                stop_reason = stop_reason or existing.get("unknown_reason")
                if stopped:
                    break
            if existing.get("status") == "scored":
                if existing.get("policy_snapshot_id") not in {
                    None,
                    arm_snapshot.get("policy_snapshot_id"),
                }:
                    raise RuntimeError(f"evaluation_snapshot_changed:{arm}:{task.task_id}")
                continue
        if stopped:
            break
        guard.check(f"evaluation:{arm}:{task.task_id}")
        episode_id = f"{arm}-{task.task_id}"
        trajectory_path: str | None = None
        qualification: dict[str, Any] | None = None
        try:
            if hasattr(engine, "evaluate"):
                episode = engine.evaluate(
                    task,
                    policy_snapshot=arm_snapshot,
                    episode_id=episode_id,
                    seed=evaluation_seed,
                )
            else:
                episode = engine.rollout(
                    task,
                    round_index=-1,
                    sample_index=index,
                    seed=evaluation_seed,
                    policy_snapshot=arm_snapshot,
                    episode_id=episode_id,
                )
            if not isinstance(episode, EpisodeSample):
                raise QualificationError("evaluation_engine_must_return_episode")
            trajectory = episode.metadata.get("trajectory")
            if not isinstance(trajectory, dict):
                raise QualificationError(f"trajectory_missing:{episode_id}")
            trajectory_file = output / "trajectories" / f"{episode_id}.json"
            immutable_write_json(trajectory_file, trajectory)
            trajectory_path = str(trajectory_file.relative_to(run_dir))
            if require_qualified_evidence:
                qualification = qualify_episode_evidence(
                    episode,
                    run_dir=run_dir,
                    expected_stage="evaluation",
                    expected_arm=arm,
                    trajectory_path=trajectory_path,
                )
            else:
                qualification = {
                    "status": "NOT_REQUIRED",
                    "trajectory_sha256": sha256_file(trajectory_file),
                }
        except BudgetError:
            raise
        except (EvidenceError, ImmutableWriteError, OSError, ValueError, TypeError) as exc:
            unknown += 1
            unknown_row = _technical_unknown(
                task=task,
                episode_id=episode_id,
                policy_snapshot_id=str(arm_snapshot.get("policy_snapshot_id")),
                reason="evidence_qualification_failure",
                error=_error_value(exc),
                trajectory_path=trajectory_path,
                qualification=qualification,
            )
            append_jsonl(results_path, unknown_row)
            prior_rows[task.task_id] = unknown_row
            stopped = True
            stop_reason = "evidence_qualification_failure"
            break

        try:
            score = guard.run_callable(
                lambda task=task, episode=episode: worker.score(task, episode),
                phase=f"evaluation:{arm}:verifier:{task.task_id}",
            )
            score.validate()
        except BudgetError:
            raise
        except Exception as exc:
            score = ScoreResult(
                episode_id=episode_id,
                task_id=task.task_id,
                status="unknown",
                infra_error=_error_value(exc),
                error=_error_value(exc),
                source_kind="real",
            )
        row = {
            **score.to_dict(),
            "task_id": task.task_id,
            "policy_snapshot_id": arm_snapshot.get("policy_snapshot_id"),
            "raw_saved": True,
            "verifier_called": True,
            "trajectory_path": trajectory_path,
            "trajectory_sha256": qualification.get("trajectory_sha256") if qualification else None,
            "qualification": qualification,
        }
        append_jsonl(results_path, row)
        prior_rows[task.task_id] = row
        scored += int(score.status == "scored")
        unknown += int(score.status == "unknown")
        successes += int(bool(score.success)) if score.status == "scored" else 0

    result = {
        "schema_version": "grpo-evaluation-v2",
        "status": "COMPLETED" if scored == len(tasks) and unknown == 0 else "INCOMPLETE",
        "arm": arm,
        "input": fingerprint,
        "planned": len(tasks),
        "attempted": scored + unknown,
        "scored": scored,
        "unknown": unknown,
        "successes": successes,
        "stopped_before_verifier": bool(stopped),
        "stop_reason": stop_reason,
        "results_path": str(results_path.relative_to(run_dir)),
        "evaluation_seed": evaluation_seed,
        "evaluation_mode": "greedy",
        "arm_snapshot": arm_snapshot,
        "arm_metadata": arm_metadata or {},
    }
    atomic_write_json(output / "arm-manifest.json", result)
    atomic_write_json(marker_path, result)
    return result
