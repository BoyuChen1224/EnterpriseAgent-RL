"""Frozen N/RL evaluation using the same injected engine/worker boundary."""

# GRPO 训练后的冻结比较入口；区别于 rollout 探索，它使用一致设置评测指定模型臂。
# 先区分已评分业务失败与技术 unknown，再用固定计划分母比较成功数和成功率。
# 普通运行路径在 frozen_evaluation.py；本文件保留旧接口兼容，阅读时注意分支选择。

from __future__ import annotations

from pathlib import Path
from typing import Any

from .budget import BudgetError, BudgetGuard
from .contracts import EpisodeSample, TaskInput
from .frozen_evaluation import evaluate_frozen as _strict_evaluate_frozen
from .io import append_jsonl, atomic_write_json, canonical_hash, read_json, read_jsonl


# 保留历史路径，不能仅凭旧函数名把输出当成当前正式冻结评测口径。
def _legacy_evaluate_frozen(
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
) -> dict[str, Any]:
    if arm not in {"N", "RL"}:
        raise ValueError("evaluation_arm_must_be_N_or_RL")
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
    # N and RL are distinct fresh model objects even though both derive from
    # the same frozen final snapshot. Keep their physical identities distinct
    # in every persisted evaluation row and resume marker.
    arm_snapshot["policy_snapshot_id"] = (
        f"{policy_snapshot.get('policy_snapshot_id', 'final')}-{arm}"
    )
    fingerprint = {
        "arm": arm,
        "policy_snapshot_id": arm_snapshot.get("policy_snapshot_id"),
        "tasks": [task.task_id for task in tasks],
        "evaluation_seed": evaluation_seed,
        "evaluation_mode": "greedy",
    }
    prior = read_json(marker_path, {}) or {}
    if prior.get("status") == "COMPLETED":
        if prior.get("input") != fingerprint:
            raise RuntimeError(f"evaluation_input_changed:{arm}")
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
        prior_rows[task_id] = row
    if prior.get("input") is not None and prior.get("input") != fingerprint:
        raise RuntimeError(f"evaluation_input_changed:{arm}")
    scored = sum(row.get("status") == "scored" for row in prior_rows.values())
    unknown = sum(row.get("status") == "unknown" for row in prior_rows.values())
    successes = sum(
        bool(row.get("success")) for row in prior_rows.values() if row.get("status") == "scored"
    )
    for index, task in enumerate(tasks):
        existing = prior_rows.get(task.task_id)
        if existing is not None:
            if existing.get("policy_snapshot_id") not in {
                None,
                arm_snapshot.get("policy_snapshot_id"),
            }:
                raise RuntimeError(f"evaluation_snapshot_changed:{arm}:{task.task_id}")
            continue
        guard.check(f"evaluation:{arm}:{task.task_id}")
        episode_id = f"{arm}-{task.task_id}"
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
                raise TypeError("evaluation_engine_must_return_episode")
            score = guard.run_callable(
                lambda task=task, episode=episode: worker.score(task, episode),
                phase=f"evaluation:{arm}:verifier:{task.task_id}",
            )
            score.validate()
        except Exception as exc:
            if isinstance(exc, BudgetError):
                raise
            unknown += 1
            unknown_row = {
                "record_type": "RESULT",
                "task_id": task.task_id,
                "episode_id": episode_id,
                "status": "unknown",
                "unknown_reason": "evaluation_infrastructure_error",
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                "policy_snapshot_id": arm_snapshot.get("policy_snapshot_id"),
            }
            append_jsonl(results_path, unknown_row)
            prior_rows[task.task_id] = unknown_row
            continue
        row = {
            **score.to_dict(),
            "task_id": task.task_id,
            "policy_snapshot_id": arm_snapshot.get("policy_snapshot_id"),
            "raw_saved": True,
        }
        trajectory = episode.metadata.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory_path = output / "trajectories" / f"{episode_id}.json"
            atomic_write_json(trajectory_path, trajectory)
            row["trajectory_path"] = str(trajectory_path.relative_to(run_dir))
        append_jsonl(results_path, row)
        prior_rows[task.task_id] = row
        scored += int(score.status == "scored")
        unknown += int(score.status == "unknown")
        successes += int(bool(score.success)) if score.status == "scored" else 0
    result = {
        "schema_version": "grpo-evaluation-v1",
        "status": "COMPLETED" if scored == len(tasks) and unknown == 0 else "INCOMPLETE",
        "arm": arm,
        "input": fingerprint,
        "planned": len(tasks),
        "attempted": scored + unknown,
        "scored": scored,
        "unknown": unknown,
        "successes": successes,
        "results_path": str(results_path.relative_to(run_dir)),
        "evaluation_seed": evaluation_seed,
        "evaluation_mode": "greedy",
        "arm_snapshot": arm_snapshot,
        "arm_metadata": arm_metadata or {},
    }
    atomic_write_json(output / "arm-manifest.json", result)
    atomic_write_json(marker_path, result)
    return result


# 选择正式冻结评测实现或兼容分支；模型加载、记录与评分的详细流程继续到对应实现。
def evaluate_frozen(*args, **kwargs):
    """Public frozen evaluator; the strict implementation is shared by CLI/tests."""

    return _strict_evaluate_frozen(*args, **kwargs)
