"""Task-grouped online rollout, durable token binding, and advantage building."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .budget import BudgetError, BudgetGuard
from .contracts import EpisodeSample, ScoreResult, TaskInput, TokenRecord
from .evidence import qualify_episode_evidence
from .io import (
    append_jsonl,
    atomic_write_json,
    canonical_hash,
    immutable_write_json,
    read_json,
    read_jsonl,
)


def _seed_for(
    round_index: int,
    group_index: int,
    sample_index: int,
    seed_start: int,
    group_count: int = 4,
    samples_per_task: int = 4,
) -> int:
    return (
        seed_start
        + round_index * group_count * samples_per_task
        + group_index * samples_per_task
        + sample_index
    )


def _score_error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:1000]}


def _task_fingerprint(tasks: Iterable[TaskInput], policy_snapshot: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "group_id": task.group_id,
                    "public_task": task.public_task,
                    "database_path": task.database_path,
                    "metadata": task.metadata,
                }
                for task in tasks
            ],
            "policy_snapshot": policy_snapshot,
        }
    )


def _load_existing_scores(path: Path) -> dict[str, ScoreResult]:
    result: dict[str, ScoreResult] = {}
    for row in read_jsonl(path):
        if row.get("record_type", "RESULT") != "RESULT":
            continue
        score = ScoreResult.from_dict(row)
        score.validate()
        prior = result.get(score.episode_id)
        if prior is not None and prior.to_dict() != score.to_dict():
            if prior.status == "unknown" and score.status == "scored":
                # A retryable infrastructure failure may be superseded after a
                # local fix; preserve both JSONL rows and use the latest score.
                result[score.episode_id] = score
                continue
            raise RuntimeError(f"conflicting_score_duplicate:{score.episode_id}")
        result[score.episode_id] = score
    return result


def _load_existing_episodes(path: Path) -> dict[str, EpisodeSample]:
    result: dict[str, EpisodeSample] = {}
    for row in read_jsonl(path):
        if row.get("record_type") not in {None, "RAW", "EPISODE"}:
            continue
        episode = EpisodeSample.from_dict(row)
        episode.validate()
        prior = result.get(episode.episode_id)
        if prior is not None and prior.to_dict() != episode.to_dict():
            if (
                prior.failed_action is not None
                and prior.failed_action.get("status") == "engine_error"
                and episode.token_records
            ):
                # Keep the failed attempt as evidence and let a later valid
                # retry become the effective episode for this logical slot.
                result[episode.episode_id] = episode
                continue
            raise RuntimeError(f"conflicting_episode_duplicate:{episode.episode_id}")
        result[episode.episode_id] = episode
    return result


class OnlineRollout:
    """The same runner is used by the real backend and CPU fakes."""

    def __init__(
        self,
        *,
        run_dir: Path,
        guard: BudgetGuard,
        seed_start: int = 74,
        group_count: int = 4,
    ):
        self.run_dir = run_dir
        self.guard = guard
        self.seed_start = seed_start
        self.group_count = group_count

    def run_round(
        self,
        *,
        round_index: int,
        tasks: list[TaskInput],
        samples_per_task: int,
        policy_snapshot: dict[str, Any],
        engine: Any,
        worker: Any,
    ) -> dict[str, Any]:
        if not tasks:
            raise ValueError("round_requires_tasks")
        if samples_per_task < 1:
            raise ValueError("samples_per_task_must_be_positive")
        round_dir = self.run_dir / "rounds" / f"r{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _task_fingerprint(tasks, policy_snapshot)
        marker_path = round_dir / "rollout-status.json"
        existing_marker = read_json(marker_path, {}) or {}
        if existing_marker.get("status") == "COMPLETED":
            if existing_marker.get("input_fingerprint") != fingerprint:
                raise RuntimeError(f"completed_round_input_fingerprint_changed:{round_index}")
            dataset = read_json(round_dir / "training" / "dataset-index.json")
            if not dataset:
                raise RuntimeError(f"completed_round_dataset_missing:{round_index}")
            if dataset.get("input_fingerprint") != fingerprint:
                raise RuntimeError(f"completed_round_dataset_fingerprint_changed:{round_index}")
            if dataset.get("policy_snapshot_id") != policy_snapshot.get("policy_snapshot_id"):
                raise RuntimeError(f"completed_round_dataset_snapshot_changed:{round_index}")
            return {**existing_marker, "dataset": dataset, "resumed": True}

        episodes_path = round_dir / "rollouts" / "episodes.jsonl"
        tokens_path = round_dir / "rollouts" / "token-rollouts.jsonl"
        scores_path = round_dir / "rollouts" / "formal-scores.jsonl"
        existing_episodes = _load_existing_episodes(episodes_path)
        existing_scores = _load_existing_scores(scores_path)
        expected: list[tuple[int, TaskInput, int]] = []
        for group_index, task in enumerate(tasks):
            for sample_index in range(samples_per_task):
                expected.append((group_index, task, sample_index))

        technical_errors: list[dict[str, Any]] = []
        for group_index, task, sample_index in expected:
            episode_id = f"r{round_index}-{task.group_id}-s{sample_index}"
            prior_score = existing_scores.get(episode_id)
            if prior_score is not None:
                if prior_score.status == "unknown":
                    technical_errors.append(
                        {
                            "episode_id": episode_id,
                            "error": prior_score.infra_error or prior_score.error,
                        }
                    )
                    previous_episode = existing_episodes.get(episode_id)
                    retryable_engine_failure = bool(
                        previous_episode is not None
                        and not previous_episode.token_records
                        and previous_episode.failed_action is not None
                        and previous_episode.failed_action.get("status") == "engine_error"
                    )
                    if retryable_engine_failure:
                        existing_scores.pop(episode_id, None)
                        existing_episodes.pop(episode_id, None)
                    else:
                        continue
                else:
                    continue
            self.guard.check(f"round{round_index}:episode:{episode_id}")
            seed = _seed_for(
                round_index,
                group_index,
                sample_index,
                self.seed_start,
                self.group_count,
                samples_per_task,
            )
            episode: EpisodeSample
            episode = existing_episodes.get(episode_id)
            if episode is None:
                try:

                    def invoke_engine(
                        task=task,
                        round_index=round_index,
                        sample_index=sample_index,
                        seed=seed,
                        policy_snapshot=policy_snapshot,
                        episode_id=episode_id,
                    ):
                        return self._call_engine(
                            engine,
                            task,
                            round_index=round_index,
                            sample_index=sample_index,
                            seed=seed,
                            policy_snapshot=policy_snapshot,
                            episode_id=episode_id,
                        )

                    episode = self.guard.run_callable(invoke_engine, phase=f"generate:{episode_id}")
                    if episode.episode_id != episode_id:
                        raise ValueError("engine_episode_id_mismatch")
                    if episode.policy_snapshot_id != policy_snapshot.get("policy_snapshot_id"):
                        raise ValueError("engine_policy_snapshot_mismatch")
                    episode.validate()
                except BudgetError:
                    raise
                except Exception as exc:
                    episode = EpisodeSample(
                        episode_id=episode_id,
                        task_id=task.task_id,
                        group_id=task.group_id,
                        sample_index=sample_index,
                        seed=seed,
                        policy_snapshot_id=str(
                            policy_snapshot.get("policy_snapshot_id", "unknown")
                        ),
                        model_ref_id=str(policy_snapshot.get("model_ref_id", "unknown")),
                        token_records=[],
                        failed_action={"status": "engine_error", "error": _score_error(exc)},
                    )
                    technical_errors.append({"episode_id": episode_id, "error": _score_error(exc)})
                episode_row = {"record_type": "RAW", **episode.to_dict(), "raw_saved": True}
                append_jsonl(episodes_path, episode_row)
                for token in episode.token_records:
                    append_jsonl(
                        tokens_path, {"record_type": "TOKEN", **token.to_dict(), "raw_saved": True}
                    )
                existing_episodes[episode_id] = episode
            trajectory_path = self._persist_trajectory(round_dir, episode)
            qualification = None
            if getattr(engine, "evidence", None) is not None:
                qualification = qualify_episode_evidence(
                    episode,
                    run_dir=self.run_dir,
                    expected_stage="training",
                    expected_arm="train",
                    trajectory_path=trajectory_path,
                )
            try:
                self.guard.check(f"round{round_index}:worker:{episode_id}")

                def invoke_worker(task=task, episode=episode):
                    return worker.score(task, episode)

                score = self.guard.run_callable(
                    invoke_worker,
                    phase=f"worker:{episode_id}",
                )
                score.validate()
            except BudgetError:
                raise
            except Exception as exc:
                score = ScoreResult(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    status="unknown",
                    infra_error=_score_error(exc),
                    error=_score_error(exc),
                    source_kind="real",
                )
                technical_errors.append({"episode_id": episode_id, "error": _score_error(exc)})
            score_row = {
                **score.to_dict(),
                "raw_saved": True,
                "episode_path": str(episodes_path.relative_to(self.run_dir)),
            }
            if trajectory_path is not None:
                score_row["trajectory_path"] = trajectory_path
            if qualification is not None:
                score_row["qualification"] = qualification
            append_jsonl(scores_path, score_row)
            existing_scores[episode_id] = score

        scores = [
            existing_scores.get(f"r{round_index}-{task.group_id}-s{sample_index}")
            for _, task, sample_index in expected
        ]
        if any(score is None for score in scores):
            raise RuntimeError(f"round_score_cardinality:{round_index}")
        score_values = [score for score in scores if score is not None]
        advantages_by_episode: dict[str, float] = {}
        grouped: dict[str, list[ScoreResult]] = defaultdict(list)
        for score in score_values:
            if score.status == "scored" and score.reward_value is not None:
                grouped[score.task_id].append(score)
        for group_scores in grouped.values():
            rewards = [float(score.reward_value) for score in group_scores]
            mean = statistics.mean(rewards)
            std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            for score in group_scores:
                advantages_by_episode[score.episode_id] = (
                    0.0 if std == 0.0 else (float(score.reward_value) - mean) / (std + 1e-4)
                )

        dataset_records: list[dict[str, Any]] = []
        active = 0
        zero = 0
        total_tokens = 0
        empty_token_records = 0
        for _, task, sample_index in expected:
            episode_id = f"r{round_index}-{task.group_id}-s{sample_index}"
            episode = existing_episodes.get(episode_id)
            score = existing_scores[episode_id]
            if episode is None or score.status != "scored":
                continue
            advantage = float(advantages_by_episode.get(episode_id, 0.0))
            for token in episode.token_records:
                if not token.generated_token_ids:
                    # Keep the failed Action in the raw episode/evidence files,
                    # but never send an empty completion to the GRPO loss.
                    empty_token_records += 1
                    continue
                row = token.to_dict()
                row["advantage"] = advantage
                row["reward_value"] = score.reward_value
                row["success"] = score.success
                row["verifier_record_id"] = score.verifier_record_id or episode_id
                row["source_kind"] = score.source_kind
                dataset_records.append(row)
                total_tokens += len(token.generated_token_ids)
                if advantage == 0.0:
                    zero += 1
                else:
                    active += 1

        dataset = {
            "schema_version": "grpo-dataset-index-v1",
            "round_index": round_index,
            "input_fingerprint": fingerprint,
            "source_fingerprint": fingerprint,
            "policy_snapshot_id": policy_snapshot.get("policy_snapshot_id"),
            "records": dataset_records,
            "counts": {
                "planned_episodes": len(expected),
                "scored_episodes": sum(score.status == "scored" for score in score_values),
                "unknown_episodes": sum(score.status == "unknown" for score in score_values),
                "physical_requests": len(dataset_records),
                "total_generated_tokens": total_tokens,
                "active_physical_requests": active,
                "zero_advantage_physical_requests": zero,
                "empty_token_records_skipped": empty_token_records,
                "nondegenerate_groups": sum(
                    len(
                        {
                            float(score.reward_value)
                            for score in values
                            if score.reward_value is not None
                        }
                    )
                    > 1
                    for values in grouped.values()
                ),
            },
        }
        atomic_write_json(round_dir / "training" / "dataset-index.json", dataset)
        status = (
            "COMPLETED"
            if dataset["counts"]["scored_episodes"] == len(expected)
            and dataset["counts"]["unknown_episodes"] == 0
            else "INCOMPLETE"
        )
        result = {
            "schema_version": "grpo-round-rollout-v1",
            "round_index": round_index,
            "status": status,
            "input_fingerprint": fingerprint,
            "planned": len(expected),
            "attempted": len(existing_scores),
            "scored": dataset["counts"]["scored_episodes"],
            "unknown": dataset["counts"]["unknown_episodes"],
            "technical_errors": technical_errors,
            "source_kind": sorted({score.source_kind for score in score_values}),
            "dataset": dataset,
        }
        atomic_write_json(marker_path, result)
        return result

    def _persist_trajectory(self, round_dir: Path, episode: EpisodeSample) -> str | None:
        trajectory = episode.metadata.get("trajectory")
        if not isinstance(trajectory, dict):
            return None
        path = round_dir / "rollouts" / "trajectories" / f"{episode.episode_id}.json"
        immutable_write_json(path, trajectory)
        return str(path.relative_to(self.run_dir))

    @staticmethod
    def _call_engine(
        engine: Any, task: TaskInput, *, episode_id: str, **kwargs: Any
    ) -> EpisodeSample:
        if hasattr(engine, "rollout"):
            episode = engine.rollout(task, **kwargs)
            if not isinstance(episode, EpisodeSample):
                raise TypeError("engine_rollout_must_return_episode_sample")
            return episode
        if not hasattr(engine, "sample"):
            raise TypeError("engine_requires_rollout_or_sample")
        sample = engine.sample(task, **kwargs)
        if isinstance(sample, EpisodeSample):
            return sample
        if not isinstance(sample, TokenRecord):
            raise TypeError("engine_sample_must_return_token_record_or_episode")
        return EpisodeSample(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=int(kwargs["sample_index"]),
            seed=int(kwargs["seed"]),
            policy_snapshot_id=str(kwargs["policy_snapshot"]["policy_snapshot_id"]),
            model_ref_id=sample.model_ref_id,
            token_records=[sample],
        )
