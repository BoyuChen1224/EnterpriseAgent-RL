"""Serializable contracts between rollout, worker, and update components."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


def _as_float_list(values: list[Any], label: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"non_finite_{label}")
    return result


@dataclass
class TaskInput:
    task_id: str
    group_id: str
    public_task: dict[str, Any]
    database_path: str | None = None
    verifier_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenRecord:
    episode_id: str
    task_id: str
    group_id: str
    sample_index: int
    request_index: int
    seed: int
    policy_snapshot_id: str
    model_ref_id: str
    request_id: str
    response_id: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    response_mask: list[int]
    raw_policy_log_probs: list[float]
    behavior_sampling_log_probs: list[float]
    action_text: str | None = None
    action_status: str = "accepted"
    error: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    advantage: float | None = None

    @classmethod
    def new_id(cls) -> str:
        return str(uuid.uuid4())

    def validate(self) -> None:
        if len(self.generated_token_ids) != len(self.raw_policy_log_probs):
            raise ValueError(
                f"token_logprob_length_mismatch:{self.episode_id}:{self.request_index}"
            )
        if len(self.generated_token_ids) != len(self.behavior_sampling_log_probs):
            raise ValueError(
                f"token_behavior_length_mismatch:{self.episode_id}:{self.request_index}"
            )
        if len(self.response_mask) != len(self.prompt_token_ids) + len(self.generated_token_ids):
            raise ValueError(
                f"response_mask_length_mismatch:{self.episode_id}:{self.request_index}"
            )
        expected = [0] * len(self.prompt_token_ids) + [1] * len(self.generated_token_ids)
        if self.response_mask != expected:
            raise ValueError(f"response_mask_contract:{self.episode_id}:{self.request_index}")
        _as_float_list(self.raw_policy_log_probs, "raw_log_probs")
        _as_float_list(self.behavior_sampling_log_probs, "behavior_log_probs")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "grpo-token-record-v1",
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "sample_index": self.sample_index,
            "request_index": self.request_index,
            "physical_request_index": int(
                self.metadata.get("physical_request_index", self.request_index)
            ),
            "seed": self.seed,
            "policy_snapshot_id": self.policy_snapshot_id,
            "model_ref_id": self.model_ref_id,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "response_mask": self.response_mask,
            "raw_policy_log_probs": self.raw_policy_log_probs,
            "old_per_token_logps": self.raw_policy_log_probs,
            "behavior_sampling_log_probs": self.behavior_sampling_log_probs,
            "completion_mask": [1] * len(self.generated_token_ids),
            "generated_token_count": len(self.generated_token_ids),
            "action_text": self.action_text,
            "action_status": self.action_status,
            "error": self.error,
            "metadata": self.metadata,
            "advantage": self.advantage,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TokenRecord:
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value["task_id"]),
            group_id=str(value["group_id"]),
            sample_index=int(value["sample_index"]),
            request_index=int(value.get("request_index", value.get("physical_request_index", 0))),
            seed=int(value["seed"]),
            policy_snapshot_id=str(value["policy_snapshot_id"]),
            model_ref_id=str(value.get("model_ref_id", "unknown")),
            request_id=str(value.get("request_id", value.get("event_id", "unknown"))),
            response_id=str(value.get("response_id", value.get("event_id", "unknown"))),
            prompt_token_ids=[int(item) for item in value.get("prompt_token_ids", [])],
            generated_token_ids=[int(item) for item in value.get("generated_token_ids", [])],
            response_mask=[int(item) for item in value.get("response_mask", [])],
            raw_policy_log_probs=[float(item) for item in value.get("raw_policy_log_probs", [])],
            behavior_sampling_log_probs=[
                float(item) for item in value.get("behavior_sampling_log_probs", [])
            ],
            action_text=value.get("action_text"),
            action_status=str(value.get("action_status", "accepted")),
            error=value.get("error"),
            metadata=dict(value.get("metadata") or {}),
            advantage=value.get("advantage"),
        )


@dataclass
class EpisodeSample:
    episode_id: str
    task_id: str
    group_id: str
    sample_index: int
    seed: int
    policy_snapshot_id: str
    model_ref_id: str
    token_records: list[TokenRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    failed_action: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.token_records and self.failed_action is None:
            raise ValueError(f"episode_has_no_evidence:{self.episode_id}")
        for record in self.token_records:
            record.validate()
            if record.episode_id != self.episode_id:
                raise ValueError("episode_token_binding_mismatch")
            if record.policy_snapshot_id != self.policy_snapshot_id:
                raise ValueError("episode_snapshot_binding_mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "grpo-episode-sample-v1",
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "policy_snapshot_id": self.policy_snapshot_id,
            "model_ref_id": self.model_ref_id,
            "token_records": [row.to_dict() for row in self.token_records],
            "metadata": self.metadata,
            "failed_action": self.failed_action,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodeSample:
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value["task_id"]),
            group_id=str(value["group_id"]),
            sample_index=int(value["sample_index"]),
            seed=int(value["seed"]),
            policy_snapshot_id=str(value["policy_snapshot_id"]),
            model_ref_id=str(value.get("model_ref_id", "unknown")),
            token_records=[TokenRecord.from_dict(row) for row in value.get("token_records", [])],
            metadata=dict(value.get("metadata") or {}),
            failed_action=value.get("failed_action"),
        )


@dataclass
class ScoreResult:
    episode_id: str
    task_id: str
    status: str
    reward_value: float | None = None
    success: bool | None = None
    reason: str | None = None
    sql_action_count: int = 0
    sql_ok_count: int = 0
    worker_count: int | None = None
    runtime_worker_sql_count: int | None = None
    runtime_worker_non_sql_count: int | None = None
    verifier_sql_count: int | None = None
    budget_rejection_count: int = 0
    infra_error: Any = None
    error: Any = None
    verifier_record_id: str | None = None
    source_kind: str = "real"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.status not in {"scored", "unknown"}:
            raise ValueError(f"invalid_score_status:{self.status}")
        if self.status == "scored":
            if self.reward_value is None or self.success is None:
                raise ValueError("scored_result_missing_reward")
            if not math.isfinite(float(self.reward_value)):
                raise ValueError("scored_result_nonfinite_reward")
        for value in (
            self.sql_action_count,
            self.sql_ok_count,
            self.worker_count,
            self.budget_rejection_count,
            self.runtime_worker_sql_count,
            self.runtime_worker_non_sql_count,
            self.verifier_sql_count,
        ):
            if value is not None and int(value) < 0:
                raise ValueError("negative_score_counter")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "grpo-score-v1",
            "record_type": "RESULT",
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "status": self.status,
            "reward_value": self.reward_value,
            "success": self.success,
            "reason": self.reason,
            "sql_action_count": self.sql_action_count,
            "sql_ok_count": self.sql_ok_count,
            "worker_count": self.worker_count,
            "runtime_worker_sql_count": self.runtime_worker_sql_count,
            "runtime_worker_non_sql_count": self.runtime_worker_non_sql_count,
            "verifier_sql_count": self.verifier_sql_count,
            "budget_rejection_count": self.budget_rejection_count,
            "infra_error": self.infra_error,
            "error": self.error,
            "verifier_record_id": self.verifier_record_id or self.episode_id,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScoreResult:
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value["task_id"]),
            status=str(value["status"]),
            reward_value=value.get("reward_value"),
            success=value.get("success"),
            reason=value.get("reason"),
            sql_action_count=int(value.get("sql_action_count", 0)),
            sql_ok_count=int(value.get("sql_ok_count", 0)),
            worker_count=(
                int(value["worker_count"]) if value.get("worker_count") is not None else None
            ),
            runtime_worker_sql_count=(
                int(value["runtime_worker_sql_count"])
                if value.get("runtime_worker_sql_count") is not None
                else None
            ),
            runtime_worker_non_sql_count=(
                int(value["runtime_worker_non_sql_count"])
                if value.get("runtime_worker_non_sql_count") is not None
                else None
            ),
            verifier_sql_count=(
                int(value["verifier_sql_count"])
                if value.get("verifier_sql_count") is not None
                else None
            ),
            budget_rejection_count=int(value.get("budget_rejection_count", 0)),
            infra_error=value.get("infra_error"),
            error=value.get("error"),
            verifier_record_id=value.get("verifier_record_id"),
            source_kind=str(value.get("source_kind", "real")),
            metadata=dict(value.get("metadata") or {}),
        )


class RolloutEngine(Protocol):
    def rollout(
        self,
        task: TaskInput,
        *,
        round_index: int,
        sample_index: int,
        seed: int,
        policy_snapshot: dict[str, Any],
    ) -> EpisodeSample: ...


class ScoreWorker(Protocol):
    def score(self, task: TaskInput, episode: EpisodeSample) -> ScoreResult: ...


class TrainablePolicy(Protocol):
    def policy_snapshot(self, *, round_index: int, successful_updates: int) -> dict[str, Any]: ...

    def update(
        self, records: list[dict[str, Any]], *, logical_step: int, total_tokens: int
    ) -> dict[str, Any]: ...

    def save_checkpoint(
        self, run_dir: Any, *, logical_step: int, metadata: dict[str, Any]
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...
