"""Continuous optimizer/scheduler/scaler update orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .budget import BudgetError, BudgetGuard
from .io import atomic_write_json, canonical_hash, read_json
from .state import recursive_fingerprint


def _policy_snapshot(policy: Any, *, round_index: int, successful_updates: int) -> dict[str, Any]:
    if hasattr(policy, "policy_snapshot"):
        value = policy.policy_snapshot(
            round_index=round_index, successful_updates=successful_updates
        )
    else:
        fingerprint = (
            policy.policy_fingerprint() if hasattr(policy, "policy_fingerprint") else "unknown"
        )
        value = {"delta_fingerprint": fingerprint}
    if not isinstance(value, dict):
        raise TypeError("policy_snapshot_must_be_object")
    if not value.get("policy_snapshot_id"):
        value = {
            **value,
            "policy_snapshot_id": (
                f"step-{successful_updates}-{value.get('delta_fingerprint', 'unknown')}"
            ),
        }
    value.setdefault("successful_updates", successful_updates)
    value.setdefault("round_index", round_index)
    return value


def _state_fingerprint(policy: Any) -> dict[str, Any]:
    if hasattr(policy, "state_fingerprints"):
        value = policy.state_fingerprints()
        if not isinstance(value, dict):
            raise TypeError("policy_state_fingerprints_must_be_object")
        return value
    values = {}
    for name in ("optimizer", "scheduler", "scaler"):
        instance = getattr(policy, name, None)
        if instance is not None and hasattr(instance, "state_dict"):
            state = instance.state_dict()
            try:
                values[name] = recursive_fingerprint(state)
            except TypeError:
                # Keep the fallback usable for injected torch-like policies;
                # production QwenPolicy supplies deterministic fingerprints.
                values[name] = recursive_fingerprint(_jsonable_state(state))
    return values


def _jsonable_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_state(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_state(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


class ContinuousTrainer:
    """Run exactly two in-process updates per round without rebuilding policy state."""

    def __init__(self, *, run_dir: Path, guard: BudgetGuard, policy: Any):
        self.run_dir = run_dir
        self.guard = guard
        self.policy = policy

    def train_round(
        self, *, round_index: int, dataset: dict[str, Any], first_step: int, updates: int = 2
    ) -> dict[str, Any]:
        if updates not in (1, 2):
            raise ValueError("updates_per_call_must_be_one_or_two")
        counts = dataset.get("counts", {})
        records = list(dataset.get("records", []))
        if not records or int(counts.get("total_generated_tokens", 0)) <= 0:
            raise RuntimeError(f"round_has_no_training_tokens:{round_index}")
        if int(counts.get("active_physical_requests", 0)) + int(
            counts.get("zero_advantage_physical_requests", 0)
        ) != len(records):
            raise RuntimeError("active_zero_advantage_contract")
        if int(counts.get("nondegenerate_groups", 0)) < 1:
            raise RuntimeError(f"NO_LEARNING_SIGNAL:round{round_index}")
        state_path = self.run_dir / "training" / "state.json"
        state = read_json(state_path, {}) or {}
        completed = int(state.get("successful_updates", first_step))
        if completed != first_step:
            raise RuntimeError(f"continuous_state_cursor_mismatch:{completed}:{first_step}")
        round_dir = self.run_dir / "rounds" / f"r{round_index}" / "training"
        round_dir.mkdir(parents=True, exist_ok=True)
        input_hash = canonical_hash(records)
        metrics_path = round_dir / "metrics.json"
        prior_metrics = read_json(metrics_path, {}) or {}
        prior_rows = {
            int(row["logical_step"]): row
            for row in prior_metrics.get("updates", [])
            if isinstance(row, dict) and row.get("logical_step") is not None
        }
        result_rows: list[dict[str, Any]] = [prior_rows[key] for key in sorted(prior_rows)]
        logical_step = first_step
        for _ in range(updates):
            logical_step += 1
            if logical_step in prior_rows:
                continue
            self.guard.check(f"round{round_index}:update:{logical_step}")
            before = _state_fingerprint(self.policy)
            try:
                self.guard.record_call("optimizer_update_attempts")

                def update_policy(records=records, logical_step=logical_step):
                    return self.policy.update(
                        records,
                        logical_step=logical_step,
                        total_tokens=int(counts["total_generated_tokens"]),
                    )

                update = self.guard.run_callable(
                    update_policy,
                    phase=f"update:{logical_step}",
                )
            except BudgetError:
                raise
            if not isinstance(update, dict):
                raise TypeError("policy_update_must_return_object")
            after = _state_fingerprint(self.policy)
            snapshot = _policy_snapshot(
                self.policy, round_index=round_index, successful_updates=logical_step
            )
            row = {
                "schema_version": "grpo-update-v1",
                "round_index": round_index,
                "logical_step": logical_step,
                "input_fingerprint": input_hash,
                "policy_snapshot_id": snapshot["policy_snapshot_id"],
                "before_state": before,
                "after_state": after,
                "update": update,
            }
            if hasattr(self.policy, "save_checkpoint"):
                checkpoint_metadata = {
                    "round_index": round_index,
                    "input_fingerprint": input_hash,
                    "source_fingerprint": dataset.get("source_fingerprint"),
                    "cursor": {
                        "round_index": round_index,
                        "logical_step": logical_step,
                        "record_count": len(records),
                    },
                    "policy_snapshot": snapshot,
                }

                def save_checkpoint(
                    logical_step=logical_step, checkpoint_metadata=checkpoint_metadata
                ):
                    return self.policy.save_checkpoint(
                        self.run_dir,
                        logical_step=logical_step,
                        metadata=checkpoint_metadata,
                    )

                checkpoint = self.guard.run_callable(
                    save_checkpoint,
                    phase=f"checkpoint:{logical_step}",
                )
                row["checkpoint"] = checkpoint
            result_rows.append(row)
            prior_rows[logical_step] = row
            self.guard.record_call("successful_optimizer_updates")
            metrics_history = (
                read_json(self.run_dir / "training" / "metrics-history.json", []) or []
            )
            if not isinstance(metrics_history, list):
                raise RuntimeError("metrics_history_invalid")
            history_by_step = {
                int(item["logical_step"]): item
                for item in metrics_history
                if isinstance(item, dict) and item.get("logical_step") is not None
            }
            history_by_step[logical_step] = row
            cumulative_metrics = [history_by_step[key] for key in sorted(history_by_step)]
            atomic_write_json(
                self.run_dir / "training" / "metrics-history.json",
                cumulative_metrics,
            )
            state = {
                "schema_version": "grpo-training-state-v1",
                "status": "UPDATE_COMPLETE",
                "successful_updates": logical_step,
                "round_index": round_index,
                "input_fingerprint": input_hash,
                "policy_snapshot": snapshot,
                "optimizer_state": after.get("optimizer"),
                "scheduler_state": after.get("scheduler"),
                "scaler_state": after.get("scaler"),
                "rng_fingerprint": after.get("rng"),
                # The durable run state is cumulative across both rounds;
                # round-local metrics remain in rounds/r*/training/metrics.json.
                "metrics": cumulative_metrics,
                "source_fingerprint": dataset.get("source_fingerprint"),
            }
            atomic_write_json(state_path, state)
        summary = {
            "schema_version": "grpo-round-training-v1",
            "status": "PASS",
            "round_index": round_index,
            "logical_updates": logical_step - first_step,
            "successful_updates_total": logical_step,
            "input_fingerprint": input_hash,
            "updates": result_rows,
            "final_policy_snapshot": _policy_snapshot(
                self.policy, round_index=round_index, successful_updates=logical_step
            ),
        }
        atomic_write_json(metrics_path, summary)
        return summary

    def current_snapshot(self, *, round_index: int, successful_updates: int) -> dict[str, Any]:
        return _policy_snapshot(
            self.policy, round_index=round_index, successful_updates=successful_updates
        )
