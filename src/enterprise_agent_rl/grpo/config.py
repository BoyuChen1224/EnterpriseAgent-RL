"""Explicit repository/run/config context and GRPO configuration validation."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class RunContext:
    """All filesystem context is passed in; no historical path is inferred."""

    repo_root: Path
    run_dir: Path
    config_path: Path

    def resolved(self) -> RunContext:
        return RunContext(
            repo_root=self.repo_root.expanduser().resolve(),
            run_dir=self.run_dir.expanduser().resolve(),
            config_path=self.config_path.expanduser().resolve(),
        )


@dataclass(frozen=True)
class GrpoConfig:
    raw: dict[str, Any]
    model: dict[str, Any]
    lora: dict[str, Any]
    rollout: dict[str, Any]
    runtime: dict[str, Any]
    optimizer: dict[str, Any]
    budget: dict[str, Any]
    data: dict[str, Any]
    evaluation: dict[str, Any]
    slurm: dict[str, Any]
    limits: dict[str, Any]

    @property
    def rounds(self) -> int:
        return int(self.rollout["rounds"])

    @property
    def updates_per_round(self) -> int:
        return int(self.rollout["updates_per_round"])

    @property
    def samples_per_task(self) -> int:
        return int(self.rollout["samples_per_task"])

    @property
    def group_count(self) -> int:
        return int(self.rollout["group_count"])

    @property
    def planned_samples_per_round(self) -> int:
        return self.group_count * self.samples_per_task

    def path(self, key: str, context: RunContext) -> Path:
        value = self.data.get(key) or self.evaluation.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"config_path_missing:{key}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else context.repo_root / candidate


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    result = values.get(name, {})
    if not isinstance(result, dict):
        raise ValueError(f"config_section_not_table:{name}")
    return dict(result)


def _require(values: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    if key not in values:
        raise ValueError(f"config_key_missing:{key}")
    value = values[key]
    if not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool):
        raise ValueError(f"config_key_type:{key}:{type(value).__name__}")
    return value


def validate_values(values: dict[str, Any]) -> None:
    model = _section(values, "model")
    lora = _section(values, "lora")
    rollout = _section(values, "rollout")
    runtime = _section(values, "runtime")
    optimizer = _section(values, "optimizer")
    budget = _section(values, "budget")
    data = _section(values, "data")
    evaluation = _section(values, "evaluation")
    slurm = _section(values, "slurm")

    if _require(model, "model_id", str) != MODEL_ID:
        raise ValueError("unsupported_model_id")
    if _require(model, "revision", str) != MODEL_REVISION:
        raise ValueError("unsupported_model_revision")
    for key in ("parent_adapter", "parent_adapter_sha256", "parent_adapter_config_sha256"):
        _require(model, key, str)
    if _require(model, "max_context_tokens", int) < 1:
        raise ValueError("invalid_max_context_tokens")
    if _require(model, "attention_backend", str) != "sdpa_math":
        raise ValueError("attention_backend_must_be_sdpa_math")

    if _require(lora, "rank", int) != 8 or _require(lora, "alpha", (int, float)) != 16:
        raise ValueError("unsupported_lora_shape")
    if float(_require(lora, "dropout", (int, float))) != 0:
        raise ValueError("lora_dropout_must_be_zero")
    targets = tuple(_require(lora, "target_modules", list))
    if targets != TARGET_MODULES:
        raise ValueError("lora_target_modules_mismatch")

    for key in ("rounds", "updates_per_round", "samples_per_task", "group_count"):
        if int(_require(rollout, key, int)) < 1:
            raise ValueError(f"invalid_rollout_{key}")
    if int(rollout["rounds"]) != 2 or int(rollout["updates_per_round"]) != 2:
        raise ValueError("online_entrypoint_requires_two_rounds_two_updates")
    seed_start = _require(rollout, "seed_start", int)
    seed_end = _require(rollout, "seed_end", int)
    expected_seed_end = (
        seed_start
        + int(rollout["rounds"]) * int(rollout["group_count"]) * int(rollout["samples_per_task"])
        - 1
    )
    if seed_end != expected_seed_end:
        raise ValueError(f"seed_range_mismatch:{seed_start}:{seed_end}:{expected_seed_end}")
    generation = _section(rollout, "generation")
    if float(_require(generation, "temperature", (int, float))) != 1.0:
        raise ValueError("generation_temperature_mismatch")
    if float(_require(generation, "top_p", (int, float))) != 1.0:
        raise ValueError("generation_top_p_mismatch")
    if (
        int(_require(generation, "top_k", int)) != 0
        or int(_require(generation, "max_tokens", int)) != 512
    ):
        raise ValueError("generation_sampling_mismatch")

    for key in ("max_actions", "max_tool_calls", "sql_max_rows", "sql_max_bytes", "sql_memory_mb"):
        if int(_require(runtime, key, int)) <= 0:
            raise ValueError(f"invalid_runtime_{key}")
    for key in ("episode_timeout_s", "inference_timeout_s", "sql_timeout_s"):
        if float(_require(runtime, key, (int, float))) <= 0:
            raise ValueError(f"invalid_runtime_{key}")

    if float(_require(optimizer, "learning_rate", (int, float))) != 2e-6:
        raise ValueError("learning_rate_mismatch")
    if tuple(_require(optimizer, "betas", list)) != (0.9, 0.999):
        raise ValueError("optimizer_betas_mismatch")
    if bool(_require(optimizer, "foreach", bool)) or bool(_require(optimizer, "fused", bool)):
        raise ValueError("foreach_or_fused_optimizer_not_allowed")
    if float(_require(optimizer, "clip_norm", (int, float))) != 1:
        raise ValueError("clip_norm_mismatch")

    for key in ("job_hard_seconds", "run_hard_seconds", "soft_seconds", "cleanup_reserve_seconds"):
        if int(_require(budget, key, int)) <= 0:
            raise ValueError(f"invalid_budget_{key}")
    if int(budget["soft_seconds"]) >= int(budget["job_hard_seconds"]):
        raise ValueError("soft_budget_must_be_before_job_hard_budget")
    if int(budget["job_hard_seconds"]) > int(budget["run_hard_seconds"]):
        raise ValueError("job_budget_exceeds_run_budget")
    if int(budget["cleanup_reserve_seconds"]) >= int(budget["job_hard_seconds"]):
        raise ValueError("cleanup_reserve_too_large")

    for key in ("train_index", "evaluation_plan", "public_panel"):
        if not isinstance(data.get(key) or evaluation.get(key), str):
            raise ValueError(f"config_external_input_missing:{key}")
    if not isinstance(slurm.get("account"), str) or not isinstance(slurm.get("qos"), str):
        raise ValueError("slurm_account_qos_required_for_future_entry")

    limits = _section(values, "limits")
    for key, value in limits.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"invalid_call_limit:{key}")
    eval_generation = _section(evaluation, "generation")
    if eval_generation:
        if float(_require(eval_generation, "temperature", (int, float))) != 0.0:
            raise ValueError("evaluation_temperature_must_be_zero")
        if float(_require(eval_generation, "top_p", (int, float))) != 1.0:
            raise ValueError("evaluation_top_p_mismatch")
        if int(_require(eval_generation, "top_k", int)) != 0:
            raise ValueError("evaluation_top_k_mismatch")
        if int(_require(eval_generation, "max_tokens", int)) != 512:
            raise ValueError("evaluation_max_tokens_mismatch")
        if int(_require(eval_generation, "seed", int)) != 17:
            raise ValueError("evaluation_seed_mismatch")


def load_config(
    config_path: Path, context: RunContext | None = None
) -> tuple[GrpoConfig, RunContext]:
    config_path = config_path.expanduser().resolve()
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("config_root_not_table")
    validate_values(values)
    if context is None:
        repo_root_value = os.environ.get("EARL_REPO_ROOT")
        repo_root = Path(repo_root_value) if repo_root_value else Path.cwd()
        run_dir = Path(
            os.environ.get("EARL_GRPO_RUN", "artifacts/grpo-training-entrypoint/t022-v1")
        )
        context = RunContext(repo_root=repo_root, run_dir=run_dir, config_path=config_path)
    resolved = context.resolved()
    if resolved.config_path != config_path:
        resolved = RunContext(resolved.repo_root, resolved.run_dir, config_path)
    return (
        GrpoConfig(
            raw=values,
            model=_section(values, "model"),
            lora=_section(values, "lora"),
            rollout=_section(values, "rollout"),
            runtime=_section(values, "runtime"),
            optimizer=_section(values, "optimizer"),
            budget=_section(values, "budget"),
            data=_section(values, "data"),
            evaluation=_section(values, "evaluation"),
            slurm=_section(values, "slurm"),
            limits=_section(values, "limits"),
        ),
        resolved,
    )
