"""Dedicated Qwen3-8B inference-only loading and rollout configuration.

The formal GRPO configuration intentionally remains a Qwen2.5/1.5B training
contract.  This module is the explicit, smaller contract for T-033: it
validates one fixed Qwen3 revision and one frozen manual-LoRA parent, then
loads a model without constructing any optimizer, scaler, scheduler, or
trainable delta.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from enterprise_agent_rl.grpo.config import GrpoConfig, RunContext
from enterprise_agent_rl.grpo.io import atomic_write_json, canonical_hash, sha256_file
from enterprise_agent_rl.grpo.torch_backend import (
    QwenPolicy,
    _record_gpu_device,
    _require_v100_device,
    _torch,
)
from enterprise_agent_rl.inference_stability import transformers_attention_implementation
from enterprise_agent_rl.model_protocol import render_chat_template

QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_DTYPE = "float16"
QWEN3_ATTENTION_BACKEND = "sdpa_math"
QWEN3_CONTEXT_TOKENS = 4096
QWEN3_MODULE_COUNT = 252
QWEN3_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN3_PARENT_WEIGHT_SHA256 = "413d1b0211ba93b2df322dcdaeb1fa1c3f019210c8ded510a2920e5ad6e114e9"
QWEN3_PARENT_CONFIG_SHA256 = "560d44e512b9745703519005087804dbcf6b819e2b6e1694d33571f9e346ed37"

QWEN3_MODULE_NAMES = tuple(
    f"model.layers.{layer}.{prefix}.{module}"
    for layer in range(36)
    for prefix, module in (
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    )
)


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    result = values.get(name, {})
    if not isinstance(result, dict):
        raise ValueError(f"qwen3_rollout_section_not_table:{name}")
    return dict(result)


def _require(values: dict[str, Any], key: str) -> Any:
    if key not in values:
        raise ValueError(f"qwen3_rollout_key_missing:{key}")
    return values[key]


def _resolved_path(value: str, context: RunContext) -> Path:
    path = Path(value)
    return path if path.is_absolute() else context.repo_root / path


def validate_qwen3_rollout_values(values: dict[str, Any]) -> None:
    """Fail closed on every experiment-defining value."""

    model = _section(values, "model")
    lora = _section(values, "lora")
    rollout = _section(values, "rollout")
    runtime = _section(values, "runtime")
    budget = _section(values, "budget")
    slurm = _section(values, "slurm")
    limits = _section(values, "limits")

    if _require(model, "model_id") != QWEN3_MODEL_ID:
        raise ValueError("qwen3_rollout_model_id_not_allowlisted")
    if _require(model, "revision") != QWEN3_REVISION:
        raise ValueError("qwen3_rollout_revision_not_allowlisted")
    if _require(model, "dtype") != QWEN3_DTYPE:
        raise ValueError("qwen3_rollout_dtype_mismatch")
    if _require(model, "attention_backend") != QWEN3_ATTENTION_BACKEND:
        raise ValueError("qwen3_rollout_attention_backend_mismatch")
    if int(_require(model, "max_context_tokens")) != QWEN3_CONTEXT_TOKENS:
        raise ValueError("qwen3_rollout_context_mismatch")
    if _require(model, "enable_thinking") is not False:
        raise ValueError("qwen3_rollout_requires_hard_thinking_off")
    if _require(model, "parent_adapter_sha256") != QWEN3_PARENT_WEIGHT_SHA256:
        raise ValueError("qwen3_rollout_parent_weight_identity_mismatch")
    if _require(model, "parent_adapter_config_sha256") != QWEN3_PARENT_CONFIG_SHA256:
        raise ValueError("qwen3_rollout_parent_config_identity_mismatch")

    if int(_require(lora, "rank")) != 8 or float(_require(lora, "alpha")) != 16:
        raise ValueError("qwen3_rollout_lora_shape_mismatch")
    if float(_require(lora, "dropout")) != 0:
        raise ValueError("qwen3_rollout_lora_dropout_mismatch")
    if tuple(_require(lora, "target_modules")) != QWEN3_TARGET_MODULES:
        raise ValueError("qwen3_rollout_lora_targets_mismatch")
    if int(_require(lora, "module_count")) != QWEN3_MODULE_COUNT:
        raise ValueError("qwen3_rollout_lora_module_count_mismatch")

    if int(_require(rollout, "rounds")) != 1:
        raise ValueError("qwen3_rollout_requires_one_round")
    if int(_require(rollout, "group_count")) != 4:
        raise ValueError("qwen3_rollout_group_count_mismatch")
    if int(_require(rollout, "samples_per_task")) != 4:
        raise ValueError("qwen3_rollout_samples_per_task_mismatch")
    if int(_require(rollout, "seed_start")) != 74 or int(_require(rollout, "seed_end")) != 89:
        raise ValueError("qwen3_rollout_seed_range_mismatch")
    generation = _section(rollout, "generation")
    if float(_require(generation, "temperature")) != 1.0:
        raise ValueError("qwen3_rollout_temperature_mismatch")
    if float(_require(generation, "top_p")) != 1.0:
        raise ValueError("qwen3_rollout_top_p_mismatch")
    if int(_require(generation, "top_k")) != 0:
        raise ValueError("qwen3_rollout_top_k_mismatch")
    if int(_require(generation, "max_tokens")) != 512:
        raise ValueError("qwen3_rollout_max_tokens_mismatch")
    if int(_require(generation, "format_retries")) != 1:
        raise ValueError("qwen3_rollout_format_retries_mismatch")
    if _require(generation, "parse_mode") != "json_fence":
        raise ValueError("qwen3_rollout_parse_mode_mismatch")
    if _require(generation, "enable_thinking") is not False:
        raise ValueError("qwen3_rollout_generation_thinking_mismatch")
    if _require(generation, "sampling_seed_rule") != (
        "episode_seed + physical_request_index_zero_based"
    ):
        raise ValueError("qwen3_rollout_sampling_seed_rule_mismatch")

    expected_runtime = {
        "seed": 17,
        "max_actions": 20,
        "max_tool_calls": 14,
        "episode_timeout_s": 600,
        "inference_timeout_s": 120,
        "sql_timeout_s": 2,
        "sql_max_rows": 100,
        "sql_max_bytes": 65536,
        "sql_memory_mb": 256,
        "step_penalty": 0.0,
        "repeat_penalty": 0.0,
        "error_penalty": 0.0,
        "recovery_bonus": 0.0,
    }
    for key, expected in expected_runtime.items():
        if _require(runtime, key) != expected:
            raise ValueError(f"qwen3_rollout_runtime_mismatch:{key}")

    expected_budget = {
        "run_hard_seconds": 5400,
        "job_hard_seconds": 3600,
        "soft_seconds": 3480,
        "cleanup_reserve_seconds": 120,
        "gpu_count": 1,
    }
    for key, expected in expected_budget.items():
        if int(_require(budget, key)) != expected:
            raise ValueError(f"qwen3_rollout_budget_mismatch:{key}")
    for name, value in limits.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"qwen3_rollout_limit_invalid:{name}")
    if _require(slurm, "account") != "pi_daipeishan" or _require(slurm, "qos") != "gpuq":
        raise ValueError("qwen3_rollout_slurm_identity_mismatch")
    partitions = tuple(_require(slurm, "partitions"))
    if partitions != ("gpu2Q", "gpu4Q", "gpu4RQ", "gpu8Q"):
        raise ValueError("qwen3_rollout_partitions_mismatch")


def load_qwen3_rollout_config(
    config_path: Path,
    context: RunContext,
    *,
    cache_dir: Path | None = None,
    parent_adapter: Path | None = None,
) -> tuple[GrpoConfig, RunContext]:
    """Load the explicit rollout-only TOML without invoking old validation."""

    config_path = config_path.expanduser().resolve()
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("qwen3_rollout_config_root_not_table")
    validate_qwen3_rollout_values(values)
    resolved_context = context.resolved()
    model = _section(values, "model")
    if cache_dir is not None:
        model["cache_dir"] = str(cache_dir.expanduser().resolve())
    else:
        model["cache_dir"] = str(
            _resolved_path(str(_require(model, "cache_dir")), resolved_context)
        )
    if parent_adapter is not None:
        model["parent_adapter"] = str(parent_adapter.expanduser().resolve())
    else:
        model["parent_adapter"] = str(
            _resolved_path(str(_require(model, "parent_adapter")), resolved_context)
        )
    values["model"] = model
    resolved_context = RunContext(resolved_context.repo_root, resolved_context.run_dir, config_path)
    return (
        GrpoConfig(
            raw=values,
            model=model,
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
        resolved_context,
    )


def _adapter_identity(adapter_dir: Path, config: GrpoConfig) -> dict[str, Any]:
    weight_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weight_path.is_file() or not config_path.is_file():
        raise RuntimeError("qwen3_rollout_parent_adapter_files_missing")
    weight_sha = sha256_file(weight_path)
    config_sha = sha256_file(config_path)
    if weight_sha != QWEN3_PARENT_WEIGHT_SHA256 or weight_sha != str(
        config.model["parent_adapter_sha256"]
    ):
        raise RuntimeError("qwen3_rollout_parent_adapter_weight_hash_mismatch")
    if config_sha != QWEN3_PARENT_CONFIG_SHA256 or config_sha != str(
        config.model["parent_adapter_config_sha256"]
    ):
        raise RuntimeError("qwen3_rollout_parent_adapter_config_hash_mismatch")
    adapter = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter, dict):
        raise RuntimeError("qwen3_rollout_parent_adapter_config_not_object")
    expected = {
        "base_model_name_or_path": QWEN3_MODEL_ID,
        "base_model_revision": QWEN3_REVISION,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": list(QWEN3_TARGET_MODULES),
        "implementation": "enterprise_agent_rl.manual_lora_v1",
    }
    for key, value in expected.items():
        if adapter.get(key) != value:
            raise RuntimeError(f"qwen3_rollout_parent_config_mismatch:{key}")
    if adapter.get("module_names") != list(QWEN3_MODULE_NAMES):
        raise RuntimeError("qwen3_rollout_parent_module_names_mismatch")
    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_model_sha256": weight_sha,
        "adapter_config_sha256": config_sha,
        "module_count": len(adapter["module_names"]),
        "rank": int(adapter["r"]),
        "alpha": float(adapter["lora_alpha"]),
        "implementation": adapter["implementation"],
        "base_model_id": adapter["base_model_name_or_path"],
        "base_model_revision": adapter["base_model_revision"],
    }


def _tokenizer_preflight(tokenizer: Any, context: RunContext, config: GrpoConfig) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("qwen3_rollout_tokenizer_chat_template_missing")
    template_bytes = template.encode("utf-8")
    rendered = render_chat_template(
        tokenizer,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("qwen3_rollout_chat_template_render_not_string")
    inputs = tokenizer(rendered, return_tensors="pt")
    prompt_length = int(inputs["input_ids"].shape[-1])
    result = {
        "schema_version": "t033-tokenizer-preflight-v1",
        "model_id": config.model["model_id"],
        "revision": config.model["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "raw_template_utf8_bytes": len(template_bytes),
        "raw_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "canonical_template_sha256": canonical_hash(template),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "hard_thinking_off": True,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "rendered_prompt_token_count": prompt_length,
        "context_limit": int(config.model["max_context_tokens"]),
        "max_new_tokens": int(config.rollout["generation"]["max_tokens"]),
        "context_boundary_accepts": prompt_length + int(config.rollout["generation"]["max_tokens"])
        <= int(config.model["max_context_tokens"]),
    }
    atomic_write_json(context.run_dir / "setup" / "tokenizer-preflight.json", result)
    return result


def load_qwen3_rollout_policy(context: RunContext, config: GrpoConfig, guard: Any) -> QwenPolicy:
    """Fresh Qwen3 base + one frozen SFT merge; no train state is touched."""

    torch = _torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("qwen3_rollout_requires_one_visible_cuda_device")
    device_info = _require_v100_device(torch)
    _record_gpu_device(context, device_info, mode="rollout", arm=None)
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("qwen3_rollout_determinism_environment_mismatch:PYTHONHASHSEED")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("qwen3_rollout_determinism_environment_mismatch:CUBLAS_WORKSPACE_CONFIG")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    model_cfg = config.model
    cache_dir = Path(str(model_cfg["cache_dir"])).resolve()
    parent = Path(str(model_cfg["parent_adapter"])).resolve()
    adapter_identity = _adapter_identity(parent, config)
    guard.check("rollout-setup:parent-assets-verified")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["model_id"]),
        revision=str(model_cfg["revision"]),
        cache_dir=str(cache_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_preflight = _tokenizer_preflight(tokenizer, context, config)
    guard.check("rollout-setup:tokenizer-preflight")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg["model_id"]),
        revision=str(model_cfg["revision"]),
        cache_dir=str(cache_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        attn_implementation=transformers_attention_implementation(QWEN3_ATTENTION_BACKEND),
    )
    model.to("cuda:0")
    guard.check("rollout-setup:fresh-base-loaded")

    from enterprise_agent_rl.lora import merge_lora_adapter

    parent_merge = merge_lora_adapter(
        model,
        parent,
        expected_model_id=QWEN3_MODEL_ID,
        expected_revision=QWEN3_REVISION,
    )
    if int(parent_merge.get("module_count", -1)) != QWEN3_MODULE_COUNT:
        raise RuntimeError(
            f"qwen3_rollout_merged_module_count_mismatch:{parent_merge.get('module_count')}"
        )
    guard.check("rollout-setup:parent-sft-merged")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("qwen3_rollout_parameter_freeze_failed")
    model.config.use_cache = True
    model.eval()

    policy = QwenPolicy(
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        scheduler=None,
        scaler=None,
        context=context,
        config=config,
        guard=guard,
        module_names=[],
        parent_merge={
            "fresh_base_load_count": 1,
            "parent_merge_count": 1,
            "delta_merge_count": 0,
            "module_count": QWEN3_MODULE_COUNT,
            "adapter_identity": adapter_identity,
            "tokenizer_preflight": tokenizer_preflight,
            "optimizer_created": False,
            "scheduler_created": False,
            "scaler_created": False,
            "training_state_loaded": False,
        },
        device=torch.device("cuda:0"),
        mode="rollout_only",
        evaluation_arm=None,
    )
    policy.model_ref_id = f"{QWEN3_MODEL_ID}@{QWEN3_REVISION}@sft-rollout"
    policy.rollout_identity = {
        "model_id": QWEN3_MODEL_ID,
        "revision": QWEN3_REVISION,
        "dtype": QWEN3_DTYPE,
        "attention_backend": QWEN3_ATTENTION_BACKEND,
        "enable_thinking": False,
        "fresh_base_load_count": 1,
        "parent_adapter_merge_count": 1,
        "parent_adapter": adapter_identity,
        "module_count": QWEN3_MODULE_COUNT,
        "model_object_id": id(model),
        "all_parameters_frozen": True,
        "optimizer_created": False,
        "scheduler_created": False,
        "scaler_created": False,
        "training_state_loaded": False,
    }
    atomic_write_json(context.run_dir / "setup" / "rollout-load.json", policy.rollout_identity)
    return policy


def qwen3_policy_snapshot(config: GrpoConfig, policy: QwenPolicy) -> dict[str, Any]:
    """Stable identity supplied to every physical model request."""

    identity = dict(getattr(policy, "rollout_identity", {}))
    identity_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "t033-qwen3-rollout-policy-snapshot-v1",
        "policy_snapshot_id": f"qwen3-sft-t032-{QWEN3_PARENT_WEIGHT_SHA256[:16]}",
        "base_model": config.model["model_id"],
        "base_revision": config.model["revision"],
        "model_ref_id": policy.model_ref_id,
        "parent_N_adapter_sha256": QWEN3_PARENT_WEIGHT_SHA256,
        "parent_N_adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
        "delta_fingerprint": None,
        "model_weight_fingerprint": identity_hash,
        "rollout_identity_sha256": identity_hash,
        "optimizer_or_scheduler_changed_by_sampling": False,
        "training_calls": {
            "backward": 0,
            "optimizer_create": 0,
            "optimizer_step": 0,
            "updates": 0,
        },
    }


def expected_module_names() -> tuple[str, ...]:
    """Public test hook for the fixed 36x7 Qwen3 identity."""

    return QWEN3_MODULE_NAMES


__all__ = [
    "QWEN3_ATTENTION_BACKEND",
    "QWEN3_CONTEXT_TOKENS",
    "QWEN3_DTYPE",
    "QWEN3_MODEL_ID",
    "QWEN3_MODULE_COUNT",
    "QWEN3_PARENT_CONFIG_SHA256",
    "QWEN3_PARENT_WEIGHT_SHA256",
    "QWEN3_REVISION",
    "QWEN3_TARGET_MODULES",
    "expected_module_names",
    "load_qwen3_rollout_config",
    "load_qwen3_rollout_policy",
    "qwen3_policy_snapshot",
    "validate_qwen3_rollout_values",
]
