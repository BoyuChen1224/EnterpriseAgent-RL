#!/usr/bin/env python3
"""Run the fixed 54-decision Qwen3 Runtime SFT recipe.

Stages are intentionally separate so a fresh process can perform reload and
the validation workers can load the final adapter without retaining training
references.  ``prepare`` and ``tiny`` are CPU-only; ``train`` and ``reload``
require one visible CUDA device on a Slurm compute node.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from enterprise_agent_rl.inference_stability import attention_backend_context
from enterprise_agent_rl.lora import (
    inject_lora_modules,
    iter_lora_modules,
    load_lora_adapter,
    merge_lora_adapter,
    save_adapter,
)
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.qwen3_sft import (
    DATASET_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_MODULES,
    Qwen3SFTError,
    build_full_tokenized_dataset,
    load_tokenized_dataset,
    schedule_manifest,
    set_seed,
    sha256_file,
    train_lora_schedule,
    write_json,
    write_tokenized_artifacts,
)
from enterprise_agent_rl.sft_training import load_decision_rows

RUN_DEFAULT = Path("artifacts/qwen3-runtime-sft-comparison/t032-v1")
TRAINING_CONFIG_DEFAULT = Path("configs/training/qwen3-runtime-sft.toml")
MODEL_CONFIG_DEFAULT = Path("configs/model/qwen3-8b.toml")
DATASET_DEFAULT = Path(
    "artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl"
)
CACHE_DEFAULT = Path("artifacts/model-cache-qwen3-8b")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"status": "unavailable", "error": str(exc)}
    return {"status": result.returncode, "query": result.stdout.strip() or result.stderr.strip()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    temporary.replace(path)


def _validate_config(cfg: ModelSettings, train_cfg: dict[str, Any]) -> None:
    if cfg.model_id != MODEL_ID or cfg.revision != MODEL_REVISION:
        raise Qwen3SFTError("qwen3_model_identity_mismatch")
    if cfg.dtype != "float16" or cfg.attention_backend != "sdpa_math":
        raise Qwen3SFTError("qwen3_runtime_contract_mismatch")
    if cfg.generation.enable_thinking is not False:
        raise Qwen3SFTError("qwen3_requires_hard_thinking_off")
    if train_cfg.get("dataset_sha256") != DATASET_SHA256:
        raise Qwen3SFTError("training_dataset_sha256_config_mismatch")
    if train_cfg.get("max_samples") != 54 or train_cfg.get("max_context_tokens") != 2048:
        raise Qwen3SFTError("training_sample_contract_mismatch")
    if train_cfg.get("enable_thinking") is not False:
        raise Qwen3SFTError("training_thinking_contract_mismatch")
    if train_cfg.get("lora") != {
        "r": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_modules": list(TARGET_MODULES),
    }:
        raise Qwen3SFTError("lora_config_mismatch")
    training = train_cfg.get("training", {})
    required = {
        "seed": 42,
        "microbatch": 1,
        "gradient_accumulation_steps": 4,
        "epochs": 2,
        "successful_optimizer_updates": 27,
        "learning_rate": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 0.00000001,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "clip_norm": 1.0,
        "amp_initial_scale": 1024.0,
        "max_backward": 216,
        "max_amp_skip_replays": 3,
        "checkpoint_use_reentrant": True,
        "checkpoint_steps": [9, 18, 27],
    }
    if training != required:
        raise Qwen3SFTError(f"training_config_mismatch:{training}")


def _cache_data(cache: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = cache / "qwen-cache-manifest.json"
    if not manifest_path.is_file():
        raise Qwen3SFTError("cache_integrity_manifest_missing")
    data = json.loads(manifest_path.read_text())
    if data.get("model") != MODEL_ID or data.get("revision") != MODEL_REVISION:
        raise Qwen3SFTError("cache_integrity_manifest_mismatch")
    return manifest_path, data


def _load_model(torch_module: Any, cfg: ModelSettings, cache: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_module.float16,
        attn_implementation="sdpa",
        output_attentions=False,
        use_safetensors=True,
    ).to("cuda:0")
    if next(model.parameters()).dtype != torch_module.float16:
        raise Qwen3SFTError("model_not_fp16")
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise Qwen3SFTError("model_attention_backend_mismatch")
    if getattr(model.config, "output_attentions", False):
        raise Qwen3SFTError("output_attentions_must_be_false_for_sdpa")
    if getattr(model.config, "num_hidden_layers", None) != 36:
        raise Qwen3SFTError("qwen3_layer_count_mismatch")
    return model


def _environment(
    *,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
    cache_manifest: Path | None = None,
    cache_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "created_at_utc": _now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": {
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
            "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "nvidia_smi": _nvidia_smi(),
    }
    if torch_module is not None:
        capability = list(torch_module.cuda.get_device_capability(0))
        value.update(
            {
                "torch": torch_module.__version__,
                "cuda_runtime": torch_module.version.cuda,
                "device": "cuda:0",
                "gpu_name": torch_module.cuda.get_device_name(0),
                "compute_capability": capability,
                "cuda_arch_list": torch_module.cuda.get_arch_list(),
                "dtype": "float16",
                "attention_backend": "sdpa_math",
                "enable_thinking": False,
                "checkpoint_use_reentrant": True,
                "input_requires_grad": True,
            }
        )
    if transformers_module is not None:
        value["transformers"] = transformers_module.__version__
    if cache_manifest is not None and cache_data is not None:
        value["cache_manifest_sha256"] = sha256_file(cache_manifest)
        value["cache_size_bytes_unique"] = cache_data.get("cache_size_bytes_unique")
    return value


def _tokenizer(cfg: ModelSettings, cache: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare(args: argparse.Namespace) -> int:
    output = args.output
    train_cfg = tomllib.loads(args.training_config.read_text())
    cfg = ModelSettings.load(args.model_config)
    _validate_config(cfg, train_cfg)
    if sha256_file(args.dataset) != DATASET_SHA256:
        raise Qwen3SFTError("supervision_dataset_sha256_mismatch")
    tokenizer = _tokenizer(cfg, args.cache)
    rows = load_decision_rows(args.dataset, expected_sha256=DATASET_SHA256)
    examples, formatted, identity = build_full_tokenized_dataset(
        tokenizer,
        rows,
        max_context_tokens=2048,
        enable_thinking=False,
    )
    training_dir = output / "training"
    dataset_manifest = write_tokenized_artifacts(
        training_dir,
        source_path=args.dataset,
        source_sha256=DATASET_SHA256,
        examples=examples,
        formatted=formatted,
        config_identity=identity,
    )
    schedule = schedule_manifest(
        examples,
        epochs=train_cfg["training"]["epochs"],
        accumulation_steps=train_cfg["training"]["gradient_accumulation_steps"],
    )
    if schedule["microbatches"] != 108 or schedule["successful_updates"] != 27:
        raise Qwen3SFTError("schedule_contract_mismatch")
    write_json(training_dir / "schedule.json", schedule)
    write_json(
        output / "frozen-config.json",
        {
            "experiment": "T-032/E-029",
            "model": cfg.model_dump(),
            "training_config": train_cfg,
            "dataset_sha256": DATASET_SHA256,
            "source_row_count": len(rows),
            "source_action_kind_counts": Counter(row["target_action"]["kind"] for row in rows),
            "final_checkpoint_policy": "step27_only_for_evaluation",
            "base_adapter": "NONE",
            "validation_not_read_by_training": True,
            "dataset_manifest_sha256": sha256_file(training_dir / "dataset-manifest.json"),
            "schedule_sha256": sha256_file(training_dir / "schedule.json"),
        },
    )
    environment = _environment()
    environment.update(
        {
            "stage": "prepare",
            "tokenizer": {
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "class": type(tokenizer).__name__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "pad_token_id": tokenizer.pad_token_id,
            },
            "source_dataset": str(args.dataset.resolve()),
            "source_dataset_sha256": DATASET_SHA256,
            "output_dataset_sha256": dataset_manifest["output_dataset_sha256"],
            "validation_split_accessed": False,
            "test_split_accessed": False,
        }
    )
    write_json(output / "preflight.json", {"status": "passed", **environment})
    write_json(output / "environment.json", environment)
    return 0


def _validate_cuda(torch_module: Any, cache: Path) -> tuple[list[int], Path, dict[str, Any]]:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise Qwen3SFTError("requires_one_visible_cuda_device")
    capability = list(torch_module.cuda.get_device_capability(0))
    if tuple(capability) < (7, 0) or "sm_70" not in torch_module.cuda.get_arch_list():
        raise Qwen3SFTError("wheel_missing_volta_support_or_unsupported_device")
    cache_manifest, cache_data = _cache_data(cache)
    return capability, cache_manifest, cache_data


def _load_resume_state(torch_module: Any, checkpoint: Path) -> dict[str, Any]:
    cursor = json.loads((checkpoint / "cursor.json").read_text())
    return {
        **cursor,
        "optimizer_state": torch_module.load(
            checkpoint / "optimizer.pt", map_location="cuda:0", weights_only=False
        ),
        "scaler_state": torch_module.load(
            checkpoint / "scaler.pt", map_location="cpu", weights_only=False
        ),
        "rng_state": torch_module.load(
            checkpoint / "rng_state.pth", map_location="cpu", weights_only=False
        ),
    }


def _save_checkpoint(
    *,
    output: Path,
    state: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    cfg: ModelSettings,
    train_cfg: dict[str, Any],
    module_names: list[str],
    schedule_sha256: str,
) -> Path:
    step = int(state["update_index"])
    checkpoint = output / "training" / "checkpoints" / f"step-{step}"
    if checkpoint.exists() and any(checkpoint.iterdir()):
        existing = checkpoint / "checkpoint-manifest.json"
        if existing.is_file():
            return checkpoint
        raise Qwen3SFTError(f"refusing_nonempty_checkpoint:{checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    adapter_summary = save_adapter(
        model,
        checkpoint / "adapter",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        rank=8,
        alpha=16,
        dropout=0,
        target_modules=TARGET_MODULES,
        module_names=module_names,
    )
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch_module.save(state["optimizer"].state_dict(), checkpoint / "optimizer.pt")
    torch_module.save(state["scaler"].state_dict(), checkpoint / "scaler.pt")
    torch_module.save(state["rng_state"], checkpoint / "rng_state.pth")
    write_json(
        checkpoint / "cursor.json",
        {
            "cursor": state["cursor"],
            "successful_updates": state["update_index"],
            "backward_count": state["backward_count"],
            "replayed_groups": state["replayed_groups"],
            "schedule_sha256": schedule_sha256,
        },
    )
    write_json(
        checkpoint / "checkpoint-manifest.json",
        {
            "schema_version": "1.0",
            "status": "valid",
            "step": step,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapter": adapter_summary,
            "adapter_sha256": sha256_file(checkpoint / "adapter" / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_file(checkpoint / "adapter" / "adapter_config.json"),
            "optimizer_sha256": sha256_file(checkpoint / "optimizer.pt"),
            "scaler_sha256": sha256_file(checkpoint / "scaler.pt"),
            "rng_sha256": sha256_file(checkpoint / "rng_state.pth"),
            "config_sha256": sha256_file(output / "frozen-config.json"),
            "schedule_sha256": schedule_sha256,
            "base_adapter": "NONE",
            "training_config": train_cfg["training"],
            "final_checkpoint_rule": "step27_only_for_evaluation",
        },
    )
    return checkpoint


def _train(args: argparse.Namespace) -> int:
    output = args.output
    train_cfg = tomllib.loads(args.training_config.read_text())
    cfg = ModelSettings.load(args.model_config)
    _validate_config(cfg, train_cfg)
    tokenized_path = output / "training" / "tokenized-dataset.jsonl"
    schedule_path = output / "training" / "schedule.json"
    if not tokenized_path.is_file() or not schedule_path.is_file():
        raise Qwen3SFTError("prepare_stage_missing")
    if not os.environ.get("SLURM_JOB_ID"):
        raise Qwen3SFTError("qwen3_sft_requires_slurm_job")
    import torch
    import transformers

    capability, cache_manifest, cache_data = _validate_cuda(torch, args.cache)
    tokenizer = _tokenizer(cfg, args.cache)
    tokenized = load_tokenized_dataset(tokenized_path)
    set_seed(train_cfg["training"]["seed"], torch)
    model = _load_model(torch, cfg, args.cache)
    model.config.use_cache = False
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise Qwen3SFTError("gradient_checkpointing_not_supported")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        raise Qwen3SFTError("gradient_checkpointing_not_enabled")
    if not hasattr(model, "enable_input_require_grads"):
        raise Qwen3SFTError("input_requires_grad_not_supported")
    model.enable_input_require_grads()
    module_names = inject_lora_modules(
        model,
        rank=8,
        alpha=16,
        dropout=0,
        target_modules=TARGET_MODULES,
    )
    target_counts = Counter(name.rsplit(".", 1)[-1] for name in module_names)
    if len(module_names) != 252 or any(target_counts[target] != 36 for target in TARGET_MODULES):
        raise Qwen3SFTError(f"qwen3_lora_module_count_mismatch:{len(module_names)}:{target_counts}")
    if set(module_names) != {name for name, _ in iter_lora_modules(model)}:
        raise Qwen3SFTError("lora_injected_module_names_mismatch")

    resume_state = None
    if args.resume is not None:
        checkpoint = args.resume
        manifest = json.loads((checkpoint / "checkpoint-manifest.json").read_text())
        if manifest.get("status") != "valid" or manifest.get("model_revision") != MODEL_REVISION:
            raise Qwen3SFTError("invalid_resume_checkpoint")
        load_lora_adapter(
            model,
            checkpoint / "adapter",
            expected_model_id=MODEL_ID,
            expected_revision=MODEL_REVISION,
        )
        resume_state = _load_resume_state(torch, checkpoint)

    schedule_sha256 = sha256_file(schedule_path)
    training_dir = output / "training"

    def on_update(state: dict[str, Any], model=model) -> None:
        if state["checkpoint_requested"]:
            _save_checkpoint(
                output=output,
                state=state,
                model=model,
                tokenizer=tokenizer,
                torch_module=torch,
                cfg=cfg,
                train_cfg=train_cfg,
                module_names=module_names,
                schedule_sha256=schedule_sha256,
            )

    environment = _environment(
        torch_module=torch,
        transformers_module=transformers,
        cache_manifest=cache_manifest,
        cache_data=cache_data,
    )
    environment.update(
        {
            "stage": "train",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "module_count": len(module_names),
            "target_counts": dict(target_counts),
            "dataset_sha256": DATASET_SHA256,
            "schedule_sha256": schedule_sha256,
            "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
            "base_adapter": "NONE",
            "capability": capability,
        }
    )
    write_json(training_dir / "environment.json", environment)
    try:
        result = train_lora_schedule(
            model,
            tokenized,
            torch_module=torch,
            device="cuda:0",
            seed=train_cfg["training"]["seed"],
            microbatch_size=train_cfg["training"]["microbatch"],
            accumulation_steps=train_cfg["training"]["gradient_accumulation_steps"],
            epochs=train_cfg["training"]["epochs"],
            learning_rate=train_cfg["training"]["learning_rate"],
            betas=tuple(train_cfg["training"]["betas"]),
            eps=train_cfg["training"]["eps"],
            weight_decay=train_cfg["training"]["weight_decay"],
            foreach=train_cfg["training"]["foreach"],
            fused=train_cfg["training"]["fused"],
            clip_norm=train_cfg["training"]["clip_norm"],
            amp_initial_scale=train_cfg["training"]["amp_initial_scale"],
            max_backward=train_cfg["training"]["max_backward"],
            max_replays_per_group=train_cfg["training"]["max_amp_skip_replays"],
            checkpoint_steps=tuple(train_cfg["training"]["checkpoint_steps"]),
            resume_state=resume_state,
            on_update=on_update,
        )
    except Exception as exc:
        write_json(
            training_dir / "failure.json",
            {
                "status": "failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "dataset_sha256": DATASET_SHA256,
                "schedule_sha256": schedule_sha256,
                "environment": environment,
            },
        )
        raise

    final_checkpoint = training_dir / "checkpoints" / "step-27"
    if not (final_checkpoint / "checkpoint-manifest.json").is_file():
        raise Qwen3SFTError("final_step27_checkpoint_missing")
    final_adapter = training_dir / "final-adapter"
    if final_adapter.exists():
        existing_hash = sha256_file(final_adapter / "adapter_model.safetensors")
        final_hash = sha256_file(final_checkpoint / "adapter" / "adapter_model.safetensors")
        if existing_hash != final_hash:
            raise Qwen3SFTError("final_adapter_conflicts_with_step27")
    else:
        shutil.copytree(final_checkpoint / "adapter", final_adapter)
    final_adapter_sha = sha256_file(final_adapter / "adapter_model.safetensors")
    metrics = {
        "schema_version": "1.0",
        "status": "SFT_TRAIN_COMPLETE",
        "experiment": "T-032/E-029",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "enable_thinking": False,
        "sample_count": 54,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "epochs": 2,
        "microbatches_scheduled": 108,
        "microbatches_processed": result["microbatches_processed"],
        "successful_optimizer_updates": result["successful_optimizer_updates"],
        "skipped_optimizer_updates": result["skipped_optimizer_updates"],
        "replayed_groups": result["replayed_groups"],
        "backward_count": result["backward_count"],
        "backward_limit": result["backward_limit"],
        "optimizer_only_lora": True,
        "optimizer_state_parameter_count": result["optimizer_state_parameter_count"],
        "trainable_parameter_names_count": len(result["trainable_parameter_names"]),
        "trainable_parameter_count": result["trainable_parameter_count"],
        "module_count": len(module_names),
        "module_count_expected": 252,
        "target_counts": dict(target_counts),
        "base_parameter_hash_before": result["base_parameter_hash_before"],
        "base_parameter_hash_after": result["base_parameter_hash_after"],
        "base_parameters_changed": result["base_parameters_changed"],
        "base_gradients_present": result["base_gradients_present"],
        "lora_changed_parameter_count": result["lora_changed_parameter_count"],
        "lora_changed_parameters": result["lora_changed_parameters"],
        "gradients_finite": result["gradients_finite"],
        "overall_lora_gradient_norm_positive": result["overall_lora_gradient_norm_positive"],
        "attention_context_scope": "forward_backward_and_checkpoint_recompute",
        "gradient_checkpointing_use_reentrant": True,
        "input_requires_grad_enabled": True,
        "checkpoint_steps": [9, 18, 27],
        "checkpoints": [
            str((training_dir / "checkpoints" / f"step-{step}").resolve()) for step in (9, 18, 27)
        ],
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_adapter": str(final_adapter.resolve()),
        "final_adapter_sha256": final_adapter_sha,
        "dataset_sha256": DATASET_SHA256,
        "schedule_sha256": schedule_sha256,
        "final_checkpoint_rule": "step27_only_for_evaluation",
        "microbatch_metrics": result["microbatch_metrics"],
        "update_metrics": result["update_metrics"],
        "replay_metrics": result["replay_metrics"],
        "environment": environment,
    }
    write_json(training_dir / "training-metrics.json", metrics)
    write_json(
        output / "checks.json",
        {
            "status": "SFT_TRAIN_RELOAD_PENDING",
            "base_panel_sealed": (output / "evaluation" / "base" / "metrics.json").is_file(),
            "sft_train_complete": metrics["status"] == "SFT_TRAIN_COMPLETE",
            "sample_count": metrics["sample_count"],
            "microbatches_processed": metrics["microbatches_processed"],
            "successful_optimizer_updates": metrics["successful_optimizer_updates"],
            "final_adapter_sha256": final_adapter_sha,
        },
    )
    del model, result
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _reload(args: argparse.Namespace) -> int:
    output = args.output
    train_cfg = tomllib.loads(args.training_config.read_text())
    cfg = ModelSettings.load(args.model_config)
    _validate_config(cfg, train_cfg)
    adapter = output / "training" / "final-adapter"
    tokenized_path = output / "training" / "tokenized-dataset.jsonl"
    if not (adapter / "adapter_model.safetensors").is_file() or not tokenized_path.is_file():
        raise Qwen3SFTError("final_adapter_or_dataset_missing")
    if not os.environ.get("SLURM_JOB_ID"):
        raise Qwen3SFTError("qwen3_reload_requires_slurm_job")
    import torch
    import transformers

    capability, cache_manifest, cache_data = _validate_cuda(torch, args.cache)
    model = _load_model(torch, cfg, args.cache).eval()
    adapter_metadata = merge_lora_adapter(
        model,
        adapter,
        expected_model_id=MODEL_ID,
        expected_revision=MODEL_REVISION,
    )
    tokenized = load_tokenized_dataset(tokenized_path)
    input_ids = torch.tensor([tokenized[0]["input_ids"]], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    with attention_backend_context(torch, "sdpa_math"):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
    finite = bool(torch.isfinite(outputs.logits).all().item())
    if not finite or adapter_metadata["module_count"] != 252:
        raise Qwen3SFTError("fresh_reload_forward_failed")
    reload_check = {
        "schema_version": "1.0",
        "status": "SFT_TRAIN_RELOAD_PASS",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model_load_count_in_reload_process": 1,
        "adapter_merge_count_in_reload_process": 1,
        "adapter": adapter_metadata,
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "input_example_id": tokenized[0]["example_id"],
        "input_tokens": int(input_ids.shape[-1]),
        "logits_shape": list(outputs.logits.shape),
        "logits_dtype": str(outputs.logits.dtype),
        "finite_logits": finite,
        "device": "cuda:0",
        "compute_capability": capability,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "environment": _environment(
            torch_module=torch,
            transformers_module=transformers,
            cache_manifest=cache_manifest,
            cache_data=cache_data,
        ),
    }
    write_json(output / "training" / "reload-check.json", reload_check)
    checks = (
        json.loads((output / "checks.json").read_text())
        if (output / "checks.json").is_file()
        else {}
    )
    checks.update(
        {
            "status": "SFT_TRAIN_RELOAD_PASS",
            "sft_train_complete": True,
            "reload_finite": finite,
            "final_adapter_sha256": reload_check["adapter_sha256"],
        }
    )
    write_json(output / "checks.json", checks)
    return 0


def _tiny(args: argparse.Namespace) -> int:
    """CPU gate using a random, tiny Qwen3 implementation and the real schedule."""

    import torch
    import transformers

    if not hasattr(transformers, "Qwen3Config"):
        raise Qwen3SFTError("transformers_qwen3_not_available_for_tiny_gate")
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
    )
    model = transformers.Qwen3ForCausalLM(config)
    set_seed(42, torch)
    inject_lora_modules(
        model,
        rank=8,
        alpha=16,
        dropout=0,
        target_modules=TARGET_MODULES,
    )
    examples = []
    generator = torch.Generator().manual_seed(42)
    for index in range(54):
        ids = torch.randint(0, 128, (16,), generator=generator).tolist()
        examples.append(
            {
                "example_id": f"tiny-{index:02}",
                "input_ids": ids,
                "labels": [-100] + ids[1:],
                "global_microbatch_index": index,
            }
        )
    result = train_lora_schedule(
        model,
        examples,
        torch_module=torch,
        device="cpu",
        seed=42,
        microbatch_size=1,
        accumulation_steps=4,
        epochs=2,
        learning_rate=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
        clip_norm=1.0,
        amp_initial_scale=1024.0,
        max_backward=216,
        max_replays_per_group=3,
    )
    metrics = {
        "schema_version": "1.0",
        "status": "TINY_SFT_GATE_PASS",
        "random_qwen3": True,
        "sample_count": 54,
        "microbatches_scheduled": 108,
        "microbatches_processed": result["microbatches_processed"],
        "successful_optimizer_updates": result["successful_optimizer_updates"],
        "backward_count": result["backward_count"],
        "base_gradients_present": result["base_gradients_present"],
        "base_parameters_changed": result["base_parameters_changed"],
        "lora_changed_parameter_count": result["lora_changed_parameter_count"],
        "overall_lora_gradient_norm_positive": result["overall_lora_gradient_norm_positive"],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "fake_or_tiny": True,
        },
    }
    write_json(args.output / "training" / "tiny-metrics.json", metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train", "reload", "tiny"), required=True)
    parser.add_argument("--output", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG_DEFAULT)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG_DEFAULT)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        return _prepare(args)
    if args.stage == "train":
        return _train(args)
    if args.stage == "reload":
        return _reload(args)
    return _tiny(args)


if __name__ == "__main__":
    raise SystemExit(main())
