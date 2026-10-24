#!/usr/bin/env python3
"""Run the fixed eight-example, two-update Qwen3 LoRA feasibility pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from enterprise_agent_rl.inference_stability import attention_backend_context
from enterprise_agent_rl.lora import (
    inject_lora_modules,
    iter_lora_modules,
    lora_parameter_names,
    merge_lora_adapter,
    save_adapter,
    trainable_lora_parameters,
)
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.sft_training import (
    load_decision_rows,
    select_decision_examples,
    sha256_file,
)

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DATASET_SHA256 = "a5bbf4f2e0df521a6ad56885f8390b070ef3dee8cd25f0fa858e566b847e64e1"
EXPECTED_LAYERS = 36
EXPECTED_MODULE_COUNT = EXPECTED_LAYERS * len(TARGET_MODULES)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _tensor_hash(parameter: Any) -> str:
    digest = hashlib.sha256()
    tensor = parameter.detach().cpu().contiguous()
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_hash(model: Any, *, include_lora: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        is_lora = name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}
        if is_lora != include_lora:
            continue
        digest.update(name.encode())
        digest.update(_tensor_hash(parameter).encode())
    return digest.hexdigest()


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


def _load_model(torch: Any, transformers: Any, cfg: ModelSettings, cache: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        output_attentions=False,
        use_safetensors=True,
    ).to("cuda:0")
    if next(model.parameters()).dtype != torch.float16:
        raise RuntimeError("model_not_fp16")
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("model_attention_backend_mismatch")
    if getattr(model.config, "output_attentions", False):
        raise RuntimeError("output_attentions_must_be_false_for_sdpa")
    if getattr(model.config, "num_hidden_layers", None) != EXPECTED_LAYERS:
        raise RuntimeError(
            "qwen3_layer_count_mismatch:"
            f"{getattr(model.config, 'num_hidden_layers', None)}"
        )
    return model


def _pad_batch(example: dict[str, Any], pad_token_id: int, torch: Any) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda:0"),
        "attention_mask": torch.ones(
            (1, len(example["input_ids"])), dtype=torch.long, device="cuda:0"
        ),
        "labels": torch.tensor([example["labels"]], dtype=torch.long, device="cuda:0"),
    }


def _scaler(torch: Any, initial_scale: float):
    try:
        return torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=initial_scale, growth_interval=2000
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=True, init_scale=initial_scale, growth_interval=2000
        )


def _validate_fixed_config(cfg: ModelSettings, train_cfg: dict[str, Any]) -> None:
    if cfg.model_id != MODEL_ID or cfg.revision != MODEL_REVISION:
        raise RuntimeError("qwen3_model_identity_mismatch")
    if cfg.dtype != "float16" or cfg.attention_backend != "sdpa_math":
        raise RuntimeError("qwen3_runtime_contract_mismatch")
    if cfg.generation.enable_thinking is not False:
        raise RuntimeError("qwen3_training_requires_hard_thinking_off")
    lora = train_cfg.get("lora", {})
    training = train_cfg.get("training", {})
    expected_lora = {
        "r": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_modules": list(TARGET_MODULES),
    }
    if lora != expected_lora:
        raise RuntimeError(f"lora_config_mismatch:{lora}")
    expected_training = {
        "seed": 42,
        "microbatch": 1,
        "gradient_accumulation_steps": 4,
        "successful_optimizer_updates": 2,
        "learning_rate": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 0.00000001,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "clip_norm": 1.0,
        "amp_initial_scale": 1024.0,
        "checkpoint_use_reentrant": True,
    }
    if training != expected_training:
        raise RuntimeError(f"training_config_mismatch:{training}")
    if train_cfg.get("max_samples") != 8 or train_cfg.get("max_context_tokens") != 1024:
        raise RuntimeError("training_sample_contract_mismatch")


def _write_dataset_artifacts(
    output: Path,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    formatted: list[dict[str, Any]],
) -> None:
    dataset_manifest = {
        "schema_version": "1.0",
        "source_dataset": (
            "artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/"
            "tokenized-dataset.jsonl"
        ),
        "source_dataset_sha256": DATASET_SHA256,
        "source_row_count": len(rows),
        "selection": {
            "sort": ["task_id", "step_index", "example_id"],
            "max_context_tokens": 1024,
            "max_samples": 8,
            "enable_thinking": False,
            "old_input_ids_used": False,
            "old_labels_used": False,
            "old_token_spans_used": False,
            "selection_depended_on_base_success": False,
        },
        "selected": [
            {
                "example_id": row["example_id"],
                "task_id": row["task_id"],
                "step_index": row["step_index"],
                "source_row_sha256": row["source_row_sha256"],
                "target_action_sha256": hashlib.sha256(
                    json.dumps(row["target_action"], sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
            for row in selected
        ],
    }
    mask_audit = {
        "schema_version": "1.0",
        "tokenizer_template": "tokenizer.apply_chat_template",
        "enable_thinking": False,
        "assistant_only": True,
        "examples": [
            {
                "example_id": item["example_id"],
                "task_id": item["task_id"],
                "full_sequence_tokens": item["full_sequence_tokens"],
                "assistant_spans": item["assistant_spans"],
                "supervised_assistant_tokens": item["supervised_assistant_tokens"],
                "masked_context_tokens": item["masked_context_tokens"],
                "last_label_is_template_termination": item["labels"][-1] != -100,
            }
            for item in formatted
        ],
        "total_supervised_tokens": sum(item["supervised_assistant_tokens"] for item in formatted),
        "prefix_offset_alignment_checked": True,
    }
    _write_json(output / "dataset-manifest.json", dataset_manifest)
    _write_json(output / "mask-audit.json", mask_audit)


def _reload_saved_adapter(
    *,
    output: Path,
    cache: Path,
    model_config_path: Path,
    training_config_path: Path,
    dataset: Path,
    prior_reload_failure: str | None,
) -> int:
    """Reload only a persisted adapter after a recoverable post-training failure."""

    train_cfg = tomllib.loads(training_config_path.read_text())
    cfg = ModelSettings.load(model_config_path)
    _validate_fixed_config(cfg, train_cfg)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("supervision_dataset_sha256_mismatch")

    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_one_visible_cuda_device")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("wheel_missing_volta_support_or_unsupported_device")
    cache_manifest = cache / "qwen-cache-manifest.json"
    if not cache_manifest.is_file():
        raise RuntimeError("cache_integrity_manifest_missing")
    cache_data = json.loads(cache_manifest.read_text())
    if cache_data.get("model") != MODEL_ID or cache_data.get("revision") != MODEL_REVISION:
        raise RuntimeError("cache_integrity_manifest_mismatch")
    adapter = output / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("saved_adapter_missing")

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
    rows = load_decision_rows(dataset, expected_sha256=DATASET_SHA256)
    _, formatted = select_decision_examples(
        tokenizer,
        rows,
        max_samples=8,
        max_context_tokens=1024,
        enable_thinking=False,
    )

    reload_model = _load_model(torch, transformers, cfg, cache).eval()
    merge_summary = merge_lora_adapter(
        reload_model,
        adapter,
        expected_model_id=MODEL_ID,
        expected_revision=MODEL_REVISION,
    )
    input_ids = torch.tensor([formatted[0]["input_ids"]], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    with attention_backend_context(torch, "sdpa_math"):
        with torch.inference_mode():
            reloaded = reload_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
    finite_logits = bool(torch.isfinite(reloaded.logits).all().item())
    if not finite_logits or merge_summary["module_count"] != EXPECTED_MODULE_COUNT:
        raise RuntimeError("adapter_reload_forward_failed")
    reload_check = {
        "status": "LORA_UPDATE_RELOAD_PASS",
        "recovery_mode": "reload_only_after_prior_reload_oom"
        if prior_reload_failure
        else "reload_only",
        "prior_reload_failure": prior_reload_failure,
        "base_model_load_count_in_reload_process": 1,
        "base_model_load_count_observed_total": 3 if prior_reload_failure else 2,
        "adapter_merge_count_in_reload_process": 1,
        "successful_adapter_merge_count_total": 1,
        "injection_count_in_training": 1,
        "merge": merge_summary,
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "input_tokens": int(input_ids.shape[-1]),
        "logits_shape": list(reloaded.logits.shape),
        "logits_dtype": str(reloaded.logits.dtype),
        "finite_logits": finite_logits,
        "output_exact_fp16_equal_required": False,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cuda:0",
        "compute_capability": list(capability),
    }
    _write_json(output / "reload-check.json", reload_check)
    run_root = output.parent
    checks_path = run_root / "checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.is_file() else {}
    checks.update(
        {
            "lora_update_reload_pass": True,
            "successful_optimizer_updates": 2,
            "module_count": EXPECTED_MODULE_COUNT,
            "adapter_reloaded_finite": finite_logits,
            "reload_recovery_mode": reload_check["recovery_mode"],
        }
    )
    _write_json(checks_path, checks)
    del reloaded, reload_model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache-qwen3-8b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/qwen3-8b.toml"))
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/training/qwen3-lora-smoke.toml")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl"
        ),
    )
    parser.add_argument("--reload-only", action="store_true")
    parser.add_argument("--prior-reload-failure")
    args = parser.parse_args()
    output = args.output
    if args.reload_only:
        return _reload_saved_adapter(
            output=output,
            cache=args.cache,
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            dataset=args.dataset,
            prior_reload_failure=args.prior_reload_failure,
        )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_nonempty_lora_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    run_root = output.parent
    (run_root / "lora").mkdir(parents=True, exist_ok=True)

    train_cfg = tomllib.loads(args.training_config.read_text())
    cfg = ModelSettings.load(args.model_config)
    _validate_fixed_config(cfg, train_cfg)
    if sha256_file(args.dataset) != DATASET_SHA256:
        raise RuntimeError("supervision_dataset_sha256_mismatch")

    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_one_visible_cuda_device")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("wheel_missing_volta_support_or_unsupported_device")
    cache_manifest = args.cache / "qwen-cache-manifest.json"
    if not cache_manifest.is_file():
        raise RuntimeError("cache_integrity_manifest_missing")
    cache_data = json.loads(cache_manifest.read_text())
    if cache_data.get("model") != MODEL_ID or cache_data.get("revision") != MODEL_REVISION:
        raise RuntimeError("cache_integrity_manifest_mismatch")

    _set_seed(42, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        cache_dir=str(args.cache),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_decision_rows(args.dataset, expected_sha256=DATASET_SHA256)
    selected, formatted = select_decision_examples(
        tokenizer,
        rows,
        max_samples=8,
        max_context_tokens=1024,
        enable_thinking=False,
    )
    _write_dataset_artifacts(output, rows, selected, formatted)

    environment = {
        "schema_version": "1.0",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": {
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
            "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": cfg.revision,
        "dtype": "float16",
        "attention_backend": "sdpa_math",
        "enable_thinking": False,
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "cuda_arch_list": torch.cuda.get_arch_list(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cwd": str(Path.cwd()),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "cache_size_bytes_unique": cache_data.get("cache_size_bytes_unique"),
        "nvidia_smi": _nvidia_smi(),
        "checkpoint_use_reentrant": True,
        "input_requires_grad": True,
        "attention_context_scope": "forward_backward_and_checkpoint_recompute",
    }
    _write_json(output / "environment.json", environment)
    _write_json(
        run_root / "environment.json",
        {"lora": environment, "stage": "lora", "created_at_utc": _now()},
    )
    _write_json(
        run_root / "frozen-config.json",
        {
            "model": cfg.model_dump(),
            "training": train_cfg,
            "dataset_sha256": DATASET_SHA256,
            "base_adapter": "NONE",
            "base_panel_results_sealed_before_lora": True,
        },
    )
    _write_json(
        run_root / "preflight.json",
        {
            "status": "passed",
            "stage": "lora",
            "model": cfg.model_dump(),
            "environment": environment,
            "selected_samples": len(selected),
            "dataset_sha256": DATASET_SHA256,
            "cache_manifest_sha256": sha256_file(cache_manifest),
        },
    )

    model = _load_model(torch, transformers, cfg, args.cache)
    model.config.use_cache = False
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError("gradient_checkpointing_not_supported")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("gradient_checkpointing_not_enabled")
    if not hasattr(model, "enable_input_require_grads"):
        raise RuntimeError("input_requires_grad_not_supported")
    model.enable_input_require_grads()

    module_names = inject_lora_modules(
        model,
        rank=8,
        alpha=16,
        dropout=0,
        target_modules=TARGET_MODULES,
    )
    target_counts = Counter(name.rsplit(".", 1)[-1] for name in module_names)
    if len(module_names) != EXPECTED_MODULE_COUNT or any(
        target_counts[target] != EXPECTED_LAYERS for target in TARGET_MODULES
    ):
        raise RuntimeError(f"qwen3_lora_module_count_mismatch:{len(module_names)}:{target_counts}")
    injected_module_names = [name for name, _ in iter_lora_modules(model)]
    if set(injected_module_names) != set(module_names):
        raise RuntimeError("lora_injected_module_names_mismatch")
    trainable = trainable_lora_parameters(model)
    trainable_names = lora_parameter_names(model)
    expected_names = {
        f"{name}.{suffix}" for name in module_names for suffix in ("lora_A", "lora_B")
    }
    if set(trainable_names) != expected_names:
        raise RuntimeError("lora_trainable_parameter_names_mismatch")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in expected_names
    ):
        raise RuntimeError("base_model_trainable_parameter_count_nonzero")

    base_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name not in expected_names
    ]
    lora_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in expected_names
    }
    base_hash_before = _parameter_hash(model, include_lora=False)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer_parameters_do_not_match_lora_parameters")
    scaler = _scaler(torch, 1024.0)
    optimizer.zero_grad(set_to_none=True)

    groups = [formatted[:4], formatted[4:]]
    group_index = 0
    backward_count = 0
    update_count = 0
    skipped_updates = 0
    replayed_groups = 0
    microbatch_metrics = []
    update_metrics = []
    started = time.monotonic()
    while update_count < 2:
        if backward_count >= 24:
            raise RuntimeError("amp_replay_backward_budget_exhausted")
        group = groups[group_index] if group_index < len(groups) else groups[0]
        if group_index >= len(groups):
            replayed_groups += 1
        group_index += 1
        for example in group:
            model.train()
            batch = _pad_batch(example, tokenizer.pad_token_id, torch)
            with attention_backend_context(torch, "sdpa_math"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                        output_attentions=False,
                    )
                    logits = outputs.logits
                    loss = outputs.loss
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError("non_finite_logits")
                if not bool(torch.isfinite(loss).all().item()):
                    raise RuntimeError("non_finite_loss")
                scaler.scale(loss / 4).backward()
            backward_count += 1
            microbatch_metrics.append(
                {
                    "backward_index": backward_count,
                    "example_id": example["example_id"],
                    "loss": float(loss.detach().float().item()),
                    "finite_loss": True,
                }
            )
            del outputs, logits, loss, batch
            torch.cuda.empty_cache()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        grad_norm_value = float(grad_norm.detach().float().item())
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable
        )
        if not math.isfinite(grad_norm_value) or not finite_gradients:
            raise RuntimeError("non_finite_lora_gradient")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        skipped = scale_after < scale_before
        if skipped:
            skipped_updates += 1
            optimizer.zero_grad(set_to_none=True)
            update_metrics.append(
                {
                    "successful": False,
                    "skipped_by_amp": True,
                    "grad_norm_unscaled": grad_norm_value,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                }
            )
            continue
        update_count += 1
        update_metrics.append(
            {
                "successful": True,
                "update_index": update_count,
                "skipped_by_amp": False,
                "grad_norm_unscaled": grad_norm_value,
                "gradients_finite": finite_gradients,
                "scale_before": scale_before,
                "scale_after": scale_after,
            }
        )
        optimizer.zero_grad(set_to_none=True)

    base_hash_after = _parameter_hash(model, include_lora=False)
    lora_changed = []
    for name, parameter in model.named_parameters():
        if name not in lora_before:
            continue
        delta = (parameter.detach().cpu().float() - lora_before[name].float()).abs()
        if bool((delta > 0).any().item()):
            lora_changed.append(
                {
                    "name": name,
                    "max_abs_delta": float(delta.max().item()),
                    "l2_delta": float(delta.norm().item()),
                }
            )
    base_grads = [name for name, parameter in base_parameters if parameter.grad is not None]
    optimizer_state_parameter_count = len(optimizer.state)
    if update_count != 2 or backward_count > 24:
        raise RuntimeError("optimizer_update_contract_failed")
    if base_hash_before != base_hash_after or base_grads:
        raise RuntimeError("base_model_changed_or_received_gradient")
    if not lora_changed:
        raise RuntimeError("lora_parameters_did_not_change")
    if not all(item["successful"] for item in update_metrics if item.get("successful")):
        raise RuntimeError("successful_update_metric_invalid")

    checkpoint = output / "checkpoints" / "pilot-final"
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
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scaler.state_dict(), checkpoint / "scaler.pt")
    torch.save(_capture_rng(torch), checkpoint / "rng_state.pth")
    final_adapter = output / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(checkpoint / "adapter", final_adapter)
    optimizer_only_lora = optimizer_state_parameter_count > 0 and optimizer_ids == {
        id(parameter) for parameter in trainable
    }
    training_metrics = {
        "schema_version": "1.0",
        "status": "completed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "enable_thinking": False,
        "sample_count": len(selected),
        "microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "microbatches_processed": backward_count,
        "successful_optimizer_updates": update_count,
        "skipped_optimizer_updates": skipped_updates,
        "replayed_groups": replayed_groups,
        "backward_count": backward_count,
        "backward_limit": 24,
        "optimizer_only_lora": optimizer_only_lora,
        "optimizer_state_parameter_count": optimizer_state_parameter_count,
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in trainable),
        "module_count": len(module_names),
        "module_count_expected": EXPECTED_MODULE_COUNT,
        "target_counts": dict(target_counts),
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_parameters_changed": base_hash_before != base_hash_after,
        "base_gradients_present": base_grads,
        "lora_changed_parameter_count": len(lora_changed),
        "lora_changed_parameters": lora_changed[:16],
        "gradients_finite": all(item.get("gradients_finite", True) for item in update_metrics),
        "overall_lora_gradient_norm_positive": all(
            item["grad_norm_unscaled"] > 0
            for item in update_metrics
            if item.get("successful")
        ),
        "attention_context_scope": "forward_backward_and_checkpoint_recompute",
        "gradient_checkpointing_use_reentrant": True,
        "input_requires_grad_enabled": True,
        "microbatch_metrics": microbatch_metrics,
        "update_metrics": update_metrics,
        "elapsed_s": time.monotonic() - started,
        "adapter_summary": adapter_summary,
        "checkpoint": str(checkpoint.resolve()),
        "adapter": str(final_adapter.resolve()),
        "optimizer_state": str((checkpoint / "optimizer.pt").resolve()),
        "scaler_state": str((checkpoint / "scaler.pt").resolve()),
        "rng_state": str((checkpoint / "rng_state.pth").resolve()),
    }
    _write_json(output / "training-metrics.json", training_metrics)
    _write_json(
        output / "lora-preflight.json",
        {
            "status": "passed",
            "matched_module_count": len(module_names),
            "expected_module_count": EXPECTED_MODULE_COUNT,
            "trainable_parameter_names_count": len(trainable_names),
            "optimizer_only_lora": optimizer_only_lora,
            "base_frozen": not base_grads,
            "initial_amp_scale": 1024.0,
        },
    )

    del (
        model,
        optimizer,
        scaler,
        trainable,
        base_parameters,
        lora_before,
        optimizer_ids,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    reload_model = _load_model(torch, transformers, cfg, args.cache).eval()
    merge_summary = merge_lora_adapter(
        reload_model,
        final_adapter,
        expected_model_id=MODEL_ID,
        expected_revision=MODEL_REVISION,
    )
    input_ids = torch.tensor([formatted[0]["input_ids"]], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    with attention_backend_context(torch, "sdpa_math"):
        with torch.inference_mode():
            reloaded = reload_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
    finite_logits = bool(torch.isfinite(reloaded.logits).all().item())
    if not finite_logits or merge_summary["module_count"] != EXPECTED_MODULE_COUNT:
        raise RuntimeError("adapter_reload_forward_failed")
    reload_check = {
        "status": "LORA_UPDATE_RELOAD_PASS",
        "base_model_load_count": 2,
        "adapter_merge_count": 1,
        "injection_count": 1,
        "merge": merge_summary,
        "adapter_sha256": sha256_file(final_adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(final_adapter / "adapter_config.json"),
        "input_tokens": int(input_ids.shape[-1]),
        "logits_shape": list(reloaded.logits.shape),
        "logits_dtype": str(reloaded.logits.dtype),
        "finite_logits": finite_logits,
        "output_exact_fp16_equal_required": False,
    }
    _write_json(output / "reload-check.json", reload_check)
    _write_json(
        run_root / "checks.json",
        {
            "base_panel_sealed": (output.parent / "base" / "metrics.json").is_file(),
            "lora_update_reload_pass": reload_check["status"] == "LORA_UPDATE_RELOAD_PASS",
            "successful_optimizer_updates": update_count,
            "module_count": len(module_names),
            "base_frozen": base_hash_before == base_hash_after and not base_grads,
            "adapter_reloaded_finite": finite_logits,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
