#!/usr/bin/env python3
"""Run the one fixed, single-GPU LoRA SFT pilot.

The script deliberately implements only the controlled pilot needed by this
repository.  It uses Transformers for the base model and a small manual LoRA
wrapper because PEFT/Accelerate are not in the locked environment.  No
validation/test task payload is read here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from enterprise_agent_rl.inference_stability import attention_backend_context
from enterprise_agent_rl.lora import (
    inject_lora_modules,
    iter_lora_modules,
    lora_parameter_names,
    save_adapter,
    trainable_lora_parameters,
)
from enterprise_agent_rl.sft_training import (
    audit_tokenized_dataset,
    format_sft_example,
    load_dataset_rows,
    sha256_file,
    yaml_dump,
)

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


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _nvidia_smi() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "query_status": query.returncode,
        "query": query.stdout.strip() or query.stderr.strip(),
    }


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


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


def _save_rng(path: Path, torch: Any) -> None:
    torch.save(_capture_rng(torch), path)


def _pad_batch(examples: list[dict[str, Any]], pad_token_id: int, torch: Any) -> dict[str, Any]:
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for example in examples:
        padding = max_length - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda:0"),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device="cuda:0"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cuda:0"),
    }


def _linear_schedule(optimizer: Any, warmup_steps: int, total_steps: int, torch: Any):
    def rate(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining = max(0, total_steps - step)
        denominator = max(1, total_steps - warmup_steps)
        return float(remaining) / float(denominator)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)


def _write_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    torch: Any,
    output: Path,
    *,
    epoch: int,
    global_step: int,
    module_names: list[str],
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = output / "checkpoints" / f"checkpoint-epoch-{epoch}"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    adapter = save_adapter(
        model,
        checkpoint / "adapter",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        rank=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        module_names=module_names,
    )
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(scaler.state_dict(), checkpoint / "scaler.pt")
    _save_rng(checkpoint / "rng_state.pth", torch)
    state = {
        "schema_version": "1.0",
        "epoch": epoch,
        "global_step": global_step,
        "adapter": adapter,
        "config": config,
        "latest_metrics": metrics[-1] if metrics else None,
        "created_at_utc": _now(),
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    )
    return state


def _copy_final_adapter(checkpoint: Path, output: Path) -> None:
    destination = output / "adapter"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(checkpoint / "adapter", destination)


def _environment(
    torch: Any,
    transformers: Any,
    *,
    args: argparse.Namespace,
    dataset_hash: str,
    max_seq_length: int,
) -> dict[str, Any]:
    capability = list(torch.cuda.get_device_capability(0))
    return {
        "schema_version": "1.0",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": {
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "qos": os.environ.get("SLURM_JOB_QOS") or os.environ.get("SLURM_QOS"),
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float16",
        "attention_backend": "sdpa_math",
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": capability,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": _package_version("peft"),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "dataset_sha256": dataset_hash,
        "max_seq_length": max_seq_length,
        "nvidia_smi": _nvidia_smi(),
        "model_weights_loaded_on_slurm_node": True,
        "created_at_utc": _now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/sft-data-train-teacher-v2/dataset-metadata.json"),
    )
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("sft_training_requires_slurm_job")
    if args.output.exists():
        raise RuntimeError(f"refusing_existing_output:{args.output}")
    args.output.mkdir(parents=True)
    (args.output / "checkpoints").mkdir()

    rows, metadata = load_dataset_rows(args.dataset, args.metadata)
    dataset_hash = sha256_file(args.dataset)
    training_config: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "controlled_sft_pilot",
        "model": {
            "name": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "float16",
        },
        "attention": {
            "implementation": "sdpa_math",
            "transformers_value": "sdpa",
            "flash_attention_2": False,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(TARGET_MODULES),
        },
        "training": {
            "seed": args.seed,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "gradient_checkpointing": True,
            "fp16": True,
            "bf16": False,
        },
        "data": {
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_hash,
            "metadata": str(args.metadata),
            "source_split": metadata["source_split"],
            "validation_used_for_training": metadata["validation_used_for_training"],
            "test_used_for_training": metadata["test_used_for_training"],
            "sample_count": len(rows),
            "teacher_source_distribution": metadata["trajectory_source_distribution"],
            "truncation": "refused",
        },
    }
    (args.output / "training-config.yaml").write_text(yaml_dump(training_config))
    shutil.copy2(args.metadata, args.output / "dataset-metadata.json")
    (args.output / "dataset-hash.txt").write_text(dataset_hash + "\n")
    (args.output / "git-state.txt").write_text(
        f"commit: {_git_value('rev-parse', 'HEAD')}\n"
        f"branch: {_git_value('branch', '--show-current')}\n"
        f"status:\n{_git_value('status', '--short')}\n"
    )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires_one_visible_cuda_device")
    if torch.cuda.get_device_capability(0) < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("wheel_missing_volta_support_or_unsupported_device")
    _set_seed(args.seed, torch)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(args.cache),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audit = audit_tokenized_dataset(tokenizer, rows, max_context_tokens=8192)
    max_seq_length = int(audit["full_sequence_tokens"]["max"])
    (args.output / "sft-preflight.json").write_text(
        json.dumps(
            {
                **audit,
                "dataset_sha256": dataset_hash,
                "source_split": metadata["source_split"],
                "validation_used_for_training": metadata["validation_used_for_training"],
                "test_used_for_training": metadata["test_used_for_training"],
                "teacher_source_distribution": metadata["trajectory_source_distribution"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_class": type(tokenizer).__name__,
                "decision": "READY_FOR_SFT_TRAINING",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    training_config["data"]["max_seq_length"] = max_seq_length
    (args.output / "training-config.yaml").write_text(yaml_dump(training_config))
    (args.output / "environment.json").write_text(
        json.dumps(
            _environment(
                torch,
                transformers,
                args=args,
                dataset_hash=dataset_hash,
                max_seq_length=max_seq_length,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    formatted = [format_sft_example(tokenizer, row, max_length=max_seq_length) for row in rows]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(args.cache),
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
    model.config.use_cache = False
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError("gradient_checkpointing_not_supported")
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    module_names = inject_lora_modules(
        model,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
    )
    target_counts = Counter(name.rsplit(".", 1)[-1] for name in module_names)
    missing_targets = sorted(set(TARGET_MODULES) - set(target_counts))
    if missing_targets:
        raise RuntimeError(f"lora_target_modules_missing:{missing_targets}")
    if not module_names:
        raise RuntimeError("lora_injected_module_count_zero")
    injected_module_names = [name for name, _ in iter_lora_modules(model)]
    if set(injected_module_names) != set(module_names):
        raise RuntimeError(
            "lora_injected_module_names_mismatch:"
            f"expected={len(module_names)}:actual={len(injected_module_names)}"
        )

    trainable = trainable_lora_parameters(model)
    trainable_names = lora_parameter_names(model)
    expected_lora_names = {
        f"{module_name}.{suffix}" for module_name in module_names for suffix in ("lora_A", "lora_B")
    }
    if set(trainable_names) != expected_lora_names:
        raise RuntimeError(
            "lora_trainable_parameter_names_mismatch:"
            f"expected={len(expected_lora_names)}:actual={len(trainable_names)}"
        )
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in expected_lora_names
    ]
    if unexpected_trainable:
        raise RuntimeError("unexpected_base_parameters_trainable:" + repr(unexpected_trainable[:8]))
    trainable_count = sum(int(parameter.numel()) for parameter in trainable)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_ratio = trainable_count / total_parameter_count if total_parameter_count else 0.0
    base_trainable_count = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if name not in expected_lora_names and parameter.requires_grad
    )
    if trainable_count <= 0:
        raise RuntimeError("lora_trainable_parameter_count_zero")
    if base_trainable_count != 0:
        raise RuntimeError(f"base_model_trainable_parameter_count:{base_trainable_count}")
    if total_parameter_count <= 0 or not 0.0 < trainable_ratio < 1.0:
        raise RuntimeError(f"lora_trainable_parameter_ratio_invalid:{trainable_ratio}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_parameter_count = sum(
        len(group.get("params", [])) for group in optimizer.param_groups
    )
    if not optimizer.param_groups or optimizer_parameter_count <= 0:
        raise RuntimeError("optimizer_parameter_groups_empty")
    if {
        id(parameter) for group in optimizer.param_groups for parameter in group.get("params", [])
    } != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer_parameters_do_not_match_lora_parameters")

    lora_preflight = {
        "schema_version": "1.0",
        "status": "passed",
        "target_modules": list(TARGET_MODULES),
        "matched_module_names": module_names,
        "matched_module_count": len(module_names),
        "matched_module_counts_by_target": {
            target: target_counts.get(target, 0) for target in TARGET_MODULES
        },
        "missing_target_modules": missing_targets,
        "injected_module_names": injected_module_names,
        "injected_module_count": len(injected_module_names),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "total_model_parameter_count": total_parameter_count,
        "trainable_parameter_ratio": trainable_ratio,
        "base_trainable_parameter_count": base_trainable_count,
        "all_trainable_parameters_are_lora": not unexpected_trainable,
        "unexpected_trainable_parameter_names": unexpected_trainable,
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_parameter_count": optimizer_parameter_count,
        "optimizer_parameter_groups_nonempty": True,
        "optimizer_parameters_match_lora_parameters": True,
        "created_at_utc": _now(),
    }
    (args.output / "lora-preflight.json").write_text(
        json.dumps(lora_preflight, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "lora_runtime_assertions": "passed",
                "matched_module_names": module_names,
                "matched_module_count": len(module_names),
                "trainable_parameter_count": trainable_count,
                "trainable_parameter_ratio": trainable_ratio,
                "optimizer_group_count": len(optimizer.param_groups),
                "optimizer_parameter_count": optimizer_parameter_count,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(formatted) / args.batch_size) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = max(1, math.ceil(total_steps * args.warmup_ratio))
    scheduler = _linear_schedule(optimizer, warmup_steps, total_steps, torch)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    training_metrics: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(formatted)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []
        optimizer_steps = 0
        for batch_start in range(0, len(order), args.batch_size):
            batch_indices = order[batch_start : batch_start + args.batch_size]
            batch = _pad_batch(
                [formatted[index] for index in batch_indices],
                tokenizer.pad_token_id,
                torch,
            )
            with attention_backend_context(torch, "sdpa_math"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                        output_attentions=False,
                    )
                    logits = outputs.logits
                    if not bool(torch.isfinite(logits).all().item()):
                        raise RuntimeError("non_finite_logits")
                    loss = outputs.loss
            if not bool(torch.isfinite(loss).all().item()):
                raise RuntimeError("non_finite_loss")
            loss_value = float(loss.detach().float().item())
            epoch_losses.append(loss_value)
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            should_step = (
                batch_start // args.batch_size + 1
            ) % args.gradient_accumulation_steps == 0 or batch_start + args.batch_size >= len(order)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                grad_norm_value = float(grad_norm.detach().float().item())
                if not math.isfinite(grad_norm_value):
                    raise RuntimeError("non_finite_gradient_norm")
                epoch_grad_norms.append(grad_norm_value)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                optimizer_steps += 1
            del outputs, logits, loss, batch
            torch.cuda.empty_cache()

        elapsed = time.monotonic() - started
        metric = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "mean_loss": sum(epoch_losses) / len(epoch_losses),
            "min_loss": min(epoch_losses),
            "max_loss": max(epoch_losses),
            "mean_grad_norm": sum(epoch_grad_norms) / len(epoch_grad_norms),
            "max_grad_norm_observed": max(epoch_grad_norms),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_s": elapsed,
            "finite_loss": True,
            "finite_gradients": True,
        }
        training_metrics.append(metric)
        (args.output / "training-metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "running",
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "trainable_parameter_count": trainable_count,
                    "module_count": len(module_names),
                    "metrics": training_metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        _write_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            torch,
            args.output,
            epoch=epoch,
            global_step=global_step,
            module_names=module_names,
            config=training_config,
            metrics=training_metrics,
        )

    final_checkpoint = args.output / "checkpoints" / f"checkpoint-epoch-{args.epochs}"
    _copy_final_adapter(final_checkpoint, args.output)
    final_metrics = {
        "schema_version": "1.0",
        "status": "completed",
        "started_at_utc": _now(),
        "global_step": global_step,
        "total_steps": total_steps,
        "epochs_completed": args.epochs,
        "trainable_parameter_count": trainable_count,
        "lora_module_count": len(module_names),
        "module_names": module_names,
        "metrics": training_metrics,
        "elapsed_s": time.monotonic() - started,
        "finite_loss": True,
        "finite_gradients": True,
        "non_finite_logits_detected": False,
        "oom": False,
        "cuda_errors": False,
        "checkpoint": str(final_checkpoint),
        "adapter": str(args.output / "adapter"),
        "resume_files": [
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ],
    }
    (args.output / "training-metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
