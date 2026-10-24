"""Reusable, auditable pieces for the fixed Qwen3 Runtime SFT recipe.

The module deliberately has no top-level torch/transformers import.  CPU
preparation and tests can therefore validate the data contract without
loading a model.  The GPU entrypoint owns model loading and calls the
gradient schedule below after the base model has been frozen and LoRA has
been injected.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from enterprise_agent_rl.inference_stability import attention_backend_context
from enterprise_agent_rl.lora import lora_parameter_names, trainable_lora_parameters
from enterprise_agent_rl.sft_training import (
    SFTDataError,
    decision_example,
    format_sft_example,
    sha256_file,
)

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_SHA256 = "a5bbf4f2e0df521a6ad56885f8390b070ef3dee8cd25f0fa858e566b847e64e1"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_ACTION_COUNTS = {
    "plan": 10,
    "dictionary": 7,
    "sql": 12,
    "check": 10,
    "finish": 10,
    "schema": 5,
}


class Qwen3SFTError(RuntimeError):
    """Raised when the fixed SFT contract cannot be proved."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON artifact, including parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def capture_rng(torch_module: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch_module.get_rng_state().clone(),
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch_module.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = []
    return state


def restore_rng(torch_module: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch_module.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def rng_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode())
    digest.update(state["torch_cpu"].detach().cpu().contiguous().numpy().tobytes())
    for item in state.get("torch_cuda", []):
        digest.update(item.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(parameter: Any) -> str:
    tensor = parameter.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def parameter_hash(model: Any, *, include_lora: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        is_lora = name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}
        if is_lora != include_lora:
            continue
        digest.update(name.encode())
        digest.update(tensor_hash(parameter).encode())
    return digest.hexdigest()


def _nested_state_digest(value: Any, digest: Any) -> None:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _nested_state_digest(value[key], digest)
    elif isinstance(value, list | tuple):
        for item in value:
            _nested_state_digest(item, digest)
    elif hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    else:
        digest.update(repr(value).encode())


def optimizer_state_hash(optimizer: Any) -> str:
    digest = hashlib.sha256()
    _nested_state_digest(optimizer.state_dict(), digest)
    return digest.hexdigest()


def build_full_tokenized_dataset(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_context_tokens: int,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Re-render every approved decision row and fail closed on any omission."""

    if len(rows) != 54:
        raise Qwen3SFTError(f"decision_row_count:{len(rows)}")
    ordered = sorted(
        rows,
        key=lambda row: (str(row["task_id"]), int(row["step_index"]), str(row["example_id"])),
    )
    if len({row.get("example_id") for row in ordered}) != len(ordered):
        raise Qwen3SFTError("duplicate_decision_example_id")
    if any(row.get("example_id") is None for row in ordered):
        raise Qwen3SFTError("decision_example_id_missing")

    examples: list[dict[str, Any]] = []
    formatted: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    for row in ordered:
        example = decision_example(row)
        try:
            item = format_sft_example(
                tokenizer,
                example,
                max_length=max_context_tokens,
                enable_thinking=enable_thinking,
            )
        except SFTDataError as exc:
            raise Qwen3SFTError(str(exc)) from exc
        spans = item.get("assistant_spans", [])
        if len(spans) != 1 or spans[0][1] != len(item["input_ids"]):
            raise Qwen3SFTError(f"target_span_not_final:{row['example_id']}")
        start, end = spans[0]
        labels = item["labels"]
        if any(label != -100 for label in labels[:start]):
            raise Qwen3SFTError(f"prefix_not_masked:{row['example_id']}")
        if labels[start:end] != item["input_ids"][start:end]:
            raise Qwen3SFTError(f"target_labels_mismatch:{row['example_id']}")
        if any(label != -100 for label in labels[end:]):
            raise Qwen3SFTError(f"suffix_not_masked:{row['example_id']}")
        examples.append(example)
        formatted.append(item)
        kind = str(row["target_action"]["kind"])
        action_counts[kind] = action_counts.get(kind, 0) + 1

    if action_counts != EXPECTED_ACTION_COUNTS:
        raise Qwen3SFTError(f"action_kind_counts:{action_counts}")
    return (
        examples,
        formatted,
        {
            "sort": ["task_id", "step_index", "example_id"],
            "sample_count": len(ordered),
            "max_context_tokens": max_context_tokens,
            "enable_thinking": enable_thinking,
            "old_input_ids_used": False,
            "old_labels_used": False,
            "old_token_spans_used": False,
            "source_rows_retained": len(ordered),
            "action_kind_counts": action_counts,
        },
    )


def write_tokenized_artifacts(
    output: Path,
    *,
    source_path: Path,
    source_sha256: str,
    examples: list[dict[str, Any]],
    formatted: list[dict[str, Any]],
    config_identity: dict[str, Any],
) -> dict[str, Any]:
    """Persist the complete re-tokenized rows and a machine-checkable mask audit."""

    if len(examples) != len(formatted):
        raise Qwen3SFTError("tokenized_example_count_mismatch")
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "tokenized-dataset.jsonl"
    lines = []
    audit_examples = []
    for example, item in zip(examples, formatted, strict=True):
        row = {
            "example_id": example["example_id"],
            "task_id": example["task_id"],
            "step_index": example["step_index"],
            "source_policy_kind": example.get("source_policy_kind"),
            "teacher_trajectory_sha256": example.get("teacher_trajectory_sha256"),
            "source_row_sha256": example.get("source_row_sha256"),
            "context_messages": example["messages"][:-1],
            "target_action": example["target_action"],
            "messages": example["messages"],
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "assistant_spans": item["assistant_spans"],
            "full_sequence_tokens": item["full_sequence_tokens"],
            "supervised_assistant_tokens": item["supervised_assistant_tokens"],
            "masked_context_tokens": item["masked_context_tokens"],
            "enable_thinking": False,
        }
        lines.append(canonical_json(row))
        start, end = item["assistant_spans"][0]
        audit_examples.append(
            {
                "example_id": example["example_id"],
                "task_id": example["task_id"],
                "step_index": example["step_index"],
                "full_sequence_tokens": len(item["input_ids"]),
                "assistant_spans": item["assistant_spans"],
                "target_span_is_final": end == len(item["input_ids"]),
                "prefix_masked": all(label == -100 for label in item["labels"][:start]),
                "target_unmasked": item["labels"][start:end] == item["input_ids"][start:end],
                "suffix_masked": all(label == -100 for label in item["labels"][end:]),
                "target_action_sha256": sha256_json(example["target_action"]),
                "termination_token_id": item["input_ids"][-1],
                "supervised_assistant_tokens": item["supervised_assistant_tokens"],
                "masked_context_tokens": item["masked_context_tokens"],
            }
        )
    temporary = dataset_path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(dataset_path)
    audit = {
        "schema_version": "1.0",
        "tokenizer_template": "tokenizer.apply_chat_template",
        "enable_thinking": False,
        "assistant_only": True,
        "target_policy": "last_assistant_action_plus_template_termination_only",
        "prefix_offset_alignment_checked": True,
        "sample_count": len(audit_examples),
        "total_supervised_tokens": sum(item["supervised_assistant_tokens"] for item in formatted),
        "examples": audit_examples,
    }
    write_json(output / "mask-audit.json", audit)
    dataset_manifest = {
        "schema_version": "1.0",
        "source_dataset": str(source_path),
        "source_dataset_sha256": source_sha256,
        "source_row_count": len(examples),
        "output_dataset": str(dataset_path),
        "output_dataset_sha256": sha256_file(dataset_path),
        "selection": config_identity,
        "mask_audit": str((output / "mask-audit.json").resolve()),
        "mask_audit_sha256": sha256_file(output / "mask-audit.json"),
    }
    write_json(output / "dataset-manifest.json", dataset_manifest)
    return dataset_manifest


def make_schedule(
    examples: list[dict[str, Any]], *, epochs: int, accumulation_steps: int
) -> list[list[dict[str, Any]]]:
    if epochs <= 0 or accumulation_steps <= 0:
        raise Qwen3SFTError("invalid_schedule_config")
    if not examples or len(examples) * epochs % accumulation_steps:
        raise Qwen3SFTError("schedule_not_divisible_by_accumulation")
    repeated = []
    for epoch in range(epochs):
        for position, example in enumerate(examples):
            item = dict(example)
            item["epoch"] = epoch
            item["epoch_position"] = position
            item["global_microbatch_index"] = len(repeated)
            repeated.append(item)
    return [
        repeated[index : index + accumulation_steps]
        for index in range(0, len(repeated), accumulation_steps)
    ]


def schedule_manifest(
    examples: list[dict[str, Any]], *, epochs: int, accumulation_steps: int
) -> dict[str, Any]:
    groups = make_schedule(examples, epochs=epochs, accumulation_steps=accumulation_steps)
    return {
        "schema_version": "1.0",
        "sample_count": len(examples),
        "epochs": epochs,
        "microbatches": len(examples) * epochs,
        "accumulation_steps": accumulation_steps,
        "successful_updates": len(groups),
        "cross_epoch_accumulation": any(
            group[0]["epoch"] != group[-1]["epoch"] for group in groups
        ),
        "groups": [
            {
                "update_index": index + 1,
                "global_microbatch_indices": [item["global_microbatch_index"] for item in group],
                "example_ids": [item["example_id"] for item in group],
                "epochs": [item["epoch"] for item in group],
            }
            for index, group in enumerate(groups)
        ],
    }


def _new_scaler(torch_module: Any, *, device: str, initial_scale: float, enabled: bool) -> Any:
    try:
        return torch_module.amp.GradScaler(
            device.split(":", 1)[0], enabled=enabled, init_scale=initial_scale
        )
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=enabled, init_scale=initial_scale)


def _autocast(torch_module: Any, *, device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch_module.autocast(device_type=device.split(":", 1)[0], dtype=torch_module.float16)


def _batch(torch_module: Any, item: dict[str, Any], device: str) -> dict[str, Any]:
    input_ids = torch_module.tensor([item["input_ids"]], dtype=torch_module.long, device=device)
    labels = torch_module.tensor([item["labels"]], dtype=torch_module.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch_module.ones_like(input_ids),
        "labels": labels,
    }


def train_lora_schedule(
    model: Any,
    examples: list[dict[str, Any]],
    *,
    torch_module: Any,
    device: str,
    seed: int,
    microbatch_size: int,
    accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    foreach: bool,
    fused: bool,
    clip_norm: float,
    amp_initial_scale: float,
    max_backward: int,
    max_replays_per_group: int,
    checkpoint_steps: tuple[int, ...] = (),
    resume_state: dict[str, Any] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    gradient_injector: Callable[[Any, int, int], None] | None = None,
    amp_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run the fixed 54x2 schedule with native AMP skip/replay semantics.

    ``gradient_injector`` is intentionally test-only plumbing.  A real run
    leaves it unset; CPU tests use it to force one native GradScaler skip.
    """

    if microbatch_size != 1:
        raise Qwen3SFTError("only_microbatch_one_is_supported")
    if len(examples) != 54:
        raise Qwen3SFTError(f"training_examples_must_be_54:{len(examples)}")
    groups = make_schedule(examples, epochs=epochs, accumulation_steps=accumulation_steps)
    if any(len(group) != accumulation_steps for group in groups):
        raise Qwen3SFTError("incomplete_accumulation_group")
    set_seed(seed, torch_module)
    trainable = trainable_lora_parameters(model)
    trainable_names = lora_parameter_names(model)
    expected_names = {
        f"{name}.{suffix}"
        for name in (module_name for module_name, _ in _iter_lora_modules(model))
        for suffix in ("lora_A", "lora_B")
    }
    if set(trainable_names) != expected_names:
        raise Qwen3SFTError("lora_trainable_parameter_names_mismatch")
    base_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name not in expected_names
    ]
    if any(parameter.requires_grad for _, parameter in base_parameters):
        raise Qwen3SFTError("base_model_trainable_parameter_count_nonzero")
    base_hash_before = parameter_hash(model, include_lora=False)
    lora_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in expected_names
    }
    optimizer = torch_module.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=foreach,
        fused=fused,
    )
    amp_active = device.startswith("cuda") if amp_enabled is None else amp_enabled
    scaler = _new_scaler(
        torch_module,
        device=device,
        initial_scale=amp_initial_scale,
        enabled=amp_active,
    )
    cursor = 0
    update_count = 0
    backward_count = 0
    skipped_updates = 0
    replayed_groups = 0
    microbatch_metrics: list[dict[str, Any]] = []
    update_metrics: list[dict[str, Any]] = []
    replay_metrics: list[dict[str, Any]] = []

    if resume_state is not None:
        cursor = int(resume_state["cursor"])
        update_count = int(resume_state["successful_updates"])
        backward_count = int(resume_state["backward_count"])
        replayed_groups = int(resume_state.get("replayed_groups", 0))
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scaler.load_state_dict(resume_state["scaler_state"])
        restore_rng(torch_module, resume_state["rng_state"])
        if cursor % accumulation_steps or update_count != cursor // accumulation_steps:
            raise Qwen3SFTError("resume_cursor_not_at_update_boundary")
        if cursor > len(examples) * epochs:
            raise Qwen3SFTError("resume_cursor_out_of_range")

    while cursor < len(examples) * epochs:
        group_index = cursor // accumulation_steps
        group = groups[group_index]
        retry = 0
        while True:
            if backward_count + len(group) > max_backward:
                raise Qwen3SFTError("amp_replay_backward_budget_exhausted")
            group_rng = capture_rng(torch_module)
            optimizer.zero_grad(set_to_none=True)
            for item in group:
                model.train()
                batch = _batch(torch_module, item, device)
                # Keep the explicit SDPA math backend active through forward,
                # backward, and reentrant checkpoint recomputation.  The
                # context must surround backward as well as model() because
                # reentrant recomputation happens during backward.
                with attention_backend_context(torch_module, "sdpa_math"):
                    with _autocast(
                        torch_module,
                        device=device,
                        enabled=amp_active and device.startswith("cuda"),
                    ):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            use_cache=False,
                            output_attentions=False,
                        )
                        loss = outputs.loss
                    if not bool(torch_module.isfinite(loss).all().item()):
                        raise Qwen3SFTError("non_finite_loss")
                    scaler.scale(loss / accumulation_steps).backward()
                backward_count += 1
                if gradient_injector is not None:
                    gradient_injector(model, backward_count, group_index)
                microbatch_metrics.append(
                    {
                        "backward_index": backward_count,
                        "global_microbatch_index": item["global_microbatch_index"],
                        "group_index": group_index,
                        "group_attempt": retry,
                        "example_id": item["example_id"],
                        "loss": float(loss.detach().float().item()),
                        "finite_loss": True,
                    }
                )
                del outputs, loss, batch

            scaler.unscale_(optimizer)
            missing_gradients = [
                name
                for name, parameter in zip(trainable_names, trainable, strict=True)
                if parameter.grad is None
            ]
            if missing_gradients:
                raise Qwen3SFTError(f"missing_lora_gradients:{missing_gradients[:3]}")
            finite_gradients = all(
                bool(torch_module.isfinite(parameter.grad).all().item()) for parameter in trainable
            )
            if finite_gradients:
                grad_norm = torch_module.nn.utils.clip_grad_norm_(trainable, clip_norm)
                grad_norm_value = float(grad_norm.detach().float().item())
                if not math.isfinite(grad_norm_value):
                    raise Qwen3SFTError("non_finite_lora_gradient_norm")
            else:
                grad_norm_value = float("inf")

            parameter_hash_before_step = parameter_hash(model, include_lora=True)
            optimizer_hash_before_step = optimizer_state_hash(optimizer)
            scale_before = float(scaler.get_scale())
            # This call is deliberately unconditional.  Native GradScaler
            # records found_inf during unscale_ and skips the optimizer step.
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            parameter_hash_after_step = parameter_hash(model, include_lora=True)
            optimizer_hash_after_step = optimizer_state_hash(optimizer)
            skipped = scale_after < scale_before

            if not finite_gradients or skipped:
                if not skipped:
                    raise Qwen3SFTError("native_scaler_did_not_skip_non_finite_gradient")
                if parameter_hash_before_step != parameter_hash_after_step:
                    raise Qwen3SFTError("amp_skip_changed_parameters")
                if optimizer_hash_before_step != optimizer_hash_after_step:
                    raise Qwen3SFTError("amp_skip_changed_optimizer_state")
                skipped_updates += 1
                replayed_groups += 1
                replay_metrics.append(
                    {
                        "group_index": group_index,
                        "group_attempt": retry,
                        "example_ids": [item["example_id"] for item in group],
                        "cursor_before": cursor,
                        "cursor_after": cursor,
                        "rng_before": rng_digest(group_rng),
                        "rng_restored": True,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "native_scaler_step_called": True,
                        "parameters_changed": False,
                        "optimizer_state_changed": False,
                    }
                )
                update_metrics.append(
                    {
                        "successful": False,
                        "update_index": update_count + 1,
                        "group_index": group_index,
                        "skipped_by_amp": True,
                        "grad_norm_unscaled": grad_norm_value,
                        "gradients_finite": finite_gradients,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                restore_rng(torch_module, group_rng)
                retry += 1
                if retry > max_replays_per_group:
                    raise Qwen3SFTError("amp_replay_limit_exhausted")
                continue

            update_count += 1
            cursor += accumulation_steps
            update_metrics.append(
                {
                    "successful": True,
                    "update_index": update_count,
                    "group_index": group_index,
                    "skipped_by_amp": False,
                    "grad_norm_unscaled": grad_norm_value,
                    "gradients_finite": finite_gradients,
                    "overall_lora_gradient_norm_positive": grad_norm_value > 0,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                }
            )
            optimizer.zero_grad(set_to_none=True)
            if on_update is not None:
                on_update(
                    {
                        "update_index": update_count,
                        "cursor": cursor,
                        "backward_count": backward_count,
                        "replayed_groups": replayed_groups,
                        "optimizer": optimizer,
                        "scaler": scaler,
                        "rng_state": capture_rng(torch_module),
                        "checkpoint_requested": update_count in checkpoint_steps,
                    }
                )
            break

    base_hash_after = parameter_hash(model, include_lora=False)
    base_grads = [name for name, parameter in base_parameters if parameter.grad is not None]
    if base_hash_before != base_hash_after or base_grads:
        raise Qwen3SFTError("base_model_changed_or_received_gradient")
    changed = []
    for name, parameter in model.named_parameters():
        if name not in lora_before:
            continue
        delta = (parameter.detach().cpu().float() - lora_before[name].float()).abs()
        if bool((delta > 0).any().item()):
            changed.append(
                {
                    "name": name,
                    "max_abs_delta": float(delta.max().item()),
                    "l2_delta": float(delta.norm().item()),
                }
            )
    if not changed:
        raise Qwen3SFTError("lora_parameters_did_not_change")
    if update_count != len(groups):
        raise Qwen3SFTError(f"optimizer_update_contract_failed:{update_count}:{len(groups)}")
    if backward_count > max_backward:
        raise Qwen3SFTError("backward_limit_exceeded")
    successful_updates = [item for item in update_metrics if item.get("successful")]
    if not successful_updates or not all(
        item["grad_norm_unscaled"] > 0 for item in successful_updates
    ):
        raise Qwen3SFTError("overall_lora_gradient_norm_not_positive")
    return {
        "optimizer": optimizer,
        "scaler": scaler,
        "microbatch_metrics": microbatch_metrics,
        "update_metrics": update_metrics,
        "replay_metrics": replay_metrics,
        "microbatches_processed": cursor,
        "successful_optimizer_updates": update_count,
        "skipped_optimizer_updates": skipped_updates,
        "replayed_groups": replayed_groups,
        "backward_count": backward_count,
        "backward_limit": max_backward,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in trainable),
        "optimizer_state_parameter_count": len(optimizer.state),
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_parameters_changed": base_hash_before != base_hash_after,
        "base_gradients_present": base_grads,
        "lora_changed_parameter_count": len(changed),
        "lora_changed_parameters": changed[:32],
        "gradients_finite": all(
            item.get("gradients_finite", True) for item in update_metrics if item.get("successful")
        ),
        "overall_lora_gradient_norm_positive": True,
        "cursor": cursor,
        "schedule_groups": len(groups),
        "schedule_microbatches": len(examples) * epochs,
    }


def _iter_lora_modules(model: Any):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base"):
            yield name, module


def load_tokenized_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 54:
        raise Qwen3SFTError(f"tokenized_dataset_row_count:{len(rows)}")
    return rows
