"""Diagnose Qwen2 numerical stability on the pinned V100 software stack.

This is a raw Transformers diagnostic only.  It deliberately does not import
the EnterpriseAgent server, protocol, agent, benchmark, SFT, or GRPO code.
The checkpoint and its generation_config are read-only inputs.  The four
execution paths are kept independent in the output report:

* FP32 + eager reference;
* FP16 + eager with module and Qwen2Attention internal probes;
* FP16 + eager with FP32 attention-score accumulation (diagnostic reference);
* FP16 + SDPA with only the PyTorch math backend enabled.

The attention probes and reference implementation are local runtime wrappers;
the installed Transformers package is never edited.  A fail-fast logits guard
is installed around generation so non-finite logits cannot silently become
token ID 0.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import tomllib
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_WEIGHT_BYTES = 3_087_467_144
EXPECTED_WEIGHT_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors",
)

# This is the same prompt used as prompt_B by Job 20385865.  Its token count
# is recorded and asserted across every experiment; no prompt shortening or
# sequence padding is introduced by this diagnostic.
PROMPT = "用一句话说明你如何帮助分析企业数据。"
MAX_NEW_TOKENS = 32
GPU_QUERY_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "memory.used",
    "memory.total",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_result(argv: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # pragma: no cover - node-dependent
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verify_cache(cache: Path) -> dict[str, Any]:
    repo = cache / "models--Qwen--1.5B-Instruct"
    # The actual Hugging Face cache directory has the namespace in its name.
    repo = cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    snapshot = repo / "snapshots" / REVISION
    manifest_path = cache / "qwen-cache-manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    file_records: dict[str, Any] = {}
    for name in REQUIRED_FILES:
        path = snapshot / name
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "is_symlink": path.is_symlink(),
        }
        if path.is_file():
            record.update(
                {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "resolved_target": str(path.resolve()),
                }
            )
            if manifest:
                manifest_record = manifest.get("files", {}).get(name, {})
                record["manifest_bytes"] = manifest_record.get("bytes")
                record["manifest_sha256"] = manifest_record.get("sha256")
                record["matches_manifest"] = (
                    record["bytes"] == manifest_record.get("bytes")
                    and record["sha256"] == manifest_record.get("sha256")
                )
        file_records[name] = record

    incomplete_blobs = sorted(path.name for path in (repo / "blobs").glob("*.incomplete"))
    weight = file_records["model.safetensors"]
    weight_matches_official = (
        weight.get("bytes") == EXPECTED_WEIGHT_BYTES
        and weight.get("sha256") == EXPECTED_WEIGHT_SHA256
    )
    all_files_match_manifest = all(
        record.get("matches_manifest") is True for record in file_records.values()
    )
    revision_is_fixed = snapshot.name == REVISION and snapshot.parent.name == "snapshots"
    complete = bool(
        snapshot.is_dir()
        and all(record["exists"] for record in file_records.values())
        and not incomplete_blobs
        and weight_matches_official
        and all_files_match_manifest
        and revision_is_fixed
    )
    return {
        "status": "pass" if complete else "fail",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "cache": str(cache),
        "snapshot": str(snapshot),
        "revision_is_fixed": revision_is_fixed,
        "required_files": list(REQUIRED_FILES),
        "incomplete_blobs": incomplete_blobs,
        "manifest_path": str(manifest_path),
        "manifest_revision": manifest.get("revision") if manifest else None,
        "manifest_model": manifest.get("model") if manifest else None,
        "all_files_match_manifest": all_files_match_manifest,
        "official_weight_bytes": EXPECTED_WEIGHT_BYTES,
        "official_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "weight_matches_official": weight_matches_official,
        "files": file_records,
    }


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_to_list(tensor: Any) -> list[int]:
    return [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]


def prepare_inputs(tokenizer: Any, device: Any) -> tuple[Any, str, dict[str, Any]]:
    messages = [{"role": "user", "content": PROMPT}]
    raw_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    input_ids = tensor_to_list(inputs["input_ids"][0])
    attention_mask = tensor_to_list(inputs["attention_mask"][0])
    record = {
        "prompt": PROMPT,
        "chat_template_text": raw_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_token_count": len(input_ids),
        "decoded_input": tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }
    return inputs, raw_text, record


def json_number(value: Any) -> float | str | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else str(number)


def tensor_stats(tensor: Any) -> dict[str, Any]:
    """Return non-mutating statistics without replacing NaN/Inf values."""

    import torch

    if tensor is None or not torch.is_tensor(tensor):
        return {
            "dtype": None,
            "shape": None,
            "finite_ratio": None,
            "finite_count": None,
            "total_count": None,
            "nan_count": None,
            "posinf_count": None,
            "neginf_count": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
            "raw_min": None,
            "raw_max": None,
        }
    detached = tensor.detach()
    values = detached.float()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    total_count = int(finite.numel())
    nan_count = int(torch.isnan(values).sum().item())
    posinf_count = int(torch.isposinf(values).sum().item())
    neginf_count = int(torch.isneginf(values).sum().item())
    finite_values = values[finite]
    raw_min = values.min().item() if total_count else None
    raw_max = values.max().item() if total_count else None
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "finite_ratio": finite_count / total_count if total_count else None,
        "finite_count": finite_count,
        "total_count": total_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "min": json_number(finite_values.min().item()) if finite_count else None,
        "max": json_number(finite_values.max().item()) if finite_count else None,
        "mean": json_number(finite_values.mean().item()) if finite_count else None,
        "std": json_number(finite_values.std(unbiased=False).item()) if finite_count else None,
        "abs_max": json_number(finite_values.abs().max().item()) if finite_count else None,
        "raw_min": json_number(raw_min),
        "raw_max": json_number(raw_max),
    }


def logits_sanity(tokenizer: Any, outputs: Any, label: str) -> dict[str, Any]:
    import torch

    logits = outputs.logits.detach()
    stats = tensor_stats(logits)
    final_logits = logits[0, -1].float()
    final_finite = torch.isfinite(final_logits)
    top_tokens: list[dict[str, Any]] | None = None
    top1_probability: float | None = None
    top5_mass: float | None = None
    if bool(final_finite.all()):
        top_values, top_ids = torch.topk(final_logits, k=min(20, final_logits.numel()))
        probabilities = torch.softmax(final_logits, dim=-1)
        top_tokens = []
        for value, token_id in zip(top_values.tolist(), top_ids.tolist(), strict=True):
            token_id = int(token_id)
            top_tokens.append(
                {
                    "token_id": token_id,
                    "token_string": tokenizer.convert_ids_to_tokens(token_id),
                    "decoded_token": tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "logit": float(value),
                    "probability": float(probabilities[token_id].item()),
                }
            )
        top1_probability = top_tokens[0]["probability"] if top_tokens else None
        top5_mass = float(probabilities[top_ids[:5]].sum().item())
    return {
        "prompt": label,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        **stats,
        "final_position_all_finite": bool(final_finite.all()),
        "final_position_finite_ratio": float(final_finite.float().mean().item()),
        "final_position_top_20": top_tokens,
        "top_1_probability": top1_probability,
        "top_5_probability_mass": top5_mass,
        "top_token_is_exclamation": bool(
            top_tokens and top_tokens[0]["decoded_token"] == "!"
        ),
    }


class TraceRecorder:
    """Store scalar summaries for ordered module and attention events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, name: str, tensor: Any, kind: str) -> None:
        event = {
            "event_index": len(self.events),
            "name": name,
            "kind": kind,
            **tensor_stats(tensor),
        }
        self.events.append(event)

    @staticmethod
    def is_nonfinite(event: dict[str, Any]) -> bool:
        ratio = event.get("finite_ratio")
        return ratio is not None and ratio < 1.0

    def first_nonfinite(self, kind: str | None = None) -> dict[str, Any] | None:
        for event in self.events:
            if (kind is None or event["kind"] == kind) and self.is_nonfinite(event):
                return event
        return None

    def report(self) -> dict[str, Any]:
        first_any = self.first_nonfinite()
        first_module = self.first_nonfinite("module_output")
        return {
            "event_count": len(self.events),
            "first_nonfinite": first_any,
            "first_nonfinite_module_output": first_module,
            "events": self.events,
        }


def first_tensor(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def install_module_hooks(model: Any, recorder: TraceRecorder) -> list[Any]:
    targets: list[tuple[str, Any]] = [("embed_tokens", model.model.embed_tokens)]
    for index, layer in enumerate(model.model.layers):
        prefix = f"decoder_layer[{index}]"
        targets.extend(
            [
                (prefix, layer),
                (f"{prefix}.input_layernorm", layer.input_layernorm),
                (f"{prefix}.self_attn", layer.self_attn),
                (f"{prefix}.self_attn.q_proj", layer.self_attn.q_proj),
                (f"{prefix}.self_attn.k_proj", layer.self_attn.k_proj),
                (f"{prefix}.self_attn.v_proj", layer.self_attn.v_proj),
                (f"{prefix}.self_attn.o_proj", layer.self_attn.o_proj),
                (f"{prefix}.post_attention_layernorm", layer.post_attention_layernorm),
                (f"{prefix}.mlp.gate_proj", layer.mlp.gate_proj),
                (f"{prefix}.mlp.up_proj", layer.mlp.up_proj),
                (f"{prefix}.mlp.down_proj", layer.mlp.down_proj),
            ]
        )
    targets.extend(
        [
            ("final_norm", model.model.norm),
            ("lm_head", model.lm_head),
        ]
    )
    handles = []
    for name, module in targets:
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: recorder.record(
                    name, first_tensor(output), "module_output"
                )
            )
        )
    return handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def attention_trace_hook(self: Any, name: str, tensor: Any) -> None:
    recorder = getattr(self, "_stability_trace_recorder", None)
    if recorder is not None:
        recorder.record(
            f"decoder_layer[{self.layer_idx}].self_attn.{name}",
            tensor,
            "attention_internal",
        )


def attention_forward_impl(
    self: Any,
    hidden_states: Any,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Any = None,
    position_embeddings: Any = None,
    *,
    score_dtype: Any = None,
) -> tuple[Any, Any, Any]:
    """Runtime copy of Qwen2Attention.forward with optional FP32 score math."""

    import math as _math
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    attention_trace_hook(self, "query_linear", query_states)
    attention_trace_hook(self, "key_linear", key_states)
    attention_trace_hook(self, "value_linear", value_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    attention_trace_hook(self, "query_states", query_states)
    attention_trace_hook(self, "key_states", key_states)
    attention_trace_hook(self, "value_states", value_states)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    attention_trace_hook(self, "rotated_query", query_states)
    attention_trace_hook(self, "rotated_key", key_states)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attention_trace_hook(self, "repeated_key", key_states)
    attention_trace_hook(self, "repeated_value", value_states)

    if score_dtype is None:
        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3))
        attention_trace_hook(self, "attention_scores_qk", attention_scores)
        scaled_scores = attention_scores / _math.sqrt(self.head_dim)
        attention_trace_hook(self, "attention_scores_scaled", scaled_scores)
        score_values = scaled_scores
        value_for_output = value_states
    else:
        query_for_scores = query_states.to(score_dtype)
        key_for_scores = key_states.to(score_dtype)
        value_for_output = value_states.to(score_dtype)
        attention_scores = torch.matmul(query_for_scores, key_for_scores.transpose(2, 3))
        attention_trace_hook(self, "attention_scores_qk_fp32_reference", attention_scores)
        scaled_scores = attention_scores / _math.sqrt(self.head_dim)
        attention_trace_hook(self, "attention_scores_scaled_fp32_reference", scaled_scores)
        score_values = scaled_scores

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        score_values = score_values + causal_mask
        attention_trace_hook(self, "attention_scores_masked", score_values)

    attention_weights = torch.nn.functional.softmax(score_values, dim=-1, dtype=torch.float32)
    attention_trace_hook(self, "softmax_output_fp32", attention_weights)
    if score_dtype is None:
        attention_weights = attention_weights.to(query_states.dtype)
    attention_weights = torch.nn.functional.dropout(
        attention_weights, p=self.attention_dropout, training=self.training
    )
    attention_output = torch.matmul(attention_weights, value_for_output)
    if score_dtype is not None:
        attention_output = attention_output.to(query_states.dtype)
    attention_trace_hook(self, "attention_output", attention_output)

    if attention_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"attention output has unexpected size {attention_output.size()}"
        )
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(bsz, q_len, self.hidden_size)
    attention_output = self.o_proj(attention_output)
    attention_trace_hook(self, "attention_output_projected", attention_output)

    if not output_attentions:
        attention_weights = None
    return attention_output, attention_weights, past_key_value


def traced_attention_forward(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return attention_forward_impl(self, *args, **kwargs, score_dtype=None)


def fp32_score_attention_forward(
    self: Any, *args: Any, **kwargs: Any
) -> tuple[Any, Any, Any]:
    import torch

    return attention_forward_impl(self, *args, **kwargs, score_dtype=torch.float32)


def patch_attention_forwards(
    model: Any,
    function: Callable[..., tuple[Any, Any, Any]],
    recorder: TraceRecorder | None = None,
) -> list[tuple[Any, Any, bool]]:
    patches: list[tuple[Any, Any, bool]] = []
    for layer in model.model.layers:
        attention = layer.self_attn
        had_instance_forward = "forward" in attention.__dict__
        old_forward = attention.__dict__.get("forward")
        patches.append((attention, old_forward, had_instance_forward))
        if recorder is not None:
            attention._stability_trace_recorder = recorder
        attention.forward = types.MethodType(function, attention)
    return patches


def restore_attention_forwards(patches: list[tuple[Any, Any, bool]]) -> None:
    for attention, old_forward, had_instance_forward in patches:
        if had_instance_forward:
            attention.forward = old_forward
        else:
            attention.__dict__.pop("forward", None)
        attention.__dict__.pop("_stability_trace_recorder", None)


class GpuMonitor:
    """Capture nvidia-smi samples while one generation is running."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.stderr_path = output_path.with_suffix(".stderr.txt")
        self.process: subprocess.Popen[str] | None = None
        self.stdout: Any = None
        self.stderr: Any = None

    def __enter__(self) -> "GpuMonitor":
        self.stdout = self.output_path.open("w")
        self.stderr = self.stderr_path.open("w")
        query = ",".join(GPU_QUERY_FIELDS)
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    "-l",
                    "1",
                ],
                stdout=self.stdout,
                stderr=self.stderr,
                text=True,
            )
        except Exception:
            self.stdout.close()
            self.stderr.close()
            self.stdout = None
            self.stderr = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()


def number(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def summarize_gpu_monitor(path: Path) -> dict[str, Any]:
    metric_names = (
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "power_draw_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "memory_used_mib",
        "memory_total_mib",
    )
    metrics = {name: [] for name in metric_names}
    rows = 0
    malformed = 0
    if path.is_file():
        with path.open(newline="") as stream:
            for values in csv.reader(stream):
                if len(values) < len(GPU_QUERY_FIELDS):
                    malformed += 1
                    continue
                rows += 1
                for key, value in zip(metric_names, values[2:], strict=True):
                    parsed = number(value)
                    if parsed is not None:
                        metrics[key].append(parsed)
    summary: dict[str, Any] = {
        "sample_count": rows,
        "malformed_rows": malformed,
        "raw_file": str(path),
    }
    for key, values in metrics.items():
        summary[key] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "samples": len(values),
        }
    return summary


def gpu_metadata(torch: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu_name": runtime.get("gpu_name"),
        "compute_capability": runtime.get("compute_capability"),
        "torch": runtime.get("torch"),
        "transformers": runtime.get("transformers"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "device": "cuda:0",
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


class NonFiniteLogitsError(RuntimeError):
    error_code = "non_finite_logits"

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(self.error_code)


class FiniteLogitGuard:
    """Fail before Transformers makes a sampling/greedy decision."""

    def __init__(
        self,
        *,
        dtype: str,
        backend: str,
        prompt_token_count: int,
        runtime: dict[str, Any],
    ) -> None:
        self.dtype = dtype
        self.backend = backend
        self.prompt_token_count = prompt_token_count
        self.runtime = runtime
        self.forward_count = 0
        self.forward_started_at: float | None = None
        self.forward_timings: list[dict[str, Any]] = []
        self.failure: dict[str, Any] | None = None

    def pre_hook(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        self.forward_started_at = time.perf_counter()

    def post_hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        finished = time.perf_counter()
        started = self.forward_started_at or finished
        elapsed = finished - started
        logits = getattr(output, "logits", None)
        input_ids = _kwargs.get("input_ids") if isinstance(_kwargs, dict) else None
        if input_ids is None and _args:
            input_ids = _args[0]
        input_length = int(input_ids.shape[-1]) if input_ids is not None else None
        timing = {
            "forward_index": self.forward_count,
            "generation_step": self.forward_count,
            "input_sequence_length": input_length,
            "latency_s": elapsed,
        }
        self.forward_timings.append(timing)
        self.forward_count += 1
        if logits is None:
            return output
        stats = tensor_stats(logits)
        finite = bool(torch_isfinite_all(logits))
        if not finite:
            payload = {
                "error_code": NonFiniteLogitsError.error_code,
                "dtype": self.dtype,
                "layer": "lm_head",
                "backend": self.backend,
                "prompt_token_count": self.prompt_token_count,
                "generation_step": timing["generation_step"],
                "input_sequence_length": input_length,
                "logits_statistics": stats,
                "gpu_metadata": gpu_metadata_from_runtime(self.runtime),
                "forward_timing": timing,
            }
            self.failure = payload
            raise NonFiniteLogitsError(payload)
        return output


def torch_isfinite_all(tensor: Any) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def gpu_metadata_from_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu_name": runtime.get("gpu_name"),
        "compute_capability": runtime.get("compute_capability"),
        "torch": runtime.get("torch"),
        "transformers": runtime.get("transformers"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "device": runtime.get("device", "cuda:0"),
    }


def register_guard(model: Any, guard: FiniteLogitGuard) -> list[Any]:
    """Use kwargs-aware hooks on pinned PyTorch; retain a compatibility fallback."""

    try:
        return [
            model.register_forward_pre_hook(guard.pre_hook, with_kwargs=True),
            model.register_forward_hook(guard.post_hook, with_kwargs=True),
        ]
    except TypeError:  # pragma: no cover - only for older torch APIs
        def pre_legacy(module: Any, args: tuple[Any, ...]) -> None:
            guard.pre_hook(module, args, {})

        def post_legacy(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            return guard.post_hook(module, args, {}, output)

        return [
            model.register_forward_pre_hook(pre_legacy),
            model.register_forward_hook(post_legacy),
        ]


def remove_guard(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


@contextlib.contextmanager
def forced_sdpa_math(torch: Any) -> Iterator[dict[str, Any]]:
    """Force the old PyTorch 2.2 API's math-only SDPA backend when available."""

    legacy = getattr(getattr(torch, "backends", None), "cuda", None)
    legacy_kernel = getattr(legacy, "sdp_kernel", None)
    if legacy_kernel is not None:
        with legacy_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield {
                "status": "forced",
                "api": "torch.backends.cuda.sdp_kernel",
                "flash": False,
                "math": True,
                "mem_efficient": False,
            }
        return

    attention = getattr(getattr(torch, "nn", None), "attention", None)
    kernel = getattr(attention, "sdpa_kernel", None)
    SDPBackend = getattr(attention, "SDPBackend", None)
    if kernel is not None and SDPBackend is not None:
        with kernel(SDPBackend.MATH):
            yield {
                "status": "forced",
                "api": "torch.nn.attention.sdpa_kernel",
                "backend": "MATH",
            }
        return
    raise RuntimeError("pytorch_sdpa_math_backend_api_unavailable")


def load_model(
    *,
    torch: Any,
    AutoModelForCausalLM: Any,
    model_id: str,
    cache: Path,
    dtype: Any,
    attention: str,
    runtime: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    load_kwargs = {
        "revision": REVISION,
        "cache_dir": str(cache),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attention,
        **load_kwargs,
    ).to("cuda:0").eval()
    sync_cuda(torch)
    elapsed = time.perf_counter() - started
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    buffer_dtypes = sorted({str(buffer.dtype) for buffer in model.buffers()})
    generation_config = model.generation_config.to_dict()
    metadata = {
        "model_class": type(model).__name__,
        "config_model_type": model.config.model_type,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "requested_dtype": str(dtype),
        "parameter_dtypes": parameter_dtypes,
        "parameter_devices": parameter_devices,
        "buffer_dtypes": buffer_dtypes,
        "buffer_devices": buffer_devices,
        "all_parameters_on_cuda_0": parameter_devices == ["cuda:0"],
        "all_parameters_requested_dtype": parameter_dtypes == [str(dtype)],
        "parameter_count": model.num_parameters(),
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "vocab_size": model.config.vocab_size,
        "generation_config_before": generation_config,
        "load_latency_s": elapsed,
        "load_current_allocated_bytes": int(torch.cuda.memory_allocated()),
        "load_current_reserved_bytes": int(torch.cuda.memory_reserved()),
        "load_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "load_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "runtime_gpu": gpu_metadata(torch, runtime),
    }
    return model, metadata


def forward_once(
    torch: Any,
    model: Any,
    inputs: Any,
    *,
    use_cache: bool = False,
) -> tuple[dict[str, Any], Any]:
    torch.cuda.reset_peak_memory_stats()
    sync_cuda(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=use_cache)
    sync_cuda(torch)
    elapsed = time.perf_counter() - started
    return {
        "latency_s": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }, outputs


def run_generation(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    inputs: Any,
    experiment: str,
    dtype: str,
    backend: str,
    prompt_token_count: int,
    runtime: dict[str, Any],
    output_dir: Path,
    generation_config_before: dict[str, Any],
    context: contextlib.AbstractContextManager[Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    monitor_path = output_dir / f"gpu-monitor-{experiment}.csv"
    guard = FiniteLogitGuard(
        dtype=dtype,
        backend=backend,
        prompt_token_count=prompt_token_count,
        runtime=runtime,
    )
    handles = register_guard(model, guard)
    started_at = dt.datetime.now(dt.UTC)
    started = time.perf_counter()
    status = "complete"
    error: dict[str, Any] | None = None
    sequences = None
    try:
        with GpuMonitor(monitor_path):
            scope = context if context is not None else contextlib.nullcontext()
            with scope:
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
                sync_cuda(torch)
    except NonFiniteLogitsError as exc:
        status = "failed"
        error = exc.payload
        write_json(output_dir / f"guard-failure-{experiment}.json", error)
    except Exception as exc:  # pragma: no cover - node-dependent
        status = "failed"
        error = {
            "error_code": "generation_exception",
            "type": type(exc).__name__,
            "message": str(exc),
            "dtype": dtype,
            "backend": backend,
        }
    finally:
        remove_guard(handles)
    elapsed = time.perf_counter() - started
    generation_config_after = model.generation_config.to_dict()
    result: dict[str, Any] = {
        "experiment": experiment,
        "status": status,
        "dtype": dtype,
        "backend": backend,
        "prompt": PROMPT,
        "prompt_token_count": prompt_token_count,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_kwargs": {"do_sample": False, "use_cache": True},
        "generation_error": error,
        "guard_error_code": error.get("error_code") if error else None,
        "latency_s": elapsed,
        "forward_timings": guard.forward_timings,
        "ttft_s": guard.forward_timings[0]["latency_s"] if guard.forward_timings else None,
        "decode_forward_count": max(len(guard.forward_timings) - 1, 0),
        "decode_latency_s": (
            sum(item["latency_s"] for item in guard.forward_timings[1:])
            if len(guard.forward_timings) > 1
            else None
        ),
        "decode_tokens_per_second": (
            (len(guard.forward_timings) - 1)
            / sum(item["latency_s"] for item in guard.forward_timings[1:])
            if len(guard.forward_timings) > 1
            and sum(item["latency_s"] for item in guard.forward_timings[1:]) > 0
            else None
        ),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "gpu_monitor": summarize_gpu_monitor(monitor_path),
        "generation_config_after": generation_config_after,
        "generation_config_before": generation_config_before,
        "generation_config_unchanged": generation_config_before == generation_config_after,
        "generation_config_was_not_assigned_by_diagnostic": True,
    }
    if sequences is not None:
        prompt_ids = tensor_to_list(inputs["input_ids"][0])
        all_ids = tensor_to_list(sequences[0])
        generated_ids = all_ids[len(prompt_ids) :]
        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = [int(token_id) for token_id in (eos_ids or [])]
        frequencies = Counter(generated_ids)
        eos_positions = [
            index for index, token_id in enumerate(generated_ids) if token_id in eos_ids
        ]
        result.update(
            {
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "unique_generated_token_ids": list(dict.fromkeys(generated_ids)),
                "token_frequency": {
                    str(token_id): count for token_id, count in sorted(frequencies.items())
                },
                "all_generated_are_same_token": len(frequencies) == 1 if generated_ids else False,
                "eos_token_ids": eos_ids,
                "eos_positions": eos_positions,
                "finish_reason": "stop"
                if eos_positions
                else "length"
                if len(generated_ids) >= MAX_NEW_TOKENS
                else "unknown",
                "output": tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "decoded_tokens": [
                    tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for token_id in generated_ids
                ],
                "generated_exclamation_count": sum(
                    generated_ids.count(token_id)
                    for token_id in tokenizer.encode("!", add_special_tokens=False)
                ),
            }
        )
    else:
        result["generated_token_ids"] = None
        result["generated_token_count"] = 0
        result["output"] = None
    return result


def release_model(torch: Any, model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    sync_cuda(torch)


def model_generation_config_unchanged(model: Any, before: dict[str, Any]) -> dict[str, Any]:
    after = model.generation_config.to_dict()
    return {
        "before": before,
        "after": after,
        "unchanged": before == after,
    }


def write_conclusion(path: Path, report: dict[str, Any]) -> None:
    experiments = report.get("experiments", {})
    fp32 = experiments.get("A_fp32_reference", {})
    fp16 = experiments.get("B_fp16_eager_layer_trace", {})
    sdpa = experiments.get("C_fp16_sdpa_math", {})
    reference = experiments.get("D_fp16_fp32_attention_scores", {})
    trace = fp16.get("layer_trace", {})
    first = trace.get("first_nonfinite") or {}
    qk_names = {
        event.get("name")
        for event in trace.get("events", [])
        if event.get("kind") == "attention_internal"
        and "attention_scores" in event.get("name", "")
        and event.get("finite_ratio") is not None
        and event.get("finite_ratio") < 1.0
    }
    lines = [
        "# GPU inference stability conclusion",
        "",
        f"- Job: `{report.get('job_id')}`",
        f"- GPU: `{report.get('runtime', {}).get('gpu_name')}`, CC `{report.get('runtime', {}).get('compute_capability')}`",
        f"- Model: `{MODEL_ID}`, revision `{REVISION}`",
        "- Raw Transformers only; no Protocol, Agent, Base, SFT, or GRPO.",
        "",
        "## Acceptance answers",
        "",
        f"1. FP32 normal: `{fp32.get('raw_forward', {}).get('final_position_all_finite') is True}`; generation status `{fp32.get('generation', {}).get('status')}`.",
        f"2. FP16 eager first non-finite: `{first.get('name')}` (kind `{first.get('kind')}`), or `none`.",
        f"3. Attention QK/softmax involved: `{bool(qk_names)}`; events `{sorted(qk_names)}`.",
        f"4. FP16 SDPA math-only normal: `{sdpa.get('raw_forward', {}).get('final_position_all_finite') is True}`; generation status `{sdpa.get('generation', {}).get('status')}`.",
        f"5. FP32 attention-score accumulation normal: `{reference.get('raw_forward', {}).get('final_position_all_finite') is True}`; generation status `{reference.get('generation', {}).get('status')}`.",
        "6. Recommended V100 path: FP16 weights with the finite path established by this run; do not treat FP32 as the final fix until the backend evidence is reviewed.",
        "7. Put the finite-logit guard on the model output immediately before the generation sampling/greedy decision; this run uses a model forward hook and fails with `non_finite_logits`.",
        "",
        "## Root-cause wording",
        "",
        "The evidence tests the narrower hypothesis that Qwen2.5 BF16 checkpoint values overflow on the current V100 + PyTorch 2.2.2 + Transformers 4.46.3 FP16 execution path. It does not claim that FP16 is categorically invalid on Volta; Qwen recommends FP16 for Compute Capability below 8.0.",
        "",
        "## Performance boundary",
        "",
        "Only finite-logit, semantically normal paths should be used as performance baselines. The previous 0.941 tok/s abnormal generation is not reused here.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "status": "started",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "pid": os.getpid(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "http_server_used": False,
        "agent_used": False,
        "protocol_used": False,
        "base_evaluation_used": False,
        "sft_used": False,
        "grpo_used": False,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git": command_result(["git", "rev-parse", "--show-toplevel"]),
        "git_commit": command_result(["git", "rev-parse", "HEAD"]),
        "git_status": command_result(["git", "status", "--short", "--branch"]),
        "python": sys.version,
        "platform": platform.platform(),
        "experiments": {},
    }
    write_json(args.output / "diagnostic-progress.json", report)
    try:
        config = tomllib.loads(args.model_config.read_text())
        if config.get("model_id") != MODEL_ID or config.get("revision") != REVISION:
            raise RuntimeError("model_config_is_not_the_pinned_qwen_revision")
        report["model_config"] = config
        cache_report = verify_cache(args.cache)
        report["cache_integrity"] = cache_report
        write_json(args.output / "cache-integrity.json", cache_report)
        if cache_report["status"] != "pass":
            raise RuntimeError("local_pinned_cache_integrity_failed")

        before = command_result(["nvidia-smi"], timeout=30)
        (args.output / "nvidia-smi-before.txt").write_text(before["stdout"] + before["stderr"])
        write_json(args.output / "nvidia-smi-before.json", before)
        environment = {
            "nvidia_smi_before": before,
            "slurm_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("SLURM_") or key in {"CUDA_VISIBLE_DEVICES", "CUDA_HOME"}
            },
        }
        write_json(args.output / "environment.json", environment)
        (args.output / "pip-freeze.txt").write_text(
            command_result([sys.executable, "-m", "pip", "freeze"], timeout=60)["stdout"]
        )

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires_exactly_one_visible_cuda_device")
        capability = torch.cuda.get_device_capability(0)
        runtime = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "cuda_arch_list": torch.cuda.get_arch_list(),
            "device_properties": {
                "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                "multi_processor_count": int(
                    torch.cuda.get_device_properties(0).multi_processor_count
                ),
            },
        }
        report["runtime"] = runtime
        write_json(args.output / "runtime.json", runtime)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(args.cache),
            local_files_only=True,
            trust_remote_code=False,
        )
        inputs, chat_template_text, prompt_record = prepare_inputs(tokenizer, torch.device("cuda:0"))
        report["prompt_record"] = prompt_record
        report["chat_template_text"] = chat_template_text
        write_json(args.output / "prompt.json", prompt_record | {"chat_template_text": chat_template_text})
        prompt_token_count = prompt_record["prompt_token_count"]

        # A: FP32/eager reference.
        fp32_model, fp32_meta = load_model(
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_id=MODEL_ID,
            cache=args.cache,
            dtype=torch.float32,
            attention="eager",
            runtime=runtime,
        )
        fp32_forward_timing, fp32_outputs = forward_once(torch, fp32_model, inputs)
        fp32_raw = logits_sanity(tokenizer, fp32_outputs, "prompt_B")
        del fp32_outputs
        fp32_generation = run_generation(
            torch=torch,
            tokenizer=tokenizer,
            model=fp32_model,
            inputs=inputs,
            experiment="A_fp32_reference",
            dtype="torch.float32",
            backend="eager",
            prompt_token_count=prompt_token_count,
            runtime=runtime,
            output_dir=args.output,
            generation_config_before=fp32_meta["generation_config_before"],
        )
        fp32_config = model_generation_config_unchanged(
            fp32_model, fp32_meta["generation_config_before"]
        )
        report["experiments"]["A_fp32_reference"] = {
            "model": fp32_meta,
            "raw_forward_timing": fp32_forward_timing,
            "raw_forward": fp32_raw,
            "generation": fp32_generation,
            "generation_config_integrity": fp32_config,
        }
        write_json(args.output / "A-fp32-reference.json", report["experiments"]["A_fp32_reference"])
        write_json(args.output / "diagnostic-progress.json", report | {"phase": "A_complete"})
        release_model(torch, fp32_model)

        # B: FP16/eager module hooks plus an exact local attention trace.
        fp16_model, fp16_meta = load_model(
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_id=MODEL_ID,
            cache=args.cache,
            dtype=torch.float16,
            attention="eager",
            runtime=runtime,
        )
        trace = TraceRecorder()
        module_handles = install_module_hooks(fp16_model, trace)
        attention_patches = patch_attention_forwards(
            fp16_model, traced_attention_forward, recorder=trace
        )
        try:
            fp16_forward_timing, fp16_outputs = forward_once(torch, fp16_model, inputs)
            fp16_raw = logits_sanity(tokenizer, fp16_outputs, "prompt_B")
            del fp16_outputs
        finally:
            restore_attention_forwards(attention_patches)
            remove_hooks(module_handles)
        layer_trace = trace.report()
        write_json(args.output / "B-fp16-eager-layer-trace.json", layer_trace)
        fp16_generation = run_generation(
            torch=torch,
            tokenizer=tokenizer,
            model=fp16_model,
            inputs=inputs,
            experiment="B_fp16_eager",
            dtype="torch.float16",
            backend="eager",
            prompt_token_count=prompt_token_count,
            runtime=runtime,
            output_dir=args.output,
            generation_config_before=fp16_meta["generation_config_before"],
        )
        fp16_config = model_generation_config_unchanged(
            fp16_model, fp16_meta["generation_config_before"]
        )
        report["experiments"]["B_fp16_eager_layer_trace"] = {
            "model": fp16_meta,
            "raw_forward_timing": fp16_forward_timing,
            "raw_forward": fp16_raw,
            "layer_trace": layer_trace,
            "generation": fp16_generation,
            "generation_config_integrity": fp16_config,
        }
        write_json(
            args.output / "B-fp16-eager-report.json",
            report["experiments"]["B_fp16_eager_layer_trace"],
        )
        write_json(args.output / "diagnostic-progress.json", report | {"phase": "B_complete"})

        # D: run only when B identifies non-finite QK/score/softmax tensors.
        qk_nonfinite = any(
            event["kind"] == "attention_internal"
            and "attention_scores" in event["name"]
            and event.get("finite_ratio") is not None
            and event["finite_ratio"] < 1.0
            for event in trace.events
        )
        if qk_nonfinite:
            reference_patches = patch_attention_forwards(fp16_model, fp32_score_attention_forward)
            try:
                reference_forward_timing, reference_outputs = forward_once(
                    torch, fp16_model, inputs
                )
                reference_raw = logits_sanity(tokenizer, reference_outputs, "prompt_B")
                del reference_outputs
                reference_generation = run_generation(
                    torch=torch,
                    tokenizer=tokenizer,
                    model=fp16_model,
                    inputs=inputs,
                    experiment="D_fp16_fp32_attention_scores",
                    dtype="torch.float16",
                    backend="eager_fp32_attention_scores",
                    prompt_token_count=prompt_token_count,
                    runtime=runtime,
                    output_dir=args.output,
                    generation_config_before=fp16_meta["generation_config_before"],
                )
            finally:
                restore_attention_forwards(reference_patches)
            report["experiments"]["D_fp16_fp32_attention_scores"] = {
                "status": "complete",
                "trigger": "FP16 eager attention_scores/softmax trace contained non-finite values",
                "qkv_parameter_dtype": "torch.float16",
                "qk_matmul_dtype": "torch.float32",
                "scale_dtype": "torch.float32",
                "softmax_dtype": "torch.float32",
                "attention_output_cast_dtype": "torch.float16",
                "raw_forward_timing": reference_forward_timing,
                "raw_forward": reference_raw,
                "generation": reference_generation,
                "generation_config_integrity": model_generation_config_unchanged(
                    fp16_model, fp16_meta["generation_config_before"]
                ),
            }
        else:
            report["experiments"]["D_fp16_fp32_attention_scores"] = {
                "status": "skipped",
                "reason": "FP16 eager trace did not identify non-finite attention score or softmax tensors",
            }
        write_json(
            args.output / "D-fp16-fp32-attention-scores.json",
            report["experiments"]["D_fp16_fp32_attention_scores"],
        )
        write_json(args.output / "diagnostic-progress.json", report | {"phase": "D_complete"})
        release_model(torch, fp16_model)

        # C: FP16/SDPA, with math-only backend forcing recorded explicitly.
        sdpa_model, sdpa_meta = load_model(
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_id=MODEL_ID,
            cache=args.cache,
            dtype=torch.float16,
            attention="sdpa",
            runtime=runtime,
        )
        try:
            with forced_sdpa_math(torch) as sdpa_backend:
                sdpa_forward_timing, sdpa_outputs = forward_once(torch, sdpa_model, inputs)
                sdpa_raw = logits_sanity(tokenizer, sdpa_outputs, "prompt_B")
                del sdpa_outputs
                sdpa_generation = run_generation(
                    torch=torch,
                    tokenizer=tokenizer,
                    model=sdpa_model,
                    inputs=inputs,
                    experiment="C_fp16_sdpa_math",
                    dtype="torch.float16",
                    backend="sdpa_math_only",
                    prompt_token_count=prompt_token_count,
                    runtime=runtime,
                    output_dir=args.output,
                    generation_config_before=sdpa_meta["generation_config_before"],
                    context=forced_sdpa_math(torch),
                )
            sdpa_result: dict[str, Any] = {
                "status": "complete",
                "backend_control": sdpa_backend,
                "model": sdpa_meta,
                "raw_forward_timing": sdpa_forward_timing,
                "raw_forward": sdpa_raw,
                "generation": sdpa_generation,
                "generation_config_integrity": model_generation_config_unchanged(
                    sdpa_model, sdpa_meta["generation_config_before"]
                ),
            }
        except Exception as exc:
            sdpa_result = {
                "status": "failed_or_unavailable",
                "model": sdpa_meta,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "reason": "SDPA math-only API unavailable or execution failed; environment was not upgraded",
                },
            }
        report["experiments"]["C_fp16_sdpa_math"] = sdpa_result
        write_json(args.output / "C-fp16-sdpa-math.json", sdpa_result)
        write_json(args.output / "diagnostic-progress.json", report | {"phase": "C_complete"})
        release_model(torch, sdpa_model)

        after = command_result(["nvidia-smi"], timeout=30)
        (args.output / "nvidia-smi-after.txt").write_text(after["stdout"] + after["stderr"])
        write_json(args.output / "nvidia-smi-after.json", after)
        report["nvidia_smi_after"] = after
        report["status"] = "complete"
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "diagnostic-report.json", report)
        write_conclusion(args.output / "diagnostic-conclusion.md", report)
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "diagnostic-report.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
