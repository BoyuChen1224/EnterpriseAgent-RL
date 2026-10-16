"""Run a self-contained raw Transformers diagnostic for the pinned Qwen checkpoint.

This script intentionally does not import or start the EnterpriseAgent HTTP server.
It loads one local checkpoint in one Python process, records cache/model/tokenizer
evidence, runs logits and generation checks, and samples GPU telemetry while each
generation is in progress.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

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
PROMPTS = {
    "prompt_A": "Who are you? Answer in one short sentence.",
    "prompt_B": "用一句话说明你如何帮助分析企业数据。",
}
GENERATION_PROMPT = "prompt_B"
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_result(argv: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # pragma: no cover - depends on the compute node
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
                record["matches_manifest"] = record["bytes"] == manifest_record.get(
                    "bytes"
                ) and record["sha256"] == manifest_record.get("sha256")
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


def normalize_eos_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def tensor_to_list(tensor) -> list[int]:
    return [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]


def prepare_inputs(
    tokenizer, messages: list[dict[str, str]], method: str, device
) -> tuple[Any, str]:
    if method == "A_official_tokenize_true":
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return inputs, raw_text
    if method == "B_enterprise_tokenize_false":
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(raw_text, return_tensors="pt").to(device)
        return inputs, raw_text
    raise ValueError(f"unknown tokenizer method: {method}")


def input_record(tokenizer, inputs, raw_text: str) -> dict[str, Any]:
    input_ids = tensor_to_list(inputs["input_ids"][0])
    attention_mask = tensor_to_list(inputs["attention_mask"][0])
    return {
        "chat_template_text": raw_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_token_count": len(input_ids),
        "first_30_token_ids": input_ids[:30],
        "last_30_token_ids": input_ids[-30:],
        "decoded_input": tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def first_difference(left: list[int], right: list[int]) -> dict[str, Any] | None:
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return {"index": index, "left": left_value, "right": right_value}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left": left[min(len(left), len(right))] if len(left) > len(right) else None,
            "right": right[min(len(left), len(right))] if len(right) > len(left) else None,
        }
    return None


def compare_tokenization(tokenizer, records: dict[str, Any]) -> dict[str, Any]:
    official = records["A_official_tokenize_true"]
    enterprise = records["B_enterprise_tokenize_false"]
    ids_equal = official["input_ids"] == enterprise["input_ids"]
    masks_equal = official["attention_mask"] == enterprise["attention_mask"]
    raw_equal = official["chat_template_text"] == enterprise["chat_template_text"]
    direct_raw = tokenizer(official["chat_template_text"], return_tensors="pt")["input_ids"][
        0
    ].tolist()
    direct_raw = [int(item) for item in direct_raw]
    if ids_equal and masks_equal and raw_equal:
        diagnosis = "A/B exact match: both paths render and tokenize the same prompt."
    elif not raw_equal:
        diagnosis = "The two paths rendered different chat-template text."
    elif not ids_equal:
        diagnosis = (
            "The rendered text is identical but chat_template(tokenize=True) and standalone "
            "tokenizer(text) produced different input IDs; inspect special-token handling."
        )
    else:
        diagnosis = "The input IDs match but attention masks or raw template text differ."
    return {
        "input_ids_equal": ids_equal,
        "attention_mask_equal": masks_equal,
        "chat_template_text_equal": raw_equal,
        "official_ids_match_direct_tokenizer": official["input_ids"] == direct_raw,
        "enterprise_ids_match_direct_tokenizer": enterprise["input_ids"] == direct_raw,
        "first_input_id_difference": first_difference(
            official["input_ids"], enterprise["input_ids"]
        ),
        "first_attention_mask_difference": first_difference(
            official["attention_mask"], enterprise["attention_mask"]
        ),
        "diagnosis": diagnosis,
    }


def token_detail(tokenizer, token_id: int, probability: float | None = None) -> dict[str, Any]:
    return {
        "token_id": token_id,
        "token_string": tokenizer.convert_ids_to_tokens(token_id),
        "decoded_token": tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "probability": probability,
    }


def logits_sanity(tokenizer, outputs, label: str) -> dict[str, Any]:
    import torch

    logits = outputs.logits.detach()
    finite_mask = torch.isfinite(logits)
    finite_values = logits[finite_mask].float()
    final_logits = logits[0, -1].float()
    final_finite = torch.isfinite(final_logits)
    safe_final = torch.nan_to_num(
        final_logits,
        nan=-torch.finfo(final_logits.dtype).max,
        posinf=torch.finfo(final_logits.dtype).max,
        neginf=-torch.finfo(final_logits.dtype).max,
    )
    top_count = min(20, safe_final.numel())
    top_values, top_ids = torch.topk(safe_final, k=top_count)
    probabilities = None
    if bool(final_finite.all()):
        probabilities = torch.softmax(final_logits, dim=-1)
    top_tokens = []
    for value, token_id in zip(top_values.tolist(), top_ids.tolist(), strict=True):
        probability = float(probabilities[token_id].item()) if probabilities is not None else None
        top_tokens.append(
            token_detail(tokenizer, int(token_id), probability) | {"logit": float(value)}
        )
    top1_probability = top_tokens[0]["probability"] if top_tokens else None
    top5_mass = (
        float(probabilities[top_ids[: min(5, top_count)]].sum().item())
        if probabilities is not None
        else None
    )
    finite_stats = {
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() else None,
    }
    return {
        "prompt": label,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_finite_ratio": float(finite_mask.float().mean().item()),
        "logits_finite_count": int(finite_mask.sum().item()),
        "logits_total_count": int(finite_mask.numel()),
        "logits_min": finite_stats["min"],
        "logits_max": finite_stats["max"],
        "logits_mean": finite_stats["mean"],
        "logits_std": finite_stats["std"],
        "final_position_all_finite": bool(final_finite.all()),
        "final_position_top_20": top_tokens,
        "top_1_probability": top1_probability,
        "top_5_probability_mass": top5_mass,
        "top_token_is_exclamation": bool(top_tokens and top_tokens[0]["decoded_token"] == "!"),
    }


class GpuMonitor:
    """Capture nvidia-smi samples while one generation is running."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.stderr_path = output_path.with_suffix(".stderr.txt")
        self.process: subprocess.Popen[str] | None = None
        self.stdout = None
        self.stderr = None

    def __enter__(self) -> GpuMonitor:
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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
    metrics = {
        "utilization_gpu_percent": [],
        "utilization_memory_percent": [],
        "power_draw_w": [],
        "sm_clock_mhz": [],
        "memory_clock_mhz": [],
        "temperature_c": [],
        "memory_used_mib": [],
        "memory_total_mib": [],
    }
    rows = 0
    malformed = 0
    if path.is_file():
        with path.open(newline="") as stream:
            for values in csv.reader(stream):
                if len(values) < len(GPU_QUERY_FIELDS):
                    malformed += 1
                    continue
                rows += 1
                for key, value in zip(metrics, values[2:], strict=False):
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


def generated_record(
    tokenizer,
    model,
    inputs,
    experiment: str,
    input_method: str,
    kwargs: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor_path = output_dir / f"gpu-monitor-{experiment}.csv"
    wall_started = dt.datetime.now(dt.UTC)
    started = time.perf_counter()
    monitor_error = None
    try:
        with GpuMonitor(monitor_path):
            with torch.inference_mode():
                sequences = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    **kwargs,
                )
            torch.cuda.synchronize()
    except Exception as exc:
        monitor_error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    elapsed = time.perf_counter() - started
    wall_finished = dt.datetime.now(dt.UTC)
    prompt_ids = tensor_to_list(inputs["input_ids"][0])
    all_ids = tensor_to_list(sequences[0])
    generated_ids = all_ids[len(prompt_ids) :]
    eos_ids = normalize_eos_ids(model.generation_config.eos_token_id)
    eos_positions = [index for index, token_id in enumerate(generated_ids) if token_id in eos_ids]
    if eos_positions:
        finish_reason = "stop"
    elif len(generated_ids) >= MAX_NEW_TOKENS:
        finish_reason = "length"
    else:
        finish_reason = "unknown"
    frequencies = Counter(generated_ids)
    bang_ids = tokenizer.encode("!")
    bang_ids_no_special = tokenizer.encode("!", add_special_tokens=False)
    return {
        "experiment": experiment,
        "input_method": input_method,
        "prompt": GENERATION_PROMPT,
        "prompt_token_count": len(prompt_ids),
        "generation_kwargs": kwargs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_config_eos_token_id_used": model.generation_config.eos_token_id,
        "generation_config_pad_token_id_used": model.generation_config.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "output": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "decoded_tokens": [
            tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for token_id in generated_ids
        ],
        "generated_token_ids": generated_ids,
        "unique_generated_token_ids": list(dict.fromkeys(generated_ids)),
        "token_frequency": {
            str(token_id): count for token_id, count in sorted(frequencies.items())
        },
        "all_generated_are_same_token": len(frequencies) == 1 if generated_ids else False,
        "eos_token_ids": eos_ids,
        "eos_positions": eos_positions,
        "finish_reason": finish_reason,
        "tokenizer_encode_exclamation_default": bang_ids,
        "tokenizer_encode_exclamation_no_specials": bang_ids_no_special,
        "generated_exclamation_count": sum(
            generated_ids.count(token_id) for token_id in bang_ids_no_special
        ),
        "latency_s": elapsed,
        "tokens_per_second": len(generated_ids) / elapsed if elapsed else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "started_at_utc": wall_started.isoformat(),
        "finished_at_utc": wall_finished.isoformat(),
        "gpu_monitor": summarize_gpu_monitor(monitor_path),
        "gpu_monitor_error": monitor_error,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text())
    if config.get("model_id") != MODEL_ID or config.get("revision") != REVISION:
        raise RuntimeError("model_config_is_not_the_pinned_qwen_revision")
    return config


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
        "schema_version": "1.0",
        "status": "started",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "pid": os.getpid(),
        "git": command_result(["git", "rev-parse", "--show-toplevel"]),
        "git_commit": command_result(["git", "rev-parse", "HEAD"]),
        "git_status": command_result(["git", "status", "--short", "--branch"]),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "http_server_used": False,
        "agent_used": False,
        "protocol_used": False,
        "base_evaluation_used": False,
    }
    write_json(args.output / "diagnostic-progress.json", report)
    try:
        config = load_config(args.model_config)
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
        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(0)
        report["runtime"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "cuda_arch_list": torch.cuda.get_arch_list(),
            "device_properties": {
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "major": torch.cuda.get_device_properties(0).major,
                "minor": torch.cuda.get_device_properties(0).minor,
            },
        }

        load_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        load_kwargs = {
            "revision": REVISION,
            "cache_dir": str(args.cache),
            "local_files_only": True,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **load_kwargs)
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                use_safetensors=True,
                **load_kwargs,
            )
            .to(device)
            .eval()
        )
        torch.cuda.synchronize()
        load_elapsed = time.perf_counter() - load_started

        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
        buffer_dtypes = sorted({str(buffer.dtype) for buffer in model.buffers()})
        input_embedding = model.get_input_embeddings().weight
        output_embedding = model.get_output_embeddings().weight
        model_report = {
            "model_class": type(model).__name__,
            "model_config_model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "vocab_size": model.config.vocab_size,
            "embedding_shape": list(input_embedding.shape),
            "lm_head_shape": list(output_embedding.shape),
            "tie_word_embeddings": model.config.tie_word_embeddings,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_num_parameters": model.num_parameters(),
            "first_parameter_dtype": str(next(model.parameters()).dtype),
            "first_parameter_device": str(next(model.parameters()).device),
            "parameter_devices": parameter_devices,
            "parameter_dtypes": parameter_dtypes,
            "buffer_devices": buffer_devices,
            "buffer_dtypes": buffer_dtypes,
            "all_parameters_on_cuda_0": parameter_devices == ["cuda:0"],
            "all_parameters_fp16": parameter_dtypes == ["torch.float16"],
            "config_eos_token_id": model.config.eos_token_id,
            "generation_config_eos_token_id": model.generation_config.eos_token_id,
            "generation_config_pad_token_id": model.generation_config.pad_token_id,
            "generation_config": model.generation_config.to_dict(),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_len": len(tokenizer),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "load_latency_s": load_elapsed,
            "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        report["model"] = model_report
        write_json(args.output / "model-structure.json", model_report)
        if not model_report["all_parameters_on_cuda_0"] or not model_report["all_parameters_fp16"]:
            raise RuntimeError("model_parameter_placement_or_dtype_check_failed")
        if model_report["model_class"] != "Qwen2ForCausalLM":
            raise RuntimeError("unexpected_model_class")

        messages = {name: [{"role": "user", "content": prompt}] for name, prompt in PROMPTS.items()}
        prepared: dict[str, dict[str, Any]] = {}
        tokenizer_ab: dict[str, Any] = {"prompts": {}}
        for prompt_name, prompt_messages in messages.items():
            prepared[prompt_name] = {}
            tokenizer_ab["prompts"][prompt_name] = {
                "prompt": PROMPTS[prompt_name],
                "methods": {},
            }
            for method in ("A_official_tokenize_true", "B_enterprise_tokenize_false"):
                inputs, raw_text = prepare_inputs(tokenizer, prompt_messages, method, device)
                prepared[prompt_name][method] = inputs
                tokenizer_ab["prompts"][prompt_name]["methods"][method] = input_record(
                    tokenizer, inputs, raw_text
                )
            tokenizer_ab["prompts"][prompt_name]["comparison"] = compare_tokenization(
                tokenizer, tokenizer_ab["prompts"][prompt_name]["methods"]
            )
        report["tokenizer_ab"] = tokenizer_ab
        write_json(args.output / "tokenizer-ab.json", tokenizer_ab)

        bang_default = tokenizer.encode("!")
        bang_no_special = tokenizer.encode("!", add_special_tokens=False)
        bang_check = {
            "tokenizer_encode_exclamation_default": bang_default,
            "tokenizer_encode_exclamation_no_specials": bang_no_special,
            "decoded_default": tokenizer.decode(
                bang_default, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "decoded_no_specials": tokenizer.decode(
                bang_no_special, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        }
        report["exclamation_token_check"] = bang_check

        logits_report: dict[str, Any] = {}
        for prompt_name in PROMPTS:
            with torch.inference_mode():
                logits_output = model(**prepared[prompt_name]["A_official_tokenize_true"])
            logits_report[prompt_name] = logits_sanity(tokenizer, logits_output, prompt_name)
            del logits_output
        report["logits_sanity"] = logits_report
        write_json(args.output / "logits-sanity.json", logits_report)

        generation_results = {}
        experiment_specs = {
            "A_greedy_cache": (
                "A_official_tokenize_true",
                {"do_sample": False, "use_cache": True},
            ),
            "B_official_sampling": (
                "A_official_tokenize_true",
                {
                    "do_sample": True,
                    "use_cache": True,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repetition_penalty": 1.1,
                },
            ),
            "C_enterprise_agent_config": (
                "B_enterprise_tokenize_false",
                {
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": tokenizer.eos_token_id,
                },
            ),
        }
        for experiment, (method, generation_kwargs) in experiment_specs.items():
            generation_results[experiment] = generated_record(
                tokenizer,
                model,
                prepared[GENERATION_PROMPT][method],
                experiment,
                method,
                generation_kwargs,
                args.output,
            )
            write_json(
                args.output / f"generation-{experiment}.json", generation_results[experiment]
            )
        generation_results["C_enterprise_agent_config"]["enterprise_agent_config"] = {
            "do_sample": False,
            "use_cache": True,
            "temperature_in_model_server": 1.0,
            "top_p_in_model_server": 1.0,
            "top_k_in_model_server": 50,
            "pad_token_id_in_model_server": tokenizer.eos_token_id,
            "note": "temperature/top_p/top_k are irrelevant when do_sample=False",
        }
        report["generation_results"] = generation_results

        official_messages = [{"role": "user", "content": "Who are you?"}]
        official_inputs, official_text = prepare_inputs(
            tokenizer, official_messages, "A_official_tokenize_true", device
        )
        official_result = generated_record(
            tokenizer,
            model,
            official_inputs,
            "official_minimal_example",
            "A_official_tokenize_true",
            {},
            args.output,
        )
        official_result["prompt"] = "Who are you?"
        official_result["chat_template_text"] = official_text
        official_result["input_ids"] = tensor_to_list(official_inputs["input_ids"][0])
        official_result["attention_mask"] = tensor_to_list(official_inputs["attention_mask"][0])
        report["official_minimal_example"] = official_result
        write_json(args.output / "official-minimal-example.json", official_result)

        after = command_result(["nvidia-smi"], timeout=30)
        (args.output / "nvidia-smi-after.txt").write_text(after["stdout"] + after["stderr"])
        write_json(args.output / "nvidia-smi-after.json", after)
        report["nvidia_smi_after"] = after
        report["status"] = "complete"
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "diagnostic-report.json", report)
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(args.output / "diagnostic-report.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
