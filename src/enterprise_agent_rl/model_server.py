"""Single GPU Transformers server. No model imports or loads on module import."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import threading
import time
import tomllib
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import Field

from enterprise_agent_rl.common import Contract
from enterprise_agent_rl.gpu_guard import require_compute_node
from enterprise_agent_rl.inference_stability import (
    AttentionBackend,
    FiniteLogitsProcessor,
    NonFiniteLogitsError,
    attention_backend_context,
    describe_attention_backend,
    transformers_attention_implementation,
)
from enterprise_agent_rl.model_protocol import GenerationSettings


class ModelSettings(Contract):
    model_id: Literal["Qwen/Qwen2.5-1.5B-Instruct"]
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dtype: Literal["float16"] = "float16"
    attention_backend: AttentionBackend = "sdpa_math"
    max_context_tokens: int = Field(default=8192, ge=512, le=16384)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    @classmethod
    def load(cls, path: Path) -> ModelSettings:
        values = tomllib.loads(path.read_text())
        # Accept the pre-stabilization key while making the persisted config
        # and model dump use the explicit runtime setting.
        if "attention" in values and "attention_backend" not in values:
            values["attention_backend"] = values.pop("attention")
        if values.get("attention_backend") == "sdpa":
            values["attention_backend"] = "sdpa_math"
        return cls.model_validate(values)


class Message(Contract):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=100000)


class CompletionRequest(Contract):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=150)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    seed: int = Field(default=17, ge=0)
    timeout_s: float = Field(default=120, gt=0, le=1800)
    stream: Literal[False] = False


def append_event(path: Path, event: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class QwenEngine:
    def __init__(self, cfg: ModelSettings, cache: Path, output: Path):
        require_compute_node()
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.cfg = torch, cfg
        self.events = output / "generations.jsonl"
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "qwen-cache-manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("cache_integrity_manifest_missing")
        cache_manifest = json.loads(manifest_path.read_text())
        if (
            cache_manifest.get("model") != cfg.model_id
            or cache_manifest.get("revision") != cfg.revision
            or not cache_manifest.get("weight_files")
        ):
            raise RuntimeError("cache_integrity_manifest_mismatch")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires_one_visible_cuda_device")
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
            raise RuntimeError("wheel_missing_volta_support_or_unsupported_device")
        attention_implementation = transformers_attention_implementation(cfg.attention_backend)
        backend_control = describe_attention_backend(torch, cfg.attention_backend)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        kwargs = dict(
            revision=cfg.revision,
            cache_dir=str(cache),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kwargs)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.float16,
                attn_implementation=attention_implementation,
                output_attentions=False,
                use_safetensors=True,
                **kwargs,
            )
            .to("cuda:0")
            .eval()
        )
        torch.cuda.synchronize()
        if next(self.model.parameters()).dtype != torch.float16:
            raise RuntimeError("model_not_fp16")
        if self.model.config._attn_implementation != attention_implementation:
            raise RuntimeError("model_attention_backend_mismatch")
        if getattr(self.model.config, "output_attentions", False):
            raise RuntimeError("output_attentions_must_be_false_for_sdpa")
        if getattr(self.model.generation_config, "remove_invalid_values", False):
            raise RuntimeError("generation_config_non_finite_repair_enabled")
        model_dtype = str(next(self.model.parameters()).dtype)
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = list(capability)
        load_time = time.monotonic() - started
        self.device = "cuda:0"
        self.model_dtype = model_dtype
        self.gpu_name = gpu_name
        self.compute_capability = compute_capability
        self.load_time = load_time
        self.backend_control = backend_control
        self.metadata = {
            "model": cfg.model_dump(),
            "model_revision": cfg.revision,
            "model_dtype": model_dtype,
            "device": self.device,
            "attention_backend": cfg.attention_backend,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "effective_dtype": model_dtype,
            "effective_attention": self.model.config._attn_implementation,
            "attention_implementation": self.model.config._attn_implementation,
            "output_attentions": False,
            "sdpa_backend_control": backend_control,
            "checkpoint_generation_config": self.model.generation_config.to_dict(),
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "architectures": torch.cuda.get_arch_list(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "node": os.uname().nodename,
            "pid": os.getpid(),
            "load_time": load_time,
            "load_latency_s": load_time,
            "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cache_manifest": {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "size_bytes": manifest_path.stat().st_size,
                "cache_size_bytes_unique": cache_manifest.get("cache_size_bytes_unique"),
                "integrity": cache_manifest.get("integrity"),
            },
        }
        (output / "model-load.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def _base_telemetry(self) -> dict:
        return {
            "model_revision": self.cfg.revision,
            "model_dtype": self.model_dtype,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "attention_backend": self.cfg.attention_backend,
            "sdpa_backend_control": self.backend_control,
            "input_tokens": None,
            "output_tokens": 0,
            "load_time": self.load_time,
            "generation_latency": None,
            "throughput": None,
            "peak_allocated": None,
            "peak_reserved": None,
            "non_finite_logits_detected": False,
        }

    def _save_diagnostic_context(self, event_id: str, payload: dict) -> None:
        path = self.events.parent / "diagnostics" / f"non-finite-logits-{event_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload["diagnostic_path"] = str(path)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        except OSError as exc:  # Preserve the original numerical failure.
            payload["diagnostic_context_save_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    def generate(self, request: CompletionRequest) -> dict:
        from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        torch = self.torch
        started = time.monotonic()
        deadline = started + request.timeout_s

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        event = {
            "id": str(uuid.uuid4()),
            "request": request.model_dump(),
            "error": None,
            "diagnostic_context": None,
            "telemetry": self._base_telemetry(),
        }
        inputs = generated = new_tokens = None
        torch.cuda.reset_peak_memory_stats()
        try:
            if request.model != self.cfg.model_id:
                raise ValueError("model_id_mismatch")
            text = self.tokenizer.apply_chat_template(
                [m.model_dump() for m in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda:0")
            prompt_tokens = inputs.input_ids.shape[-1]
            event["telemetry"]["input_tokens"] = prompt_tokens
            if prompt_tokens + request.max_tokens > self.cfg.max_context_tokens:
                raise ValueError("context_length_exceeded")
            torch.manual_seed(request.seed)
            kwargs = {"do_sample": request.temperature > 0}
            if request.temperature > 0:
                kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=0)
            guard = FiniteLogitsProcessor(
                prompt_tokens=prompt_tokens,
                context={
                    "attention_backend": self.cfg.attention_backend,
                    "model_revision": self.cfg.revision,
                    "model_dtype": self.model_dtype,
                    "device": self.device,
                    "gpu_name": self.gpu_name,
                    "compute_capability": self.compute_capability,
                    "sdpa_backend_control": self.backend_control,
                },
            )
            with attention_backend_context(torch, self.cfg.attention_backend) as backend_control:
                event["telemetry"]["sdpa_backend_control"] = backend_control
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([Deadline()]),
                        logits_processor=LogitsProcessorList([guard]),
                        output_attentions=False,
                        **kwargs,
                    )
            torch.cuda.synchronize()
            new_tokens = generated[0, prompt_tokens:]
            content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            elapsed = time.monotonic() - started
            output_tokens = len(new_tokens)
            event.update(
                raw_output=content,
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens},
            )
            event["telemetry"].update(
                output_tokens=output_tokens,
                generation_latency=elapsed,
                throughput=output_tokens / elapsed if elapsed else None,
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("inference_timeout")
            if not content.strip():
                raise ValueError("empty_output")
            return {
                "id": event["id"],
                "model": self.cfg.model_id,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "length"
                        if len(new_tokens) >= request.max_tokens
                        else "stop",
                    }
                ],
                "usage": event["usage"],
                "telemetry": event["telemetry"],
            }
        except NonFiniteLogitsError as exc:
            event["diagnostic_context"] = exc.payload
            event["telemetry"]["non_finite_logits_detected"] = True
            self._save_diagnostic_context(event["id"], exc.payload)
            event["error"] = {
                "code": NonFiniteLogitsError.error_code,
                "type": type(exc).__name__,
                "message": str(exc),
                "diagnostic_context": exc.payload,
            }
            return {
                "error": event["error"],
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "telemetry": event.get("telemetry", {}),
                "diagnostic_context": event["diagnostic_context"],
            }
        except torch.cuda.OutOfMemoryError:
            event["error"] = {"code": "oom"}
            return {
                "error": event["error"],
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "telemetry": event.get("telemetry", {}),
                "diagnostic_context": event.get("diagnostic_context"),
            }
        except Exception as exc:
            event["error"] = {
                "code": "inference_timeout"
                if isinstance(exc, TimeoutError)
                else str(exc)
                if isinstance(exc, ValueError)
                else "generation_error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            return {
                "error": event["error"],
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "telemetry": event.get("telemetry", {}),
                "diagnostic_context": event.get("diagnostic_context"),
            }
        finally:
            elapsed = time.monotonic() - started
            event["elapsed_s"] = elapsed
            event["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            event["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            output_tokens = event["telemetry"].get("output_tokens") or 0
            event["telemetry"].update(
                generation_latency=event["telemetry"].get("generation_latency") or elapsed,
                throughput=output_tokens / elapsed if output_tokens and elapsed else None,
                peak_allocated=event["peak_allocated_bytes"],
                peak_reserved=event["peak_reserved_bytes"],
                peak_allocated_bytes=event["peak_allocated_bytes"],
                peak_reserved_bytes=event["peak_reserved_bytes"],
                generation_latency_s=event["telemetry"].get("generation_latency") or elapsed,
                generation_tokens_per_s=(
                    output_tokens / elapsed if output_tokens and elapsed else None
                ),
            )
            append_event(self.events, event)
            del inputs, generated, new_tokens
            gc.collect()
            torch.cuda.empty_cache()


def create_model_app(engine) -> FastAPI:
    app = FastAPI(title="Slurm-local Qwen FP16")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ready", "metadata": engine.metadata}

    @app.post("/v1/chat/completions")
    def complete(request: CompletionRequest):
        if not lock.acquire(blocking=False):
            return JSONResponse({"error": {"code": "model_busy"}}, status_code=429)
        try:
            result = engine.generate(request)
            return JSONResponse(result, status_code=503 if "error" in result else 200)
        finally:
            lock.release()

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model/qwen2.5-1.5b.toml"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    require_compute_node()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        engine = QwenEngine(ModelSettings.load(args.config), args.cache, args.output)
    except Exception as exc:
        append_event(
            args.output / "startup-errors.jsonl",
            {"code": "model_load_failed", "type": type(exc).__name__, "message": str(exc)[:1000]},
        )
        raise
    import uvicorn

    uvicorn.run(create_model_app(engine), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
