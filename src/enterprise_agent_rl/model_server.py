"""Single GPU Transformers server. No model imports or loads on module import."""

from __future__ import annotations

import argparse
import gc
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
from enterprise_agent_rl.model_protocol import GenerationSettings


class ModelSettings(Contract):
    model_id: Literal["Qwen/Qwen2.5-1.5B-Instruct"]
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dtype: Literal["float16"] = "float16"
    attention: Literal["eager"] = "eager"
    max_context_tokens: int = Field(default=8192, ge=512, le=16384)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    @classmethod
    def load(cls, path: Path) -> ModelSettings:
        return cls.model_validate(tomllib.loads(path.read_text()))


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
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires_one_visible_cuda_device")
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
            raise RuntimeError("wheel_missing_volta_support_or_unsupported_device")
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
                attn_implementation="eager",
                use_safetensors=True,
                **kwargs,
            )
            .to("cuda:0")
            .eval()
        )
        # Do not inherit sampling defaults from the checkpoint's generation_config.
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        torch.cuda.synchronize()
        if next(self.model.parameters()).dtype != torch.float16:
            raise RuntimeError("model_not_fp16")
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("model_not_eager_attention")
        self.metadata = {
            "model": cfg.model_dump(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "effective_dtype": str(next(self.model.parameters()).dtype),
            "effective_attention": self.model.config._attn_implementation,
            "checkpoint_generation_config": self.model.generation_config.to_dict(),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": capability,
            "architectures": torch.cuda.get_arch_list(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "node": os.uname().nodename,
            "pid": os.getpid(),
            "load_latency_s": time.monotonic() - started,
            "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        (output / "model-load.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def generate(self, request: CompletionRequest) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList

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
            "telemetry": {},
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
            if prompt_tokens + request.max_tokens > self.cfg.max_context_tokens:
                raise ValueError("context_length_exceeded")
            torch.manual_seed(request.seed)
            kwargs = {"do_sample": request.temperature > 0}
            if request.temperature > 0:
                kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=0)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([Deadline()]),
                    **kwargs,
                )
            torch.cuda.synchronize()
            new_tokens = generated[0, prompt_tokens:]
            content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            elapsed = time.monotonic() - started
            event.update(
                raw_output=content,
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(new_tokens)},
                telemetry={
                    "generation_latency_s": elapsed,
                    "generation_tokens_per_s": len(new_tokens) / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
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
        except torch.cuda.OutOfMemoryError:
            event["error"] = {"code": "oom"}
            return {
                "error": event["error"],
                "raw_output": event.get("raw_output"),
                "usage": event.get("usage", {}),
                "telemetry": event.get("telemetry", {}),
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
            }
        finally:
            event["elapsed_s"] = time.monotonic() - started
            event["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            event["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            event["telemetry"].update(
                peak_allocated_bytes=event["peak_allocated_bytes"],
                peak_reserved_bytes=event["peak_reserved_bytes"],
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
