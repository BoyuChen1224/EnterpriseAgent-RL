"""Small, dependency-light helpers for stable autoregressive inference."""

# 集中控制 attention 后端与非有限 logits 检查，供模型服务、SFT 和 GRPO 复用。
# Transformers 的 sdpa 只是接口选项；要强制 math，还需 PyTorch SDPA 后端上下文。
# 训练时上下文应覆盖 forward/backward/重算，推理时包住 generate 的实际模型调用。

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Literal

AttentionBackend = Literal["eager", "sdpa_math"]


def transformers_attention_implementation(backend: AttentionBackend) -> str:
    """Map the explicit runtime setting to Transformers' official API value."""

    if backend == "eager":
        return "eager"
    if backend == "sdpa_math":
        return "sdpa"
    raise ValueError(f"unsupported_attention_backend:{backend}")


# 兼容两代 PyTorch SDPA 控制 API。没有可用的显式 math 控制时拒绝静默退回其他 GPU 后端。
def _sdpa_backend_control(torch_module: Any, backend: AttentionBackend) -> dict[str, Any]:
    if backend == "eager":
        return {
            "requested": backend,
            "status": "not_applicable",
            "attention_implementation": "eager",
        }

    if backend != "sdpa_math":
        raise ValueError(f"unsupported_attention_backend:{backend}")

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {
            "requested": backend,
            "status": "not_applicable",
            "reason": "cuda_unavailable",
            "attention_implementation": "sdpa",
            "flash": False,
            "math": True,
            "mem_efficient": False,
        }

    legacy = getattr(getattr(torch_module, "backends", None), "cuda", None)
    legacy_kernel = getattr(legacy, "sdp_kernel", None)
    if legacy_kernel is not None:
        return {
            "requested": backend,
            "status": "forced",
            "api": "torch.backends.cuda.sdp_kernel",
            "attention_implementation": "sdpa",
            "flash": False,
            "math": True,
            "mem_efficient": False,
        }

    attention = getattr(getattr(torch_module, "nn", None), "attention", None)
    kernel = getattr(attention, "sdpa_kernel", None)
    sdp_backend = getattr(attention, "SDPBackend", None)
    if kernel is not None and sdp_backend is not None:
        return {
            "requested": backend,
            "status": "forced",
            "api": "torch.nn.attention.sdpa_kernel",
            "attention_implementation": "sdpa",
            "backend": "MATH",
            "flash": False,
            "math": True,
            "mem_efficient": False,
        }
    raise RuntimeError("pytorch_sdpa_math_backend_api_unavailable")


def describe_attention_backend(torch_module: Any, backend: AttentionBackend) -> dict[str, Any]:
    """Return the backend control that will be used without entering a context."""

    return _sdpa_backend_control(torch_module, backend).copy()


# 在当前动态作用域内强制指定后端，退出时恢复；CPU 路径不套用 CUDA 控制。
@contextlib.contextmanager
def attention_backend_context(
    torch_module: Any, backend: AttentionBackend
) -> Iterator[dict[str, Any]]:
    """Apply the selected attention policy only for the current model call.

    PyTorch 2.2.2 exposes the legacy ``torch.backends.cuda.sdp_kernel`` API.
    The newer ``torch.nn.attention.sdpa_kernel`` spelling is retained only as
    a fallback for environments that provide it.  CPU execution deliberately
    uses a no-op context because CUDA backend controls are not applicable.
    """

    control = _sdpa_backend_control(torch_module, backend)
    if control["status"] != "forced":
        yield control
        return

    if control["api"] == "torch.backends.cuda.sdp_kernel":
        legacy = torch_module.backends.cuda.sdp_kernel
        with legacy(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield control
        return

    attention = torch_module.nn.attention
    with attention.sdpa_kernel(attention.SDPBackend.MATH):
        yield control


class NonFiniteLogitsError(RuntimeError):
    """Raised before a non-finite logits tensor can select a token."""

    error_code = "non_finite_logits"

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(self.error_code)


def _logits_statistics(logits: Any) -> dict[str, Any]:
    import torch

    detached = logits.detach()
    values = detached.float()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total_count = int(finite.numel())
    finite_values = values[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_ratio": finite_count / total_count if total_count else None,
        "finite_count": finite_count,
        "total_count": total_count,
        "nan_count": int(torch.isnan(detached).sum().item()),
        "posinf_count": int(torch.isposinf(detached).sum().item()),
        "neginf_count": int(torch.isneginf(detached).sum().item()),
        "min": float(finite_values.min().item()) if finite_count else None,
        "max": float(finite_values.max().item()) if finite_count else None,
    }


# 检查原始模型 logits 的 NaN/Inf 并记录诊断。出现非有限值时报告问题，不伪造正常回答。
def check_finite_logits(
    logits: Any,
    *,
    generation_step: int,
    context: Mapping[str, Any],
) -> Any:
    """Return finite logits, or fail with a complete diagnostic payload."""

    import torch

    finite = torch.isfinite(logits)
    if bool(finite.all().item()):
        return logits

    statistics = _logits_statistics(logits)
    payload = {
        "error_code": NonFiniteLogitsError.error_code,
        "generation_step": generation_step,
        "logits_shape": statistics["shape"],
        "dtype": statistics["dtype"],
        "device": statistics["device"],
        "finite_ratio": statistics["finite_ratio"],
        "min": statistics["min"],
        "max": statistics["max"],
        "nan_count": statistics["nan_count"],
        "posinf_count": statistics["posinf_count"],
        "neginf_count": statistics["neginf_count"],
        "attention_backend": context.get("attention_backend"),
        "model_revision": context.get("model_revision"),
        "gpu_name": context.get("gpu_name"),
        "model_dtype": context.get("model_dtype"),
        "diagnostic_context": dict(context),
    }
    raise NonFiniteLogitsError(payload)


# 接入自回归生成的每一步，及时捕获中途才出现的数值异常。
class FiniteLogitsProcessor:
    """Transformers-compatible processor checked on every generation step."""

    def __init__(self, *, prompt_tokens: int, context: Mapping[str, Any]):
        self.prompt_tokens = prompt_tokens
        self.context = dict(context)
        self.steps_checked = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        input_tokens = int(input_ids.shape[-1])
        generation_step = max(input_tokens - self.prompt_tokens, 0)
        self.steps_checked += 1
        return check_finite_logits(
            scores,
            generation_step=generation_step,
            context=self.context,
        )
