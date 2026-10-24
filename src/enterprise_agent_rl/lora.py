"""Small dependency-free LoRA adapter helpers.

PEFT is intentionally not part of the locked GPU environment.  The helpers
below therefore use the same conceptual adapter layout (``adapter_config``
plus ``adapter_model.safetensors``) while keeping base-model loading and
production inference unchanged when no adapter is supplied.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _get_submodule(root: Any, name: str) -> Any:
    current = root
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _set_submodule(root: Any, name: str, value: Any) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    setattr(_get_submodule(root, parent_name), child_name, value)


def _is_target_linear(name: str, module: Any, target_modules: set[str]) -> bool:
    import torch

    return isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in target_modules


def inject_lora_modules(
    model: Any,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
) -> list[str]:
    """Replace target Linear modules with trainable low-rank residuals."""

    import torch
    import torch.nn.functional as F

    if rank <= 0:
        raise ValueError("lora_rank_must_be_positive")
    target_set = set(target_modules)

    class LoRALinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.lora_A = torch.nn.Parameter(
                torch.empty(
                    rank,
                    base.in_features,
                    dtype=torch.float32,
                    device=base.weight.device,
                )
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(
                    base.out_features,
                    rank,
                    dtype=torch.float32,
                    device=base.weight.device,
                )
            )
            torch.nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            self.scaling = float(alpha) / float(rank)
            self.lora_dropout = torch.nn.Dropout(dropout)

        def forward(self, inputs):
            result = self.base(inputs)
            residual = F.linear(
                F.linear(self.lora_dropout(inputs).to(self.lora_A.dtype), self.lora_A),
                self.lora_B,
            )
            return result + residual.to(result.dtype) * self.scaling

    module_names = [
        name
        for name, module in list(model.named_modules())
        if name and _is_target_linear(name, module, target_set)
    ]
    if not module_names:
        raise ValueError(f"lora_target_modules_not_found:{sorted(target_set)}")
    for name in module_names:
        _set_submodule(model, name, LoRALinear(_get_submodule(model, name)))
    return module_names


def iter_lora_modules(model: Any):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base"):
            yield name, module


def trainable_lora_parameters(model: Any) -> list[Any]:
    parameters = []
    for name, parameter in model.named_parameters():
        # lora_A/lora_B are registered directly as Parameters, so PyTorch
        # names them ``...lora_A`` rather than ``...lora_A.weight``.
        if name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}:
            parameter.requires_grad_(True)
            parameters.append(parameter)
        else:
            parameter.requires_grad_(False)
    if not parameters:
        raise ValueError("no_trainable_lora_parameters")
    return parameters


def lora_parameter_names(model: Any) -> list[str]:
    """Return the registered parameter names belonging to manual LoRA."""

    return [
        name
        for name, _ in model.named_parameters()
        if name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}
    ]


def adapter_state_dict(model: Any) -> dict[str, Any]:
    """Return PEFT-shaped keys without serializing the frozen base model."""

    state: dict[str, Any] = {}
    for module_name, module in iter_lora_modules(model):
        prefix = f"base_model.model.{module_name}"
        state[f"{prefix}.lora_A.weight"] = module.lora_A.detach().cpu().contiguous()
        state[f"{prefix}.lora_B.weight"] = module.lora_B.detach().cpu().contiguous()
    if not state:
        raise ValueError("empty_lora_adapter_state")
    return state


def save_adapter(
    model: Any,
    adapter_dir: Path,
    *,
    model_id: str,
    revision: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
    module_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write adapter config and safetensors, returning a manifest summary."""

    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    modules = list(module_names or (name for name, _ in iter_lora_modules(model)))
    state = adapter_state_dict(model)
    save_file(state, str(adapter_dir / "adapter_model.safetensors"))
    config = {
        "base_model_name_or_path": model_id,
        "base_model_revision": revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": list(target_modules),
        "module_names": modules,
        "bias": "none",
        "inference_mode": False,
        "implementation": "enterprise_agent_rl.manual_lora_v1",
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "path": str(adapter_dir),
        "weight_file": str(adapter_dir / "adapter_model.safetensors"),
        "module_count": len(modules),
        "parameter_count": sum(int(value.numel()) for value in state.values()),
        "config": config,
    }


def merge_lora_adapter(
    model: Any,
    adapter_dir: Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Merge an adapter into a loaded base model for production inference."""

    import torch
    from safetensors.torch import load_file

    config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError("adapter_files_missing")
    config = json.loads(config_path.read_text())
    if config.get("implementation") != "enterprise_agent_rl.manual_lora_v1":
        raise RuntimeError("unsupported_adapter_implementation")
    if expected_model_id and config.get("base_model_name_or_path") != expected_model_id:
        raise RuntimeError("adapter_base_model_mismatch")
    if expected_revision and config.get("base_model_revision") != expected_revision:
        raise RuntimeError("adapter_base_revision_mismatch")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0:
        raise RuntimeError("adapter_rank_invalid")
    module_names = config.get("module_names")
    if not isinstance(module_names, list) or not module_names:
        raise RuntimeError("adapter_module_names_missing")
    state = load_file(str(weight_path), device="cpu")
    expected_keys = {
        key
        for name in module_names
        for key in (
            f"base_model.model.{name}.lora_A.weight",
            f"base_model.model.{name}.lora_B.weight",
        )
    }
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        extra = sorted(set(state) - expected_keys)
        raise RuntimeError(f"adapter_state_keys_mismatch:missing={missing[:2]}:extra={extra[:2]}")

    merged = 0
    delta_norm = 0.0
    scaling = alpha / rank
    with torch.no_grad():
        for name in module_names:
            module = _get_submodule(model, name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"adapter_target_not_linear:{name}")
            a = state[f"base_model.model.{name}.lora_A.weight"].to(
                device=module.weight.device, dtype=torch.float32
            )
            b = state[f"base_model.model.{name}.lora_B.weight"].to(
                device=module.weight.device, dtype=torch.float32
            )
            if tuple(a.shape) != (rank, module.in_features):
                raise RuntimeError(f"adapter_a_shape_mismatch:{name}")
            if tuple(b.shape) != (module.out_features, rank):
                raise RuntimeError(f"adapter_b_shape_mismatch:{name}")
            delta = (b @ a) * scaling
            if not bool(torch.isfinite(delta).all().item()):
                raise RuntimeError(f"adapter_non_finite_delta:{name}")
            module.weight.add_(delta.to(dtype=module.weight.dtype))
            delta_norm += float(delta.norm().item())
            merged += 1
    return {
        "adapter_dir": str(adapter_dir),
        "implementation": config["implementation"],
        "module_count": merged,
        "rank": rank,
        "alpha": alpha,
        "delta_norm_sum": delta_norm,
        "merged_into_base": True,
    }


def load_lora_adapter(
    model: Any,
    adapter_dir: Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Load a saved manual-LoRA state into already-injected modules.

    This is distinct from :func:`merge_lora_adapter`: checkpoints resume
    training and must restore the low-rank tensors without adding their delta
    to the frozen base weights.
    """

    import torch
    from safetensors.torch import load_file

    config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError("adapter_files_missing")
    config = json.loads(config_path.read_text())
    if config.get("implementation") != "enterprise_agent_rl.manual_lora_v1":
        raise RuntimeError("unsupported_adapter_implementation")
    if expected_model_id and config.get("base_model_name_or_path") != expected_model_id:
        raise RuntimeError("adapter_base_model_mismatch")
    if expected_revision and config.get("base_model_revision") != expected_revision:
        raise RuntimeError("adapter_base_revision_mismatch")
    module_names = config.get("module_names")
    if not isinstance(module_names, list) or not module_names:
        raise RuntimeError("adapter_module_names_missing")
    state = load_file(str(weight_path), device="cpu")
    expected_keys = {
        key
        for name in module_names
        for key in (
            f"base_model.model.{name}.lora_A.weight",
            f"base_model.model.{name}.lora_B.weight",
        )
    }
    if set(state) != expected_keys:
        raise RuntimeError("adapter_state_keys_mismatch")
    loaded = 0
    with torch.no_grad():
        for name in module_names:
            module = _get_submodule(model, name)
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                raise RuntimeError(f"adapter_target_not_injected:{name}")
            a = state[f"base_model.model.{name}.lora_A.weight"]
            b = state[f"base_model.model.{name}.lora_B.weight"]
            if tuple(a.shape) != tuple(module.lora_A.shape) or tuple(b.shape) != tuple(
                module.lora_B.shape
            ):
                raise RuntimeError(f"adapter_tensor_shape_mismatch:{name}")
            module.lora_A.copy_(a.to(device=module.lora_A.device, dtype=module.lora_A.dtype))
            module.lora_B.copy_(b.to(device=module.lora_B.device, dtype=module.lora_B.dtype))
            loaded += 1
    return {
        "adapter_dir": str(adapter_dir),
        "implementation": config["implementation"],
        "module_count": loaded,
        "rank": int(config["r"]),
        "alpha": float(config["lora_alpha"]),
        "loaded_without_base_merge": True,
    }
