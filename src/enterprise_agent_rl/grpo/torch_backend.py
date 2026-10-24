"""Validated Qwen/LoRA backend adapter for the reusable online loop.

Imports of torch/Transformers are deliberately lazy.  Configuration validation,
finalization, and CPU fake integration therefore never load model assets.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_agent_rl.agent import SYSTEM_PROMPT, run_episode
from enterprise_agent_rl.common import Decision, PublicTask, Settings
from enterprise_agent_rl.inference_stability import (
    attention_backend_context,
    transformers_attention_implementation,
)
from enterprise_agent_rl.model_protocol import GenerationSettings, ModelFailure, parse_action
from enterprise_agent_rl.trajectories import Trajectory

from .budget import BudgetError, BudgetGuard
from .config import GrpoConfig, RunContext
from .contracts import EpisodeSample, ScoreResult, TaskInput, TokenRecord
from .evidence import AttemptHandle, EvidencePersistenceError, EvidenceStore
from .io import atomic_write_json, sha256_bytes, sha256_file
from .state import capture_rng_state, recursive_fingerprint, restore_rng_state, seed_everything
from .upstream import LocalGRPOAdapter


def _torch():
    import torch

    return torch


def _digest_tensor_state(state: dict[str, Any]) -> str:
    return recursive_fingerprint(state)


def _require_v100_device(torch: Any) -> dict[str, Any]:
    """Reject a compatible CUDA allocation that is not the frozen V100 contract."""

    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    capability = tuple(torch.cuda.get_device_capability(0))
    memory_bytes = int(properties.total_memory)
    if "V100" not in name.upper() or capability != (7, 0):
        raise RuntimeError(
            f"v100_contract_mismatch:name={name}:capability={capability}:memory={memory_bytes}"
        )
    if not 30 * 1024**3 <= memory_bytes <= 36 * 1024**3:
        raise RuntimeError(f"v100_memory_contract_mismatch:{name}:{memory_bytes}")
    return {
        "name": name,
        "compute_capability": list(capability),
        "total_memory_bytes": memory_bytes,
        "device_index": 0,
    }


def _record_gpu_device(context: RunContext, device: dict[str, Any], *, mode: str, arm: str | None):
    output = context.run_dir / "setup" / f"gpu-device-{mode}-{arm or 'train'}.json"
    atomic_write_json(
        output,
        {"schema_version": "t023-gpu-device-v1", **device, "mode": mode, "arm": arm},
    )


@dataclass
class GenerationResult:
    content: str
    token_record: TokenRecord
    usage: dict[str, int]


class QwenPolicy:
    """Fresh base → merge frozen N once → train one LoRA delta."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        context: RunContext,
        config: GrpoConfig,
        guard: BudgetGuard,
        module_names: list[str],
        parent_merge: dict[str, Any],
        device: Any,
        mode: str = "train",
        evaluation_arm: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.context = context
        self.config = config
        self.guard = guard
        self.module_names = module_names
        self.parent_merge = parent_merge
        self.device = device
        self.mode = mode
        self.evaluation_arm = evaluation_arm
        self.adapter = LocalGRPOAdapter(
            epsilon=float(config.optimizer["epsilon"]), beta=float(config.optimizer["beta"])
        )
        self.model_ref_id = f"{config.model['model_id']}@{config.model['revision']}"
        self.closed = False
        self.amp_skip_replays_total = 0
        self.forward_backward_count = 0

    @classmethod
    def load(cls, context: RunContext, config: GrpoConfig, guard: BudgetGuard) -> QwenPolicy:
        torch = _torch()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("real_grpo_requires_one_visible_cuda_device")
        device_info = _require_v100_device(torch)
        _record_gpu_device(context, device_info, mode="train", arm=None)
        for name, value in {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
            if os.environ.get(name) != value:
                raise RuntimeError(f"determinism_environment_mismatch:{name}")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_everything(int(config.raw.get("run", {}).get("training_seed", 42)))
        guard.check("setup:model-load")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = config.model
        cache_dir = Path(str(model_cfg["cache_dir"]))
        if not cache_dir.is_absolute():
            cache_dir = context.repo_root / cache_dir
        parent = Path(str(model_cfg["parent_adapter"]))
        if not parent.is_absolute():
            parent = context.repo_root / parent
        expected_parent = str(model_cfg["parent_adapter_sha256"])
        expected_parent_config = str(model_cfg["parent_adapter_config_sha256"])
        parent_weight = parent / "adapter_model.safetensors"
        if not parent_weight.is_file() or sha256_file(parent_weight) != expected_parent:
            raise RuntimeError("parent_N_adapter_hash_mismatch")
        parent_config = parent / "adapter_config.json"
        if not parent_config.is_file() or sha256_file(parent_config) != expected_parent_config:
            raise RuntimeError("parent_N_adapter_config_hash_mismatch")
        guard.check("setup:parent-assets-verified")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_cfg["model_id"]),
            revision=str(model_cfg["revision"]),
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation=transformers_attention_implementation("sdpa_math"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg["model_id"]),
            revision=str(model_cfg["revision"]),
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        guard.check("setup:model-and-tokenizer-loaded")
        model.to("cuda:0")
        from enterprise_agent_rl.lora import (
            inject_lora_modules,
            merge_lora_adapter,
            trainable_lora_parameters,
        )

        parent_merge = merge_lora_adapter(
            model,
            parent,
            expected_model_id=str(model_cfg["model_id"]),
            expected_revision=str(model_cfg["revision"]),
        )
        guard.check("setup:parent-N-merged")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        module_names = inject_lora_modules(
            model,
            rank=int(config.lora["rank"]),
            alpha=float(config.lora["alpha"]),
            dropout=float(config.lora["dropout"]),
            target_modules=tuple(config.lora["target_modules"]),
        )
        guard.check("setup:trainable-delta-injected")
        trainable = trainable_lora_parameters(model)
        if len(module_names) != 196:
            raise RuntimeError(f"unexpected_qwen_lora_module_count:{len(module_names)}")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
        model.train()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config.optimizer["learning_rate"]),
            betas=tuple(config.optimizer["betas"]),
            eps=float(config.optimizer["eps"]),
            weight_decay=float(config.optimizer["weight_decay"]),
            foreach=False,
            fused=False,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        scaler = torch.cuda.amp.GradScaler(
            init_scale=float(config.optimizer["grad_scaler_init"]),
            backoff_factor=float(config.optimizer["grad_scaler_backoff"]),
            growth_interval=int(config.optimizer["grad_scaler_growth_interval"]),
            enabled=True,
        )
        guard.check("setup:optimizer-ready")
        return cls(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            context=context,
            config=config,
            guard=guard,
            module_names=module_names,
            parent_merge=parent_merge,
            device=torch.device("cuda:0"),
        )

    @classmethod
    def load_evaluation(
        cls,
        context: RunContext,
        config: GrpoConfig,
        guard: BudgetGuard,
        *,
        arm: str,
    ) -> QwenPolicy:
        """Load a fresh, inference-only Base, N, or N+delta evaluation arm."""

        if arm not in {"Base", "N", "RL"}:
            raise ValueError("evaluation_arm_must_be_Base_N_or_RL")
        torch = _torch()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("real_grpo_requires_one_visible_cuda_device")
        device_info = _require_v100_device(torch)
        _record_gpu_device(context, device_info, mode="evaluation", arm=arm)
        if os.environ.get("PYTHONHASHSEED") != "42" or os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ) != ":4096:8":
            raise RuntimeError("determinism_environment_mismatch")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_everything(int(config.evaluation.get("generation", {}).get("seed", 17)))
        model_cfg = config.model
        cache_dir = Path(str(model_cfg["cache_dir"]))
        if not cache_dir.is_absolute():
            cache_dir = context.repo_root / cache_dir
        parent: Path | None = None
        if arm in {"N", "RL"}:
            parent = Path(str(model_cfg["parent_adapter"]))
            if not parent.is_absolute():
                parent = context.repo_root / parent
            parent_weight = parent / "adapter_model.safetensors"
            parent_config = parent / "adapter_config.json"
            if not parent_weight.is_file() or sha256_file(parent_weight) != str(
                model_cfg["parent_adapter_sha256"]
            ):
                raise RuntimeError("parent_N_adapter_hash_mismatch")
            if not parent_config.is_file() or sha256_file(parent_config) != str(
                model_cfg["parent_adapter_config_sha256"]
            ):
                raise RuntimeError("parent_N_adapter_config_hash_mismatch")
        guard.check(f"evaluation:{arm}:model-load")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg["model_id"]),
            revision=str(model_cfg["revision"]),
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        template = tokenizer.chat_template
        if not isinstance(template, str):
            raise RuntimeError("tokenizer_chat_template_missing")
        raw_template_sha = sha256_bytes(template.encode("utf-8"))
        canonical_template_sha = sha256_bytes(
            json.dumps(
                template, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        expected_raw = str(config.evaluation.get("tokenizer_raw_template_sha256", ""))
        expected_canonical = str(
            config.evaluation.get("tokenizer_canonical_json_sha256", "")
        )
        if expected_raw and raw_template_sha != expected_raw:
            raise RuntimeError("tokenizer_chat_template_raw_hash_mismatch")
        if expected_canonical and canonical_template_sha != expected_canonical:
            raise RuntimeError("tokenizer_chat_template_canonical_hash_mismatch")
        atomic_write_json(
            context.run_dir / "setup" / f"tokenizer-preflight-{arm}.json",
            {
                "schema_version": "t024-tokenizer-preflight-v1",
                "model_id": model_cfg["model_id"],
                "revision": model_cfg["revision"],
                "tokenizer_class": type(tokenizer).__name__,
                "raw_utf8_bytes": len(template.encode("utf-8")),
                "raw_sha256": raw_template_sha,
                "canonical_json_utf8_bytes": len(
                    json.dumps(
                        template,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "canonical_json_sha256": canonical_template_sha,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "input_plus_max_tokens_contract": "checked_per_model_attempt",
            },
        )
        guard.check(f"evaluation:{arm}:tokenizer-preflight")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_cfg["model_id"]),
            revision=str(model_cfg["revision"]),
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation=transformers_attention_implementation("sdpa_math"),
        )
        model.to("cuda:0")
        from enterprise_agent_rl.lora import merge_lora_adapter

        parent_merge: dict[str, Any] | None = None
        if parent is not None:
            parent_merge = merge_lora_adapter(
                model,
                parent,
                expected_model_id=str(model_cfg["model_id"]),
                expected_revision=str(model_cfg["revision"]),
            )
        delta_merge: dict[str, Any] | None = None
        if arm == "RL":
            delta_value = config.evaluation.get("rl_delta_adapter") or model_cfg.get(
                "rl_delta_adapter"
            )
            if not isinstance(delta_value, str) or not delta_value:
                raise RuntimeError("fixed_rl_delta_path_missing_for_RL_evaluation")
            delta = Path(delta_value)
            if not delta.is_absolute():
                delta = context.repo_root / delta
            delta_weight = delta / "adapter_model.safetensors"
            delta_config = delta / "adapter_config.json"
            expected_delta = str(config.evaluation.get("rl_delta_sha256", ""))
            expected_delta_config = str(
                config.evaluation.get("rl_delta_config_sha256", "")
            )
            if not delta_weight.is_file() or not expected_delta or sha256_file(
                delta_weight
            ) != expected_delta:
                raise RuntimeError("fixed_rl_delta_adapter_hash_mismatch")
            if not delta_config.is_file() or not expected_delta_config or sha256_file(
                delta_config
            ) != expected_delta_config:
                raise RuntimeError("fixed_rl_delta_adapter_config_hash_mismatch")
            guard.check("evaluation:RL:delta-assets-verified")
            delta_merge = merge_lora_adapter(
                model,
                delta,
                expected_model_id=str(model_cfg["model_id"]),
                expected_revision=str(model_cfg["revision"]),
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.config.use_cache = True
        model.eval()
        policy = cls(
            model=model,
            tokenizer=tokenizer,
            optimizer=None,
            scheduler=None,
            scaler=None,
            context=context,
            config=config,
            guard=guard,
            module_names=[],
            parent_merge={
                "parent": parent_merge,
                "delta": delta_merge,
                "parent_merge_count": int(parent_merge is not None),
                "delta_merge_count": int(delta_merge is not None),
                "rl_delta_semantic_fingerprint": config.evaluation.get(
                    "rl_delta_semantic_fingerprint"
                ),
            },
            device=torch.device("cuda:0"),
            mode="evaluation",
            evaluation_arm=arm,
        )
        policy.model_ref_id = f"{config.model['model_id']}@{config.model['revision']}@{arm}"
        return policy

    def _adapter_state(self) -> dict[str, Any]:
        from enterprise_agent_rl.lora import adapter_state_dict

        return {
            key: value.detach().cpu().clone()
            for key, value in adapter_state_dict(self.model).items()
        }

    def policy_fingerprint(self) -> str:
        return _digest_tensor_state(self._adapter_state())

    def policy_snapshot(self, *, round_index: int, successful_updates: int) -> dict[str, Any]:
        fingerprint = self.policy_fingerprint()
        return {
            "schema_version": "grpo-policy-snapshot-v1",
            "round_index": round_index,
            "successful_updates": successful_updates,
            "policy_snapshot_id": f"grpo-step{successful_updates}-delta-{fingerprint[:16]}",
            "base_model": self.config.model["model_id"],
            "base_revision": self.config.model["revision"],
            "parent_N_adapter_sha256": self.config.model["parent_adapter_sha256"],
            "delta_fingerprint": fingerprint,
            "model_ref_id": self.model_ref_id,
            "optimizer_or_scheduler_changed_by_sampling": False,
        }

    def evaluation_identity(self, arm: str) -> dict[str, Any]:
        return {
            "evaluation_arm": arm,
            "model_ref_id": self.model_ref_id,
            "model_object_id": id(self.model),
            "model_weight_fingerprint": _digest_tensor_state(
                {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
            ),
            "parent_merge_count": int(self.parent_merge.get("parent_merge_count", 1)),
            "delta_merge_count": int(self.parent_merge.get("delta_merge_count", 0)),
            "rl_delta_semantic_fingerprint": self.config.evaluation.get(
                "rl_delta_semantic_fingerprint"
            ),
            "fresh_base": True,
        }

    def state_fingerprints(self) -> dict[str, Any]:
        return {
            "optimizer": _digest_tensor_state(self.optimizer.state_dict())
            if self.optimizer is not None
            else None,
            "scheduler": _digest_tensor_state(self.scheduler.state_dict())
            if self.scheduler is not None
            else None,
            "scaler": _digest_tensor_state(self.scaler.state_dict())
            if self.scaler is not None
            else None,
            "rng": recursive_fingerprint(capture_rng_state()),
        }

    def _inputs(self, record: dict[str, Any]) -> dict[str, Any]:
        torch = _torch()
        prompt = torch.tensor([record["prompt_token_ids"]], dtype=torch.long, device=self.device)
        completion = torch.tensor(
            [record["generated_token_ids"]], dtype=torch.long, device=self.device
        )
        if completion.size(1) == 0:
            raise ValueError(f"empty_completion:{record.get('episode_id')}")
        return {
            "prompt_ids": prompt,
            "prompt_mask": torch.ones_like(prompt),
            "completion_ids": completion,
            "completion_mask": torch.ones_like(completion, dtype=torch.float32),
            "old_per_token_logps": torch.tensor(
                [record["old_per_token_logps"]], dtype=torch.float32, device=self.device
            ),
            "advantages": torch.tensor(
                [float(record["advantage"])], dtype=torch.float32, device=self.device
            ),
        }

    def update(
        self, records: list[dict[str, Any]], *, logical_step: int, total_tokens: int
    ) -> dict[str, Any]:
        torch = _torch()
        active = [row for row in records if float(row.get("advantage", 0.0)) != 0.0]
        if not active:
            return {
                "status": "SKIPPED_ZERO_ADVANTAGE",
                "active_records": 0,
                "logical_step": logical_step,
            }
        if total_tokens <= 0:
            raise ValueError("total_tokens_must_be_positive")
        self.model.train()
        before = self._adapter_state()
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_scheduler = copy.deepcopy(self.scheduler.state_dict())
        replay_count = 0
        before_rng = capture_rng_state()
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            self.adapter._metrics["train"].clear()
            for row in active:
                self.guard.record_call("forward_backward")
                self.forward_backward_count += 1
                self.guard.check(
                    "update:{}:microbatch:{}".format(
                        logical_step,
                        row.get("physical_request_index", row.get("request_index")),
                    ),
                    allow_soft=True,
                )
                inputs = self._inputs(row)
                with attention_backend_context(torch, "sdpa_math"):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.device.type == "cuda",
                    ):
                        loss = self.adapter.compute_loss(self.model, inputs)
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("nonfinite_grpo_loss")
                    weighted = loss * (float(len(row["generated_token_ids"])) / float(total_tokens))
                    self.scaler.scale(weighted).backward()
                losses.append(float(loss.detach().float().cpu().item()))
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(), float(self.config.optimizer["clip_norm"])
                ).item()
            )
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            after = self._adapter_state()
            changed = _digest_tensor_state(after) != _digest_tensor_state(before)
            scale_after = float(self.scaler.get_scale())
            if not changed:
                if (
                    math.isfinite(grad_norm) is False
                    and scale_after >= scale_before
                ) or scale_after >= scale_before:
                    raise FloatingPointError("nonfinite_gradient_or_amp_skip_without_backoff")
                if self.amp_skip_replays_total >= int(
                    self.config.optimizer["max_amp_skip_replays"]
                ):
                    raise FloatingPointError("amp_skip_replay_limit_exceeded")
                reduced_scaler = copy.deepcopy(self.scaler.state_dict())
                self._load_adapter_state(before)
                self.optimizer.load_state_dict(before_optimizer)
                self.scheduler.load_state_dict(before_scheduler)
                restore_rng_state(before_rng)
                self.scaler.load_state_dict(reduced_scaler)
                replay_count += 1
                self.amp_skip_replays_total += 1
                self.guard.record_call("amp_skip_replays")
                continue
            if not changed:
                raise RuntimeError("successful_update_without_delta_change")
            self.scheduler.step()
            return {
                "status": "UPDATED",
                "logical_step": logical_step,
                "active_records": len(active),
                "zero_advantage_skips": len(records) - len(active),
                "loss_mean": sum(losses) / len(losses),
                "gradient_norm": grad_norm,
                "amp_skip_replays": replay_count,
                "amp_skip_replays_total": self.amp_skip_replays_total,
                "forward_backward_count": self.forward_backward_count,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "policy_fingerprint": self.policy_fingerprint(),
                "optimizer_state_count": len(self.optimizer.state_dict().get("state", {})),
                "scheduler_last_epoch": self.scheduler.state_dict().get("last_epoch"),
            }

    def _trainable_parameters(self) -> list[Any]:
        from enterprise_agent_rl.lora import trainable_lora_parameters

        return trainable_lora_parameters(self.model)

    def _load_adapter_state(self, state: dict[str, Any]) -> None:
        from enterprise_agent_rl.lora import iter_lora_modules

        for name, module in iter_lora_modules(self.model):
            for suffix, target in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
                key = f"base_model.model.{name}.{suffix}.weight"
                if key not in state:
                    raise RuntimeError(f"adapter_state_missing:{key}")
                with _torch().no_grad():
                    target.copy_(state[key].to(device=target.device, dtype=target.dtype))

    def save_checkpoint(
        self, run_dir: Path, *, logical_step: int, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        torch = _torch()
        from enterprise_agent_rl.lora import save_adapter

        path = run_dir / "training" / "checkpoints" / f"step-{logical_step}"
        path.mkdir(parents=True, exist_ok=True)
        adapter = save_adapter(
            self.model,
            path / "delta",
            model_id=str(self.config.model["model_id"]),
            revision=str(self.config.model["revision"]),
            rank=int(self.config.lora["rank"]),
            alpha=float(self.config.lora["alpha"]),
            dropout=float(self.config.lora["dropout"]),
            target_modules=tuple(self.config.lora["target_modules"]),
            module_names=self.module_names,
        )
        torch.save(
            {
                "delta_state": self._adapter_state(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "rng": capture_rng_state(),
                "amp_skip_replays_total": self.amp_skip_replays_total,
                "forward_backward_count": self.forward_backward_count,
                "metadata": metadata,
            },
            path / "state.pt",
        )
        atomic_write_json(
            path / "checkpoint.json",
            {
                "schema_version": "grpo-checkpoint-v1",
                "logical_step": logical_step,
                "adapter": adapter,
                "policy_fingerprint": self.policy_fingerprint(),
                "state_fingerprints": self.state_fingerprints(),
            },
        )
        return {
            "path": str(path.relative_to(run_dir)),
            "policy_fingerprint": self.policy_fingerprint(),
        }

    def load_run_state(self, run_dir: Path, state: dict[str, Any]) -> None:
        torch = _torch()
        step = int(state["successful_updates"])
        path = run_dir / "training" / "checkpoints" / f"step-{step}" / "state.pt"
        if not path.is_file():
            raise RuntimeError(f"resume_checkpoint_missing:{step}")
        value = torch.load(path, map_location="cpu")
        self._load_adapter_state(value["delta_state"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.scheduler.load_state_dict(value["scheduler"])
        self.scaler.load_state_dict(value["scaler"])
        metadata = value.get("metadata") or {}
        expected_input = state.get("input_fingerprint")
        if expected_input is not None and metadata.get("input_fingerprint") not in {
            None,
            expected_input,
        }:
            raise RuntimeError("resume_checkpoint_input_fingerprint_mismatch")
        if value.get("rng") is not None:
            restore_rng_state(value["rng"])
        self.amp_skip_replays_total = int(value.get("amp_skip_replays_total", 0))
        self.forward_backward_count = int(value.get("forward_backward_count", 0))
        self.model.train()

    def freeze_final(self, run_dir: Path, *, snapshot: dict[str, Any]) -> dict[str, Any]:
        from enterprise_agent_rl.lora import save_adapter

        target = run_dir / "training" / "final-delta"
        adapter = save_adapter(
            self.model,
            target,
            model_id=str(self.config.model["model_id"]),
            revision=str(self.config.model["revision"]),
            rank=int(self.config.lora["rank"]),
            alpha=float(self.config.lora["alpha"]),
            dropout=float(self.config.lora["dropout"]),
            target_modules=tuple(self.config.lora["target_modules"]),
            module_names=self.module_names,
        )
        parent_stack = {
            "schema_version": "grpo-parent-stack-v1",
            "load_order": [
                "fresh Qwen base",
                "merge frozen parent N once",
                "merge final delta once for RL",
            ],
            "fresh_base": {
                "model_id": self.config.model["model_id"],
                "revision": self.config.model["revision"],
                "dtype": "float16",
            },
            "parent_N": {
                "path": str(Path(self.config.model["parent_adapter"])),
                "adapter_model_sha256": self.config.model["parent_adapter_sha256"],
                "adapter_config_sha256": self.config.model["parent_adapter_config_sha256"],
                "merge_count": 1,
            },
            "delta": {
                "path": str(target.relative_to(run_dir)),
                "adapter_model_sha256": sha256_file(target / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(target / "adapter_config.json"),
                "depends_on_parent_N": True,
                "standalone_fresh_base_load_allowed": False,
                "merge_count_for_RL": 1,
            },
            "final_snapshot": snapshot,
        }
        atomic_write_json(run_dir / "training" / "parent-stack.json", parent_stack)
        return {
            "status": "FINAL_DELTA_FROZEN",
            "snapshot": snapshot,
            "adapter": adapter,
            "parent_stack_path": "training/parent-stack.json",
            "merge_order": parent_stack["load_order"],
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            del self.model
            del self.optimizer
            del self.scheduler
            del self.scaler
        finally:
            gc.collect()
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class HfRolloutEngine:
    """Direct model-facing engine; Runtime and Reward remain public interfaces."""

    def __init__(
        self,
        *,
        policy: QwenPolicy,
        run_dir: Path,
        config: GrpoConfig,
        evaluation: bool = False,
    ):
        self.policy = policy
        self.run_dir = run_dir
        self.config = config
        self.evaluation = evaluation
        self.request_counter = 0
        self.stage = "evaluation" if evaluation else "training"
        self.arm = str(getattr(policy, "evaluation_arm", None) or "train")
        self.evidence = EvidenceStore(run_dir, stage=self.stage, arm=self.arm)

    def _settings(self) -> Settings:
        return Settings.model_validate(self.config.runtime)

    def _generation_settings(self, seed: int) -> tuple[GenerationSettings, dict[str, Any]]:
        values = (
            self.config.evaluation.get("generation", {})
            if self.evaluation
            else self.config.rollout.get("generation", {})
        )
        supported = {
            key: values[key]
            for key in ("temperature", "top_p", "max_tokens", "format_retries", "parse_mode")
            if key in values
        }
        supported["seed"] = seed
        generation = GenerationSettings.model_validate(supported)
        return generation, {
            "temperature": float(values.get("temperature", generation.temperature)),
            "top_p": float(values.get("top_p", generation.top_p)),
            "top_k": int(values.get("top_k", 0)),
            "max_tokens": int(values.get("max_tokens", generation.max_tokens)),
            "format_retries": int(values.get("format_retries", generation.format_retries)),
            "parse_mode": str(values.get("parse_mode", generation.parse_mode)),
            "seed": seed,
            "mode": "greedy" if self.evaluation else "sampling",
        }

    def rollout(
        self,
        task: TaskInput,
        *,
        round_index: int,
        sample_index: int,
        seed: int,
        policy_snapshot: dict[str, Any],
        episode_id: str | None = None,
    ) -> EpisodeSample:
        if not task.database_path:
            raise ValueError(f"database_path_required:{task.task_id}")
        episode_id = episode_id or f"r{round_index}-{task.group_id}-s{sample_index}"
        public = PublicTask.model_validate(task.public_task)
        generation, _ = self._generation_settings(seed)
        direct = _DirectPolicy(
            self, task, episode_id, sample_index, seed, policy_snapshot, generation
        )
        trajectory = run_episode(
            public,
            direct,
            Path(task.database_path),
            self._settings(),
            self.run_dir / "runtime" / "sql-audit.jsonl",
        )
        # Accrue the completed episode without discarding its known trajectory;
        # the next worker/episode guard decides whether new work may start.
        self.policy.guard.current(f"rollout:after-episode:{episode_id}")
        return EpisodeSample(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            seed=seed,
            policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
            model_ref_id=self.policy.model_ref_id,
            token_records=direct.records,
            metadata={
                "trajectory": trajectory.model_dump(mode="json"),
                "terminal_reason": trajectory.terminal_reason,
            },
            failed_action=direct.failed_action,
        )

    def evaluate(
        self, task: TaskInput, *, policy_snapshot: dict[str, Any], episode_id: str, seed: int
    ) -> EpisodeSample:
        return self.rollout(
            task,
            round_index=-1,
            sample_index=0,
            seed=seed,
            policy_snapshot=policy_snapshot,
            episode_id=episode_id,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        task: TaskInput,
        episode_id: str,
        sample_index: int,
        seed: int,
        policy_snapshot: dict[str, Any],
        request_index: int,
    ) -> GenerationResult:
        torch = _torch()
        self.request_counter += 1
        logical_request_index = self.request_counter
        self.policy.guard.record_call("model_requests")
        model_id = str(self.config.model["model_id"])
        generation, hf_generation = self._generation_settings(seed)
        max_tokens = int(hf_generation["max_tokens"])
        generation_parameters = {
            "configured": dict(hf_generation),
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "do_sample": not self.evaluation,
            "num_beams": 1,
            "num_return_sequences": 1,
            "repetition_penalty": 1.0,
            "pad_token_id": self.policy.tokenizer.pad_token_id
            or self.policy.tokenizer.eos_token_id,
            "eos_token_id": self.policy.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_scores": True,
        }
        if self.evaluation:
            # Greedy decoding does not consume sampling controls.  Supplying
            # neutral values explicitly prevents the model's generation
            # config (for example temperature=.7/top_p=.8/top_k=20) from
            # leaking into the effective HF kwargs or its warnings.
            generation_parameters.update(
                {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
            )
        else:
            generation_parameters.update(
                {
                    "temperature": generation.temperature,
                    "top_p": generation.top_p,
                    "top_k": hf_generation["top_k"],
                }
            )
        model_identity = {
            "model_id": model_id,
            "revision": str(self.config.model["revision"]),
            "model_ref_id": self.policy.model_ref_id,
            "model_object_id": id(self.policy.model),
            "policy_snapshot_id": policy_snapshot.get("policy_snapshot_id"),
            "model_weight_fingerprint": policy_snapshot.get("model_weight_fingerprint"),
            "delta_fingerprint": policy_snapshot.get("delta_fingerprint")
            or (
                self.config.evaluation.get("rl_delta_semantic_fingerprint")
                if policy_snapshot.get("evaluation_arm") == "RL"
                else None
            ),
            "protocol_hash": policy_snapshot.get("protocol_hash"),
        }
        # The intent is durable before tokenization/model execution.  A fresh
        # engine can therefore restart its local request counter at 1 without
        # ever reusing the prior request/response/event paths.
        handle = self.evidence.start_attempt(
            episode_id=episode_id,
            task_id=task.task_id,
            sample_index=sample_index,
            messages=messages,
            generation_settings=generation_parameters,
            model_identity=model_identity,
        )
        prompt_ids: list[int] = []
        model_execution_started = False
        started = time.monotonic()
        try:
            text = self.policy.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.policy.tokenizer(text, return_tensors="pt")
            inputs = {
                key: value.to(self.policy.device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            prompt_ids = [int(value) for value in inputs["input_ids"][0].detach().cpu().tolist()]
            if len(prompt_ids) + max_tokens > int(self.config.model["max_context_tokens"]):
                record = self._error_record(
                    task,
                    episode_id,
                    sample_index,
                    seed,
                    policy_snapshot,
                    logical_request_index,
                    "context_length_exceeded",
                    prompt_ids,
                    handle=handle,
                )
                self.evidence.persist_response(
                    handle,
                    record,
                    model_id=model_id,
                    model_identity=model_identity,
                    generation_settings=generation_parameters,
                    content="",
                    error=record.error,
                    model_execution_started=False,
                    model_execution_completed=False,
                    completion_status="not_started",
                )
                return GenerationResult("", record, {})
            self.policy.guard.check(
                f"rollout:before-generation:{handle.attempt_index}", allow_soft=False
            )
            model_execution_started = True
            self.policy.model.eval()
            with attention_backend_context(torch, "sdpa_math"):
                with torch.inference_mode():
                    rng_context = (
                        torch.random.fork_rng(devices=[self.policy.device.index or 0])
                        if self.policy.device.type == "cuda"
                        else contextlib.nullcontext()
                    )
                    with rng_context:
                        torch.manual_seed(seed + request_index)
                        generation_kwargs = dict(generation_parameters)
                        generation_kwargs.pop("configured", None)
                        generated = self.policy.guard.run_callable(
                            lambda: self.policy.model.generate(**inputs, **generation_kwargs),
                            phase=f"generate:{handle.attempt_index}",
                        )
            self.policy.guard.current(f"rollout:after-generation:{handle.attempt_index}")
            sequences = generated.sequences
            full_ids = [int(value) for value in sequences[0].detach().cpu().tolist()]
            raw_logits = tuple(getattr(generated, "logits", ()) or ())
            scores = tuple(getattr(generated, "scores", ()) or ())
            if len(raw_logits) != len(scores):
                raise RuntimeError("raw_behavior_step_count_mismatch")
            generated_ids = full_ids[len(prompt_ids) : len(prompt_ids) + len(raw_logits)]
            raw_log_probs: list[float] = []
            behavior_log_probs: list[float] = []
            for index, token_id in enumerate(generated_ids):
                raw_vector = raw_logits[index][0]
                behavior_vector = scores[index][0]
                if not bool(torch.isfinite(raw_vector).all().item()):
                    raise FloatingPointError("nonfinite_generation_logits")
                raw_log_probs.append(
                    float(torch.log_softmax(raw_vector.float(), dim=-1)[token_id].cpu().item())
                )
                behavior_log_probs.append(
                    float(torch.log_softmax(behavior_vector.float(), dim=-1)[token_id].cpu().item())
                )
            record = TokenRecord(
                episode_id=episode_id,
                task_id=task.task_id,
                group_id=task.group_id,
                sample_index=sample_index,
                request_index=logical_request_index,
                seed=seed,
                policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
                model_ref_id=self.policy.model_ref_id,
                request_id=handle.request_id,
                response_id=handle.response_id,
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated_ids,
                response_mask=[0] * len(prompt_ids) + [1] * len(generated_ids),
                raw_policy_log_probs=raw_log_probs,
                behavior_sampling_log_probs=behavior_log_probs,
                action_text=self.policy.tokenizer.decode(generated_ids, skip_special_tokens=True),
                action_status="accepted",
                metadata={
                    "physical_request_index": handle.attempt_index,
                    "elapsed_s": time.monotonic() - started,
                    "generation_settings": hf_generation,
                },
            )
            record.validate()
            self.evidence.persist_response(
                handle,
                record,
                model_id=model_id,
                model_identity=model_identity,
                generation_settings=generation_parameters,
                content=record.action_text,
                error=record.error,
                model_execution_started=True,
                model_execution_completed=True,
                completion_status="completed",
            )
            return GenerationResult(
                record.action_text or "",
                record,
                {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated_ids)},
            )
        except BudgetError:
            self.evidence.persist_incomplete(
                handle,
                error={"type": "BudgetError", "message": "budget_stop"},
                model_execution_started=model_execution_started,
            )
            raise
        except EvidencePersistenceError:
            # A missing response/event/journal is a technical evidence stop;
            # never turn it into an ordinary model-error response.
            raise
        except Exception as exc:
            record = self._error_record(
                task,
                episode_id,
                sample_index,
                seed,
                policy_snapshot,
                logical_request_index,
                str(exc),
                prompt_ids,
                handle=handle,
            )
            self.evidence.persist_response(
                handle,
                record,
                model_id=model_id,
                model_identity=model_identity,
                generation_settings=generation_parameters,
                content=record.action_text,
                error=record.error,
                model_execution_started=model_execution_started,
                model_execution_completed=model_execution_started,
                completion_status="completed_with_error",
            )
            return GenerationResult("", record, {})

    def _error_record(
        self,
        task: TaskInput,
        episode_id: str,
        sample_index: int,
        seed: int,
        snapshot: dict[str, Any],
        request_index: int,
        message: str,
        prompt_ids: list[int],
        *,
        handle: AttemptHandle,
    ) -> TokenRecord:
        return TokenRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            request_index=request_index,
            seed=seed,
            policy_snapshot_id=str(snapshot["policy_snapshot_id"]),
            model_ref_id=self.policy.model_ref_id,
            request_id=handle.request_id,
            response_id=handle.response_id,
            prompt_token_ids=prompt_ids,
            generated_token_ids=[],
            response_mask=[0] * len(prompt_ids),
            raw_policy_log_probs=[],
            behavior_sampling_log_probs=[],
            action_text="",
            action_status="model_error",
            error={"type": "generation_error", "message": message[:1000]},
            metadata={"physical_request_index": handle.attempt_index},
        )


class _DirectPolicy:
    # Trajectory's public policy-kind contract uses the existing local-LLM
    # category; the GRPO adapter is an implementation detail, not a new enum.
    kind = "local_llm"

    def __init__(
        self,
        engine: HfRolloutEngine,
        task: TaskInput,
        episode_id: str,
        sample_index: int,
        seed: int,
        snapshot: dict[str, Any],
        generation: GenerationSettings,
    ):
        self.engine = engine
        self.task = task
        self.episode_id = episode_id
        self.sample_index = sample_index
        self.seed = seed
        self.snapshot = snapshot
        self.generation = generation
        self.version = str(snapshot["policy_snapshot_id"])
        self.records: list[TokenRecord] = []
        self.failed_action: dict[str, Any] | None = None

    def decide(
        self, task: PublicTask, observation: Any, history: list[Any], timeout_s: float
    ) -> Decision:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
        for step in history:
            raw = (
                step.model_attempts[-1]["raw_output"]
                if step.model_attempts
                else step.action.model_dump_json()
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": step.tool_result.model_dump_json()})
        if not history:
            messages.append({"role": "user", "content": observation.model_dump_json()})
        attempts: list[dict[str, Any]] = []
        for retry in range(self.generation.format_retries + 1):
            result = self.engine.generate(
                messages=messages,
                task=self.task,
                episode_id=self.episode_id,
                sample_index=self.sample_index,
                seed=self.seed + retry,
                policy_snapshot=self.snapshot,
                request_index=len(self.records),
            )
            self.records.append(result.token_record)
            attempt = {
                "attempt_index": retry,
                "raw_output": result.content,
                "format_valid": False,
                "strict_format_valid": False,
                "usage": result.usage,
                "error": None,
            }
            evidence_metadata = result.token_record.metadata
            attempt.update(
                {
                    "request_id": result.token_record.request_id,
                    "response_id": result.token_record.response_id,
                    "attempt_id": evidence_metadata.get("attempt_id"),
                    "evidence_attempt_index": evidence_metadata.get(
                        "attempt_index", result.token_record.request_index
                    ),
                    "request_path": evidence_metadata.get("request_path"),
                    "response_path": evidence_metadata.get("response_path"),
                    "engine_event_path": evidence_metadata.get("engine_event_path"),
                    "content_sha256": evidence_metadata.get("content_sha256"),
                }
            )
            try:
                action, normalized = parse_action(result.content, self.generation.parse_mode)
                attempt.update(
                    format_valid=True,
                    strict_format_valid=not normalized,
                    normalized_json_fence=normalized,
                    parsed_action=action.model_dump(),
                )
            except ValueError as exc:
                attempt["error"] = str(exc)
            attempts.append(attempt)
            if attempt["format_valid"]:
                return Decision(
                    action=action,
                    model_attempts=attempts,
                    prompt_tokens=sum(item["usage"].get("prompt_tokens", 0) for item in attempts),
                    completion_tokens=sum(
                        item["usage"].get("completion_tokens", 0) for item in attempts
                    ),
                    token_usage_known=True,
                )
            messages.extend(
                [
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": (
                            f"Format error: {attempt['error']}. Return one valid JSON Action."
                        ),
                    },
                ]
            )
        self.failed_action = {"status": "format_failed", "attempts": attempts}
        raise ModelFailure(str(attempts[-1].get("error") or "format_failed"), attempts)


class RealVerifierWorker:
    """Runtime/Verifier adapter; reference fields stay on the verifier side."""

    def __init__(
        self,
        verifier: Callable[[TaskInput, Trajectory], Any] | None = None,
        guard: BudgetGuard | None = None,
    ):
        self.verifier = verifier
        self.guard = guard

    def score(self, task: TaskInput, episode: EpisodeSample) -> ScoreResult:
        trajectory_value = episode.metadata.get("trajectory")
        if not isinstance(trajectory_value, dict):
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="unknown",
                infra_error="trajectory_missing",
                source_kind="real",
            )
        trajectory = Trajectory.model_validate(trajectory_value)
        sql_actions = [step for step in trajectory.steps if step.tool_name == "sql"]
        sql_ok = sum(bool(step.tool_result.ok) for step in sql_actions)
        non_sql_actions = [
            step
            for step in trajectory.steps
            if step.tool_name in {"schema", "dictionary", "check"}
        ]
        def budget_errors(step):
            return "budget" in str(step.tool_result.error or "").lower()
        runtime_worker_sql = sum(not budget_errors(step) for step in sql_actions)
        runtime_worker_non_sql = sum(not budget_errors(step) for step in non_sql_actions)
        worker_count = runtime_worker_sql + runtime_worker_non_sql
        if self.guard is not None:
            self.guard.record_call("accepted_actions", len(trajectory.steps))
            self.guard.record_call(
                "tool_actions", len(sql_actions) + len(non_sql_actions)
            )
            self.guard.record_call("runtime_workers", worker_count)
        budget_rejections = 0
        error_values: list[Any] = []
        for step in sql_actions:
            result = step.tool_result
            error_text = str(result.error or "")
            if "budget" in error_text.lower():
                budget_rejections += 1
            if not result.ok:
                error_values.append(result.error)
        if self.verifier is None:
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="unknown",
                sql_action_count=len(sql_actions),
                sql_ok_count=sql_ok,
                worker_count=worker_count,
                runtime_worker_sql_count=runtime_worker_sql,
                runtime_worker_non_sql_count=runtime_worker_non_sql,
                verifier_sql_count=None,
                budget_rejection_count=budget_rejections,
                infra_error="verifier_not_injected",
                error=error_values,
                source_kind="real",
            )
        if self.guard is not None:
            self.guard.record_call("verifier_calls")
            self.guard.check(f"verifier:before:{episode.episode_id}")
        verifier_sql_count = 1 + int(
            isinstance(task.verifier_payload, dict)
            and task.verifier_payload.get("required_probe_sql") is not None
        )
        if self.guard is not None:
            self.guard.record_call("verifier_sql", verifier_sql_count)
        try:
            if self.guard is not None:
                reward = self.guard.run_callable(
                    lambda: self.verifier(task, trajectory),
                    phase=f"verifier:{episode.episode_id}",
                )
            else:
                reward = self.verifier(task, trajectory)
        except BudgetError:
            raise
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="unknown",
                sql_action_count=len(sql_actions),
                sql_ok_count=sql_ok,
                worker_count=worker_count,
                runtime_worker_sql_count=runtime_worker_sql,
                runtime_worker_non_sql_count=runtime_worker_non_sql,
                verifier_sql_count=verifier_sql_count,
                budget_rejection_count=budget_rejections,
                infra_error=error,
                error=error,
                source_kind="real",
            )
        if self.guard is not None:
            self.guard.current(f"verifier:after:{episode.episode_id}")
        if hasattr(reward, "model_dump"):
            reward = reward.model_dump()
        if not isinstance(reward, dict):
            raise TypeError("verifier_must_return_reward_mapping")
        return ScoreResult(
            episode_id=episode.episode_id,
            task_id=task.task_id,
            status="scored",
            reward_value=float(reward["value"]),
            success=bool(reward["success"]),
            reason=str(reward.get("reason", "unknown")),
            sql_action_count=len(sql_actions),
            sql_ok_count=sql_ok,
            worker_count=worker_count,
            runtime_worker_sql_count=runtime_worker_sql,
            runtime_worker_non_sql_count=runtime_worker_non_sql,
            verifier_sql_count=verifier_sql_count,
            budget_rejection_count=budget_rejections,
            error=error_values,
            source_kind="real",
            metadata={"reward": reward},
        )
