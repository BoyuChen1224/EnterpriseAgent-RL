"""CPU-only injected components used by the formal integration tests/CLI demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .budget import BudgetGuard
from .contracts import EpisodeSample, ScoreResult, TaskInput, TokenRecord
from .evidence import EvidenceStore
from .io import atomic_write_json


def tiny_qwen_policy(run_dir: Path, guard: BudgetGuard, config: Any):
    """Build a CPU model that exercises the production QwenPolicy.update path."""

    import torch

    from enterprise_agent_rl.lora import inject_lora_modules, trainable_lora_parameters

    from .torch_backend import QwenPolicy

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 16)
            self.block = torch.nn.Module()
            self.block.proj = torch.nn.Linear(16, 512)

        def forward(self, input_ids, attention_mask=None, use_cache=False, **_kwargs):
            return SimpleNamespace(logits=self.block.proj(self.embedding(input_ids)))

    model = TinyCausalLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module_names = inject_lora_modules(
        model, rank=2, alpha=2.0, dropout=0.0, target_modules=("proj",)
    )
    trainable = trainable_lora_parameters(model)
    optimizer = torch.optim.AdamW(trainable, lr=2e-6, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return QwenPolicy(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        context=SimpleNamespace(run_dir=run_dir),
        config=config,
        guard=guard,
        module_names=module_names,
        parent_merge={"parent_merge_count": 0, "delta_merge_count": 0},
        device=torch.device("cpu"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class TinyPolicy:
    """A tiny torch-backed adapter with the same update/checkpoint boundary."""

    def __init__(self, run_dir: Path, guard: BudgetGuard | None = None):
        import torch

        self.torch = torch
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.optimizer = torch.optim.AdamW([self.weight], lr=0.05, foreach=False, fused=False)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _step: 1.0
        )
        self.run_dir = run_dir
        self.guard = guard
        # A resumed run must identify the same synthetic base consistently;
        # object identity would make a checkpoint restore look like a new model.
        self.model_ref_id = "tiny-model-fixture"
        self.closed = False

    def policy_fingerprint(self) -> str:
        return hashlib.sha256(self.weight.detach().cpu().numpy().tobytes()).hexdigest()

    def policy_snapshot(self, *, round_index: int, successful_updates: int) -> dict[str, Any]:
        digest = self.policy_fingerprint()
        return {
            "schema_version": "grpo-policy-snapshot-v1",
            "round_index": round_index,
            "successful_updates": successful_updates,
            "policy_snapshot_id": f"tiny-step{successful_updates}-delta-{digest[:16]}",
            "delta_fingerprint": digest,
            "model_ref_id": self.model_ref_id,
            "base_model": "tiny-test-model",
            "base_revision": "fixture",
        }

    def state_fingerprints(self) -> dict[str, str]:
        return {
            "optimizer": _hash(self.optimizer.state_dict()),
            "scheduler": _hash(self.scheduler.state_dict()),
            "scaler": "disabled-on-cpu",
        }

    def update(
        self, records: list[dict[str, Any]], *, logical_step: int, total_tokens: int
    ) -> dict[str, Any]:
        active = [row for row in records if float(row.get("advantage", 0.0)) != 0.0]
        if not active:
            return {
                "status": "SKIPPED_ZERO_ADVANTAGE",
                "active_records": 0,
                "logical_step": logical_step,
            }
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.weight * 0.0
        for row in active:
            if self.guard is not None:
                self.guard.check(f"tiny-update:{logical_step}:{row['episode_id']}", allow_soft=True)
            loss = loss - self.weight * float(row["advantage"]) * (
                len(row["generated_token_ids"]) / max(1, total_tokens)
            )
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_([self.weight], 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return {
            "status": "UPDATED",
            "logical_step": logical_step,
            "active_records": len(active),
            "zero_advantage_skips": len(records) - len(active),
            "loss": float(loss.detach()),
            "weight": float(self.weight.detach()),
            "policy_fingerprint": self.policy_fingerprint(),
        }

    def save_checkpoint(
        self, run_dir: Path, *, logical_step: int, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        path = run_dir / "training" / "checkpoints" / f"step-{logical_step}"
        path.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "weight": self.weight.detach().cpu(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "metadata": metadata,
            },
            path / "state.pt",
        )
        atomic_write_json(
            path / "checkpoint.json",
            {
                "logical_step": logical_step,
                "policy_fingerprint": self.policy_fingerprint(),
                "state_fingerprints": self.state_fingerprints(),
            },
        )
        return {
            "path": str(path.relative_to(run_dir)),
            "policy_fingerprint": self.policy_fingerprint(),
        }

    def load_run_state(self, run_dir: Path, state: dict[str, Any]) -> None:
        value = self.torch.load(
            run_dir
            / "training"
            / "checkpoints"
            / f"step-{int(state['successful_updates'])}"
            / "state.pt",
            map_location="cpu",
        )
        with self.torch.no_grad():
            self.weight.copy_(value["weight"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.scheduler.load_state_dict(value["scheduler"])

    def freeze_final(self, run_dir: Path, *, snapshot: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(
            run_dir / "training" / "final-delta.json",
            {"weight": float(self.weight.detach()), "snapshot": snapshot},
        )
        return {
            "status": "FINAL_DELTA_FROZEN",
            "snapshot": snapshot,
            "weight": float(self.weight.detach()),
        }

    def close(self) -> None:
        self.closed = True


class FakeRolloutEngine:
    def __init__(self, *, model_ref_id: str = "fake-model-object"):
        self.model_ref_id = model_ref_id
        self.calls: list[dict[str, Any]] = []

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
        episode_id = episode_id or f"r{round_index}-{task.group_id}-s{sample_index}"
        self.calls.append(
            {
                "round_index": round_index,
                "task_id": task.task_id,
                "sample_index": sample_index,
                "policy_snapshot_id": policy_snapshot["policy_snapshot_id"],
            }
        )
        prompt = [11, 12, 13, 14]
        generated = [100 + round_index, 200 + sample_index, 300 + len(self.calls)]
        record = TokenRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            request_index=len(self.calls),
            seed=seed,
            policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
            model_ref_id=self.model_ref_id,
            request_id=f"request-{len(self.calls)}",
            response_id=f"response-{len(self.calls)}",
            prompt_token_ids=prompt,
            generated_token_ids=generated,
            response_mask=[0] * len(prompt) + [1] * len(generated),
            raw_policy_log_probs=[-0.1, -0.2, -0.3],
            behavior_sampling_log_probs=[-0.1, -0.2, -0.3],
            action_text='{"kind":"sql","arguments":{"query":"SELECT 1"}}',
            action_status="format_failed" if sample_index == 3 else "accepted",
            error="synthetic-format-failure" if sample_index == 3 else None,
            metadata={
                "request_path": f"requests/{len(self.calls):04d}.json",
                "response_path": f"responses/{len(self.calls):04d}.json",
            },
        )
        return EpisodeSample(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            seed=seed,
            policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
            model_ref_id=self.model_ref_id,
            token_records=[record],
            metadata={
                "trajectory": {
                    "steps": [
                        {
                            "tool_name": "sql",
                            "action": {"kind": "sql"},
                            "tool_result": {"ok": True, "worker_calls": 1},
                        }
                    ]
                }
            },
            failed_action={"status": "format_failed", "message": "synthetic"}
            if sample_index == 3
            else None,
        )


class FakeWorker:
    def __init__(self):
        self.calls: list[str] = []

    def score(self, task: TaskInput, episode: EpisodeSample) -> ScoreResult:
        self.calls.append(episode.episode_id)
        success = episode.sample_index in {0, 3}
        return ScoreResult(
            episode_id=episode.episode_id,
            task_id=task.task_id,
            status="scored",
            reward_value=1.0 if success else 0.0,
            success=success,
            reason="synthetic_success" if success else "synthetic_business_failure",
            sql_action_count=1,
            sql_ok_count=1 if success else 0,
            worker_count=1,
            runtime_worker_sql_count=1,
            runtime_worker_non_sql_count=0,
            verifier_sql_count=1,
            budget_rejection_count=0,
            error={"code": "synthetic_business_failure"} if not success else None,
            source_kind="cpu_fake",
        )


class EvidenceFakeRolloutEngine:
    """CPU-only evaluation engine that exercises the real evidence guard."""

    def __init__(self, run_dir: Path, *, arm: str, model_ref_id: str):
        self.run_dir = run_dir
        self.arm = arm
        self.model_ref_id = model_ref_id
        self.calls: list[dict[str, Any]] = []
        self.evidence = EvidenceStore(run_dir, stage="evaluation", arm=arm)

    def evaluate(
        self,
        task: TaskInput,
        *,
        policy_snapshot: dict[str, Any],
        episode_id: str,
        seed: int,
    ) -> EpisodeSample:
        self.calls.append(
            {
                "task_id": task.task_id,
                "episode_id": episode_id,
                "seed": seed,
                "policy_snapshot_id": policy_snapshot["policy_snapshot_id"],
            }
        )
        messages = [
            {"role": "system", "content": "synthetic evaluation system"},
            {"role": "user", "content": str(task.public_task.get("prompt", "synthetic"))},
        ]
        generation = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 512,
            "format_retries": 1,
            "parse_mode": "json_fence",
            "seed": seed,
            "mode": "greedy",
        }
        identity = {
            "model_id": "tiny/fake",
            "revision": "fixture",
            "model_ref_id": self.model_ref_id,
            "model_object_id": id(self),
            "policy_snapshot_id": policy_snapshot["policy_snapshot_id"],
            "model_weight_fingerprint": policy_snapshot.get("model_weight_fingerprint"),
            "protocol_hash": policy_snapshot.get("protocol_hash"),
        }
        handle = self.evidence.start_attempt(
            episode_id=episode_id,
            task_id=task.task_id,
            sample_index=0,
            messages=messages,
            generation_settings=generation,
            model_identity=identity,
        )
        content = '{"kind":"finish","arguments":{"evidence_step":0,"summary":"synthetic"}}'
        record = TokenRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=0,
            request_index=1,
            seed=seed,
            policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
            model_ref_id=self.model_ref_id,
            request_id=handle.request_id,
            response_id=handle.response_id,
            prompt_token_ids=[1, 2, 3],
            generated_token_ids=[4, 5, 6],
            response_mask=[0, 0, 0, 1, 1, 1],
            raw_policy_log_probs=[-0.1, -0.2, -0.3],
            behavior_sampling_log_probs=[-0.1, -0.2, -0.3],
            action_text=content,
            action_status="accepted",
            metadata={"physical_request_index": handle.attempt_index},
        )
        self.evidence.persist_response(
            handle,
            record,
            model_id="tiny/fake",
            model_identity=identity,
            generation_settings=generation,
            content=content,
            error=None,
            model_execution_started=True,
            model_execution_completed=True,
            completion_status="completed",
        )
        evidence = record.metadata
        attempt = {
            "attempt_index": 0,
            "raw_output": content,
            "format_valid": True,
            "strict_format_valid": True,
            "error": None,
            "usage": {"prompt_tokens": 3, "completion_tokens": 3},
            "request_id": record.request_id,
            "response_id": record.response_id,
            "attempt_id": evidence["attempt_id"],
            "evidence_attempt_index": evidence["attempt_index"],
            "request_path": evidence["request_path"],
            "response_path": evidence["response_path"],
            "engine_event_path": evidence["engine_event_path"],
            "content_sha256": evidence["content_sha256"],
        }
        trajectory = {
            "model_attempts": [attempt],
            "steps": [
                {
                    "tool_name": None,
                    "action": {"kind": "finish"},
                    "tool_result": {"ok": True},
                }
            ],
        }
        return EpisodeSample(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=0,
            seed=seed,
            policy_snapshot_id=str(policy_snapshot["policy_snapshot_id"]),
            model_ref_id=self.model_ref_id,
            token_records=[record],
            metadata={"trajectory": trajectory, "terminal_reason": "completed"},
        )
