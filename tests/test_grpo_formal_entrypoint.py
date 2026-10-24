from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.grpo.aggregation import aggregate_source
from enterprise_agent_rl.grpo.budget import BudgetGuard, BudgetHardStop
from enterprise_agent_rl.grpo.config import RunContext, load_config
from enterprise_agent_rl.grpo.contracts import ScoreResult, TaskInput
from enterprise_agent_rl.grpo.io import append_jsonl, read_json
from enterprise_agent_rl.grpo.orchestration import run_online_training
from enterprise_agent_rl.grpo.rollout import _seed_for
from enterprise_agent_rl.grpo.testing import (
    FakeRolloutEngine,
    FakeWorker,
    TinyPolicy,
    tiny_qwen_policy,
)
from enterprise_agent_rl.grpo.torch_backend import HfRolloutEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training" / "grpo-online.toml"
SMOKE_CONFIG = REPO_ROOT / "configs" / "training" / "grpo-integration-smoke.toml"


def _tasks(count: int = 4) -> list[TaskInput]:
    return [
        TaskInput(
            task_id=f"formal-task-{index}",
            group_id=f"formal-group-{index}",
            public_task={"task_id": f"formal-task-{index}", "prompt": "synthetic"},
        )
        for index in range(count)
    ]


def _guard(path: Path, config, *, job_id: str = "TEST") -> BudgetGuard:
    guard = BudgetGuard(
        path,
        run_id="formal-test",
        run_hard_seconds=1000,
        job_hard_seconds=900,
        soft_seconds=800,
        cleanup_reserve_seconds=60,
        call_limits=config.limits,
    )
    guard.start_job(job_id=job_id)
    return guard


def test_seed_formula_and_smoke_config_are_not_hardcoded_to_four_groups(tmp_path):
    config, _ = load_config(
        SMOKE_CONFIG, RunContext(REPO_ROOT, tmp_path / "smoke", SMOKE_CONFIG)
    )
    assert config.group_count == 2
    assert [
        _seed_for(0, group, sample, 106, config.group_count, config.samples_per_task)
        for group in range(2)
        for sample in range(4)
    ] == list(range(106, 114))
    assert [
        _seed_for(1, group, sample, 106, config.group_count, config.samples_per_task)
        for group in range(2)
        for sample in range(4)
    ] == list(range(114, 122))
    default, _ = load_config(
        DEFAULT_CONFIG, RunContext(REPO_ROOT, tmp_path / "default", DEFAULT_CONFIG)
    )
    assert default.rollout["seed_end"] == 105


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "synthetic prompt"

    def __call__(self, text, *, return_tensors):
        import torch

        assert text == "synthetic prompt"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens is True
        return '{"kind":"finish","arguments":{"summary":"ok"}}'


class _FakeGenerationModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch

        self.calls.append(kwargs)
        logits = (torch.zeros((1, 8)), torch.zeros((1, 8)))
        return SimpleNamespace(
            sequences=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            logits=logits,
            scores=logits,
        )


def test_hf_generation_separates_top_k_and_greedy_evaluation(tmp_path):
    config, _ = load_config(
        DEFAULT_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", DEFAULT_CONFIG)
    )
    guard = _guard(tmp_path / "budget.json", config)
    model = _FakeGenerationModel()
    policy = SimpleNamespace(
        model=model,
        tokenizer=_FakeTokenizer(),
        device=__import__("torch").device("cpu"),
        guard=guard,
        model_ref_id="fake-model",
    )
    task = TaskInput(
        task_id="task",
        group_id="group",
        public_task={"task_id": "task", "prompt": "prompt"},
    )
    try:
        training_engine = HfRolloutEngine(
            policy=policy, run_dir=tmp_path / "run", config=config
        )
        training = training_engine.generate(
            messages=[{"role": "user", "content": "prompt"}],
            task=task,
            episode_id="episode-train",
            sample_index=0,
            seed=74,
            policy_snapshot={"policy_snapshot_id": "train-snapshot"},
            request_index=0,
        )
        assert training.token_record.metadata["generation_settings"]["top_k"] == 0
        assert model.calls[-1]["top_k"] == 0

        evaluation_engine = HfRolloutEngine(
            policy=policy, run_dir=tmp_path / "run", config=config, evaluation=True
        )
        evaluation = evaluation_engine.generate(
            messages=[{"role": "user", "content": "prompt"}],
            task=task,
            episode_id="episode-eval",
            sample_index=0,
            seed=17,
            policy_snapshot={"policy_snapshot_id": "eval-snapshot"},
            request_index=0,
        )
        assert evaluation.token_record.metadata["generation_settings"]["mode"] == "greedy"
        assert evaluation.token_record.metadata["generation_settings"]["seed"] == 17
        assert model.calls[-1]["do_sample"] is False
        assert model.calls[-1]["temperature"] == 1.0
        assert model.calls[-1]["top_p"] == 1.0
        assert model.calls[-1]["top_k"] == 0
    finally:
        guard.close_job(status="COMPLETED")


class _OneSkipScaler:
    """CPU scaler double that exercises the native skip/replay branch."""

    def __init__(self):
        self.scale_value = 4096.0
        self.skipped = False
        self.pending_backoff = False

    def scale(self, value):
        return value * self.scale_value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)

    def step(self, optimizer):
        if not self.skipped:
            self.skipped = True
            self.pending_backoff = True
            return
        optimizer.step()

    def update(self):
        if self.pending_backoff:
            self.scale_value *= 0.0625
            self.pending_backoff = False

    def get_scale(self):
        return self.scale_value

    def state_dict(self):
        return {"scale": self.scale_value}

    def load_state_dict(self, value):
        self.scale_value = float(value["scale"])


def test_production_qwen_update_freezes_base_and_replays_one_amp_skip(tmp_path):
    config, _ = load_config(
        DEFAULT_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", DEFAULT_CONFIG)
    )
    guard = _guard(tmp_path / "budget.json", config)
    policy = tiny_qwen_policy(tmp_path / "run", guard, config)
    policy.scaler = _OneSkipScaler()
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.model.named_parameters()
        if not parameter.requires_grad
    }
    before_delta = policy.policy_fingerprint()
    records = [
        {
            "episode_id": "a",
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [3, 4],
            "old_per_token_logps": [-1.0, -1.0],
            "advantage": 1.0,
        },
        {
            "episode_id": "b",
            "prompt_token_ids": [2, 3],
            "generated_token_ids": [4, 5],
            "old_per_token_logps": [-1.0, -1.0],
            "advantage": -1.0,
        },
    ]
    try:
        result = policy.update(records, logical_step=1, total_tokens=4)
        assert result["status"] == "UPDATED"
        assert result["amp_skip_replays"] == 1
        assert result["scale_after"] == pytest.approx(256.0)
        assert policy.policy_fingerprint() != before_delta
        assert all(
            parameter.detach().equal(base_before[name])
            for name, parameter in policy.model.named_parameters()
            if name in base_before
        )
        assert read_json(tmp_path / "budget.json")["call_counters"]["amp_skip_replays"] == 1
    finally:
        policy.close()
        guard.close_job(status="COMPLETED")


@pytest.mark.parametrize(("failed_step", "second_round_calls"), [(1, 16), (3, 0)])
def test_step_one_and_step_three_resume_without_repeating_rollouts(
    tmp_path, failed_step, second_round_calls
):
    config, _ = load_config(
        DEFAULT_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", DEFAULT_CONFIG)
    )
    run_dir = tmp_path / "run"
    fail_marker = tmp_path / "failed-once"
    captures: list[tuple[object, object]] = []

    class FailOncePolicy(TinyPolicy):
        def update(self, records, *, logical_step, total_tokens):
            result = super().update(records, logical_step=logical_step, total_tokens=total_tokens)
            if logical_step == failed_step and not fail_marker.exists():
                fail_marker.write_text("injected", encoding="utf-8")
                raise RuntimeError(f"injected_failure_{failed_step}")
            return result

    def factory(guard, context, _config):
        policy = FailOncePolicy(context.run_dir, guard)
        engine = FakeRolloutEngine()
        captures.append((policy, engine))
        return policy, engine, FakeWorker()

    with pytest.raises(RuntimeError, match=f"injected_failure_{failed_step}"):
        run_online_training(
            RunContext(REPO_ROOT, run_dir, DEFAULT_CONFIG),
            config,
            _tasks(),
            component_factory=factory,
            job_id=f"FAIL-{failed_step}",
        )
    first_engine = captures[0][1]
    assert len(first_engine.calls) == 16 if failed_step == 1 else 32

    result = run_online_training(
        RunContext(REPO_ROOT, run_dir, DEFAULT_CONFIG),
        config,
        _tasks(),
        component_factory=factory,
        job_id=f"RESUME-{failed_step}",
    )
    assert result["status"] == "COMPLETED"
    assert result["updates"] == 4
    assert len(captures[1][1].calls) == second_round_calls
    state = read_json(run_dir / "training" / "state.json")
    assert [row["logical_step"] for row in state["metrics"]] == [1, 2, 3, 4]
    assert len(read_json(run_dir / "training" / "metrics-history.json")) == 4


def test_watchdog_persists_stop_before_interrupting_callable(tmp_path):
    guard = BudgetGuard(
        tmp_path / "budget.json",
        run_id="watchdog-test",
        run_hard_seconds=4,
        job_hard_seconds=4,
        soft_seconds=3,
        cleanup_reserve_seconds=1,
    )
    guard.start_job(job_id="WATCHDOG")
    with pytest.raises(BudgetHardStop, match="watchdog_hard_stop"):
        guard.run_callable(lambda: time.sleep(30), phase="blocked-generate")
    state = read_json(tmp_path / "budget.json")
    assert state["status"] == "HARD_STOP_REQUESTED"
    assert any(event["event"] == "watchdog_stop_requested" for event in state["events"])
    elapsed = state["active_attempt"]["elapsed_gpu_seconds"]
    guard.recover_interrupted(elapsed_gpu_seconds=elapsed, reason="watchdog-test")
    assert read_json(tmp_path / "budget.json")["active_attempt"] is None


def test_evaluation_arms_have_independent_snapshots_and_aggregation(tmp_path):
    config, _ = load_config(
        DEFAULT_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", DEFAULT_CONFIG)
    )

    def factory(guard, context, _config):
        return TinyPolicy(context.run_dir, guard), FakeRolloutEngine(), FakeWorker()

    def evaluation_factory(arm, _guard, _context, _config):
        return {
            "policy": None,
            "engine": FakeRolloutEngine(model_ref_id=f"fresh-{arm}"),
            "worker": FakeWorker(),
            "metadata": {
                "model_ref_id": f"fresh-{arm}",
                "model_object_id": f"object-{arm}",
                "model_weight_fingerprint": f"weight-{arm}",
                "parent_merge_count": 1,
                "delta_merge_count": int(arm == "RL"),
                "fresh_base": True,
            },
        }

    result = run_online_training(
        RunContext(REPO_ROOT, tmp_path / "run", DEFAULT_CONFIG),
        config,
        _tasks(),
        component_factory=factory,
        evaluation_tasks={"N": _tasks(), "RL": _tasks()},
        evaluation_factory=evaluation_factory,
        run_evaluation=True,
        job_id="EVAL-ARMS",
    )
    assert result["comparison"]["status"] == "COMPLETE"
    n = read_json(tmp_path / "run" / "evaluation" / "N" / "arm-manifest.json")
    rl = read_json(tmp_path / "run" / "evaluation" / "RL" / "arm-manifest.json")
    assert n["arm_snapshot"]["policy_snapshot_id"].endswith("-N")
    assert rl["arm_snapshot"]["policy_snapshot_id"].endswith("-RL")
    assert n["arm_metadata"]["model_object_id"] != rl["arm_metadata"]["model_object_id"]
    assert n["arm_metadata"]["delta_merge_count"] == 0
    assert rl["arm_metadata"]["delta_merge_count"] == 1


def test_missing_worker_audit_is_null_not_zero(tmp_path):
    index = tmp_path / "verifier.jsonl"
    append_jsonl(
        index,
        {
            "record_type": "RESULT",
            "task_id": "task-a",
            "episode_id": "episode-a",
            "status": "scored",
            "raw_saved": True,
            "success": True,
            "reward_value": 1.0,
            "trajectory": {"steps": [{"action": {"kind": "finish"}}]},
        },
    )
    result = aggregate_source(
        source_root=tmp_path,
        verifier_index=index,
        identity_field="task_id",
        expected_count=1,
        source_kind="formal-test",
    )
    assert result["unknown"] == 0
    assert result["worker_count"] is None
    assert result["runtime_worker_sql_count"] is None
    assert result["runtime_worker_non_sql_count"] is None
    assert result["verifier_sql_count"] is None
    assert ScoreResult.from_dict(
        {"episode_id": "e", "task_id": "t", "status": "unknown"}
    ).worker_count is None
