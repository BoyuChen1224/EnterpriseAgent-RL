from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.grpo.budget import BudgetGuard
from enterprise_agent_rl.grpo.config import RunContext, load_config
from enterprise_agent_rl.grpo.contracts import ScoreResult, TaskInput
from enterprise_agent_rl.grpo.evaluation import evaluate_frozen
from enterprise_agent_rl.grpo.evidence import EvidencePersistenceError
from enterprise_agent_rl.grpo.io import (
    ImmutableWriteError,
    exclusive_write_json,
    sha256_file,
)
from enterprise_agent_rl.grpo.testing import EvidenceFakeRolloutEngine, FakeWorker
from enterprise_agent_rl.grpo.torch_backend import HfRolloutEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "training" / "grpo-frozen-evaluation.toml"


def _config(tmp_path: Path):
    return load_config(CONFIG_PATH, RunContext(REPO_ROOT, tmp_path / "run", CONFIG_PATH))[0]


def _guard(tmp_path: Path, config, *, name: str = "evidence-test") -> BudgetGuard:
    guard = BudgetGuard(
        tmp_path / "budget.json",
        run_id=name,
        run_hard_seconds=1000,
        job_hard_seconds=900,
        soft_seconds=800,
        cleanup_reserve_seconds=60,
        call_limits=config.limits,
    )
    guard.start_job(job_id=name)
    return guard


def _task(task_id: str = "task") -> TaskInput:
    return TaskInput(
        task_id=task_id,
        group_id=task_id,
        public_task={"task_id": task_id, "prompt": "synthetic"},
    )


class _Tokenizer:
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
        return '{"kind":"finish","arguments":{"evidence_step":0,"summary":"ok"}}'


class _GenerationModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch

        logits = (torch.zeros((1, 8)), torch.zeros((1, 8)))
        return SimpleNamespace(
            sequences=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            logits=logits,
            scores=logits,
        )


def _hf_engine(tmp_path: Path, guard: BudgetGuard, *, arm: str) -> HfRolloutEngine:
    import torch

    policy = SimpleNamespace(
        model=_GenerationModel(),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        guard=guard,
        model_ref_id=f"fake-model-{arm}",
        evaluation_arm=arm,
    )
    return HfRolloutEngine(
        policy=policy,
        run_dir=tmp_path / "run",
        config=_config(tmp_path),
        evaluation=True,
    )


def _generate(engine: HfRolloutEngine, *, task_id: str = "task"):
    return engine.generate(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ],
        task=_task(task_id),
        episode_id=f"N-{task_id}",
        sample_index=0,
        seed=17,
        policy_snapshot={
            "policy_snapshot_id": "frozen-test",
            "model_weight_fingerprint": "weight-test",
            "protocol_hash": "protocol-test",
        },
        request_index=0,
    )


def test_exclusive_evidence_write_preserves_original_bytes(tmp_path):
    path = tmp_path / "request.json"
    exclusive_write_json(path, {"value": 1})
    before = sha256_file(path)
    with pytest.raises(ImmutableWriteError):
        exclusive_write_json(path, {"value": 2})
    assert sha256_file(path) == before


def test_public_engine_restart_uses_local_request_one_and_persistent_unique_paths(tmp_path):
    config = _config(tmp_path)
    guard = _guard(tmp_path, config)
    try:
        first = _generate(_hf_engine(tmp_path, guard, arm="N"), task_id="first")
        second = _generate(_hf_engine(tmp_path, guard, arm="N"), task_id="second")
        other = _generate(_hf_engine(tmp_path, guard, arm="Base"), task_id="other")
        assert first.token_record.request_index == 1
        assert second.token_record.request_index == 1
        assert other.token_record.request_index == 1
        paths = [
            first.token_record.metadata["request_path"],
            second.token_record.metadata["request_path"],
            other.token_record.metadata["request_path"],
        ]
        assert len(set(paths)) == 3
        assert all("evaluation-" in path and "-a00000" in path for path in paths)
        hashes = {path: sha256_file(tmp_path / "run" / path) for path in paths}
        _generate(_hf_engine(tmp_path, guard, arm="N"), task_id="third")
        assert all(sha256_file(tmp_path / "run" / path) == value for path, value in hashes.items())
    finally:
        guard.close_job(status="COMPLETED")


def test_response_persistence_failure_is_not_downgraded_to_model_failure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    guard = _guard(tmp_path, config, name="persist-failure")
    engine = _hf_engine(tmp_path, guard, arm="N")

    def fail(*_args, **_kwargs):
        raise EvidencePersistenceError("forced_response_failure")

    monkeypatch.setattr(engine.evidence, "persist_response", fail)
    try:
        with pytest.raises(EvidencePersistenceError):
            _generate(engine)
        assert list((tmp_path / "run" / "engine-evidence" / "requests").glob("*.json"))
        assert not list((tmp_path / "run" / "engine-evidence" / "responses").glob("*.json"))
    finally:
        guard.close_job(status="FAILED")


def test_qualification_failure_stops_before_verifier_and_keeps_unknown(tmp_path):
    config = _config(tmp_path)
    guard = _guard(tmp_path, config, name="qualification-failure")
    run_dir = tmp_path / "run"
    engine = EvidenceFakeRolloutEngine(run_dir, arm="Base", model_ref_id="fake-base")
    task = _task("broken")
    snapshot = {"policy_snapshot_id": "snapshot", "protocol_hash": "protocol"}
    episode = engine.evaluate(task, policy_snapshot=snapshot, episode_id="Base-broken", seed=17)
    response_path = run_dir / episode.token_records[0].metadata["response_path"]
    response_path.write_text(response_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    class ExistingEpisodeEngine:
        def evaluate(self, *_args, **_kwargs):
            return episode

    worker = FakeWorker()
    try:
        result = evaluate_frozen(
            run_dir=run_dir,
            arm="Base",
            tasks=[task],
            engine=ExistingEpisodeEngine(),
            worker=worker,
            guard=guard,
            policy_snapshot=snapshot,
            require_qualified_evidence=True,
        )
        assert result["unknown"] == 1
        assert result["stopped_before_verifier"] is True
        assert worker.calls == []
    finally:
        guard.close_job(status="COMPLETED")


def test_qualified_business_failure_is_scored_and_does_not_stop(tmp_path):
    config = _config(tmp_path)
    guard = _guard(tmp_path, config, name="business-failure")
    run_dir = tmp_path / "run"
    tasks = [_task("business-a"), _task("business-b")]
    engine = EvidenceFakeRolloutEngine(run_dir, arm="Base", model_ref_id="fake-base")

    class BusinessFailureWorker:
        def __init__(self):
            self.calls = []

        def score(self, task, episode):
            self.calls.append(task.task_id)
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="scored",
                reward_value=0.0,
                success=False,
                reason="normal_business_failure",
                source_kind="cpu_fake",
            )

    worker = BusinessFailureWorker()
    try:
        result = evaluate_frozen(
            run_dir=run_dir,
            arm="Base",
            tasks=tasks,
            engine=engine,
            worker=worker,
            guard=guard,
            policy_snapshot={"policy_snapshot_id": "snapshot", "protocol_hash": "protocol"},
            require_qualified_evidence=True,
        )
        assert result["status"] == "COMPLETED"
        assert result["scored"] == 2
        assert result["unknown"] == 0
        assert result["successes"] == 0
        assert worker.calls == ["business-a", "business-b"]
    finally:
        guard.close_job(status="COMPLETED")
