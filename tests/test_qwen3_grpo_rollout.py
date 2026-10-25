from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from enterprise_agent_rl.grpo.budget import BudgetGuard
from enterprise_agent_rl.grpo.config import RunContext
from enterprise_agent_rl.grpo.contracts import EpisodeSample, ScoreResult, TaskInput, TokenRecord
from enterprise_agent_rl.grpo.evidence import EvidencePersistenceError
from enterprise_agent_rl.grpo.io import read_jsonl
from enterprise_agent_rl.grpo.qwen3_rollout import (
    QWEN3_MODEL_ID,
    QWEN3_MODULE_COUNT,
    QWEN3_PARENT_CONFIG_SHA256,
    QWEN3_PARENT_WEIGHT_SHA256,
    QWEN3_REVISION,
    expected_module_names,
    load_qwen3_rollout_config,
    load_qwen3_rollout_policy,
    validate_qwen3_rollout_values,
)
from enterprise_agent_rl.grpo.torch_backend import HfRolloutEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/qwen3-grpo-rollout.toml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _config(tmp_path: Path):
    return load_qwen3_rollout_config(
        CONFIG_PATH,
        RunContext(REPO_ROOT, tmp_path / "run", CONFIG_PATH),
        cache_dir=tmp_path / "cache",
        parent_adapter=tmp_path / "parent",
    )[0]


def _guard(tmp_path: Path, name: str) -> BudgetGuard:
    guard = BudgetGuard(
        tmp_path / f"{name}-budget.json",
        run_id=name,
        run_hard_seconds=1000,
        job_hard_seconds=900,
        soft_seconds=800,
        cleanup_reserve_seconds=60,
        call_limits={
            "model_requests": 640,
            "tool_actions": 240,
            "runtime_workers": 240,
            "verifier_calls": 24,
            "verifier_sql": 48,
        },
    )
    guard.start_job(job_id=name)
    return guard


def _tasks() -> list[TaskInput]:
    return [
        TaskInput(
            task_id=task_id,
            group_id=task_id,
            public_task={"task_id": task_id, "prompt": f"synthetic {task_id}"},
            database_path=None,
        )
        for task_id in (
            "train-0-schema",
            "train-0-region",
            "train-1-product",
            "train-1-recovery",
        )
    ]


def _schedule(run_dir: Path, config):
    from scripts.evaluation.run_qwen3_grpo_rollout import _write_schedule

    rows = [
        {
            "task_id": task.task_id,
            "group_id": task.group_id,
            "database_id": "train-0" if task.task_id.startswith("train-0") else "train-1",
            "template": task.task_id.rsplit("-", 1)[-1],
        }
        for task in _tasks()
    ]
    return _write_schedule(run_dir, rows, config)


class _TinySignalEngine:
    evidence = None

    def __init__(self, *, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls: list[str] = []

    def rollout(self, task, *, round_index, sample_index, seed, policy_snapshot, episode_id):
        self.calls.append(episode_id)
        if self.fail_first and len(self.calls) == 1:
            raise EvidencePersistenceError("synthetic_persist_failure")
        generated = [4, 5, 6]
        record = TokenRecord(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            request_index=len(self.calls),
            seed=seed,
            policy_snapshot_id=policy_snapshot["policy_snapshot_id"],
            model_ref_id="tiny-rollout",
            request_id=f"request-{len(self.calls)}",
            response_id=f"response-{len(self.calls)}",
            prompt_token_ids=[1, 2],
            generated_token_ids=generated,
            response_mask=[0, 0, 1, 1, 1],
            raw_policy_log_probs=[-0.1, -0.2, -0.3],
            behavior_sampling_log_probs=[-0.1, -0.2, -0.3],
            action_text='{"kind":"sql","arguments":{"query":"SELECT 1"}}',
            metadata={
                "eos_token_id": 6,
                "eos_generated": True,
                "native_score_count": 3,
                "native_score_log_probs": [-0.1, -0.2, -0.3],
                "native_raw_log_probs": [-0.1, -0.2, -0.3],
            },
        )
        return EpisodeSample(
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            sample_index=sample_index,
            seed=seed,
            policy_snapshot_id=policy_snapshot["policy_snapshot_id"],
            model_ref_id="tiny-rollout",
            token_records=[record],
            metadata={
                "trajectory": {"terminal_reason": "completed", "model_attempts": [], "steps": []}
            },
            failed_action=(
                {"status": "format_failed", "error": "synthetic_format_failure"}
                if sample_index == 3
                else None
            ),
        )


class _MixedWorker:
    def __init__(self, *, unknown_at: str | None = None):
        self.calls: list[str] = []
        self.unknown_at = unknown_at

    def score(self, task, episode):
        self.calls.append(episode.episode_id)
        if episode.episode_id == self.unknown_at:
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="unknown",
                infra_error="synthetic_transport_failure",
                source_kind="real",
            )
        success = episode.sample_index == 0
        return ScoreResult(
            episode_id=episode.episode_id,
            task_id=task.task_id,
            status="scored",
            reward_value=1.0 if success else 0.0,
            success=success,
            reason="success" if success else "business_failure",
            source_kind="real",
        )


def _fake_policy():
    return SimpleNamespace(
        model_ref_id="tiny-rollout",
        rollout_identity={
            "model_id": QWEN3_MODEL_ID,
            "revision": QWEN3_REVISION,
            "parent_adapter": {
                "adapter_model_sha256": QWEN3_PARENT_WEIGHT_SHA256,
                "adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
            },
            "fresh_base_load_count": 1,
            "parent_adapter_merge_count": 1,
            "all_parameters_frozen": True,
            "module_count": QWEN3_MODULE_COUNT,
        },
    )


def _run_fake_schedule(tmp_path: Path, *, unknown_at: str | None = None, fail_first: bool = False):
    from scripts.evaluation.run_qwen3_grpo_rollout import _run_schedule

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _config(tmp_path)
    schedule = _schedule(run_dir, config)
    guard = _guard(tmp_path, "schedule")
    engine = _TinySignalEngine(fail_first=fail_first)
    worker = _MixedWorker(unknown_at=unknown_at)
    snapshot = {
        "policy_snapshot_id": "tiny-snapshot",
        "training_calls": {"backward": 0, "optimizer_create": 0, "optimizer_step": 0, "updates": 0},
    }
    result = _run_schedule(
        {"run": run_dir},
        config,
        guard,
        _tasks(),
        _fake_policy(),
        engine=engine,
        worker=worker,
        snapshot=snapshot,
    )
    guard.close_job(status="COMPLETED")
    return run_dir, result, engine, worker, schedule


def test_qwen3_contract_is_fixed_and_module_identity_is_36x7():
    assert len(expected_module_names()) == QWEN3_MODULE_COUNT
    assert expected_module_names()[0] == "model.layers.0.self_attn.q_proj"
    assert expected_module_names()[-1] == "model.layers.35.mlp.down_proj"
    values = json.loads(json.dumps(__import__("tomllib").loads(CONFIG_PATH.read_text())))
    validate_qwen3_rollout_values(values)
    values["model"]["revision"] = "arbitrary"
    with pytest.raises(ValueError, match="revision_not_allowlisted"):
        validate_qwen3_rollout_values(values)


def test_fixed_schedule_has_four_groups_and_contiguous_seeds(tmp_path):
    config = _config(tmp_path)
    schedule = _schedule(tmp_path, config)
    assert len(schedule) == 16
    assert [row["seed"] for row in schedule] == list(range(74, 90))
    assert [row["task_id"] for row in schedule[::4]] == [
        "train-0-schema",
        "train-0-region",
        "train-1-product",
        "train-1-recovery",
    ]


def test_fake_schedule_preserves_business_zero_and_format_failure(tmp_path):
    run_dir, result, engine, worker, _ = _run_fake_schedule(tmp_path)
    assert result["status"] == "COMPLETED"
    assert len(engine.calls) == 16
    rows = read_jsonl(run_dir / "case-results.jsonl")
    assert len(rows) == 16
    assert sum(row["reward_value"] == 0.0 for row in rows) == 12
    assert all(row["status"] == "scored" for row in rows)
    failed = [row for row in rows if row["sample_index"] == 3]
    assert len(failed) == 4
    assert all(row["episode_summary"]["format_failure"] is True for row in failed)
    assert all(row["source_kind"] == "real" for row in rows)


def test_group_metrics_requires_four_scored_members_and_uses_population_variance():
    from scripts.evaluation.run_qwen3_grpo_rollout import TASK_IDS, _group_metrics

    schedule = [
        {
            "episode_id": f"e-{group}-{sample}",
            "task_id": group,
            "group_id": group,
            "sample_index": sample,
        }
        for group in TASK_IDS
        for sample in range(4)
    ]
    scores = {}
    values = {
        TASK_IDS[0]: [1.0, 0.0, 1.0, 0.0],
        TASK_IDS[1]: [1.0] * 4,
        TASK_IDS[2]: [0.0] * 4,
        TASK_IDS[3]: [1.0] * 4,
    }
    for row in schedule:
        scores[row["episode_id"]] = {
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "status": "scored",
            "reward_value": values[row["task_id"]][row["sample_index"]],
        }
    metrics = _group_metrics(schedule, scores)
    assert metrics["groups"][0]["population_variance"] == pytest.approx(0.25)
    assert metrics["nondegenerate_group_count"] == 1
    scores[f"e-{TASK_IDS[0]}-3"]["status"] = "unknown"
    incomplete = _group_metrics(schedule, scores)
    assert incomplete["groups"][0]["complete"] is False
    assert incomplete["groups"][0]["population_variance"] is None
    assert incomplete["groups"][0]["expected_denominator"] == 4


def test_unknown_stops_new_episodes_and_does_not_fill_missing_reward(tmp_path):
    unknown_at = "r0-train-0-region-s0"
    run_dir, result, engine, worker, _ = _run_fake_schedule(tmp_path, unknown_at=unknown_at)
    assert result["status"] == "INCOMPLETE"
    assert result["unknown"] == 1
    assert len(engine.calls) == 5
    rows = read_jsonl(run_dir / "case-results.jsonl")
    assert rows[-1]["episode_id"] == unknown_at
    assert rows[-1]["status"] == "unknown"
    assert all(row["task_id"] != "train-1-product" for row in rows)


def test_persistence_failure_stops_and_resume_is_idempotent(tmp_path):
    run_dir, result, failed_engine, _, schedule = _run_fake_schedule(tmp_path, fail_first=True)
    assert result["status"] == "INCOMPLETE"
    assert len(failed_engine.calls) == 1
    failures = read_jsonl(run_dir / "raw/engine-failures.jsonl")
    assert len(failures) == 1
    assert not (run_dir / "raw/episodes.jsonl").exists()

    from scripts.evaluation.run_qwen3_grpo_rollout import _run_schedule

    config = _config(tmp_path)
    guard = _guard(tmp_path, "resume")
    engine = _TinySignalEngine()
    worker = _MixedWorker()
    result = _run_schedule(
        {"run": run_dir},
        config,
        guard,
        _tasks(),
        _fake_policy(),
        engine=engine,
        worker=worker,
        snapshot={
            "policy_snapshot_id": "tiny-snapshot",
            "training_calls": {
                "backward": 0,
                "optimizer_create": 0,
                "optimizer_step": 0,
                "updates": 0,
            },
        },
    )
    guard.close_job(status="COMPLETED")
    assert result["status"] == "COMPLETED"
    assert len(engine.calls) == 16
    guard = _guard(tmp_path, "resume-idempotent")
    second_engine = _TinySignalEngine()
    result = _run_schedule(
        {"run": run_dir},
        config,
        guard,
        _tasks(),
        _fake_policy(),
        engine=second_engine,
        worker=_MixedWorker(),
        snapshot={
            "policy_snapshot_id": "tiny-snapshot",
            "training_calls": {
                "backward": 0,
                "optimizer_create": 0,
                "optimizer_step": 0,
                "updates": 0,
            },
        },
    )
    guard.close_job(status="COMPLETED")
    assert result["status"] == "COMPLETED"
    assert second_engine.calls == []
    assert len(read_jsonl(run_dir / "case-results.jsonl")) == 16


def test_reference_sql_audit_ignores_model_generated_assistant_history(tmp_path):
    from scripts.evaluation.run_qwen3_grpo_rollout import _scan_reference_sql_policy_inputs

    request_root = tmp_path / "requests"
    request_root.mkdir()
    (request_root / "assistant.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "public instructions"},
                    {"role": "assistant", "content": "SELECT private_benchmark"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (request_root / "input.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "SELECT private_benchmark"},
                ]
            }
        ),
        encoding="utf-8",
    )

    scan = _scan_reference_sql_policy_inputs(
        request_root,
        ["SELECT private_benchmark"],
        tmp_path,
    )
    assert scan["non_assistant_input_matches"] == ["requests/input.json"]
    assert scan["assistant_generated_matches"] == ["requests/assistant.json"]


def test_artifact_manifest_excludes_source_snapshot_and_mutable_command_ledger(tmp_path):
    from scripts.evaluation.run_qwen3_grpo_rollout import _write_artifact_manifest

    (tmp_path / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "commands.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "source-snapshot").mkdir()
    (tmp_path / "source-snapshot/manifest.json").write_text("{}\n", encoding="utf-8")

    manifest = _write_artifact_manifest(tmp_path)
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "result.json"
    assert not any(row["path"].startswith("source-snapshot/") for row in manifest["files"])
    assert (tmp_path / "artifact-manifest.json").is_file()


class _FakeTokenizer:
    chat_template = "{{ messages }}"
    pad_token_id = 0
    eos_token_id = 5

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "synthetic prompt"

    def __call__(self, text, *, return_tensors):
        import torch

        assert text == "synthetic prompt"
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens is True
        return '{"kind":"finish","arguments":{"evidence_step":0,"summary":"ok"}}'


class _FakeGenerateModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch

        logits = (torch.zeros((1, 8)), torch.zeros((1, 8)))
        return SimpleNamespace(
            sequences=torch.tensor([[1, 2, 4, 5]], dtype=torch.long),
            logits=logits,
            scores=logits,
        )


def test_native_scores_eos_and_completion_mask_are_persisted(tmp_path):
    config = _config(tmp_path)
    guard = _guard(tmp_path, "native-token-contract")
    policy = SimpleNamespace(
        model=_FakeGenerateModel(),
        tokenizer=_FakeTokenizer(),
        device=__import__("torch").device("cpu"),
        guard=guard,
        model_ref_id="tiny-qwen3",
        evaluation_arm=None,
    )
    engine = HfRolloutEngine(
        policy=policy,
        run_dir=tmp_path / "evidence",
        config=config,
        evaluation=False,
    )
    result = engine.generate(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        task=TaskInput(
            task_id="task", group_id="task", public_task={"task_id": "task", "prompt": "u"}
        ),
        episode_id="episode",
        sample_index=0,
        seed=74,
        policy_snapshot={"policy_snapshot_id": "snapshot"},
        request_index=0,
    )
    guard.close_job(status="COMPLETED")
    row = result.token_record.to_dict()
    assert row["generated_token_ids"] == [4, 5]
    assert row["completion_mask"] == [1, 1]
    assert row["response_mask"] == [0, 0, 1, 1]
    assert row["metadata"]["eos_generated"] is True
    assert row["metadata"]["native_score_count"] == 2
    assert row["metadata"]["sampling_seed"] == 74


def test_rollout_loader_never_constructs_optimizer_or_training_state(tmp_path, monkeypatch):
    import enterprise_agent_rl.grpo.qwen3_rollout as qwen3

    config = _config(tmp_path)
    context = RunContext(REPO_ROOT, tmp_path / "loader-run", CONFIG_PATH)
    context.run_dir.mkdir()
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    class ForbiddenOptim:
        def __getattribute__(self, name):
            raise AssertionError(f"optimizer path touched: {name}")

    class FakeTorch:
        float16 = "float16"
        optim = ForbiddenOptim()
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        )

        @staticmethod
        def use_deterministic_algorithms(*_args, **_kwargs):
            return None

        @staticmethod
        def device(value):
            return value

    class FakeParameter:
        requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = bool(value)
            return self

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self.parameters_seen = [FakeParameter(), FakeParameter()]

        def to(self, _device):
            return self

        def parameters(self):
            return self.parameters_seen

        def eval(self):
            return self

    fake_model = FakeModel()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: fake_model
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(qwen3, "_torch", lambda: FakeTorch())
    monkeypatch.setattr(
        qwen3,
        "_require_v100_device",
        lambda _torch: {
            "name": "V100",
            "compute_capability": [7, 0],
            "total_memory_bytes": 32 * 1024**3,
            "device_index": 0,
        },
    )
    monkeypatch.setattr(
        qwen3,
        "_adapter_identity",
        lambda _path, _config: {
            "adapter_model_sha256": QWEN3_PARENT_WEIGHT_SHA256,
            "adapter_config_sha256": QWEN3_PARENT_CONFIG_SHA256,
            "module_count": QWEN3_MODULE_COUNT,
        },
    )
    monkeypatch.setattr(
        "enterprise_agent_rl.lora.merge_lora_adapter",
        lambda *_args, **_kwargs: {"module_count": QWEN3_MODULE_COUNT},
    )

    class Guard:
        def check(self, _phase):
            return None

    policy = load_qwen3_rollout_policy(context, config, Guard())
    assert policy.optimizer is None
    assert policy.scheduler is None
    assert policy.scaler is None
    assert all(parameter.requires_grad is False for parameter in fake_model.parameters_seen)
    assert policy.parent_merge["fresh_base_load_count"] == 1
    assert policy.parent_merge["parent_merge_count"] == 1
