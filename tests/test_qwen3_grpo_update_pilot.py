from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.grpo.budget import BudgetGuard
from enterprise_agent_rl.grpo.config import RunContext, load_config
from enterprise_agent_rl.grpo.qwen3_update import (
    EXPECTED_ACTIVE_RECORDS,
    EXPECTED_ACTIVE_TOKENS,
    EXPECTED_MAX_SEQUENCE,
    EXPECTED_TOKEN_ROWS,
    EXPECTED_TOTAL_TOKENS,
    EXPECTED_ZERO_RECORDS,
    build_frozen_train_batch,
    load_qwen3_update_config,
    validate_train_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
T034_CONFIG = REPO_ROOT / "configs/training/qwen3-grpo-update-pilot.toml"
T033_RUN = REPO_ROOT / "artifacts/qwen3-grpo-rollout-signal/t033-v1"
LEGACY_CONFIG = REPO_ROOT / "configs/training/grpo-online.toml"


def _guard(path: Path, *, name: str = "t034-cpu") -> BudgetGuard:
    guard = BudgetGuard(
        path,
        run_id=name,
        run_hard_seconds=1000,
        job_hard_seconds=900,
        soft_seconds=800,
        cleanup_reserve_seconds=60,
        call_limits={
            "forward_backward": 100,
            "optimizer_update_attempts": 10,
            "successful_optimizer_updates": 10,
            "amp_skip_replays": 4,
        },
    )
    guard.start_job(job_id=name)
    return guard


def test_fixed_config_has_frozen_model_and_module_count(tmp_path):
    config, _ = load_qwen3_update_config(
        T034_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", T034_CONFIG)
    )
    assert config.model["model_id"] == "Qwen/Qwen3-8B"
    assert config.lora["module_count"] == 252


@pytest.mark.local_artifacts
@pytest.mark.skipif(
    not (T033_RUN / "artifact-manifest.json").is_file(),
    reason="The frozen T-033 experiment is a local, Git-ignored artifact bundle.",
)
def test_real_batch_has_frozen_counts(tmp_path):
    output = tmp_path / "train-batch.jsonl"
    summary = build_frozen_train_batch(T033_RUN, output)
    assert summary["counts"] == {
        "records": EXPECTED_TOKEN_ROWS,
        "active_records": EXPECTED_ACTIVE_RECORDS,
        "zero_records": EXPECTED_ZERO_RECORDS,
        "total_tokens": EXPECTED_TOTAL_TOKENS,
        "active_tokens": EXPECTED_ACTIVE_TOKENS,
        "max_full_sequence_tokens": EXPECTED_MAX_SEQUENCE,
    }
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 118
    assert all(row["advantage"] is not None for row in rows)
    assert sum(float(row["advantage"]) == 0 for row in rows) == 48
    assert all(row["old_per_token_logps"] for row in rows)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == summary["output_sha256"]


def _synthetic_batch():
    lengths = [49] * 69 + [77] + [26] * 47 + [65]
    return [
        {
            "record_type": "TOKEN",
            "request_id": f"synthetic-{index}",
            "advantage": 1.0 if index < 70 else 0.0,
            "prompt_token_ids": [1] * (EXPECTED_MAX_SEQUENCE - length if index == 0 else 2),
            "generated_token_ids": [2] * length,
            "completion_mask": [1] * length,
            "old_per_token_logps": [-1.0] * length,
        }
        for index, length in enumerate(lengths)
    ]


def test_batch_validator_accepts_complete_synthetic_contract():
    assert validate_train_batch(_synthetic_batch())["total_tokens"] == EXPECTED_TOTAL_TOKENS


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), "0"])
def test_batch_validator_rejects_null_or_nonfinite_advantage(bad_value):
    rows = _synthetic_batch()
    rows[0]["advantage"] = bad_value
    with pytest.raises(ValueError, match="advantage_invalid"):
        validate_train_batch(rows)


def test_local_loss_uses_last_completion_causal_positions():
    torch = pytest.importorskip("torch")
    from enterprise_agent_rl.grpo.upstream import LocalGRPOAdapter

    class PredictNext(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            values = torch.zeros(input_ids.size(0), input_ids.size(1), 16, dtype=torch.float32)
            for index in range(input_ids.size(1) - 1):
                values[:, index, input_ids[:, index + 1]] = 20.0
            return SimpleNamespace(logits=values)

    adapter = LocalGRPOAdapter()
    result = adapter._get_per_token_logps(
        PredictNext(),
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.ones((1, 5), dtype=torch.long),
        logits_to_keep=2,
    )
    assert tuple(result.shape) == (1, 2)
    assert torch.isfinite(result).all()
    assert float(result[0, 0]) > -1.0
    assert float(result[0, 1]) > -1.0


def test_qwen_policy_inputs_preserve_completion_mask():
    torch = pytest.importorskip("torch")
    from enterprise_agent_rl.grpo.torch_backend import QwenPolicy

    policy = object.__new__(QwenPolicy)
    policy.device = torch.device("cpu")
    policy.config = SimpleNamespace(model={"max_context_tokens": 4096})
    inputs = policy._inputs(
        {
            "episode_id": "mask-test",
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [3, 4, 5],
            "completion_mask": [1, 1, 0],
            "old_per_token_logps": [-1.0, -1.0, -1.0],
            "advantage": 1.0,
        }
    )
    assert inputs["completion_mask"].tolist() == [[1.0, 1.0, 0.0]]


def test_checkpoint_resume_continues_at_step_two_without_repeating_step_one(tmp_path):
    torch = pytest.importorskip("torch")
    from enterprise_agent_rl.grpo.testing import tiny_qwen_policy

    config, _ = load_config(LEGACY_CONFIG, RunContext(REPO_ROOT, tmp_path / "run", LEGACY_CONFIG))
    records = [
        {
            "episode_id": "a",
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [3, 4],
            "completion_mask": [1, 1],
            "old_per_token_logps": [-1.0, -1.0],
            "advantage": 1.0,
        },
        {
            "episode_id": "b",
            "prompt_token_ids": [2, 3],
            "generated_token_ids": [4, 5],
            "completion_mask": [1, 1],
            "old_per_token_logps": [-1.0, -1.0],
            "advantage": -1.0,
        },
    ]
    run_dir = tmp_path / "run"
    torch.manual_seed(42)
    guard_a = _guard(tmp_path / "a-budget.json", name="t034-a")
    continuous = tiny_qwen_policy(run_dir, guard_a, config)
    try:
        continuous.update(records, logical_step=1, total_tokens=4)
        continuous.save_checkpoint(
            run_dir,
            logical_step=1,
            metadata={"input_fingerprint": "fixed", "logical_step": 1},
        )
        continuous.update(records, logical_step=2, total_tokens=4)
        continuous_fingerprint = continuous.policy_fingerprint()
    finally:
        continuous.close()
        guard_a.close_job(status="COMPLETED")

    torch.manual_seed(42)
    guard_b = _guard(tmp_path / "b-budget.json", name="t034-b")
    resumed = tiny_qwen_policy(run_dir, guard_b, config)
    try:
        resumed.load_run_state(
            run_dir,
            {"successful_updates": 1, "input_fingerprint": "fixed"},
        )
        resumed.update(records, logical_step=2, total_tokens=4)
        assert resumed.policy_fingerprint() == continuous_fingerprint
    finally:
        resumed.close()
        guard_b.close_job(status="COMPLETED")


def test_t033_prepare_reuses_frozen_controls_without_new_calls(tmp_path, monkeypatch):
    import scripts.evaluation.run_qwen3_grpo_rollout as rollout

    task_ids = list(rollout.TASK_IDS)
    selected = []
    oracle = []

    def fake_task(instance, template):
        task_id = f"train-{instance}-{template}"
        public = SimpleNamespace(model_dump=lambda: {"task_id": task_id, "prompt": task_id})
        return SimpleNamespace(public=public, reference_sql=f"SELECT '{task_id}'")

    monkeypatch.setattr(rollout, "_selected_task", fake_task)
    for instance, template in rollout.TASK_ORDER:
        task_id = f"train-{instance}-{template}"
        selected.append(
            {"task_id": task_id, "public_task": {"task_id": task_id, "prompt": task_id}}
        )
        oracle.append({"task_id": task_id, "oracle_sha256": f"oracle-{task_id}"})
    run_dir = tmp_path / "run"
    panel_dir = run_dir / "panel" / "private"
    panel_dir.mkdir(parents=True)
    public_rows = [
        {"task_id": task_id, "public_task": {"task_id": task_id, "prompt": task_id}}
        for task_id in task_ids
    ]
    private_rows = [
        {
            "task_id": task_id,
            "reference_sql_sha256": hashlib.sha256(f"SELECT '{task_id}'".encode()).hexdigest(),
        }
        for task_id in task_ids
    ]
    control_rows = [
        {
            "task_id": task_id,
            "reference_sql_sha256": private["reference_sql_sha256"],
            "oracle_sha256": f"oracle-{task_id}",
            "policy_messages_read_reference_sql": False,
        }
        for task_id, private in zip(task_ids, private_rows, strict=True)
    ]
    for path, rows in (
        (run_dir / "panel" / "public-panel.jsonl", public_rows),
        (run_dir / "panel" / "private" / "verifier-inputs.jsonl", private_rows),
        (run_dir / "panel" / "private" / "cpu-controls.jsonl", control_rows),
    ):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    ledger = rollout._control_ledger_path(run_dir)
    for task_id in task_ids:
        for control_type in ("reference_sql", "python_oracle"):
            rollout._append_control_call(
                run_dir, {"control_type": control_type, "task_id": task_id}
            )
    panel = {"tasks": selected, "oracle_checks": oracle}
    first = rollout._load_frozen_panel({"run": run_dir}, panel)
    before = ledger.read_bytes()
    second = rollout._load_frozen_panel({"run": run_dir}, panel)
    assert first == second
    assert ledger.read_bytes() == before
    assert rollout._control_counts(run_dir) == {"reference_sql": 4, "python_oracle": 4, "total": 8}
