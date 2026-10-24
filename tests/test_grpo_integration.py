from __future__ import annotations

from pathlib import Path

from enterprise_agent_rl.grpo.budget import BudgetGuard
from enterprise_agent_rl.grpo.config import RunContext, load_config
from enterprise_agent_rl.grpo.contracts import ScoreResult, TaskInput
from enterprise_agent_rl.grpo.io import atomic_write_json, read_json, read_jsonl
from enterprise_agent_rl.grpo.orchestration import run_online_training
from enterprise_agent_rl.grpo.rollout import OnlineRollout
from enterprise_agent_rl.grpo.testing import FakeRolloutEngine, FakeWorker, TinyPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "training" / "grpo-online.toml"


def _tasks(count: int = 4) -> list[TaskInput]:
    return [
        TaskInput(
            task_id=f"integration-task-{index}",
            group_id=f"integration-group-{index}",
            public_task={"task_id": f"integration-task-{index}", "prompt": "synthetic"},
        )
        for index in range(count)
    ]


def test_public_orchestration_runs_two_rounds_with_continuous_state(tmp_path):
    run_dir = tmp_path / "run"
    config, _ = load_config(CONFIG_PATH, RunContext(REPO_ROOT, run_dir, CONFIG_PATH))
    captured: dict[str, object] = {}

    def factory(guard, context, _config):
        policy = TinyPolicy(context.run_dir, guard)
        engine = FakeRolloutEngine()
        worker = FakeWorker()
        captured.update(policy=policy, engine=engine, worker=worker)
        return policy, engine, worker

    result = run_online_training(
        RunContext(REPO_ROOT, run_dir, CONFIG_PATH),
        config,
        _tasks(),
        component_factory=factory,
        job_id="INTEGRATION-CPU",
    )
    assert result["status"] == "COMPLETED"
    assert result["updates"] == 4
    assert len(captured["engine"].calls) == 32

    r0_snapshot = read_json(run_dir / "rounds" / "r0" / "policy-snapshot.json")
    r1_snapshot = read_json(run_dir / "rounds" / "r1" / "policy-snapshot.json")
    final_snapshot = read_json(run_dir / "training" / "final-step4.json")["snapshot"]
    assert r0_snapshot["successful_updates"] == 0
    assert r1_snapshot["successful_updates"] == 2
    assert r0_snapshot["delta_fingerprint"] != r1_snapshot["delta_fingerprint"]
    assert final_snapshot["successful_updates"] == 4
    assert final_snapshot["delta_fingerprint"] != r1_snapshot["delta_fingerprint"]

    all_updates = []
    for round_index in range(2):
        dataset = read_json(
            run_dir / "rounds" / f"r{round_index}" / "training" / "dataset-index.json"
        )
        assert dataset["counts"]["planned_episodes"] == 16
        assert dataset["counts"]["scored_episodes"] == 16
        assert dataset["counts"]["unknown_episodes"] == 0
        token_rows = read_jsonl(
            run_dir / "rounds" / f"r{round_index}" / "rollouts" / "token-rollouts.jsonl"
        )
        assert len(token_rows) == 16
        assert all(row["old_per_token_logps"] == row["raw_policy_log_probs"] for row in token_rows)
        assert all(
            row["policy_snapshot_id"]
            == (r0_snapshot if round_index == 0 else r1_snapshot)["policy_snapshot_id"]
            for row in token_rows
        )
        metrics = read_json(run_dir / "rounds" / f"r{round_index}" / "training" / "metrics.json")
        all_updates.extend(metrics["updates"])
    assert [row["logical_step"] for row in all_updates] == [1, 2, 3, 4]
    assert all(
        row["before_state"]["optimizer"] != row["after_state"]["optimizer"] for row in all_updates
    )
    assert all(
        row["before_state"]["scheduler"] != row["after_state"]["scheduler"] for row in all_updates
    )
    assert all("scaler" in row["after_state"] for row in all_updates)

    episodes = read_jsonl(run_dir / "rounds" / "r0" / "rollouts" / "episodes.jsonl")
    scores = read_jsonl(run_dir / "rounds" / "r0" / "rollouts" / "formal-scores.jsonl")
    failed_episode = next(row for row in episodes if row["sample_index"] == 3)
    failed_score = next(row for row in scores if row["episode_id"].endswith("-s3"))
    business_failure_score = next(row for row in scores if row["episode_id"].endswith("-s1"))
    assert failed_episode["failed_action"]["status"] == "format_failed"
    assert failed_score["status"] == "scored"
    assert business_failure_score["status"] == "scored"
    assert business_failure_score["success"] is False
    budget = read_json(run_dir / "budget-state.json")
    assert budget["active_attempt"] is None
    assert budget["status"] == "COMPLETED"

    resumed_capture: dict[str, object] = {}

    def resumed_factory(guard, context, _config):
        policy = TinyPolicy(context.run_dir, guard)
        engine = FakeRolloutEngine()
        worker = FakeWorker()
        resumed_capture.update(policy=policy, engine=engine, worker=worker)
        return policy, engine, worker

    resumed_result = run_online_training(
        RunContext(REPO_ROOT, run_dir, CONFIG_PATH),
        config,
        _tasks(),
        component_factory=resumed_factory,
        job_id="INTEGRATION-CPU-RESUME",
    )
    assert resumed_result["status"] == "COMPLETED"
    assert resumed_result["updates"] == 4
    assert resumed_capture["engine"].calls == []
    assert len(read_json(run_dir / "budget-state.json")["attempts"]) == 2


def test_rollout_resume_uses_persisted_episode_and_zero_advantage_is_valid(tmp_path):
    run_dir = tmp_path / "run"
    clock_guard = BudgetGuard(
        run_dir / "budget-state.json",
        run_id="rollout-test",
        run_hard_seconds=1000,
        job_hard_seconds=900,
        soft_seconds=800,
        cleanup_reserve_seconds=60,
    )
    clock_guard.start_job(job_id="ROLLOUT")
    rollout = OnlineRollout(run_dir=run_dir, guard=clock_guard, seed_start=74)
    task = _tasks(1)[0]
    snapshot = {"policy_snapshot_id": "snapshot-0", "model_ref_id": "fake"}
    first_engine = FakeRolloutEngine()

    class EqualWorker(FakeWorker):
        def score(self, task, episode):
            self.calls.append(episode.episode_id)
            return ScoreResult(
                episode_id=episode.episode_id,
                task_id=task.task_id,
                status="scored",
                reward_value=1.0,
                success=True,
                source_kind="cpu_fake",
            )

    worker = EqualWorker()
    first = rollout.run_round(
        round_index=0,
        tasks=[task],
        samples_per_task=4,
        policy_snapshot=snapshot,
        engine=first_engine,
        worker=worker,
    )
    assert first["status"] == "COMPLETED"
    assert first["dataset"]["counts"]["zero_advantage_physical_requests"] == 4

    score_path = run_dir / "rounds" / "r0" / "rollouts" / "formal-scores.jsonl"
    score_path.write_text(
        "\n".join(score_path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8"
    )
    marker_path = run_dir / "rounds" / "r0" / "rollout-status.json"
    marker = read_json(marker_path)
    marker["status"] = "INCOMPLETE"
    atomic_write_json(marker_path, marker)
    second_engine = FakeRolloutEngine()
    resumed = rollout.run_round(
        round_index=0,
        tasks=[task],
        samples_per_task=4,
        policy_snapshot=snapshot,
        engine=second_engine,
        worker=worker,
    )
    assert resumed["status"] == "COMPLETED"
    assert second_engine.calls == []
    clock_guard.close_job(status="COMPLETED")
