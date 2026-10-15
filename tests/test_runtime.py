import json

import pytest
from pydantic import ValidationError

from enterprise_agent_rl.agent import DeterministicPolicy, run_episode
from enterprise_agent_rl.common import Action, Decision, PublicTask, Settings
from enterprise_agent_rl.data import benchmark_tasks
from enterprise_agent_rl.environment import DataEnvironment
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import Trajectory, TrajectoryStore, export_jsonl, replay


@pytest.mark.parametrize("task", benchmark_tasks(), ids=lambda t: t.public.task_id)
def test_complete_multiturn_task(database, settings, task):
    trajectory = run_episode(task.public, DeterministicPolicy(), database, settings)
    assert trajectory.terminal_reason == "completed"
    assert len(trajectory.steps) >= 6
    assert trajectory.steps[0].action.kind == "plan"
    assert trajectory.steps[-1].action.kind == "finish"
    assert verify(task, trajectory, database).success
    assert trajectory.final_answer["evidence_step"] in [
        s.step_index for s in trajectory.steps if s.tool_name == "sql"
    ]
    if task.min_sql_calls == 2:
        assert sum(s.tool_name == "sql" for s in trajectory.steps) == 2


def test_recovery_and_no_reward_farming(database):
    task = benchmark_tasks()[0]
    cfg = Settings(step_penalty=0.01, error_penalty=0.02, recovery_bonus=0.02, repeat_penalty=0.02)
    good = run_episode(task.public, DeterministicPolicy(), database, cfg)
    recovered = run_episode(task.public, DeterministicPolicy(recover_error=True), database, cfg)
    assert verify(task, recovered, database).success
    assert verify(task, recovered, database).components["error_recovered"]
    assert verify(task, recovered, database).value <= verify(task, good, database).value
    assert any(not s.tool_result.ok for s in recovered.steps)


def test_budget_failure_has_no_success_reward(database):
    task = benchmark_tasks()[0]
    cfg = Settings(max_tool_calls=1)
    trajectory = run_episode(task.public, DeterministicPolicy(), database, cfg)
    assert trajectory.terminal_reason == "tool_budget_exhausted"
    assert trajectory.final_answer is None
    assert verify(task, trajectory, database).value == 0
    short = run_episode(task.public, DeterministicPolicy(), database, Settings(max_actions=2))
    assert short.terminal_reason == "action_budget_exhausted"


def test_grounded_finish_reset_and_truncation(database, settings):
    task = benchmark_tasks()[0].public
    env = DataEnvironment(database, settings)
    with pytest.raises(RuntimeError):
        env.step(Action(kind="schema"))
    env.reset(task)
    assert not env.step(Action(kind="finish", arguments={"evidence_step": 0, "summary": "fake"})).ok
    assert not env.step(Action(kind="sql", arguments={"query": 42})).ok
    result = env.step(
        Action(kind="sql", arguments={"query": "SELECT id FROM customers ORDER BY id"})
    )
    idx = result.step_index
    assert not env.step(
        Action(kind="check", arguments={"evidence_step": idx, "expected_columns": ["wrong"]})
    ).ok
    assert env.step(
        Action(kind="check", arguments={"evidence_step": idx, "expected_columns": ["id"]})
    ).ok
    end = env.step(Action(kind="finish", arguments={"evidence_step": idx, "summary": "done"}))
    assert end.terminated and end.result["rows"][0] == [1]
    with pytest.raises(RuntimeError):
        env.step(Action(kind="schema"))
    env.reset(task)
    assert env.sql_results == {} and not env.checked
    tiny = DataEnvironment(database, Settings(sql_max_rows=1))
    tiny.reset(task)
    result = tiny.step(Action(kind="sql", arguments={"query": "SELECT id FROM customers"}))
    assert not tiny.step(
        Action(
            kind="check", arguments={"evidence_step": result.step_index, "expected_columns": ["id"]}
        )
    ).ok


def test_wrong_but_executable_result_gets_zero(database, settings):
    task = benchmark_tasks()[0]

    class WrongPolicy:
        kind = "deterministic_fixture"
        version = "wrong-test-only"

        def decide(self, task, observation, history, timeout_s):
            index = len(history)
            actions = [
                Action(kind="sql", arguments={"query": "SELECT 0 AS region,0 AS revenue_cents"}),
                Action(
                    kind="check",
                    arguments={"evidence_step": 0, "expected_columns": ["region", "revenue_cents"]},
                ),
                Action(kind="finish", arguments={"evidence_step": 0, "summary": "done"}),
            ]
            return Decision(action=actions[index])

    trajectory = run_episode(task.public, WrongPolicy(), database, settings)
    reward = verify(task, trajectory, database)
    assert reward.value == 0 and not reward.success
    assert reward.components["sql_execution_rate"] == 1


def test_trajectory_roundtrip_replay_and_tamper(database, tmp_path, settings):
    task = benchmark_tasks()[0]
    trajectory = run_episode(
        task.public, DeterministicPolicy(recover_error=True), database, settings
    )
    store = TrajectoryStore(tmp_path / "traces")
    path = store.save(trajectory)
    restored = store.load(path)
    assert restored == trajectory
    assert replay(restored, database)["matched"]
    export_jsonl([restored], tmp_path / "episodes.jsonl")
    assert Trajectory.model_validate_json((tmp_path / "episodes.jsonl").read_text()) == restored
    tampered = json.loads(path.read_text())
    tampered["steps"][0]["step_index"] = 9
    with pytest.raises(ValidationError):
        Trajectory.model_validate(tampered)
    tampered = json.loads(path.read_text())
    tampered["schema_version"] = "99"
    with pytest.raises(ValidationError):
        Trajectory.model_validate(tampered)
    tampered = json.loads(path.read_text())
    tampered["final_answer"]["rows"] = [[123]]
    with pytest.raises(ValidationError):
        Trajectory.model_validate(tampered)
    with database.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        replay(restored, database)


def test_unsupported_prompt_and_episode_timeout(database, settings):
    trajectory = run_episode(
        PublicTask(task_id="unknown", prompt="unknown task"),
        DeterministicPolicy(),
        database,
        settings,
    )
    assert trajectory.terminal_reason.startswith("policy_or_runtime_error")
    env = DataEnvironment(database, Settings(episode_timeout_s=0.001))
    env.reset(benchmark_tasks()[0].public)
    env.started -= 10
    assert env.step(Action(kind="schema")).reason == "episode_timeout"


def test_task_verifier_mismatch_rejected(database, settings):
    tasks = benchmark_tasks()
    trajectory = run_episode(tasks[0].public, DeterministicPolicy(), database, settings)
    with pytest.raises(ValueError, match="task_mismatch"):
        verify(tasks[1], trajectory, database)


def test_repeated_final_query_cannot_fake_required_business_probe(database, settings):
    task = next(t for t in benchmark_tasks() if t.public.task_id == "top-2025")

    class RepeatPolicy:
        kind = "deterministic_fixture"
        version = "adversarial-repeat-test"

        def decide(self, task_public, observation, history, timeout_s):
            actions = [
                Action(kind="sql", arguments={"query": task.reference_sql}),
                Action(kind="sql", arguments={"query": task.reference_sql}),
                Action(
                    kind="check",
                    arguments={
                        "evidence_step": 1,
                        "expected_columns": ["customer", "revenue_cents"],
                    },
                ),
                Action(kind="finish", arguments={"evidence_step": 1, "summary": "done"}),
            ]
            return Decision(action=actions[len(history)])

    trajectory = run_episode(task.public, RepeatPolicy(), database, settings)
    reward = verify(task, trajectory, database)
    assert reward.components["final_correct"]
    assert not reward.components["business_probe_verified"]
    assert reward.value == 0 and not reward.success
