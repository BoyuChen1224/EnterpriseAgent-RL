import json

import pytest
from fastapi.testclient import TestClient

from enterprise_agent_rl.agent import DeterministicPolicy, LocalModelPolicy, run_episode
from enterprise_agent_rl.common import Action, Observation, PublicTask
from enterprise_agent_rl.data import benchmark_tasks
from enterprise_agent_rl.evaluation import compare_reports, evaluate, summarize
from enterprise_agent_rl.serving import JobStore, create_app


def test_evaluation_reports_and_comparability(database, tmp_path, settings):
    tasks = [t for t in benchmark_tasks() if t.split == "test"]
    regular = evaluate(DeterministicPolicy(), tasks, database, settings, tmp_path / "regular")
    recovery = evaluate(
        DeterministicPolicy(recover_error=True), tasks, database, settings, tmp_path / "recovery"
    )
    assert regular["metrics"]["task_success_rate"] == 1
    assert regular["metrics"]["executable_sql_rate"] == 1
    assert regular["metrics"]["llm_capability_result"] is False
    assert regular["metrics"]["average_tokens"] == 0
    assert recovery["metrics"]["executable_sql_rate"] < 1
    assert recovery["metrics"]["average_tool_calls"] > regular["metrics"]["average_tool_calls"]
    assert len(compare_reports([regular, recovery])["results"]) == 2
    different = json.loads(json.dumps(regular))
    different["task_ids"] = ["different"]
    with pytest.raises(ValueError):
        compare_reports([regular, different])
    with pytest.raises(ValueError):
        compare_reports([regular])
    with pytest.raises(ValueError):
        summarize([], settings)
    unscored = run_episode(tasks[0].public, DeterministicPolicy(), database, settings)
    with pytest.raises(ValueError):
        summarize([unscored], settings)


def test_api_end_to_end_and_persistence(database, tmp_path):
    output = tmp_path / "api"
    task = benchmark_tasks()[0].public
    with TestClient(create_app(database, output)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/docs").status_code == 200
        assert "/tasks" in client.get("/openapi.json").json()["paths"]
        assert all("reference_sql" not in t for t in client.get("/benchmarks").json()["tasks"])
        assert len(client.get("/models").json()["models"]) == 2
        response = client.post(
            "/tasks", json={"prompt": task.prompt, "benchmark_task_id": task.task_id}
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        status = client.get(f"/tasks/{job_id}").json()
        assert status["status"] == "completed" and status["reward"]["success"]
        trajectory = client.get(f"/tasks/{job_id}/trajectory").json()
        assert trajectory["final_answer"]["rows"]
        assert trajectory["policy_kind"] == "deterministic_fixture"
        assert client.get("/tasks/nonexistent").status_code == 404
        assert (
            client.post(
                "/tasks", json={"prompt": task.prompt, "model_version": "imaginary-sft"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/tasks", json={"prompt": "wrong prompt", "benchmark_task_id": task.task_id}
            ).status_code
            == 422
        )
        assert client.post("/tasks", json={"prompt": ""}).status_code == 422
        compare = client.post(
            "/comparisons",
            json={
                "prompt": task.prompt,
                "model_versions": ["fixture-v1", "fixture-v1-recovery"],
                "benchmark_task_id": task.task_id,
            },
        )
        assert compare.status_code == 202
        assert all(
            client.get(f"/tasks/{j['id']}").json()["reward"]["success"]
            for j in compare.json()["tasks"]
        )
        assert (
            client.post(
                "/comparisons",
                json={"prompt": task.prompt, "model_versions": ["fixture-v1", "fixture-v1"]},
            ).status_code
            == 422
        )
        custom = client.post("/tasks", json={"prompt": task.prompt}).json()
        custom_status = client.get(f"/tasks/{custom['id']}").json()
        assert custom_status["reward"] is None
        assert custom_status["evaluation_status"] == "unscored_custom_prompt"
        failed = client.post("/tasks", json={"prompt": "unsupported question"}).json()
        assert client.get(f"/tasks/{failed['id']}").json()["status"] == "failed"
    with TestClient(create_app(database, output)) as client:
        assert client.get(f"/tasks/{job_id}").json()["reward"]["success"]


def test_interrupted_jobs_are_explicitly_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.put({"id": "interrupted", "status": "running"})
    restarted = JobStore(tmp_path / "jobs.sqlite3")
    assert restarted.get("interrupted")["error"] == "server_restarted"


def test_real_model_adapter_slurm_gate_and_contract(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="slurm"):
        LocalModelPolicy("http://127.0.0.1:8001/v1/chat/completions", "qwen", "base")
    monkeypatch.setenv("SLURM_JOB_ID", "contract-test-not-real-inference")
    with pytest.raises(ValueError, match="loopback"):
        LocalModelPolicy("http://example.com/v1/chat/completions", "qwen", "base")
    policy = LocalModelPolicy("http://127.0.0.1:8001/v1/chat/completions", "qwen", "base")
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps(
                {
                    "choices": [{"message": {"content": Action(kind="schema").model_dump_json()}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 8},
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    decision = policy.decide(
        PublicTask(task_id="public", prompt="统计收入"), Observation(step_index=-1, ok=True), [], 1
    )
    assert decision.action.kind == "schema"
    assert decision.prompt_tokens == 30 and decision.completion_tokens == 8
    assert decision.token_usage_known
    assert "reference_sql" not in json.dumps(captured)


def test_comparison_capacity_failure_releases_all_slots(database, tmp_path, monkeypatch):
    import threading

    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(
        "enterprise_agent_rl.serving.threading.BoundedSemaphore", lambda n: semaphore
    )
    task = benchmark_tasks()[0].public
    with TestClient(create_app(database, tmp_path / "limited")) as client:
        response = client.post(
            "/comparisons",
            json={"prompt": task.prompt, "model_versions": ["fixture-v1", "fixture-v1-recovery"]},
        )
        assert response.status_code == 429
        # A rejected batch must leave capacity for the next single task.
        assert client.post("/tasks", json={"prompt": task.prompt}).status_code == 202


def test_failed_local_usage_stays_unknown(database, settings, tmp_path):
    from enterprise_agent_rl.rewards import verify

    class FailedLocalPolicy:
        version = "failed-local-contract"
        kind = "local_llm"

        def decide(self, *args):
            raise TimeoutError("contract test: no model invoked")

    task = benchmark_tasks()[0]
    trajectory = run_episode(task.public, FailedLocalPolicy(), database, settings)
    trajectory = trajectory.model_copy(update={"reward": verify(task, trajectory, database)})
    result = summarize([trajectory], settings)
    assert result["average_tokens"] is None
    assert result["average_end_to_end_cost"] is None
