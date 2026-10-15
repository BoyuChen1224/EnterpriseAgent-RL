"""CPU protocol/failure tests only: none of these results measure Qwen capability."""

import io
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_agent_rl.agent import LocalModelPolicy, run_episode
from enterprise_agent_rl.common import Observation, PublicTask, Settings
from enterprise_agent_rl.data import benchmark_tasks
from enterprise_agent_rl.evaluation import summarize
from enterprise_agent_rl.gpu_guard import require_compute_node
from enterprise_agent_rl.model_protocol import GenerationSettings, parse_action
from enterprise_agent_rl.model_server import QwenEngine, create_model_app
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import TrajectoryStore


@pytest.mark.parametrize(
    "raw,code",
    [
        ("", "empty_output"),
        ("Here is JSON: {}", "invalid_json"),
        ('{"kind":"delete","arguments":{}}', "invalid_action"),
        ('{"kind":"sql","arguments":{"query":42}}', "invalid_action_arguments"),
        ('{"kind":"plan","arguments":{"steps":42}}', "invalid_action_arguments"),
        (
            '{"kind":"check","arguments":{"evidence_step":true,"expected_columns":["id"]}}',
            "invalid_action_arguments",
        ),
        ('{"kind":"schema","arguments":{},"sql":"SELECT 1"}', "invalid_action"),
        ('{"kind":"schema"} {"kind":"finish"}', "invalid_json"),
    ],
)
def test_no_semantic_action_repair(raw, code):
    with pytest.raises(ValueError, match=code):
        parse_action(raw)


def test_only_complete_json_fences_can_be_normalized():
    raw = '```json\n{"kind":"schema","arguments":{}}\n```'
    action, normalized = parse_action(raw)
    assert action.kind == "schema" and normalized
    with pytest.raises(ValueError, match="invalid_json"):
        parse_action(raw, "strict")
    with pytest.raises(ValueError, match="invalid_json"):
        parse_action("reasoning\n" + raw)


def install_responses(monkeypatch, outputs):
    captured = []
    iterator = iter(outputs)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps(
                {
                    "choices": [{"message": {"content": next(iterator)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "telemetry": {"generation_latency_s": 0.5, "peak_allocated_bytes": 100},
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            captured.append((json.loads(request.data), timeout))
            return Response()

    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test-only-no-real-inference")
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    return captured


def test_retry_records_all_outputs_and_keeps_budget(monkeypatch):
    captured = install_responses(monkeypatch, ["not json", '{"kind":"schema","arguments":{}}'])
    policy = LocalModelPolicy("http://127.0.0.1:8001/v1/chat/completions", "qwen", "contract")
    result = policy.decide(
        PublicTask(task_id="p", prompt="公开问题"), Observation(step_index=-1, ok=True), [], 90
    )
    assert [a["raw_output"] for a in result.model_attempts] == [
        "not json",
        '{"kind":"schema","arguments":{}}',
    ]
    assert result.prompt_tokens == 20 and result.completion_tokens == 10
    assert captured[0][1] > 30 and captured[1][1] <= captured[0][1]
    assert captured[1][0]["messages"][-2]["content"] == "not json"
    assert "SELECT" not in captured[1][0]["messages"][-1]["content"]


def test_failed_generations_survive_without_tool_step(monkeypatch, database, tmp_path):
    install_responses(monkeypatch, ["", '{"kind":"bad"}'])
    policy = LocalModelPolicy("http://localhost:8001/v1/chat/completions", "qwen", "contract")
    task = benchmark_tasks()[0]
    trajectory = run_episode(task.public, policy, database, Settings())
    assert trajectory.terminal_reason == "invalid_action"
    assert not trajectory.steps and len(trajectory.model_attempts) == 2
    scored = trajectory.model_copy(update={"reward": verify(task, trajectory, database)})
    path = TrajectoryStore(tmp_path).save(scored)
    assert TrajectoryStore.load(path).model_attempts == trajectory.model_attempts
    metrics = summarize([scored], Settings())
    assert metrics["invalid_output_count"] == 2
    assert metrics["action_format_validity_rate"] == 0
    assert metrics["gpu_peak_allocated_bytes"] == 100
    assert metrics["generation_throughput_tokens_per_s"] == 10


@pytest.mark.parametrize("code", ["oom", "inference_timeout"])
def test_structured_service_errors(monkeypatch, database, code):
    monkeypatch.setenv("SLURM_JOB_ID", "contract-only")

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "failed",
                {},
                io.BytesIO(json.dumps({"error": {"code": code}}).encode()),
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    policy = LocalModelPolicy("http://localhost:8001/v1/chat/completions", "qwen", "contract")
    trajectory = run_episode(benchmark_tasks()[0].public, policy, database, Settings())
    assert trajectory.terminal_reason == code
    assert trajectory.model_attempts[0]["server_error"]["code"] == code


def test_compute_gate_before_model_import(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="slurm"):
        QwenEngine(None, tmp_path, tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "compute01")
    monkeypatch.setattr("socket.gethostname", lambda: "login01")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="compute01\n")
    )
    with pytest.raises(RuntimeError, match="host_not_in"):
        require_compute_node()


def test_service_schema_and_structured_error_passthrough():
    class ContractEngine:
        metadata = {"test_double": True}

        def generate(self, request):
            return {"error": {"code": "oom"}}

    with TestClient(create_model_app(ContractEngine())) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 503 and response.json()["error"]["code"] == "oom"
        assert (
            client.post(
                "/v1/chat/completions", json={"model": "qwen", "messages": [], "stream": True}
            ).status_code
            == 422
        )


def test_independent_inference_timeout(monkeypatch, database):
    captured = install_responses(monkeypatch, ['{"kind":"schema","arguments":{}}'])
    policy = LocalModelPolicy(
        "http://localhost:8001/v1/chat/completions",
        "qwen",
        "contract",
        GenerationSettings(format_retries=0),
    )
    trajectory = run_episode(
        benchmark_tasks()[0].public,
        policy,
        database,
        Settings(max_actions=2, episode_timeout_s=600, inference_timeout_s=75),
    )
    assert 74 <= captured[0][1] <= 75
    assert trajectory.steps[0].model_attempts[0]["raw_output"].startswith('{"kind"')


def test_server_cli_rejects_login_node_before_torch(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "enterprise_agent_rl.model_server",
            "--cache",
            str(tmp_path),
            "--output",
            str(tmp_path / "server"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "requires_slurm_compute_node" in result.stderr
    assert not (tmp_path / "server").exists()


def test_executed_action_must_match_saved_generation(monkeypatch, database):
    from pydantic import ValidationError

    from enterprise_agent_rl.trajectories import Trajectory

    install_responses(monkeypatch, ['{"kind":"schema","arguments":{}}'])
    policy = LocalModelPolicy("http://localhost:8001/v1/chat/completions", "qwen", "contract")
    trajectory = run_episode(benchmark_tasks()[0].public, policy, database, Settings(max_actions=2))
    data = trajectory.model_dump()
    data["steps"][0]["model_attempts"][-1]["raw_output"] = '{"kind":"dictionary","arguments":{}}'
    with pytest.raises(ValidationError, match="executed_action_differs"):
        Trajectory.model_validate(data)


def test_owned_model_process_group_is_reaped():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts/inference/run_qwen.py"
    spec = importlib.util.spec_from_file_location("qwen_supervisor", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        module.stop_model_process(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
