"""CPU-only contracts for the T-031 Qwen3 pilot; no capability result is inferred."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from enterprise_agent_rl.agent import LocalModelPolicy, run_episode
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks
from enterprise_agent_rl.lora import inject_lora_modules, trainable_lora_parameters
from enterprise_agent_rl.model_protocol import GenerationSettings, render_chat_template
from enterprise_agent_rl.model_server import ModelSettings
from enterprise_agent_rl.sft_training import format_sft_example
from enterprise_agent_rl.trajectories import TrajectoryStore


class _HardOffTokenizer:
    def __init__(self):
        self.flags = []

    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking=None
    ):
        self.flags.append(enable_thinking)
        tokens = []
        for message in conversation:
            tokens.extend([f"<{message['role']}>", message["content"], "</message>"])
        if add_generation_prompt:
            tokens.append("<assistant>")
        return list(range(len(tokens))) if tokenize else " ".join(tokens)


def test_qwen3_config_and_hard_off_template_are_explicit():
    config = ModelSettings.load(Path("configs/model/qwen3-8b.toml"))
    assert config.model_id == "Qwen/Qwen3-8B"
    assert config.generation.enable_thinking is False
    tokenizer = _HardOffTokenizer()
    example = {
        "example_id": "qwen3-cpu",
        "task_id": "train-0-schema",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"kind":"schema","arguments":{}}'},
        ],
        "assistant_message_indices": [2],
        "assistant_only_loss": True,
    }
    formatted = format_sft_example(tokenizer, example, enable_thinking=False)
    assert formatted["assistant_spans"]
    assert tokenizer.flags and all(flag is False for flag in tokenizer.flags)


def test_old_qwen25_default_omits_new_template_keyword():
    tokenizer = _HardOffTokenizer()
    render_chat_template(
        tokenizer,
        [{"role": "user", "content": "old-default"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert tokenizer.flags == [None]
    assert GenerationSettings().enable_thinking is None


def test_qwen3_http_rollout_persists_cross_task_requests(tmp_path, monkeypatch):
    events = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib handler contract
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            events.append(payload)
            assistant_count = sum(message["role"] == "assistant" for message in payload["messages"])
            actions = [
                {"kind": "plan", "arguments": {"steps": ["query"]}},
                {"kind": "schema", "arguments": {}},
                {"kind": "dictionary", "arguments": {}},
                {"kind": "sql", "arguments": {"query": "SELECT 1 AS value"}},
                {
                    "kind": "check",
                    "arguments": {"evidence_step": 3, "expected_columns": ["value"]},
                },
                {
                    "kind": "finish",
                    "arguments": {"evidence_step": 3, "summary": "done"},
                },
            ]
            content = json.dumps(actions[assistant_count], ensure_ascii=False)
            response = {
                "id": f"fake-event-{len(events)}",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "telemetry": {
                    "event_id": f"fake-event-{len(events)}",
                    "generation_latency_s": 0.001,
                    "peak_allocated_bytes": 1,
                },
            }
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SLURM_JOB_ID", "qwen3-cpu-http")
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        policy = LocalModelPolicy(
            endpoint,
            "Qwen/Qwen3-8B",
            "qwen3-cpu-http",
            GenerationSettings(enable_thinking=False, format_retries=0),
        )
        db_path = tmp_path / "business.sqlite3"
        from enterprise_agent_rl.data import initialize_database

        initialize_database(db_path)
        trajectories = []
        for task in benchmark_tasks()[:2]:
            trajectory = run_episode(
                task.public,
                policy,
                db_path,
                Settings(max_actions=20, max_tool_calls=14),
                tmp_path / "sql-audit.jsonl",
            )
            trajectories.append(trajectory)
            TrajectoryStore(tmp_path / "trajectories").save(trajectory)
            assert trajectory.terminal_reason == "completed"
            assert all(attempt.get("engine_event_id") for attempt in trajectory.model_attempts)
        assert len(events) == 12
        assert len({payload["messages"][1]["content"] for payload in events}) == 2
        (tmp_path / "responses.jsonl").write_text(
            "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in events)
        )
        assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 12
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_tiny_random_qwen3_lora_freezes_base_and_has_gradient():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("installed Transformers has no Qwen3 implementation")
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = transformers.Qwen3ForCausalLM(config)
    modules = inject_lora_modules(
        model,
        rank=8,
        alpha=16,
        dropout=0,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    assert len(modules) == 14
    trainable = trainable_lora_parameters(model)
    input_ids = torch.randint(0, 128, (1, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
