"""CPU-only contracts for T-032/E-029 implementation paths.

These tests use a synthetic tokenizer/model or temporary benchmark database;
they do not load the real Qwen3 weights and do not claim a capability result.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_agent_rl.lora import inject_lora_modules
from enterprise_agent_rl.qwen3_sft import (
    DATASET_SHA256,
    build_full_tokenized_dataset,
    make_schedule,
    train_lora_schedule,
    write_tokenized_artifacts,
)
from enterprise_agent_rl.sft_training import load_decision_rows


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _HardOffTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self):
        self.flags: list[bool | None] = []

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=None,
    ):
        self.flags.append(enable_thinking)
        pieces = []
        for message in conversation:
            pieces.extend([f"<{message['role']}>", message["content"], "</message>"])
        if add_generation_prompt:
            pieces.append("<assistant>")
        return list(range(len(pieces))) if tokenize else " ".join(pieces)

    def save_pretrained(self, _path):
        return None


def test_full_source_is_rerendered_and_schedule_is_fixed(tmp_path: Path):
    source = Path(
        "artifacts/sft-runtime-aligned-decision-pilot/t017-r2-v1/decision/tokenized-dataset.jsonl"
    )
    rows = load_decision_rows(source, expected_sha256=DATASET_SHA256)
    tokenizer = _HardOffTokenizer()
    examples, formatted, identity = build_full_tokenized_dataset(
        tokenizer, rows, max_context_tokens=2048, enable_thinking=False
    )
    assert len(examples) == len(formatted) == 54
    assert identity["old_input_ids_used"] is False
    assert identity["old_labels_used"] is False
    assert all(flag is False for flag in tokenizer.flags)
    assert all(item["assistant_spans"][0][1] == len(item["input_ids"]) for item in formatted)
    assert all(
        all(label == -100 for label in item["labels"][: item["assistant_spans"][0][0]])
        for item in formatted
    )
    schedule = make_schedule(examples, epochs=2, accumulation_steps=4)
    assert len(schedule) == 27
    assert sum(len(group) for group in schedule) == 108
    assert any(group[0]["epoch"] != group[-1]["epoch"] for group in schedule)
    manifest = write_tokenized_artifacts(
        tmp_path / "training",
        source_path=source,
        source_sha256=DATASET_SHA256,
        examples=examples,
        formatted=formatted,
        config_identity=identity,
    )
    assert manifest["source_row_count"] == 54
    saved = [
        json.loads(line)
        for line in (tmp_path / "training" / "tokenized-dataset.jsonl").read_text().splitlines()
    ]
    assert len(saved) == 54
    assert all("reference_sql" not in row for row in saved)


class _TinyCausalLM:
    """Small torch model with one target Linear for schedule tests."""

    def __init__(self, torch):
        self.torch = torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Module()
                self.layers.q_proj = nn.Linear(8, 8)
                self.lm_head = nn.Linear(8, 16)

            def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
                del attention_mask
                x = torch.nn.functional.one_hot(input_ids % 8, num_classes=8).float()
                x = self.layers.q_proj(x)
                logits = self.lm_head(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100
                )
                return SimpleNamespace(logits=logits, loss=loss)

        self.model = Model()


def test_native_amp_skip_replays_same_group_without_advancing_cursor():
    torch = pytest.importorskip("torch")
    tiny = _TinyCausalLM(torch)
    inject_lora_modules(tiny.model, rank=2, alpha=4, dropout=0, target_modules=("q_proj",))
    examples = []
    generator = torch.Generator().manual_seed(7)
    for index in range(54):
        ids = torch.randint(0, 16, (5,), generator=generator).tolist()
        examples.append(
            {
                "example_id": f"tiny-{index:02}",
                "input_ids": ids,
                "labels": [-100] + ids[1:],
            }
        )
    target = next(
        parameter for name, parameter in tiny.model.named_parameters() if name.endswith("lora_B")
    )
    inject = {"enabled": True}

    def inject_once(_model, _backward_index, group_index):
        if group_index == 0 and inject["enabled"]:
            inject["enabled"] = False

    hook = target.register_hook(
        lambda gradient: gradient * float("inf") if inject["enabled"] else gradient
    )
    result = train_lora_schedule(
        tiny.model,
        examples,
        torch_module=torch,
        device="cpu",
        seed=42,
        microbatch_size=1,
        accumulation_steps=4,
        epochs=2,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
        clip_norm=1.0,
        amp_initial_scale=1024.0,
        max_backward=216,
        max_replays_per_group=3,
        gradient_injector=inject_once,
        amp_enabled=True,
    )
    hook.remove()
    assert result["successful_optimizer_updates"] == 27
    assert result["microbatches_processed"] == 108
    assert result["backward_count"] == 112
    assert result["skipped_optimizer_updates"] == 1
    assert result["replayed_groups"] == 1
    replay = result["replay_metrics"][0]
    assert replay["cursor_before"] == replay["cursor_after"] == 0
    assert replay["example_ids"] == [f"tiny-{index:02}" for index in range(4)]
    assert replay["native_scaler_step_called"] is True
    assert replay["parameters_changed"] is False
    assert replay["optimizer_state_changed"] is False
    assert result["base_parameters_changed"] is False
    assert result["base_gradients_present"] == []
    assert result["lora_changed_parameter_count"] > 0
    assert result["overall_lora_gradient_norm_positive"] is True


def test_validation_prepare_separates_public_and_private_verifier(tmp_path: Path):
    panel = _load_script("qwen3_runtime_panel", "scripts/evaluation/run_qwen3_runtime_panel.py")
    manifest = panel.prepare_validation(
        tmp_path, agent_config=Path("configs/agent/qwen3-base-panel.toml")
    )
    assert manifest["planned"] == 20
    assert manifest["task_templates"] == [
        "schema",
        "region",
        "top",
        "customer",
        "product",
        "empty",
        "constraints",
        "recovery",
        "status",
        "monthly",
    ]
    public_text = (tmp_path / "evaluation/validation/public-panel.jsonl").read_text()
    private_text = (tmp_path / "evaluation/validation/private/verifier-inputs.jsonl").read_text()
    assert "reference_sql" not in public_text
    assert "reference_sql" in private_text
    assert "expected" not in public_text
    assert manifest["test_payload_accessed"] is False


def test_engine_event_persistence_is_unique_and_append_only(tmp_path: Path):
    panel = _load_script(
        "qwen3_runtime_panel_events", "scripts/evaluation/run_qwen3_runtime_panel.py"
    )
    engine = tmp_path / "engine"
    response = tmp_path / "response"
    engine.mkdir()
    first = {"id": "event-1", "request": {"messages": []}, "usage": {}}
    (engine / "generations.jsonl").write_text(json.dumps(first) + "\n")
    assert len(panel.persist_engine_events(engine, response)) == 1
    assert (response / "request-event-1.json").is_file()
    second = {"id": "event-2", "request": {"messages": []}, "usage": {}}
    with (engine / "generations.jsonl").open("a") as stream:
        stream.write(json.dumps(second) + "\n")
    assert len(panel.persist_engine_events(engine, response)) == 2
    with (engine / "generations.jsonl").open("a") as stream:
        stream.write(json.dumps(first) + "\n")
    with pytest.raises(panel.PanelError, match="not_unique"):
        panel.persist_engine_events(engine, response)
