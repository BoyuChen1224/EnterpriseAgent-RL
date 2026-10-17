"""CPU-only tests for the SFT dataset gate and assistant-only mask."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_agent_rl.lora import lora_parameter_names, trainable_lora_parameters
from enterprise_agent_rl.sft_training import (
    SFTDataError,
    audit_tokenized_dataset,
    format_sft_example,
    load_dataset_rows,
)


class FakeParameter:
    def __init__(self, count: int):
        self._count = count
        self.requires_grad = True

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self

    def numel(self):
        return self._count


class FakeLoRAModel:
    def __init__(self):
        self.parameters_by_name = [
            ("model.layers.0.q_proj.base.weight", FakeParameter(32)),
            ("model.layers.0.q_proj.lora_A", FakeParameter(8)),
            ("model.layers.0.q_proj.lora_B", FakeParameter(8)),
        ]

    def named_parameters(self):
        return iter(self.parameters_by_name)


class FakeTokenizer:
    """A deterministic chat-template tokenizer for mask-boundary tests."""

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        rendered = []
        for message in conversation:
            rendered.extend(
                [f"<{message['role']}>", message["content"], "</message>"]
            )
        if add_generation_prompt:
            rendered.extend(["<assistant>"])
        if tokenize:
            return list(range(len(rendered)))
        return " ".join(rendered)


def _example():
    return {
        "example_id": "train-0:test",
        "task_id": "train-0-test",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"kind":"plan","arguments":{}}'},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": '{"kind":"finish","arguments":{}}'},
        ],
        "assistant_message_indices": [2, 4],
        "assistant_only_loss": True,
    }


def test_lora_parameter_filter_freezes_base_and_keeps_only_manual_lora():
    model = FakeLoRAModel()

    trainable = trainable_lora_parameters(model)

    assert len(trainable) == 2
    assert lora_parameter_names(model) == [
        "model.layers.0.q_proj.lora_A",
        "model.layers.0.q_proj.lora_B",
    ]
    assert model.parameters_by_name[0][1].requires_grad is False
    assert all(parameter.requires_grad for parameter in trainable)


def test_mask_targets_only_assistant_spans():
    formatted = format_sft_example(FakeTokenizer(), _example())
    assert formatted["full_sequence_tokens"] == 15
    assert formatted["assistant_spans"] == [[7, 9], [13, 15]]
    assert all(label == -100 for label in formatted["labels"][:7])
    assert formatted["labels"][7:9] == [7, 8]
    assert all(label == -100 for label in formatted["labels"][9:13])
    assert formatted["labels"][13:15] == [13, 14]


def test_mask_refuses_truncation():
    with pytest.raises(SFTDataError, match="sequence_exceeds_context"):
        format_sft_example(FakeTokenizer(), _example(), max_length=14)


def test_audit_reports_complete_targets():
    report = audit_tokenized_dataset(FakeTokenizer(), [_example()], max_context_tokens=64)
    assert report["sample_count"] == 1
    assert report["truncated_sample_count"] == 0
    assert report["empty_target_count"] == 0
    assert report["supervised_assistant_tokens"]["values"] == [4]


def test_dataset_gate_rejects_validation_metadata(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(_example()) + "\n")
    metadata = tmp_path / "dataset-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "source_split": "train",
                "validation_used_for_training": True,
                "test_used_for_training": False,
                "final_decision": "READY_FOR_SFT_TRAINING",
                "trajectory_count": 1,
            }
        )
    )
    with pytest.raises(SFTDataError, match="validation_used"):
        load_dataset_rows(dataset, metadata)
