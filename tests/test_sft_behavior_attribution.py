"""CPU-only tests for the teacher-forced behavior attribution diagnostic."""

from __future__ import annotations

from enterprise_agent_rl.model_protocol import GenerationSettings
from scripts.diagnostics.sft_behavior_attribution import (
    ACTION_KINDS,
    PARSE_MODE,
    build_targets,
    classify_output,
    cpu_smoke,
    fixed_generation_evidence,
    metrics_for,
)


def _row() -> dict:
    return {
        "schema_version": "1.0",
        "example_id": "example-1",
        "task_id": "train-0-schema",
        "assistant_only_loss": True,
        "source": {
            "source_split": "train",
            "trajectory_source": "deterministic_fixture_expert",
            "template_id": "schema",
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": '{"kind":"schema","arguments":{}}'},
            {"role": "user", "content": '{"ok":true}'},
            {
                "role": "assistant",
                "content": '{"kind":"finish","arguments":{"evidence_step":0,"summary":"ok"}}',
            },
            {"role": "user", "content": '{"ok":true}'},
        ],
        "assistant_message_indices": [2, 4],
    }


def test_target_history_excludes_current_gold_message():
    targets = build_targets([_row()])
    assert len(targets) == 2
    assert targets[0].history[-1]["content"] == "question"
    assert all(
        target.raw_gold_action not in [message["content"] for message in target.history]
        for target in targets
    )
    assert targets[1].history[-1]["content"] == '{"ok":true}'


def test_raw_json_and_runtime_parser_are_reported_separately():
    accepted = classify_output(
        raw_output='```json\n{"kind":"schema","arguments":{}}\n```',
        response_error=None,
        finish_reason="stop",
    )
    assert accepted["raw_json_valid"] is False
    assert accepted["strict_action_envelope_valid"] is False
    assert accepted["runtime_parser_accepted"] is True
    assert accepted["parser_normalized"] is True


def test_invalid_envelope_and_arguments_are_distinct():
    envelope = classify_output(
        raw_output='{"kind":"unknown","arguments":{}}',
        response_error=None,
        finish_reason="stop",
    )
    arguments = classify_output(
        raw_output='{"kind":"sql","arguments":{"query":42}}',
        response_error=None,
        finish_reason="stop",
    )
    assert envelope["invalid_envelope"] is True
    assert arguments["invalid_arguments"] is True
    assert not envelope["runtime_parser_accepted"]
    assert not arguments["runtime_parser_accepted"]


def test_tool_argument_denominator_and_full_match():
    parsed = classify_output(
        raw_output='{"kind":"schema","arguments":{}}',
        response_error=None,
        finish_reason="stop",
    )
    record = {
        "gold_action": {"kind": "schema", "arguments": {}},
        "gold_kind": "schema",
        "teacher_source": "deterministic_fixture_expert",
        "template_id": "schema",
        "raw_output": '{"kind":"schema","arguments":{}}',
        "input_tokens": 3,
        "output_tokens": 4,
        **parsed,
    }
    metrics = metrics_for([record])
    assert metrics["tool_target_count"] == 1
    assert metrics["tool_argument_exact_match"]["denominator"] == 1
    assert metrics["full_action_exact_match"]["count"] == 1


def test_cpu_smoke_is_weight_free():
    result = cpu_smoke()
    assert result["status"] == "passed"
    assert result["model_weights_loaded"] is False
    assert result["gpu_used"] is False


def test_fixed_generation_is_greedy_and_retry_free():
    class Config:
        attention_backend = "sdpa_math"
        generation = GenerationSettings()

    evidence = fixed_generation_evidence(Config())
    assert evidence["requested"]["temperature"] == 0
    assert evidence["requested"]["top_p"] == 1
    assert evidence["requested"]["max_tokens"] == 512
    assert evidence["requested"]["format_retries"] == 0
    assert evidence["effective_transformers_generation"]["do_sample"] is False
    assert evidence["effective_transformers_generation"]["top_p_applied"] is False
    assert PARSE_MODE == "json_fence"
    assert ACTION_KINDS == ("plan", "schema", "dictionary", "sql", "check", "finish")
