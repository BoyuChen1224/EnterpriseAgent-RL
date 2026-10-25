"""CPU-only safety tests for the train-only Runtime protocol ablation."""

from __future__ import annotations

import json

import pytest

from enterprise_agent_rl.common import Observation, Settings
from enterprise_agent_rl.sft_training import load_dataset_rows
from scripts.diagnostics.sft_runtime_protocol_ablation import (
    GROUPS,
    ProtocolAblationPolicy,
    SequenceMockBackend,
    _aggregate,
    build_request_messages,
    canonical_json,
    compare_group_matrix,
    load_train_only_cases,
    run_cpu_mock_checks,
    run_protocol_self_checks,
)


@pytest.fixture(scope="module")
def real_cases(train_protocol_data):
    return load_train_only_cases(**train_protocol_data)[0]


def test_streaming_loader_is_train_only(real_cases):
    assert len(real_cases) == 20
    assert {case.database_id for case in real_cases} == {"train-0", "train-1"}
    assert {case.template_id for case in real_cases} == {
        "category",
        "constraints",
        "customer",
        "empty",
        "inactive",
        "product",
        "recovery",
        "region",
        "schema",
        "top",
    }
    assert all(case.public_task.task_id.startswith("train-") for case in real_cases)


def test_protocol_matrix_changes_only_three_factors():
    evidence = compare_group_matrix()
    assert set(evidence["allowed_factors"]) == {
        "retain_initial_observation",
        "canonical_history",
        "format_retries",
    }
    assert all(
        set(item["changed_fields"]).issubset(set(evidence["allowed_factors"]))
        for item in evidence["pairwise"].values()
    )


def test_initial_observation_is_retained_once_in_correct_position(real_cases):
    case = next(case for case in real_cases if case.template_id == "schema")
    initial = Observation(
        step_index=-1,
        ok=True,
        result={"task": case.public_task.model_dump(), "seed": 17},
    )
    messages, _ = build_request_messages(
        task=case.public_task,
        initial_observation=initial,
        current_observation=initial,
        history=[],
        group=GROUPS[1],
    )
    initial_text = initial.model_dump_json()
    assert messages[2] == {"role": "user", "content": initial_text}
    assert sum(message.get("content") == initial_text for message in messages) == 1


def test_canonical_serializer_matches_training_formatter(real_cases, train_protocol_data):
    case = next(case for case in real_cases if case.template_id == "schema")
    rows, _ = load_dataset_rows(
        train_protocol_data["dataset_path"],
        train_protocol_data["metadata_path"],
    )
    row = next(row for row in rows if row["task_id"] == case.public_task.task_id)
    action_message = next(
        row["messages"][index]
        for index in row["assistant_message_indices"]
        if json.loads(row["messages"][index]["content"])["kind"] == "schema"
    )
    action = json.loads(action_message["content"])
    assert canonical_json(action) == action_message["content"]


def test_no_format_retry_makes_one_model_call(real_cases):
    case = next(case for case in real_cases if case.template_id == "schema")
    initial = Observation(step_index=-1, ok=True, result={"task": case.public_task.model_dump()})
    backend = SequenceMockBackend(["not json", canonical_json({"kind": "schema", "arguments": {}})])
    policy = ProtocolAblationPolicy(backend, GROUPS[3])
    with pytest.raises(Exception, match="invalid_json"):
        policy.decide(case.public_task, initial, [], 10)
    assert len(backend.calls) == 1
    assert len(policy.request_log) == 1


def test_public_request_guard_excludes_verifier_fields(real_cases):
    case = next(case for case in real_cases if case.template_id == "schema")
    initial = Observation(step_index=-1, ok=True, result={"task": case.public_task.model_dump()})
    backend = SequenceMockBackend([canonical_json({"kind": "schema", "arguments": {}})])
    policy = ProtocolAblationPolicy(
        backend,
        GROUPS[0],
        reference_sql_guard=case.verifier_task.reference_sql,
    )
    policy.decide(case.public_task, initial, [], 10)
    for request in policy.request_log:
        for message in request["messages"]:
            if message["role"] in {"system", "user"}:
                assert "reference_sql" not in message["content"]
                assert case.verifier_task.reference_sql not in message["content"]


def test_protocol_self_checks_and_mock_episode(real_cases, tmp_path):
    settings = Settings(
        max_actions=20,
        max_tool_calls=14,
        episode_timeout_s=600,
        inference_timeout_s=120,
    )
    assert run_protocol_self_checks(real_cases, settings)["status"] == "passed"
    result = run_cpu_mock_checks(
        cases=real_cases,
        settings=settings,
        output_dir=tmp_path / "cpu-mock",
    )
    assert result["status"] == "passed"
    assert set(result["groups"]) == {group.name for group in GROUPS}


def test_aggregate_does_not_recurse_forever():
    record = {
        "success": False,
        "parser_attempts": 1,
        "parser_acceptances": 1,
        "first_action_legal": True,
        "schema_dictionary_before_first_sql": False,
        "sql_attempts": 0,
        "executable_sql_count": 0,
        "sql_error_count": 0,
        "recovery_count": 0,
        "recovery_success_count": 0,
        "repeated_sql": 0,
        "check_present": False,
        "finish_present": False,
        "budget_exhausted": False,
        "tool_calls": 0,
        "latency_s": 0,
        "input_tokens": 1,
        "output_tokens": 1,
        "reward": 0,
        "invalid_json": 0,
        "invalid_envelope": 0,
        "invalid_arguments": 0,
        "format_retry_count": 0,
        "action_kind_counts": {},
        "sql_error_types": {},
        "teacher_provenance": "mock",
    }
    result = _aggregate([record])
    assert result["teacher_provenance"]["mock"]["task_count"] == 1
