"""Auditable data formatting helpers for the controlled SFT pilot.

The repository intentionally keeps the training implementation small and
dependency-light.  This module owns the part that must be identical between
the CPU preflight and the GPU trainer: dataset gates, chat-template rendering,
and assistant-only label spans.  It does not import torch or load a model.
"""

# SFT 输入和 token 监督范围的公共实现，尽量不在导入时加载 GPU 依赖。
# 旧 load_dataset_rows 对应固定 20 条整轨迹；load_decision_rows 对应固定 54 条逐决策输入。
# 先确认文本、来源和模板，再构造 labels；不能把旧模型的 token IDs 直接用于 Qwen3。

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from enterprise_agent_rl.common import Action
from enterprise_agent_rl.model_protocol import parse_action, render_chat_template


class ChatTokenizer(Protocol):
    """Small tokenizer surface used by the CPU preflight and unit tests."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


class SFTDataError(ValueError):
    """Raised when a dataset cannot be safely used for assistant-only SFT."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or (value and isinstance(value[0], list)):
        raise SFTDataError("chat_template_tokenization_must_return_one_dimensional_ids")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SFTDataError("chat_template_tokenization_returned_non_integer_ids") from exc


# 装载历史整轨迹数据，校验元数据、train 来源和固定样本分布；不是任意 JSONL 的通用加载器。
def load_dataset_rows(
    dataset_path: Path, metadata_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the frozen train-only JSONL and its metadata."""

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source_split") != "train":
        raise SFTDataError("metadata_source_split_not_train")
    if metadata.get("validation_used_for_training") is not False:
        raise SFTDataError("metadata_validation_used_for_training_not_false")
    if metadata.get("test_used_for_training") is not False:
        raise SFTDataError("metadata_test_used_for_training_not_false")
    if metadata.get("final_decision") != "READY_FOR_SFT_TRAINING":
        raise SFTDataError("metadata_final_decision_not_ready")

    rows: list[dict[str, Any]] = []
    with dataset_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTDataError(f"invalid_jsonl:{line_number}") from exc
            if not isinstance(row, dict):
                raise SFTDataError(f"dataset_row_not_object:{line_number}")
            rows.append(row)

    if len(rows) != 20:
        raise SFTDataError(f"dataset_row_count:{len(rows)}")
    if metadata.get("trajectory_count") != len(rows):
        raise SFTDataError("metadata_trajectory_count_mismatch")

    task_ids: set[str] = set()
    for row in rows:
        source = row.get("source")
        if not isinstance(source, dict):
            raise SFTDataError("row_source_missing")
        if source.get("source_split") != "train":
            raise SFTDataError("row_source_split_not_train")
        if source.get("validation_used_for_training") is True:
            raise SFTDataError("row_validation_used_for_training")
        if source.get("test_used_for_training") is True:
            raise SFTDataError("row_test_used_for_training")
        if _contains_key(row, "reference_sql"):
            raise SFTDataError("reference_sql_leakage")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id.startswith("train-"):
            raise SFTDataError("non_train_task_id")
        if task_id in task_ids:
            raise SFTDataError("duplicate_task_id")
        task_ids.add(task_id)

        messages = row.get("messages")
        indices = row.get("assistant_message_indices")
        if not isinstance(messages, list) or not messages:
            raise SFTDataError("messages_missing")
        if not isinstance(indices, list) or not indices:
            raise SFTDataError("assistant_message_indices_missing")
        if row.get("assistant_only_loss") is not True:
            raise SFTDataError("assistant_only_loss_not_enabled")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(messages):
                raise SFTDataError("assistant_message_index_invalid")
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise SFTDataError("assistant_target_is_not_assistant")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise SFTDataError("empty_assistant_target")
            try:
                parse_action(content)
            except ValueError as exc:
                raise SFTDataError("invalid_assistant_action") from exc

    source_counts = Counter(row["source"]["trajectory_source"] for row in rows)
    expected_sources = {"deterministic_fixture_expert": 6, "reference_sql_teacher": 14}
    if dict(source_counts) != expected_sources:
        raise SFTDataError(f"teacher_source_distribution:{dict(source_counts)}")
    return rows, metadata


def _chat_ids(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> list[int]:
    return _token_list(
        render_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    )


# 根据当前 tokenizer 自带的 chat template 定位 assistant 内容与结束符。
# 返回 input_ids 和 labels；标签为 -100 的位置仅充当上下文，不进入语言模型交叉熵。
def format_sft_example(
    tokenizer: ChatTokenizer,
    example: dict[str, Any],
    *,
    max_length: int | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Render one example and return labels with exact assistant-only spans.

    The span boundaries are derived from the tokenizer's own chat template.
    Prefix equality checks intentionally fail closed: if tokenization of a
    prefix cannot be proven to align with the full conversation, no GPU
    training is allowed to proceed.
    """

    messages = example["messages"]
    indices = example["assistant_message_indices"]
    full_ids = _chat_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not full_ids:
        raise SFTDataError(f"empty_rendered_example:{example.get('example_id')}")

    spans: list[tuple[int, int]] = []
    previous_end = 0
    for index in indices:
        prefix_ids = _chat_ids(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        end_ids = _chat_ids(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        # 模板前缀可能受分词边界影响，必须与完整序列逐 token 对齐，不能凭字符偏移猜标签位置。
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise SFTDataError(f"assistant_prefix_alignment_failed:{example.get('example_id')}")
        if full_ids[: len(end_ids)] != end_ids:
            raise SFTDataError(f"assistant_end_alignment_failed:{example.get('example_id')}")
        start, end = len(prefix_ids), len(end_ids)
        if end <= start or start < previous_end:
            raise SFTDataError(f"empty_or_overlapping_assistant_span:{example.get('example_id')}")
        spans.append((start, end))
        previous_end = end

    # 超长直接拒绝，不截掉工具反馈或目标动作后仍声称该样本训练完成。
    if max_length is not None and len(full_ids) > max_length:
        raise SFTDataError(
            f"sequence_exceeds_context_without_truncation:{example.get('example_id')}"
        )
    # 先全部屏蔽，再只打开审核过的 assistant 区间；工具反馈和 system/user 文本不作为目标。
    labels = [-100] * len(full_ids)
    for start, end in spans:
        labels[start:end] = full_ids[start:end]
    supervised = sum(label != -100 for label in labels)
    if supervised <= 0:
        raise SFTDataError(f"empty_supervised_target:{example.get('example_id')}")

    return {
        "example_id": example.get("example_id"),
        "task_id": example.get("task_id"),
        "input_ids": full_ids,
        "labels": labels,
        "assistant_spans": [list(span) for span in spans],
        "full_sequence_tokens": len(full_ids),
        "supervised_assistant_tokens": supervised,
        "masked_context_tokens": len(full_ids) - supervised,
    }


# 在训练前检查每条的实际监督范围和上下文长度，而不只相信 assistant_only_loss 标记。
def audit_tokenized_dataset(
    tokenizer: ChatTokenizer,
    rows: list[dict[str, Any]],
    *,
    max_context_tokens: int,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Tokenize every complete trajectory without truncation and summarize it."""

    formatted = [
        format_sft_example(
            tokenizer,
            row,
            max_length=max_context_tokens,
            enable_thinking=enable_thinking,
        )
        for row in rows
    ]
    lengths = [item["full_sequence_tokens"] for item in formatted]
    supervised = [item["supervised_assistant_tokens"] for item in formatted]

    def percentile(values: list[int], fraction: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "schema_version": "1.0",
        "sample_count": len(formatted),
        "max_context_tokens": max_context_tokens,
        "truncated_sample_count": 0,
        "empty_target_count": 0,
        "invalid_action_count": 0,
        "full_sequence_tokens": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "values": lengths,
        },
        "supervised_assistant_tokens": {
            "min": min(supervised),
            "max": max(supervised),
            "mean": sum(supervised) / len(supervised),
            "values": supervised,
        },
        "assistant_message_count_distribution": dict(
            sorted(Counter(len(row["assistant_message_indices"]) for row in rows).items())
        ),
        "examples": [
            {
                "example_id": item["example_id"],
                "task_id": item["task_id"],
                "full_sequence_tokens": item["full_sequence_tokens"],
                "supervised_assistant_tokens": item["supervised_assistant_tokens"],
                "masked_context_tokens": item["masked_context_tokens"],
                "assistant_spans": item["assistant_spans"],
            }
            for item in formatted
        ],
    }


# 只提取冻结数据的公开历史与当前目标 Action，并验证 train 身份和未来结果缺失。
# 旧 input_ids、labels、token span 不作为当前模型的训练数据使用。
def load_decision_rows(dataset_path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    """Load the frozen train-only decision rows without using old token IDs.

    The source file contains historical Qwen2.5 tokenization beside the
    public context and current teacher Action.  This loader intentionally
    consumes only the latter two fields and fails closed on future results or
    verifier-only answers.
    """

    actual_sha256 = sha256_file(dataset_path)
    if actual_sha256 != expected_sha256:
        raise SFTDataError(f"decision_dataset_sha256_mismatch:{actual_sha256}")
    rows: list[dict[str, Any]] = []
    with dataset_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTDataError(f"invalid_decision_jsonl:{line_number}") from exc
            if not isinstance(row, dict):
                raise SFTDataError(f"decision_row_not_object:{line_number}")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.startswith("train-"):
                raise SFTDataError("decision_row_not_train")
            context = row.get("context_messages")
            if not isinstance(context, list) or not context:
                raise SFTDataError("decision_context_missing")
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in context
            ):
                raise SFTDataError("decision_context_invalid")
            if context[-1].get("role") != "user":
                raise SFTDataError("decision_context_must_end_with_user")
            if _contains_key(row, "reference_sql") or _contains_key(row, "expected_result"):
                raise SFTDataError("decision_verifier_data_leakage")
            if row.get("future_result_absent") is not True:
                raise SFTDataError("decision_future_result_not_absent")
            try:
                action = Action.model_validate(row.get("target_action"))
            except Exception as exc:
                raise SFTDataError("decision_target_action_invalid") from exc
            target_content = json.dumps(
                action.model_dump(), ensure_ascii=False, separators=(",", ":")
            )
            if not target_content.strip():
                raise SFTDataError("decision_target_content_empty")
            rows.append(
                {
                    "example_id": row.get("example_id"),
                    "task_id": task_id,
                    "step_index": row.get("step_index"),
                    "context_messages": context,
                    "target_action": action.model_dump(),
                    "target_content": target_content,
                    "source_policy_kind": row.get("source_policy_kind"),
                    "teacher_trajectory_sha256": row.get("teacher_trajectory_sha256"),
                    "source_row_sha256": hashlib.sha256(line.strip().encode()).hexdigest(),
                }
            )
    if len(rows) != 54:
        raise SFTDataError(f"decision_row_count:{len(rows)}")
    return rows


# 将一个决策点转为标准 SFT 样本：已有上下文 + 最后一个 assistant 目标。
# 过去的 assistant 动作虽在上下文中，本样本仅监督最后这个决策。
def decision_example(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one assistant-only example from public context and Action."""

    messages = list(row["context_messages"]) + [
        {"role": "assistant", "content": row["target_content"]}
    ]
    return {
        "example_id": row["example_id"],
        "task_id": row["task_id"],
        "step_index": row["step_index"],
        "messages": messages,
        "assistant_message_indices": [len(messages) - 1],
        "assistant_only_loss": True,
        "source_policy_kind": row.get("source_policy_kind"),
        "teacher_trajectory_sha256": row.get("teacher_trajectory_sha256"),
        "source_row_sha256": row.get("source_row_sha256"),
        "target_action": row["target_action"],
    }


# 最小可行性 smoke 的有限选样入口；正式 Qwen3 配方由 qwen3_sft 保留全部 54 条。
def select_decision_examples(
    tokenizer: ChatTokenizer,
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    max_context_tokens: int,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort, re-render, and select the first bounded Qwen3 examples."""

    if max_samples <= 0:
        raise SFTDataError("decision_max_samples_invalid")
    ordered = sorted(
        rows,
        key=lambda row: (str(row["task_id"]), int(row["step_index"]), str(row["example_id"])),
    )
    selected: list[dict[str, Any]] = []
    formatted: list[dict[str, Any]] = []
    for row in ordered:
        example = decision_example(row)
        try:
            item = format_sft_example(
                tokenizer,
                example,
                max_length=max_context_tokens,
                enable_thinking=enable_thinking,
            )
        except SFTDataError as exc:
            if str(exc).startswith("sequence_exceeds_context_without_truncation"):
                continue
            raise
        selected.append(example)
        formatted.append(item)
        if len(selected) == max_samples:
            break
    if len(selected) != max_samples:
        raise SFTDataError(f"decision_bounded_examples_insufficient:{len(selected)}:{max_samples}")
    return selected, formatted


def yaml_scalar(value: Any) -> str:
    """Serialize the small, fixed config shape without a PyYAML dependency."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#{}[],&*?|<>!=%@`\"'\n"):
        return json.dumps(text, ensure_ascii=False)
    return text


def yaml_dump(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict | list):
                lines.append(f"{prefix}{key}:")
                lines.append(yaml_dump(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                lines.append(f"{prefix}-")
                lines.append(yaml_dump(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{yaml_scalar(value)}")
    return "\n".join(lines) + "\n"
