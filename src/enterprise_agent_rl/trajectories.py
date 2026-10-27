"""Versioned, validated, atomic episode storage and action replay."""

# 轨迹是运行、训练和评测共享的事实记录，不只是最终回答。
# 一条 StepRecord 同时保存执行前观察、模型动作和执行后观察；Trajectory 将它们串成一条链。
# 回放会重新执行已保存的动作，验证环境行为；它不会重新调用语言模型。

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from enterprise_agent_rl.common import Action, Contract, Observation, PublicTask, Settings
from enterprise_agent_rl.environment import DataEnvironment
from enterprise_agent_rl.model_protocol import parse_action


# 一个逻辑动作对应一条记录，多次模型格式重试保存在该步的 model_attempts 中。
class StepRecord(Contract):
    step_index: int = Field(ge=0)
    observation: Observation
    action: Action
    tool_name: str | None
    tool_arguments: dict
    tool_result: Observation
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    token_usage_known: bool
    model_attempts: list[dict] = Field(default_factory=list)
    elapsed_s: float = Field(ge=0)


class RewardResult(Contract):
    value: float
    success: bool
    reason: str
    components: dict[str, float | bool]


# 只包含公开任务和真实交互。reward 可在独立评分后补入，不能用缺省零代替未评分。
class Trajectory(Contract):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    runtime_errors: list[dict] = Field(default_factory=list)
    policy_metadata: dict = Field(default_factory=dict)
    run_id: str
    task: PublicTask
    policy_version: str
    policy_kind: Literal[
        "deterministic_fixture",
        "local_llm",
        "programmatic_reference_sql_teacher",
    ]
    database_sha256: str
    settings: Settings
    initial_observation: Observation
    steps: list[StepRecord]
    final_answer: dict | None
    terminal_reason: str
    reward: RewardResult | None = None
    model_attempts: list[dict] = Field(default_factory=list)
    elapsed_s: float = Field(ge=0)

    # 检查步骤连续、前后观察衔接、动作参数一致，并校验 LLM 原始输出与执行 Action 对应。
    # completed 的最终答案必须就是终态观察中的执行结果，不能另造一份最终表格。
    @model_validator(mode="after")
    def validate_chain(self) -> Trajectory:
        previous = self.initial_observation
        for index, step in enumerate(self.steps):
            if step.step_index != index or step.tool_result.step_index != index:
                raise ValueError("non_contiguous_steps")
            if step.observation != previous or previous.terminated:
                raise ValueError("broken_observation_chain")
            if step.tool_arguments != step.action.arguments:
                raise ValueError("tool_arguments_mismatch")
            expected_tool = step.action.kind if step.action.kind not in ("plan", "finish") else None
            if step.tool_name != expected_tool:
                raise ValueError("tool_name_mismatch")
            previous = step.tool_result
        if self.schema_version == "1.1" and self.policy_kind == "local_llm":
            recorded = []
            for step in self.steps:
                if not step.model_attempts:
                    raise ValueError("local_step_requires_actual_generation")
                parsed, _ = parse_action(step.model_attempts[-1]["raw_output"])
                if parsed != step.action:
                    raise ValueError("executed_action_differs_from_model_output")
                recorded.extend(step.model_attempts)
            if self.model_attempts[: len(recorded)] != recorded:
                raise ValueError("generation_history_mismatch")
        if self.terminal_reason == "completed":
            if not self.steps or not previous.terminated or previous.reason != "completed":
                raise ValueError("completion_requires_terminal_step")
            if self.final_answer != previous.result:
                raise ValueError("final_answer_not_grounded")
        elif self.final_answer is not None:
            raise ValueError("failed_episode_has_final_answer")
        return self


# 这里计算数据库文件的物理 SHA-256，不是忽略文件布局的逻辑内容指纹。
def database_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrajectoryStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    # 先重新验证轨迹，再用同目录临时文件、fsync 和原子替换写出，避免留下半份 JSON。
    # 原子写入不表示不可覆盖：相同 run_id 会替换同名文件，调用方需维护运行目录纪律。
    def save(self, trajectory: Trajectory) -> Path:
        # Revalidate even if a caller constructed an object without validation.
        validated = Trajectory.model_validate_json(trajectory.model_dump_json())
        if not validated.run_id or any(c not in "0123456789abcdef-" for c in validated.run_id):
            raise ValueError("invalid_run_id")
        path = self.directory / f"{validated.run_id}.json"
        fd, temporary = tempfile.mkstemp(dir=self.directory, prefix=".trajectory-")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(validated.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @staticmethod
    def load(path: Path) -> Trajectory:
        return Trajectory.model_validate_json(path.read_text())


# 先核对数据库字节指纹，再按原配置重执行 Action 并逐步比较观察。
# 超时等结果可能依赖机器，遇到差异直接报告具体步骤，不宣称所有环境都逐位可复现。
def replay(trajectory: Trajectory, database: Path) -> dict:
    if database_fingerprint(database) != trajectory.database_sha256:
        raise ValueError("database_fingerprint_mismatch")
    env = DataEnvironment(database, trajectory.settings)
    if env.reset(trajectory.task) != trajectory.initial_observation:
        raise ValueError("initial_observation_mismatch")
    for record in trajectory.steps:
        actual = env.step(record.action)
        # Timing/resource termination is machine-dependent; report differences explicitly.
        if actual != record.tool_result:
            return {
                "matched": False,
                "mismatch_step": record.step_index,
                "expected": record.tool_result.model_dump(),
                "actual": actual.model_dump(),
            }
    return {
        "matched": True,
        "steps": len(trajectory.steps),
        "terminal_reason": trajectory.terminal_reason,
    }


def export_jsonl(trajectories: list[Trajectory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for trajectory in trajectories:
            stream.write(json.dumps(trajectory.model_dump(), ensure_ascii=False) + "\n")
