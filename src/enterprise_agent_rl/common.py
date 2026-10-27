"""Shared contracts; policy-facing objects contain no verifier answers."""

# 阅读入口：这里定义模型、Runtime 和轨迹共同使用的数据契约。
# 数据流是 PublicTask → Decision(Action) → Observation → 下一次 Decision。
# 完整 BenchmarkTask 含评分专用信息；业务 Policy 只能接收这里的 PublicTask。

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# 统一拒绝未知字段并冻结字段赋值，尽早发现接口拼写错误；这不等于深度冻结嵌套容器。
class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# 单次 episode 的执行与资源限制。动作数、工具数、SQL 时限和 episode 时限是不同预算。
# 惩罚/奖励系数是可配置项；判断实验是否启用它们，要查看实际运行配置。
class Settings(Contract):
    seed: int = 17
    # max_actions 包含 plan/finish；max_tool_calls 只限制环境工具分支，二者会在不同位置耗尽。
    max_actions: int = Field(default=16, ge=2, le=100)
    max_tool_calls: int = Field(default=10, ge=1, le=50)
    episode_timeout_s: float = Field(default=30, gt=0, le=7200)
    inference_timeout_s: float = Field(default=120, gt=0, le=1800)
    sql_timeout_s: float = Field(default=2, gt=0, le=10)
    sql_max_rows: int = Field(default=100, ge=1, le=1000)
    sql_max_bytes: int = Field(default=65536, ge=256, le=1048576)
    sql_memory_mb: int = Field(default=256, ge=128, le=1024)
    step_penalty: float = Field(default=0, ge=0, le=0.02)
    repeat_penalty: float = Field(default=0, ge=0, le=0.05)
    error_penalty: float = Field(default=0, ge=0, le=0.05)
    recovery_bonus: float = Field(default=0, ge=0, le=0.02)
    # 价格只是调用方提供的估算参数；未提供真实计费来源时，零配置不代表集群计算免费。
    token_price_per_million: float = Field(default=0, ge=0)
    tool_price: float = Field(default=0, ge=0)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        return cls.model_validate(tomllib.loads(path.read_text()) if path else {})


# 只放模型可以看到的任务标识和问题正文，不携带参考 SQL、期望行或隐藏验收条件。
class PublicTask(Contract):
    task_id: str
    prompt: str = Field(min_length=1, max_length=4000)


# 统一动作外壳。kind 限制动作种类，各种 arguments 的具体约束由解析器和环境继续检查。
class Action(Contract):
    kind: Literal["plan", "schema", "dictionary", "sql", "check", "finish"]
    arguments: dict[str, Any] = Field(default_factory=dict)


# 环境返回的状态：ok 表示这一步是否执行成功，terminated 表示 episode 是否结束。
# SQL 执行成功不代表答案正确，completed 也仍需由独立 Verifier 评分。
class Observation(Contract):
    step_index: int
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    terminated: bool = False
    reason: str | None = None


# 一次逻辑决策只交付一个 Action，但 model_attempts 可包含多次物理生成和格式重试。
# 调用次数与 token 用量要累计全部尝试；缺少 usage 时不凭空填成零。
class Decision(Contract):
    action: Action
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    token_usage_known: bool = True
    model_attempts: list[dict[str, Any]] = Field(default_factory=list)


def configure_logging() -> None:
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps(
                {"level": record.levelname, "message": record.getMessage(), "logger": record.name},
                ensure_ascii=False,
            )

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
