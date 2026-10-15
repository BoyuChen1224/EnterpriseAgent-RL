"""Shared contracts; policy-facing objects contain no verifier answers."""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Settings(Contract):
    seed: int = 17
    max_actions: int = Field(default=16, ge=2, le=100)
    max_tool_calls: int = Field(default=10, ge=1, le=50)
    episode_timeout_s: float = Field(default=30, gt=0, le=300)
    sql_timeout_s: float = Field(default=2, gt=0, le=10)
    sql_max_rows: int = Field(default=100, ge=1, le=1000)
    sql_max_bytes: int = Field(default=65536, ge=256, le=1048576)
    sql_memory_mb: int = Field(default=256, ge=128, le=1024)
    step_penalty: float = Field(default=0, ge=0, le=0.02)
    repeat_penalty: float = Field(default=0, ge=0, le=0.05)
    error_penalty: float = Field(default=0, ge=0, le=0.05)
    recovery_bonus: float = Field(default=0, ge=0, le=0.02)
    token_price_per_million: float = Field(default=0, ge=0)
    tool_price: float = Field(default=0, ge=0)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        return cls.model_validate(tomllib.loads(path.read_text()) if path else {})


class PublicTask(Contract):
    task_id: str
    prompt: str = Field(min_length=1, max_length=4000)


class Action(Contract):
    kind: Literal["plan", "schema", "dictionary", "sql", "check", "finish"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class Observation(Contract):
    step_index: int
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    terminated: bool = False
    reason: str | None = None


class Decision(Contract):
    action: Action
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    token_usage_known: bool = True


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
