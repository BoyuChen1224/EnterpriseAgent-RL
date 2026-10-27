"""Immutable service profiles and preflight checks for local model serving.

The serving profile is deliberately separate from the model and runtime
implementations.  It is the small contract that binds a model endpoint to a
specific, already-validated asset and to the public benchmark subset exposed
by the Agent API.  The profile never serializes verifier-only task fields.
"""

# 把模型服务、任务 API 与固定资产绑定起来，避免启动了错误 Adapter 却只检查端口存活。
# 当前正式 profile 是 Qwen2.5-1.5B 的 sft-n-step27；不能当成通用 Qwen3 部署配置。
# HTTP health、持久 model-load、任务结果中的模型身份必须相互一致。

from __future__ import annotations

import hashlib
import json
import socket
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field

from enterprise_agent_rl.benchmark import database_content_fingerprint
from enterprise_agent_rl.common import Contract, Settings
from enterprise_agent_rl.data import BenchmarkTask
from enterprise_agent_rl.model_protocol import GenerationSettings

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
N_ADAPTER_SHA256 = "e67372df6d805d4ece08e6e698ccbe76d73cc8f83fe96b12f6c7939028f088b9"
N_ADAPTER_CONFIG_SHA256 = "bae7f0af3379a71ae24b0386131d1728eea4ed48fef8c7a0f3ba22780ce17c22"
TRAIN_RECORDS_SHA256 = "88259942516ea57396bb0f17660804cab945e3e8210e36bb6da15f0686c151dc"
TRAIN_DATABASE_PHYSICAL_SHA256 = "99e56b8f6511219eaecf19b50668ae02effb03d2533737d8dd3da4f7d65451bf"
TRAIN_DATABASE_CONTENT_SHA256 = "01b0da1528d8cef072cf10f801039dd4a5d3849744bf2f69b485734d018ad4eb"
TRAIN_TASK_IDS = ("train-1-schema", "train-1-empty")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _require_loopback(url: str, label: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(f"{label}_must_be_loopback_http")
    if not parsed.port:
        raise ValueError(f"{label}_port_required")


# 保存固定模型/Adapter、数据库、生成设置、Runtime 及本地端点，路径变更不能绕过身份校验。
class ServiceProfile(Contract):
    """Resolved, reproducible settings for one local Agent service."""

    profile_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    model_version: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    model_id: str = MODEL_ID
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_config_path: Path
    cache_dir: Path
    adapter_dir: Path
    benchmark_records_path: Path
    database_path: Path
    output_dir: Path
    model_completion_endpoint: str
    model_health_endpoint: str
    agent_endpoint: str
    adapter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    benchmark_records_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    database_physical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    database_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    benchmark_task_ids: list[str] = Field(min_length=1)
    runtime: Settings = Field(default_factory=Settings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    model_load_timeout_s: float = Field(default=300, gt=0, le=1800)
    health_timeout_s: float = Field(default=10, gt=0, le=120)

    @classmethod
    def load(cls, path: Path, *, repo_root: Path | None = None) -> ServiceProfile:
        """Load a TOML profile and resolve every path against an explicit root."""

        profile_path = path.resolve()
        values = tomllib.loads(profile_path.read_text())
        root = (repo_root or Path.cwd()).resolve()
        raw_runtime = values.get("runtime", {})
        raw_generation = values.get("generation", {})
        paths = values.get("paths", {})
        endpoints = values.get("endpoints", {})
        identity = values.get("identity", {})
        service = values.get("service", {})
        return cls(
            profile_id=values["profile_id"],
            model_version=values["model_version"],
            model_id=values["model_id"],
            revision=values["revision"],
            model_config_path=_resolve_path(root, paths["model_config"]),
            cache_dir=_resolve_path(root, paths["cache"]),
            adapter_dir=_resolve_path(root, paths["adapter"]),
            benchmark_records_path=_resolve_path(root, paths["benchmark_records"]),
            database_path=_resolve_path(root, paths["database"]),
            output_dir=_resolve_path(root, paths["output"]),
            model_completion_endpoint=endpoints["model_completion"],
            model_health_endpoint=endpoints["model_health"],
            agent_endpoint=endpoints["agent"],
            adapter_sha256=identity["adapter_sha256"],
            adapter_config_sha256=identity["adapter_config_sha256"],
            benchmark_records_sha256=identity["benchmark_records_sha256"],
            database_physical_sha256=identity["database_physical_sha256"],
            database_content_sha256=identity["database_content_sha256"],
            benchmark_task_ids=list(identity["benchmark_task_ids"]),
            runtime=raw_runtime,
            generation=raw_generation,
            model_load_timeout_s=service.get("model_load_timeout_s", 300),
            health_timeout_s=service.get("health_timeout_s", 10),
        )

    def with_paths(
        self, *, database: Path | None = None, output: Path | None = None
    ) -> ServiceProfile:
        """Return a profile with run-local writable paths while preserving identity."""

        return self.model_copy(
            update={
                "database_path": database.resolve() if database is not None else self.database_path,
                "output_dir": output.resolve() if output is not None else self.output_dir,
            }
        )

    def with_endpoints(
        self,
        *,
        model_completion: str | None = None,
        model_health: str | None = None,
        agent: str | None = None,
    ) -> ServiceProfile:
        """Return a profile with loopback transport endpoints changed only."""

        return self.model_copy(
            update={
                "model_completion_endpoint": model_completion or self.model_completion_endpoint,
                "model_health_endpoint": model_health or self.model_health_endpoint,
                "agent_endpoint": agent or self.agent_endpoint,
            }
        )

    def identity_summary(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "adapter_sha256": self.adapter_sha256,
            "adapter_config_sha256": self.adapter_config_sha256,
            "benchmark_task_ids": list(self.benchmark_task_ids),
            "runtime": self.runtime.model_dump(),
            "generation": self.generation.model_dump(),
        }

    def expected_endpoint_identity(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "adapter_sha256": self.adapter_sha256,
            "adapter_config_sha256": self.adapter_config_sha256,
        }

    # 不加载权重、不生成文本的资产检查；按开关核对数据库和 benchmark 记录。
    # 模型 ID、revision、权重/config SHA、缓存和任务范围都要匹配，不能仅验证文件存在。
    def preflight(
        self,
        *,
        check_database: bool = True,
        check_benchmark_records: bool = True,
    ) -> dict[str, Any]:
        """Validate immutable assets without loading a model or generating text."""

        if self.model_id != MODEL_ID:
            raise RuntimeError("unsupported_profile_model_id")
        if self.revision != MODEL_REVISION:
            raise RuntimeError("unsupported_profile_model_revision")
        if self.profile_id == "sft-n-step27":
            fixed_identity = {
                "model_version": "sft-n-step27",
                "adapter_sha256": N_ADAPTER_SHA256,
                "adapter_config_sha256": N_ADAPTER_CONFIG_SHA256,
                "benchmark_records_sha256": TRAIN_RECORDS_SHA256,
                "database_physical_sha256": TRAIN_DATABASE_PHYSICAL_SHA256,
                "database_content_sha256": TRAIN_DATABASE_CONTENT_SHA256,
                "benchmark_task_ids": list(TRAIN_TASK_IDS),
            }
            actual_identity = {
                "model_version": self.model_version,
                "adapter_sha256": self.adapter_sha256,
                "adapter_config_sha256": self.adapter_config_sha256,
                "benchmark_records_sha256": self.benchmark_records_sha256,
                "database_physical_sha256": self.database_physical_sha256,
                "database_content_sha256": self.database_content_sha256,
                "benchmark_task_ids": self.benchmark_task_ids,
            }
            if actual_identity != fixed_identity:
                raise RuntimeError("sft_n_frozen_identity_mismatch")
        _require_loopback(self.model_completion_endpoint, "model_completion_endpoint")
        _require_loopback(self.model_health_endpoint, "model_health_endpoint")
        _require_loopback(self.agent_endpoint, "agent_endpoint")
        if self.runtime != Settings.model_validate(self.runtime.model_dump()):
            raise RuntimeError("invalid_runtime_settings")

        checks: dict[str, Any] = {
            "profile_id": self.profile_id,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "adapter": {},
            "cache": {},
            "benchmark": {},
            "database": {},
            "generation": self.generation.model_dump(),
            "runtime": self.runtime.model_dump(),
        }

        if not self.model_config_path.is_file():
            raise FileNotFoundError("model_config_missing")
        model_config = tomllib.loads(self.model_config_path.read_text())
        if model_config.get("model_id") != self.model_id:
            raise RuntimeError("profile_model_config_id_mismatch")
        if model_config.get("revision") != self.revision:
            raise RuntimeError("profile_model_config_revision_mismatch")
        config_generation = GenerationSettings.model_validate(model_config.get("generation", {}))
        if config_generation != self.generation:
            raise RuntimeError("profile_generation_model_config_mismatch")
        checks["model_config"] = {
            "path": str(self.model_config_path),
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
        }

        manifest_path = self.cache_dir / "qwen-cache-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("cache_integrity_manifest_missing")
        cache_manifest = json.loads(manifest_path.read_text())
        if (
            cache_manifest.get("model") != self.model_id
            or cache_manifest.get("revision") != self.revision
            or not cache_manifest.get("weight_files")
        ):
            raise RuntimeError("cache_integrity_manifest_mismatch")
        snapshot = cache_manifest.get("snapshot")
        if snapshot:
            snapshot_path = Path(snapshot)
            candidates = [
                snapshot_path,
                self.cache_dir / Path(snapshot).name,
                self.cache_dir
                / "models--Qwen--Qwen2.5-1.5B-Instruct"
                / "snapshots"
                / self.revision,
            ]
            if not any(candidate.is_dir() for candidate in candidates):
                raise FileNotFoundError("cache_snapshot_missing")
        checks["cache"] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "model": cache_manifest["model"],
            "revision": cache_manifest["revision"],
            "weight_files": list(cache_manifest["weight_files"]),
        }

        config_path = self.adapter_dir / "adapter_config.json"
        weight_path = self.adapter_dir / "adapter_model.safetensors"
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError("adapter_files_missing")
        actual_adapter_sha = sha256_file(weight_path)
        actual_config_sha = sha256_file(config_path)
        if actual_adapter_sha != self.adapter_sha256:
            raise RuntimeError("adapter_weight_sha256_mismatch")
        if actual_config_sha != self.adapter_config_sha256:
            raise RuntimeError("adapter_config_sha256_mismatch")
        adapter_config = json.loads(config_path.read_text())
        if adapter_config.get("base_model_name_or_path") != self.model_id:
            raise RuntimeError("adapter_base_model_mismatch")
        if adapter_config.get("base_model_revision") != self.revision:
            raise RuntimeError("adapter_base_revision_mismatch")
        if adapter_config.get("implementation") != "enterprise_agent_rl.manual_lora_v1":
            raise RuntimeError("unsupported_adapter_implementation")
        checks["adapter"] = {
            "path": str(self.adapter_dir),
            "weight_sha256": actual_adapter_sha,
            "config_sha256": actual_config_sha,
            "implementation": adapter_config["implementation"],
            "module_count": len(adapter_config.get("module_names", [])),
        }

        if check_benchmark_records:
            if not self.benchmark_records_path.is_file():
                raise FileNotFoundError("benchmark_records_missing")
            actual_records_sha = sha256_file(self.benchmark_records_path)
            if actual_records_sha != self.benchmark_records_sha256:
                raise RuntimeError("benchmark_records_sha256_mismatch")
            registry = self.load_benchmark_registry()
            if set(registry) != set(self.benchmark_task_ids):
                raise RuntimeError("benchmark_registry_task_set_mismatch")
            checks["benchmark"] = {
                "path": str(self.benchmark_records_path),
                "sha256": actual_records_sha,
                "task_ids": sorted(registry),
                "public_only_endpoint_fields": ["task_id", "prompt", "split"],
            }

        if check_database:
            checks["database"] = self.check_database(self.database_path)
        return checks

    # 物理字节和逻辑内容指纹分别核对；复制文件权限只读不等于内容一定匹配。
    def check_database(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError("business_database_missing")
        physical = sha256_file(path)
        if physical != self.database_physical_sha256:
            raise RuntimeError("database_physical_sha256_mismatch")
        content = database_content_fingerprint(path)
        if content != self.database_content_sha256:
            raise RuntimeError("database_content_sha256_mismatch")
        return {
            "path": str(path),
            "physical_sha256": physical,
            "content_sha256": content,
            "mode": oct(path.stat().st_mode & 0o777),
        }

    # 建立获准任务的评分注册表；公开接口只返回公开字段，不把参考 SQL 暴露给 Policy。
    def load_benchmark_registry(self) -> dict[str, BenchmarkTask]:
        """Load only the explicitly selected train tasks for verifier-side use."""

        payload = json.loads(self.benchmark_records_path.read_text())
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("benchmark_records_invalid")
        registry: dict[str, BenchmarkTask] = {}
        for record in records:
            task_payload = record.get("task") if isinstance(record, dict) else None
            if not isinstance(task_payload, dict):
                raise ValueError("benchmark_task_record_invalid")
            task = BenchmarkTask.model_validate(task_payload)
            task_id = task.public.task_id
            if task_id in self.benchmark_task_ids:
                if task.split != "train" or record.get("database_id") != "train-1":
                    raise RuntimeError("benchmark_task_scope_mismatch")
                registry[task_id] = task
        if set(registry) != set(self.benchmark_task_ids):
            raise RuntimeError("benchmark_task_set_incomplete")
        return registry


# 本地 HTTP 探针绕过集群代理，并限制超时和响应大小，避免 ready 检查一直挂起。
def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    """Fetch one loopback JSON endpoint without honoring proxy environment variables."""

    _require_loopback(url, "endpoint")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            raw = response.read(2 * 1024 * 1024)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("model_health_unreachable") from exc
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("model_health_invalid_json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("model_health_invalid_payload")
    return payload


# ready 只是必要条件；还要核对 Adapter/revision、实际加载次数和持久加载记录。
def validate_model_health(profile: ServiceProfile, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the server's persisted model-load identity before accepting work."""

    if payload.get("status") != "ready":
        raise RuntimeError("model_endpoint_not_ready")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("model_health_metadata_missing")
    expected = profile.expected_endpoint_identity()
    service_profile = metadata.get("service_profile") or {}
    actual = {
        "profile_id": metadata.get("profile_id") or service_profile.get("profile_id"),
        "model_version": metadata.get("model_version") or service_profile.get("model_version"),
        "model_id": metadata.get("model_id") or metadata.get("model", {}).get("model_id"),
        "revision": metadata.get("model_revision") or metadata.get("model", {}).get("revision"),
        "adapter_sha256": metadata.get("adapter_sha256")
        or metadata.get("adapter", {}).get("adapter_sha256"),
        "adapter_config_sha256": metadata.get("adapter_config_sha256")
        or metadata.get("adapter", {}).get("adapter_config_sha256"),
    }
    if actual != expected:
        raise RuntimeError(
            "model_endpoint_identity_mismatch:"
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    if metadata.get("base_model_load_count") != 1:
        raise RuntimeError("model_base_load_count_mismatch")
    if metadata.get("adapter_merge_count") != 1:
        raise RuntimeError("model_adapter_merge_count_mismatch")
    model_load_path = metadata.get("model_load_path")
    if not isinstance(model_load_path, str) or not Path(model_load_path).is_file():
        raise RuntimeError("model_load_metadata_missing")
    try:
        persisted = json.loads(Path(model_load_path).read_text())
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("model_load_metadata_invalid") from exc
    for key in (
        "profile_id",
        "model_version",
        "model_id",
        "model_revision",
        "adapter_sha256",
        "adapter_config_sha256",
        "base_model_load_count",
        "adapter_merge_count",
    ):
        if persisted.get(key) != metadata.get(key):
            raise RuntimeError(f"model_load_metadata_mismatch:{key}")
    return {
        "status": payload["status"],
        "metadata": metadata,
        "validated_identity": actual,
    }


def probe_free_loopback_port(host: str = "127.0.0.1") -> int:
    """Return an ephemeral loopback port for the launcher, without probing others."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
