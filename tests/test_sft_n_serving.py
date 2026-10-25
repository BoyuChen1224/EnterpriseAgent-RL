"""CPU-only T-026 service-contract tests with a real loopback HTTP fake engine."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import threading
import tomllib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from fastapi.testclient import TestClient

from enterprise_agent_rl.benchmark import database_content_fingerprint
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import BenchmarkTask, PublicTask, initialize_database
from enterprise_agent_rl.service_config import (
    MODEL_ID,
    MODEL_REVISION,
    ServiceProfile,
    sha256_file,
    validate_model_health,
)
from enterprise_agent_rl.serving import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reject_real_service_assets(monkeypatch, tmp_path):
    """Make accidental reuse of repository service assets a hard test failure."""

    real_artifacts = os.path.abspath(ROOT / "artifacts")
    allowed_root = os.path.abspath(tmp_path)

    def guard_path(path) -> str:
        raw = os.path.abspath(os.fsdecode(os.fspath(path)))
        if raw == real_artifacts or raw.startswith(real_artifacts + os.sep):
            raise AssertionError(f"real service asset access forbidden: {raw}")
        return raw

    def guard_tmp_path(path) -> None:
        raw = os.fsdecode(path)
        if raw.startswith("file:"):
            raw = unquote(urlsplit(raw).path)
        raw = os.path.abspath(raw)
        if raw != allowed_root and not raw.startswith(allowed_root + os.sep):
            raise AssertionError(f"non-tmp SQLite path forbidden: {raw}")

    for method_name in ("open", "read_text", "read_bytes", "is_file", "is_dir", "stat"):
        original = getattr(Path, method_name)

        def guarded(self, *args, _original=original, **kwargs):
            guard_path(self)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)

    real_connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        if isinstance(database, str | bytes | os.PathLike):
            value = os.fsdecode(database)
            if value != ":memory:":
                guard_tmp_path(value)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)

    real_run = subprocess.run

    def guarded_run(*args, **kwargs):
        request = kwargs.get("input")
        if isinstance(request, str):
            try:
                payload = json.loads(request)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("database"):
                guard_tmp_path(payload["database"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


def _synthetic_service_profile(tmp_path: Path) -> ServiceProfile:
    """Build every service asset inside tmp_path; never read the production TOML."""

    root = tmp_path / "cpu-fixture-assets"
    model_config = root / "model-config.toml"
    cache = root / "cache"
    snapshot = cache / "snapshot"
    adapter = root / "adapter"
    records_path = root / "benchmark-records.json"
    database = root / "business.sqlite3"
    output = tmp_path / "service-output"
    snapshot.mkdir(parents=True)
    adapter.mkdir(parents=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(database)

    model_config.write_text(
        "\n".join(
            [
                f'model_id = "{MODEL_ID}"',
                f'revision = "{MODEL_REVISION}"',
                'dtype = "float16"',
                'attention_backend = "sdpa_math"',
                "max_context_tokens = 8192",
                "",
                "[generation]",
                "temperature = 0.0",
                "top_p = 1.0",
                "max_tokens = 512",
                "format_retries = 1",
                'parse_mode = "json_fence"',
                "seed = 17",
                "",
            ]
        )
    )
    cache_manifest = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_files": ["synthetic-not-loaded.safetensors"],
        "snapshot": str(snapshot),
        "fixture": "cpu-fixture",
    }
    cache_manifest_path = cache / "qwen-cache-manifest.json"
    cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest_path.write_text(json.dumps(cache_manifest, sort_keys=True) + "\n")

    adapter_config = {
        "base_model_name_or_path": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "implementation": "enterprise_agent_rl.manual_lora_v1",
        "module_names": ["synthetic.module"],
    }
    adapter_config_path = adapter / "adapter_config.json"
    adapter_path = adapter / "adapter_model.safetensors"
    adapter_config_path.write_text(json.dumps(adapter_config, sort_keys=True) + "\n")
    adapter_path.write_bytes(b"cpu-fixture-adapter-bytes\n")

    schema_query = "SELECT name AS customer,region_id FROM customers ORDER BY customer"
    empty_query = (
        "SELECT DISTINCT c.name AS customer FROM customers c "
        "JOIN orders o ON o.customer_id=c.id "
        "WHERE o.status='paid' AND o.ordered_at>='2026-01-01' "
        "AND o.ordered_at<'2027-01-01' ORDER BY customer"
    )
    tasks = [
        BenchmarkTask(
            public=PublicTask(
                task_id="train-1-schema",
                prompt=(
                    "先使用 schema 确认客户表字段，列出客户名称和区域编号，"
                    "返回 customer, region_id，按客户名排序。"
                ),
            ),
            split="train",
            reference_sql=schema_query,
            ordered=True,
            required_schema=True,
        ),
        BenchmarkTask(
            public=PublicTask(
                task_id="train-1-empty",
                prompt="列出 2026 年有已支付订单的客户，返回 customer，按客户名排序。",
            ),
            split="train",
            reference_sql=empty_query,
            ordered=True,
        ),
    ]
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        json.dumps(
            {
                "fixture": "cpu-fixture",
                "records": [
                    {
                        "task": task.model_dump(),
                        "database_id": "train-1",
                        "database_version": "cpu-fixture-v1",
                    }
                    for task in tasks
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    return ServiceProfile(
        profile_id="cpu-fixture-sft-n",
        model_version="cpu-fixture-sft-n",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_config_path=model_config,
        cache_dir=cache,
        adapter_dir=adapter,
        benchmark_records_path=records_path,
        database_path=database,
        output_dir=output,
        model_completion_endpoint="http://127.0.0.1:18001/v1/chat/completions",
        model_health_endpoint="http://127.0.0.1:18001/health",
        agent_endpoint="http://127.0.0.1:18000",
        adapter_sha256=sha256_file(adapter_path),
        adapter_config_sha256=sha256_file(adapter_config_path),
        benchmark_records_sha256=sha256_file(records_path),
        database_physical_sha256=sha256_file(database),
        database_content_sha256=database_content_fingerprint(database),
        benchmark_task_ids=[task.public.task_id for task in tasks],
        runtime=Settings(),
    )


class FakeModelHTTP:
    def __init__(self, profile: ServiceProfile):
        self.profile = profile
        self.mode = "normal"
        self.requests: list[dict] = []
        self.response_ids: list[str] = []
        metadata = {
            "profile_id": profile.profile_id,
            "model_version": profile.model_version,
            "model_id": profile.model_id,
            "model_revision": profile.revision,
            "adapter_sha256": profile.adapter_sha256,
            "adapter_config_sha256": profile.adapter_config_sha256,
            "base_model_load_count": 1,
            "adapter_merge_count": 1,
            "model": {"model_id": profile.model_id, "revision": profile.revision},
        }
        profile.output_dir.mkdir(parents=True, exist_ok=True)
        model_load_path = profile.output_dir / "model-load.json"
        metadata["model_load_path"] = str(model_load_path)
        model_load_path.write_text(json.dumps(metadata) + "\n")
        self.health = {"status": "ready", "metadata": metadata}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def send_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - stdlib handler API
                if self.path == "/health":
                    self.send_json(200, owner.health)
                elif self.path == "/models":
                    metadata = owner.health["metadata"]
                    self.send_json(
                        200,
                        {
                            "models": [
                                {
                                    "model": metadata["model_id"],
                                    "model_version": metadata["model_version"],
                                    "revision": metadata["model_revision"],
                                    "adapter_sha256": metadata["adapter_sha256"],
                                }
                            ]
                        },
                    )
                else:
                    self.send_json(404, {"error": "not_found"})

            def do_POST(self):  # noqa: N802 - stdlib handler API
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                owner.requests.append(request)
                if self.path != "/v1/chat/completions":
                    self.send_json(404, {"error": "not_found"})
                    return
                if owner.mode == "transport_error":
                    self.send_json(503, {"error": {"code": "oom"}})
                    return
                if owner.mode == "invalid_action":
                    content = '{"kind":"invalid","arguments":{}}'
                else:
                    prompt = request["messages"][1]["content"]
                    history_count = sum(
                        message["role"] == "assistant" for message in request["messages"]
                    )
                    if history_count == 0:
                        content = '{"kind":"plan","arguments":{"steps":["query"]}}'
                    elif history_count == 1:
                        content = '{"kind":"schema","arguments":{}}'
                    elif history_count == 2 and "客户表字段" in prompt:
                        content = (
                            '{"kind":"sql","arguments":{"query":"SELECT name AS customer,'
                            'region_id FROM customers ORDER BY customer"}}'
                        )
                    elif history_count == 2:
                        query = (
                            "SELECT DISTINCT c.name AS customer FROM customers c "
                            "JOIN orders o ON o.customer_id=c.id "
                            "WHERE o.status='paid' AND o.ordered_at>='2026-01-01' "
                            "AND o.ordered_at<'2027-01-01' ORDER BY customer"
                        )
                        content = json.dumps(
                            {"kind": "sql", "arguments": {"query": query}},
                            ensure_ascii=False,
                        )
                    elif history_count == 3:
                        columns = (
                            ["customer", "region_id"] if "客户表字段" in prompt else ["customer"]
                        )
                        content = json.dumps(
                            {
                                "kind": "check",
                                "arguments": {"evidence_step": 2, "expected_columns": columns},
                            },
                            ensure_ascii=False,
                        )
                    else:
                        content = json.dumps(
                            {
                                "kind": "finish",
                                "arguments": {"evidence_step": 2, "summary": "done"},
                            }
                        )
                event_id = str(uuid.uuid4())
                owner.response_ids.append(event_id)
                self.send_json(
                    200,
                    {
                        "id": event_id,
                        "model": owner.profile.model_id,
                        "choices": [{"message": {"role": "assistant", "content": content}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "telemetry": {"event_id": event_id, "generation_latency_s": 0.01},
                    },
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def service_profile(tmp_path):
    return _synthetic_service_profile(tmp_path)


def test_sft_n_profile_preflight_and_identity_rejection(service_profile):
    checks = service_profile.preflight()
    assert checks["adapter"]["weight_sha256"] == service_profile.adapter_sha256
    assert checks["benchmark"]["task_ids"] == ["train-1-empty", "train-1-schema"]
    assert checks["database"]["physical_sha256"] == service_profile.database_physical_sha256
    assert service_profile.profile_id == "cpu-fixture-sft-n"
    assert all(
        str(path).startswith(str(service_profile.database_path.parents[1]))
        for path in (
            service_profile.model_config_path,
            service_profile.cache_dir,
            service_profile.adapter_dir,
            service_profile.benchmark_records_path,
            service_profile.database_path,
        )
    )

    bad = service_profile.model_copy(
        update={
            "profile_id": "sft-n-step27",
            "model_version": "wrong-frozen-identity",
            "adapter_sha256": "0" * 64,
        }
    )
    with pytest.raises(RuntimeError, match="sft_n_frozen_identity_mismatch"):
        bad.preflight(check_database=False, check_benchmark_records=False)

    bad_health = {
        "status": "ready",
        "metadata": {
            "profile_id": service_profile.profile_id,
            "model_version": service_profile.model_version,
            "model_id": service_profile.model_id,
            "model_revision": "0" * 40,
            "adapter_sha256": service_profile.adapter_sha256,
            "adapter_config_sha256": service_profile.adapter_config_sha256,
            "base_model_load_count": 1,
            "adapter_merge_count": 1,
        },
    }
    with pytest.raises(RuntimeError, match="model_endpoint_identity_mismatch"):
        validate_model_health(service_profile, bad_health)


def test_production_n_identity_is_checked_before_any_asset_io(monkeypatch, service_profile):
    production = tomllib.loads((ROOT / "configs/serving/sft-n.toml").read_text())
    assert production["profile_id"] == "sft-n-step27"
    assert production["identity"]["benchmark_task_ids"] == [
        "train-1-schema",
        "train-1-empty",
    ]

    def fail_asset_io(*_args, **_kwargs):
        raise AssertionError("asset IO occurred before frozen identity rejection")

    monkeypatch.setattr(Path, "is_file", fail_asset_io)
    monkeypatch.setattr(Path, "read_text", fail_asset_io)
    bad = service_profile.model_copy(
        update={"profile_id": "sft-n-step27", "model_version": "not-sft-n"}
    )
    with pytest.raises(RuntimeError, match="sft_n_frozen_identity_mismatch"):
        bad.preflight(check_database=False, check_benchmark_records=False)


def test_real_http_local_policy_agent_api_and_business_technical_split(
    service_profile, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-contract-only")
    fake = FakeModelHTTP(service_profile)
    service_profile = service_profile.with_endpoints(
        model_completion=f"http://127.0.0.1:{fake.port}/v1/chat/completions",
        model_health=f"http://127.0.0.1:{fake.port}/health",
        agent="http://127.0.0.1:18000",
    )
    fake.profile = service_profile
    fake.health["metadata"].update(
        {
            "profile_id": service_profile.profile_id,
            "model_version": service_profile.model_version,
        }
    )
    fake.start()
    try:
        app = create_app(
            service_profile.database_path,
            service_profile.output_dir,
            profile=service_profile,
        )
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["status"] == "ok"
            assert health["profile"]["model_version"] == service_profile.model_version
            models = client.get("/models").json()["models"]
            assert models[0]["adapter_sha256"] == service_profile.adapter_sha256
            benchmark_payload = client.get("/benchmarks").json()
            assert {task["task_id"] for task in benchmark_payload["tasks"]} == {
                "train-1-schema",
                "train-1-empty",
            }
            assert all("reference_sql" not in task for task in benchmark_payload["tasks"])

            jobs = []
            for task in benchmark_payload["tasks"]:
                response = client.post(
                    "/tasks",
                    json={"prompt": task["prompt"], "benchmark_task_id": task["task_id"]},
                    headers={"X-Request-ID": f"cpu-{task['task_id']}"},
                )
                assert response.status_code == 202
                jobs.append(response.json())
            for submitted in jobs:
                status = client.get(f"/tasks/{submitted['id']}").json()
                assert status["status"] == "completed"
                assert status["qualification"] == "qualified"
                assert status["evaluation_status"] == "verified_benchmark"
                assert status["reward"]["success"] is True
                trajectory = client.get(f"/tasks/{submitted['id']}/trajectory").json()
                assert (
                    trajectory["policy_metadata"]["model_version"] == service_profile.model_version
                )
                assert trajectory["database_sha256"] == service_profile.database_physical_sha256
                event_ids = [
                    attempt["telemetry"]["event_id"]
                    for attempt in trajectory["model_attempts"]
                    if attempt.get("telemetry", {}).get("event_id")
                ]
                assert event_ids
                assert set(event_ids).issubset(set(fake.response_ids))

            # A custom prompt retains the old unscored semantics.
            custom = client.post("/tasks", json={"prompt": "列出客户"}).json()
            custom_status = client.get(f"/tasks/{custom['id']}").json()
            assert custom_status["reward"] is None
            assert custom_status["evaluation_status"] == "unscored_custom_prompt"

            fake.mode = "invalid_action"
            invalid = client.post(
                "/tasks",
                json={
                    "prompt": benchmark_payload["tasks"][0]["prompt"],
                    "benchmark_task_id": "train-1-schema",
                },
            ).json()
            invalid_status = client.get(f"/tasks/{invalid['id']}").json()
            assert invalid_status["qualification"] == "qualified"
            assert invalid_status["evaluation_status"] == "verified_benchmark"
            assert invalid_status["reward"]["success"] is False

            fake.mode = "transport_error"
            failed = client.post(
                "/tasks",
                json={
                    "prompt": benchmark_payload["tasks"][0]["prompt"],
                    "benchmark_task_id": "train-1-schema",
                },
            ).json()
            failed_status = client.get(f"/tasks/{failed['id']}").json()
            assert failed_status["qualification"] == "technical_failure"
            assert failed_status["evaluation_status"] == "technical_unknown"
            assert failed_status["reward"] is None
    finally:
        fake.close()


def test_profile_rejects_non_loopback_endpoint(service_profile):
    bad = service_profile.with_endpoints(model_health="https://example.com/health")
    with pytest.raises(ValueError, match="loopback"):
        bad.preflight(check_database=False, check_benchmark_records=False)


def test_launcher_reaps_only_its_owned_process_group():
    path = ROOT / "scripts/serving/run_sft_n_api.py"
    spec = importlib.util.spec_from_file_location("t026_launcher", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        returncode = module.stop_process_group(process, grace_s=1)
        assert returncode is not None and process.poll() is not None
    finally:
        if process.poll() is None:
            module.stop_process_group(process, grace_s=1)
