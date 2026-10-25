"""Local FastAPI demo; durable status records and bounded background work."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import Field

from enterprise_agent_rl.agent import DeterministicPolicy, LocalModelPolicy, Policy, run_episode
from enterprise_agent_rl.common import Contract, PublicTask, Settings
from enterprise_agent_rl.data import BenchmarkTask, benchmark_tasks
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.service_config import ServiceProfile, fetch_json, validate_model_health
from enterprise_agent_rl.trajectories import TrajectoryStore


class SubmitRequest(Contract):
    prompt: str = Field(min_length=1, max_length=4000)
    model_version: str | None = None
    benchmark_task_id: str | None = None


class CompareRequest(Contract):
    prompt: str = Field(min_length=1, max_length=4000)
    model_versions: list[str] = Field(min_length=2, max_length=3)
    benchmark_task_id: str | None = None


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
            # Single-process demo: a previous worker cannot survive this process restart.
            for job_id, payload in db.execute("SELECT id,payload FROM jobs").fetchall():
                job = json.loads(payload)
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", error="server_restarted")
                    db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(job), job_id))

    def put(self, job: dict) -> None:
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))

    def get(self, job_id: str) -> dict:
        with sqlite3.connect(self.path, timeout=5) as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown_task")
        return json.loads(row[0])


TECHNICAL_TERMINAL_PREFIXES = (
    "inference_",
    "model_",
    "policy_or_runtime_error",
)
TECHNICAL_TERMINAL_REASONS = {
    "oom",
    "model_http_error",
    "model_transport_error",
    "model_response_too_large",
}


def is_technical_failure(trajectory) -> bool:
    """Separate transport/runtime qualification failures from business outcomes."""

    return bool(trajectory.runtime_errors) or (
        trajectory.terminal_reason in TECHNICAL_TERMINAL_REASONS
        or trajectory.terminal_reason.startswith(TECHNICAL_TERMINAL_PREFIXES)
    )


def create_app(
    database: Path,
    output: Path,
    settings: Settings | None = None,
    policies: dict[str, Callable[[], Policy]] | None = None,
    *,
    profile: ServiceProfile | None = None,
    benchmark_registry: dict[str, BenchmarkTask] | None = None,
) -> FastAPI:
    if not database.is_file():
        raise FileNotFoundError("initialize_business_database_first")
    if profile is not None:
        if settings is not None and settings != profile.runtime:
            raise RuntimeError("service_runtime_settings_mismatch")
        profile.preflight(check_database=False, check_benchmark_records=True)
        profile.check_database(database)
        endpoint_health = validate_model_health(
            profile, fetch_json(profile.model_health_endpoint, profile.health_timeout_s)
        )
        cfg = profile.runtime
        registry = benchmark_registry or profile.load_benchmark_registry()
        if set(registry) != set(profile.benchmark_task_ids):
            raise RuntimeError("service_benchmark_registry_mismatch")
        if policies is None:
            local = LocalModelPolicy(
                profile.model_completion_endpoint,
                profile.model_id,
                profile.model_version,
                profile.generation,
            )
            factories = {profile.model_version: lambda: local}
        else:
            factories = policies
            if set(factories) != {profile.model_version}:
                raise RuntimeError("service_policy_version_mismatch")
        default_version = profile.model_version
        descriptors = {
            profile.model_version: {
                "version": profile.model_version,
                "kind": "local_llm",
                **profile.identity_summary(),
                "endpoint": profile.model_completion_endpoint,
                "health": endpoint_health,
            }
        }
    else:
        cfg = settings or Settings()
        factories = policies or {
            "fixture-v1": DeterministicPolicy,
            "fixture-v1-recovery": lambda: DeterministicPolicy(recover_error=True),
        }
        registry = {t.public.task_id: t for t in benchmark_tasks()}
        default_version = "fixture-v1"
        descriptors = {
            version: {
                "version": version,
                "kind": "deterministic_fixture" if version.startswith("fixture-") else "local_llm",
            }
            for version in factories
        }

    def require_current_model_health() -> None:
        if profile is None:
            return
        nonlocal endpoint_health
        try:
            endpoint_health = validate_model_health(
                profile, fetch_json(profile.model_health_endpoint, profile.health_timeout_s)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    store = TrajectoryStore(output / "trajectories")
    raw_store = TrajectoryStore(output / "raw-trajectories")
    jobs = JobStore(output / "jobs.sqlite3")
    slots = threading.BoundedSemaphore(8)
    app = FastAPI(title="EnterpriseAgent-RL", version="0.1.0")

    def execute(job: dict, request: SubmitRequest) -> None:
        raw_path = None
        try:
            job.update(status="running")
            jobs.put(job)
            public = PublicTask(
                task_id=request.benchmark_task_id or job["id"], prompt=request.prompt
            )
            trajectory = run_episode(
                public,
                factories[job["model_version"]](),
                database,
                cfg,
                output / "sql-audit.jsonl",
            )
            if profile is not None:
                trajectory = trajectory.model_copy(
                    update={
                        "policy_metadata": {
                            **trajectory.policy_metadata,
                            "service_profile": profile.profile_id,
                            "model_version": profile.model_version,
                            "model_id": profile.model_id,
                            "model_revision": profile.revision,
                            "adapter_sha256": profile.adapter_sha256,
                        }
                    }
                )
            raw_path = raw_store.save(trajectory)
            technical = is_technical_failure(trajectory)
            if request.benchmark_task_id and not technical:
                trajectory = trajectory.model_copy(
                    update={
                        "reward": verify(
                            registry[request.benchmark_task_id], trajectory, database, cfg
                        )
                    }
                )
            trajectory_path = store.save(trajectory)
            reward = trajectory.reward.model_dump() if trajectory.reward and not technical else None
            job.update(
                status="completed" if trajectory.terminal_reason == "completed" else "failed",
                run_id=trajectory.run_id,
                terminal_reason=trajectory.terminal_reason,
                reward=reward,
                qualification="technical_failure" if technical else "qualified",
                evaluation_status=(
                    "technical_unknown"
                    if technical and request.benchmark_task_id
                    else "verified_benchmark"
                    if request.benchmark_task_id
                    else "unscored_custom_prompt"
                ),
                raw_trajectory_path=str(raw_path),
                trajectory_path=str(trajectory_path),
            )
            jobs.put(job)
        except Exception as exc:
            job.update(
                status="failed",
                error=type(exc).__name__,
                qualification="technical_failure",
                evaluation_status="technical_unknown",
            )
            jobs.put(job)
        finally:
            slots.release()

    def resolve_version(request: SubmitRequest) -> str:
        version = request.model_version or default_version
        if version not in factories:
            raise HTTPException(status_code=422, detail="unknown_model_version")
        return version

    def validate(request: SubmitRequest) -> str:
        version = resolve_version(request)
        require_current_model_health()
        if request.benchmark_task_id:
            task = registry.get(request.benchmark_task_id)
            if task is None or task.public.prompt != request.prompt:
                raise HTTPException(status_code=422, detail="benchmark_prompt_mismatch")
        return version

    def submit_batch(
        requests: list[SubmitRequest],
        background: BackgroundTasks,
        request_ids: list[str | None] | None = None,
    ) -> list[dict]:
        versions = [validate(request) for request in requests]
        reserved = 0
        for _ in requests:
            if not slots.acquire(blocking=False):
                for _ in range(reserved):
                    slots.release()
                raise HTTPException(status_code=429, detail="task_capacity_reached")
            reserved += 1
        ids = request_ids or [None] * len(requests)
        queued = [
            {
                "id": str(uuid.uuid4()),
                "status": "queued",
                "model_version": version,
                "request_id": request_id,
            }
            for version, request_id in zip(versions, ids, strict=True)
        ]
        try:
            for job in queued:
                jobs.put(job)
        except Exception:
            for job in queued:
                try:
                    jobs.put({**job, "status": "failed", "error": "submission_failed"})
                except sqlite3.Error:
                    pass
            for _ in range(reserved):
                slots.release()
            raise
        for job, request in zip(queued, requests, strict=True):
            background.add_task(execute, job, request)
        return [dict(job) for job in queued]

    @app.get("/health")
    def health():
        result = {"status": "ok"}
        if profile is not None:
            try:
                require_current_model_health()
                result.update(
                    {
                        "profile": profile.identity_summary(),
                        "model_health": endpoint_health,
                    }
                )
            except HTTPException as exc:
                result.update(
                    {
                        "status": "degraded",
                        "profile": profile.identity_summary(),
                        "error": exc.detail,
                    }
                )
        return result

    @app.get("/models")
    def models():
        return {"models": list(descriptors.values())}

    @app.get("/benchmarks")
    def benchmarks():
        return {
            "tasks": [
                {"task_id": t.public.task_id, "prompt": t.public.prompt, "split": t.split}
                for t in registry.values()
            ]
        }

    @app.post("/tasks", status_code=202)
    def submit_task(
        request: SubmitRequest,
        background: BackgroundTasks,
        x_request_id: str | None = Header(default=None),
    ):
        return submit_batch([request], background, [x_request_id])[0]

    @app.get("/tasks/{job_id}")
    def get_task(job_id: str):
        return jobs.get(job_id)

    @app.get("/tasks/{job_id}/trajectory")
    def get_trajectory(job_id: str):
        job = jobs.get(job_id)
        if not job.get("run_id"):
            raise HTTPException(status_code=409, detail="trajectory_not_ready")
        return store.load(store.directory / f"{job['run_id']}.json").model_dump()

    @app.post("/comparisons", status_code=202)
    def comparison(
        request: CompareRequest,
        background: BackgroundTasks,
        x_request_id: str | None = Header(default=None),
    ):
        requests = [
            SubmitRequest(
                prompt=request.prompt,
                model_version=version,
                benchmark_task_id=request.benchmark_task_id,
            )
            for version in request.model_versions
        ]
        for item in requests:
            validate(item)
        if len(set(request.model_versions)) != len(request.model_versions):
            raise HTTPException(status_code=422, detail="duplicate_model_versions")
        return {
            "tasks": submit_batch(requests, background, [x_request_id] * len(requests)),
            "note": (
                "Compare task correctness and efficiency separately; fixtures are not LLM results."
            ),
        }

    return app
