"""Local FastAPI demo; durable status records and bounded background work."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import Field

from enterprise_agent_rl.agent import DeterministicPolicy, Policy, run_episode
from enterprise_agent_rl.common import Contract, PublicTask, Settings
from enterprise_agent_rl.data import benchmark_tasks
from enterprise_agent_rl.rewards import verify
from enterprise_agent_rl.trajectories import TrajectoryStore


class SubmitRequest(Contract):
    prompt: str = Field(min_length=1, max_length=4000)
    model_version: str = "fixture-v1"
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


def create_app(
    database: Path,
    output: Path,
    settings: Settings | None = None,
    policies: dict[str, Callable[[], Policy]] | None = None,
) -> FastAPI:
    if not database.is_file():
        raise FileNotFoundError("initialize_business_database_first")
    cfg = settings or Settings()
    factories = policies or {
        "fixture-v1": DeterministicPolicy,
        "fixture-v1-recovery": lambda: DeterministicPolicy(recover_error=True),
    }
    store = TrajectoryStore(output / "trajectories")
    jobs = JobStore(output / "jobs.sqlite3")
    slots = threading.BoundedSemaphore(8)
    registry = {t.public.task_id: t for t in benchmark_tasks()}
    app = FastAPI(title="EnterpriseAgent-RL", version="0.1.0")

    def execute(job: dict, request: SubmitRequest) -> None:
        try:
            job.update(status="running")
            jobs.put(job)
            public = PublicTask(
                task_id=request.benchmark_task_id or job["id"], prompt=request.prompt
            )
            trajectory = run_episode(
                public,
                factories[request.model_version](),
                database,
                cfg,
                output / "sql-audit.jsonl",
            )
            if request.benchmark_task_id:
                trajectory = trajectory.model_copy(
                    update={
                        "reward": verify(registry[request.benchmark_task_id], trajectory, database)
                    }
                )
            store.save(trajectory)
            job.update(
                status="completed" if trajectory.terminal_reason == "completed" else "failed",
                run_id=trajectory.run_id,
                terminal_reason=trajectory.terminal_reason,
                reward=trajectory.reward.model_dump() if trajectory.reward else None,
                evaluation_status="verified_benchmark"
                if trajectory.reward
                else "unscored_custom_prompt",
            )
            jobs.put(job)
        except Exception as exc:
            job.update(status="failed", error=type(exc).__name__)
            jobs.put(job)
        finally:
            slots.release()

    def validate(request: SubmitRequest) -> None:
        if request.model_version not in factories:
            raise HTTPException(status_code=422, detail="unknown_model_version")
        if request.benchmark_task_id:
            task = registry.get(request.benchmark_task_id)
            if task is None or task.public.prompt != request.prompt:
                raise HTTPException(status_code=422, detail="benchmark_prompt_mismatch")

    def submit_batch(requests: list[SubmitRequest], background: BackgroundTasks) -> list[dict]:
        for request in requests:
            validate(request)
        reserved = 0
        for _ in requests:
            if not slots.acquire(blocking=False):
                for _ in range(reserved):
                    slots.release()
                raise HTTPException(status_code=429, detail="task_capacity_reached")
            reserved += 1
        queued = [
            {"id": str(uuid.uuid4()), "status": "queued", "model_version": request.model_version}
            for request in requests
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
        return {"status": "ok"}

    @app.get("/models")
    def models():
        return {
            "models": [
                {"version": version, "kind": factory().kind}
                for version, factory in factories.items()
            ]
        }

    @app.get("/benchmarks")
    def benchmarks():
        return {
            "tasks": [
                {"task_id": t.public.task_id, "prompt": t.public.prompt, "split": t.split}
                for t in registry.values()
            ]
        }

    @app.post("/tasks", status_code=202)
    def submit_task(request: SubmitRequest, background: BackgroundTasks):
        return submit_batch([request], background)[0]

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
    def comparison(request: CompareRequest, background: BackgroundTasks):
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
            "tasks": submit_batch(requests, background),
            "note": (
                "Compare task correctness and efficiency separately; fixtures are not LLM results."
            ),
        }

    return app
