"""Small full-stack check; safe without a GPU, credentials or model weights."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from enterprise_agent_rl.agent import DeterministicPolicy
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.evaluation import evaluate
from enterprise_agent_rl.serving import create_app
from enterprise_agent_rl.trajectories import TrajectoryStore, replay


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="earl-smoke-") as directory:
        root = Path(directory)
        database = initialize_database(root / "business.sqlite3")
        tasks = [t for t in benchmark_tasks() if t.split == "test"]
        report = evaluate(
            DeterministicPolicy(recover_error=True), tasks, database, Settings(), root / "eval"
        )
        assert report["metrics"]["task_success_rate"] == 1
        paths = list((root / "eval" / "trajectories").glob("*.json"))
        assert len(paths) == len(tasks)
        for path in paths:
            assert replay(TrajectoryStore.load(path), database)["matched"]
        with TestClient(create_app(database, root / "api")) as client:
            task = tasks[0].public
            response = client.post(
                "/tasks", json={"prompt": task.prompt, "benchmark_task_id": task.task_id}
            )
            assert response.status_code == 202
            job_id = response.json()["id"]
            assert client.get(f"/tasks/{job_id}").json()["reward"]["success"]
            assert client.get(f"/tasks/{job_id}/trajectory").status_code == 200
        print(
            json.dumps(
                {"smoke": "passed", "replayed": len(paths), "metrics": report["metrics"]},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
