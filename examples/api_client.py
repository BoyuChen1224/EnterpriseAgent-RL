"""Run `earl serve` first; this client calls only the loopback demo service."""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with OPENER.open(BASE + path, timeout=5) as response:
        return json.load(response)


def main():
    task = next(t for t in get("/benchmarks")["tasks"] if t["task_id"] == "top-2025")
    request = urllib.request.Request(
        BASE + "/tasks",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"prompt": task["prompt"], "benchmark_task_id": task["task_id"]}).encode(),
    )
    with OPENER.open(request, timeout=5) as response:
        job_id = json.load(response)["id"]
    for _ in range(100):
        status = get(f"/tasks/{job_id}")
        if status["status"] in ("completed", "failed"):
            print(json.dumps(status, ensure_ascii=False, indent=2))
            print(json.dumps(get(f"/tasks/{job_id}/trajectory"), ensure_ascii=False, indent=2))
            return
        time.sleep(0.1)
    raise TimeoutError("demo task did not finish within client polling budget")


if __name__ == "__main__":
    main()
