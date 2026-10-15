"""Reproducible CPU commands; real model requests are Slurm-gated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_agent_rl.agent import DeterministicPolicy, LocalModelPolicy
from enterprise_agent_rl.common import Settings, configure_logging
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.evaluation import compare_reports, evaluate
from enterprise_agent_rl.model_protocol import GenerationSettings
from enterprise_agent_rl.serving import create_app
from enterprise_agent_rl.trajectories import TrajectoryStore, replay


def main() -> None:
    parser = argparse.ArgumentParser(prog="earl")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-db")
    init.add_argument("--database", type=Path, default=Path("artifacts/business.sqlite3"))
    ev = sub.add_parser("evaluate")
    ev.add_argument("--database", type=Path, default=Path("artifacts/business.sqlite3"))
    ev.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    ev.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    ev.add_argument("--policy", choices=["fixture", "fixture-recovery", "local"], default="fixture")
    ev.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    ev.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ev.add_argument("--policy-version", default="local-untrained")
    ev.add_argument("--temperature", type=float, default=0)
    ev.add_argument("--max-tokens", type=int, default=512)
    ev.add_argument("--format-retries", type=int, default=1)
    rp = sub.add_parser("replay")
    rp.add_argument("trajectory", type=Path)
    rp.add_argument("--database", type=Path, default=Path("artifacts/business.sqlite3"))
    cp = sub.add_parser("compare")
    cp.add_argument("reports", type=Path, nargs="+")
    cp.add_argument("--output", type=Path, default=Path("artifacts/comparison.json"))
    api = sub.add_parser("serve")
    api.add_argument("--database", type=Path, default=Path("artifacts/business.sqlite3"))
    api.add_argument("--output", type=Path, default=Path("artifacts/api"))
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--local-endpoint")
    api.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    api.add_argument("--policy-version", default="local-untrained")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.load(args.config)
    if args.command == "init-db":
        print(initialize_database(args.database, settings.seed))
    elif args.command == "evaluate":
        policy = (
            LocalModelPolicy(
                args.endpoint,
                args.model,
                args.policy_version,
                GenerationSettings(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    format_retries=args.format_retries,
                ),
            )
            if args.policy == "local"
            else DeterministicPolicy(recover_error=args.policy == "fixture-recovery")
        )
        tasks = [t for t in benchmark_tasks() if args.split == "all" or t.split == args.split]
        report = evaluate(policy, tasks, args.database, settings, args.output)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "replay":
        result = replay(TrajectoryStore.load(args.trajectory), args.database)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["matched"]:
            raise SystemExit(1)
    elif args.command == "compare":
        result = compare_reports([json.loads(path.read_text()) for path in args.reports])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(args.output)
    elif args.command == "serve":
        import uvicorn

        policies = None
        if args.local_endpoint:
            # Fail fast on the login node before binding a server.
            local = LocalModelPolicy(args.local_endpoint, args.model, args.policy_version)
            policies = {local.version: lambda: local}
        uvicorn.run(
            create_app(args.database, args.output, settings, policies),
            host="127.0.0.1",
            port=args.port,
            workers=1,
        )


if __name__ == "__main__":
    main()
