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
from enterprise_agent_rl.service_config import ServiceProfile
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
    ev.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ev.add_argument("--policy-version", default="local-untrained")
    ev.add_argument("--temperature", type=float, default=0)
    ev.add_argument("--max-tokens", type=int, default=512)
    ev.add_argument("--format-retries", type=int, default=1)
    ev.add_argument("--top-p", type=float, default=1)
    ev.add_argument("--generation-seed", type=int, default=17)
    ev.add_argument("--parse-mode", choices=["strict", "json_fence"], default="json_fence")
    rp = sub.add_parser("replay")
    rp.add_argument("trajectory", type=Path)
    rp.add_argument("--database", type=Path, default=Path("artifacts/business.sqlite3"))
    cp = sub.add_parser("compare")
    cp.add_argument("reports", type=Path, nargs="+")
    cp.add_argument("--output", type=Path, default=Path("artifacts/comparison.json"))
    api = sub.add_parser("serve")
    api.add_argument("--database", type=Path)
    api.add_argument("--output", type=Path)
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--profile", type=Path)
    api.add_argument("--repo-root", type=Path)
    api.add_argument("--model-endpoint")
    api.add_argument("--model-health-endpoint")
    api.add_argument("--local-endpoint")
    api.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    api.add_argument("--policy-version", default="local-untrained")
    args = parser.parse_args()
    configure_logging()
    profile = None
    if args.command == "serve" and args.profile is not None:
        profile = ServiceProfile.load(args.profile, repo_root=args.repo_root)
        profile = profile.with_paths(
            database=args.database,
            output=args.output,
        ).with_endpoints(
            model_completion=args.model_endpoint,
            model_health=args.model_health_endpoint,
        )
        settings = profile.runtime
    else:
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
                    top_p=args.top_p,
                    seed=args.generation_seed,
                    parse_mode=args.parse_mode,
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

        if profile is not None:
            if args.local_endpoint or args.model != "Qwen/Qwen2.5-1.5B-Instruct":
                raise SystemExit("--profile fixes the local model endpoint and model identity")
            local = LocalModelPolicy(
                profile.model_completion_endpoint,
                profile.model_id,
                profile.model_version,
                profile.generation,
            )
            policies = {profile.model_version: lambda: local}
            app = create_app(
                profile.database_path,
                profile.output_dir,
                profile.runtime,
                policies,
                profile=profile,
                benchmark_registry=profile.load_benchmark_registry(),
            )
        else:
            database = args.database or Path("artifacts/business.sqlite3")
            output = args.output or Path("artifacts/api")
            policies = None
            if args.profile or args.model_endpoint or args.model_health_endpoint:
                raise SystemExit("--profile is required for explicit service endpoints")
            if args.local_endpoint:
                # Fail fast on the login node before binding a server.
                local = LocalModelPolicy(args.local_endpoint, args.model, args.policy_version)
                policies = {local.version: lambda: local}
            app = create_app(database, output, settings, policies)
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            workers=1,
        )


if __name__ == "__main__":
    main()
