"""Audit train-only expert trajectory coverage; this is not a Base-model result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from enterprise_agent_rl.agent import DeterministicPolicy
from enterprise_agent_rl.base_evaluation import evaluate_suite
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.trajectories import TrajectoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/m2a/manifest.json"))
    parser.add_argument("--database-root", type=Path, default=Path("artifacts/m2a-databases"))
    parser.add_argument("--work-output", type=Path, default=Path("artifacts/m2b-train-audit"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/m2b-preparation/train-data-audit.json")
    )
    args = parser.parse_args()
    report = evaluate_suite(
        DeterministicPolicy(),
        args.manifest,
        args.database_root,
        "train",
        Settings(),
        args.work_output,
    )
    store = TrajectoryStore(args.work_output / "trajectories")
    trajectories = [
        store.load(path) for path in sorted((args.work_output / "trajectories").glob("*.json"))
    ]
    action_sequences = {
        tuple(step.action.kind for step in trajectory.steps) for trajectory in trajectories
    }
    tool_sequences = {
        tuple(step.tool_name for step in trajectory.steps if step.tool_name)
        for trajectory in trajectories
    }
    templates = {
        record["template"]
        for record in json.loads(args.manifest.read_text())["records"]
        if record["task"]["split"] == "train"
    }
    summary = {
        "schema_version": "1.0",
        "source": "deterministic_fixture_expert",
        "llm_capability_result": False,
        "reference_answers_in_model_context": False,
        "tool_execution_and_independent_verification": True,
        "scheduled_tasks": report["scheduled_episodes"],
        "verified_trajectories": report["scored_episodes"],
        "successful_trajectories": sum(bool(t.reward and t.reward.success) for t in trajectories),
        "failed_trajectories": sum(not bool(t.reward and t.reward.success) for t in trajectories),
        "failure_types": dict(
            sorted(
                Counter(
                    (t.reward.reason if t.reward else t.terminal_reason) for t in trajectories
                ).items()
            )
        ),
        "task_templates": sorted(templates),
        "task_template_count": len(templates),
        "database_count": len({t.database_sha256 for t in trajectories}),
        "action_sequence_count": len(action_sequences),
        "tool_sequence_count": len(tool_sequences),
        "average_steps": (
            sum(len(t.steps) for t in trajectories) / len(trajectories) if trajectories else None
        ),
        "average_tool_calls": (
            sum(sum(step.tool_name is not None for step in t.steps) for t in trajectories)
            / len(trajectories)
            if trajectories
            else None
        ),
        "evaluation_report": str(args.work_output / "report.json"),
        "recommendation": (
            "The 20 train tasks can yield verified fixture expert traces, but are too small for "
            "strong generalization. Expand databases, schemas, business templates, recovery and "
            "negative traces before SFT/GRPO; keep Base trajectories and expert traces distinct."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
