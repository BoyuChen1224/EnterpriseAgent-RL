"""Public online GRPO orchestration shared by real and CPU backends."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .aggregation import aggregate_source, paired_comparison, three_arm_comparison
from .budget import BudgetError, BudgetGuard, Clock
from .config import GrpoConfig, RunContext
from .evaluation import evaluate_frozen
from .evidence import evidence_journal_summary
from .io import atomic_write_json, canonical_hash, read_json
from .rollout import OnlineRollout
from .trainer import ContinuousTrainer


def _components(value: Any) -> tuple[Any, Any, Any]:
    if isinstance(value, dict):
        return value["policy"], value["engine"], value["worker"]
    if isinstance(value, tuple) and len(value) == 3:
        return value
    raise TypeError("component_factory_must_return_policy_engine_worker")


def run_online_training(
    context: RunContext,
    config: GrpoConfig,
    tasks: list[Any],
    *,
    component_factory: Callable[[BudgetGuard, RunContext, GrpoConfig], Any] | None = None,
    policy: Any | None = None,
    engine: Any | None = None,
    worker: Any | None = None,
    evaluation_tasks: dict[str, list[Any]] | None = None,
    evaluation_factory: Callable[..., Any] | None = None,
    run_evaluation: bool = False,
    job_id: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run rollout → advantage → two updates × two rounds → freeze → eval.

    The only difference between CPU integration and a future GPU run is the
    injected component triple.  The durable stage markers, budget checks, and
    data contracts are shared.
    """

    context = context.resolved()
    context.run_dir.mkdir(parents=True, exist_ok=True)
    if len(tasks) != config.group_count:
        raise ValueError(f"train_group_count_mismatch:{len(tasks)}:{config.group_count}")
    budget = config.budget
    guard = BudgetGuard(
        context.run_dir / "budget-state.json",
        run_id=str(config.raw.get("run", {}).get("run_id", "grpo-online")),
        run_hard_seconds=int(budget["run_hard_seconds"]),
        job_hard_seconds=int(budget["job_hard_seconds"]),
        soft_seconds=int(budget["soft_seconds"]),
        cleanup_reserve_seconds=int(budget["cleanup_reserve_seconds"]),
        gpu_count=int(budget.get("allocated_gpu_count", 1)),
        call_limits=config.limits,
        clock=clock,
    )
    guard.start_job(job_id=job_id or os.environ.get("SLURM_JOB_ID"))

    def budget_info(phase: str) -> dict[str, Any]:
        try:
            return guard.current(phase).as_dict()
        except Exception as exc:
            return {"status": "UNAVAILABLE", "type": type(exc).__name__, "message": str(exc)}

    created_policy = False
    status = "FAILED"
    failure: dict[str, Any] | None = None
    try:
        guard.check("setup:component-factory")
        if component_factory is not None:
            policy, engine, worker = _components(
                guard.run_callable(
                    lambda: component_factory(guard, context, config),
                    phase="setup:component-factory",
                )
            )
            created_policy = True
        if policy is None or engine is None or worker is None:
            raise ValueError("policy_engine_worker_required")
        prior_state = read_json(context.run_dir / "training" / "state.json", {}) or {}
        if prior_state.get("successful_updates", 0) and hasattr(policy, "load_run_state"):
            guard.check("setup:resume")
            policy.load_run_state(context.run_dir, prior_state)
        rollout = OnlineRollout(
            run_dir=context.run_dir,
            guard=guard,
            seed_start=int(config.rollout["seed_start"]),
            group_count=config.group_count,
        )
        trainer = ContinuousTrainer(run_dir=context.run_dir, guard=guard, policy=policy)
        updates = int(prior_state.get("successful_updates", 0))
        round_summaries: list[dict[str, Any]] = []
        for round_index in range(config.rounds):
            expected_before = round_index * config.updates_per_round
            if updates >= expected_before + config.updates_per_round:
                marker = (
                    read_json(
                        context.run_dir
                        / "rounds"
                        / f"r{round_index}"
                        / "training"
                        / "metrics.json",
                        {},
                    )
                    or {}
                )
                if not marker:
                    raise RuntimeError(f"completed_round_metrics_missing:{round_index}")
                if (
                    marker.get("status") != "PASS"
                    or marker.get("round_index") != round_index
                    or marker.get("successful_updates_total")
                    != expected_before + config.updates_per_round
                    or len(marker.get("updates", [])) != config.updates_per_round
                ):
                    raise RuntimeError(f"completed_round_metrics_invalid:{round_index}")
                round_summaries.append(marker)
                continue
            if updates < expected_before:
                raise RuntimeError(
                    f"round_order_violation:{round_index}:{updates}:{expected_before}"
                )
            guard.check(f"round{round_index}:policy-snapshot")
            snapshot = trainer.current_snapshot(round_index=round_index, successful_updates=updates)
            snapshot_path = context.run_dir / "rounds" / f"r{round_index}" / "policy-snapshot.json"
            previous_snapshot = read_json(snapshot_path)
            if previous_snapshot:
                for key in (
                    "base_model",
                    "base_revision",
                    "model_ref_id",
                ):
                    # The snapshot is the sampling policy for the whole round.
                    # Its delta changes after step 1/3, so only immutable
                    # identity fields may be compared on resume.
                    if key in previous_snapshot and previous_snapshot.get(key) != snapshot.get(key):
                        raise RuntimeError(f"policy_snapshot_changed:{round_index}:{key}")
                snapshot = previous_snapshot
            else:
                atomic_write_json(snapshot_path, snapshot)
            rollout_result = rollout.run_round(
                round_index=round_index,
                tasks=tasks,
                samples_per_task=config.samples_per_task,
                policy_snapshot=snapshot,
                engine=engine,
                worker=worker,
            )
            if rollout_result.get("status") != "COMPLETED":
                raise RuntimeError(
                    f"round_rollout_incomplete:{round_index}:{rollout_result.get('status')}"
                )
            dataset = rollout_result["dataset"]
            remaining = config.updates_per_round - (updates - expected_before)
            summary = trainer.train_round(
                round_index=round_index,
                dataset=dataset,
                first_step=updates,
                updates=remaining,
            )
            updates += int(summary["logical_updates"])
            round_summaries.append(summary)
            guard.check(f"round{round_index}:post-update", allow_soft=True)

        if updates != config.rounds * config.updates_per_round:
            raise RuntimeError(f"final_update_count_incomplete:{updates}")
        guard.check("finalize:freeze")
        final_snapshot = trainer.current_snapshot(
            round_index=config.rounds, successful_updates=updates
        )
        if hasattr(policy, "freeze_final"):
            final_artifact = policy.freeze_final(context.run_dir, snapshot=final_snapshot)
        else:
            final_artifact = {
                "status": "METADATA_ONLY",
                "policy_snapshot_id": final_snapshot.get("policy_snapshot_id"),
            }
        atomic_write_json(
            context.run_dir / "training" / "final-step4.json",
            {
                "schema_version": "grpo-final-step-v1",
                "status": "FINAL_STEP4_FROZEN",
                "snapshot": final_snapshot,
                "artifact": final_artifact,
            },
        )
        evaluations: dict[str, Any] = {}
        evaluation_aggregates: dict[str, Any] = {}
        comparison: dict[str, Any] | None = None
        if run_evaluation:
            if not evaluation_tasks or set(evaluation_tasks) != {"N", "RL"}:
                raise ValueError("evaluation_tasks_N_and_RL_required")
            if policy is not None and evaluation_factory is not None:
                # The training model is no longer needed after final-delta
                # freeze.  Freeing it allows two genuinely fresh arms on a
                # 32 GB V100 instead of reusing the training object.
                policy.close()
                policy = None
            arm_results: dict[str, dict[str, Any]] = {}
            for arm in ("N", "RL"):
                arm_policy = None
                arm_engine = engine
                arm_worker = worker
                arm_metadata: dict[str, Any] = {"isolated_model": False}
                if evaluation_factory is not None:
                    created = guard.run_callable(
                        lambda arm=arm: evaluation_factory(arm, guard, context, config),
                        phase=f"evaluation:{arm}:setup",
                    )
                    if isinstance(created, dict):
                        arm_policy = created.get("policy")
                        arm_engine = created["engine"]
                        arm_worker = created["worker"]
                        arm_metadata = dict(created.get("metadata") or {})
                    elif isinstance(created, tuple) and len(created) in {3, 4}:
                        arm_policy, arm_engine, arm_worker = created[:3]
                        if len(created) == 4:
                            arm_metadata = dict(created[3] or {})
                    else:
                        raise TypeError("evaluation_factory_invalid_components")
                evaluations[arm] = evaluate_frozen(
                    run_dir=context.run_dir,
                    arm=arm,
                    tasks=evaluation_tasks[arm],
                    engine=arm_engine,
                    worker=arm_worker,
                    guard=guard,
                    policy_snapshot=final_snapshot,
                    arm_metadata=arm_metadata,
                    evaluation_seed=int(config.evaluation.get("generation", {}).get("seed", 17)),
                )
                if arm_policy is not None and hasattr(arm_policy, "close"):
                    guard.run_callable(arm_policy.close, phase=f"evaluation:{arm}:cleanup")
                arm_results[arm] = evaluations[arm]
                evaluation_aggregates[arm] = aggregate_source(
                    source_root=context.run_dir,
                    verifier_index=context.run_dir / "evaluation" / arm / "verifier-results.jsonl",
                    identity_field="task_id",
                    expected_count=len(evaluation_tasks[arm]),
                    source_kind=f"T023-E025-real-validation-{arm}",
                )
            comparison = paired_comparison(
                evaluation_aggregates["N"],
                evaluation_aggregates["RL"],
                expected_count=len(evaluation_tasks["N"]),
            )
            atomic_write_json(context.run_dir / "comparison.json", comparison)
            atomic_write_json(context.run_dir / "metrics.json", evaluation_aggregates)
        status = (
            "COMPLETED"
            if not run_evaluation
            or all(value.get("status") == "COMPLETED" for value in evaluations.values())
            else "INCOMPLETE_EVALUATION"
        )
        budget_snapshot = budget_info("run-complete")
        result = {
            "schema_version": "grpo-online-run-v1",
            "status": status,
            "updates": updates,
            "rounds": round_summaries,
            "final_snapshot": final_snapshot,
            "final_artifact": final_artifact,
            "evaluations": evaluations,
            "evaluation_aggregates": evaluation_aggregates,
            "comparison": comparison,
            "budget": budget_snapshot,
        }
        atomic_write_json(context.run_dir / "run-status.json", result)
        return result
    except BudgetError as exc:
        failure = {"status": "BUDGET_STOP", "type": type(exc).__name__, "message": str(exc)}
        failure["budget"] = budget_info("budget-stop")
        atomic_write_json(context.run_dir / "run-status.json", failure)
        raise
    except Exception as exc:
        failure = {"status": "FAILED", "type": type(exc).__name__, "message": str(exc)}
        failure["budget"] = budget_info("failure")
        atomic_write_json(context.run_dir / "run-status.json", failure)
        raise
    finally:
        if policy is not None and (created_policy or hasattr(policy, "close")):
            try:
                guard.check("cleanup:before-policy-close", allow_soft=True)
            except BudgetError as exc:
                atomic_write_json(
                    context.run_dir / "cleanup-budget-stop.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
            try:
                guard.run_callable(policy.close, phase="cleanup:policy-close")
            except Exception as exc:
                atomic_write_json(
                    context.run_dir / "cleanup-error.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
        close_status = (
            "BUDGET_STOP" if failure and failure.get("status") == "BUDGET_STOP" else status
        )
        try:
            guard.close_job(status=close_status, reason=failure["message"] if failure else None)
        except Exception as exc:
            atomic_write_json(
                context.run_dir / "budget-close-error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )


def run_frozen_evaluation(
    context: RunContext,
    config: GrpoConfig,
    evaluation_tasks: list[Any],
    *,
    evaluation_factory: Callable[..., Any],
    job_id: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Run Base → N → RL evaluation without constructing the train stack."""

    context = context.resolved()
    context.run_dir.mkdir(parents=True, exist_ok=True)
    budget = config.budget
    guard = BudgetGuard(
        context.run_dir / "budget-state.json",
        run_id=str(config.raw.get("run", {}).get("run_id", "grpo-frozen-evaluation")),
        run_hard_seconds=int(budget["run_hard_seconds"]),
        job_hard_seconds=int(budget["job_hard_seconds"]),
        soft_seconds=int(budget["soft_seconds"]),
        cleanup_reserve_seconds=int(budget["cleanup_reserve_seconds"]),
        gpu_count=int(budget.get("allocated_gpu_count", 1)),
        call_limits=config.limits,
        clock=clock,
    )
    recovery_seconds = os.environ.get("EARL_FROZEN_RECOVERY_ELAPSED")
    if recovery_seconds is not None:
        guard.recover_interrupted(
            elapsed_gpu_seconds=float(recovery_seconds),
            reason=os.environ.get(
                "EARL_FROZEN_RECOVERY_REASON", "previous_job_stopped_before_frozen_protocol"
            ),
        )
    guard.start_job(job_id=job_id or os.environ.get("SLURM_JOB_ID"))
    evaluations: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    active_policy: Any | None = None
    failure: dict[str, Any] | None = None
    final_status = "FAILED"
    budget_stopped = False
    protocol_hash = canonical_hash(
        {
            "config": config.raw,
            "evaluation_task_ids": [task.task_id for task in evaluation_tasks],
            "evaluation_seed": int(config.evaluation.get("generation", {}).get("seed", 17)),
        }
    )
    base_snapshot = {
        "schema_version": "grpo-frozen-evaluation-snapshot-v1",
        "policy_snapshot_id": f"frozen-{protocol_hash[:16]}",
        "base_model": config.model["model_id"],
        "base_revision": config.model["revision"],
        "protocol_hash": protocol_hash,
        "evaluation_seed": int(config.evaluation.get("generation", {}).get("seed", 17)),
        "training_calls": 0,
    }

    def empty_arm_source(arm: str, error: dict[str, Any]) -> None:
        output = context.run_dir / "evaluation" / arm
        output.mkdir(parents=True, exist_ok=True)
        index = output / "verifier-results.jsonl"
        for task in evaluation_tasks:
            from .io import append_jsonl

            append_jsonl(
                index,
                {
                    "record_type": "RESULT",
                    "task_id": task.task_id,
                    "episode_id": f"{arm}-{task.task_id}",
                    "status": "unknown",
                    "unknown_reason": "arm_setup_failure",
                    "technical_unknown": True,
                    "error": error,
                    "policy_snapshot_id": f"{base_snapshot['policy_snapshot_id']}-{arm}",
                    "raw_saved": True,
                    "verifier_called": False,
                },
            )

    try:
        for arm in ("Base", "N", "RL"):
            arm_policy = None
            try:
                guard.check(f"evaluation:{arm}:setup")
                created = guard.run_callable(
                    lambda arm=arm: evaluation_factory(arm, guard, context, config),
                    phase=f"evaluation:{arm}:setup",
                )
                if isinstance(created, dict):
                    arm_policy = created.get("policy")
                    engine = created["engine"]
                    worker = created["worker"]
                    metadata = dict(created.get("metadata") or {})
                elif isinstance(created, tuple) and len(created) in {3, 4}:
                    arm_policy, engine, worker = created[:3]
                    metadata = dict(created[3] or {}) if len(created) == 4 else {}
                else:
                    raise TypeError("evaluation_factory_invalid_components")
                active_policy = arm_policy
                evaluations[arm] = evaluate_frozen(
                    run_dir=context.run_dir,
                    arm=arm,
                    tasks=evaluation_tasks,
                    engine=engine,
                    worker=worker,
                    guard=guard,
                    policy_snapshot=base_snapshot,
                    arm_metadata=metadata,
                    evaluation_seed=int(config.evaluation.get("generation", {}).get("seed", 17)),
                    require_qualified_evidence=True,
                )
            except BudgetError:
                raise
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                empty_arm_source(arm, error)
                evaluations[arm] = {
                    "schema_version": "grpo-evaluation-v2",
                    "status": "INCOMPLETE",
                    "arm": arm,
                    "planned": len(evaluation_tasks),
                    "attempted": 0,
                    "scored": 0,
                    "unknown": len(evaluation_tasks),
                    "successes": 0,
                    "stopped_before_verifier": True,
                    "stop_reason": "arm_setup_failure",
                    "input": {
                        "arm": arm,
                        "policy_snapshot_id": f"{base_snapshot['policy_snapshot_id']}-{arm}",
                        "tasks": [task.task_id for task in evaluation_tasks],
                        "evaluation_seed": int(
                            config.evaluation.get("generation", {}).get("seed", 17)
                        ),
                        "evaluation_mode": "greedy",
                        "require_qualified_evidence": True,
                    },
                    "error": error,
                }
                atomic_write_json(
                    context.run_dir / "evaluation" / arm / "arm-manifest.json",
                    evaluations[arm],
                )
                failure = {"arm": arm, **error}
            finally:
                if active_policy is not None and hasattr(active_policy, "close"):
                    try:
                        guard.run_callable(active_policy.close, phase=f"evaluation:{arm}:cleanup")
                    finally:
                        active_policy = None
            aggregates[arm] = aggregate_source(
                source_root=context.run_dir,
                verifier_index=context.run_dir / "evaluation" / arm / "verifier-results.jsonl",
                identity_field="task_id",
                expected_count=len(evaluation_tasks),
                source_kind=f"T024-E026-real-validation-{arm}",
            )
            if failure is not None or evaluations[arm].get("stopped_before_verifier"):
                break

        for arm in ("Base", "N", "RL"):
            if arm not in aggregates:
                aggregates[arm] = {
                    "planned": len(evaluation_tasks),
                    "attempted": 0,
                    "scored": 0,
                    "unknown": len(evaluation_tasks),
                    "qualified": 0,
                    "successes": 0,
                    "success_rate": None,
                    "entries": [],
                }
        comparison = three_arm_comparison(
            aggregates["Base"],
            aggregates["N"],
            aggregates["RL"],
            expected_count=len(evaluation_tasks),
        )
        atomic_write_json(context.run_dir / "metrics.json", aggregates)
        atomic_write_json(context.run_dir / "comparison.json", comparison)
        result = {
            "schema_version": "grpo-frozen-evaluation-run-v1",
            "status": "COMPLETED"
            if comparison["status"] == "COMPLETE"
            else "INCOMPLETE_EVALUATION",
            "arms": evaluations,
            "aggregates": aggregates,
            "comparison": comparison,
            "evidence": evidence_journal_summary(context.run_dir),
            "budget": guard.current("run-complete").as_dict(),
            "failure": failure,
        }
        final_status = result["status"]
        atomic_write_json(context.run_dir / "run-status.json", result)
        if failure is not None:
            raise RuntimeError(f"frozen_evaluation_failed:{failure}")
        return result
    except BudgetError as exc:
        budget_stopped = True
        failure = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(
            context.run_dir / "run-status.json",
            {
                "schema_version": "grpo-frozen-evaluation-run-v1",
                "status": "BUDGET_STOP",
                "error": failure,
            },
        )
        raise
    finally:
        if active_policy is not None and hasattr(active_policy, "close"):
            try:
                active_policy.close()
            except Exception:
                pass
        try:
            guard.close_job(
                status=("BUDGET_STOP" if budget_stopped else final_status),
                reason=failure.get("message") if failure else None,
            )
        except Exception:
            pass
