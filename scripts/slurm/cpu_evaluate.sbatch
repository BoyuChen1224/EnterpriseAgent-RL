#!/usr/bin/env bash
#SBATCH --job-name=earl-cpu-evaluate
#SBATCH --partition=freecpuQ
#SBATCH --account=free
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PYCODE'
from pathlib import Path
from enterprise_agent_rl.agent import DeterministicPolicy
from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import benchmark_tasks, initialize_database
from enterprise_agent_rl.evaluation import evaluate, compare_reports
import json, os
root=Path("artifacts")/f"slurm-evaluation-{os.environ['SLURM_JOB_ID']}"
root.mkdir(parents=True,exist_ok=True)
database=initialize_database(root/"business.sqlite3")
tasks=[t for t in benchmark_tasks() if t.split=="test"]
reports=[]
for name,recover in [("fixture",False),("recovery",True)]:
    report=evaluate(DeterministicPolicy(recover_error=recover),tasks,database,Settings(),root/name)
    report["execution"]={"slurm_job_id":os.environ["SLURM_JOB_ID"],
                         "node":os.environ.get("SLURMD_NODENAME"),
                         "experiment_kind":"CPU deterministic fixture; no LLM executed"}
    (root/name/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"policy":name,"metrics":report["metrics"]},ensure_ascii=False))
    assert report["metrics"]["task_success_rate"]==1, "Incomplete fixture evaluation; preserve artifacts"
    reports.append(report)
(root/"comparison.json").write_text(json.dumps(compare_reports(reports),ensure_ascii=False,indent=2)+"\n")
print(f"Validated CPU reports: {root}")
PYCODE
