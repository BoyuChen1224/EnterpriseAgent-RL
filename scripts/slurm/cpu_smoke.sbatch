#!/usr/bin/env bash
#SBATCH --job-name=earl-cpu-smoke
#SBATCH --partition=freecpuQ
#SBATCH --account=free
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@30
set -euo pipefail
trap 'echo "CPU smoke interrupted by Slurm"; exit 143' TERM INT
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluation/cpu_smoke.py
