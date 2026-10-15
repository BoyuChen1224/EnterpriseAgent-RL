#!/usr/bin/env bash
#SBATCH --job-name=earl-gpu-inventory
#SBATCH --partition=gpu2Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=slurm-%x-%j.out
# Paid queue: prepare only; submit only after explicit user authorization.
# Supply --account and --qos explicitly at submission, never infer free GPU access.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if type module >/dev/null 2>&1; then module load singularity/3.11.2; fi
python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
srun "$python_bin" scripts/inference/gpu_inventory.py --output "artifacts/gpu-inventory-${SLURM_JOB_ID:?}"
