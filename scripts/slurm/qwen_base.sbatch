#!/usr/bin/env bash
#SBATCH --job-name=earl-qwen-base
#SBATCH --partition=gpu2Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@30
# Paid queue; user authorization required before submission. validation split by default.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
exec bash scripts/slurm/qwen_job.sh base
