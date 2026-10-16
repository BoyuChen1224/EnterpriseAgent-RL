#!/usr/bin/env bash
# Protocol smoke: real Qwen JSON Action plus a real environment tool call.
# Paid queue; user authorization required before submission.
#SBATCH --job-name=earl-qwen-protocol
#SBATCH --partition=gpu2Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@30
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
exec bash scripts/slurm/qwen_job.sh protocol
