#!/usr/bin/env bash
# Raw Transformers GPU diagnostic only. This job must not run Protocol, Agent, Base,
# SFT, or GRPO evaluation.
#SBATCH --job-name=earl-gpu-inference-diagnostic
#SBATCH --account=pi_daipeishan
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@30

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
run_dir="artifacts/gpu-inference-diagnostic-${SLURM_JOB_ID:?}"
tmp_dir="${SLURM_TMPDIR:-$PWD/artifacts/.tmp-gpu-inference-diagnostic-${SLURM_JOB_ID}}"
mkdir -p "$tmp_dir"
export TMPDIR="$tmp_dir"

exec "$python_bin" scripts/inference/gpu_inference_diagnostic.py \
  --cache "${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}" \
  --model-config configs/model/qwen2.5-1.5b.toml \
  --output "$run_dir"
