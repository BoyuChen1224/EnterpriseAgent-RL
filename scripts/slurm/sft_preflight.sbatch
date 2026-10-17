#!/usr/bin/env bash
# CPU-only SFT data/tokenizer preflight. No model weights, validation payload,
# or test payload are loaded.
#SBATCH --job-name=earl-sft-preflight
#SBATCH --account=free
#SBATCH --partition=freecpuQ
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-earl-sft-preflight-%j.out
#SBATCH --signal=B:TERM@30

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
exec "$python_bin" scripts/training/sft_preflight.py \
  --dataset artifacts/sft-data-train-teacher-v2/dataset.jsonl \
  --metadata artifacts/sft-data-train-teacher-v2/dataset-metadata.json \
  --cache "${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}" \
  --output artifacts/sft-data-train-teacher-v2/sft-preflight.json
