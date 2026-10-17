#!/usr/bin/env bash
# One-shot teacher-forced SFT behavior attribution.  No training, GRPO,
# validation/test benchmark, Environment, Verifier, Reward, or SQL execution.
# The caller must pass the already-created unique run directory and a passing
# CPU preflight.  This script never submits another job.
#SBATCH --job-name=earl-sft-behavior-attribution
#SBATCH --account=pi_daipeishan
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@30

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"

run_dir="${SFT_ATTRIBUTION_RUN_DIR:?pass SFT_ATTRIBUTION_RUN_DIR for the unique run directory}"
python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$run_dir/preflight.json" ]]; then
  echo "missing preflight.json: $run_dir" >&2
  exit 2
fi

exec "$python_bin" scripts/diagnostics/sft_behavior_attribution.py \
  --mode gpu \
  --root "$PWD" \
  --run-dir "$run_dir" \
  --dataset "$PWD/artifacts/sft-data-train-teacher-v2/dataset.jsonl" \
  --metadata "$PWD/artifacts/sft-data-train-teacher-v2/dataset-metadata.json" \
  --cache "$cache_dir" \
  --model-config "$PWD/configs/model/qwen2.5-1.5b.toml" \
  --timeout-s 120
