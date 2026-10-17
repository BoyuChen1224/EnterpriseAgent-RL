#!/usr/bin/env bash
# One-shot train-only SFT Runtime protocol ablation.
# No training, GRPO, validation benchmark, or test split access.
# The caller supplies the already-selected partition and unique run directory.
# This script never retries or submits another job.
#SBATCH --job-name=earl-sft-runtime-ablation
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

run_dir="${SFT_RUNTIME_ABLATION_RUN_DIR:?pass SFT_RUNTIME_ABLATION_RUN_DIR}"
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

set +e
"$python_bin" scripts/diagnostics/sft_runtime_protocol_ablation.py \
  --mode gpu \
  --root "$PWD" \
  --run-dir "$run_dir" \
  --dataset "$PWD/artifacts/sft-data-train-teacher-v2/dataset.jsonl" \
  --metadata "$PWD/artifacts/sft-data-train-teacher-v2/dataset-metadata.json" \
  --manifest "$PWD/benchmarks/m2a/manifest.json" \
  --database-root "$PWD/artifacts/m2a-databases-final" \
  --adapter "$PWD/artifacts/sft-pilot-20390015/adapter" \
  --cache "$cache_dir" \
  --agent-config "$PWD/configs/agent/qwen.toml" \
  --model-config "$PWD/configs/model/qwen2.5-1.5b.toml"
status=$?
set -e

"$python_bin" scripts/diagnostics/sft_runtime_protocol_ablation.py \
  --mode finalize \
  --run-dir "$run_dir" \
  --exit-code "$status" || true
exit "$status"
