#!/usr/bin/env bash
# Validation-only SFT adapter benchmark using the existing Agent Runtime.
#SBATCH --job-name=earl-sft-eval
#SBATCH --account=pi_daipeishan
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-earl-sft-eval-%j.out
#SBATCH --signal=B:TERM@30

set -u -o pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}"
adapter_dir="${EARL_SFT_ADAPTER:?set EARL_SFT_ADAPTER to the completed adapter directory}"
run_dir="artifacts/sft-agent-benchmark-${SLURM_JOB_ID:?}"

finalize() {
  local status=$?
  trap - EXIT TERM INT
  mkdir -p "$run_dir"
  local slurm_output="slurm-earl-sft-eval-${SLURM_JOB_ID}.out"
  if [[ -f "$slurm_output" ]]; then
    cp "$slurm_output" "$run_dir/raw-slurm-output" || true
  else
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/raw-slurm-output" 2>&1 || true
  fi
  if [[ ! -f "$run_dir/benchmark-report.json" ]]; then
    "$python_bin" scripts/evaluation/base_agent_benchmark.py \
      --manifest benchmarks/m2a/manifest.json \
      --database-root artifacts/m2a-databases \
      --model-config configs/model/qwen2.5-1.5b.toml \
      --agent-config configs/agent/qwen.toml \
      --split validation \
      --output "$run_dir" \
      --finalize-incomplete \
      --error-type supervisor_exit_code \
      --error-message "SFT evaluation supervisor exited with status $status" \
      > "$run_dir/finalizer-output.json" 2>&1 || true
  fi
  exit "$status"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

status=0
"$python_bin" scripts/inference/run_qwen.py base-benchmark \
  --cache "$cache_dir" \
  --model-config configs/model/qwen2.5-1.5b.toml \
  --agent-config configs/agent/qwen.toml \
  --manifest benchmarks/m2a/manifest.json \
  --database-root artifacts/m2a-databases \
  --split validation \
  --adapter "$adapter_dir" \
  --output "$run_dir/runtime" || status=$?
exit "$status"
