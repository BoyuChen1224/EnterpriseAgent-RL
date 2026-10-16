#!/usr/bin/env bash
# Formal Base Agent benchmark: real Qwen, validation split only, one V100.
# Business failures remain scored episodes; runtime/infrastructure failures make
# the preserved benchmark report BLOCKED.
#SBATCH --job-name=earl-base-agent-benchmark
#SBATCH --account=pi_daipeishan
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@30

set -u -o pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}"
run_dir="artifacts/base-agent-benchmark-${SLURM_JOB_ID:?}"
runtime_dir="$run_dir/runtime"
mkdir -p "$run_dir"

finalize_artifacts() {
  local exit_code=$?
  trap - EXIT TERM INT
  # Keep the scheduler's combined stdout/stderr beside structured artifacts.
  local slurm_output="slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  if [[ -f "$slurm_output" ]]; then
    cp "$slurm_output" "$run_dir/raw-slurm-output" || true
  else
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/raw-slurm-output" 2>&1 || true
  fi

  # If the Python supervisor died before its own finalizer ran, preserve a
  # BLOCKED report without rerunning any task or changing benchmark inputs.
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
      --error-message "run_qwen.py exited with status $exit_code" \
      > "$run_dir/finalizer-output.json" 2>&1 || true
  fi
  exit "$exit_code"
}
trap finalize_artifacts EXIT
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
  --output "$runtime_dir" || status=$?

exit "$status"
