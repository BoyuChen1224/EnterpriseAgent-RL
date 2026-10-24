#!/bin/bash
# T-024/E-026 frozen Base/N/RL evaluation entrypoint.
# Override the partition at submission time after checking live inventory.
#SBATCH --account=pi_daipeishan
#SBATCH --qos=gpuq
#SBATCH --partition=gpu4RQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=50:00

set -euo pipefail

REPO_ROOT="${EARL_REPO_ROOT:-$(pwd)}"
CONFIG_PATH="${EARL_GRPO_CONFIG:-$REPO_ROOT/configs/training/grpo-frozen-evaluation.toml}"
RUN_DIR="${EARL_GRPO_RUN:-$REPO_ROOT/artifacts/frozen-base-sft-rl-evaluation/t024-v1}"
COMMAND="${EARL_GRPO_COMMAND:-evaluate-frozen}"
PYTHON_BIN="${EARL_GRPO_PYTHON:-$REPO_ROOT/.venv-gpu/bin/python}"

if [ -e "$RUN_DIR" ]; then
  suffix=2
  while [ -e "${RUN_DIR}-r${suffix}" ]; do
    suffix=$((suffix + 1))
  done
  RUN_DIR="${RUN_DIR}-r${suffix}"
fi

export EARL_FROZEN_RUN_RESERVED=1

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$RUN_DIR/logs"
{
  echo "repo_root=$REPO_ROOT"
  echo "config_path=$CONFIG_PATH"
  echo "run_dir=$RUN_DIR"
  echo "command=$COMMAND"
  echo "python_bin=$PYTHON_BIN"
  echo "slurm_job_id=${SLURM_JOB_ID:-UNKNOWN}"
  command -v nvidia-smi || true
} > "$RUN_DIR/logs/launcher-env.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "$RUN_DIR/logs/nvidia-smi-L.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    > "$RUN_DIR/logs/nvidia-smi-summary.csv"
  nvidia-smi -q > "$RUN_DIR/logs/nvidia-smi-query.txt"
else
  echo "nvidia-smi unavailable; torch device contract remains mandatory" \
    > "$RUN_DIR/logs/nvidia-smi-unavailable.txt"
fi

start_epoch=$(date +%s)
set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/training/run_grpo.py" \
  "$COMMAND" \
  --repo-root "$REPO_ROOT" \
  --config "$CONFIG_PATH" \
  --run "$RUN_DIR" \
  > "$RUN_DIR/logs/stdout.txt" \
  2> "$RUN_DIR/logs/stderr.txt"
status=$?
set -e
end_epoch=$(date +%s)
cat > "$RUN_DIR/logs/wrapper-elapsed.json" <<EOF
{"job_id":"${SLURM_JOB_ID:-UNKNOWN}","command":"$COMMAND","start_epoch":$start_epoch,"end_epoch":$end_epoch,"elapsed_seconds":$((end_epoch - start_epoch)),"exit_code":$status}
EOF
exit "$status"
