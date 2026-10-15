#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?must run in Slurm}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_bin="${EARL_GPU_PYTHON:-.venv-gpu/bin/python}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache}"
run_dir="artifacts/qwen-${1:?mode}-${SLURM_JOB_ID}"
child_pid=''
cleanup() {
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
}
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT
# Python owns and cleans its model process group; Slurm also reaps job steps on cancellation.
"$python_bin" scripts/inference/run_qwen.py "$1" --cache "$cache_dir" --output "$run_dir" \
    --split "${EARL_EVAL_SPLIT:-validation}" \
    --database-root "${EARL_DATABASE_ROOT:-artifacts/m2a-databases}" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
child_pid=''
exit "$status"
