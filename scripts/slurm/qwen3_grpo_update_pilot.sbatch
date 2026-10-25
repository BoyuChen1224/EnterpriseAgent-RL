#!/usr/bin/env bash
# T-034/E-031.  One executable job at a time; stages are submitted separately
# so train/reload/compare are independent processes.  Compatible partitions:
# gpu2Q/gpu4Q/gpu4RQ/gpu8Q.  This wrapper records actual allocation evidence.
#SBATCH --job-name=earl-qwen3-t034
#SBATCH --partition=gpu2Q
#SBATCH --account=pi_daipeishan
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-qwen3-t034-%j.out

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_dir="${EARL_T034_RUN:-$PWD/artifacts/qwen3-grpo-update-pilot/t034-v1}"
config="${EARL_T034_CONFIG:-$PWD/configs/training/qwen3-grpo-update-pilot.toml}"
python_bin="${EARL_QWEN3_PYTHON:-$PWD/.venv-qwen3/bin/python}"
stage="${T034_STAGE:-train}"
stage_started="$(date +%s.%N)"
mkdir -p "$run_dir/logs"

visible_gpu_count=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi

record_stage() {
    local return_code="$1"
    local status="FAILED"
    [[ "$return_code" -eq 0 ]] && status="COMPLETED"
    local ended elapsed evidence
    ended="$(date +%s.%N)"
    elapsed="$(awk -v a="$stage_started" -v b="$ended" 'BEGIN {print b-a}')"
    evidence="$run_dir/logs/slurm-root-${SLURM_JOB_ID}.txt"
    if command -v sacct >/dev/null 2>&1; then
        sacct -X -j "$SLURM_JOB_ID" \
            --format=JobID,State,ElapsedRaw,Start,End,AllocTRES,AllocCPUS,NodeList \
            --parsable2 > "$evidence" 2>&1 || true
    else
        env | sort | grep -E '^(SLURM_|CUDA_VISIBLE_DEVICES=)' > "$evidence" || true
    fi
    set +e
    T034_STAGE="$stage" "$python_bin" scripts/training/run_qwen3_grpo_pilot.py \
        --stage record-allocation --output "$run_dir" --config "$config" \
        --record-status "$status" --record-exit-code "$return_code" \
        --record-elapsed-s "$elapsed" --record-gpu-count "$visible_gpu_count" \
        --record-slurm-evidence "$evidence"
    set -e
}

finish() {
    local return_code="$?"
    record_stage "$return_code"
    exit "$return_code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "$visible_gpu_count" -ne 1 ]]; then
    echo "T-034 GPU stage requires exactly one visible GPU" >&2
    exit 2
fi

set +e
"$python_bin" scripts/training/run_qwen3_grpo_pilot.py \
    --stage "$stage" --output "$run_dir" --config "$config" \
    > "$run_dir/logs/${stage}-${SLURM_JOB_ID}.stdout" \
    2> "$run_dir/logs/${stage}-${SLURM_JOB_ID}.stderr"
return_code="$?"
set -e
exit "$return_code"
