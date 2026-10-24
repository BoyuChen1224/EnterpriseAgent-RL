#!/usr/bin/env bash
# T-033/E-030: one rollout-only Qwen3 job.  Compatible partitions are
# gpu2Q/gpu4Q/gpu4RQ/gpu8Q; change partition only for this pending job, never
# broaden the task scope or submit a second executable job concurrently.
#SBATCH --job-name=earl-qwen3-t033
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
#SBATCH --output=slurm-qwen3-t033-%j.out

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

run_dir="${EARL_T033_RUN:-$PWD/artifacts/qwen3-grpo-rollout-signal/t033-v1}"
config="${EARL_T033_CONFIG:-$PWD/configs/evaluation/qwen3-grpo-rollout.toml}"
panel_manifest="${EARL_T033_PANEL_MANIFEST:-$PWD/artifacts/qwen3-base-lora-pilot/t031-v1/panel-manifest.json}"
source_db_root="${EARL_T033_SOURCE_DB_ROOT:-$PWD/artifacts/qwen3-base-lora-pilot/t031-v1/base/databases}"
cache_dir="${EARL_T033_CACHE:-$PWD/artifacts/model-cache-qwen3-8b}"
parent_adapter="${EARL_T033_PARENT_ADAPTER:-$PWD/artifacts/qwen3-runtime-sft-comparison/t032-v1/training/final-adapter}"
python_bin="${EARL_QWEN3_PYTHON:-$PWD/.venv-qwen3/bin/python}"
mkdir -p "$run_dir/logs"

visible_gpu_count=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi

record_stage() {
    local return_code="$1"
    local evidence="$run_dir/logs/slurm-root-${SLURM_JOB_ID}.txt"
    if command -v sacct >/dev/null 2>&1; then
        sacct -X -j "$SLURM_JOB_ID" \
            --format=JobID,State,ElapsedRaw,Start,End,AllocTRES,AllocCPUS,NodeList \
            --parsable2 > "$evidence" 2>&1 || true
    else
        env | sort | grep -E '^(SLURM_|CUDA_VISIBLE_DEVICES=)' > "$evidence" || true
    fi
    "$python_bin" scripts/evaluation/run_qwen3_grpo_rollout.py \
        --stage record-command \
        --repo-root "$PWD" \
        --run "$run_dir" \
        --config "$config" \
        --panel-manifest "$panel_manifest" \
        --source-db-root "$source_db_root" \
        --cache "$cache_dir" \
        --parent-adapter "$parent_adapter" \
        --command-id "slurm-rollout-${SLURM_JOB_ID}" \
        --command "${python_bin} scripts/evaluation/run_qwen3_grpo_rollout.py --stage rollout --repo-root ${PWD} --run ${run_dir} --config ${config} --panel-manifest ${panel_manifest} --source-db-root ${source_db_root} --cache ${cache_dir} --parent-adapter ${parent_adapter}" \
        --exit-code "$return_code" \
        --stdout-path "$run_dir/logs/rollout-${SLURM_JOB_ID}.stdout" \
        --stderr-path "$run_dir/logs/rollout-${SLURM_JOB_ID}.stderr" \
        --scope gpu-rollout-wrapper || true
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
    echo "T-033 requires exactly one visible GPU before importing torch" >&2
    exit 2
fi

set +e
"$python_bin" scripts/evaluation/run_qwen3_grpo_rollout.py \
    --stage rollout \
    --repo-root "$PWD" \
    --run "$run_dir" \
    --config "$config" \
    --panel-manifest "$panel_manifest" \
    --source-db-root "$source_db_root" \
    --cache "$cache_dir" \
    --parent-adapter "$parent_adapter" \
    --stdout-path "$run_dir/logs/rollout-${SLURM_JOB_ID}.stdout" \
    --stderr-path "$run_dir/logs/rollout-${SLURM_JOB_ID}.stderr" \
    --scope gpu-rollout > "$run_dir/logs/rollout-${SLURM_JOB_ID}.stdout" \
    2> "$run_dir/logs/rollout-${SLURM_JOB_ID}.stderr"
return_code="$?"
set -e
exit "$return_code"
