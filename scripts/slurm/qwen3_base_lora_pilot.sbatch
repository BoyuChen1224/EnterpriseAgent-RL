#!/usr/bin/env bash
# T-031/E-028. Submit one stage at a time; override --gres/--mem for the
# one-GPU LoRA stage when submitting it.
#SBATCH --job-name=earl-qwen3-t031
#SBATCH --partition=gpu2Q
#SBATCH --account=pi_daipeishan
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-qwen3-t031-%j.out
#SBATCH --signal=B:TERM@120
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

run_dir="${EARL_QWEN3_RUN:-$PWD/artifacts/qwen3-base-lora-pilot/t031-v1}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache-qwen3-8b}"
python_bin="${EARL_QWEN3_PYTHON:-$PWD/.venv-qwen3/bin/python}"
stage="${PILOT_STAGE:-base}"
lora_mode="${PILOT_LORA_MODE:-train}"
mkdir -p "$run_dir/logs"
stage_started="$(date +%s.%N)"
gpu_count=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi
record_stage() {
    local status="$1"
    local code="$2"
    local ended elapsed
    local ledger_stage="$stage"
    if [[ "$stage" == "lora" ]]; then
        ledger_stage="lora-${lora_mode}"
    fi
    ended="$(date +%s.%N)"
    elapsed="$(awk -v a="$stage_started" -v b="$ended" 'BEGIN {print b-a}')"
    "$python_bin" scripts/evaluation/record_qwen3_allocation.py \
        --run "$run_dir" --stage "$ledger_stage" --status "$status" --exit-code "$code" \
        --elapsed-s "$elapsed" --gpu-count "$gpu_count" || true
}
finish() {
    local code="$?"
    if declare -F cleanup_servers >/dev/null 2>&1; then
        cleanup_servers
    fi
    if [[ "$code" -eq 0 ]]; then
        record_stage COMPLETED 0
    else
        record_stage FAILED "$code"
    fi
    exit "$code"
}
trap finish EXIT

if [[ "$stage" == "base" ]]; then
    IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
    if [[ "${#allocated_gpus[@]}" -lt 2 ]]; then
        echo "base stage requires two allocated single-GPU workers" >&2
        exit 2
    fi
    pids=()
    ports=("$((19000 + SLURM_JOB_ID % 500))" "$((19500 + SLURM_JOB_ID % 500))")
    cleanup_servers() {
        for pid in "${pids[@]:-}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        for pid in "${pids[@]:-}"; do
            wait "$pid" 2>/dev/null || true
        done
    }
    trap 'cleanup_servers; exit 143' TERM
    trap 'cleanup_servers; exit 130' INT
    for index in 0 1; do
        worker_dir="$run_dir/base/worker-$index"
        mkdir -p "$worker_dir/engine" "$worker_dir/request" "$worker_dir/response"
        CUDA_VISIBLE_DEVICES="${allocated_gpus[$index]}" "$python_bin" -m enterprise_agent_rl.model_server \
            --config configs/model/qwen3-8b.toml --cache "$cache_dir" \
            --output "$worker_dir/engine" --port "${ports[$index]}" \
            > "$run_dir/logs/base-server-$index.log" 2>&1 &
        pids+=("$!")
    done
    # Two independent 8B loads can spend several minutes reading five local
    # shards before the HTTP health endpoint exists.  This is startup only;
    # episode and inference budgets remain fixed in the agent config.
    for attempt in $(seq 1 360); do
        ready=0
        for index in 0 1; do
            if curl --noproxy '*' -fsS "http://127.0.0.1:${ports[$index]}/health" >/dev/null 2>&1; then
                ready=$((ready + 1))
            fi
        done
        [[ "$ready" -eq 2 ]] && break
        sleep 1
        if [[ "$attempt" -eq 120 ]]; then
            echo "model server health timeout" >&2
            exit 2
        fi
    done
    worker_status=0
    for index in 0 1; do
        worker_dir="$run_dir/base/worker-$index"
        gpu="${allocated_gpus[$index]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/evaluation/run_qwen3_base_panel.py \
            --output "$run_dir" --worker-index "$index" --worker-count 2 \
            --endpoint "http://127.0.0.1:${ports[$index]}/v1/chat/completions" \
            --engine-dir "$worker_dir/engine" \
            > "$run_dir/logs/base-worker-$index.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]:2}"; do
        if ! wait "$pid"; then
            worker_status=2
        fi
    done
    cleanup_servers
    pids=()
    "$python_bin" scripts/evaluation/run_qwen3_base_panel.py --output "$run_dir" --merge
    exit "$worker_status"
fi

if [[ "$stage" == "lora" ]]; then
    if [[ "$gpu_count" -ne 1 ]]; then
        echo "lora stage must be submitted with one GPU" >&2
        exit 2
    fi
    if [[ ! -f "$run_dir/base/metrics.json" ]]; then
        echo "base panel is not sealed" >&2
        exit 2
    fi
    lora_args=(
        --cache "$cache_dir" --output "$run_dir/lora"
        --model-config configs/model/qwen3-8b.toml
        --training-config configs/training/qwen3-lora-smoke.toml
    )
    if [[ "$lora_mode" == "reload-only" ]]; then
        lora_args+=(
            --reload-only
            --prior-reload-failure
            "CUDA OOM after training model references were not released"
        )
    elif [[ "$lora_mode" != "train" ]]; then
        echo "unknown PILOT_LORA_MODE=$lora_mode" >&2
        exit 2
    fi
    "$python_bin" scripts/training/run_qwen3_lora_smoke.py "${lora_args[@]}" \
        > "$run_dir/logs/lora-${lora_mode}.log" 2>&1
    exit 0
fi

echo "unknown PILOT_STAGE=$stage" >&2
exit 2
