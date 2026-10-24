#!/usr/bin/env bash
# T-032/E-029.  Submit stages separately so training/reload are independent
# processes and only the comparison stage holds two single-GPU workers.
# For comparison override: --gres=gpu:tesla:2 --cpus-per-task=8 --mem=96G
# --time=01:00:00.  Compatible partitions: gpu2Q/gpu4Q/gpu4RQ/gpu8Q.
#SBATCH --job-name=earl-qwen3-t032
#SBATCH --partition=gpu2Q
#SBATCH --account=pi_daipeishan
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-qwen3-t032-%j.out
#SBATCH --signal=B:TERM@120
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

run_dir="${EARL_QWEN3_RUN:-$PWD/artifacts/qwen3-runtime-sft-comparison/t032-v1}"
cache_dir="${EARL_MODEL_CACHE:-$PWD/artifacts/model-cache-qwen3-8b}"
python_bin="${EARL_QWEN3_PYTHON:-$PWD/.venv-qwen3/bin/python}"
stage="${T032_STAGE:-train}"
stage_started="$(date +%s.%N)"
mkdir -p "$run_dir/logs"

visible_gpu_count=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi

cleanup_servers() {
    for pid in "${server_pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    server_pids=()
}

record_stage() {
    local code="$1"
    local status="FAILED"
    [[ "$code" -eq 0 ]] && status="COMPLETED"
    local ended elapsed evidence
    ended="$(date +%s.%N)"
    elapsed="$(awk -v a="$stage_started" -v b="$ended" 'BEGIN {print b-a}')"
    evidence="$run_dir/logs/slurm-${SLURM_JOB_ID}.txt"
    if command -v sacct >/dev/null 2>&1; then
        sacct -X -j "$SLURM_JOB_ID" \
            --format=JobID,State,ElapsedRaw,Start,End,AllocTRES,AllocCPUS,AllocGRES,NodeList \
            --parsable2 > "$evidence" 2>&1 || true
    else
        env | sort | grep -E '^(SLURM_|CUDA_VISIBLE_DEVICES=)' > "$evidence" || true
    fi
    "$python_bin" scripts/evaluation/record_t032_allocation.py \
        --run "$run_dir" --stage "$stage" --status "$status" --exit-code "$code" \
        --elapsed-s "$elapsed" --gpu-count "$visible_gpu_count" \
        --slurm-evidence "$evidence" || true
}

finish() {
    local code="$?"
    cleanup_servers
    record_stage "$code"
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "$stage" == "train" ]]; then
    if [[ "$visible_gpu_count" -ne 1 ]]; then
        echo "train stage requires one visible GPU" >&2
        exit 2
    fi
    "$python_bin" scripts/training/run_qwen3_sft.py \
        --stage train --output "$run_dir" --cache "$cache_dir" \
        --model-config configs/model/qwen3-8b.toml \
        --training-config configs/training/qwen3-runtime-sft.toml \
        > "$run_dir/logs/train.log" 2>&1
    exit 0
fi

if [[ "$stage" == "reload" ]]; then
    if [[ "$visible_gpu_count" -ne 1 ]]; then
        echo "reload stage requires one visible GPU" >&2
        exit 2
    fi
    "$python_bin" scripts/training/run_qwen3_sft.py \
        --stage reload --output "$run_dir" --cache "$cache_dir" \
        --model-config configs/model/qwen3-8b.toml \
        --training-config configs/training/qwen3-runtime-sft.toml \
        > "$run_dir/logs/reload.log" 2>&1
    exit 0
fi

if [[ "$stage" == "compare" ]]; then
    if [[ "$visible_gpu_count" -lt 2 ]]; then
        echo "compare stage requires two visible GPUs" >&2
        exit 2
    fi
    if [[ ! -f "$run_dir/evaluation/validation/validation-manifest.json" ]]; then
        echo "validation manifest is missing; run --prepare first" >&2
        exit 2
    fi
    if [[ ! -f "$run_dir/training/reload-check.json" ]]; then
        echo "fresh reload check is missing" >&2
        exit 2
    fi
    IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
    server_pids=()
    ports=("$((19000 + SLURM_JOB_ID % 500))" "$((19500 + SLURM_JOB_ID % 500))")
    arm_names=(base sft)
    for index in 0 1; do
        arm="${arm_names[$index]}"
        engine_dir="$run_dir/evaluation/$arm/engine"
        mkdir -p "$engine_dir"
        adapter_args=()
        if [[ "$arm" == "sft" ]]; then
            adapter_args+=(--adapter "$run_dir/training/final-adapter")
        fi
        CUDA_VISIBLE_DEVICES="${allocated_gpus[$index]}" "$python_bin" \
            -m enterprise_agent_rl.model_server \
            --config configs/model/qwen3-8b.toml --cache "$cache_dir" \
            --output "$engine_dir" --port "${ports[$index]}" "${adapter_args[@]}" \
            > "$run_dir/logs/${arm}-server.log" 2>&1 &
        server_pids+=("$!")
    done
    for attempt in $(seq 1 360); do
        ready=0
        for index in 0 1; do
            if curl --noproxy '*' -fsS "http://127.0.0.1:${ports[$index]}/health" >/dev/null 2>&1; then
                ready=$((ready + 1))
            fi
        done
        [[ "$ready" -eq 2 ]] && break
        sleep 1
        if [[ "$attempt" -eq 360 ]]; then
            echo "model server health timeout" >&2
            exit 2
        fi
    done
    worker_pids=()
    worker_status=0
    for index in 0 1; do
        arm="${arm_names[$index]}"
        adapter_args=()
        if [[ "$arm" == "sft" ]]; then
            adapter_args+=(--adapter "$run_dir/training/final-adapter")
        fi
        CUDA_VISIBLE_DEVICES="${allocated_gpus[$index]}" "$python_bin" \
            scripts/evaluation/run_qwen3_runtime_panel.py \
            --output "$run_dir" --arm "$arm" --worker-index 0 --worker-count 1 \
            --endpoint "http://127.0.0.1:${ports[$index]}/v1/chat/completions" \
            --engine-dir "$run_dir/evaluation/$arm/engine" "${adapter_args[@]}" \
            > "$run_dir/logs/${arm}-worker.log" 2>&1 &
        worker_pids+=("$!")
    done
    for pid in "${worker_pids[@]}"; do
        if ! wait "$pid"; then
            worker_status=2
        fi
    done
    cleanup_servers
    "$python_bin" scripts/evaluation/run_qwen3_runtime_panel.py \
        --output "$run_dir" --arm base --merge > "$run_dir/logs/base-merge.log" 2>&1 || worker_status=2
    "$python_bin" scripts/evaluation/run_qwen3_runtime_panel.py \
        --output "$run_dir" --arm sft --merge > "$run_dir/logs/sft-merge.log" 2>&1 || worker_status=2
    "$python_bin" scripts/evaluation/run_qwen3_runtime_panel.py \
        --output "$run_dir" --compare > "$run_dir/logs/compare.log" 2>&1 || worker_status=2
    exit "$worker_status"
fi

echo "unknown T032_STAGE=$stage" >&2
exit 2
