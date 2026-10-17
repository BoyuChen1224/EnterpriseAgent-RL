#!/usr/bin/env bash
# One fixed, single-GPU LoRA SFT pilot on the locked Qwen/V100 stack.
#SBATCH --job-name=earl-sft-train
#SBATCH --account=pi_daipeishan
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-earl-sft-train-%j.out
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
run_dir="artifacts/sft-pilot-${SLURM_JOB_ID:?}"

finalize() {
  local status=$?
  trap - EXIT TERM INT
  mkdir -p "$run_dir"
  local slurm_output="slurm-earl-sft-train-${SLURM_JOB_ID}.out"
  if [[ -f "$slurm_output" ]]; then
    cp "$slurm_output" "$run_dir/raw-slurm-output" || true
  else
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/raw-slurm-output" 2>&1 || true
  fi
  exit "$status"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

status=0
"$python_bin" scripts/training/run_sft.py \
  --dataset artifacts/sft-data-train-teacher-v2/dataset.jsonl \
  --metadata artifacts/sft-data-train-teacher-v2/dataset-metadata.json \
  --cache "$cache_dir" \
  --output "$run_dir" \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 0.0001 \
  --warmup-ratio 0.05 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --seed 42 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 || status=$?
exit "$status"
