#!/usr/bin/env bash
# T-027: one bounded SFT-N model + Agent API lifecycle attempt.
# Override --partition at submission time only with a compatible V100 queue.
# The Python supervisor owns and reaps the two loopback service process groups.
#SBATCH --job-name=earl-sft-n-api
#SBATCH --partition=gpu4RQ
#SBATCH --account=pi_daipeishan
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
python_bin="$repo_root/.venv-gpu/bin/python"
profile="$repo_root/configs/serving/sft-n.toml"
run_dir="${EARL_RUN_DIR:-$repo_root/artifacts/service-lifecycle/t027-v1-slurm-${SLURM_JOB_ID:-local-$$}}"
if [[ "$run_dir" != /* ]]; then
  run_dir="$repo_root/$run_dir"
fi
if [[ -e "$run_dir" ]]; then
  echo "run directory already exists; choose a unique EARL_RUN_DIR: $run_dir" >&2
  exit 2
fi

if [[ ! -x "$python_bin" ]]; then
  echo "missing GPU virtualenv interpreter: $python_bin" >&2
  exit 2
fi

supervisor_pid=""
forward_signal() {
  local signal_name="$1"
  if [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    kill "-$signal_name" "$supervisor_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

set +e
"$python_bin" "$repo_root/scripts/serving/run_sft_n_api.py" \
  --repo-root "$repo_root" \
  --profile "$profile" \
  --run-dir "$run_dir" \
  --mode "${EARL_MODE:-smoke}" \
  --soft-cutoff "${EARL_SOFT_CUTOFF:-1320}" \
  --cleanup-reserve "${EARL_CLEANUP_RESERVE:-60}" \
  --auto-ports &
supervisor_pid=$!
rc=1
while true; do
  wait "$supervisor_pid"
  rc=$?
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    break
  fi
done
set -e
trap - TERM INT

# Keep the scheduler's own allocation reading separate from the supervisor's
# lower-bound wall clock.  A site may not expose sacct during the job; that is
# recorded as unavailable rather than fabricated as zero.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sacct >/dev/null 2>&1; then
  sacct -X -j "$SLURM_JOB_ID" --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES -P \
    >"$run_dir/slurm-sacct.txt" 2>&1 || true
fi

exit "$rc"
