#!/usr/bin/env bash
#SBATCH --job-name=earl-gpu-inventory
#SBATCH --partition=gpu2Q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%x-%j.out
# Prepared only: not submitted during Milestone 0-1. Choose account/QOS when authorized.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run via sbatch, never on login'; exit 2; }
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi
command -v nvcc >/dev/null && nvcc --version
ldd --version | head -1
.venv/bin/python - <<'PYCODE'
import importlib.util
if importlib.util.find_spec("torch") is None:
    print("torch absent in selected environment; metadata only")
else:
    import torch
    print("torch", torch.__version__, "compiled CUDA", torch.version.cuda)
    if torch.cuda.is_available():
        print("device",torch.cuda.get_device_name(0))
        print("compute capability",torch.cuda.get_device_capability(0))
        print("memory bytes",torch.cuda.get_device_properties(0).total_memory)
        print("compiled architectures",torch.cuda.get_arch_list())
PYCODE
