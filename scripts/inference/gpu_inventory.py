"""Run exclusively in an allocated compute node; retain failed probes as evidence."""

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from enterprise_agent_rl.gpu_guard import require_compute_node

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
require_compute_node()
args.output.mkdir(parents=True, exist_ok=False)


def probe(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {
            "command": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc)}


report = {
    "node": platform.node(),
    "python": platform.python_version(),
    "glibc": platform.libc_ver(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "gpu": probe(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"]
    ),
    "nvidia_smi": probe(["nvidia-smi"]),
    "nvcc": probe(["nvcc", "--version"]),
    "job": probe(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
    "account": probe(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            "user=" + os.environ["USER"],
            "format=Account,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmit,MaxWall",
        ]
    ),
    "quota": probe(["lfs", "quota", "-u", os.environ["USER"], "/public"]),
    "cache": os.environ.get("EARL_MODEL_CACHE", "artifacts/model-cache"),
    "disk": probe(["df", "-h", "."]),
}
if importlib.util.find_spec("torch"):
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "architectures": torch.cuda.get_arch_list(),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            report["torch"].update(
                name=torch.cuda.get_device_name(0),
                cc=torch.cuda.get_device_capability(0),
                memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            )
        # CUDA driver API version is distinct from runtime and toolkit versions.
        version = ctypes.c_int()
        driver = ctypes.CDLL("libcuda.so.1")
        status = driver.cuDriverGetVersion(ctypes.byref(version))
        report["cuda_driver_api"] = {"status": status, "version": version.value}
        runtime_version = ctypes.c_int()
        runtime = ctypes.CDLL("libcudart.so.12")
        status = runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
        report["cuda_runtime_api"] = {"status": status, "version": runtime_version.value}
    except Exception as exc:
        report["torch_probe_error"] = {"type": type(exc).__name__, "message": str(exc)}
else:
    report["torch"] = {"status": "not_installed_in_selected_python"}
image = os.environ.get("EARL_AUDIT_IMAGE")
if shutil.which("singularity"):
    report["singularity"] = probe(["singularity", "--version"])
    report["gpu_passthrough"] = (
        probe(["singularity", "exec", "--nv", image, "nvidia-smi"])
        if image
        else {"status": "pending_no_local_image"}
    )
else:
    report["gpu_passthrough"] = {"status": "pending_singularity_unavailable"}
(args.output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
