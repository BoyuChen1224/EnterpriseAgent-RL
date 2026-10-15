"""Check allocation membership before importing torch or touching a GPU."""

from __future__ import annotations

import os
import socket
import subprocess


def require_compute_node(*, gpu: bool = True) -> None:
    job = os.environ.get("SLURM_JOB_ID", "")
    nodes = os.environ.get("SLURM_JOB_NODELIST", "")
    if not job.isdigit() or not nodes:
        raise RuntimeError("requires_slurm_compute_node")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodes],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    hosts = {h.split(".")[0] for h in result.stdout.split()}
    if socket.gethostname().split(".")[0] not in hosts:
        raise RuntimeError("host_not_in_slurm_allocation")
    if gpu and not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise RuntimeError("requires_slurm_gpu_allocation")
