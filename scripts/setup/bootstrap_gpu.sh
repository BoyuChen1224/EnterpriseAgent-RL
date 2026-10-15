#!/usr/bin/env bash
# Installs dependencies only; never probes GPU or loads weights on the login node.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --index-url https://pypi.org/simple -r requirements/gpu-py312-linux.lock
.venv-gpu/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-gpu/bin/python -m pip check
