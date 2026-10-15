#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 -m venv .venv
.venv/bin/python -m pip --disable-pip-version-check install --index-url https://pypi.org/simple -r requirements/cpu-py312-linux.lock
.venv/bin/python -m pip --disable-pip-version-check install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
