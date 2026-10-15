#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
particle_venv="${PARTICLE_VENV:-/workspace/.venvs/particle}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/particle-pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/particle-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache/particle-xdg}"
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ ! -x "$particle_venv/bin/python" ]]; then
    python -m venv --system-site-packages "$particle_venv"
fi
cd "$project_dir"
"$particle_venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1'
"$particle_venv/bin/python" -m pip install -r requirements.txt -c constraints-cpu.txt
"$particle_venv/bin/python" -m unittest discover -s tests -v
"$particle_venv/bin/python" run_particle_pinn.py --help >/dev/null
printf 'Python environment: %s\n' "$particle_venv/bin/python"
