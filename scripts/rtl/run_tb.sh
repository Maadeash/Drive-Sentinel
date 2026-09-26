#!/usr/bin/env bash
# Re-run the whole RTL verification plan in one command (bash / Git Bash).
#
#   scripts/rtl/run_tb.sh                 # V-4, V-3, V-5, then V-1 (the long one)
#   scripts/rtl/run_tb.sh v4 v3 v5        # skip V-1
#   VIVADO_BIN=/d/Xilinx/2024.1/bin scripts/rtl/run_tb.sh
#
# V-1 is 16,211 windows at about 1.74 million cycles each and takes hours; it is
# run in resumable chunks across SHARDS (default 2, measured) XSim processes over disjoint window ranges
# (docs/rtl_declarations.md D-6 allows splitting and forbids subsampling).
#
# Everything lands in artifacts/rtl/verify.json; docs/results_rtl.md is generated
# from it by scripts/rtl/render_results.py.
set -euo pipefail
cd "$(dirname "$0")/../.."          # $readmemh paths are relative to the repo root
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python
STAGES=${*:-all}
exec "$PY" scripts/rtl/run_verify.py --stages compile $STAGES --shards "${SHARDS:-2}"
