#!/usr/bin/env bash
# Gated pilot runner. Reads config from $CONFIG (default: config.yaml in $PWD).
#
# Resumability: every pilot writes an append-only JSONL checkpoint plus atomic
# .npy files. If killed mid-run (OOM, SIGKILL, nohup hangup), rerunning this
# script picks up exactly where it left off — no duplicate forward passes.
#
# Usage:
#   CONFIG=config.yaml bash scripts/run_gated.sh
#   CONFIG=configs/llama8b.yaml CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_gated.sh
#
# Environment:
#   CONFIG               path to a YAML config (default: config.yaml)
#   CUDA_VISIBLE_DEVICES which physical GPUs are visible to the process
#   PYTHONUNBUFFERED     set to 1 for line-buffered stdout under nohup
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CONFIG:=config.yaml}"
export CONFIG
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "CONFIG=$CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "PWD=$(pwd)"
date
echo "============================================================"

run() {
    echo ">>> Running $1"
    python -m "$1"
}

check_pass() {
    python -c "
import json, sys
with open('$1') as f:
    r = json.load(f)
sys.exit(0 if r.get('decision_rule_passed') else 1)
"
}

# Derive the out_dir prefix from the config so parallel launches don't collide.
OUT_BASE=$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    print(yaml.safe_load(f)['logging']['out_dir'])
")

run src.pilot_b1_attractor
if check_pass "$OUT_BASE/pilot_b1_attractor/results.json"; then
    echo ">>> B1 PASSED. Proceeding to B2."
    run src.pilot_b2_steering
    if check_pass "$OUT_BASE/pilot_b2_steering/results.json"; then
        echo ">>> B2 PASSED. Proceeding to B3."
        run src.pilot_b3_gradient
    else
        echo ">>> B2 FAILED. Skipping B3."
    fi
else
    echo ">>> B1 FAILED. Skipping B2 and B3."
fi

echo "============================================================"
echo ">>> Running Pilot 5 (independent of B track)"
echo "============================================================"
run src.pilot_5_self_other

echo "============================================================"
date
echo "All pilots done. Inspect $OUT_BASE/*/results.json"
echo "============================================================"
