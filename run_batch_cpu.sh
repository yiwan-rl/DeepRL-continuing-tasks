#!/bin/bash

set -euo pipefail

export PYTHONPATH=.
export MUJOCO_GL=disable

# Configurable parameters
NUM_CPUS=$(nproc)
OUT_DIR="experiments/mujoco_no_resets/outputs/"
CONFIG="experiments/mujoco_no_resets/inputs.json"

# Setup
mkdir -p "$OUT_DIR"

# Create a list of all jobs to run
for i in {0..199}; do
    echo "$i"
done | parallel --jobs "$NUM_CPUS" \
    --joblog "$OUT_DIR/joblog.txt" \
    --tagstring "[Job {#}]" \
    --line-buffer \
    "echo 'Starting job {}' >&2 && 
     python3 run.py --config-file $CONFIG \
                    --out-dir $OUT_DIR \
                    --base-id {} > $OUT_DIR/{}.log 2>&1 && 
     echo 'Completed job {}' >&2"

echo "All jobs completed!" 