#!/bin/bash

set -euo pipefail

export PYTHONPATH=.
export MUJOCO_GL=disable

# Configurable parameters
NUM_GPUS=4
SLOTS_PER_GPU=12
MAX_PARALLEL=$((NUM_GPUS * SLOTS_PER_GPU))
OUT_DIR="experiments/mujoco_no_resets/outputs/"
CONFIG="experiments/mujoco_no_resets/inputs.json"

# Setup
mkdir -p "$OUT_DIR"

# Create a unique token queue file
TOKEN_FILE="/tmp/gpu_tokens_$$.txt"
LOCK_FILE="/tmp/gpu_tokens_lock_$$.lock"

# Initialize token file with GPU slots
for ((g=0; g<NUM_GPUS; g++)); do
    for ((s=0; s<SLOTS_PER_GPU; s++)); do
        echo "$g"
    done
done > "$TOKEN_FILE"

# Cleanup on exit
cleanup() {
    rm -f "$TOKEN_FILE" "$LOCK_FILE"
}
trap cleanup EXIT

# Function to run a single job
run_job() {
    local i=$1
    local gpu_id=""
    
    # Try to acquire GPU token
    while true; do
        {
            flock 200
            gpu_id=$(head -n 1 "$TOKEN_FILE" 2>/dev/null)
            if [[ -n "$gpu_id" ]]; then
                tail -n +2 "$TOKEN_FILE" > "$TOKEN_FILE.tmp" && mv "$TOKEN_FILE.tmp" "$TOKEN_FILE"
                break
            fi
        } 200>"$LOCK_FILE"
        sleep 1  # Wait before retrying
    done
    
    echo "[INFO] Starting job $i on GPU $gpu_id" >&2
    
    # Run the job
    python3 run.py --config-file "$CONFIG" \
                   --out-dir "$OUT_DIR" \
                   --base-id "$i" \
                   --gpu-id "$gpu_id" > "$OUT_DIR/$i.log" 2>&1
    
    # Return GPU token
    {
        flock 200
        echo "$gpu_id" >> "$TOKEN_FILE"
    } 200>"$LOCK_FILE"
    
    echo "[INFO] Completed job $i on GPU $gpu_id" >&2
}

# Export necessary variables and functions
export -f run_job
export TOKEN_FILE LOCK_FILE CONFIG OUT_DIR

# Create a list of all jobs to run
jobs=()
for run_id in {0..4}; do
    for i in {0..39}; do
        jobs+=("$((run_id * 40 + i))")
    done
done

# Run jobs in parallel with proper queue management
for job_id in "${jobs[@]}"; do
    run_job "$job_id" &
    # Small delay to prevent overwhelming the system
    sleep 0.1
done

# Wait for all jobs to complete
wait
echo "All jobs completed!"