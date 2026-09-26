#!/bin/bash
set -o pipefail
SEEDS=(42 123 777)
CONDITION="competition_with_memory"
CONFIG="config_two_agent_I0.yaml"
LOG_DIR="logs/two_agent_I0"

mkdir -p "${LOG_DIR}"

for seed in "${SEEDS[@]}"; do
    log_file="${LOG_DIR}/seed${seed}.log"
    echo "=========================================="
    echo "Starting: I0 seed=${seed}"
    echo "Started at: $(date)"
    echo "=========================================="
    python3 src/run_two_agent_experiment.py "${CONDITION}" "${seed}" "${CONFIG}" \
        > "${log_file}" 2>&1
    if [ $? -eq 0 ]; then
        echo "COMPLETED: I0 seed${seed} at $(date)"
    else
        echo "FAILED: I0 seed${seed} at $(date) -- check ${log_file}"
    fi
done
echo "ALL TWO-AGENT I0 RUNS COMPLETE at $(date)"
