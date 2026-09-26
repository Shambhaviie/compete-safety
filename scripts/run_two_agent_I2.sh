#!/bin/bash
set -o pipefail
SEEDS=(42 123 777)
CONDITION="competition_with_memory"
LOG_DIR="logs/two_agent_I2"
mkdir -p "${LOG_DIR}"

for interaction in I2_I3_FULL I2_I3_COMPLIANCE_ONLY; do
    CONFIG="config_two_agent_${interaction}.yaml"
    for seed in "${SEEDS[@]}"; do
        log_file="${LOG_DIR}/${interaction}_seed${seed}.log"
        echo "=========================================="
        echo "Starting: ${interaction} seed=${seed}"
        echo "Started at: $(date)"
        echo "=========================================="
        python3 src/run_two_agent_experiment.py "${CONDITION}" "${seed}" "${CONFIG}" \
            > "${log_file}" 2>&1
        if [ $? -eq 0 ]; then
            echo "COMPLETED: ${interaction} seed${seed} at $(date)"
        else
            echo "FAILED: ${interaction} seed${seed} at $(date) -- check ${log_file}"
        fi
    done
done
echo "ALL TWO-AGENT I2 RUNS COMPLETE at $(date)"
