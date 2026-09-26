#!/bin/bash
set -o pipefail
CONDITIONS=("control" "competition_no_memory" "competition_with_memory")
SEEDS=(42 123 777)
LOG_DIR="logs/task6_regression"
CONFIG="config_task6_regression.yaml"

for condition in "${CONDITIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    log_file="${LOG_DIR}/${condition}_seed${seed}.log"
    echo "=========================================="
    echo "Starting: condition=${condition} seed=${seed}"
    echo "Started at: $(date)"
    echo "=========================================="
    python3 src/run_real_experiment.py "${condition}" "${seed}" "${CONFIG}" > "${log_file}" 2>&1
    if [ $? -eq 0 ]; then
      echo "COMPLETED: ${condition} seed${seed} at $(date)"
    else
      echo "FAILED: ${condition} seed${seed} at $(date) -- check ${log_file}"
    fi
  done
done
echo "ALL TASK 6 RUNS COMPLETE at $(date)"
