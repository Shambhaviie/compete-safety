#!/bin/bash
set -o pipefail
SEEDS=(42 123 777)
ESCALATION_TYPES=("E1" "E2" "E3")
LOG_DIR="logs/escalation"
CONDITION="competition_with_memory"

for seed in "${SEEDS[@]}"; do
  for etype in "${ESCALATION_TYPES[@]}"; do
    log_file="${LOG_DIR}/${etype}_seed${seed}.log"
    echo "=========================================="
    echo "Starting: escalation=${etype} seed=${seed}"
    echo "Started at: $(date)"
    echo "=========================================="
    python3 src/run_real_experiment.py "${CONDITION}" "${seed}" "config_escalation_${etype}.yaml" > "${log_file}" 2>&1
    if [ $? -eq 0 ]; then
      echo "COMPLETED: ${etype} seed${seed} at $(date)"
    else
      echo "FAILED: ${etype} seed${seed} at $(date) -- check ${log_file}"
    fi
  done
done
echo "ALL ESCALATION RUNS COMPLETE at $(date)"
