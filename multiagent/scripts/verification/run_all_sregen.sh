#!/bin/bash
# Run baseline verification for all sregen-compatible models sequentially
# Skips LS, GS, QS (too large for 24GB RTX 4090)

set -e

SCRIPT="multiagent/scripts/verification/verify_model_sregen.py"
MODELS="GJ LR DS PH TU QJ"

echo "============================================================"
echo "CSS v3 Baseline Verification — sregen-4090"
echo "Models: $MODELS"
echo "Started: $(date)"
echo "============================================================"

for agent in $MODELS; do
    echo ""
    echo "============================================================"
    echo "Starting $agent at $(date)"
    echo "============================================================"
    python3 $SCRIPT --agent $agent
    echo "$agent completed at $(date)"
done

echo ""
echo "============================================================"
echo "ALL MODELS COMPLETE: $(date)"
echo "============================================================"
