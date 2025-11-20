#!/usr/bin/env bash
set -euo pipefail

# run with: bash federated_learning/plotting/run_a01_thr035.sh

# Fixe Parameter
THRESHOLDS=(0.323)   
ROUNDS=5
ALPHA="alpha03"  # falls du verschiedene Alphas testest, ändere hier

# Optional: Logs-Verzeichnis erstellen
mkdir -p logs

for THR in "${THRESHOLDS[@]}"; do
  # Threshold für Verzeichnisnamen formatieren (0.35 -> 035)
  THR_TAG=$(echo "$THR" | sed 's/\.//g')
  
  echo ""
  echo "=========================================="
  echo "Starting runs for threshold=$THR"
  echo "=========================================="
  
  # Log-Verzeichnis für diesen Threshold erstellen
  mkdir -p "logs/${ALPHA}/thr_${THR_TAG}"
  
  for run in $(seq 1 $ROUNDS); do
    echo "▶️  Run $run/$ROUNDS | thr=$THR | run-tag=${run}"

    flwr run --run-config "eval-threshold=${THR} run-tag=${run}" > /dev/null 2>&1
    
    echo "   ✓ Completed run $run for threshold $THR"
  done
  
  echo "✅ Finished all runs for threshold=$THR"
  echo "   Logs: logs/${ALPHA}/thr_${THR_TAG}/run_*.log"
  echo "   Results: result/alpha03/thr_${THR}/run_*.json"
done

echo ""
echo "=========================================="
echo "✅ All runs completed!"
echo "=========================================="
echo "Logs organized by threshold:"
for THR in "${THRESHOLDS[@]}"; do
  THR_TAG=$(echo "$THR" | sed 's/\.//g')
  echo "  - logs/${ALPHA}/thr_${THR_TAG}/"
done
echo ""
echo "Results organized by threshold:"
for THR in "${THRESHOLDS[@]}"; do
  echo "  - result/idd/thr_${THR}/"
done