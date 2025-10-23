threshold=0.42
mkdir -p logs/thr_${threshold}

for run in {1..10}; do
  echo "▶️  Run $run for threshold=${threshold}"
  flwr run > logs/thr_${threshold}/run_${run}.log 2>&1
done