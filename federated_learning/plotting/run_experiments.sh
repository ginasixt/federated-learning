# how to run:
# bash federated_learning/plotting/run_experiments.sh

threshold=0.35
mkdir -p logs/alpha03/thr_${threshold}

for run in {1..10}; do
  echo "▶️  Run $run for threshold=${threshold}"
  flwr run > logs/alpha03/thr_${threshold}/run_${run}.log 2>&1
done