#!/usr/bin/env bash
set -e
MU_LIST=("0" "1e-4" "1e-3" "5e-3" "1e-2")
TARGET_RECALL="0.95"
ROUNDS="20"
for MU in "${MU_LIST[@]}"; do
  OUTDIR="results/mu=${MU}"
  mkdir -p "${OUTDIR}"
  RUN_CFG=$(cat <<JSON
{"mu": ${MU},
 "eval-threshold-mode": "hist",
 "return-histogram": true,
 "target-recall": ${TARGET_RECALL},
 "threshold-grid-size": 101,
 "privacy.mode": "dp",
 "privacy.k_min": 50,
 "privacy.dp.mechanism": "laplace",
 "privacy.dp.epsilon": 1.0,
 "privacy.dp.delta": 1e-5,
 "privacy.clip.max_c": 1,
 "num-server-rounds": ${ROUNDS}}
JSON
)
  echo "${RUN_CFG}" > "${OUTDIR}/run_config.json"
  echo ">>> flwr run --run-config '${RUN_CFG}'"
  flwr run --run-config "${RUN_CFG}" | tee "${OUTDIR}/stdout.log"
done
