#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

MU_VALUES = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
TARGET_RECALL = 0.95
ROUNDS = 20

BASE_RUN_CONFIG = {
    "eval-threshold-mode": "hist",
    "return-histogram": True,
    "target-recall": TARGET_RECALL,
    "threshold-grid-size": 101,
    "privacy.mode": "dp",
    "privacy.k_min": 50,
    "privacy.dp.mechanism": "laplace",
    "privacy.dp.epsilon": 1.0,
    "privacy.dp.delta": 1e-5,
    "privacy.clip.max_c": 1,
    "num-server-rounds": ROUNDS,
}

def run_once(mu: float):
    outdir = Path("results") / f"mu={mu}"
    outdir.mkdir(parents=True, exist_ok=True)
    run_cfg = dict(BASE_RUN_CONFIG)
    run_cfg["mu"] = float(mu)
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    cmd = ["flwr", "run", "--run-config", json.dumps(run_cfg)]
    print(">>>", " ".join(cmd))
    with open(outdir / "stdout.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)

def main():
    for mu in MU_VALUES:
        run_once(mu)

if __name__ == "__main__":
    main()
