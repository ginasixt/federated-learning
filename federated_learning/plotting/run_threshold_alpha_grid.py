import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import re

import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
PYPROJECT = ROOT / "pyproject.toml"
PY_BAK = ROOT / "pyproject.toml.bak"
RESULT_JSON = ROOT / "results" / "screening_best_round.json"
OUT_PLOT = ROOT / "results" / "recall_vs_spec.png"

def replace_run_config(split_path: str, eval_threshold: float):
    txt = PYPROJECT.read_text(encoding="utf8")
    # Simple replace for split-path = "..." under tool.flwr.app.config
    txt = re.sub(r'^(split-path\s*=\s*").*(")', rf'\1{split_path}\2', txt, flags=re.M)
    txt = re.sub(r'^(eval-threshold\s*=\s*).+$', rf'\1{eval_threshold}', txt, flags=re.M)
    PYPROJECT.write_text(txt, encoding="utf8")

def restore_pyproject():
    if PY_BAK.exists():
        shutil.move(str(PY_BAK), str(PYPROJECT))

def run_flwr_and_wait(timeout=None, logfile=None):
    cmd = ["flwr", "run", "."]
    with (open(logfile, "w") if logfile else open("/dev/null", "w")) as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
            raise

def collect_metrics():
    if not RESULT_JSON.exists():
        return None
    d = json.loads(RESULT_JSON.read_text(encoding="utf8"))
    # expecting keys like 'tpr' and 'spec' from server aggregation
    tpr = float(d.get("tpr", 0.0))
    spec = float(d.get("spec", 0.0))
    return {"tpr": tpr, "spec": spec, "raw": d}

def find_splits_for_alpha(alpha):
    # match filenames like splits_dirichlet_*_a0.1.json or a01/a03/a10 variants
    res = []
    for p in ROOT.glob("splits_dirichlet*"):
        if f"_a{str(alpha).replace('.','')}" in p.name or f"_a{alpha}" in p.name or f"_a{alpha:.2g}" in p.name:
            res.append(p)
        # also accept a01, a03, a10 style
        if re.search(rf"_a0*{int(alpha*10)}", p.name):
            res.append(p)
    # fallback: include any file that contains _a{alpha} literally
    res = list(dict.fromkeys(res))
    return res

def main(args):
    alphas = [float(x) for x in args.alphas.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    if not PYPROJECT.exists():
        print("pyproject.toml not found", file=sys.stderr); sys.exit(1)

    # backup
    shutil.copy2(PYPROJECT, PY_BAK)

    all_results = []
    try:
        for a in alphas:
            splits = find_splits_for_alpha(a)
            if not splits:
                print(f"No split file found for alpha={a}, skipping.")
                continue
            for split in splits:
                for thr in thresholds:
                    print(f"RUN: split={split.name} alpha={a} thresh={thr}")
                    replace_run_config(str(split), thr)
                    if args.dry_run:
                        print("dry-run: updated pyproject.toml (not running).")
                        # collect nothing in dry-run
                        all_results.append({"split": split.name, "alpha": a, "threshold": thr, "tpr": None, "spec": None})
                        # restore backup after dry-run update to keep repo clean
                        shutil.copy2(PY_BAK, PYPROJECT)
                        continue

                    logfile = ROOT / "logs" / f"run_{split.stem}_thr{str(thr).replace('.','_')}.log"
                    logfile.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        run_flwr_and_wait(timeout=args.timeout, logfile=str(logfile))
                        time.sleep(1)
                        metrics = collect_metrics()
                        if metrics is None:
                            print("No results/screening_best_round.json produced for this run.", file=sys.stderr)
                            all_results.append({"split": split.name, "alpha": a, "threshold": thr, "tpr": None, "spec": None})
                        else:
                            all_results.append({"split": split.name, "alpha": a, "threshold": thr, "tpr": metrics["tpr"], "spec": metrics["spec"]})
                            print("-> tpr:", metrics["tpr"], "spec:", metrics["spec"])
                    except Exception as e:
                        print("Run failed:", e, file=sys.stderr)
                        all_results.append({"split": split.name, "alpha": a, "threshold": thr, "tpr": None, "spec": None})
                    finally:
                        # restore original pyproject between runs
                        shutil.copy2(PY_BAK, PYPROJECT)

    finally:
        restore_pyproject()

    # Save CSV of results
    out_csv = ROOT / "results" / "threshold_alpha_grid_results.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(json.dumps(all_results, indent=2), encoding="utf8")
    print("Wrote results:", out_csv)

    # Plot recall (tpr) vs spec
    xs = [r["spec"] for r in all_results if r["spec"] is not None]
    ys = [r["tpr"] for r in all_results if r["tpr"] is not None]
    labels = [f'{r["split"]}\na={r["alpha"]}\nthr={r["threshold"]}' for r in all_results if r["spec"] is not None]

    if xs and ys:
        plt.figure(figsize=(6,6))
        plt.scatter(xs, ys, c="C0")
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), fontsize=7, alpha=0.8)
        plt.xlabel("Specificity")
        plt.ylabel("Recall (TPR)")
        plt.title("Recall vs Specificity per (split, alpha, threshold)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_PLOT, dpi=200)
        print("Saved plot:", OUT_PLOT)
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", default="0.1,0.3,1.0")
    p.add_argument("--thresholds", default="0.35,0.42,0.5")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=None, help="timeout per flwr run in seconds")
    args = p.parse_args()
    main(args)