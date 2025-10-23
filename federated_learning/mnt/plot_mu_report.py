from pathlib import Path
import csv, math
from typing import List, Dict
import matplotlib.pyplot as plt

def load_rows() -> List[Dict]:
    rows = []
    for metrics in Path("results").glob("mu=*/metrics.csv"):
        with open(metrics, newline="") as f:
            for r in csv.DictReader(f):
                rr = {}
                for k, v in r.items():
                    if k == "mu":
                        rr[k] = float(v)
                    else:
                        try:
                            rr[k] = float(v)
                        except:
                            rr[k] = math.nan
                rows.append(rr)
    rows.sort(key=lambda x: x["mu"])
    return rows

def ensure_data(rows: List[Dict]):
    if not rows:
        raise SystemExit("No results found. Expected files under results/mu=*/metrics.csv")

def plot_xy(rows: List[Dict], x_key: str, y_key: str, fname: str, ylabel: str = None):
    xs = [r[x_key] for r in rows]
    ys = [r[y_key] for r in rows]
    import matplotlib
    matplotlib.use("Agg")
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_key)
    plt.ylabel(ylabel or y_key)
    plt.title(f"{y_key} vs {x_key}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()

def main():
    rows = load_rows()
    ensure_data(rows)
    plot_xy(rows, "mu", "recall", "mu_vs_recall.png", "Recall (Sensitivity)")
    plot_xy(rows, "mu", "specificity", "mu_vs_specificity.png", "Spezifität")
    plot_xy(rows, "mu", "auc", "mu_vs_auc.png", "ROC-AUC")
    plot_xy(rows, "mu", "alerts_per_1000", "mu_vs_alerts_per_1000.png", "Alerts pro 1000")
    plot_xy(rows, "mu", "threshold", "mu_vs_threshold.png", "gewählter Threshold")
    if any(not math.isnan(r.get("brier", float("nan"))) for r in rows):
        plot_xy(rows, "mu", "brier", "mu_vs_brier.png", "Brier Score")
    if any(not math.isnan(r.get("ece", float("nan"))) for r in rows):
        plot_xy(rows, "mu", "ece", "mu_vs_ece.png", "ECE")
    print("μ-sweep summary:")
    for r in rows:
        print(
            f"mu={r['mu']:.3g} | thr={r.get('threshold', float('nan')):.3f} "
            f"| R={r.get('recall', float('nan')):.3f} "
            f"| S={r.get('specificity', float('nan')):.3f} "
            f"| AUC={r.get('auc', float('nan')):.3f} "
            f"| alerts/1k={r.get('alerts_per_1000', float('nan')):.1f}"
        )

if __name__ == "__main__":
    main()
