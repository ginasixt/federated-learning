import argparse, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# run with_:
# python3 federated_learning/more_screening_plots.py --csv out/agg_runs.csv --outdir out/more_plots

def safe_savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()

def bubbles(x, y, size, hue, xlabel, ylabel, title, outpath):
    plt.figure()
    # group by threshold to get a legend
    for t in sorted(pd.Series(hue).dropna().unique()):
        m = (hue == t)
        plt.scatter(x[m], y[m], s=np.clip(size[m], 10, None), alpha=0.8, label=f"thr={t}")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend()
    safe_savefig(outpath)

def pareto_frontier(x, y):
    # lower x is better (Alerts), higher y is better (Youden) -> sort by x asc
    pts = np.array(sorted(zip(x, y), key=lambda k: (np.isnan(k[0]), k[0])))
    front = []
    best_y = -np.inf
    for xi, yi in pts:
        if np.isnan(xi) or np.isnan(yi): 
            continue
        if yi > best_y:
            front.append((xi, yi))
            best_y = yi
    return zip(*front) if front else ([], [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ["threshold","recall","spec","ppv","npv","alerts_per_1000","youden","f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) PPV vs Recall (size = Alerts/1000, color = threshold)
    if {"ppv","recall","alerts_per_1000","threshold"}.issubset(df.columns):
        bubbles(df["ppv"], df["recall"], df["alerts_per_1000"], df["threshold"],
                "PPV (Precision)", "Recall (Sensitivity)",
                "PPV vs Recall (bubble = Alerts/1000, color = threshold)",
                os.path.join(args.outdir, "bubble_ppv_vs_recall.png"))

    # 2) NPV vs Recall (size = Alerts/1000, color = threshold)
    if {"npv","recall","alerts_per_1000","threshold"}.issubset(df.columns):
        bubbles(df["npv"], df["recall"], df["alerts_per_1000"], df["threshold"],
                "NPV", "Recall (Sensitivity)",
                "NPV vs Recall (bubble = Alerts/1000, color = threshold)",
                os.path.join(args.outdir, "bubble_npv_vs_recall.png"))

    # 3) Youden vs Alerts/1000 + Pareto frontier
    if {"youden","alerts_per_1000","threshold"}.issubset(df.columns):
        plt.figure()
        for t in sorted(df["threshold"].dropna().unique()):
            sub = df[df["threshold"]==t]
            plt.scatter(sub["alerts_per_1000"], sub["youden"], alpha=0.8, label=f"thr={t}")
        xpf, ypf = pareto_frontier(df["alerts_per_1000"].values, df["youden"].values)
        if len(list(xpf))>0:
            xpf, ypf = pareto_frontier(df["alerts_per_1000"].values, df["youden"].values)  # recompute iterators
            plt.plot(list(xpf), list(ypf), linewidth=2)
        plt.xlabel("Alerts per 1000 (workload)"); plt.ylabel("Youden’s J")
        plt.title("Youden vs Alerts/1000 (Pareto frontier line)")
        plt.legend()
        safe_savefig(os.path.join(args.outdir, "youden_vs_alerts_pareto.png"))

    # 4) F1 vs Alerts/1000 (by threshold)
    if {"f1","alerts_per_1000","threshold"}.issubset(df.columns):
        plt.figure()
        for t in sorted(df["threshold"].dropna().unique()):
            sub = df[df["threshold"]==t]
            plt.scatter(sub["alerts_per_1000"], sub["f1"], alpha=0.8, label=f"thr={t}")
        plt.xlabel("Alerts per 1000"); plt.ylabel("F1")
        plt.title("F1 vs Alerts/1000 (color = threshold)")
        plt.legend()
        safe_savefig(os.path.join(args.outdir, "f1_vs_alerts.png"))

    # 5) Workload-Efficiency: Recall per 1000 alerts
    if {"recall","alerts_per_1000","threshold"}.issubset(df.columns):
        eff = df.copy()
        eff["recall_per_1000_alerts"] = eff["recall"] / eff["alerts_per_1000"]
        plt.figure()
        for t in sorted(eff["threshold"].dropna().unique()):
            sub = eff[eff["threshold"]==t]
            plt.scatter(sub["alerts_per_1000"], sub["recall_per_1000_alerts"], alpha=0.8, label=f"thr={t}")
        plt.xlabel("Alerts per 1000"); plt.ylabel("Recall / Alerts per 1000")
        plt.title("Workload-Efficiency (higher = more TP per alert budget)")
        plt.legend()
        safe_savefig(os.path.join(args.outdir, "efficiency_recall_per_alert.png"))

if __name__ == "__main__":
    main()
