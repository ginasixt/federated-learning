import argparse, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# run with: 
# python3 federated_learning/compare_thresholds_report.py \
#   --csv out/agg_runs.csv \
#   --outdir out/compare_thresholds

def safe_savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()

def add_ci(series):
    x = series.dropna().values.astype(float)
    n = len(x)
    mean = np.mean(x) if n else np.nan
    sd = np.std(x, ddof=1) if n > 1 else np.nan
    ci = 1.96 * (sd / np.sqrt(n)) if n > 1 else np.nan
    return mean, sd, ci, n

def mean_ci_plot(df, metric, outdir):
    order = sorted(df["threshold"].dropna().unique())
    means, cis = [], []
    for t in order:
        m, sd, ci, n = add_ci(df[df["threshold"] == t][metric])
        means.append(m); cis.append(ci)
    x = np.arange(len(order))
    plt.figure()
    plt.errorbar(x, means, yerr=cis, fmt='o-')
    plt.xticks(x, [str(t) for t in order])
    plt.xlabel("Threshold"); plt.ylabel(metric.replace("_"," ").title())
    plt.title(f"{metric.replace('_',' ').title()} mean ± 95% CI by Threshold")
    safe_savefig(os.path.join(outdir, f"mean_ci_{metric}.png"))

def boxplot_metric(df, metric, outdir):
    order = sorted(df["threshold"].dropna().unique())
    data = [df[df["threshold"] == t][metric].values for t in order]
    plt.figure()
    plt.boxplot(data, labels=[str(t) for t in order], showfliers=False)
    plt.xlabel("Threshold"); plt.ylabel(metric.replace("_"," ").title())
    plt.title(f"{metric.replace('_',' ').title()} by Threshold (per-run best round)")
    safe_savefig(os.path.join(outdir, f"box_{metric}.png"))

def scatter_tradeoff(df, outdir):
    order = sorted(df["threshold"].dropna().unique())
    plt.figure()
    for t in order:
        sub = df[df["threshold"] == t]
        plt.scatter(sub["spec"], sub["recall"], alpha=0.8, s=40, label=f"thr={t}")
    plt.axvline(0.80, linestyle="--", color="gray")
    plt.axhline(0.85, linestyle="--", color="gray")
    plt.xlabel("Specificity"); plt.ylabel("Recall")
    plt.title("Recall vs Specificity (points = runs; color by threshold)")
    plt.legend()
    safe_savefig(os.path.join(outdir, "scatter_recall_vs_spec.png"))

def summary_table(df, outdir):
    rows = []
    for thr, g in df.groupby("threshold"):
        for m in ["recall","spec","f1","ppv","npv","alerts_per_1000","youden","auc"]:
            mean, sd, ci, n = add_ci(g[m])
            rows.append({
                "threshold": thr, "metric": m,
                "mean": mean, "sd": sd, "ci95": ci, "n_runs": n
            })
    tab = pd.DataFrame(rows).sort_values(["metric","threshold"])
    path = os.path.join(outdir, "summary_table.csv")
    tab.to_csv(path, index=False)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV from parse_flwr_logs_fixed.py")
    ap.add_argument("--outdir", required=True, help="Output dir for plots/tables")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    for metric in ["recall","spec","f1","ppv","npv","alerts_per_1000","youden"]:
        boxplot_metric(df, metric, args.outdir)
        mean_ci_plot(df, metric, args.outdir)
    scatter_tradeoff(df, args.outdir)
    path = summary_table(df, args.outdir)
    print(f"Wrote {path} and plots in {args.outdir}")

if __name__ == "__main__":
    main()
