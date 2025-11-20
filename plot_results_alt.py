
import argparse
import json
import math
import re
from pathlib import Path
from typing import Tuple

# How to run
# python plot_results.py --root /Users/ginasixt/federated-learning/result --out /Users/ginasixt/federated-learning/plots_out


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA_DIR_RE = re.compile(r"alpha(\d+)", re.IGNORECASE)
THR_DIR_RE = re.compile(r"thr_([0-9]*\.?[0-9]+)")

def parse_alpha_from_dir(name: str) -> float:
    m = ALPHA_DIR_RE.search(name)
    if not m:
        try:
            return float(name)
        except Exception:
            return np.nan
    token = m.group(1)
    if "." in token:
        try:
            return float(token)
        except Exception:
            pass
    if token == "10":
        return 1.0
    if len(token) >= 2:
        return float(token[:-1] + "." + token[-1])
    return float(token)

def parse_thr_from_dir(name: str) -> float:
    m = THR_DIR_RE.search(name)
    if m:
        return float(m.group(1))
    try:
        return float(name)
    except Exception:
        return np.nan

def collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    for alpha_dir in sorted(root.glob("alpha*")):
        if not alpha_dir.is_dir():
            continue
        alpha_val = parse_alpha_from_dir(alpha_dir.name)
        for thr_dir in sorted(alpha_dir.glob("thr_*")):
            if not thr_dir.is_dir():
                continue
            thr_val = parse_thr_from_dir(thr_dir.name)
            for run_file in sorted(thr_dir.glob("run_*.json")):
                try:
                    with open(run_file, "r") as f:
                        data = json.load(f)
                    metrics = data.get("metrics", {})
                    rows.append({
                        "alpha": alpha_val,
                        "threshold": thr_val,
                        "run_path": str(run_file),
                        "round": data.get("round", np.nan),
                        "recall": metrics.get("recall", np.nan),
                        "spec": metrics.get("spec", np.nan),
                        "tpr": metrics.get("tpr", np.nan),
                        "fpr": metrics.get("fpr", np.nan),
                        "precision": metrics.get("precision", np.nan),
                        "ppv": metrics.get("ppv", np.nan),
                        "npv": metrics.get("npv", np.nan),
                        "f1": metrics.get("f1", np.nan),
                        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                        "youden": metrics.get("youden", np.nan),
                        "alerts_per_1000": metrics.get("alerts_per_1000", np.nan),
                        "auc": metrics.get("auc", np.nan),
                        "tp": metrics.get("tp", np.nan),
                        "fp": metrics.get("fp", np.nan),
                        "tn": metrics.get("tn", np.nan),
                        "fn": metrics.get("fn", np.nan),
                        "prevalence": metrics.get("prevalence", np.nan),
                    })
                except Exception as e:
                    print(f"[WARN] Failed to read {run_file}: {e}")
    df = pd.DataFrame(rows)
    if "ppv" in df.columns and "precision" in df.columns:
        df["ppv"] = df["ppv"].fillna(df["precision"])
    return df

def mean_std_ci(x: pd.Series, ci: float = 0.95):
    x = x.dropna()
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mu = x.mean()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if len(x) > 1:
        se = sd / math.sqrt(len(x))
        z = 1.96 if ci == 0.95 else 1.96
        lo, hi = mu - z*se, mu + z*se
    else:
        lo = hi = mu
    return (mu, sd, hi - mu)

def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def plot_scatter_recall_vs_spec(df: pd.DataFrame, outdir: Path):
    ensure_out(outdir)
    plt.figure()
    alphas = sorted(df["alpha"].dropna().unique().tolist())
    thrs = sorted(df["threshold"].dropna().unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    marker_map = {thr: markers[i % len(markers)] for i, thr in enumerate(thrs)}
    for a in alphas:
        sub = df[df["alpha"] == a]
        for thr in thrs:
            sub2 = sub[sub["threshold"] == thr]
            if sub2.empty:
                continue
            plt.scatter(sub2["spec"], sub2["recall"], marker=marker_map[thr], label=f"α={a}, thr={thr}", alpha=0.8)
    plt.xlabel("Specificity")
    plt.ylabel("Recall")
    plt.title("Recall vs Specificity (per run)")
    plt.grid(True, linestyle="--", alpha=0.4)
    from matplotlib.lines import Line2D
    thr_handles = [Line2D([0],[0], marker=marker_map[t], linestyle='', label=f"thr={t}") for t in thrs]
    alpha_handles = [Line2D([0],[0], marker='o', linestyle='', label=f"α={a}") for a in alphas]
    first_legend = plt.legend(handles=alpha_handles, title="Alpha", loc="lower left")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=thr_handles, title="Threshold", loc="lower right")
    outpath = outdir / "scatter_recall_vs_spec.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath

def plot_means_recall_spec_errorbars(df: pd.DataFrame, outdir: Path):
    ensure_out(outdir)
    grp = df.groupby(["alpha", "threshold"])
    rows = []
    for (a, t), g in grp:
        mu_r, sd_r, err_r = mean_std_ci(g["recall"])
        mu_s, sd_s, err_s = mean_std_ci(g["spec"])
        rows.append({"alpha": a, "threshold": t, "recall_mu": mu_r, "recall_err": err_r, "spec_mu": mu_s, "spec_err": err_s, "n": len(g)})
    agg = pd.DataFrame(rows).sort_values(["alpha", "threshold"])
    plt.figure()
    for a in sorted(agg["alpha"].unique().tolist()):
        sub = agg[agg["alpha"] == a]
        plt.errorbar(sub["spec_mu"], sub["recall_mu"], xerr=sub["spec_err"], yerr=sub["recall_err"], fmt="o", capsize=3, label=f"α={a}")
    plt.xlabel("Specificity (mean ± ~95% CI)")
    plt.ylabel("Recall (mean ± ~95% CI)")
    plt.title("Recall vs Specificity (means per α, threshold)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Alpha")
    outpath = outdir / "means_recall_spec_errorbars.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath

def plot_balacc_bars(df: pd.DataFrame, outdir: Path):
    ensure_out(outdir)
    grp = df.groupby(["alpha", "threshold"])
    rows = []
    for (a, t), g in grp:
        mu, sd, err = mean_std_ci(g["balanced_accuracy"])
        rows.append({"alpha": a, "threshold": t, "balacc_mu": mu, "balacc_err": err, "n": len(g)})
    agg = pd.DataFrame(rows).sort_values(["alpha", "threshold"])
    alphas = sorted(agg["alpha"].unique().tolist())
    thrs = sorted(agg["threshold"].unique().tolist())
    x_positions = []
    heights = []
    errors = []
    for a in alphas:
        for t in thrs:
            row = agg[(agg["alpha"] == a) & (agg["threshold"] == t)]
            if row.empty:
                continue
            x_positions.append(f"α={a}\nthr={t}")
            heights.append(row["balacc_mu"].values[0])
            errors.append(row["balacc_err"].values[0])
    plt.figure()
    x = np.arange(len(x_positions))
    plt.bar(x, heights, yerr=errors, capsize=3)
    plt.xticks(x, x_positions, rotation=0)
    plt.ylabel("Balanced Accuracy (mean ± ~95% CI)")
    plt.title("Balanced Accuracy by α and threshold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    outpath = outdir / "bar_balanced_accuracy.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath

def plot_auc_box_by_alpha(df: pd.DataFrame, outdir: Path):
    ensure_out(outdir)
    plt.figure()
    data = []
    labels = []
    for a in sorted(df["alpha"].dropna().unique().tolist()):
        vals = df[df["alpha"] == a]["auc"].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"α={a}")
    if not data:
        return None
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("AUC")
    plt.title("AUC distribution by α")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    outpath = outdir / "box_auc_by_alpha.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath

def export_aggregates(df: pd.DataFrame, outdir: Path):
    ensure_out(outdir)
    metrics = ["recall", "spec", "f1", "precision", "npv", "balanced_accuracy", "youden", "alerts_per_1000", "auc"]
    agg = df.groupby(["alpha", "threshold"])[metrics].agg(["mean", "std", "count"])
    outpath = outdir / "aggregates_by_alpha_threshold.csv"
    agg.to_csv(outpath)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Plot FL results from result/alpha*/thr_*/run_*.json")
    ap.add_argument("--root", type=Path, default=Path("result"), help="Root directory containing alpha*/thr_*/run_*.json")
    ap.add_argument("--out", type=Path, default=Path("plots_out"), help="Output directory for figures and CSV")
    args = ap.parse_args()

    df = collect_runs(args.root)
    if df.empty:
        print(f"[ERROR] No run_*.json found under {args.root}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "all_runs_flat.csv", index=False)

    paths = []
    p1 = plot_scatter_recall_vs_spec(df, args.out)
    if p1: paths.append(p1)
    p2 = plot_means_recall_spec_errorbars(df, args.out)
    if p2: paths.append(p2)
    p3 = plot_balacc_bars(df, args.out)
    if p3: paths.append(p3)
    p4 = plot_auc_box_by_alpha(df, args.out)
    if p4: paths.append(p4)
    p5 = export_aggregates(df, args.out)
    if p5: paths.append(p5)

    print("[OK] Created:")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()
