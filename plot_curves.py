
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss

def ece_score(y_true, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def decision_curve(y_true, p, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = y_true.astype(int)
    N = len(y_true)

    t_list, nb_m, nb_all, nb_none = [], [], [], []
    for pt in thresholds:
        yhat = (p >= pt).astype(int)
        TP = int(((y_true==1)&(yhat==1)).sum())
        FP = int(((y_true==0)&(yhat==1)).sum())
        nb = (TP/N) - (FP/N) * (pt / (1-pt))
        TP_all = int((y_true==1).sum())
        FP_all = int((y_true==0).sum())
        nb_all_pt = (TP_all/N) - (FP_all/N) * (pt / (1-pt))
        nb_none_pt = 0.0
        t_list.append(pt)
        nb_m.append(nb)
        nb_all.append(nb_all_pt)
        nb_none.append(nb_none_pt)
    return pd.DataFrame({"threshold": t_list, "nb_model": nb_m, "nb_all": nb_all, "nb_none": nb_none})

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_roc(y, p, outpath: Path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p) if len(np.unique(y))==2 else np.nan
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return auc

def plot_pr(y, p, outpath: Path, title="Precision-Recall Curve"):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p) if len(np.unique(y))==2 else np.nan
    plt.figure()
    plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return ap

def plot_calibration(y, p, outpath: Path, title="Calibration (Reliability)"):
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        mask = idx == b
        if not np.any(mask):
            continue
        xs.append(p[mask].mean())
        ys.append(y[mask].mean())
    ece = ece_score(y, p, n_bins=10)
    brier = brier_score_loss(y, p)
    plt.figure()
    plt.plot([0,1],[0,1],"--", label="Perfectly calibrated")
    plt.plot(xs, ys, "o-", label=f"Model (ECE={ece:.3f}, Brier={brier:.3f})")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return ece, brier

def plot_decision(df_nb: pd.DataFrame, outpath: Path, title="Decision Curve"):
    plt.figure()
    plt.plot(df_nb["threshold"], df_nb["nb_model"], label="Model")
    plt.plot(df_nb["threshold"], df_nb["nb_all"], "--", label="Treat-All")
    plt.plot(df_nb["threshold"], df_nb["nb_none"], "--", label="Treat-None")
    plt.xlabel("Risk threshold (pt)")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def load_round_scores(round_dir: Path) -> pd.DataFrame:
    files = list(round_dir.glob("client_*.parquet"))
    if not files:
        files = list(round_dir.glob("global_round.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["p","y_true"])
    df["p"] = df["p"].astype(float)
    df["y_true"] = df["y_true"].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="Plot ROC/PR/Calibration/Decision Curves from saved scores (p,y).")
    ap.add_argument("--root", type=Path, default=Path("result_scores"), help="Root folder containing run_*/split-*/round_*/client_*.parquet")
    ap.add_argument("--out", type=Path, default=Path("plots_curves"), help="Output directory")
    ap.add_argument("--rounds", type=str, default="", help="Comma-separated round numbers to plot (e.g., 1,5,10). Empty = all.")
    args = ap.parse_args()

    ensure_dir(args.out)
    rounds_filter = set(int(x) for x in args.rounds.split(",") if x.strip().isdigit()) if args.rounds else None

    summary_rows = []

    for run_dir in sorted(args.root.glob("run_*")):
        for split_dir in sorted(run_dir.glob("split-*")):
            for round_dir in sorted(split_dir.glob("round_*")):
                try:
                    rnum = int(round_dir.name.split("_")[-1])
                except Exception:
                    rnum = None
                if rounds_filter and rnum not in rounds_filter:
                    continue

                df = load_round_scores(round_dir)
                if df.empty:
                    continue

                y = df["y_true"].to_numpy()
                p = df["p"].to_numpy()

                outdir = args.out / run_dir.name / split_dir.name / round_dir.name
                ensure_dir(outdir)

                auc = plot_roc(y, p, outdir / "roc.png", title=f"ROC – {run_dir.name} | {split_dir.name} | round {rnum}")
                ap = plot_pr(y, p, outdir / "pr.png", title=f"PR – {run_dir.name} | {split_dir.name} | round {rnum}")
                ece, brier = plot_calibration(y, p, outdir / "calibration.png", title=f"Calibration – {run_dir.name} | {split_dir.name} | round {rnum}")
                nb_df = decision_curve(y, p)
                plot_decision(nb_df, outdir / "decision_curve.png", title=f"Decision – {run_dir.name} | {split_dir.name} | round {rnum}")
                nb_df.to_csv(outdir / "decision_curve.csv", index=False)

                summary_rows.append({
                    "run": run_dir.name,
                    "split": split_dir.name,
                    "round": rnum,
                    "auc_roc": auc,
                    "ap": ap,
                    "ece_10bins": ece,
                    "brier": brier,
                    "n": len(y),
                    "prevalence": float(y.mean()) if len(y)>0 else np.nan,
                })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.sort_values(["run","split","round"], inplace=True)
        summary.to_csv(args.out / "summary_metrics.csv", index=False)
        print("[OK] Wrote:", args.out / "summary_metrics.csv")
    else:
        print("[WARN] No score files found. Expected under run_*/split-*/round_*/")

if __name__ == "__main__":
    main()
