# plot_mu_hist_and_middle.py
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TARGETS = [0.90, 0.92, 0.95]  # mehrere Recall-Ziele
FIG_DPI = 160

def load_all_hist():
    rows = []  # (mu, round, pos_hist, neg_hist, edges, summary)
    for mu_dir in sorted(Path("results").glob("mu=*/")):
        try:
            mu = float(mu_dir.name.split("=")[1])
        except:
            continue
        for rdir in sorted(mu_dir.glob("round=*/")):
            npz = rdir / "hist_agg.npz"
            if not npz.exists():
                continue
            data = np.load(npz)
            pos = data["pos_hist"].astype(float)
            neg = data["neg_hist"].astype(float)
            edges = data["edges"].astype(float)
            summary = {}
            if (rdir / "summary.json").exists():
                summary = json.loads((rdir / "summary.json").read_text())
            rows.append((mu, rdir.name, pos, neg, edges, summary))
    return rows

def pick_last_round_per_mu(rows):
    # nimm pro μ die letzte Runde (höchste round=XXX)
    per_mu = {}
    for mu, rname, pos, neg, edges, summary in rows:
        rid = int(rname.split("=")[1])
        if mu not in per_mu or rid > per_mu[mu][0]:
            per_mu[mu] = (rid, pos, neg, edges, summary)
    # sortiert nach mu
    out = []
    for mu in sorted(per_mu.keys()):
        rid, pos, neg, edges, summary = per_mu[mu]
        out.append((mu, rid, pos, neg, edges, summary))
    return out

def tpr_fpr_curves_from_hist(pos, neg):
    # cum ab hoher Schwelle
    tp_c = np.cumsum(pos[::-1])[::-1]
    fp_c = np.cumsum(neg[::-1])[::-1]
    P = pos.sum()
    N = neg.sum()
    if P <= 0 or N <= 0:
        return None, None
    tpr = tp_c / P
    fpr = fp_c / N
    return tpr, fpr  # Index i entspricht edges[i]

def threshold_for_target_recall(pos, neg, edges, target_recall):
    tpr, fpr = tpr_fpr_curves_from_hist(pos, neg)
    if tpr is None:
        return 0.5, dict(recall=0.0, specificity=0.0, precision=0.0, alerts_per_1000=0.0)
    # kleinster Index mit TPR >= Ziel
    idx = None
    for i in range(len(tpr)):
        if tpr[i] >= target_recall:
            idx = i; break
    if idx is None:
        idx = 0
    P = pos.sum(); N = neg.sum()
    tp = tpr[idx]*P
    fp = fpr[idx]*N
    fn = P - tp
    tn = N - fp
    spec = tn / max(N, 1e-12)
    prec = tp / max(tp+fp, 1e-12)
    alerts1k = 1000.0 * (tp+fp) / max(P+N, 1e-12)
    thr = float(edges[idx])
    return thr, dict(recall=float(tpr[idx]), specificity=float(spec),
                     precision=float(prec), alerts_per_1000=float(alerts1k))

def plot_hist(mu, edges, pos, neg, outdir):
    # einfache Linienplots der Häufigkeiten
    x = 0.5*(edges[:-1]+edges[1:])
    plt.figure()
    plt.plot(x, pos, label="Positiv (y=1)")
    plt.plot(x, neg, label="Negativ (y=0)")
    plt.xlabel("Score")
    plt.ylabel("Häufigkeit (Bin-Zählung)")
    plt.title(f"Histogramme (μ={mu:g})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    (outdir / f"hist_mu={mu:g}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"hist_mu={mu:g}.png", dpi=FIG_DPI)
    plt.close()

def main():
    rows = load_all_hist()
    if not rows:
        print("Keine Histogramme gefunden. Erwarte files unter results/mu=*/round=*/hist_agg.npz")
        return
    per_mu = pick_last_round_per_mu(rows)

    # 1) Histograms plotten und 2) Metriken für mehrere Ziel-Recalls berechnen
    outdir = Path("mu_report")
    outdir.mkdir(exist_ok=True)
    table = []  # sammelt per μ: mittel-Spez, Varianz, bestes μ etc.

    for mu, rid, pos, neg, edges, summary in per_mu:
        # 1) Speichere Histogramm-Plot
        plot_hist(mu, edges, pos, neg, outdir)

        # 2) Metriken über Ziel-Recalls
        specs = []
        thr_map = {}
        for tgt in TARGETS:
            thr, m = threshold_for_target_recall(pos, neg, edges, tgt)
            specs.append(m["specificity"])
            thr_map[tgt] = dict(threshold=thr, **m)
        mean_spec = float(np.mean(specs)) if specs else float("nan")
        std_spec  = float(np.std(specs)) if specs else float("nan")
        # Score für "perfekte Mitte": hohe mittlere Spez + geringe Streuung
        score = mean_spec - 0.5*std_spec

        table.append({
            "mu": mu,
            "round": rid,
            "mean_specificity": mean_spec,
            "std_specificity": std_spec,
            "score": score,
            "per_targets": thr_map
        })

    # 3) „perfekte Mitte“ wählen
    table_sorted = sorted(table, key=lambda r: (-r["score"], r["mu"]))
    best = table_sorted[0]

    # 4) Zusammenfassung als JSON
    report = {
        "targets": TARGETS,
        "ranking": table_sorted,
        "best_mu": best["mu"],
        "best_round": best["round"],
        "best_mean_specificity": best["mean_specificity"],
        "best_std_specificity": best["std_specificity"],
    }
    (outdir / "mu_middle_report.json").write_text(json.dumps(report, indent=2))

    # 5) zusätzlich: μ→Spezifität pro Ziel als Kurven
    mus = [r["mu"] for r in per_mu]
    for tgt in TARGETS:
        y = []
        for _, _, pos, neg, edges, _ in per_mu:
            _, m = threshold_for_target_recall(pos, neg, edges, tgt)
            y.append(m["specificity"])
        plt.figure()
        plt.plot(mus, y, marker="o")
        plt.xlabel("μ")
        plt.ylabel(f"Spezifität @ Recall ≥ {tgt}")
        plt.title(f"Spezifität vs μ bei Recall-Ziel {tgt}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"mu_vs_specificity_at_recall_{tgt}.png", dpi=FIG_DPI)
        plt.close()

    print(f"Beste 'Mitte' nach Score: μ={best['mu']} (Round {best['round']})")
    print(f"Details: mu_report/mu_middle_report.json")
    print("Histogramm-Plots liegen in: mu_report/*.png")

if __name__ == "__main__":
    main()
