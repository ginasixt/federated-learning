from pathlib import Path
import csv

CSV_HEADER = [
    "mu","threshold","recall","specificity","precision","npv",
    "auc","brier","ece","alerts_per_1000","loss"
]

def append_metrics(mu: float, outdir: str, **kwargs):
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    csv_path = p / "metrics.csv"
    write_header = not csv_path.exists()
    row = {h: "" for h in CSV_HEADER}
    row["mu"] = mu
    for k, v in kwargs.items():
        if k in row:
            row[k] = v
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)
