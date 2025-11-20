
import re
import os
import argparse
import pandas as pd
from typing import Dict, Any, Optional

# how to run:
# python3 logs/parse_flwr_logs.py --logs-root logs --min-spec 0.70 --out-csv out/agg_runs.csv 

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
INFO_PREFIX_RE = re.compile(r"^\s*(?:\x1b\[[0-9;]*[mK])?INFO\s*:\s*", re.IGNORECASE)
MARKER_RE = re.compile(r"History\s*\(metrics,\s*distributed,\s*evaluate\)\s*:\s*$", re.IGNORECASE)
THRESHOLD_HINT_RE = re.compile(r"(?:thr(?:eshold)?[_-]?)(\d+(?:\.\d+)?)", re.IGNORECASE)

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def strip_info_prefix(s: str) -> str:
    return INFO_PREFIX_RE.sub("", s)

def extract_evaluate_block(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if MARKER_RE.search(strip_ansi(line)):
            # Collect following lines, strip prefixes, until braces are balanced
            buf = []
            brace = 0
            started = False
            for j in range(i+1, len(lines)):
                raw = lines[j]
                s = strip_info_prefix(strip_ansi(raw)).strip()
                if not s:
                    continue
                # track braces
                brace += s.count("{")
                brace -= s.count("}")
                if "{" in s:
                    started = True
                if started:
                    buf.append(s)
                if started and brace <= 0:
                    break
            block = "\n".join(buf)
            # Ensure we have outer braces
            if not block.startswith("{"):
                # try to find first '{'
                k = block.find("{")
                if k != -1:
                    block = block[k:]
            if block and block.strip().endswith("}"):
                return block.strip()
    return None

def parse_metrics_dict(block: str) -> Dict[str, Dict[int, float]]:
    # Evaluate a python-like dict safely: convert single quotes to double quotes where needed
    # but tuples (r, v) must be parsed via regex
    METRIC_RE = re.compile(r"'([\w_]+)'\s*:\s*\[([^\]]*)\]")
    PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*([0-9eE\.\-]+)\s*\)")
    out: Dict[str, Dict[int, float]] = {}
    for m in METRIC_RE.finditer(block):
        name = m.group(1)
        arr = m.group(2)
        rd: Dict[int, float] = {}
        for p in PAIR_RE.finditer(arr):
            r = int(p.group(1)); v = float(p.group(2)); rd[r] = v
        if rd:
            out[name] = rd
    return out

def pick_best_round(metrics: Dict[str, Dict[int, float]], min_spec: float = 0.80) -> Optional[int]:
    recall = metrics.get("recall", {}); spec = metrics.get("spec", {}); youden = metrics.get("youden", {})
    alerts = metrics.get("alerts_per_1000", {})
    candidates = [r for r, s in spec.items() if s >= min_spec]
    if candidates:
        return max(candidates, key=lambda r: (recall.get(r, float("-inf")), youden.get(r, float("-inf")), -alerts.get(r, float("inf"))))
    if youden:
        return max(youden, key=lambda r: youden[r])
    if recall:
        return max(recall, key=lambda r: recall[r])
    return None

def sniff_threshold_from_path(path: str) -> Optional[float]:
    m = THRESHOLD_HINT_RE.search(path)
    return float(m.group(1)) if m else None

def parse_single_log(path: str, min_spec: float) -> Optional[Dict[str, Any]]:
    try:
        text = open(path, "r", errors="ignore").read()
    except Exception:
        return None
    block = extract_evaluate_block(text)
    if not block:
        return None
    metrics = parse_metrics_dict(block)
    if not metrics:
        return None
    best_r = pick_best_round(metrics, min_spec=min_spec)
    if best_r is None:
        return None
    row = {"run_id": os.path.splitext(os.path.basename(path))[0],
           "source_path": os.path.abspath(path),
           "threshold": sniff_threshold_from_path(path) or sniff_threshold_from_path(os.path.dirname(path)),
           "best_round": best_r}
    for k in ["recall","spec","f1","ppv","npv","fpr","youden","alerts_per_1000","auc","prevalence","tp","fp","tn","fn"]:
        row[k] = metrics.get(k, {}).get(best_r, None)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--min-spec", type=float, default=0.80)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    for root, _, files in os.walk(args.logs_root):
        for fn in files:
            if not fn.lower().endswith((".log",".txt",".out")): 
                continue
            path = os.path.join(root, fn)
            r = parse_single_log(path, min_spec=args.min_spec)
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Parsed {len(df)} logs -> {args.out_csv}")

if __name__ == "__main__":
    main()
