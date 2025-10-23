"""
server_app.py — robust gegen Flower-Versionen/Strukturen
- CustomStrategy(FedAvg).aggregate_evaluate(...) wertet Client-Ergebnisse direkt aus
- Rekursive Extraktion der Histogramm-Metriken aus beliebigen Rückgabeformaten
- CSV wird direkt hier geschrieben (kein externer Import)
- on_fit_config_fn / on_evaluate_config_fn: Clients bekommen mu & Histogramm-Config
- Kompatibler Rückgabetyp via _ServerAppComponentsShim (falls deine Version .server erwartet)
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
import os, csv
import numpy as np

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


# ------------------------------------------------------------
# Kompatibilitäts-Shim (für Flower-Builds, die components.server erwarten)
# ------------------------------------------------------------
class _ServerAppComponentsShim:
    def __init__(self, strategy, config, server=None, client_manager=None):
        self.strategy = strategy
        self.config = config
        self.server = server
        self.client_manager = client_manager


# ------------------------------------------------------------
# Utilities: Histogram-Threshold & CSV
# ------------------------------------------------------------
def _find_threshold_for_target_recall(pos_hist: np.ndarray, neg_hist: np.ndarray, target_recall: float):
    pos_hist = np.asarray(pos_hist, dtype=float)
    neg_hist = np.asarray(neg_hist, dtype=float)

    tp_cum_r = np.cumsum(pos_hist[::-1])
    fp_cum_r = np.cumsum(neg_hist[::-1])
    tp_cum = tp_cum_r[::-1]
    fp_cum = fp_cum_r[::-1]

    P = float(pos_hist.sum()); N = float(neg_hist.sum())
    if P <= 0 or N < 0:
        return 0.5, dict(
            recall=0.0, specificity=0.0, precision=0.0, alerts_per_1000=0.0,
            tp=0.0, fp=0.0, tn=0.0, fn=0.0,
        )

    best_idx = None
    for i in range(len(pos_hist)):
        tp = tp_cum[i]
        if tp / P >= target_recall:
            best_idx = i
            break
    if best_idx is None:
        best_idx = 0

    tp = tp_cum[best_idx]; fp = fp_cum[best_idx]
    fn = P - tp; tn = max(0.0, N - fp)

    recall = tp / P
    specificity = tn / max(1e-12, N)
    precision = tp / max(1e-12, (tp + fp))
    alerts1k = 1000.0 * (tp + fp) / max(1e-12, (P + N))

    edges = np.linspace(0.0, 1.0, len(pos_hist) + 1)
    thr = float(edges[best_idx])

    return thr, dict(
        recall=float(recall),
        specificity=float(specificity),
        precision=float(precision),
        alerts_per_1000=float(alerts1k),
        tp=float(tp), fp=float(fp), tn=float(tn), fn=float(fn),
    )


def _append_metrics_row(mu_val: float, outdir: str, row: dict):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics.csv")
    header = ["mu","threshold","recall","specificity","precision","npv","auc","brier","ece","alerts_per_1000","loss"]
    exists = os.path.exists(path)
    full = {h: "" for h in header}
    full.update({
        "mu": mu_val,
        "threshold": row.get("threshold",""),
        "recall": row.get("recall",""),
        "specificity": row.get("specificity",""),
        "precision": row.get("precision",""),
        "npv": row.get("npv",""),
        "auc": row.get("auc",""),
        "brier": row.get("brier",""),
        "ece": row.get("ece",""),
        "alerts_per_1000": row.get("alerts_per_1000",""),
        "loss": row.get("loss",""),
    })
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(full)


# ------------------------------------------------------------
# Rekursiver Metrics-Extractor (findet hist_bins/pos_bin_*/neg_bin_* überall)
# ------------------------------------------------------------
def _find_metric_dict_anywhere(obj) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        keys = obj.keys()
        if ("hist_bins" in obj) or any(isinstance(k, str) and (k.startswith("pos_bin_") or k.startswith("neg_bin_")) for k in keys):
            return obj
        for v in obj.values():
            found = _find_metric_dict_anywhere(v)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for el in obj:
            found = _find_metric_dict_anywhere(el)
            if found is not None:
                return found
    return None


# ------------------------------------------------------------
# Custom Strategy: Aggregation direkt in aggregate_evaluate(...)
# ------------------------------------------------------------
class CustomStrategy(FedAvg):
    def __init__(self, run_cfg: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc = dict(run_cfg)

    def aggregate_evaluate(self, server_round: int, results, failures):
        """
        results: je nach Flower-Version
          - List[Tuple[ClientProxy, EvaluateRes]]
          - List[EvaluateRes]
          - oder exotischer (wir fangen alle Varianten per Rekursion ab)
        """
        # 1) Weighted average Loss (wie FedAvg)
        loss_aggregated = None
        loss_sum = 0.0
        num_examples_sum = 0
        metric_dicts: List[Dict[str, Any]] = []

        for item in results or []:
            # Versuche, loss / num_examples / metrics robust herauszuziehen
            res = None
            if isinstance(item, tuple) and len(item) == 2:
                # (ClientProxy, EvaluateRes) ODER (num_examples, metrics)?
                maybe = item[1]
                if hasattr(maybe, "loss") and hasattr(maybe, "num_examples"):
                    res = maybe
                else:
                    # könnte eine Tuple/Dict-Struktur sein – wir extrahieren unten rekursiv die Metriken
                    pass
            elif hasattr(item, "loss") and hasattr(item, "num_examples"):
                res = item

            if res is not None:
                # Standardweg (EvaluateRes)
                try:
                    n = int(getattr(res, "num_examples", 0))
                    l = float(getattr(res, "loss", 0.0))
                    if n > 0:
                        loss_sum += l * n
                        num_examples_sum += n
                    md = getattr(res, "metrics", None)
                    if md is not None:
                        found = _find_metric_dict_anywhere(md)
                        if isinstance(found, dict):
                            metric_dicts.append(found)
                except Exception:
                    pass
            else:
                # Fallback: rekursiv in 'item' suchen
                found = _find_metric_dict_anywhere(item)
                if isinstance(found, dict):
                    metric_dicts.append(found)
                # evtl. loss/num_examples flach gespeichert?
                if isinstance(item, tuple):
                    # Varianten wie (loss, num_examples, metrics)
                    if len(item) >= 2 and isinstance(item[0], (float, int)) and isinstance(item[1], (int, float)):
                        l = float(item[0]); n = int(item[1])
                        if n > 0:
                            loss_sum += l * n
                            num_examples_sum += n

        if num_examples_sum > 0:
            loss_aggregated = loss_sum / num_examples_sum

        # Debug: erste Keys zeigen
        if metric_dicts:
            sample_keys = list(metric_dicts[0].keys())[:10]
            print(f"[SERVER] sample metric keys: {sample_keys}", flush=True)
        else:
            print("[SERVER] no metrics dicts extracted", flush=True)

        # 2) Histogramme aggregieren, Threshold wählen, CSV schreiben
        target_recall = float(self.rc.get("target-recall", 0.95))
        num_bins_full = int(self.rc.get("threshold-grid-size", 101))
        num_bins = max(2, num_bins_full) - 1

        pos = np.zeros(num_bins, dtype=float)
        neg = np.zeros(num_bins, dtype=float)
        client_count = 0
        total_n = 0

        for m in metric_dicts:
            hb = int(m.get("hist_bins", -1))
            if hb == num_bins:
                client_count += 1
                total_n += int(m.get("n_eval", 0))
                for i in range(num_bins):
                    pos[i] += float(m.get(f"pos_bin_{i}", 0.0))
                    neg[i] += float(m.get(f"neg_bin_{i}", 0.0))

        print(f"[SERVER] hist_clients={client_count} pos_sum={pos.sum()} neg_sum={neg.sum()} bins={num_bins}", flush=True)

        metrics_aggregated: Dict[str, Any] = {}
        if client_count > 0:
            thr, mm = _find_threshold_for_target_recall(pos, neg, target_recall)
            metrics_aggregated = {
                "chosen_threshold": float(thr),
                "recall": mm["recall"],
                "specificity": mm["specificity"],
                "precision": mm["precision"],
                "alerts_per_1000": mm["alerts_per_1000"],
                "tp": mm["tp"], "fp": mm["fp"], "tn": mm["tn"], "fn": mm["fn"],
                "clients_in_round": float(client_count),
                "n_eval_total": float(total_n),
            }
            # CSV schreiben
            try:
                mu_val = float(self.rc.get("mu", self.rc.get("proximal-mu", 0.0)))
                outdir = f"results/mu={mu_val}"
                _append_metrics_row(mu_val, outdir, {
                    "threshold": metrics_aggregated["chosen_threshold"],
                    "recall": metrics_aggregated["recall"],
                    "specificity": metrics_aggregated["specificity"],
                    "precision": metrics_aggregated["precision"],
                    "alerts_per_1000": metrics_aggregated["alerts_per_1000"],
                    "loss": float(loss_aggregated) if loss_aggregated is not None else "",
                })
                print(f"[SERVER] wrote CSV for mu={mu_val} -> {outdir}/metrics.csv", flush=True)
            except Exception as e:
                print("[WARN] writing metrics.csv failed:", e, flush=True)

        # Rückgabe wie FedAvg: (loss_avg, metrics_aggregated)
        return loss_aggregated, metrics_aggregated


# ------------------------------------------------------------
# Server factory
# ------------------------------------------------------------
def server_fn(context):
    rc: Dict[str, Any] = dict(context.run_config)

    # Fit-Config (mu an Clients)
    def on_fit_config_fn(server_round: int):
        return {
            "mu": float(rc.get("mu", rc.get("proximal-mu", 0.0))),
        }

    # Evaluate-Config (Histogramm-Params an Clients)
    def on_evaluate_config_fn(server_round: int):
        return {
            "eval-threshold-mode": rc.get("eval-threshold-mode", "fixed"),
            "return-histogram": bool(rc.get("return-histogram", True)),
            "threshold-grid-size": int(rc.get("threshold-grid-size", 101)),
        }

    strategy = CustomStrategy(
        run_cfg=rc,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        min_fit_clients=int(rc.get("min_fit_clients", 8)),
        min_evaluate_clients=int(rc.get("min_evaluate_clients", 10)),
        min_available_clients=int(rc.get("min_available_clients", 10)),
    )

    # num_rounds aus run_config (beide Schreibweisen akzeptieren)
    num_rounds = int(rc.get("num-server-rounds", rc.get("num_rounds", 20)))
    cfg = ServerConfig(num_rounds=num_rounds)

    # Kompatibler Rückgabewert für deine Flower-Version
    return _ServerAppComponentsShim(strategy=strategy, config=cfg)


# App-Objekt (pyproject.toml referenziert :app)
app = ServerApp(server_fn=server_fn)
