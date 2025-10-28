# federated_learning/server_app.py
from __future__ import annotations

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context
from pathlib import Path
from federated_learning.screening_policy import ScreeningPolicy
import json

try:
    # je nach Version liegt es hier:
    from flwr.server.app import ServerAppComponents
except ImportError:
    # oder hier (Fallback, falls sich der Pfad minimal unterscheidet)
    from flwr.server import ServerAppComponents  # type: ignore


# flwr run lädt über pyproject.toml serverapp = "…server_app:app".
# Flower ruft server_fn(context) auf.
#  context.run_config (die TOML-Werte) werden gelesen
# bauen FedAvg(..., on_fit_config_fn=..., evaluate_metrics_aggregation_fn=...).
# udn geben ServerAppComponents(config, strategy) zurück.
# Ab dann orchestriert Flower die Runden (Sampling, Fit, Evaluate).
def server_fn(context: Context) -> ServerAppComponents:
    """Build strategy + config. FedAvg + stabilere Settings + robuste Metrik-Aggregation."""
    rc = dict(context.run_config)

    # Screening-Policy Klasse erstellen für die Runden-Auswahl
    screening = ScreeningPolicy()
    round_counter = {"r": 0}

    # --- Pro-Runden-Konfiguration für Clients ---
    def on_fit_config_fn(rnd: int) -> dict:

        lr = float(rc.get("lr", 1e-2)) if rnd < 3 else float(rc.get("lr-after", 5e-3))
        return {
            "epochs": int(rc.get("local-epochs", 1)),
            "lr": lr,
            # FedProx-Parameter (μ=0.0 schaltet Prox aus)
            "mu": float(rc.get("mu", 1e-3)),
            # Sanfte Regularisierung/Stabilisierung
            "weight-decay": float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
            # Feste globale Eval-Schwelle (optional, z. B. 0.35 statt 0.5), das bede
            "eval-threshold": float(rc.get("eval-threshold", 0.42)),
        }

    # --- Metrik-Aggregation ---
    # Wir unterstützen zwei Pfade:
    # 1) klassisch: gew. Mittel fertiger Metrics (AUC etc.)
    # 2) robust: TP/FP/TN/FN zuerst summieren, daraus globale Präzision/Recall/F1/Spez berechnen
    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
        tpr = _safe_div(tp, tp + fn)                # = Recall / Sensitivität
        fpr = _safe_div(fp, fp + tn)
        spec = 1.0 - fpr                            # Spezifität
        ppv  = _safe_div(tp, tp + fp)               # = Precision / PPV
        npv  = _safe_div(tn, tn + fn)
        prec = ppv
        rec  = tpr
        f1   = _safe_div(2*prec*rec, prec + rec) if (prec + rec) else 0.0
        bal_acc = 0.5 * (tpr + spec)
        youden  = tpr + spec - 1.0
        prev    = _safe_div(tp + fn, tp + fp + tn + fn)
        alerts_per_1000 = _safe_div(tp + fp, tp + fp + tn + fn) * 1000.0

        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "recall": rec,
            "fpr": fpr, "spec": spec,
            "ppv": ppv, "precision": prec, "npv": npv,
            "f1": f1, "balanced_accuracy": bal_acc, "youden": youden,
            "prevalence": prev, "alerts_per_1000": alerts_per_1000,
        }

    def evaluate_metrics_aggregation_fn(eval_metrics: list[tuple[int, dict]]) -> dict:
        """
        - Jeder Client führt seine evaluate aus und schickt (loss, n, metrics) zurück.
        - Flower sammelt diese Tupel und ruft diese Funktion auf.
        eval_metrics: Liste von (num_examples, metrics_dict) vom Server gesammelt.
        Erwartet in metrics_dict mindestens: {'tp','fp','tn','fn'} und optional 'auc'.
        """
        # 1) Zählwerte micro-aggregieren (über alle Clients summieren)
        TP = FP = TN = FN = 0
        total_weight_for_auc = 0
        auc_weighted_sum = 0.0

        for n, md in eval_metrics:
            tp = int(md.get("tp", 0)); fp = int(md.get("fp", 0))
            tn = int(md.get("tn", 0)); fn = int(md.get("fn", 0))
            TP += tp; FP += fp; TN += tn; FN += fn

            # AUC sinnvoll mitteln (gewichtete Mittelung nach n oder nach (tp+fp+tn+fn))
            auc = md.get("auc", None)
            if auc is not None:
                w = int(n) if n else (tp + fp + tn + fn)
                if w:
                    auc_weighted_sum += float(auc) * w
                    total_weight_for_auc += w

        # 2) Screening-Metriken aus Gesamtsummen
        agg = _metrics_from_counts(TP, FP, TN, FN)

        # 3) AUC anhängen (falls vorhanden)
        if total_weight_for_auc:
            agg["auc"] = auc_weighted_sum / total_weight_for_auc

        # Runde hochzählen (Flower ruft diese Funktion einmal pro Runde auf)
        round_counter["r"] += 1
        rnd = round_counter["r"]
        threshold = str(rc.get("eval-threshold", "none"))
        run_tag = str(rc.get("run-tag", "none"))

        # Screening-Policy aktualisieren & speichern
        screening.add_round(rnd, agg)
        path = Path(f"result/idd/thr_{threshold}/run_{run_tag}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        screening.save_best(str(path))

        return agg


    def fit_metrics_aggregation_fn(metrics):
        n_sum = sum(n for n, _ in metrics) or 1
        keys = set().union(*(m.keys() for _, m in metrics))
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / n_sum for k in keys}

    strategy = FedAvg(
        fraction_fit=float(rc.get("fraction-fit", 0.5)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 5)),
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 10)),
        on_fit_config_fn=on_fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # server_momentum=0.9,  # FedAvgM
    )

    cfg = ServerConfig(num_rounds=int(rc.get("num-server-rounds", 10)))


    return ServerAppComponents(config=cfg, strategy=strategy)

# Create ServerApp
# ServerApp is the main entry point for the Flower server
# It orchestrates the federated learning process, manages clients, and controls training and evaluation.
# When we flwr run, flower reads from the pyproject.toml file and loads the configurations. 
# 
# server_fn prepares everything we need to run the server
# # creates the model, defines the strategy, and sets the server config (numberof rounds)

app = ServerApp(server_fn=server_fn)


