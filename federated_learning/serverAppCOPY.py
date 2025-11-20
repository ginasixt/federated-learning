# federated_learning/server_app.py
from __future__ import annotations

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context
from pathlib import Path
from federated_learning.screening_policy import ScreeningPolicy
import json
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# Task-Imports für Datenladen
from federated_learning.task import load_prepared
from federated_learning.client_app import MLP, DEVICE

try:
    from flwr.server.app import ServerAppComponents
except ImportError:
    from flwr.server import ServerAppComponents


def server_fn(context: Context) -> ServerAppComponents:
    """Build strategy + config mit Centralized Evaluation."""
    rc = dict(context.run_config)

    # ✅ NEU: Server lädt GLOBALES Test-Dataset
    print("🔄 Server: Loading global test dataset...")
    X, y, train_idx, test_idx = load_prepared(
        rc["prepared-parquet"], 
        rc["norm-stats-json"]
    )
    
    # Globales Test-Set (alle Clients nutzen diese Indizes)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = y[test_idx]
    
    # ✅ Globales Modell für Server-Evaluation (gleiche Architektur wie Clients)
    in_dim = X.shape[1]
    global_model = MLP(in_dim=in_dim, hidden_dims=[256, 128], out_dim=2).to(DEVICE)
    print(f"✅ Server: Global test set loaded ({len(test_idx)} samples)")

    # Screening-Policy + Runden-Counter
    screening = ScreeningPolicy()
    round_counter = {"r": 0}

    # --- Fit Config (unverändert) ---
    def on_fit_config_fn(rnd: int) -> dict:
        lr = float(rc.get("lr", 1e-2)) if rnd < 3 else float(rc.get("lr-after", 5e-3))
        return {
            "epochs": int(rc.get("local-epochs", 1)),
            "lr": lr,
            "mu": float(rc.get("mu", 1e-3)),
            "weight-decay": float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
            "eval-threshold": float(rc.get("eval-threshold", 0.35)),
        }

    # --- Helper Funktionen (unverändert) ---
    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        spec = 1.0 - fpr
        ppv  = _safe_div(tp, tp + fp)
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

    # --- Federated Evaluation (Client-side, unverändert) ---
    def evaluate_metrics_aggregation_fn(eval_metrics: list[tuple[int, dict]]) -> dict:
        """Aggregiere Client-Metriken (TP/FP/TN/FN, AUC)."""
        TP = FP = TN = FN = 0
        total_weight_for_auc = 0
        auc_weighted_sum = 0.0

        for n, md in eval_metrics:
            tp = int(md.get("tp", 0)); fp = int(md.get("fp", 0))
            tn = int(md.get("tn", 0)); fn = int(md.get("fn", 0))
            TP += tp; FP += fp; TN += tn; FN += fn

            auc = md.get("auc", None)
            if auc is not None:
                w = int(n) if n else (tp + fp + tn + fn)
                if w:
                    auc_weighted_sum += float(auc) * w
                    total_weight_for_auc += w

        agg = _metrics_from_counts(TP, FP, TN, FN)
        if total_weight_for_auc:
            agg["auc"] = auc_weighted_sum / total_weight_for_auc

        # Runde hochzählen
        round_counter["r"] += 1
        rnd = round_counter["r"]
        threshold = str(rc.get("eval-threshold", "none"))
        run_tag = str(rc.get("run-tag", "none"))

        # Screening-Policy speichern
        screening.add_round(rnd, agg)
        path = Path(f"result/alpha03/thr_{threshold}/run_{run_tag}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        screening.save_best(str(path))

        return agg

    # ✅ NEU: CENTRALIZED Server-Side Evaluation
    def centralized_evaluate(server_round: int, parameters, config):
        """
        Server evaluiert GLOBALE Weights auf GLOBALEM Test-Set.
        → Privacy-safe: Keine Client-Daten, nur aggregierte Weights!
        """
        if parameters is None:
            return None
        
        print(f"\n🔍 Server: Centralized evaluation (Round {server_round})...")
        
        # 1. Setze globale Weights
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.array(v)) for k, v in params_dict}
        global_model.load_state_dict(state_dict, strict=True)
        
        # 2. Evaluiere auf globalem Test-Set
        global_model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(DEVICE)
            logits = global_model(X_test_device)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        # 3. Berechne ROC-Kurve (threshold-unabhängig!)
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        auc = float(roc_auc_score(y_test, probs))
        
        # 4. Finde optimalen Threshold (Youden's Index)
        youden_index = tpr - fpr
        optimal_idx = int(np.argmax(youden_index))
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_tpr = float(tpr[optimal_idx])
        optimal_fpr = float(fpr[optimal_idx])
        
        # 5. Berechne Metriken für verschiedene Thresholds
        metrics_at_thresholds = {}
        for thr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            preds = (probs >= thr).astype(int)
            tp = int(((preds == 1) & (y_test == 1)).sum())
            fp = int(((preds == 1) & (y_test == 0)).sum())
            tn = int(((preds == 0) & (y_test == 0)).sum())
            fn = int(((preds == 0) & (y_test == 1)).sum())
            
            m = _metrics_from_counts(tp, fp, tn, fn)
            metrics_at_thresholds[f"thr_{thr:.2f}"] = {
                "recall": m["recall"],
                "spec": m["spec"],
                "precision": m["precision"],
                "f1": m["f1"],
                "alerts_per_1000": m["alerts_per_1000"],
            }
        
        # 6. Speichere ROC-Daten
        roc_data = {
            "round": server_round,
            "auc": auc,
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": optimal_tpr,
            "optimal_fpr": optimal_fpr,
            "optimal_specificity": 1.0 - optimal_fpr,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "metrics_at_thresholds": metrics_at_thresholds,
        }
        
        # Speichere in dediziertem Ordner
        threshold_config = str(rc.get("eval-threshold", "none"))
        run_tag = str(rc.get("run-tag", "none"))
        path = Path(f"result/centralized_eval/alpha03/thr_{threshold_config}/run_{run_tag}_round_{server_round}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(roc_data, f, indent=2)
        
        print(f"✅ Server: AUC={auc:.4f}, Optimal Threshold={optimal_threshold:.3f}")
        print(f"   Saved: {path}")
        
        # Return loss und metrics für Flower
        return 0.0, {"auc_centralized": auc, "optimal_threshold": optimal_threshold}

    # --- Fit Metrics (unverändert) ---
    def fit_metrics_aggregation_fn(metrics):
        n_sum = sum(n for n, _ in metrics) or 1
        keys = set().union(*(m.keys() for _, m in metrics))
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / n_sum for k in keys}

    # ✅ Strategy mit BEIDEN Evaluation-Methoden
    strategy = FedAvg(
        fraction_fit=float(rc.get("fraction-fit", 0.5)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 5)),
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 10)),
        on_fit_config_fn=on_fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Federated
        evaluate_fn=centralized_evaluate,  # ✅ NEU: Centralized!
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    cfg = ServerConfig(num_rounds=int(rc.get("num-server-rounds", 10)))

    return ServerAppComponents(config=cfg, strategy=strategy)


app = ServerApp(server_fn=server_fn)


