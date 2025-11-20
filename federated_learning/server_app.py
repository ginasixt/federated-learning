# federated_learning/server_app.py
from __future__ import annotations

import math
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context
from pathlib import Path
from federated_learning.screening_policy import ScreeningPolicy
import json
import numpy as np
from flwr.common.record import ConfigRecord

try:
    from flwr.server.app import ServerAppComponents
except ImportError:
    from flwr.server import ServerAppComponents


# flwr run lädt über pyproject.toml serverapp = "…server_app:app".
# Flower ruft server_fn(context) auf.
#  context.run_config (die TOML-Werte) werden gelesen
# bauen FedAvg(..., on_fit_config_fn=..., evaluate_metrics_aggregation_fn=...).
# udn geben ServerAppComponents(config, strategy) zurück.
# Ab dann orchestriert Flower die Runden (Sampling, Fit, Evaluate).
def server_fn(context: Context) -> ServerAppComponents:
    """Build strategy + config. FedAvg + multi-threshold optimization."""
    rc = dict(context.run_config)
    
    # Screening-Policy + Tracking
    screening = ScreeningPolicy()
    round_counter = {"r": 0}
    
    # Threshold-Grid
    threshold_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    def on_fit_config_fn(rnd: int) -> dict:
        """Config für Training"""
        lr = float(rc.get("lr", 1e-2)) if rnd < 3 else float(rc.get("lr-after", 5e-3))
        return {
            "epochs": int(rc.get("local-epochs", 1)),
            "lr": lr,
            "mu": float(rc.get("mu", 1e-3)),
            "weight-decay": float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
        }
    
    def on_evaluate_config_fn(rnd: int) -> dict:
        """Config für Evaluation"""
        return {
            "threshold_grid": json.dumps(threshold_grid),
        }
    
    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0
    
    def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
        """Berechne Metriken aus TP/FP/TN/FN"""
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        spec = 1.0 - fpr
        ppv = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        f1 = _safe_div(2*ppv*tpr, ppv + tpr) if (ppv + tpr) else 0.0
        bal_acc = 0.5 * (tpr + spec)
        youden = tpr + spec - 1.0
        prev = _safe_div(tp + fn, tp + fp + tn + fn)
        alerts_per_1000 = _safe_div(tp + fp, tp + fp + tn + fn) * 1000.0
        
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "recall": tpr,
            "fpr": fpr, "spec": spec,
            "ppv": ppv, "precision": ppv, "npv": npv,
            "f1": f1, "balanced_accuracy": bal_acc, "youden": youden,
            "prevalence": prev, "alerts_per_1000": alerts_per_1000,
        }
    
    def evaluate_metrics_aggregation_fn(eval_metrics: list[tuple[int, dict]]) -> dict:
        """
        Aggregate evaluation metrics from clients using multi-threshold optimization.
        
        """
        # Konvertiere ConfigRecord
        processed_metrics = []
        for n, md in eval_metrics:
            if isinstance(md, ConfigRecord):
                md_dict = dict(md.items())
            elif isinstance(md, dict):
                md_dict = md
            else:
                print(f"⚠️  Unknown metrics type: {type(md)}")
                continue
            processed_metrics.append((n, md_dict))
        
        # 1) AUC
        auc_weighted_sum = 0.0
        total_weight_for_auc = 0
        
        # n = number of samples per client and md = metrics dict, 
        for n, md in processed_metrics:
            if not isinstance(md, dict):
                continue
            auc = md.get("auc", None)
            if auc is not None:
                w = int(n) if n else 1 # so a Client with 1 samples does not break the sum
                auc_weighted_sum += float(auc) * w
                total_weight_for_auc += w
        
        aggregated_auc = auc_weighted_sum / total_weight_for_auc if total_weight_for_auc else 0.0
        
        # 2) Deserialize und Aggregiere Threshold-Counts
        threshold_aggregated = {}
        
        for n, md in processed_metrics:
            if not isinstance(md, dict):
                continue
            
            #  Parse JSON-Strings zu Listen
            try:
                thresholds = json.loads(md.get("thresholds_json", "[]"))
                tp_list = json.loads(md.get("tp_json", "[]"))
                fp_list = json.loads(md.get("fp_json", "[]"))
                tn_list = json.loads(md.get("tn_json", "[]"))
                fn_list = json.loads(md.get("fn_json", "[]"))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error: {e}")
                continue
            
            # Aggregiere Counts
            for i, thr in enumerate(thresholds):
                if thr not in threshold_aggregated:
                    threshold_aggregated[thr] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                
                threshold_aggregated[thr]["tp"] += int(tp_list[i]) if i < len(tp_list) else 0
                threshold_aggregated[thr]["fp"] += int(fp_list[i]) if i < len(fp_list) else 0
                threshold_aggregated[thr]["tn"] += int(tn_list[i]) if i < len(tn_list) else 0
                threshold_aggregated[thr]["fn"] += int(fn_list[i]) if i < len(fn_list) else 0
        
        # NAch Aggregation:
        # threshold_aggregated = {
        #   0.30: {tp: 3192, fp: 9554, tn: 12279, fn: 343},  # SUMME aller Clients!
        #   0.35: {tp: 3083, fp: 8489, tn: 13344, fn: 452},
        #   ...
        # }
        
        # 3) Berechne Metriken 
        threshold_results = []
        
        # Berechne Metriken für jeden Threshold
        for thr in sorted(threshold_aggregated.keys()):
            counts = threshold_aggregated[thr]
            metrics = _metrics_from_counts(
                counts["tp"], counts["fp"], counts["tn"], counts["fn"]
            )
            metrics["threshold"] = thr
            threshold_results.append(metrics)
        
        # 4) Wähle besten Threshold
        best_thr_result = None
        best_score = -1.0
        
        MIN_RECALL = 0.75
        MIN_SPEC = 0.70
        
        # Durchlaufe alle Threshold-Ergebnisse um den besten zu finden
        for result in threshold_results:
            recall = result["recall"]
            spec = result["spec"]
            f1 = result["f1"]
            
            # 4.1) Filtere nach MIN_RECALL
            if recall < MIN_RECALL:
                continue
            
            # 4.2) Spec Penalty is calculated if spec is below MIN_SPEC
            spec_penalty = 0.0 if spec >= MIN_SPEC else (MIN_SPEC - spec) * 0.5
            
            # 4.3) Score Berechnung
            score = (
                0.40 * recall +     # Recall wichtigster Faktor
                0.30 * spec +       # Spec auch 
                0.20 * f1 +
                0.10 * (1.0 - spec_penalty) 
            )
            
            # 4.4) Wähle basierend auf bestem Score den Threshold
            if score > best_score:
                best_score = score
                best_thr_result = result
        
        # Falls kein Threshold MIN_RECALL erfüllt, wähle besten Youden-Index
        if best_thr_result is None:
            print(f"⚠️  No threshold meets min_recall={MIN_RECALL}, using best youden index.")
            best_thr_result = max(threshold_results, key=lambda x: x["youden"])
        
        # 5) Logging
        print(f"\n🎯 Multi-Threshold Aggregation (Round {round_counter['r'] + 1}):")
        print(f"   ═══════════════════════════════════════════════")
        print(f"   Evaluated {len(threshold_results)} thresholds:")
        
        for result in threshold_results[:3]:
            print(f"     • Thr={result['threshold']:.2f}: "
                  f"Recall={result['recall']:.3f}, Spec={result['spec']:.3f}, F1={result['f1']:.3f}")
        
        print(f"   ...")
        
        print(f"\n   🏆 BEST Threshold: {best_thr_result['threshold']:.2f}")
        print(f"     • Recall:      {best_thr_result['recall']:.3f}")
        print(f"     • Specificity: {best_thr_result['spec']:.3f}")
        print(f"     • F1-Score:    {best_thr_result['f1']:.3f}")
        print(f"     • Score:       {best_score:.3f}")
        print(f"   ═══════════════════════════════════════════════\n")
        
        # 6) Speichere in History
        round_counter["r"] += 1
        rnd = round_counter["r"]
        
        agg = dict(best_thr_result)
        agg["auc"] = aggregated_auc
        agg["best_threshold"] = best_thr_result["threshold"]
        agg["best_score"] = best_score
        agg["all_thresholds"] = threshold_results
        
        screening.add_round(rnd, agg)
        
        # 7) Speichere Best Model
        best = screening.best()
        if best:
            run_tag = str(rc.get("run-tag", "none"))
            path = Path(f"result/alpha03/multi_thr/run_{run_tag}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            screening.save_best(str(path))
        
        del agg["all_thresholds"]
        return agg
    
    def fit_metrics_aggregation_fn(metrics):
        """Aggregiere Fit-Metriken"""
        n_sum = sum(n for n, _ in metrics) or 1
        keys = set().union(*(m.keys() for _, m in metrics))
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / n_sum for k in keys}
    
    strategy = FedAvg(
        fraction_fit=float(rc.get("fraction-fit", 0.8)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)), # 100% der verfügbaren Clients zum Eval ausgewählt
        min_fit_clients=int(rc.get("min-fit-clients", 8)), # Mindestens 8 Clients für das Training
        min_available_clients=int(rc.get("min-available-clients", 10)), # Mindestens 9 Clients für die Verfügbarkeit
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 10)), 
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    
    cfg = ServerConfig(num_rounds=int(rc.get("num-server-rounds", 20)))
    return ServerAppComponents(config=cfg, strategy=strategy)

# Create ServerApp
# ServerApp is the main entry point for the Flower server
# It orchestrates the federated learning process, manages clients, and controls training and evaluation.
# When we flwr run, flower reads from the pyproject.toml file and loads the configurations. 
# 
# server_fn prepares everything we need to run the server
# creates the model, defines the strategy, and sets the server config (numberof rounds)

app = ServerApp(server_fn=server_fn)


