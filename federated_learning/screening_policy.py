# screening_policy.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

def _safe_div(a: float, b: float) -> float:
    return float(a)/float(b) if b else 0.0

def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    tpr = _safe_div(tp, tp+fn)          # recall / sensitivity
    fpr = _safe_div(fp, fp+tn)          # false positive rate
    spec = 1.0 - fpr                    # specificity is the true negative rate, 70% of all negatives (without Diabetes) are correctly identified
    ppv  = _safe_div(tp, tp+fp)         # precision / PPV
    npv  = _safe_div(tn, tn+fn)         # negative predictive value
    f1   = _safe_div(2*ppv*tpr, ppv+tpr) if (ppv+tpr) else 0.0
    bal  = 0.5*(tpr+spec)
    youd = tpr + spec - 1.0
    prev = _safe_div(tp+fn, tp+fp+tn+fn)
    alerts_per_1000 = _safe_div(tp+fp, tp+fp+tn+fn)*1000.0
    return {
        "tpr": tpr, "fpr": fpr, "spec": spec, "ppv": ppv, "npv": npv,
        "f1": f1, "balanced_accuracy": bal, "youden": youd,
        "prevalence": prev, "alerts_per_1000": alerts_per_1000,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

def policy_ok(m: Dict[str,float], tpr_target: float,
              fpr_max: Optional[float]=None,
              alerts_max: Optional[float]=None) -> bool:
    if m["tpr"] < tpr_target:
        return False
    if fpr_max is not None and m["fpr"] > fpr_max:
        return False
    if alerts_max is not None and m["alerts_per_1000"] > alerts_max:
        return False
    return True

def policy_sort_key(m: Dict[str,float]) -> Tuple:
    """
    Sortierung für 'beste' Runde unter erfüllter Policy:
    1) höhere PPV besser (deshalb -ppv),
    2) weniger Alerts/1000,
    3) höherer Youden,
    4) höherer TPR (nice-to-have).
    """
    return (-m["ppv"], m["alerts_per_1000"], -m["youden"], -m["tpr"])
