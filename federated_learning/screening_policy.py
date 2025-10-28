import json
from typing import List, Dict, Any, Optional

class ScreeningPolicy:
    """
    Policy:
    1) Wähle Runden mit spec >= 0.75 UND recall >= 0.8. Falls vorhanden: die Runde mit höchstem recall.
    2) Sonst: Wähle aus Runden mit spec >= 0.70 die Runde mit höchstem recall.
    3) Fallback: Runde mit höchstem recall insgesamt.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_round(self, rnd: int, metrics: Dict[str, float]) -> None:
        entry = {"round": int(rnd), "metrics": {k: float(v) for k, v in metrics.items()}}
        self.history.append(entry)

    def _choose_best(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None

        def spec(m): return float(m.get("spec", 0.0))
        def recall(m): return float(m.get("recall", m.get("tpr", 0.0)))

        # 1) spec >= 0.75 & recall >= 0.8
        c1 = [h for h in self.history if spec(h["metrics"]) >= 0.75 and recall(h["metrics"]) >= 0.8]
        if c1:
            return max(c1, key=lambda h: recall(h["metrics"]))

        # 2) spec >= 0.70, pick highest recall
        c2 = [h for h in self.history if spec(h["metrics"]) >= 0.70]
        if c2:
            return max(c2, key=lambda h: recall(h["metrics"]))

        # 3) fallback: highest recall overall
        return max(self.history, key=lambda h: recall(h["metrics"]))

    def best(self) -> Optional[Dict[str, Any]]:
        return self._choose_best()

    def save_best(self, out_path: str) -> None:
        best = self.best()
        if best is None:
            return
        PathLike = out_path  # keep simple local var for type clarity
        with open(out_path, "w") as f:
            json.dump(best, f, indent=2)
