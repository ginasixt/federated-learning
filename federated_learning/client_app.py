"""federated-learning: A Flower / PyTorch app."""
# federated_learning/client_app.py
# Hält MLP und Training-/Eval-Funktionen.
from __future__ import annotations

import json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.metrics import roc_auc_score

from federated_learning.task import load_prepared, make_loaders_for_indices

# wählt GPU, sonst CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

# define a simple MLP model for Classification
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [128, 64], out_dim: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Training/Eval-Utilities ---
def train_one_epoch(model: nn.Module, loader, opt, crit, prox_mu: float, global_params: List[torch.Tensor], clip_norm: float):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        ce = crit(logits, yb)

        prox = 0.0
        if prox_mu > 0.0:
            # FedProx: (mu/2) * ||w - w_global||^2
            prox_term = 0.0
            for w, w0 in zip(model.parameters(), global_params):
                prox_term = prox_term + torch.sum((w - w0) ** 2)
            prox = 0.5 * prox_mu * prox_term

        loss = ce + prox
        loss.backward()
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        opt.step()


def evaluate(model: nn.Module, loader, crit, threshold: float) -> Tuple[float, int, dict]:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # Zähler für globale Metriken
    tp = fp = tn = fn = 0
    probs_all = []
    y_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            # Wahrscheinlichkeiten für Klasse 1
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            tp += int(((preds == 1) & (yb == 1)).sum())
            fp += int(((preds == 1) & (yb == 0)).sum())
            tn += int(((preds == 0) & (yb == 0)).sum())
            fn += int(((preds == 0) & (yb == 1)).sum())

            probs_all.append(probs.detach().cpu())
            y_all.append(yb.detach().cpu())

    # AUC ist threshold-unabhängig
    try:
        if probs_all:
            p = torch.cat(probs_all).numpy()
            y = torch.cat(y_all).numpy()
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    metrics = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "auc": auc}
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, n_samples, metrics


# --- Flower-Client ---
class FlowerClient(NumPyClient):
    def __init__(self, cid: str, rc: dict):
        super().__init__()
        # Client-ID, Run-Config von client_fn
        self.cid = str(cid)
        self.rc = rc

        # Normalisierten Daten werden für diesen Client geladen
        X, y, train_idx, test_idx = load_prepared(
            rc["prepared-parquet"], rc["norm-stats-json"] # Pfade aus run_config
        )

        # Split-JSON lesen und passende Indizes auswählen
        split_path = rc.get("split-path")
        if not split_path:
            raise RuntimeError("run_config['split-path'] fehlt.")
        mapping: Dict[str, List[int]] = json.loads(open(split_path, "r").read())
        if self.cid not in mapping:
            raise KeyError(f"cid {self.cid} fehlt in {split_path}")
        
        # indizes für diesen Client (= cid = partition-id)
        client_train_idx = mapping[self.cid]

        # Optionaler Check: alle Client-Indizes müssen im globalen Train liegen
        tr_set = set(int(i) for i in train_idx)
        if not all(int(i) in tr_set for i in client_train_idx):
            raise ValueError(f"Split {self.cid} enthält Indizes außerhalb des globalen Train-Splits.")

        # 3) DataLoader bauen (Train: nur Client-Zeilen; Test: global)
        bs = int(rc.get("batch-size", 128))
        self.train_loader, self.test_loader = make_loaders_for_indices(X, y, client_train_idx, test_idx, batch_size=bs)

        # 4) Modell + Loss
        # Modell wird initialisiert und auf Device (CPU/GPU) geladen
        self.model = MLP(in_dim=X.shape[1]).to(DEVICE)

        # Class-Weights aus *globalem* Train-Split (nicht client-lokal)
        y_train_global = y[train_idx]
        pos = int((y_train_global == 1).sum())
        neg = int((y_train_global == 0).sum())
        tot = max(1, pos + neg)
        w_pos = neg / tot
        w_neg = pos / tot
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)

        # Defaults / eval threshold
        self.default_lr = float(rc.get("lr", 1e-2))
        
        # Anzahl der lokalen Epochen aus config, (default 1)
        # the config arument is send from the server (run-config) to the client
        # the server can set the number of epochs, learning rate, etc. (look at server_app.py)
        # in the next step we then train our local model.
        self.local_epochs = int(rc.get("local-epochs", 1))
        self.eval_threshold = float(rc.get("eval-threshold", 0.42))


    
     # FOR EVERY ROUND THE FOLLOWING METHODS ARE CALLED:
     # 1) fit is called to train the model locally for each client
         # 1) the current weights are set
         # 2) the model is trained on the local data for a number of epochs
         # 3) the new weights are returned to the server
    def fit(self, parameters, config):
        # Sets the model weights and trains the model on the local data
        self.set_parameters(parameters)
        global_params = [p.clone().to(DEVICE) for p in self.model.parameters()]

        lr = float(config.get("lr", self.default_lr))
        epochs = int(config.get("epochs", self.local_epochs))
        mu = float(config.get("mu", 0.0))
        wd = float(config.get("weight-decay", 1e-4))
        clip = float(config.get("clip-grad-norm", 5.0))

        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # Training
        fit_total = 0
        fit_correct = 0
        fit_ce_sum = 0.0

        self.model.train()
        for _ in range(epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = self.model(xb)
                ce = self.crit(logits, yb)

                # FedProx-Term
                prox = 0.0
                if mu > 0.0:
                    prox_term = 0.0
                    for w, w0 in zip(self.model.parameters(), global_params):
                        prox_term = prox_term + torch.sum((w - w0) ** 2)
                    prox = 0.5 * mu * prox_term

                loss = ce + prox
                loss.backward()
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                opt.step()

                bs = xb.size(0)
                fit_total += bs
                fit_correct += (logits.argmax(1) == yb).sum().item()
                fit_ce_sum += ce.item() * bs  # nur CE loggen

        fit_metrics = {
            "fit_loss": fit_ce_sum / max(1, fit_total),
            "fit_accuracy": fit_correct / max(1, fit_total),
        }
        return self.get_parameters({}), len(self.train_loader.dataset), fit_metrics

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)
    # 1.2) After the training is complete, we return the current model weights to the server
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    # 2) this method is called to evaluate the local clients model
        # 1) the new weights are set
        # 2) the model is evaluated on the local test data
        # 3) Loss and accurany are returned
    def evaluate(self, parameters, config):
        # globale Gewichte evaluieren (post-aggregation)
        self.set_parameters(parameters)
        thr = float(config.get("eval-threshold", self.eval_threshold))
        loss, n, metrics = evaluate(self.model, self.test_loader, self.crit, threshold=thr)
        return float(loss), int(n), metrics



# wird von Flower für jeden Client aufgerufen
def client_fn(context: Context):
    # run_config als dict holen
    rc = dict(context.run_config)

    cid = None
    if hasattr(context, "node_config"):
        # cid(partition-id) wird aus der run_config gholt
        cid = context.node_config.get("partition-id")
    if cid is None:
        raise RuntimeError("No client id (cid) found in context.node_config or run_config.")
    # erzeugen von FlowerClient mit Client-ID und Run-Config
    return FlowerClient(str(cid), rc).to_client()

app = ClientApp(client_fn=client_fn)

