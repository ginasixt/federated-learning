"""federated-learning: A Flower / PyTorch app."""
# federated_learning/client_app.py
# Hält MLP und Training-/Eval-Funktionen.

import json
from typing import List, Dict
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch, torch.nn as nn, torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from federated_learning.task import load_prepared, make_loaders_for_indices

# wählt GPU, sonst CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

# define a simple MLP model for Classification
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, 2) # 2 classes (diabetes or not)
        )
    def forward(self, x): return self.net(x)

def train_one_epoch(model, loader, opt, crit):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); 
        loss = crit(model(xb), yb); 
        loss.backward(); 
        opt.step()

def evaluate(model, loader, crit):
    model.eval(); n=0; correct=0; loss_sum=0.0
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logit = model(xb)
            loss = crit(logit, yb)
            loss_sum += loss.item()*xb.size(0)
            preds = logit.argmax(1)
            correct  += (preds==yb).sum().item()
            n += xb.size(0)
            probs = torch.softmax(logit, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # Metriken berechnen
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return loss_sum/n, correct/n, auc, precision, recall, f1, specificity, cm


class FlowerClient(NumPyClient):
    def __init__(self, cid: str, rc: dict):
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
        mapping: Dict[str, List[int]] = json.loads(open(split_path).read())
        if self.cid not in mapping:
            raise KeyError(f"cid {self.cid} fehlt in {split_path}")

        # indizes für diesen Client (= cid = partition-id)
        client_train_idx = mapping[self.cid]

        # Daten für den Clienten laden
        self.train_loader, self.test_loader = make_loaders_for_indices(
            X, y, client_train_idx, test_idx, batch_size=int(rc.get("batch-size", 128))
        )

        # Jeder Client trainiert nur auf eigenen deterministischen Daten.

        # Modell wird initialisiert und auf Device (CPU/GPU) geladen
        self.model = MLP(in_dim=X.shape[1]).to(DEVICE)
        # Klassen-Gewichte berechnen TODO(ginasixt): recall FN sehr schlecht, daher Gewichte anpassen
        class_counts = np.bincount(y)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)

        # Anzahl der lokalen Epochen aus config, (default 1)
        # the config arument is send from the server (run-config) to the client
        # the server can set the number of epochs, learning rate, etc. (look at server_app.py)
        # in the next step we then train our local model.
        self.local_epochs = int(rc.get("local-epochs", 1))

     # FOR EVERY ROUND THE FOLLOWING METHODS ARE CALLED:
     # 1) fit is called to train the model locally for each client
         # 1) the current weights are set
         # 2) the model is trained on the local data for a number of epochs
         # 3) the new weights are returned to the server
    def fit(self, parameters, config):
        # Sets the model weights and trains the model on the local data
        self.set_parameters(parameters)

        opt = optim.SGD(self.model.parameters(), config.get("lr", 1e-2), momentum=0.9)
        for _ in range(int(config.get("local_epochs", self.local_epochs))):
            train_one_epoch(self.model, self.train_loader, opt, self.crit)
        
        loss, acc, auc, precision, recall, f1, specificity, cm = evaluate(self.model, self.train_loader, self.crit)
        return self.get_parameters({}), len(self.train_loader.dataset), {
            "fit_accuracy": float(acc), 
            "fit_loss":float(loss),
            "fit_auc": float(auc),
            }

    # 1.1) sets the model weights while training (fit())
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
        self.set_parameters(parameters)
        loss, acc, auc, precision, recall, f1, specificity, cm = evaluate(self.model, self.test_loader, self.crit)
        return float(loss), len(self.test_loader.dataset), {
            "accuracy": float(acc),
            "loss": float(loss),
            "auc": float(auc), 
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            
        }

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

