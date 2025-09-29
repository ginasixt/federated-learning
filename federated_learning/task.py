"""federated-learning: A Flower / PyTorch app."""

# federated_learning/task.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Lädt die vorbereiteten Daten und Normalisierungs-Statistiken
def load_prepared(parquet_path: str, stats_path: str):
    df = pd.read_parquet(parquet_path)
    meta = json.loads(Path(stats_path).read_text())

    # Features und Target extrahieren und Normalierungparameter laden
    tgt = meta["target"]
    tr_idx = np.array(meta["train_idx"])
    te_idx = np.array(meta["test_idx"])
    mean = meta["mean"]; std = meta["std"]


    y_all = df[tgt].astype(int).values 
    X_all = df.drop(columns=[tgt, "__row_id__"]).astype(float)

    # Normalisierung der Features (Mittelwert 0, Standardabweichung 1)
        # Features (z.B. Alter, BMI, Blutdruck) unterschieldiche Wertebereiche.
        # --> würde bedeuten, Features mit großen Werten dominieren Training .
        # Mit Normalisierung ( Mittelwert 0, Standardabweichung 1) werden die Features vergleichbar skaliert.
    # Also berechen wir mean und std auf Trainingsdaten (sonst Data Leakage) und normalisiwren Train und Test mit diesen Werten.
    X_all = (X_all - pd.Series(mean)) / pd.Series(std)
    X_all = X_all.values.astype("float32")
    y_all = y_all.astype("int64")

    return X_all, y_all, tr_idx, te_idx

def make_loaders_for_indices(X, y, train_indices, test_indices, batch_size=128):
    Xtr, ytr = torch.tensor(X[train_indices]), torch.tensor(y[train_indices])
    Xte, yte = torch.tensor(X[test_indices]), torch.tensor(y[test_indices])
    tr = TensorDataset(Xtr, ytr); te = TensorDataset(Xte, yte)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=1024, shuffle=False),
    )
