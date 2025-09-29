# tools/make_splits.py
# Jeder Client liest später dieses Mapping ein und lädt nur die Zeilen, die ihm zugewiesen wurden, als Trainingsdaten.
# entweder IID (gleichverteilte, zufällige Aufteilung) oder Dirichlet (nicht-iid, realistischere Verteilung)
import json, numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

# IID Partitionen: Daten werden zufällig und gleichmäßig auf die Clients verteilt
def iid_partitions(n, k, seed=123):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    parts = np.array_split(idx, k)
    return {str(i): parts[i].tolist() for i in range(k)}

# Dirichlet Partitionen: Daten werden ungleichmäßig verteilt, basierend auf einer Dirichlet-Verteilung
# Verteilung der Klassen pro Client variiert, was realistischere Szenarien simuliert
def dirichlet_partitions(labels, k, alpha=0.3, seed=123):
    rng = np.random.default_rng(seed)
    label2idx = defaultdict(list)
    for i, y in enumerate(labels):
        label2idx[int(y)].append(i)
    mapping = {str(i): [] for i in range(k)}
    for _, cls_idx in label2idx.items():
        cls_idx = np.array(cls_idx); rng.shuffle(cls_idx)
        p = rng.dirichlet([alpha]*k)
        splits = (np.cumsum(p)*len(cls_idx)).astype(int)
        parts = np.split(cls_idx, splits[:-1])
        for i in range(k):
            mapping[str(i)].extend(parts[i].tolist())
    for i in range(k): rng.shuffle(mapping[str(i)])
    return mapping

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-partitions", type=int, required=True)
    p.add_argument("--mode", choices=["iid","dirichlet"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    # Lade Daten und Metadaten 
    df = pd.read_parquet(a.parquet) 
    meta = json.loads(Path(a.stats).read_text())
    y_train = df.iloc[meta["train_idx"]][meta["target"]].astype(int).values

    # Erstelle Partitionen basierend auf dem gewählten Modus
    if a.mode == "iid":
        local = iid_partitions(len(meta["train_idx"]), a.num_partitions, seed=a.seed)
    else:
        local = dirichlet_partitions(y_train, a.num_partitions, alpha=a.alpha, seed=a.seed)

    # Die Indizes der Trainingsdaten im Parquet
    tr_idx = np.array(meta["train_idx"])
    # global_map: client_id -> list of global indices in the original dataset
    global_map = {cid: tr_idx[np.array(idxs)].tolist() for cid, idxs in local.items()}
    # Zum Beispiel Beispiel:
        # tr_idx = [10, 20, 30, 40] // Indizes der Trainingsdaten im Parquet
        # local["0"] = [0, 2] , wenn Client 0, die Daten in Position 0 und 2 hat, 
        # dann ist global_map["0"] = [10, 30]

    # Speichern das Mapping (Partionierung (Simulierte Clients umgebung)) in eine JSON Datei
    Path(a.out).write_text(json.dumps(global_map))
    print("OK ->", a.out)

            # --> 
            # global_map = {
            #   "0": [10, 30, 50, ...], // ein Krankenhaus mit den Partient:innen
            #   "1": [20, 40, 60, ...],
            #   ...
            # }
