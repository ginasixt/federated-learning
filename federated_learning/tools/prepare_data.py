# tools/prepare_data.py
import json
import os
from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare(csv_path, out_parquet, out_stats, test_size=0.2, seed=123):
    # Load Daraset from Kaggle
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(csv_path)
    
    # devide feature set and target (diabetes or not)
    target_col = "Diabetes_binary"  # ggf. anpassen
    y = df[target_col].astype(int).values # target values (0/1)
    X = df.drop(columns=[target_col]).astype(float) # feature set (alles außer target)

    # Splitting into Train und Test (stratified)
    # startified = gleiche Verteilung der Klassen in Train und Test
    idx = np.arange(len(df)) # wie viele Zeilen hat der Datensatz
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    # Normalisierungs-Statistiken speichern
    # Mean ist der Mittelwert, also Durchschnittswert eines Features
    # std die Standardabweichung (wie weit sind die Daten im Durchschnitt vom Mittelwert entfernt)
    # Berechen diese Werte aber nur auf Trainingsdaten um Data Leakage zu vermeiden
    mean = X.iloc[tr_idx].mean().to_dict() # Mittelwert - average of values
    std  = (X.iloc[tr_idx].std().replace(0, 1)).to_dict() # Standartabweichung - Streuung

    # Jetzt speichern wir die Daten im Parquet Format, 
    X[target_col] = y # fügt das target wieder in die Feature Matrix ein
    X["__row_id__"] = idx # idx matcht die Reihenfolge der Daten im Parquet File für die Indices Zuordnung
    # Parquet ist ein spaltenbasiertes, komprimiertes Format, das effiziente Speicherung und schnellen Zugriff auf große Datenmengen ermöglicht.
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True) 
    X.to_parquet(out_parquet, index=False)

    # Im Json out_stats speichern wir jetzt
        # Normalisierungs-Statistiken, 
        # die Train/Test Indices (idx) passend zu unserem Parquet File,
        # und die Target-Spalte 
    stats = {"mean": mean, "std": std, "train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist(), "target": target_col}
    Path(out_stats).write_text(json.dumps(stats))
    print("OK:", out_parquet, out_stats)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--stats", required=True)
    a = p.parse_args()
    prepare(a.csv, a.parquet, a.stats)
