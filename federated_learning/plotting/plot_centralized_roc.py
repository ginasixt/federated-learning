"""Plot ROC curves from centralized server evaluation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def load_roc_data(result_dir: Path) -> List[Dict]:
    """Load all ROC data from centralized evaluation."""
    roc_files = sorted(result_dir.glob("*.json"))
    data = []
    for f in roc_files:
        with open(f) as fp:
            data.append(json.load(fp))
    return data

def plot_roc_evolution(roc_data: List[Dict], outpath: Path):
    """Plot ROC curves for different rounds."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot für jede 5. Runde (um Plot nicht zu überladen)
    rounds_to_plot = [d for d in roc_data if d["round"] % 5 == 0 or d["round"] == roc_data[-1]["round"]]
    
    for data in rounds_to_plot:
        rnd = data["round"]
        fpr = np.array(data["roc_curve"]["fpr"])
        tpr = np.array(data["roc_curve"]["tpr"])
        auc = data["auc"]
        
        # Farbverlauf: frühe Runden → blau, späte Runden → rot
        color = plt.cm.coolwarm(rnd / max(d["round"] for d in roc_data))
        
        ax.plot(fpr, tpr, label=f'Round {rnd} (AUC={auc:.3f})', 
               color=color, linewidth=2, alpha=0.8)
        
        # Markiere optimalen Punkt
        opt_fpr = data["optimal_fpr"]
        opt_tpr = data["optimal_tpr"]
        opt_thr = data["optimal_threshold"]
        ax.scatter(opt_fpr, opt_tpr, color=color, s=100, zorder=5, marker='*')
        ax.annotate(f'{opt_thr:.2f}', (opt_fpr, opt_tpr), 
                   textcoords="offset points", xytext=(5,5), fontsize=8)
    
    # Diagonale (Random Classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves: Training Evolution (Centralized Evaluation)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"✅ ROC evolution plot saved: {outpath}")

def plot_best_roc(roc_data: List[Dict], outpath: Path):
    """Plot ROC curve for best round (highest AUC)."""
    best = max(roc_data, key=lambda x: x["auc"])
    
    fpr = np.array(best["roc_curve"]["fpr"])
    tpr = np.array(best["roc_curve"]["tpr"])
    thresholds = np.array(best["roc_curve"]["thresholds"])
    auc = best["auc"]
    rnd = best["round"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC-Kurve
    ax.plot(fpr, tpr, color='darkorange', linewidth=3, label=f'ROC (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    # Optimaler Punkt
    opt_fpr = best["optimal_fpr"]
    opt_tpr = best["optimal_tpr"]
    opt_thr = best["optimal_threshold"]
    ax.scatter(opt_fpr, opt_tpr, color='red', s=300, zorder=5, marker='*', 
              edgecolors='black', linewidth=2, label=f'Optimal (thr={opt_thr:.3f})')
    
    # Markiere auch konfigurierte Thresholds
    for thr in [0.30, 0.35, 0.40, 0.45]:
        idx = np.argmin(np.abs(thresholds - thr))
        ax.scatter(fpr[idx], tpr[idx], s=100, zorder=4, alpha=0.7)
        ax.annotate(f'{thr:.2f}', (fpr[idx], tpr[idx]), 
                   textcoords="offset points", xytext=(8,8), fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=14)
    ax.set_title(f'Best ROC Curve (Round {rnd}, AUC={auc:.4f})', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"✅ Best ROC plot saved: {outpath}")

def plot_threshold_sensitivity(roc_data: List[Dict], outpath: Path):
    """Plot metrics vs threshold for best round."""
    best = max(roc_data, key=lambda x: x["auc"])
    metrics = best["metrics_at_thresholds"]
    
    thresholds = sorted([float(k.split("_")[1]) for k in metrics.keys()])
    recalls = [metrics[f"thr_{t:.2f}"]["recall"] for t in thresholds]
    specs = [metrics[f"thr_{t:.2f}"]["spec"] for t in thresholds]
    precisions = [metrics[f"thr_{t:.2f}"]["precision"] for t in thresholds]
    f1s = [metrics[f"thr_{t:.2f}"]["f1"] for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(thresholds, recalls, marker='o', linewidth=2.5, markersize=8, 
           label='Recall (Sensitivity)', color='#2ecc71')
    ax.plot(thresholds, specs, marker='s', linewidth=2.5, markersize=8, 
           label='Specificity', color='#3498db')
    ax.plot(thresholds, precisions, marker='^', linewidth=2.5, markersize=8, 
           label='Precision (PPV)', color='#e74c3c')
    ax.plot(thresholds, f1s, marker='D', linewidth=2.5, markersize=8, 
           label='F1-Score', color='#9b59b6')
    
    # Markiere optimalen Threshold
    opt_thr = best["optimal_threshold"]
    ax.axvline(opt_thr, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Optimal (Youden) = {opt_thr:.3f}')
    
    ax.set_xlabel('Classification Threshold', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=14)
    ax.set_title(f'Threshold Sensitivity Analysis (Round {best["round"]})', fontsize=16)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    print(f"✅ Threshold sensitivity plot saved: {outpath}")

def main():
    # Lade Daten
    result_dir = Path("result/centralized_eval/alpha03/thr_0.35")
    
    if not result_dir.exists():
        print(f"❌ No centralized evaluation data found in {result_dir}")
        print("   Run 'flwr run .' first!")
        return
    
    roc_data = load_roc_data(result_dir)
    print(f"✅ Loaded {len(roc_data)} rounds of ROC data")
    
    # Output-Ordner
    out_dir = Path("plots_centralized")
    out_dir.mkdir(exist_ok=True)
    
    # Plots erstellen
    plot_roc_evolution(roc_data, out_dir / "roc_evolution.png")
    plot_best_roc(roc_data, out_dir / "roc_best.png")
    plot_threshold_sensitivity(roc_data, out_dir / "threshold_sensitivity.png")
    
    # Summary
    best = max(roc_data, key=lambda x: x["auc"])
    print(f"\n📊 SUMMARY:")
    print(f"   Best Round: {best['round']}")
    print(f"   AUC: {best['auc']:.4f}")
    print(f"   Optimal Threshold: {best['optimal_threshold']:.3f}")
    print(f"   TPR at optimal: {best['optimal_tpr']:.3f}")
    print(f"   Specificity at optimal: {best['optimal_specificity']:.3f}")

if __name__ == "__main__":
    main()