import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_recall_curve, precision_score
from sklearn.calibration import calibration_curve

def evaluate_model(y_true, y_probs, name="Model", threshold=None):
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)

    precs, recs, threshs = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precs * recs) / (precs + recs + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    
    if threshold is None:
        threshold = threshs[optimal_idx] if optimal_idx < len(threshs) else 0.5
        print(f"Calculated Optimal Threshold (Max F1): {threshold:.4f}")

    preds = (y_probs >= threshold).astype(int)
    recall = recall_score(y_true, preds)
    precision = precision_score(y_true, preds)
    f1 = f1_scores[optimal_idx] if optimal_idx < len(f1_scores) else 0.0

    print(f"--- {name} Metrics ---")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f} (threshold: {threshold:.4f})")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall: {recall:.4f}")

    return {'auprc': auprc, 'auroc': auroc, 'f1': f1, 'best_f1_threshold': threshold}

def plot_calibration_curve(y_true, y_probs, model_name="Model"):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Clinical Reliability Trace ({model_name})")
    plt.legend()
    plt.tight_layout()
    return fig
