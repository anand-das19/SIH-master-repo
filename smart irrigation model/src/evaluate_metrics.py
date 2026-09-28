import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_classification(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Computes standard binary classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else "N/A",
        "confusion_matrix": cm
    }

    print(f"\n--- Evaluation Report: {model_name} ---")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score  : {metrics['f1_score'] * 100:.2f}%")
    if auc is not None:
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"  {cm}")
    print("----------------------------------------\n")

    return metrics

if __name__ == "__main__":
    print("[*] Testing evaluate_metrics harness with baseline simulation...")
    y_test_sim = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    y_pred_sim = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    y_prob_sim = np.array([0.1, 0.9, 0.2, 0.85, 0.3, 0.45, 0.95, 0.15, 0.25, 0.88])
    evaluate_classification(y_test_sim, y_pred_sim, y_prob_sim, model_name="Mock Baseline")
