import torch
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    """
    Computes classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": acc, "f1_score": f1}
