import numpy as np

def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean().item()

def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return {"accuracy": simple_accuracy(preds, labels)}