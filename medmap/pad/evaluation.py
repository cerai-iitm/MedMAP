"""PAD-UFES-20 test metrics and per-subgroup accuracy reporting."""

import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from medmap.pad.preprocessing import AGE_LABELS, SUBGROUP_COLS, bucket_age_label


@torch.no_grad()
def compute_test_metrics(model, x_te: np.ndarray, y_te: np.ndarray,
                         device, class_names: np.ndarray):
    model.eval()
    x_t = torch.tensor(x_te, dtype=torch.float32).to(device)
    preds = model(x_t).argmax(1).cpu().numpy()

    labels = np.arange(len(class_names))
    bal_acc = balanced_accuracy_score(y_te, preds)
    macro_f1 = f1_score(y_te, preds, labels=labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=labels)

    print("\n=== Test Metrics ===")
    print(f"Accuracy:          {100*np.mean(preds == y_te):.2f}%")
    print(f"Balanced accuracy: {100*bal_acc:.2f}%")
    print(f"Macro F1:          {100*macro_f1:.2f}%")
    print("\nPer-class metrics (one-vs-rest):")
    print(f"{'Class':<8} {'TPR':>8} {'FPR':>8} {'TNR':>8} {'FNR':>8}")

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fnr = fn / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        tnr = tn / (fp + tn) if fp + tn else 0.0
        print(f"{str(cls):<8} {100*tpr:7.2f}% {100*fpr:7.2f}% "
              f"{100*tnr:7.2f}% {100*fnr:7.2f}%")

    return preds


@torch.no_grad()
def compute_subgroup_accuracies(model, x_te: np.ndarray, y_te: np.ndarray,
                                subgroup_te: dict, device,
                                run_meta: dict, subgroup_csv_path: str):
    model.eval()
    x_t = torch.tensor(x_te, dtype=torch.float32).to(device)
    preds = model(x_t).argmax(1).cpu().numpy()
    correct = preds == y_te

    write_header = not os.path.exists(subgroup_csv_path)
    csv_rows = []

    print("\n=== Subgroup Test Accuracies ===")

    age_bucket = np.array([bucket_age_label(a) for a in subgroup_te['age']])
    print("age:")
    for label in AGE_LABELS + ['Unknown']:
        mask = age_bucket == label
        n = int(mask.sum())
        if n:
            acc = correct[mask].mean() * 100
            print(f"  {label:<8}: acc={acc:6.2f}%  (n={n})")
            csv_rows.append({**run_meta, 'subgroup_column': 'age',
                             'raw_value': label, 'n': n, 'accuracy': round(float(acc), 2)})

    for col in SUBGROUP_COLS:
        print(f"{col}:")
        vals = np.asarray(subgroup_te[col], dtype=object)
        for value in sorted(pd.unique(vals).tolist(), key=str):
            mask = vals == value
            n = int(mask.sum())
            if n:
                acc = correct[mask].mean() * 100
                print(f"  {str(value):<20}: acc={acc:6.2f}%  (n={n})")
                csv_rows.append({**run_meta, 'subgroup_column': col,
                                 'raw_value': str(value), 'n': n,
                                 'accuracy': round(float(acc), 2)})

    fieldnames = list(run_meta.keys()) + ['subgroup_column', 'raw_value', 'n', 'accuracy']
    with open(subgroup_csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(csv_rows)
    print(f"Appended {len(csv_rows)} subgroup rows to {subgroup_csv_path}")
