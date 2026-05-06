import numpy as np
from typing import List, Dict, Any

def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean().item()

def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return {"accuracy": simple_accuracy(preds, labels)}

def build_compute_metrics(class_names: List[str]):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    except Exception:
        print("[Warn] scikit-learn not available; metrics disabled.")
        return None

    import re

    def _safe_div(num, den, nan_val=float("nan")):
        return float(num / den) if den > 0 else nan_val

    def _safe_mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    def _safe_key(name: str) -> str:
        # "Pleural Other" -> "Pleural_Other", "Edema/CHF" -> "Edema_CHF"
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
        return s if s else "label"

    def _per_sample_ranking_metrics(y_true01, probs, known_mask):
        """
        y_true01, probs, known_mask: shape (N, L)
        Computes LRAP, coverage_error, ranking_loss with NaN-safe masking.
        Skips samples with no known labels or with no positives.
        """
        lrap_list, cov_list, rloss_list = [], [], []
        N, L = y_true01.shape

        for i in range(N):
            m = known_mask[i]
            if not m.any():
                continue
            t = y_true01[i, m]
            s = probs[i, m]

            pos_idx = np.where(t)[0]
            neg_idx = np.where(~t)[0]
            if pos_idx.size == 0:
                continue  # undefined for these ranking metrics

            # sort by score desc
            order = np.argsort(-s)
            t_sorted = t[order]

            # LRAP
            precisions = []
            cum_pos = 0
            for k in range(len(t_sorted)):
                if t_sorted[k]:
                    cum_pos += 1
                    precisions.append(cum_pos / float(k + 1))
            if precisions:
                lrap_list.append(float(np.mean(precisions)))

                # coverage error: max rank position of a positive (1-indexed)
                pos_positions = np.where(t_sorted)[0]
                cov_list.append(float(pos_positions.max() + 1))

            # ranking loss: fraction of (pos,neg) pairs misordered
            if neg_idx.size > 0:
                pos_scores = s[pos_idx][:, None]
                neg_scores = s[neg_idx][None, :]
                mis = (pos_scores <= neg_scores).sum()
                total_pairs = pos_idx.size * neg_idx.size
                rloss_list.append(float(mis / float(total_pairs)))

        return _safe_mean(lrap_list), _safe_mean(cov_list), _safe_mean(rloss_list)

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = 1.0 / (1.0 + np.exp(-logits))

        labels = labels.astype(np.float32)
        known_mask = np.isfinite(labels)

        # boolean ground truth (NaNs ignored by known_mask)
        y_true01 = (labels >= 0.5)
        pred01 = (probs >= 0.5)

        # -------------------------
        # Micro confusion over ALL known positions
        # -------------------------
        tp = int(((pred01 == 1) & (y_true01 == 1) & known_mask).sum())
        fp = int(((pred01 == 1) & (y_true01 == 0) & known_mask).sum())
        fn = int(((pred01 == 0) & (y_true01 == 1) & known_mask).sum())
        tn = int(((pred01 == 0) & (y_true01 == 0) & known_mask).sum())
        known = int(known_mask.sum())

        micro_acc = _safe_div(tp + tn, known)
        micro_precision = _safe_div(tp, tp + fp, nan_val=0.0)
        micro_recall = _safe_div(tp, tp + fn, nan_val=0.0)
        micro_specificity = _safe_div(tn, tn + fp, nan_val=0.0)
        micro_jaccard = _safe_div(tp, tp + fp + fn, nan_val=0.0)

        # micro f1 (flattened)
        if known > 0:
            y_true_flat = y_true01[known_mask].astype(np.int32)
            y_pred_flat = pred01[known_mask].astype(np.int32)
            micro_f1 = float(f1_score(y_true_flat, y_pred_flat, average="binary", zero_division=0))
        else:
            micro_f1 = float("nan")

        # micro auc/auprc (flattened known positions)
        if known > 0:
            y_true_flat_f = labels[known_mask]
            p_flat = probs[known_mask]
            if len(np.unique(y_true_flat_f)) >= 2:
                micro_auroc = float(roc_auc_score(y_true_flat_f, p_flat))
                micro_auprc = float(average_precision_score(y_true_flat_f, p_flat))
            else:
                micro_auroc = float("nan")
                micro_auprc = float("nan")
        else:
            micro_auroc = float("nan")
            micro_auprc = float("nan")

        # -------------------------
        # Per-label + Macro aggregates (NaN-safe)
        # -------------------------
        perlabel_metrics = {}  # will be merged into return dict

        macro_f1_list = []
        macro_auroc_list = []
        macro_auprc_list = []
        macro_prec_list = []
        macro_rec_list = []
        macro_spec_list = []
        macro_acc_list = []
        macro_balacc_list = []

        N, L = labels.shape
        for j in range(L):
            m = known_mask[:, j]
            support = int(m.sum())
            if support <= 0:
                continue

            name = class_names[j] if j < len(class_names) else f"label_{j}"
            key = _safe_key(name)

            yt = y_true01[m, j]
            yp = pred01[m, j]
            pj = probs[m, j]
            yj = labels[m, j]  # float 0/1

            tpj = int(((yp == 1) & (yt == 1)).sum())
            fpj = int(((yp == 1) & (yt == 0)).sum())
            fnj = int(((yp == 0) & (yt == 1)).sum())
            tnj = int(((yp == 0) & (yt == 0)).sum())
            denomj = tpj + fpj + fnj + tnj

            precj = _safe_div(tpj, tpj + fpj, nan_val=0.0)
            recj  = _safe_div(tpj, tpj + fnj, nan_val=0.0)
            specj = _safe_div(tnj, tnj + fpj, nan_val=0.0)
            accj  = _safe_div(tpj + tnj, denomj)
            f1j   = _safe_div(2 * tpj, (2 * tpj + fpj + fnj), nan_val=0.0)
            balj  = 0.5 * (recj + specj) if np.isfinite(recj) and np.isfinite(specj) else float("nan")

            # AUC/AUPRC only if both classes exist for this label on known positions
            if len(np.unique(yj)) >= 2:
                aucj = float(roc_auc_score(yj, pj))
                auprcj = float(average_precision_score(yj, pj))
            else:
                aucj = float("nan")
                auprcj = float("nan")

            # --- store per-label metrics
            perlabel_metrics[f"perlabel_support_{key}"] = float(support)
            perlabel_metrics[f"perlabel_pos_{key}"] = float(int((yt == 1).sum()))
            perlabel_metrics[f"perlabel_prev_{key}"] = _safe_div(int((yt == 1).sum()), support)
            perlabel_metrics[f"perlabel_predpos_{key}"] = float(int((yp == 1).sum()))
            perlabel_metrics[f"perlabel_predpos_rate_{key}"] = _safe_div(int((yp == 1).sum()), support)

            perlabel_metrics[f"perlabel_precision_{key}"] = precj
            perlabel_metrics[f"perlabel_recall_{key}"] = recj
            perlabel_metrics[f"perlabel_specificity_{key}"] = specj
            perlabel_metrics[f"perlabel_f1_{key}"] = f1j
            perlabel_metrics[f"perlabel_acc_{key}"] = accj
            perlabel_metrics[f"perlabel_balanced_acc_{key}"] = balj
            perlabel_metrics[f"perlabel_auroc_{key}"] = aucj
            perlabel_metrics[f"perlabel_auprc_{key}"] = auprcj

            # --- macro aggregates (NaN-safe lists)
            macro_prec_list.append(precj)
            macro_rec_list.append(recj)
            macro_spec_list.append(specj)
            macro_acc_list.append(accj)
            macro_balacc_list.append(balj)

            # F1 macro: only meaningful if both classes exist
            if len(np.unique(yt.astype(np.int32))) >= 2:
                macro_f1_list.append(float(f1_score(yt.astype(np.int32), yp.astype(np.int32), zero_division=0)))

            if np.isfinite(aucj):
                macro_auroc_list.append(aucj)
            if np.isfinite(auprcj):
                macro_auprc_list.append(auprcj)

        macro_f1 = _safe_mean(macro_f1_list)
        macro_auroc = _safe_mean(macro_auroc_list)
        macro_auprc = _safe_mean(macro_auprc_list)
        macro_precision = _safe_mean(macro_prec_list)
        macro_recall = _safe_mean(macro_rec_list)
        macro_specificity = _safe_mean(macro_spec_list)
        macro_acc = _safe_mean(macro_acc_list)
        macro_balanced_acc = _safe_mean(macro_balacc_list)

        # -------------------------
        # Hamming loss over known positions
        # -------------------------
        hamming_loss = _safe_div(fp + fn, known)

        # -------------------------
        # Subset accuracy (NaN-aware exact match)
        # -------------------------
        row_has_any_known = known_mask.any(axis=1)
        if int(row_has_any_known.sum()) > 0:
            per_row_exact = np.logical_or(pred01 == y_true01, ~known_mask).all(axis=1)
            subset_acc = float(per_row_exact[row_has_any_known].mean())
        else:
            subset_acc = float("nan")

        # -------------------------
        # Sample-wise metrics (NaN-safe)
        # -------------------------
        samp_prec, samp_rec, samp_f1, samp_jacc = [], [], [], []
        for i in range(N):
            m = known_mask[i]
            if not m.any():
                continue
            yt = y_true01[i, m]
            yp = pred01[i, m]

            tpi = int(((yp == 1) & (yt == 1)).sum())
            fpi = int(((yp == 1) & (yt == 0)).sum())
            fni = int(((yp == 0) & (yt == 1)).sum())

            pi = _safe_div(tpi, tpi + fpi, nan_val=0.0)
            ri = _safe_div(tpi, tpi + fni, nan_val=0.0)
            fi = _safe_div(2 * tpi, 2 * tpi + fpi + fni, nan_val=0.0)

            # Jaccard: if both sets empty, define as 1
            if (tpi + fpi + fni) == 0:
                ji = 1.0
            else:
                ji = float(tpi / float(tpi + fpi + fni))

            samp_prec.append(pi)
            samp_rec.append(ri)
            samp_f1.append(fi)
            samp_jacc.append(ji)

        samples_precision = _safe_mean(samp_prec)
        samples_recall = _safe_mean(samp_rec)
        samples_f1 = _safe_mean(samp_f1)
        samples_jaccard = _safe_mean(samp_jacc)

        # -------------------------
        # Ranking-based metrics (threshold-free), NaN-safe
        # -------------------------
        lrap, coverage_error, ranking_loss = _per_sample_ranking_metrics(y_true01, probs, known_mask)

        # -------------------------
        # Sanity stats
        # -------------------------
        true_pos_known = int((y_true01 & known_mask).sum())
        pred_pos_known = int((pred01 & known_mask).sum())
        pos_rate_true = _safe_div(true_pos_known, known)
        pos_rate_pred = _safe_div(pred_pos_known, known)

        return {
            # core (what you already had)
            "micro_acc": micro_acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "hamming_loss": hamming_loss,
            "subset_acc": subset_acc,

            # extra global metrics (useful)
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_specificity": micro_specificity,
            "micro_jaccard": micro_jaccard,
            "micro_auroc": micro_auroc,
            "micro_auprc": micro_auprc,

            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_specificity": macro_specificity,
            "macro_acc": macro_acc,
            "macro_balanced_acc": macro_balanced_acc,

            "samples_precision": samples_precision,
            "samples_recall": samples_recall,
            "samples_f1": samples_f1,
            "samples_jaccard": samples_jaccard,

            "lrap": lrap,
            "coverage_error": coverage_error,
            "ranking_loss": ranking_loss,

            "pos_rate_true": pos_rate_true,
            "pos_rate_pred": pos_rate_pred,

            # per-label metrics (many keys)
            **perlabel_metrics,
        }

    return compute_metrics