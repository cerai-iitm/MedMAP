import os
import random
import numpy as np
import torch
from transformers import set_seed, Trainer
from datetime import datetime
import gc
import shutil
from typing import Any, Dict, List, Optional, Tuple

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))

def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return v.strip().lower() in ["1", "true", "yes", "y"]

def seed_everything(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_formatted_datetime_string(format_code="%Y-%m-%d_%H-%M-%S"):
    now = datetime.now()
    return now.strftime(format_code)

def _final_cleanup(trainer=None, model=None, backbone=None):
    # break reference cycles + free cuda cache
    try:
        if trainer is not None:
            try:
                trainer.accelerator.free_memory()
            except Exception:
                pass
    except Exception:
        pass

    for obj in [trainer, model, backbone]:
        try:
            del obj
        except Exception:
            pass

    try:
        gc.collect()
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # close wandb if present
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

def _is_main_process() -> bool:
    # Works for single-node DDP/accelerate too
    return int(os.environ.get("RANK", "0")) == 0

def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _remove_if_exists(path: str):
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception:
        pass

def _symlink_dir(src_dir: str, dst_link: str):
    """
    Create/replace a directory symlink dst_link -> src_dir.
    Falls back to writing a small pointer file if symlinks are not supported.
    """
    _remove_if_exists(dst_link)
    try:
        rel_src = os.path.relpath(src_dir, os.path.dirname(dst_link))
        os.symlink(rel_src, dst_link, target_is_directory=True)
    except Exception:
        # Fallback: pointer file
        try:
            with open(dst_link + ".txt", "w") as f:
                f.write(src_dir + "\n")
        except Exception:
            pass

def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""
        
def export_predictions_csv(
    trainer: Trainer,
    dataset,
    split_name: str,
    output_dir: str,
    class_names: List[str],
    threshold: float = 0.5,
    save_logits: bool = False,
    include_image_path: bool = False,
    include_text_len: bool = True,
    filename: Optional[str] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
):
    """
    Writes one row per example with:
      - metadata (split/epoch/global_step/ids)
      - y_true (0/1/NaN) per label
      - prob (sigmoid) per label
      - pred (thresholded) per label
      - optionally logits per label

    NOTE: This uses trainer.predict(dataset) (simple + robust).
    """
    import re

    _safe_makedirs(output_dir)

    # ---- build stable, unique label keys for column names ----
    def _safe_key(name: str) -> str:
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
        return s if s else "label"

    used = {}
    label_keys = []
    for name in class_names:
        k = _safe_key(name)
        if k in used:
            used[k] += 1
            k = f"{k}_{used[_safe_key(name)]}"
        else:
            used[k] = 0
        label_keys.append(k)

    out_path = os.path.join(
        output_dir,
        filename if filename is not None else f"predictions_{split_name}.csv",
    )

    # ---- run inference ----
    pred = trainer.predict(dataset)
    logits = pred.predictions
    labels = pred.label_ids

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.float32)

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred01 = (probs >= float(threshold)).astype(np.int32)
    known = np.isfinite(labels)

    # ---- optional token length (best-effort) ----
    tok = None
    try:
        # your collator has processor.tokenizer
        if hasattr(trainer, "data_collator") and hasattr(trainer.data_collator, "processor"):
            tok = getattr(trainer.data_collator.processor, "tokenizer", None)
    except Exception:
        tok = None

    rows = []
    N, L = labels.shape

    # detect available metadata columns
    col_names = set(getattr(dataset, "column_names", []))

    for i in range(N):
        ex = dataset[i]

        row = {
            "split": split_name,
            "epoch": (int(epoch) if epoch is not None else ""),
            "global_step": (int(global_step) if global_step is not None else ""),
            "row_idx": int(i),
            "threshold": float(threshold),
        }

        # IDs (if present)
        if "subject_id" in col_names:
            row["subject_id"] = ex.get("subject_id", "")
        if "study_id" in col_names:
            row["study_id"] = ex.get("study_id", "")
        if "dicom_id" in col_names:
            row["dicom_id"] = ex.get("dicom_id", "")

        # text lengths
        if include_text_len:
            txt = (ex.get("text", "") or "")
            row["text_len_chars"] = int(len(txt))
            if tok is not None:
                try:
                    row["text_len_tokens"] = int(len(tok(txt, add_special_tokens=True)["input_ids"]))
                except Exception:
                    row["text_len_tokens"] = ""
            else:
                row["text_len_tokens"] = ""

        # image path
        if include_image_path:
            img_field = ex.get("image", None)
            img_path = ""
            try:
                if isinstance(img_field, str):
                    img_path = img_field
                elif isinstance(img_field, dict):
                    img_path = img_field.get("path", "") or ""
            except Exception:
                pass
            row["image_path"] = img_path

        # row-level summary
        row_known = known[i]
        row["num_known_labels"] = int(row_known.sum())
        row["num_pos_true"] = int(((labels[i] >= 0.5) & row_known).sum())
        row["num_pos_pred"] = int((pred01[i] & row_known.astype(np.int32)).sum())

        # per-label columns
        for j in range(L):
            k = label_keys[j] if j < len(label_keys) else f"label_{j}"
            y = labels[i, j]
            is_known = bool(row_known[j])

            row[f"known__{k}"] = int(is_known)
            row[f"y_true__{k}"] = (float(y) if is_known else np.nan)
            row[f"prob__{k}"] = float(probs[i, j]) if is_known else float(probs[i, j])
            row[f"pred__{k}"] = int(pred01[i, j]) if is_known else int(pred01[i, j])

            if save_logits:
                row[f"logit__{k}"] = float(logits[i, j])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[pred_dump] wrote: {out_path} (N={len(df)})")
    