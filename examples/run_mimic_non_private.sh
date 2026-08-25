#!/usr/bin/env bash
set -euo pipefail

# MedGemma on MIMIC-CXR, non-private, both modalities.
# Edit the DATA_* variables to match your download locations
# (see docs/mimic_cxr.md for the preprocessing steps that produce them).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

TRAIN_MODULE="scripts.train_mimic"
if [[ ! -d "medmap" || ! -f "scripts/train_mimic.py" ]]; then
    echo "Error: Could not locate the MedMAP repository root (got: $(pwd))." && exit 1
fi

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="/path/to/mimic"                                                    # <- EDIT
JPG_ROOT="${DATA_ROOT}/physionet.org/files/mimic-cxr-jpg/2.1.0"
REPORT_ROOT="${DATA_ROOT}/physionet.org/files/mimic-cxr/2.1.0/files"
BEST_IMG_INDEX_ROOT="${DATA_ROOT}/best_img_index"                             # from scripts/select_mimic_images.py
DATASET_CACHE_ROOT="${DATA_ROOT}/medmap_dataset_cache"
# ---------------------------------------------------------------------------

python -m "${TRAIN_MODULE}" \
  --model_name google/medgemma-4b-pt \
  --output_root output_mimic_non_private \
  --top_level_buckets p10 p11 p12 \
  --best_img_index_root "${BEST_IMG_INDEX_ROOT}" \
  --chexpert_csv "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv" \
  --split_csv "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv" \
  --report_root "${REPORT_ROOT}" \
  --dataset_cache_root "${DATASET_CACHE_ROOT}" \
  --bf16 \
  --tf32 \
  --no-use_dp \
  --include_img \
  --include_txt \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 42 \
  --attempt 1

# Modality ablations: pass exactly one of --include_img / --include_txt.
# Text-only runs used --max_length 226; image+text runs used 426.
