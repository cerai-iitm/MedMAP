#!/usr/bin/env bash
set -euo pipefail

# MedGemma on MIMIC-CXR with differential privacy.
#   Feature-DP  : --use_dp --feature_dp     (image public, text private)
#   Full DP-SGD : --use_dp --no-feature_dp
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

TARGET_EPSILON="${TARGET_EPSILON:-4.0}"
SEED="${SEED:-42}"
# ---------------------------------------------------------------------------

python -m "${TRAIN_MODULE}" \
  --output_root output_mimic_private \
  --top_level_buckets p10 p11 p12 \
  --best_img_index_root "${BEST_IMG_INDEX_ROOT}" \
  --chexpert_csv "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv" \
  --split_csv "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv" \
  --report_root "${REPORT_ROOT}" \
  --dataset_cache_root "${DATASET_CACHE_ROOT}" \
  --no-bf16 \
  --no-tf32 \
  --use_dp \
  --feature_dp \
  --target_epsilon "${TARGET_EPSILON}" \
  --clip_norm 1.0 \
  --public_clip_norm 50.0 \
  --include_img \
  --include_txt \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-3 \
  --num_train_epochs 3.0 \
  --seed "${SEED}" \
  --attempt 1

# For full DP-SGD replace `--feature_dp` with `--no-feature_dp`.
