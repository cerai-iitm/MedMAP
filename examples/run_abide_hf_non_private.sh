#!/usr/bin/env bash
set -Eeuo pipefail

# ABIDE linear model (HF backend), non-private: three modality ablations.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/../scripts/lib/workspace.sh"
medmap_create_workspace
cd "${MEDMAP_REPO_ROOT}"

TRAIN_MODULE="scripts.train_abide_hf"
if [[ ! -d "medmap" || ! -f "scripts/train_abide_hf.py" ]]; then
    echo "Error: Could not locate the MedMAP repository root (got: $(pwd))." && exit 1
fi

# ============================================================
# Hardware / runtime
# ============================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"

# ============================================================
# Data configuration
# ============================================================
export ABIDE_DATA_DIR="${ABIDE_DATA_DIR:-${MEDMAP_RAW_ROOT}/abide}"
export ABIDE_FC_DIR="${ABIDE_FC_DIR:-${MEDMAP_PROCESSED_ROOT}/abide/fc-features}"
export FEATURE_JSON="${FEATURE_JSON:-}"
export PHENOTYPIC_CSV="${PHENOTYPIC_CSV:-${ABIDE_DATA_DIR}/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv}"
medmap_require_file "${ABIDE_FC_DIR}/X_conn.npy"
medmap_require_file "${ABIDE_FC_DIR}/y.npy"
medmap_require_file "${PHENOTYPIC_CSV}"

# ============================================================
# Reproducibility
# ============================================================
export SEED="${SEED:-81}"

# ============================================================
# Training hyperparameters
# ============================================================
export EPOCHS="${EPOCHS:-100}"
export BATCH="${BATCH:-64}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-1e-3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"

# ============================================================
# Logging / evaluation / checkpointing
# ============================================================
export LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
export EVAL_STEPS="${EVAL_STEPS:-500}"

# W&B: set WANDB_MODE=disabled to run without Weights & Biases.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ============================================================
# Privacy configuration (non-private here; exported for explicitness)
# ============================================================
export USE_DP=0
export FEATURE_DP=0

# ============================================================
# Modality / feature configuration
# ============================================================
export INCLUDE_SITE_ID_AS_FEATURE="${INCLUDE_SITE_ID_AS_FEATURE:-0}"

# ============================================================
# Output configuration
# ============================================================
BASE_OUT_DIR="${BASE_OUT_DIR:-${MEDMAP_OUTPUT_ROOT}/abide/hf-nonprivate}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUT_DIR}"

run_nonprivate() {
    local MODE_NAME="$1"
    local INCLUDE_FMRI_VALUE="$2"
    local INCLUDE_SENSITIVE_VALUE="$3"

    echo "============================================================"
    echo "Running ABIDE non-private setting: ${MODE_NAME}"
    echo "INCLUDE_FMRI=${INCLUDE_FMRI_VALUE}  INCLUDE_SENSITIVE=${INCLUDE_SENSITIVE_VALUE}"
    echo "============================================================"

    export INCLUDE_FMRI="${INCLUDE_FMRI_VALUE}"
    export INCLUDE_SENSITIVE="${INCLUDE_SENSITIVE_VALUE}"
    export OUT_DIR="${BASE_OUT_DIR}/${MODE_NAME}_seed_${SEED}_${STAMP}"
    export RUN_NAME="ABIDE-Linear-NonPrivate-${MODE_NAME}-seed-${SEED}-${STAMP}"

    "${MEDMAP_PYTHON}" -m "${TRAIN_MODULE}"
}

# 1. Non-private fMRI only
run_nonprivate "fmri_only" 1 0

# 2. Non-private sensitive only
run_nonprivate "sensitive_only" 0 1

# 3. Non-private fMRI + sensitive
run_nonprivate "fmri_sensitive" 1 1

echo "All ABIDE non-private runs completed."
