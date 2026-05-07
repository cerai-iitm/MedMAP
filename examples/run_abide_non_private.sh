#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script is located:
# MedMAP/examples/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Repository root is one level above examples/:
# MedMAP/
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# Always run from the MedMAP repository root
cd "${REPO_ROOT}"

# Training entrypoint for this script
# Use these two lines if the file is scripts/train_abide.py
TRAIN_FILE="scripts/train_abide.py"
TRAIN_MODULE="scripts.train_abide"

# Use these instead if the file is actually scripts/train_abide_linear.py
# TRAIN_FILE="scripts/train_abide_linear.py"
# TRAIN_MODULE="scripts.train_abide_linear"

# Sanity check to make sure we are really in the MedMAP root
if [[ ! -d "medmap" || ! -f "${TRAIN_FILE}" ]]; then
    echo "Error: Could not locate MedMAP repository root."
    echo "Current directory: $(pwd)"
    echo "Expected to find: medmap/ and ${TRAIN_FILE}"
    exit 1
fi

echo "Running from repository root: $(pwd)"
echo "Using training entrypoint: ${TRAIN_MODULE}"

# ============================================================
# Hardware / runtime
# ============================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"

# ============================================================
# Data configuration
# ============================================================

export ABIDE_DATA_DIR="${ABIDE_DATA_DIR:-/storage/common/datasets/ABIDE_PCP_full}"
export ABIDE_FC_DIR="${ABIDE_FC_DIR:-/storage/common/datasets/ABIDE_PCP_full/fc_features}"
export FEATURE_JSON="${FEATURE_JSON:-medmap/abide_dump/recommended_feature_lists.json}"

# Optional. Leave empty to let build_abide_linear_hf_datasets fetch/load phenotypics normally.
export PHENOTYPIC_CSV="${PHENOTYPIC_CSV:-}"

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

# ============================================================
# Privacy configuration
# For this script USE_DP=0, but these are still exported so
# all env-controlled Python arguments are explicit.
# ============================================================

export USE_DP="${USE_DP:-0}"
export SELECTIVE_DP="${SELECTIVE_DP:-0}"
export TARGET_EPS="${TARGET_EPS:-8.0}"
export CLIP_NORM="${CLIP_NORM:-1.0}"
export PUBLIC_CLIP="${PUBLIC_CLIP:-50.0}"

# ============================================================
# Modality / feature configuration
# ============================================================

export INCLUDE_SITE_ID_AS_FEATURE="${INCLUDE_SITE_ID_AS_FEATURE:-0}"

# ============================================================
# Output configuration
# ============================================================

BASE_OUT_DIR="${BASE_OUT_DIR:-/storage/kaushik/outputs_abide_linear_nonprivate}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BASE_OUT_DIR}"

# ============================================================
# Helper function
# ============================================================

run_nonprivate() {
    local MODE_NAME="$1"
    local INCLUDE_FMRI_VALUE="$2"
    local INCLUDE_SENSITIVE_VALUE="$3"

    echo "============================================================"
    echo "Running ABIDE non-private setting: ${MODE_NAME}"
    echo "USE_DP=0"
    echo "SELECTIVE_DP=0"
    echo "INCLUDE_FMRI=${INCLUDE_FMRI_VALUE}"
    echo "INCLUDE_SENSITIVE=${INCLUDE_SENSITIVE_VALUE}"
    echo "INCLUDE_SITE_ID_AS_FEATURE=${INCLUDE_SITE_ID_AS_FEATURE}"
    echo "OUT_DIR=${BASE_OUT_DIR}/${MODE_NAME}_seed_${SEED}_${STAMP}"
    echo "============================================================"

    export USE_DP=0
    export SELECTIVE_DP=0

    export INCLUDE_FMRI="${INCLUDE_FMRI_VALUE}"
    export INCLUDE_SENSITIVE="${INCLUDE_SENSITIVE_VALUE}"

    export OUT_DIR="${BASE_OUT_DIR}/${MODE_NAME}_seed_${SEED}_${STAMP}"
    export RUN_NAME="ABIDE-Linear-NonPrivate-${MODE_NAME}-seed-${SEED}-${STAMP}"

    python -m "${TRAIN_MODULE}"
}

# ============================================================
# Runs
# ============================================================

# 1. Non-private fMRI only
run_nonprivate "fmri_only" 1 0

# 2. Non-private sensitive only
run_nonprivate "sensitive_only" 0 1

# 3. Non-private fMRI + sensitive
run_nonprivate "fmri_sensitive" 1 1

echo "All ABIDE non-private runs completed."