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
TRAIN_FILE="scripts/train_abide.py"
TRAIN_MODULE="scripts.train_abide"

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
# Common configuration
# ============================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export ABIDE_DATA_DIR="${ABIDE_DATA_DIR:-/storage/common/datasets/ABIDE_PCP_full}"
export ABIDE_FC_DIR="${ABIDE_FC_DIR:-/storage/common/datasets/ABIDE_PCP_full/fc_features}"
export FEATURE_JSON="${FEATURE_JSON:-/mnt/data/recommended_feature_lists.json}"

export SEED="${SEED:-81}"
export EPOCHS="${EPOCHS:-100}"
export BATCH="${BATCH:-64}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-1e-3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"

export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"

export INCLUDE_SITE_ID_AS_FEATURE="${INCLUDE_SITE_ID_AS_FEATURE:-0}"

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
    echo "INCLUDE_FMRI=${INCLUDE_FMRI_VALUE}"
    echo "INCLUDE_SENSITIVE=${INCLUDE_SENSITIVE_VALUE}"
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