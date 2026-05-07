#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ABIDE Linear HF Model: Non-private runs
# ============================================================

TRAIN_MODULE="${TRAIN_MODULE:-scripts.train_abide}"

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

export INCLUDE_SITE_ID_AS_FEATURE="${INCLUDE_SITE_ID_AS_FEATURE:-false}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/storage/kaushik/outputs_abide_linear_nonprivate}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BASE_OUT_DIR}"

run_nonprivate() {
    local mode_name="$1"
    local include_fmri="$2"
    local include_sensitive="$3"

    echo "============================================================"
    echo "Running non-private ABIDE setting: ${mode_name}"
    echo "INCLUDE_FMRI=${include_fmri}"
    echo "INCLUDE_SENSITIVE=${include_sensitive}"
    echo "============================================================"

    USE_DP=false \
    SELECTIVE_DP=false \
    INCLUDE_FMRI="${include_fmri}" \
    INCLUDE_SENSITIVE="${include_sensitive}" \
    OUT_DIR="${BASE_OUT_DIR}/${mode_name}_seed_${SEED}_${STAMP}" \
    RUN_NAME="ABIDE-Linear-NonPrivate-${mode_name}-seed-${SEED}-${STAMP}" \
    python -m "${TRAIN_MODULE}"
}

run_nonprivate "fmri_only" true false
run_nonprivate "sensitive_only" false true
run_nonprivate "fmri_sensitive" true true

echo "All non-private ABIDE runs completed."