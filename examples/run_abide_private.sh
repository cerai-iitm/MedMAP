#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ABIDE Linear HF Model: Private runs
#   1. Full DP-SGD
#   2. Feature-DP / Selective-DP
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

export TARGET_EPS="${TARGET_EPS:-8.0}"
export CLIP_NORM="${CLIP_NORM:-1.0}"

# Used only by Feature-DP / selective-DP trainer.
# Keep large if you want the public gradient to be almost unclipped.
export PUBLIC_CLIP="${PUBLIC_CLIP:-50.0}"

export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"

export INCLUDE_SITE_ID_AS_FEATURE="${INCLUDE_SITE_ID_AS_FEATURE:-false}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/storage/kaushik/outputs_abide_linear_private}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BASE_OUT_DIR}"

run_private() {
    local mode_name="$1"
    local selective_dp="$2"

    echo "============================================================"
    echo "Running private ABIDE setting: ${mode_name}"
    echo "USE_DP=true"
    echo "SELECTIVE_DP=${selective_dp}"
    echo "TARGET_EPS=${TARGET_EPS}"
    echo "CLIP_NORM=${CLIP_NORM}"
    echo "PUBLIC_CLIP=${PUBLIC_CLIP}"
    echo "============================================================"

    USE_DP=true \
    SELECTIVE_DP="${selective_dp}" \
    INCLUDE_FMRI=true \
    INCLUDE_SENSITIVE=true \
    OUT_DIR="${BASE_OUT_DIR}/${mode_name}_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}" \
    RUN_NAME="ABIDE-Linear-${mode_name}-eps-${TARGET_EPS}-clip-${CLIP_NORM}-lr-${LR}-seed-${SEED}-${STAMP}" \
    python -m "${TRAIN_MODULE}"
}

run_private "fulldp_fmri_sensitive" false
run_private "featuredp_fmri_sensitive" true

echo "All private ABIDE runs completed."