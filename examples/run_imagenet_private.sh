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
TRAIN_FILE="scripts/train_imagenet.py"
TRAIN_MODULE="scripts.train_imagenet"

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

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

# ============================================================
# Dataset paths
# ============================================================

export CAPTIONS_JSON="/data/common/data/imagenet_captions.json"
export ROOT_ZIP_PATH="/data/common/data/imagenet_object_localization_challenge"
export TRAIN_TXT="ILSVRC/ImageSets/CLS-LOC/train_cls.txt"

# ============================================================
# Model
# ============================================================

export MODEL_NAME="HuggingFaceTB/SmolVLM-256M-Base"

# ============================================================
# Reproducibility
# ============================================================

export SEED=100
export USE_SPLIT_IDX=0

# ============================================================
# Modality setting
# Note: train_imagenet.py forces INCLUDE_IMG=True and
# INCLUDE_TEXT=True whenever USE_DP=1.
# ============================================================

export INCLUDE_IMG=1
export INCLUDE_TEXT=1

# ============================================================
# Training hyperparameters
# IMPORTANT:
# In DP mode, train_imagenet.py does:
# PER_DEVICE_BATCH = BATCH * GRAD_ACCUM
#
# With BATCH=16 and GRAD_ACCUM=64,
# per_device_train_batch_size becomes 1024.
# ============================================================

export EPOCHS=10
export BATCH=16
export GRAD_ACCUM=64
export LR=5e-4
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.03
export MAX_LENGTH=128

# ============================================================
# Logging / evaluation / checkpointing
# Keep SAVE_STRATEGY and EVAL_STRATEGY aligned because
# load_best_model_at_end=True is used in TrainingArguments.
# ============================================================

export LOGGING_STRATEGY="steps"
export LOGGING_STEPS=1

export SAVE_STRATEGY="steps"
export SAVE_STEPS=700

export EVAL_STRATEGY="steps"
export EVAL_STEPS=700

# ============================================================
# Differential privacy settings
# USE_DP=1          -> private training
# SELECTIVE_DP=0    -> full DP-SGD
# SELECTIVE_DP=1    -> Feature DP / Selective DP
# ============================================================

export USE_DP=1
export SELECTIVE_DP=0
export TARGET_EPS=1.0
export CLIP_NORM=1.0

# ============================================================
# LoRA settings
# ============================================================

export USE_LORA=1
export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05

# ============================================================
# Gradient checkpointing
# IMPORTANT:
# Keep this 0 unless you fix train_imagenet.py.
# Currently the code calls model.gradient_checkpointing_enable()
# before model = setup_model(...), so GRAD_CHECKPOINT=1 will fail.
# ============================================================

export GRAD_CHECKPOINT=0

# ============================================================
# Output / W&B run naming
# Optional, but explicit names are better for reproducibility.
# ============================================================

STAMP="$(date +'%Y%m%d_%H%M%S')"

if [[ "${SELECTIVE_DP}" == "1" ]]; then
    export OUT_DIR="/storage/kaushik/outputs_smolvlm_featdp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
    export RUN_NAME="SmolVLM-FeatDP-eps-${TARGET_EPS}-clip-${CLIP_NORM}-lr-${LR}-seed-${SEED}-${STAMP}"
else
    export OUT_DIR="/storage/kaushik/outputs_smolvlm_fulldp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
    export RUN_NAME="SmolVLM-FullDP-eps-${TARGET_EPS}-clip-${CLIP_NORM}-lr-${LR}-seed-${SEED}-${STAMP}"
fi

# ============================================================
# Launch
# ============================================================

python -m "${TRAIN_MODULE}"