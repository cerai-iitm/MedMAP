#!/usr/bin/env bash
set -euo pipefail

# SmolVLM on Tiny ImageNet (ImageNet-Captions) with differential privacy.
#   FEATURE_DP=0 -> full DP-SGD
#   FEATURE_DP=1 -> Feature-DP (image public, caption text private)
# Edit the dataset paths below (see docs/imagenet.md for downloads).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

TRAIN_MODULE="scripts.train_imagenet"
if [[ ! -d "medmap" || ! -f "scripts/train_imagenet.py" ]]; then
    echo "Error: Could not locate the MedMAP repository root (got: $(pwd))." && exit 1
fi

# ============================================================
# Hardware / runtime
# ============================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

# ============================================================
# Dataset paths (see docs/imagenet.md)
# ============================================================
export CAPTIONS_JSON="${CAPTIONS_JSON:-/path/to/imagenet_captions.json}"                     # <- EDIT
export ROOT_ZIP_PATH="${ROOT_ZIP_PATH:-/path/to/imagenet_object_localization_challenge}"     # <- EDIT
export TRAIN_TXT="ILSVRC/ImageSets/CLS-LOC/train_cls.txt"
export TINY_IMAGENET_WNIDS="${TINY_IMAGENET_WNIDS:-/path/to/tiny-imagenet-200/wnids.txt}"    # <- EDIT

# ============================================================
# Model
# ============================================================
export MODEL_NAME="HuggingFaceTB/SmolVLM-256M-Base"

# ============================================================
# Reproducibility
# ============================================================
export SEED="${SEED:-100}"
export USE_SPLIT_IDX="${USE_SPLIT_IDX:-0}"   # 1 = use the fixed packaged split indices

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

# W&B: set WANDB_MODE=disabled to run without Weights & Biases.
export WANDB_MODE="${WANDB_MODE:-online}"

# ============================================================
# Differential privacy settings
# ============================================================
export USE_DP=1
export FEATURE_DP="${FEATURE_DP:-1}"
export TARGET_EPS="${TARGET_EPS:-1.0}"
export CLIP_NORM="${CLIP_NORM:-1.0}"

# ============================================================
# LoRA settings
# ============================================================
export USE_LORA=1
export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05

# ============================================================
# Memory / checkpointing (the NameError from the original
# implementation is fixed; GRAD_CHECKPOINT=1 is safe to use)
# ============================================================
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-0}"

# ============================================================
# Output / W&B run naming
# ============================================================
STAMP="$(date +'%Y%m%d_%H%M%S')"

if [[ "${FEATURE_DP}" == "1" ]]; then
    export OUT_DIR="outputs_smolvlm_featdp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
    export RUN_NAME="SmolVLM-FeatDP-eps-${TARGET_EPS}-clip-${CLIP_NORM}-lr-${LR}-seed-${SEED}-${STAMP}"
else
    export OUT_DIR="outputs_smolvlm_fulldp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
    export RUN_NAME="SmolVLM-FullDP-eps-${TARGET_EPS}-clip-${CLIP_NORM}-lr-${LR}-seed-${SEED}-${STAMP}"
fi

python -m "${TRAIN_MODULE}"
