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

# Sanity check to make sure we are really in the MedMAP root
if [[ ! -d "medmap" || ! -f "scripts/train.py" ]]; then
    echo "Error: Could not locate MedMAP repository root."
    echo "Current directory: $(pwd)"
    echo "Expected to find: medmap/ and scripts/train.py"
    exit 1
fi

echo "Running from repository root: $(pwd)"
export CUDA_VISIBLE_DEVICES=2

python -m scripts.train_mimic \
  --output_root /storage/karthick/data/MedGemma \
  --top_level_buckets p10 p11 p12 \
  --xrv_root /data/kaushik/MedGemma/mimic/xrv_all_vs_mimic_ch_model_comparison \
  --chexpert_csv /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv \
  --split_csv /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv \
  --report_root /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr/2.1.0/files/ \
  --use_dp \
  --selective_dp \
  --target_epsilon 4.0 \
  --clip_norm 1.0 \
  --public_clip_norm 50.0 \
  --include_img \
  --include_txt \
  --per_device_train_batch_size 1024 \
  --gradient_accumulation_steps 512 \
  --learning_rate 2e-3 \
  --num_train_epochs 3.0