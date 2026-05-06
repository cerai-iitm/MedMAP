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

export CUDA_VISIBLE_DEVICES=0
export USE_DP=1
export SELECTIVE_DP=0  # Set to 1 for Feature DP
export TARGET_EPS=1.0
export CLIP_NORM=1.0
export BATCH=16
export GRAD_ACCUM=64
export LR=5e-4

python -m scripts.train_imagenet