#!/usr/bin/env bash
set -euo pipefail

# PAD-UFES-20: non-private baselines + DP-SGD + Feature DP-SGD sweep.
# The dataset downloads automatically on first run (--download; ~3.3 GB from
# Mendeley Data, CC BY 4.0). See docs/pad_ufes.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d "medmap" || ! -f "scripts/train_pad.py" ]]; then
    echo "Error: Could not locate the MedMAP repository root (got: $(pwd))." && exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${PAD_DATA_ROOT:-/path/to/PAD-UFES-20}"     # <- EDIT
LR="${LR:-1e-3}"
SEED="${SEED:-0}"
EPSILONS="${EPSILONS:-1 2 4 8 16 32 64}"

# 1) Non-private: image-only ("no meta") and image+metadata.
python -m scripts.train_pad --mode sgd --lr "${LR}" --seed "${SEED}" \
    --data_root "${DATA_ROOT}" --download
python -m scripts.train_pad --mode sgd --lr "${LR}" --seed "${SEED}" \
    --data_root "${DATA_ROOT}" --use_metadata

# 2) DP-SGD and Feature DP-SGD across the epsilon grid.
for EPS in ${EPSILONS}; do
    python -m scripts.train_pad --mode dpsgd --lr "${LR}" --seed "${SEED}" \
        --data_root "${DATA_ROOT}" --epsilon "${EPS}"
    python -m scripts.train_pad --mode feature_dpsgd --lr "${LR}" --seed "${SEED}" \
        --data_root "${DATA_ROOT}" --epsilon "${EPS}"
done

echo "All PAD-UFES-20 runs completed. Subgroup results: subgroup_results.csv"
