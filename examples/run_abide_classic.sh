#!/usr/bin/env bash
set -euo pipefail

# ABIDE classic backend (sklearn / NumPy): the full fixed-split benchmark.
# Reproduces the reported runs:
#   1. non-private LBFGS baseline (creates + persists the fixed split, tunes C)
#   2. full-DP and Feature-DP sweeps over the epsilon grid
#   3. summary tables + privacy-utility plots
#
# Set ABIDE_DATA_DIR to your ABIDE PCP root; the FC features must exist
# (scripts/extract_abide_features.py). See docs/abide.md.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d "medmap" || ! -f "scripts/train_abide_classic.py" ]]; then
    echo "Error: Could not locate the MedMAP repository root (got: $(pwd))." && exit 1
fi

ABIDE_DATA_DIR="${ABIDE_DATA_DIR:-/path/to/abide/ABIDE_PCP_full}"   # <- EDIT
SPLIT="${SPLIT:-non_iid}"          # iid | non_iid
EPSILONS="${EPSILONS:-1 2 4 8 16 32 64}"

# 1) Non-private baseline: creates the fixed split + tunes the global C.
python -m scripts.train_abide_classic \
    --split "${SPLIT}" --method lbfgs --data-dir "${ABIDE_DATA_DIR}"

# 2) DP sweeps (one target epsilon per invocation, as in the original runs).
for EPS in ${EPSILONS}; do
    python -m scripts.train_abide_classic \
        --split "${SPLIT}" --method full_dp --data-dir "${ABIDE_DATA_DIR}" --epsilon "${EPS}"
    python -m scripts.train_abide_classic \
        --split "${SPLIT}" --method feat_dp --data-dir "${ABIDE_DATA_DIR}" --epsilon "${EPS}"
done

# 3) Summaries: cross-epsilon tables + privacy-utility plots.
python -m scripts.summarize_abide --data-dir "${ABIDE_DATA_DIR}" --split "${SPLIT}" --stages all

echo "ABIDE classic ${SPLIT} benchmark completed."
