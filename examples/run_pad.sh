#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/../scripts/lib/workspace.sh"
cd "${MEDMAP_REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "${MEDMAP_PYTHON}" -m scripts.train_pad --help
fi
medmap_create_workspace

RAW_ROOT="${MEDMAP_RAW_ROOT}/pad-ufes-20"
CACHE_PATH="${MEDMAP_PROCESSED_ROOT}/pad-ufes-20/efficientnet-b3-features.npz"
RESULT_ROOT="${MEDMAP_OUTPUT_ROOT}/pad-ufes-20"
MODE="${MODE:-sgd}"

medmap_require_file "${RAW_ROOT}/metadata.csv"
medmap_require_dir "${RAW_ROOT}/all_images"
medmap_require_file "${CACHE_PATH}"

mode_args=()
case "${MODE}" in
    sgd) ;;
    dpsgd|feature_dpsgd) mode_args+=(--epsilon "${TARGET_EPSILON:-8}") ;;
    *)
        printf 'Error: MODE must be sgd, dpsgd, or feature_dpsgd.\n' >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${MEDMAP_PYTHON}" -m scripts.train_pad \
    --mode "${MODE}" \
    --lr "${LEARNING_RATE:-1e-3}" \
    --data_root "${RAW_ROOT}" \
    --img_dir "${RAW_ROOT}/all_images" \
    --meta_path "${RAW_ROOT}/metadata.csv" \
    --cache_path "${CACHE_PATH}" \
    --subgroup_csv "${RESULT_ROOT}/subgroup-results.csv" \
    --seed "${SEED:-0}" \
    "${mode_args[@]}" \
    "$@"
