#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/../scripts/lib/workspace.sh"
cd "${MEDMAP_REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "${MEDMAP_PYTHON}" -m scripts.train_imagenet --help
fi
medmap_create_workspace

RAW_ROOT="${MEDMAP_RAW_ROOT}/tiny-imagenet"
CAPTIONS_JSON="${RAW_ROOT}/captions/imagenet-captions/imagenet_captions.json"
WNIDS="${RAW_ROOT}/tiny-imagenet-200/wnids.txt"
ILSVRC_DIR="${RAW_ROOT}/ilsvrc"
ILSVRC_ZIP="${ILSVRC_DIR}/imagenet-object-localization-challenge.zip"
if [[ -f "${ILSVRC_DIR}/ILSVRC/ImageSets/CLS-LOC/train_cls.txt" ]]; then
    ILSVRC_SOURCE="${ILSVRC_DIR}"
else
    ILSVRC_SOURCE="${ILSVRC_ZIP}"
fi

medmap_require_file "${CAPTIONS_JSON}"
medmap_require_file "${WNIDS}"
if [[ ! -d "${ILSVRC_SOURCE}" && ! -f "${ILSVRC_SOURCE}" ]]; then
    printf 'Error: ILSVRC source not found under %s. Run scripts/download_tiny_imagenet.sh.\n' "${ILSVRC_DIR}" >&2
    exit 1
fi

case "${IMAGENET_PRIVACY_MODE:-feature_dp}" in
    feature_dp) privacy_args=(--feature-dp); setting="feature-dp" ;;
    full_dp) privacy_args=(--no-feature-dp); setting="full-dp" ;;
    *)
        printf 'Error: IMAGENET_PRIVACY_MODE must be feature_dp or full_dp.\n' >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
STAMP="$(date -u +'%Y%m%d_%H%M%S')"
exec "${MEDMAP_PYTHON}" -m scripts.train_imagenet \
    --captions-json "${CAPTIONS_JSON}" \
    --root-zip-path "${ILSVRC_SOURCE}" \
    --tiny-imagenet-wnids "${WNIDS}" \
    --output-dir "${MEDMAP_OUTPUT_ROOT}/tiny-imagenet/${setting}-${STAMP}" \
    --include-img --include-text --use-dp "${privacy_args[@]}" --use-fixed-split \
    --target-epsilon "${TARGET_EPSILON:-1}" --clip-norm 1 --public-clip-norm 50 \
    --epochs 10 --batch-size 16 --gradient-accumulation-steps 64 \
    --learning-rate 5e-4 --seed 100 --wandb-mode "${WANDB_MODE}" \
    "$@"
