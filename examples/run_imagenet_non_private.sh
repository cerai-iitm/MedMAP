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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
STAMP="$(date -u +'%Y%m%d_%H%M%S')"
exec "${MEDMAP_PYTHON}" -m scripts.train_imagenet \
    --captions-json "${CAPTIONS_JSON}" \
    --root-zip-path "${ILSVRC_SOURCE}" \
    --tiny-imagenet-wnids "${WNIDS}" \
    --output-dir "${MEDMAP_OUTPUT_ROOT}/tiny-imagenet/nonprivate-${STAMP}" \
    --include-img --include-text --no-use-dp --use-fixed-split \
    --epochs 10 --batch-size 32 --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 --seed 100 --wandb-mode "${WANDB_MODE}" \
    "$@"
