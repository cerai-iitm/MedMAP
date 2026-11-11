#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"

SKIP_ILSVRC=0
if [[ "${1:-}" == "--skip-ilsvrc" ]]; then
    SKIP_ILSVRC=1
    shift
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash scripts/download_tiny_imagenet.sh [--skip-ilsvrc]

Before downloading ILSVRC:
  1. Accept the ImageNet/Kaggle competition terms.
  2. Install the Kaggle CLI and configure ~/.kaggle/kaggle.json (mode 600).

--skip-ilsvrc downloads only ImageNet-Captions and tiny-imagenet-200. Use it
when the ILSVRC competition archive has already been placed in the expected
directory by another approved method.
EOF
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Error: unknown argument: %s\n' "$1" >&2
    exit 2
fi

medmap_require_command git
medmap_require_command curl
medmap_require_command unzip
medmap_create_workspace

ROOT="${MEDMAP_RAW_ROOT}/tiny-imagenet"
ILSVRC_ROOT="${ROOT}/ilsvrc"
CAPTIONS_PARENT="${ROOT}/captions"
CAPTIONS_REPO="${CAPTIONS_PARENT}/imagenet-captions"
TINY_ZIP="${ROOT}/tiny-imagenet-200.zip"
mkdir -p "${ILSVRC_ROOT}" "${CAPTIONS_PARENT}"

if [[ ! -d "${CAPTIONS_REPO}/.git" ]]; then
    git clone --depth 1 https://github.com/mlfoundations/imagenet-captions.git "${CAPTIONS_REPO}"
else
    printf 'ImageNet-Captions already exists: %s\n' "${CAPTIONS_REPO}"
fi
if [[ ! -f "${CAPTIONS_REPO}/imagenet_captions.json" ]]; then
    medmap_require_file "${CAPTIONS_REPO}/imagenet_captions.zip"
    unzip -q -o "${CAPTIONS_REPO}/imagenet_captions.zip" -d "${CAPTIONS_REPO}"
fi
medmap_require_file "${CAPTIONS_REPO}/imagenet_captions.json"

if [[ ! -f "${ROOT}/tiny-imagenet-200/wnids.txt" ]]; then
    curl --fail --location --continue-at - \
        --output "${TINY_ZIP}" \
        http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q -o "${TINY_ZIP}" -d "${ROOT}"
fi
medmap_require_file "${ROOT}/tiny-imagenet-200/wnids.txt"

if [[ "${SKIP_ILSVRC}" == "0" ]]; then
    if [[ ! -f "${ILSVRC_ROOT}/imagenet-object-localization-challenge.zip" \
          && ! -f "${ILSVRC_ROOT}/ILSVRC/ImageSets/CLS-LOC/train_cls.txt" ]]; then
        medmap_require_command kaggle
        cat <<'EOF'
Downloading ILSVRC through Kaggle. This succeeds only after competition terms
have been accepted and Kaggle credentials have been configured.
EOF
        kaggle competitions download \
            --competition imagenet-object-localization-challenge \
            --path "${ILSVRC_ROOT}"
    else
        printf 'ILSVRC source already exists under %s\n' "${ILSVRC_ROOT}"
    fi
fi

cat <<EOF
Tiny ImageNet sources are ready under ${ROOT}
  captions: ${CAPTIONS_REPO}/imagenet_captions.json
  wnids:    ${ROOT}/tiny-imagenet-200/wnids.txt
  ILSVRC:   ${ILSVRC_ROOT}/imagenet-object-localization-challenge.zip

Validate the complete join before training:
  bash examples/run_imagenet_non_private.sh --validate-only
EOF
