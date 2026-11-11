#!/usr/bin/env bash
# Shared repository-local storage contract for all MedMAP shell workflows.

MEDMAP_SCRIPT_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
MEDMAP_REPO_ROOT="$(cd -- "${MEDMAP_SCRIPT_LIB_DIR}/../.." >/dev/null 2>&1 && pwd -P)"

export MEDMAP_DATA_ROOT="${MEDMAP_DATA_ROOT:-${MEDMAP_REPO_ROOT}/data}"
export MEDMAP_RAW_ROOT="${MEDMAP_RAW_ROOT:-${MEDMAP_DATA_ROOT}/raw}"
export MEDMAP_PROCESSED_ROOT="${MEDMAP_PROCESSED_ROOT:-${MEDMAP_DATA_ROOT}/processed}"
export MEDMAP_MODEL_CACHE="${MEDMAP_MODEL_CACHE:-${MEDMAP_REPO_ROOT}/models/cache}"
export MEDMAP_OUTPUT_ROOT="${MEDMAP_OUTPUT_ROOT:-${MEDMAP_REPO_ROOT}/outputs}"
export MEDMAP_PYTHON="${MEDMAP_PYTHON:-python}"

# Keep downloaded model weights in the documented workspace unless users
# explicitly supply standard framework cache variables.
export HF_HOME="${HF_HOME:-${MEDMAP_MODEL_CACHE}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${MEDMAP_MODEL_CACHE}/torch}"

medmap_create_workspace() {
    mkdir -p \
        "${MEDMAP_RAW_ROOT}/mimic-cxr" \
        "${MEDMAP_RAW_ROOT}/abide" \
        "${MEDMAP_RAW_ROOT}/pad-ufes-20" \
        "${MEDMAP_RAW_ROOT}/tiny-imagenet" \
        "${MEDMAP_PROCESSED_ROOT}/mimic-cxr/best-image-index" \
        "${MEDMAP_PROCESSED_ROOT}/mimic-cxr/reports-mmp" \
        "${MEDMAP_PROCESSED_ROOT}/mimic-cxr/dataset-cache" \
        "${MEDMAP_PROCESSED_ROOT}/abide/fc-features" \
        "${MEDMAP_PROCESSED_ROOT}/pad-ufes-20" \
        "${MEDMAP_PROCESSED_ROOT}/tiny-imagenet" \
        "${MEDMAP_MODEL_CACHE}/huggingface" \
        "${MEDMAP_MODEL_CACHE}/torch" \
        "${MEDMAP_OUTPUT_ROOT}/mimic-cxr" \
        "${MEDMAP_OUTPUT_ROOT}/abide" \
        "${MEDMAP_OUTPUT_ROOT}/pad-ufes-20" \
        "${MEDMAP_OUTPUT_ROOT}/tiny-imagenet"
}

medmap_require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Error: required command not found: %s\n' "$1" >&2
        return 1
    fi
}

medmap_require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Error: required file not found: %s\n' "$1" >&2
        return 1
    fi
}

medmap_require_dir() {
    if [[ ! -d "$1" ]]; then
        printf 'Error: required directory not found: %s\n' "$1" >&2
        return 1
    fi
}
