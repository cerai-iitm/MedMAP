#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"
medmap_create_workspace
cd "${MEDMAP_REPO_ROOT}"

read -r -a SHARDS <<< "${MIMIC_SHARDS:-p10 p11 p12}"
MIMIC_ROOT="${MEDMAP_RAW_ROOT}/mimic-cxr"
JPG_ROOT="${MIMIC_ROOT}/physionet.org/files/mimic-cxr-jpg/2.1.0"
RAW_REPORT_ROOT="${MIMIC_ROOT}/physionet.org/files/mimic-cxr/2.1.0/files"
INDEX_ROOT="${MEDMAP_PROCESSED_ROOT}/mimic-cxr/best-image-index"
PROCESSED_REPORT_ROOT="${MEDMAP_PROCESSED_ROOT}/mimic-cxr/reports-mmp"

medmap_require_dir "${JPG_ROOT}/files"
medmap_require_dir "${RAW_REPORT_ROOT}"
medmap_require_file "${JPG_ROOT}/mimic-cxr-2.0.0-metadata.csv"
medmap_require_file "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
medmap_require_file "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv"

printf '[1/2] Selecting one representative image per study...\n'
"${MEDMAP_PYTHON}" -m scripts.select_mimic_images "${SHARDS[@]}" \
    --dataset-root "${JPG_ROOT}" \
    --output-root "${INDEX_ROOT}"

report_args=(
    --shard "${SHARDS[@]}"
    --reports-root "${RAW_REPORT_ROOT}"
    --output-root "${PROCESSED_REPORT_ROOT}"
    --fail-fast
)
if [[ "${MIMIC_PREPARE_OVERWRITE:-0}" == "1" ]]; then
    report_args+=(--overwrite)
fi

printf '[2/2] Removing Findings/Impression and masking label mentions...\n'
"${MEDMAP_PYTHON}" -m scripts.extract_mimic_reports "${report_args[@]}"

printf 'MIMIC-CXR preprocessing complete.\n'
printf '  image index:      %s\n' "${INDEX_ROOT}"
printf '  processed reports: %s\n' "${PROCESSED_REPORT_ROOT}"
printf 'Next: bash examples/run_mimic_non_private.sh [training overrides]\n'
