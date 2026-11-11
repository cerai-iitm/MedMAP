#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"
medmap_create_workspace
cd "${MEDMAP_REPO_ROOT}"

RAW_ROOT="${MEDMAP_RAW_ROOT}/abide"
FEATURE_ROOT="${MEDMAP_PROCESSED_ROOT}/abide/fc-features"

cat <<EOF
ABIDE preparation downloads the C-PAC func_preproc derivative (~100+ GB),
the MSDL atlas, and then extracts 741 connectivity features per subject.
Raw root:       ${RAW_ROOT}
Feature output: ${FEATURE_ROOT}
EOF

exec "${MEDMAP_PYTHON}" -m scripts.extract_abide_features \
    --data-dir "${RAW_ROOT}" \
    --out-dir "${FEATURE_ROOT}" \
    "$@"
