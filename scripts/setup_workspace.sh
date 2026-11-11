#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"

medmap_create_workspace

cat <<EOF
MedMAP workspace is ready.

Raw datasets (do not modify): ${MEDMAP_RAW_ROOT}
Processed data:               ${MEDMAP_PROCESSED_ROOT}
Downloaded model caches:      ${MEDMAP_MODEL_CACHE}
Run outputs:                  ${MEDMAP_OUTPUT_ROOT}

Override these locations before running any helper if needed:
  MEDMAP_DATA_ROOT=/mnt/medmap/data
  MEDMAP_MODEL_CACHE=/mnt/medmap/model-cache
  MEDMAP_OUTPUT_ROOT=/mnt/medmap/outputs
EOF
