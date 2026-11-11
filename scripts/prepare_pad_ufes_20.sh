#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"
medmap_create_workspace
cd "${MEDMAP_REPO_ROOT}"

exec "${MEDMAP_PYTHON}" -m scripts.prepare_pad \
    --data-root "${MEDMAP_RAW_ROOT}/pad-ufes-20" \
    --cache-path "${MEDMAP_PROCESSED_ROOT}/pad-ufes-20/efficientnet-b3-features.npz" \
    --download \
    "$@"
