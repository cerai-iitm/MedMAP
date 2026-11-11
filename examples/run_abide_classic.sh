#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/../scripts/lib/workspace.sh"
cd "${MEDMAP_REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "${MEDMAP_PYTHON}" -m scripts.train_abide_classic --help
fi
medmap_create_workspace

RAW_ROOT="${MEDMAP_RAW_ROOT}/abide"
FC_ROOT="${MEDMAP_PROCESSED_ROOT}/abide/fc-features"
PHENOTYPE_CSV="${RAW_ROOT}/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
METHOD="${METHOD:-lbfgs}"
SPLIT="${SPLIT:-non_iid}"

medmap_require_file "${FC_ROOT}/X_conn.npy"
medmap_require_file "${FC_ROOT}/y.npy"
medmap_require_file "${PHENOTYPE_CSV}"

method_args=()
case "${METHOD}" in
    lbfgs) ;;
    full_dp|feat_dp) method_args+=(--epsilon "${TARGET_EPSILON:-8}") ;;
    *)
        printf 'Error: METHOD must be lbfgs, full_dp, or feat_dp.\n' >&2
        exit 2
        ;;
esac

exec "${MEDMAP_PYTHON}" -m scripts.train_abide_classic \
    --split "${SPLIT}" \
    --method "${METHOD}" \
    --data-dir "${RAW_ROOT}" \
    --fc-dir "${FC_ROOT}" \
    --phenotypic-csv "${PHENOTYPE_CSV}" \
    --output-root "${MEDMAP_OUTPUT_ROOT}/abide" \
    "${method_args[@]}" \
    "$@"
