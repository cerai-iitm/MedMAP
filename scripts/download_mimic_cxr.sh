#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/lib/workspace.sh"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: PHYSIONET_USER=<username> bash scripts/download_mimic_cxr.sh

Prerequisites:
  1. Complete PhysioNet credentialing/CITI training.
  2. Sign the MIMIC-CXR and MIMIC-CXR-JPG data use agreements.
  3. Set PHYSIONET_USER. wget asks for the password interactively and the
     password is never stored by this script.

Optional: MIMIC_SHARDS="p10 p11 p12" (these are the benchmark defaults).
EOF
    exit 0
fi

: "${PHYSIONET_USER:?Set PHYSIONET_USER after completing PhysioNet access requirements.}"
medmap_require_command wget
medmap_require_command gzip
medmap_create_workspace

read -r -a SHARDS <<< "${MIMIC_SHARDS:-p10 p11 p12}"
for shard in "${SHARDS[@]}"; do
    if [[ ! "${shard}" =~ ^p1[0-9]$ ]]; then
        printf 'Error: invalid MIMIC shard: %s (expected p10..p19)\n' "${shard}" >&2
        exit 2
    fi
done

MIMIC_ROOT="${MEDMAP_RAW_ROOT}/mimic-cxr"
JPG_ROOT="${MIMIC_ROOT}/physionet.org/files/mimic-cxr-jpg/2.1.0"
mkdir -p "${JPG_ROOT}"

report_urls=()
image_urls=()
for shard in "${SHARDS[@]}"; do
    report_urls+=("https://physionet.org/files/mimic-cxr/2.1.0/files/${shard}/")
    image_urls+=("https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/${shard}/")
done

printf 'Downloading restricted reports. wget will request your PhysioNet password.\n'
wget --recursive --no-parent --continue --timestamping \
    --user "${PHYSIONET_USER}" --ask-password \
    --directory-prefix "${MIMIC_ROOT}" "${report_urls[@]}"

printf 'Downloading restricted JPG shards. wget will request your PhysioNet password.\n'
wget --recursive --no-parent --continue --timestamping \
    --user "${PHYSIONET_USER}" --ask-password \
    --directory-prefix "${MIMIC_ROOT}" "${image_urls[@]}"

printf 'Downloading metadata, CheXpert labels, and official split CSVs.\n'
for filename in \
    mimic-cxr-2.0.0-metadata.csv.gz \
    mimic-cxr-2.0.0-chexpert.csv.gz \
    mimic-cxr-2.0.0-split.csv.gz; do
    wget --continue --timestamping --user "${PHYSIONET_USER}" --ask-password \
        --directory-prefix "${JPG_ROOT}" \
        "https://physionet.org/files/mimic-cxr-jpg/2.1.0/${filename}"
    if [[ ! -f "${JPG_ROOT}/${filename%.gz}" ]]; then
        gzip --decompress --keep "${JPG_ROOT}/${filename}"
    fi
done

printf 'MIMIC-CXR download complete under %s\n' "${MIMIC_ROOT}"
printf 'Next: bash scripts/prepare_mimic_cxr.sh\n'
