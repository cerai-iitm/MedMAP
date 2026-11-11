#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# shellcheck source=scripts/lib/workspace.sh
source "${SCRIPT_DIR}/../scripts/lib/workspace.sh"
cd "${MEDMAP_REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "${MEDMAP_PYTHON}" -m scripts.train_mimic --help
fi
medmap_create_workspace

read -r -a SHARDS <<< "${MIMIC_SHARDS:-p10 p11 p12}"
for shard in "${SHARDS[@]}"; do
    if [[ ! "${shard}" =~ ^p1[0-9]$ ]]; then
        printf 'Error: invalid MIMIC shard: %s (expected p10..p19)\n' "${shard}" >&2
        exit 2
    fi
done

MIMIC_RAW="${MEDMAP_RAW_ROOT}/mimic-cxr"
JPG_ROOT="${MIMIC_RAW}/physionet.org/files/mimic-cxr-jpg/2.1.0"
INDEX_ROOT="${MEDMAP_PROCESSED_ROOT}/mimic-cxr/best-image-index"
REPORT_ROOT="${MEDMAP_PROCESSED_ROOT}/mimic-cxr/reports-mmp"
CACHE_ROOT="${MEDMAP_PROCESSED_ROOT}/mimic-cxr/dataset-cache"

medmap_require_dir "${JPG_ROOT}/files"
medmap_require_dir "${INDEX_ROOT}"
medmap_require_dir "${REPORT_ROOT}"
medmap_require_file "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
medmap_require_file "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${MEDMAP_PYTHON}" -m scripts.train_mimic \
    --model_name google/medgemma-4b-pt \
    --output_root "${MEDMAP_OUTPUT_ROOT}/mimic-cxr" \
    --top_level_buckets "${SHARDS[@]}" \
    --best_img_index_root "${INDEX_ROOT}" \
    --image_root "${JPG_ROOT}" \
    --chexpert_csv "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv" \
    --split_csv "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv" \
    --report_root "${REPORT_ROOT}" \
    --dataset_cache_root "${CACHE_ROOT}" \
    --bf16 --tf32 --no-use_dp --include_img --include_txt \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --seed 42 \
    --attempt 1 \
    --wandb_mode disabled \
    "$@"
