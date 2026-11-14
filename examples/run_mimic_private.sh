#!/usr/bin/env bash
# Private MIMIC-CXR fine-tuning (Feature-DP by default, or full DP-SGD).
#
#   MIMIC_BACKBONE=medgemma|gemma      backbone preset (default: medgemma)
#   MIMIC_PRIVACY_MODE=feature_dp|full_dp  mechanism (default: feature_dp)
#   TARGET_EPSILON=<float>             privacy budget (default: 4)
#   SEED=<int>                         run seed (default: 42)
#   LEARNING_RATE=<float>              default: reported value for backbone+mode
#                                      (DP-SGD 5e-3, Feature-DP 2e-3 for both backbones)
#   PUBLIC_CLIP_NORM=<float>           Feature-DP public clip (default: preset,
#                                      50 for medgemma, 69 for gemma)
#   MIMIC_MICROBATCH / MIMIC_GRAD_ACCUM
#                                      physical microbatch x accumulation steps =
#                                      logical DP batch (default 2 x 512 = 1024,
#                                      the reported configuration; use e.g. 2 x 2
#                                      for a smoke test)
#
# delta follows the backbone preset (gemma: n^-1.1, medgemma: 1/n); override
# with `--delta_power`. Any additional arguments are forwarded to
# scripts/train_mimic.py and override the values below.
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

BACKBONE="${MIMIC_BACKBONE:-medgemma}"
case "${BACKBONE}" in
    medgemma) default_public_clip=50 ;;
    gemma)    default_public_clip=69 ;;
    *)
        printf 'Error: MIMIC_BACKBONE must be medgemma or gemma (got %s).\n' "${BACKBONE}" >&2
        exit 2
        ;;
esac

case "${MIMIC_PRIVACY_MODE:-feature_dp}" in
    feature_dp)
        privacy_args=(--feature_dp)
        default_lr=2e-3
        ;;
    full_dp)
        privacy_args=(--no-feature_dp)
        default_lr=5e-3
        ;;
    *)
        printf 'Error: MIMIC_PRIVACY_MODE must be feature_dp or full_dp.\n' >&2
        exit 2
        ;;
esac

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
    --backbone "${BACKBONE}" \
    --output_root "${MEDMAP_OUTPUT_ROOT}/mimic-cxr" \
    --top_level_buckets "${SHARDS[@]}" \
    --best_img_index_root "${INDEX_ROOT}" \
    --image_root "${JPG_ROOT}" \
    --chexpert_csv "${JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv" \
    --split_csv "${JPG_ROOT}/mimic-cxr-2.0.0-split.csv" \
    --report_root "${REPORT_ROOT}" \
    --dataset_cache_root "${CACHE_ROOT}" \
    --no-bf16 --no-tf32 --use_dp "${privacy_args[@]}" \
    --target_epsilon "${TARGET_EPSILON:-4}" \
    --clip_norm 1 \
    --public_clip_norm "${PUBLIC_CLIP_NORM:-${default_public_clip}}" \
    --include_img --include_txt \
    --per_device_train_batch_size "${MIMIC_MICROBATCH:-2}" \
    --gradient_accumulation_steps "${MIMIC_GRAD_ACCUM:-512}" \
    --learning_rate "${LEARNING_RATE:-${default_lr}}" \
    --num_train_epochs 3 \
    --eval_strategy epoch --save_strategy epoch \
    --seed "${SEED:-42}" \
    --attempt 1 \
    --wandb_mode disabled \
    "$@"
