# MedMAP: A Clinically Grounded Multi-Modal Benchmark for Modality-Asymmetric Privacy

<p align="justify">
Multi-modal data is increasingly used in high-stakes domains such as healthcare, where different modalities often exhibit unequal privacy sensitivity. For example, clinical text reports or patient attributes may be substantially more privacy-sensitive than imaging data, while also containing highly informative signals for downstream prediction tasks. We study this setting of modality-asymmetric privacy, where only a subset of modalities requires strong differential privacy (DP) guarantees. MedMAP provides a software package for benchmarking full DP and feature-level / modality-aware DP across image, text, and tabular/phenotypic data.
</p>

The current repository supports three experiment families:

1. **ImageNet + SmolVLM**
2. **MIMIC-CXR + MedGemma**
3. **ABIDE + linear/logistic classifier**

The repository is organized so that most experiments can be launched using the bash scripts under `examples/`. These scripts resolve the repository root automatically, check that the expected training entrypoint exists, and then launch the corresponding Python module.

---

## 1. Installation

Clone the repository:

```bash
git clone https://github.com/<anonymous-or-your-username>/MedMAP.git
cd MedMAP
```

Create and activate the environment:

```bash
conda create -n medmap python=3.10 -y
conda activate medmap
```

Install the package in editable mode:

```bash
pip install -e .
```

Install additional dependencies:

```bash
pip install -r requirements.txt
```

Optional but recommended: authenticate with Weights & Biases if you want experiment logging:

```bash
wandb login
```

---

## 2. Repository entrypoints

The provided bash scripts use the following Python modules:

| Experiment | Non-private script | Private script | Python module |
|---|---|---|---|
| ImageNet + SmolVLM | `examples/run_imagenet_non_private.sh` | `examples/run_imagenet_private.sh` | `scripts.train_imagenet` |
| MIMIC-CXR + MedGemma | `examples/run_mimic_non_private.sh` | `examples/run_mimic_private.sh` | `scripts.train_mimic` |
| ABIDE + linear model | `examples/run_abide_non_private.sh` | `examples/run_abide_private.sh` | `scripts.train_abide` |

Make the scripts executable once:

```bash
chmod +x examples/run_*.sh
```

Then launch experiments with:

```bash
bash examples/run_imagenet_non_private.sh
bash examples/run_imagenet_private.sh
bash examples/run_mimic_non_private.sh
bash examples/run_mimic_private.sh
bash examples/run_abide_non_private.sh
bash examples/run_abide_private.sh
```

---

## 3. General execution notes

This repository does **not** include dataset files. Users must download datasets separately and provide the corresponding local paths through either:

1. exported environment variables in the bash scripts, or
2. command-line arguments passed inside the bash scripts.

Large datasets such as ImageNet, MIMIC-CXR, MIMIC-CXR-JPG, and ABIDE should not be committed to the repository.

Recommended `.gitignore` entries:

```bash
data/
outputs/
checkpoints/
wandb/
*.zip
*.tar.gz
*.csv.gz
```

All scripts assume they are placed inside `examples/` and that the repository root contains both:

```bash
medmap/
scripts/
```

---

# 4. ImageNet + SmolVLM

## 4.1 Dataset setup

Download the ImageNet Object Localization Challenge dataset from Kaggle:

```text
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
```

Install and configure the Kaggle CLI:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Choose a local dataset directory:

```bash
export IMAGENET_DIR=/data/common/data/imagenet_object_localization_challenge
mkdir -p "${IMAGENET_DIR}"
```

Download and extract ImageNet:

```bash
kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p "${IMAGENET_DIR}"

unzip "${IMAGENET_DIR}/imagenet-object-localization-challenge.zip" \
  -d "${IMAGENET_DIR}"
```

Download ImageNet-Captions:

```bash
export IMAGENET_CAP_DIR=/data/common/data/imagenet_captions
mkdir -p "${IMAGENET_CAP_DIR}"

git clone https://github.com/mlfoundations/imagenet-captions.git "${IMAGENET_CAP_DIR}"

unzip "${IMAGENET_CAP_DIR}/imagenet_captions.zip" \
  -d "${IMAGENET_CAP_DIR}"
```

## 4.2 ImageNet variables used by the scripts

The ImageNet scripts use environment variables inside the bash files. Before running, check or edit these values in:

```bash
examples/run_imagenet_non_private.sh
examples/run_imagenet_private.sh
```

Important dataset variables:

```bash
export CAPTIONS_JSON="/data/common/data/imagenet_captions.json"
export ROOT_ZIP_PATH="/data/common/data/imagenet_object_localization_challenge"
export TRAIN_TXT="ILSVRC/ImageSets/CLS-LOC/train_cls.txt"
```

Meaning:

| Variable | Meaning |
|---|---|
| `CAPTIONS_JSON` | Path to the ImageNet-Captions JSON file. |
| `ROOT_ZIP_PATH` | Root directory containing the ImageNet Object Localization Challenge data. |
| `TRAIN_TXT` | Relative path to ImageNet train split file, usually `ILSVRC/ImageSets/CLS-LOC/train_cls.txt`. |
| `USE_SPLIT_IDX` | If `1`, loads predefined train/val/test indices from `medmap/data/split/imagenet/`. If `0`, creates a stratified split. |

## 4.3 Running ImageNet non-private experiments

Run:

```bash
bash examples/run_imagenet_non_private.sh
```

The non-private ImageNet script supports image-only, text-only, and image+text training by editing:

```bash
export INCLUDE_IMG=1
export INCLUDE_TEXT=1
```

Use the following settings:

| Setting | `INCLUDE_IMG` | `INCLUDE_TEXT` |
|---|---:|---:|
| Image + text | `1` | `1` |
| Image only | `1` | `0` |
| Text only | `0` | `1` |

For example, for image-only non-private training:

```bash
export INCLUDE_IMG=1
export INCLUDE_TEXT=0
```

For text-only non-private training:

```bash
export INCLUDE_IMG=0
export INCLUDE_TEXT=1
```

Do not set both to `0`, because the model would receive no input modality.

## 4.4 Running ImageNet private experiments

Run:

```bash
bash examples/run_imagenet_private.sh
```

In private ImageNet training, the Python code forces both modalities to be active:

```bash
INCLUDE_IMG=1
INCLUDE_TEXT=1
```

Therefore, modality ablations should be done using the **non-private** script. The private script is for comparing privacy mechanisms on the multimodal setting.

To choose between Full DP-SGD and Feature DP / Selective DP, edit:

```bash
export USE_DP=1
export SELECTIVE_DP=1
```

Use:

| Private method | `USE_DP` | `SELECTIVE_DP` |
|---|---:|---:|
| Full DP-SGD | `1` | `0` |
| Feature DP / Selective DP | `1` | `1` |

The private script automatically names outputs based on this flag:

```bash
if [[ "${SELECTIVE_DP}" == "1" ]]; then
    export OUT_DIR="outputs_smolvlm_featdp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
else
    export OUT_DIR="outputs_smolvlm_fulldp_eps_${TARGET_EPS}_clip_${CLIP_NORM}_lr_${LR}_seed_${SEED}_${STAMP}"
fi
```

Important DP variables:

```bash
export TARGET_EPS=1.0
export CLIP_NORM=1.0
```

Important training variables:

```bash
export BATCH=16
export GRAD_ACCUM=64
export LR=5e-4
```

Note: in the ImageNet DP training code, the effective per-device batch used by HuggingFace is computed as:

```text
PER_DEVICE_BATCH = BATCH * GRAD_ACCUM
```

Thus, for `BATCH=16` and `GRAD_ACCUM=64`, the per-device train batch becomes `1024`. Reduce these values if GPU memory is insufficient.

---

# 5. MIMIC-CXR + MedGemma

## 5.1 Dataset setup

MIMIC-CXR and MIMIC-CXR-JPG require PhysioNet credentialed access. Before downloading, make sure you have:

1. a PhysioNet account,
2. completed the required credentialing/training,
3. signed the relevant data use agreement.

MIMIC-CXR reports:

```text
https://physionet.org/content/mimic-cxr/2.1.0/
```

MIMIC-CXR-JPG:

```text
https://physionet.org/content/mimic-cxr-jpg/2.1.0/
```

A typical local layout is:

```bash
/data/common/data/mimic-cxr/2.1.0/files/
/data/common/data/mimic-cxr-jpg/2.1.0/files/
/data/common/data/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz
/data/common/data/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz
```

## 5.2 MIMIC variables / arguments used by the scripts

The MIMIC scripts currently pass dataset paths as **command-line arguments** inside the bash files rather than reading all of them from environment variables.

Before running, edit these files:

```bash
examples/run_mimic_non_private.sh
examples/run_mimic_private.sh
```

Replace the placeholder paths:

```bash
--chexpert_csv /path/to/file/mimic-cxr-2.0.0-chexpert.csv
--split_csv /path/to/file/mimic-cxr-2.0.0-split.csv
--report_root /path/to/downloaded_mimic/mimic_cxr/physionet.org/files/mimic-cxr/2.1.0/files/
```

Recommended local variables while editing:

```bash
export MIMIC_JPG_ROOT=/data/common/data/mimic-cxr-jpg/2.1.0
export MIMIC_REPORT_ROOT=/data/common/data/mimic-cxr/2.1.0/files
export CHEXPERT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz
export SPLIT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-split.csv.gz
```

Then use these paths in the script arguments:

```bash
--chexpert_csv "${CHEXPERT_CSV}" \
--split_csv "${SPLIT_CSV}" \
--report_root "${MIMIC_REPORT_ROOT}" \
```

The scripts also pass:

```bash
--top_level_buckets p10 p11 p12
```

This restricts training to the selected MIMIC top-level subject buckets. To run on different buckets, edit this list.

## 5.3 Running MIMIC non-private experiments

Run:

```bash
bash examples/run_mimic_non_private.sh
```

The non-private script uses:

```bash
--no-use_dp
--include_img
--include_txt
```

To activate or deactivate modalities, edit the modality flags in the bash command:

| Setting | Image flag | Text flag |
|---|---|---|
| Image + text | `--include_img` | `--include_txt` |
| Image only | `--include_img` | `--no-include_txt` |
| Text only | `--no-include_img` | `--include_txt` |

If your local argparse version does not expose the `--no-...` form, remove the flag and set the corresponding default in `scripts/train_mimic.py`.

## 5.4 Running MIMIC private experiments

Run:

```bash
bash examples/run_mimic_private.sh
```

The private script uses:

```bash
--use_dp
--selective_dp
--target_epsilon 4.0
--clip_norm 1.0
--public_clip_norm 50.0
```

To switch between private methods:

| Private method | Required flags |
|---|---|
| Full DP-SGD | `--use_dp --no-selective_dp` or remove `--selective_dp` if the default is false |
| Feature DP / Selective DP | `--use_dp --selective_dp` |

For Feature DP, `public_clip_norm` controls clipping for the public-gradient component. A large value such as `50.0` makes the public gradient nearly unclipped.

Important private hyperparameters:

```bash
--target_epsilon 4.0
--clip_norm 1.0
--public_clip_norm 50.0
--per_device_train_batch_size 2
--gradient_accumulation_steps 2
--learning_rate 2e-3
```

---

# 6. ABIDE + linear model

## 6.1 Dataset setup

The ABIDE code uses the ABIDE PCP data and precomputed fMRI connectivity features. A typical local layout is:

```bash
/storage/common/datasets/ABIDE_PCP_full
/storage/common/datasets/ABIDE_PCP_full/fc_features
medmap/abide_dump/recommended_feature_lists.json
```

The bash scripts expose these paths as environment variables:

```bash
export ABIDE_DATA_DIR="${ABIDE_DATA_DIR:-/storage/common/datasets/ABIDE_PCP_full}"
export ABIDE_FC_DIR="${ABIDE_FC_DIR:-/storage/common/datasets/ABIDE_PCP_full/fc_features}"
export FEATURE_JSON="${FEATURE_JSON:-medmap/abide_dump/recommended_feature_lists.json}"
export PHENOTYPIC_CSV="${PHENOTYPIC_CSV:-}"
```

Meaning:

| Variable | Meaning |
|---|---|
| `ABIDE_DATA_DIR` | Root/cache directory for ABIDE PCP data. |
| `ABIDE_FC_DIR` | Directory containing precomputed fMRI functional-connectivity features. |
| `FEATURE_JSON` | JSON file defining recommended sensitive/phenotypic features. |
| `PHENOTYPIC_CSV` | Optional explicit phenotypic CSV path. Leave empty to use the default loading path. |

The ABIDE scripts use the `${VAR:-default}` pattern, so you can override paths directly from the command line:

```bash
ABIDE_DATA_DIR=/my/abide/root \
ABIDE_FC_DIR=/my/abide/root/fc_features \
FEATURE_JSON=medmap/abide_dump/recommended_feature_lists.json \
bash examples/run_abide_non_private.sh
```

## 6.2 Running ABIDE non-private experiments

Run:

```bash
bash examples/run_abide_non_private.sh
```

The script defines a helper:

```bash
run_nonprivate "MODE_NAME" INCLUDE_FMRI INCLUDE_SENSITIVE
```

The default script runs three settings:

```bash
run_nonprivate "fmri_only" 1 0
run_nonprivate "sensitive_only" 0 1
run_nonprivate "fmri_sensitive" 1 1
```

To activate or deactivate modalities, edit these calls:

| Setting | `INCLUDE_FMRI` | `INCLUDE_SENSITIVE` |
|---|---:|---:|
| fMRI only | `1` | `0` |
| Sensitive/phenotypic only | `0` | `1` |
| fMRI + sensitive/phenotypic | `1` | `1` |

Do not set both to `0`; the Python code raises an error if both modalities are disabled.

Optional site feature:

```bash
export INCLUDE_SITE_ID_AS_FEATURE=0
```

Set it to `1` only if you intentionally want to include site ID as an input feature.

## 6.3 Running ABIDE private experiments

Run:

```bash
bash examples/run_abide_private.sh
```

In private mode, the Python code forces both ABIDE views to be active:

```bash
INCLUDE_FMRI=1
INCLUDE_SENSITIVE=1
```

Therefore, modality ablations should be done using the **non-private** ABIDE script.

The private script defines:

```bash
run_private "MODE_NAME" SELECTIVE_DP_VALUE
```

Use:

| Private method | Command |
|---|---|
| Full DP-SGD | `run_private "fulldp_fmri_sensitive" 0` |
| Feature DP / Selective DP | `run_private "featuredp_fmri_sensitive" 1` |

To run only Full DP-SGD, use:

```bash
run_private "fulldp_fmri_sensitive" 0
# run_private "featuredp_fmri_sensitive" 1
```

To run only Feature DP, use:

```bash
# run_private "fulldp_fmri_sensitive" 0
run_private "featuredp_fmri_sensitive" 1
```

To run both, uncomment both lines.

Important private variables:

```bash
export TARGET_EPS="${TARGET_EPS:-8.0}"
export CLIP_NORM="${CLIP_NORM:-1.0}"
export PUBLIC_CLIP="${PUBLIC_CLIP:-50.0}"
```

As in the ImageNet DP script, the ABIDE DP code multiplies:

```text
PER_DEVICE_BATCH = BATCH * GRAD_ACCUM
```

With the default ABIDE private settings, `BATCH=64` and `GRAD_ACCUM=1`, so the per-device batch remains `64`.

---

# 7. Privacy modes summary

MedMAP supports three broad training regimes:

| Regime | Description | Typical switch |
|---|---|---|
| Non-private | Standard training without DP noise. | `USE_DP=0` or `--no-use_dp` |
| Full DP-SGD | Clips/noises the full example-level gradient. | `USE_DP=1`, `SELECTIVE_DP=0` |
| Feature DP / Selective DP | Applies DP protection selectively to the private/sensitive modality component. | `USE_DP=1`, `SELECTIVE_DP=1` |

For ImageNet and ABIDE, the private bash scripts use environment variables. For MIMIC, the private bash script uses command-line flags.

---

# 8. Modality-control summary

## ImageNet

Use in `examples/run_imagenet_non_private.sh`:

```bash
export INCLUDE_IMG=1
export INCLUDE_TEXT=1
```

| Experiment | Setting |
|---|---|
| Image + text | `INCLUDE_IMG=1`, `INCLUDE_TEXT=1` |
| Image only | `INCLUDE_IMG=1`, `INCLUDE_TEXT=0` |
| Text only | `INCLUDE_IMG=0`, `INCLUDE_TEXT=1` |

In private ImageNet runs, both modalities are forced on.

## MIMIC-CXR

Use in `examples/run_mimic_non_private.sh` or `examples/run_mimic_private.sh`:

```bash
--include_img
--include_txt
```

| Experiment | Setting |
|---|---|
| Image + text | `--include_img --include_txt` |
| Image only | `--include_img --no-include_txt` |
| Text only | `--no-include_img --include_txt` |

## ABIDE

Use in `examples/run_abide_non_private.sh`:

```bash
run_nonprivate "fmri_only" 1 0
run_nonprivate "sensitive_only" 0 1
run_nonprivate "fmri_sensitive" 1 1
```

In private ABIDE runs, both fMRI and sensitive/phenotypic features are forced on.

---

# 9. Output directories

The scripts write outputs to timestamped directories, for example:

```bash
outputs_smolvlm_cls_img_txt_<timestamp>
outputs_smolvlm_featdp_eps_<eps>_clip_<clip>_lr_<lr>_seed_<seed>_<timestamp>
outputs_abide_linear_nonprivate/<mode>_seed_<seed>_<timestamp>
outputs_abide_linear_private/<mode>_eps_<eps>_clip_<clip>_lr_<lr>_seed_<seed>_<timestamp>
output_mimic_non_private
output_mimic_private
```

Each run may include model checkpoints, metrics, trainer state, and W&B logs depending on the training script and configuration.

---

# 10. Troubleshooting

## `Error: Could not locate MedMAP repository root.`

Run the bash scripts from anywhere, but keep them inside `examples/`. Each script expects the repository root to contain:

```bash
medmap/
scripts/<training_script>.py
```

## `ModuleNotFoundError: No module named medmap`

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

## CUDA out-of-memory

Reduce one or more of:

```bash
BATCH
GRAD_ACCUM
MAX_LENGTH
```

For ImageNet and ABIDE private runs, remember that the code multiplies `BATCH * GRAD_ACCUM` before passing it to `per_device_train_batch_size`.

## Gradient checkpointing in ImageNet

The ImageNet scripts currently keep:

```bash
export GRAD_CHECKPOINT=0
```

Keep this disabled unless the gradient-checkpointing block in `scripts/train_imagenet.py` is moved after model construction.

---

# 11. Notes on data access and redistribution

ImageNet must be downloaded through Kaggle according to Kaggle's terms.

MIMIC-CXR and MIMIC-CXR-JPG must be downloaded through PhysioNet after credentialed access approval. Do not redistribute MIMIC data, reports, labels, or images.

ABIDE can be cached locally through the code or preprocessed into the expected feature directories. Do not commit raw or processed large data files to the repository.

---

# 12. Citation

If you use this software package, please cite the corresponding paper:

```bibtex
@inproceedings{medmap2026,
  title     = {MedMAP: Benchmarking Differentially Private Multimodal Learning},
  author    = {Anonymous Authors},
  booktitle = {Preprint},
  year      = {2026}
}
```

Please also cite the original datasets and model checkpoints used in each experiment.