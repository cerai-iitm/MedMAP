# MedMAP: A Clinically Grounded Multi-Modal Benchmark for Modality-Asymmetric Privacy


# MedMAP

MedMAP is a software package for benchmarking differentially private multimodal learning across image, text, and tabular/phenotypic data.

The current repository supports experiments on:

1. ImageNet + SmolVLM
2. MIMIC-CXR + MedGemma
3. ABIDE + logistic regression

---

## 1. Installation

Clone the repository:

```bash
git clone https://github.com/<anonymous-or-your-username>/MedMAP.git
cd MedMAP
````

Create and activate the environment:

```bash
conda create -n medmap python=3.10 -y
conda activate medmap
```

Install the package in editable mode:

```bash
pip install -e .
```

Install additional experiment dependencies if needed:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Setup

This repository does **not** include any dataset files.
Users must download the datasets separately and provide the corresponding paths through environment variables or command-line arguments.

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

---

# 3. ImageNet + ImageNet-Captions Setup

## 3.1 Download ImageNet

Download the ImageNet Object Localization Challenge dataset from Kaggle:

```bash
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
```

First, install and configure the Kaggle CLI:

```bash
pip install kaggle
```

Place your Kaggle API token at:

```bash
~/.kaggle/kaggle.json
```

Then set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Choose a directory where ImageNet should be downloaded:

```bash
export IMAGENET_DIR=/data/common/data/imagenet_object_localization_challenge
mkdir -p ${IMAGENET_DIR}
```

Download the dataset:

```bash
kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p ${IMAGENET_DIR}
```

Unzip the downloaded archive:

```bash
unzip ${IMAGENET_DIR}/imagenet-object-localization-challenge.zip \
  -d ${IMAGENET_DIR}
```

After extraction, `IMAGENET_DIR` should point to the root directory containing the ImageNet files.

For this project, set:

```bash
export ROOT_ZIP_PATH=${IMAGENET_DIR}
```

Example:

```bash
export ROOT_ZIP_PATH=/data/common/data/imagenet_object_localization_challenge
```

---

## 3.2 Download ImageNet-Captions

Download ImageNet-Captions from:

```bash
https://github.com/mlfoundations/imagenet-captions
```

Set the captions directory:

```bash
export IMAGENET_CAP_DIR=/data/common/data/imagenet_captions
mkdir -p ${IMAGENET_CAP_DIR}
```

Clone the repository:

```bash
git clone https://github.com/mlfoundations/imagenet-captions.git ${IMAGENET_CAP_DIR}
```

Unzip the captions file:

```bash
unzip ${IMAGENET_CAP_DIR}/imagenet_captions.zip \
  -d ${IMAGENET_CAP_DIR}
```

Set the captions JSON path:

```bash
export CAPTIONS_JSON=${IMAGENET_CAP_DIR}/imagenet_captions.json
```

Example:

```bash
export CAPTIONS_JSON=/data/common/data/imagenet_captions/imagenet_captions.json
```

The ImageNet-Captions JSON contains image-level metadata such as filename, WordNet ID, title, description, and tags.

---

## 3.3 ImageNet Environment Variables

Before running ImageNet experiments, export:

```bash
export ROOT_ZIP_PATH=/data/common/data/imagenet_object_localization_challenge
export CAPTIONS_JSON=/data/common/data/imagenet_captions/imagenet_captions.json
```

Example training command:

```bash
python scripts/train_imagenet_smolvlm.py \
  --root-zip-path ${ROOT_ZIP_PATH} \
  --captions-json ${CAPTIONS_JSON} \
  --include-img \
  --include-text \
  --output-dir outputs/imagenet_smolvlm
```

For image-only experiments:

```bash
python scripts/train_imagenet_smolvlm.py \
  --root-zip-path ${ROOT_ZIP_PATH} \
  --captions-json ${CAPTIONS_JSON} \
  --include-img \
  --no-include-text \
  --output-dir outputs/imagenet_smolvlm_image_only
```

For text-only experiments:

```bash
python scripts/train_imagenet_smolvlm.py \
  --root-zip-path ${ROOT_ZIP_PATH} \
  --captions-json ${CAPTIONS_JSON} \
  --no-include-img \
  --include-text \
  --output-dir outputs/imagenet_smolvlm_text_only
```

---

# 4. MIMIC-CXR + MIMIC-CXR-JPG Setup

MIMIC-CXR and MIMIC-CXR-JPG require PhysioNet credentialed access.
Before downloading, make sure you have:

1. A PhysioNet account
2. Completed the required credentialing/training
3. Signed the relevant data use agreement

---

## 4.1 Download MIMIC-CXR Free-Text Reports

The free-text radiology reports are available from:

```bash
https://physionet.org/content/mimic-cxr/2.1.0/
```

Download the reports archive:

```bash
mimic-cxr-reports.zip
```

Set the reports directory:

```bash
export MIMIC_CXR_DIR=/data/common/data/mimic-cxr/2.1.0
mkdir -p ${MIMIC_CXR_DIR}
```

Download using PhysioNet credentials:

```bash
wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_CXR_DIR} \
  https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
```

Unzip the reports:

```bash
unzip ${MIMIC_CXR_DIR}/mimic-cxr-reports.zip \
  -d ${MIMIC_CXR_DIR}
```

Set the report root:

```bash
export REPORT_ROOT=${MIMIC_CXR_DIR}/files
```

Example:

```bash
export REPORT_ROOT=/data/common/data/mimic-cxr/2.1.0/files
```

`REPORT_ROOT` should point to the directory containing the extracted report folders such as `p10/`, `p11/`, etc.

---

## 4.2 Download MIMIC-CXR-JPG

MIMIC-CXR-JPG is available from:

```bash
https://physionet.org/content/mimic-cxr-jpg/2.1.0/
```

Set the MIMIC-CXR-JPG root directory:

```bash
export MIMIC_JPG_ROOT=/data/common/data/mimic-cxr-jpg/2.1.0
mkdir -p ${MIMIC_JPG_ROOT}
```

It is recommended to download the dataset into:

```bash
/data/common/data/mimic-cxr-jpg/2.1.0
```

Download useful metadata files:

```bash
wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_JPG_ROOT} \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz

wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_JPG_ROOT} \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz

wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_JPG_ROOT} \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz

wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_JPG_ROOT} \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-negbio.csv.gz

wget -N -c \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -P ${MIMIC_JPG_ROOT} \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES
```

Download all JPG images listed in `IMAGE_FILENAMES`:

```bash
cd /data/common/data

wget -r -N -c -np -nH --cut-dirs=1 \
  --user YOUR_PHYSIONET_USERNAME \
  --ask-password \
  -i ${MIMIC_JPG_ROOT}/IMAGE_FILENAMES \
  --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

This will download the image files under:

```bash
/data/common/data/mimic-cxr-jpg/2.1.0/files/
```

---

## 4.3 MIMIC Environment Variables

Set the following paths before running MIMIC-CXR experiments:

```bash
export MIMIC_JPG_ROOT=/data/common/data/mimic-cxr-jpg/2.1.0
export REPORT_ROOT=/data/common/data/mimic-cxr/2.1.0/files

export CHEXPERT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz
export METADATA_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz
export SPLIT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-split.csv.gz
```

Example training command:

```bash
python scripts/train_mimic_medgemma.py \
  --image-root ${MIMIC_JPG_ROOT}/files \
  --report-root ${REPORT_ROOT} \
  --chexpert-csv ${CHEXPERT_CSV} \
  --metadata-csv ${METADATA_CSV} \
  --split-csv ${SPLIT_CSV} \
  --output-dir outputs/mimic_medgemma
```

---

# 5. ABIDE Setup

For ABIDE experiments, the code automatically downloads the dataset using `nilearn.datasets.fetch_abide_pcp`.

You may optionally specify a local cache directory:

```bash
export ABIDE_DATA_DIR=/data/common/data/ABIDE_PCP_full
```

Example command:

```bash
python scripts/train_abide_logreg.py \
  --data-dir ${ABIDE_DATA_DIR} \
  --output-dir outputs/abide_logreg
```

If the ABIDE files are not already present in `ABIDE_DATA_DIR`, the dataset will be downloaded automatically.

---

# 6. Summary of Important Environment Variables

## ImageNet

```bash
export ROOT_ZIP_PATH=/data/common/data/imagenet_object_localization_challenge
export CAPTIONS_JSON=/data/common/data/imagenet_captions/imagenet_captions.json
```

## MIMIC-CXR

```bash
export MIMIC_JPG_ROOT=/data/common/data/mimic-cxr-jpg/2.1.0
export REPORT_ROOT=/data/common/data/mimic-cxr/2.1.0/files

export CHEXPERT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz
export METADATA_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz
export SPLIT_CSV=${MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-split.csv.gz
```

## ABIDE

```bash
export ABIDE_DATA_DIR=/data/common/data/ABIDE_PCP_full
```

---

# 7. Running Experiments

## ImageNet + SmolVLM

```bash
python scripts/train_imagenet_smolvlm.py \
  --root-zip-path ${ROOT_ZIP_PATH} \
  --captions-json ${CAPTIONS_JSON} \
  --include-img \
  --include-text \
  --output-dir outputs/imagenet_smolvlm
```

## MIMIC-CXR + MedGemma

```bash
python scripts/train_mimic_medgemma.py \
  --image-root ${MIMIC_JPG_ROOT}/files \
  --report-root ${REPORT_ROOT} \
  --chexpert-csv ${CHEXPERT_CSV} \
  --metadata-csv ${METADATA_CSV} \
  --split-csv ${SPLIT_CSV} \
  --output-dir outputs/mimic_medgemma
```

## ABIDE + Logistic Regression

```bash
python scripts/train_abide_logreg.py \
  --data-dir ${ABIDE_DATA_DIR} \
  --output-dir outputs/abide_logreg
```

---

# 8. Notes on Data Access

ImageNet must be downloaded through Kaggle according to Kaggle's terms.

MIMIC-CXR and MIMIC-CXR-JPG must be downloaded through PhysioNet after credentialed access approval. Do not redistribute MIMIC data, reports, labels, or images.

ABIDE is downloaded automatically by the code through Nilearn and cached locally.

---

# 9. Citation

If you use this software package, please cite the corresponding paper:

```bibtex
@inproceedings{medmap2026,
  title     = {MedMAP: Benchmarking Differentially Private Multimodal Learning},
  author    = {Anonymous Authors},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2026}
}
```

Please also cite the original datasets used in each experiment.

```
::contentReference[oaicite:1]{index=1}
```

[1]: https://github.com/mlfoundations/imagenet-captions "GitHub - mlfoundations/imagenet-captions: Release of ImageNet-Captions · GitHub"
