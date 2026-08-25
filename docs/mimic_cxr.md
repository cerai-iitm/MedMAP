# MIMIC-CXR: chest X-rays + leakage-filtered reports (MedGemma)

Multi-label classification of the 12 CheXpert pathology labels from chest
X-ray images (public modality) and free-text radiology reports (private
modality), fine-tuning MedGemma (`google/medgemma-4b-pt`) with LoRA.

Contents: [Access & download](#1-access--download) ·
[Preprocessing](#2-preprocessing) · [Training](#3-training) ·
[Reproducibility notes](#4-reproducibility-notes) ·
[License & citations](#5-license--citations)

---

## 1. Access & download

MIMIC-CXR is a restricted-access dataset hosted on PhysioNet. You need:

1. A [PhysioNet](https://physionet.org/) account.
2. Completed CITI "Data or Specimens Only Research" training.
3. Signed data use agreements for **both** projects:
   - [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) (free-text reports)
   - [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) (JPG images + label/split CSVs)

Both are distributed under the **PhysioNet Credentialed Health Data License
1.5.0**. Do not redistribute any file from either project; this repository
contains no MIMIC data.

Download (choose a local root, called `$DATA_ROOT` below). The benchmark uses
shards **p10, p11, p12** (subject IDs starting 10/11/12); downloading only
those saves substantial space:

```bash
DATA_ROOT=/data/mimic
cd "$DATA_ROOT"

# Reports (mimic-cxr): the .txt radiology reports for p10-p12
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr/2.1.0/files/p10/ \
    https://physionet.org/files/mimic-cxr/2.1.0/files/p11/ \
    https://physionet.org/files/mimic-cxr/2.1.0/files/p12/

# Images + metadata/label/split CSVs (mimic-cxr-jpg)
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/ \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p11/ \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p12/
wget -N --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
    https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz
gunzip -k mimic-cxr-2.0.0-*.csv.gz
```

Expected layout afterwards:

```
$DATA_ROOT/physionet.org/files/
├── mimic-cxr/2.1.0/files/p10/p10000032/s50414267.txt          # reports
└── mimic-cxr-jpg/2.1.0/
    ├── files/p10/p10000032/s50414267/<dicom_id>.jpg           # images
    ├── mimic-cxr-2.0.0-metadata.csv                           # ViewPosition etc.
    ├── mimic-cxr-2.0.0-chexpert.csv                           # CheXpert labels
    └── mimic-cxr-2.0.0-split.csv                              # official split
```

(The CSVs keep their `2.0.0` names inside the 2.1.0 release.)

## 2. Preprocessing

### 2a. Representative image per study

A study can contain several images (frontal, lateral, repeats). We select
**exactly one image per study** with a deterministic, metadata-only
heuristic (`medmap/datasets/mimic_cxr/image_selection.py`):

```
PA > AP > LATERAL > OTHER_OR_UNKNOWN
```

`ViewPosition` from `mimic-cxr-2.0.0-metadata.csv` is the primary field;
`ViewCodeSequence_CodeMeaning` is the fallback. Ties within the best tier are
broken by the lexicographically smallest `dicom_id`, and images whose JPG is
missing on disk are skipped in favor of the next-best candidate. The selector
never looks at report text, labels, image content, or model predictions, so
it cannot leak label information.

*Why PA first?* The erect posteroanterior view is the standard frontal chest
projection: the heart and mediastinum sit close to the detector, so their
size and contours are depicted accurately, whereas portable AP projections
magnify these structures and are generally of lower diagnostic quality (see
the chest roentgenography chapter of *Clinical Methods*, 3rd ed. — full
reference below). Frontal-view selection is also standard practice in chest
X-ray ML pipelines (e.g., TorchXRayVision standardizes datasets to frontal
PA/AP views). Preferring the single most informative, most standardized
projection keeps exactly one well-defined image per study.

```bash
python -m scripts.select_mimic_images p10 p11 p12 \
    --dataset-root "$DATA_ROOT/physionet.org/files/mimic-cxr-jpg/2.1.0" \
    --output-root  "$DATA_ROOT/best_img_index"
```

Output: `$DATA_ROOT/best_img_index/<shard>/<shard>_best_img_final.csv` with
columns `(image_path, subject_id, study_id, dicom_id)` — one row per study.

### 2b. Leakage-filtered reports (`_mmp`)

The CheXpert labels were generated *from* the reports, so feeding raw
reports to a classifier leaks the labels. The extraction pipeline
(`medmap/datasets/mimic_cxr/report_extraction.py`) writes a leakage-filtered
sibling `s<study_id>_mmp.txt` next to every original report:

1. Sections are parsed with the **official MIT-LCP section parser**
   (vendored, pinned; the same code that produced the MIMIC-CXR-JPG labels),
   and every `FINDINGS` / `IMPRESSION` section is removed, including the
   official per-study manual overrides.
2. On the remaining text (indication, history, comparison, ...), mentions of
   the label vocabulary are masked with `___` using the **official CheXpert
   phrase lists** and the **official NegBio regex patterns** (both vendored,
   pinned — see `medmap/datasets/mimic_cxr/vendor/SOURCES.md`), with matched
   stems expanded to whole tokens (`opaci` → `opacities`).
3. Reports with no usable Findings/Impression section are listed in a
   per-shard audit CSV (`reports_without_findings_or_impression_mmp.csv`);
   those studies are excluded downstream.

Originals are never modified, and only canonical `s<study_id>.txt` files are
accepted as input, so generated files can never be re-processed.

```bash
# Read-only validation first
python -m scripts.extract_mimic_reports --shard p10 --dry-run --limit 100 \
    --reports-root "$DATA_ROOT/physionet.org/files/mimic-cxr/2.1.0/files"

# Full processing
python -m scripts.extract_mimic_reports --shard p10 p11 p12 \
    --reports-root "$DATA_ROOT/physionet.org/files/mimic-cxr/2.1.0/files"
```

Unit tests for the pipeline: `pytest -q tests/test_mimic_report_extraction.py`.

### 2c. Dataset build & cache (automatic)

`scripts/train_mimic.py` merges the best-image index, the `_mmp` reports and
the CheXpert labels (dropping `Support Devices` and `No Finding`; `-1`
uncertain → NaN, ignored by the masked loss), applies the **official
PhysioNet split**, and caches the result on disk keyed by a fingerprint of
every source file. Sweeps over seeds/epsilons reuse the cache automatically;
pass `--rebuild_dataset_cache` to force a rebuild.

## 3. Training

```bash
# Non-private, both modalities (see examples/run_mimic_non_private.sh)
python -m scripts.train_mimic \
    --output_root outputs_mimic --attempt 1 \
    --top_level_buckets p10 p11 p12 \
    --best_img_index_root "$DATA_ROOT/best_img_index" \
    --chexpert_csv "$DATA_ROOT/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv" \
    --split_csv    "$DATA_ROOT/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv" \
    --report_root  "$DATA_ROOT/physionet.org/files/mimic-cxr/2.1.0/files" \
    --dataset_cache_root "$DATA_ROOT/medmap_dataset_cache" \
    --include_img --include_txt --bf16 --tf32 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 --num_train_epochs 3.0 --seed 42

# Modality ablations: pass exactly one of --include_img / --include_txt
# (text-only runs used --max_length 226; image+text runs 426).

# Feature-DP at epsilon=4 (image public, text private); full DP-SGD with
# --no-feature_dp. See examples/run_mimic_private.sh.
python -m scripts.train_mimic ... \
    --use_dp --feature_dp --target_epsilon 4.0 \
    --clip_norm 1.0 --public_clip_norm 50.0 \
    --no-bf16 --no-tf32 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
    --learning_rate 2e-3
```

Notes:

- DP always trains on both modalities; the per-device batch is multiplied by
  `gradient_accumulation_steps` (manual microbatching inside the DP trainer).
- Privacy runs through the modified dp-transformers fork
  (`library/dp-transformers`, see its `MODIFICATIONS.md`): per-sample
  gradients are decomposed into a public (image-only) component clipped at
  `--public_clip_norm` and a private residual clipped at `--clip_norm`; only
  the private component sets the noise scale. The noise multiplier is
  calibrated to `--target_epsilon` with an RDP accountant.
- W&B is on by default; use `--wandb_mode disabled` to turn it off
  (see `docs/wandb.md`).

## 4. Reproducibility notes

- The official PhysioNet split (train/validate/test) is used; no random
  splitting.
- Seeds: 42, 100 and 123 were used for the reported runs (`--seed`).
- The image selection and report extraction are fully deterministic, so the
  dataset is reconstructible bit-for-bit from a fresh MIMIC download.
- `--attempt N` namespaces the output directory of repeated configurations.

## 5. License & citations

**Data license:** PhysioNet Credentialed Health Data License 1.5.0
(no redistribution; credentialed access only).
**Model terms:** MedGemma is subject to Google's Health AI Developer
Foundations terms of use (accepted on Hugging Face before download).

Please cite the following when using this part of the benchmark:

```bibtex
@article{johnson2019mimiccxr,
  title   = {MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author  = {Johnson, Alistair E W and Pollard, Tom J and Berkowitz, Seth J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Mark, Roger G and Horng, Steven},
  journal = {Scientific Data}, volume = {6}, number = {317}, year = {2019}
}

@article{johnson2019mimiccxrjpg,
  title   = {MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs},
  author  = {Johnson, Alistair E W and Pollard, Tom J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Peng, Yifan and Lu, Zhiyong and Mark, Roger G and Berkowitz, Seth J and Horng, Steven},
  journal = {arXiv preprint arXiv:1901.07042}, year = {2019}
}

@article{goldberger2000physionet,
  title   = {PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals},
  author  = {Goldberger, Ary L and Amaral, Luis A N and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
  journal = {Circulation}, volume = {101}, number = {23}, pages = {e215--e220}, year = {2000}
}

% Label vocabulary / masking resources (leakage filtering)
@inproceedings{irvin2019chexpert,
  title     = {CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
  booktitle = {AAAI}, volume = {33}, pages = {590--597}, year = {2019}
}

@inproceedings{peng2018negbio,
  title     = {NegBio: a high-performance tool for negation and uncertainty detection in radiology reports},
  author    = {Peng, Yifan and Wang, Xiaosong and Lu, Le and Bagheri, Mohammadhadi and Summers, Ronald and Lu, Zhiyong},
  booktitle = {AMIA Summits on Translational Science Proceedings}, pages = {188--196}, year = {2018}
}

@article{johnson2018mimic_code,
  title   = {The MIMIC Code Repository: enabling reproducibility in critical care research},
  author  = {Johnson, Alistair E W and Stone, David J and Celi, Leo A and Pollard, Tom J},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {25}, number = {1}, pages = {32--39}, year = {2018}
}

% Basis for the PA-first representative-image heuristic
@incollection{chestroentgenography1990,
  title     = {Chest Roentgenography for Pulmonary Evaluation},
  booktitle = {Clinical Methods: The History, Physical, and Laboratory Examinations},
  edition   = {3rd}, editor = {Walker, H Kenneth and Hall, W Dallas and Hurst, J Willis},
  publisher = {Butterworths}, address = {Boston}, year = {1990},
  note      = {NCBI Bookshelf NBK370: erect PA is the standard projection; bedside AP films magnify the cardiomediastinal structures}
}

@article{cohen2022torchxrayvision,
  title   = {TorchXRayVision: A library of chest X-ray datasets and models},
  author  = {Cohen, Joseph Paul and Viviano, Joseph D and Bertin, Paul and Morrison, Paul and Torabian, Parsa and Guarrera, Matteo and Lungren, Matthew P and Chaudhari, Akshay and Brooks, Rupert and Hashir, Mohammad and Bertrand, Hadrien},
  journal = {Proceedings of Machine Learning Research (MIDL)}, year = {2022},
  note    = {standardizes chest X-ray datasets to frontal (PA/AP) views}
}

% Backbone model
@article{sellergren2025medgemma,
  title   = {MedGemma Technical Report},
  author  = {Sellergren, Andrew and others},
  journal = {arXiv preprint arXiv:2507.05201}, year = {2025}
}
```
