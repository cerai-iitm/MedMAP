# MIMIC-CXR: chest X-rays + leakage-filtered reports (MedGemma)

Multi-label classification of the 12 CheXpert pathology labels from chest
X-ray images (public modality) and free-text radiology reports (private
modality), fine-tuning MedGemma (`google/medgemma-4b-pt`) with LoRA.

Contents: [Access & download](#1-access--download) ·
[Preprocessing](#2-preprocessing) · [Training](#3-training) ·
[Reproducibility notes](#4-reproducibility-notes) ·
[Parameter reference](#5-parameter-reference) · [License & citations](#6-license--citations)

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

Use the shared workspace from [data_layout.md](data_layout.md). After access is
approved, the helper downloads benchmark shards **p10, p11, p12** and places
the three root CSVs correctly inside the MIMIC-CXR-JPG version directory:

```bash
bash scripts/setup_workspace.sh
PHYSIONET_USER=your_username bash scripts/download_mimic_cxr.sh
```

`wget` requests the password interactively; the script does not store it.
Set `MIMIC_SHARDS="p10 p11 p12"` only if you need to restate/alter the shard
list. Expected raw layout:

```text
data/raw/mimic-cxr/physionet.org/files/
├── mimic-cxr/2.1.0/files/p10/p10000032/s50414267.txt
└── mimic-cxr-jpg/2.1.0/
    ├── files/p10/p10000032/s50414267/<dicom_id>.jpg
    ├── mimic-cxr-2.0.0-metadata.csv
    ├── mimic-cxr-2.0.0-chexpert.csv
    └── mimic-cxr-2.0.0-split.csv
```

The CSVs retain their `2.0.0` names inside release 2.1.0. To use an existing
approved download, copy or symlink it into exactly this tree, or set
`MEDMAP_DATA_ROOT` before running every helper.

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

The canonical helper runs this stage together with report preprocessing:

```bash
bash scripts/prepare_mimic_cxr.sh
```

It writes `data/processed/mimic-cxr/best-image-index/<shard>/` with columns
`(image_path, subject_id, study_id, dicom_id)`. `image_path` is relative to
the MIMIC-CXR-JPG version root (for example `files/p10/...jpg`), so the index
contains no user/machine path and remains valid when the workspace moves.
Direct CLI users can opt into legacy absolute paths with `--absolute-paths`.

### 2b. Leakage-filtered reports (`_mmp`)

The CheXpert labels were generated *from* the reports, so feeding raw
reports to a classifier leaks the labels. The extraction pipeline
(`medmap/datasets/mimic_cxr/report_extraction.py`) writes a mirrored,
leakage-filtered `s<study_id>_mmp.txt` under
`data/processed/mimic-cxr/reports-mmp/`:

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

Raw reports are never modified. Only canonical `s<study_id>.txt` files are
accepted, so generated files cannot be reprocessed. The `_mmp` implementation
is the packaged equivalent of the reference `extract_reports.py`/`_ex_v1`
pipeline mentioned in the original experiments; only the output suffix and
portable output location differ.

The shared helper executes both preprocessing stages and stops on the first
report error:

```bash
bash scripts/prepare_mimic_cxr.sh
# Set MIMIC_PREPARE_OVERWRITE=1 to regenerate existing processed reports.
```

For a small read-only parser check before the full run:

```bash
python -m scripts.extract_mimic_reports --shard p10 --dry-run --limit 100 \
  --reports-root data/raw/mimic-cxr/physionet.org/files/mimic-cxr/2.1.0/files \
  --output-root data/processed/mimic-cxr/reports-mmp
```

Unit tests: `pytest -q tests/test_mimic_report_extraction.py`.

### 2c. Dataset build & cache (automatic)

`scripts/train_mimic.py` merges the best-image index, the `_mmp` reports and
the CheXpert labels (dropping `Support Devices` and `No Finding`; `-1`
uncertain → NaN, ignored by the masked loss), applies the **official
PhysioNet split**, and caches the result on disk using a source/configuration
fingerprint. Sweeps over seeds/epsilons reuse the cache automatically; pass
`--rebuild_dataset_cache` after intentionally changing processed reports or
other source inputs.

## 3. Training

```bash
# All paths are supplied by the launcher; append hyperparameter overrides.
bash examples/run_mimic_non_private.sh \
  --learning_rate 5e-5 --num_train_epochs 3 --seed 42

# Text-only (image-only uses --include_img --no-include_txt)
bash examples/run_mimic_non_private.sh \
  --no-include_img --include_txt --max_length 226

# Feature-DP at epsilon 4 (default private mechanism)
TARGET_EPSILON=4 bash examples/run_mimic_private.sh --seed 42

# Full DP-SGD
MIMIC_PRIVACY_MODE=full_dp TARGET_EPSILON=4 \
  bash examples/run_mimic_private.sh --learning_rate 2e-3
```

Run either launcher with `--help` for every training option. The launcher
passes `--image_root` so relative image-index paths resolve against the raw
MIMIC-CXR-JPG directory, uses only processed `_mmp` reports, caches the joined
dataset under `data/processed/`, and writes runs under `outputs/mimic-cxr/`.

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

## 5. Parameter reference

### `scripts/select_mimic_images.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `shards` (positional) | — | Which MIMIC shards to index, e.g. `p10 p11 p12` (comma-separated also works). |
| `--dataset-root` | required | MIMIC-CXR-JPG version directory (contains `files/` and the metadata CSV). |
| `--output-root` | required | Where the per-shard index CSVs are written. |
| `--metadata-csv` | `<dataset-root>/mimic-cxr-2.0.0-metadata.csv` | Metadata CSV providing `ViewPosition` etc. |
| `--absolute-paths` | off | By default CSV image paths are portable paths relative to `--dataset-root`; enable only for legacy behavior. |
| `--skip-file-verification` | off | By default images missing on disk are skipped in favor of the next-best candidate; this flag trusts the metadata blindly. |

### `scripts/extract_mimic_reports.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--shard` | required | One or more shards to process, e.g. `--shard p10 p11 p12`. |
| `--reports-root` | required | Immutable raw report root containing `p10..p19`. |
| `--output-root` | raw root (legacy) | Processed root that mirrors shard/subject directories; the canonical helper always sets this to `data/processed/mimic-cxr/reports-mmp`. |
| `--audit-csv` | `<output-root>/<shard>/reports_without_findings_or_impression_mmp.csv` | Explicit single-shard exclusion list; leave unset for multi-shard runs. |
| `--overwrite` | off | Regenerate existing `*_mmp.txt` files (they are skipped by default; the audit CSV is always rebuilt). |
| `--dry-run` | off | Parse and count without writing anything — run this first. |
| `--limit N` | all | Process at most N reports per shard (testing aid). |
| `--progress-every N` | 100 | Progress print frequency. |
| `--fail-fast` | off | Stop on the first error instead of logging and continuing. |

### `scripts/train_mimic.py`

**Data paths** (all produced in §1–§2):

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--top_level_buckets` | required | Shards to train on, e.g. `p10 p11 p12`. |
| `--best_img_index_root` | required | Output of the image selector. |
| `--image_root` | required for relative indexes | MIMIC-CXR-JPG version root used to resolve portable `files/...` image paths. |
| `--chexpert_csv` / `--split_csv` | required | Official label and split CSVs from MIMIC-CXR-JPG. |
| `--report_root` | required | Processed report root containing `_mmp` files and audit CSVs. |
| `--dataset_cache_root` | required | Where the built HF dataset is cached; sweeps reuse it. |
| `--dataset_cache_tag` | `v1` | Free-form tag mixed into the cache key (bump to force a parallel cache). |
| `--rebuild_dataset_cache` | off | Delete and rebuild the cache. |
| `--output_root` | required | Parent directory for run outputs. |
| `--attempt` | required | Integer suffix namespacing repeated runs of the same configuration. |

**Modality & privacy:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--include_img` / `--include_txt` | off/off | Which modalities the model sees (non-private ablations). At least one required; DP forces both on. |
| `--use_dp` | off | Switch differential privacy on. |
| `--feature_dp` / `--no-feature_dp` | on | With `--use_dp`: Feature-DP (clip/noise only the private text component) vs. full DP-SGD (clip/noise the whole gradient). |
| `--target_epsilon` | 4.0 | Privacy budget ε; the noise multiplier is calibrated to reach it (δ = n^-1.1). |
| `--clip_norm` | 1.0 | Per-sample clip norm for the private gradient component (the whole gradient under full DP-SGD). |
| `--public_clip_norm` | 50.0 | Per-sample clip norm for the public (image) component under Feature-DP; large = effectively unclipped. |

**Optimization:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--num_train_epochs` | 3.0 | Training epochs. |
| `--per_device_train_batch_size` | 4 | Batch per device. Under DP this is multiplied by `--gradient_accumulation_steps` (the DP trainer microbatches internally). |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation (microbatch count under DP). |
| `--learning_rate` / `--weight_decay` / `--warmup_ratio` | 5e-5 / 0.01 / 0.03 | AdamW settings. Reported DP runs used lr 2e-3. |
| `--max_length` | 426 | Tokenizer truncation length (use 226 for text-only runs). |
| `--bf16` / `--tf32` | off | Mixed-precision switches (used for non-private runs; DP runs used fp32). |
| `--gradient_checkpointing` | off | Trade compute for memory. |
| `--use_lora`, `--lora_r`, `--lora_alpha`, `--lora_dropout` | on, 16, 32, 0.0 | LoRA adapter configuration on the MedGemma backbone. |
| `--seed` | 42 | Run seed (model init, data order, DP noise). Reported runs: 42/100/123. |
| `--resume_from_checkpoint` | — | Resume path for an interrupted run. |

**Evaluation / checkpointing / logging:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--eval_strategy`, `--eval_steps` | steps, 800 | When to run validation (`epoch` also supported). |
| `--save_strategy`, `--save_steps`, `--save_total_limit` | steps, 800, unlimited | Checkpointing cadence and retention. |
| `--load_best_model_at_end` | on | Restore the best checkpoint (by `--metric_for_best_model`, default `eval_loss`) before the final test evaluation. |
| `--metric_for_best_model` / `--greater_is_better` | eval_loss / auto | Model-selection metric; direction is inferred (loss ⇒ lower is better) unless set. |
| `--logging_steps` | 1 | Train-loss logging frequency. |
| `--wandb_project`, `--wandb_entity`, `--wandb_mode`, `--wandb_tags`, `--wandb_watch_model`, `--gpu_log_every_n_steps` | see `--help` | W&B configuration; `--wandb_mode disabled` runs without W&B (see docs/wandb.md). |

---

## 6. License & citations

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
