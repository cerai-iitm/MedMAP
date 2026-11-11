# MedMAP: A Benchmark for Multimodal Medical Machine Learning under Differential Privacy

MedMAP evaluates how much utility multimodal medical models retain when the
**sensitive modality** of each dataset is protected with differential
privacy. Every dataset pairs a *public* modality (imaging) with a *private*
modality (text or tabular clinical data), and every task is run under three
regimes on identical data pipelines:

1. **Non-private** training (with per-modality ablations),
2. **DP-SGD** — the standard baseline: every example's entire gradient is
   clipped and noised, and
3. **Feature-DP** — only the gradient component attributable to the
   private modality is clipped and noised; the public component stays exact.
   At equal privacy budget ε, Feature-DP consistently retains more utility.

## Datasets & tasks

| Dataset | Task | Public modality | Private modality | Model | Doc |
| --- | --- | --- | --- | --- | --- |
| [MIMIC-CXR](docs/mimic_cxr.md) | 12-label CheXpert pathology classification | chest X-ray | leakage-filtered radiology report | MedGemma-4B (LoRA) | [docs/mimic_cxr.md](docs/mimic_cxr.md) |
| [ABIDE I](docs/abide.md) | autism vs. control | fMRI functional connectivity | phenotypic record | logistic regression (sklearn **and** HF backends) | [docs/abide.md](docs/abide.md) |
| [PAD-UFES-20](docs/pad_ufes.md) | 6-class skin-lesion diagnosis | lesion photo (EfficientNet-B3 features) | clinical questionnaire | MLP head | [docs/pad_ufes.md](docs/pad_ufes.md) |
| [Tiny ImageNet](docs/imagenet.md) | 200-class classification | image | Flickr caption (ImageNet-Captions) | SmolVLM-256M (LoRA) | [docs/imagenet.md](docs/imagenet.md) |

No dataset content is redistributed here. Each doc page walks through
access/credentialing, exact download commands, expected directory layout,
preprocessing, training, and the required citations and license terms.

## Repository layout

```
medmap/                      the Python package
├── common/                  seeding, env helpers, shared utilities
├── datasets/
│   ├── mimic_cxr/           image selection, leakage-filtered report
│   │                        extraction (vendored official CheXpert/NegBio/
│   │                        mimic-code resources), cached dataset builder
│   ├── abide/               HF data pipeline + packaged feature lists
│   └── imagenet/            ImageNet-Captions builder + fixed split indices
├── models/                  medgemma, smolvlm, abide_linear, pad_mlp —
│                            unified forward contract (public= / feature_dp=)
│                            so models are swappable under one DP trainer
├── training/                trainers (standard + DP), collators, callbacks,
│                            metrics
├── abide_classic/           faithful packaging of the original ABIDE
│                            research scripts (sklearn LBFGS + NumPy/Numba
│                            DP-GD & Feature-DP + summarizers)
└── pad/                     PAD-UFES-20 pipeline (features, DP loops)

scripts/                     Python entry points plus workspace/download/
                             preprocessing shell helpers
examples/                    parameterized one-run launchers; all forward "$@"
docs/                        shared data-layout guide + per-dataset guides
library/dp-transformers/     vendored, MODIFIED dp-transformers fork — the
                             DP/Feature-DP backbone (see MODIFICATIONS.md)
tests/                       unit tests (report leakage filtering)
```

## Installation

```bash
git clone <repository-url> MedMAP && cd MedMAP
conda create -n medmap python=3.11 -y && conda activate medmap

# Reference environment (exact pins used for the reported results):
pip install -r requirements.txt
# ...or a flexible install:
pip install -e ".[all]"
```

The vendored dp-transformers fork is picked up automatically when running
via `python -m scripts.<name>` from the repository root (a small bootstrap
adds `library/dp-transformers/src` to `sys.path`). To use it outside the
repo: `pip install -e library/dp-transformers`.

GPU requirements: MedGemma fine-tuning was run on a single Nvidia L40S GPU
(bf16, LoRA), ABIDE, PAD and Imagenet experiments fit on much smaller
GPUs, and the ABIDE classic backend is CPU-only (NumPy/Numba).

## Quick start

All four datasets use the same repository-local workspace. Create it once;
paths can be moved without editing scripts by setting `MEDMAP_DATA_ROOT`,
`MEDMAP_MODEL_CACHE`, and `MEDMAP_OUTPUT_ROOT` first:

```bash
bash scripts/setup_workspace.sh
```

The complete cold-start workflow, exact directory tree, access requirements,
and override examples are in **[docs/data_layout.md](docs/data_layout.md)**.
The shortest per-dataset sequence is:

```bash
# MIMIC-CXR: complete both PhysioNet DUAs first
PHYSIONET_USER=your_username bash scripts/download_mimic_cxr.sh
bash scripts/prepare_mimic_cxr.sh
bash examples/run_mimic_non_private.sh --learning_rate 1e-4

# ABIDE I: download C-PAC data and extract connectivity features
bash scripts/prepare_abide.sh
SPLIT=non_iid METHOD=lbfgs bash examples/run_abide_classic.sh

# PAD-UFES-20: download, validate, and cache EfficientNet features
bash scripts/prepare_pad_ufes_20.sh
MODE=sgd bash examples/run_pad.sh --use_metadata --epochs 100

# Tiny ImageNet: accept ImageNet/Kaggle terms and configure Kaggle first
bash scripts/download_tiny_imagenet.sh
bash examples/run_imagenet_non_private.sh --validate-only
bash examples/run_imagenet_non_private.sh --epochs 5
```

Every run launcher forwards additional arguments to its Python CLI. W&B is
disabled by default for offline and anonymous use; enable it explicitly only
when desired. Dataset-specific implementation and parameter details remain in
the four guides linked above.

## Where to go next: experiment guide

Every experiment follows the same workflow: **download → preprocess once →
run the arm(s) you need → aggregate**. The table below routes you to the
right command and doc section for each step.

### MIMIC-CXR (MedGemma) — [docs/mimic_cxr.md](docs/mimic_cxr.md)

| To do this... | Go here |
| --- | --- |
| Download images + reports (PhysioNet, credentialed) | `PHYSIONET_USER=... bash scripts/download_mimic_cxr.sh` after both DUAs; see [docs/mimic_cxr.md §1](docs/mimic_cxr.md) |
| Select images + generate leakage-filtered reports | `bash scripts/prepare_mimic_cxr.sh` (writes only under `data/processed/`) |
| Non-private training (both / image-only / text-only) | `examples/run_mimic_non_private.sh`; append normal CLI overrides |
| Full DP-SGD | `MIMIC_PRIVACY_MODE=full_dp examples/run_mimic_private.sh` |
| Feature-DP | `examples/run_mimic_private.sh` (default); set `TARGET_EPSILON` and append hyperparameter overrides |
| Results | `outputs/mimic-cxr/outputs_mimic_cxr_medgemma_*/` (metrics, prediction dumps, checkpoints) |

### ABIDE I (logistic regression) — [docs/abide.md](docs/abide.md)

| To do this... | Go here |
| --- | --- |
| Download + extract FC features (once, ~hours) | `bash scripts/prepare_abide.sh` |
| Non-private baseline + create the fixed split (run first) | `SPLIT=non_iid METHOD=lbfgs bash examples/run_abide_classic.sh` |
| Full DP-GD at a given ε | `METHOD=full_dp TARGET_EPSILON=8 bash examples/run_abide_classic.sh` |
| Feature-DP at a given ε | `METHOD=feat_dp TARGET_EPSILON=8 bash examples/run_abide_classic.sh` |
| Custom hyperparameters | Append any `train_abide_classic.py` flag, e.g. `--iterations 3000 --seeds 21,22` |
| Summary tables, privacy-utility plots, subgroup/LaTeX tables | `python -m scripts.summarize_abide --data-dir data/raw/abide --output-root outputs/abide --split non_iid --stages all` |
| Same task on the HF/torch stack (advanced alternative) | `examples/run_abide_hf_non_private.sh` / `examples/run_abide_hf_private.sh` |
| Results | `outputs/abide/{iid,non_iid}_fixed_split/...` |

### PAD-UFES-20 (EfficientNet + MLP) — [docs/pad_ufes.md](docs/pad_ufes.md)

| To do this... | Go here |
| --- | --- |
| Download, validate, cache EfficientNet features | `bash scripts/prepare_pad_ufes_20.sh` |
| Non-private, image-only / image+metadata | `MODE=sgd bash examples/run_pad.sh` (append `--use_metadata` for metadata) |
| Full DP-SGD | `MODE=dpsgd TARGET_EPSILON=8 bash examples/run_pad.sh` |
| Feature-DP | `MODE=feature_dpsgd TARGET_EPSILON=8 bash examples/run_pad.sh` |
| Custom hyperparameters | Append any `train_pad.py` flag, e.g. `--epochs 50 --hidden_dim 256` |
| Results | stdout `RESULT ...` lines + `outputs/pad-ufes-20/subgroup-results.csv` |

### Tiny ImageNet (SmolVLM) — [docs/imagenet.md](docs/imagenet.md)

| To do this... | Go here |
| --- | --- |
| Download (Kaggle ILSVRC + ImageNet-Captions + Tiny WNIDs) | `bash scripts/download_tiny_imagenet.sh` after accepting ImageNet/Kaggle terms |
| Validate the complete 200-class join and fixed split | `bash examples/run_imagenet_non_private.sh --validate-only` |
| Non-private (modalities are CLI flags) | `examples/run_imagenet_non_private.sh` with appended overrides |
| Full DP-SGD | `IMAGENET_PRIVACY_MODE=full_dp examples/run_imagenet_private.sh` |
| Feature-DP | `examples/run_imagenet_private.sh` (default) |
| Results | `outputs/tiny-imagenet/` |

Common conventions across all datasets: setup/download/preparation helpers use
the shared roots in `docs/data_layout.md`; one-run launchers forward additional
CLI arguments unchanged; privacy sweeps use one target ε per invocation; seeds
are explicit; and reusable preprocessing products are cached. CLI-based entry
points expose `--help`; the legacy ABIDE HF alternative remains configured by
environment variables. Reported ε values cover the final training run
(hyperparameter selection is not composed into the budget).

## Extending MedMAP

- **New model on an existing dataset:** implement the forward contract
  (`labels=`, plus `public=` / `feature_dp=` kwargs returning the
  decomposed loss) and pass it to `medmap.training.trainers.get_trainer`;
  the collators and DP trainer are model-agnostic. `medmap/models/abide_linear.py`
  is the smallest complete example.
- **New dataset:** add a builder under `medmap/datasets/`, a collator if the
  modalities need special batching, and an entry script; the trainer stack
  and accounting are reusable as-is.

## Vendored dp-transformers (modified)

`library/dp-transformers/` is a modified copy of
[microsoft/dp-transformers](https://github.com/microsoft/dp-transformers)
(MIT) implementing the Feature-DP training step (public/private per-sample
gradient decomposition, separate clipping, private-only noise, manual
microbatching, dp-accounting-based calibration; the mechanism is exposed as
`feature_dp` throughout). The complete change log is
in [library/dp-transformers/MODIFICATIONS.md](library/dp-transformers/MODIFICATIONS.md);
all upstream licenses are retained (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Acknowledgements

Built on [dp-transformers](https://github.com/microsoft/dp-transformers)
(Microsoft), [Opacus](https://opacus.ai/), Google
[dp-accounting](https://github.com/google/differential-privacy),
Hugging Face Transformers/Datasets/PEFT, scikit-learn, nilearn, and Numba.
Leakage filtering vendors pinned resources from the
[MIT-LCP mimic-code](https://github.com/MIT-LCP/mimic-code) repository, the
[CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler), and
[NegBio](https://github.com/ncbi-nlp/NegBio).

## Licenses

- **Code:** MIT ([LICENSE](LICENSE)).
- **Third-party components** (modified dp-transformers, vendored HF Trainer
  code, official CheXpert/NegBio/mimic-code resources): see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Data:** each dataset keeps its own license/DUA — PhysioNet Credentialed
  Health Data License 1.5.0 (MIMIC), CC BY-NC-SA (ABIDE), CC BY 4.0
  (PAD-UFES-20), ImageNet terms of access (Tiny ImageNet). Details and
  required citations are on each dataset page under `docs/`.

## Citation

If you use MedMAP, please cite (reference anonymized for double-blind
review; the final citation will appear here upon publication):

```bibtex
@misc{medmap2026,
  title  = {MedMAP: A Benchmark for Multimodal Medical Machine Learning under Differential Privacy},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review}
}
```

Please additionally cite the datasets and tools you use — the exact BibTeX
entries are listed on each dataset page (`docs/*.md`) and in
`THIRD_PARTY_NOTICES.md`.
