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

scripts/                     runnable entry points (python -m scripts.<name>)
examples/                    ready-to-edit bash launchers for every arm
docs/                        per-dataset guides + W&B setup
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

Each dataset has a two-step pattern — prepare once, then sweep:

```bash
# --- MIMIC-CXR (after PhysioNet download, see docs/mimic_cxr.md) ---
python -m scripts.select_mimic_images p10 p11 p12 --dataset-root ... --output-root ...
python -m scripts.extract_mimic_reports --shard p10 p11 p12 --reports-root ...
bash examples/run_mimic_non_private.sh          # or run_mimic_private.sh

# --- ABIDE I (see docs/abide.md) ---
python -m scripts.extract_abide_features --data-dir $ABIDE_DATA_DIR
bash examples/run_abide_classic.sh              # sklearn/NumPy backend (paper results)
bash examples/run_abide_hf_private.sh           # HF backend (same DP stack as the deep models)

# --- PAD-UFES-20 (auto-downloads, see docs/pad_ufes.md) ---
bash examples/run_pad.sh

# --- Tiny ImageNet / SmolVLM (see docs/imagenet.md) ---
bash examples/run_imagenet_private.sh
```

Weights & Biases logging is on by default for the deep-learning arms and
fully optional — `--wandb_mode disabled` / `WANDB_MODE=disabled` turns it
off. Setup instructions: [docs/wandb.md](docs/wandb.md).

## Where to go next: experiment guide

Every experiment follows the same workflow: **download → preprocess once →
run the arm(s) you need → aggregate**. The table below routes you to the
right command and doc section for each step.

### MIMIC-CXR (MedGemma) — [docs/mimic_cxr.md](docs/mimic_cxr.md)

| To do this... | Go here |
| --- | --- |
| Download images + reports (PhysioNet, credentialed) | [docs/mimic_cxr.md §1](docs/mimic_cxr.md) |
| Select one image per study | `python -m scripts.select_mimic_images p10 p11 p12 --dataset-root ... --output-root ...` |
| Generate leakage-filtered `_mmp` reports | `python -m scripts.extract_mimic_reports --shard p10 p11 p12 --reports-root ...` |
| Non-private training (both / image-only / text-only) | `examples/run_mimic_non_private.sh`; choose modalities with `--include_img` / `--include_txt` |
| Full DP-SGD | `examples/run_mimic_private.sh` with `--no-feature_dp` |
| Feature-DP | `examples/run_mimic_private.sh` (default: `--feature_dp`); sweep with `TARGET_EPSILON=... SEED=...` |
| Results | `output_mimic_*/outputs_mimic_cxr_medgemma_*/` (metrics JSON, prediction dumps, checkpoints) + W&B |

### ABIDE I (logistic regression) — [docs/abide.md](docs/abide.md)

| To do this... | Go here |
| --- | --- |
| Download (nilearn, open access) | [docs/abide.md §1](docs/abide.md) |
| Extract FC features (once, ~hours) | `python -m scripts.extract_abide_features --data-dir $ABIDE_DATA_DIR` |
| Non-private baseline + create the fixed split (run this first) | `python -m scripts.train_abide_classic --split {iid,non_iid} --method lbfgs --data-dir ...` |
| Full DP-GD at a given ε | `python -m scripts.train_abide_classic --split ... --method full_dp --epsilon 8 --data-dir ...` |
| Feature-DP at a given ε | `python -m scripts.train_abide_classic --split ... --method feat_dp --epsilon 8 --data-dir ...` |
| Whole benchmark in one go (baseline → ε sweep → summaries) | `examples/run_abide_classic.sh` |
| Summary tables, privacy-utility plots, subgroup/LaTeX tables | `python -m scripts.summarize_abide --data-dir ... --split ... --stages all` |
| Same task on the HF/torch stack (swappable-model template) | `examples/run_abide_hf_non_private.sh` / `examples/run_abide_hf_private.sh` (`FEATURE_DP=1` for Feature-DP) |
| Results | `<data-dir>/outputs/{iid,non_iid}_fixed_split/...` (per-seed dirs, accounting JSONs, summary CSVs, plots) |

### PAD-UFES-20 (EfficientNet + MLP) — [docs/pad_ufes.md](docs/pad_ufes.md)

| To do this... | Go here |
| --- | --- |
| Download (automatic on first run) | add `--download` to any command below |
| Non-private, image-only / image+metadata | `python -m scripts.train_pad --mode sgd --lr 1e-3 --data_root ...` (add `--use_metadata`) |
| Full DP-SGD | `python -m scripts.train_pad --mode dpsgd --lr 1e-3 --epsilon 8 --data_root ...` |
| Feature-DP | `python -m scripts.train_pad --mode feature_dpsgd --lr 1e-3 --epsilon 8 --data_root ...` |
| Full grid (both baselines + ε sweep) | `examples/run_pad.sh` |
| Results | stdout `RESULT ...` lines + per-subgroup accuracies appended to `subgroup_results.csv` |

### Tiny ImageNet (SmolVLM) — [docs/imagenet.md](docs/imagenet.md)

| To do this... | Go here |
| --- | --- |
| Download (Kaggle ILSVRC + ImageNet-Captions + tiny-imagenet wnids) | [docs/imagenet.md §1](docs/imagenet.md) |
| Non-private (modalities via `INCLUDE_IMG` / `INCLUDE_TEXT`) | `examples/run_imagenet_non_private.sh` |
| Full DP-SGD | `FEATURE_DP=0 examples/run_imagenet_private.sh` |
| Feature-DP | `FEATURE_DP=1 examples/run_imagenet_private.sh` |
| Results | `outputs_smolvlm_*/` + W&B |

Common conventions across all datasets: `--use_dp` / `USE_DP=1` switches
privacy on and `feature_dp` selects Feature-DP vs. full DP-SGD; ε sweeps are
one target per invocation; seeds are explicit everywhere; preprocessing
results are cached so sweeps only pay the training cost. Note that reported
ε values cover the final training run (hyperparameter selection is not
composed into the budget).

Every entry point also documents all of its parameters: each dataset page
ends with a **Parameter reference** section explaining every CLI flag /
environment variable, its default, and its purpose, and `--help` works on
every script.

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
