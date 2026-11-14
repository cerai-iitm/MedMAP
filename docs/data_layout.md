# Data layout and one-command workflows

MedMAP uses one repository-local workspace by default. Run this once from the
repository root:

```bash
bash scripts/setup_workspace.sh
```

This creates the following untracked directories:

```text
data/
├── raw/                         # immutable downloads
│   ├── mimic-cxr/
│   ├── abide/
│   ├── pad-ufes-20/
│   └── tiny-imagenet/
└── processed/                   # reproducible preprocessing products
    ├── mimic-cxr/
    │   ├── best-image-index/
    │   ├── reports-mmp/
    │   └── dataset-cache/
    ├── abide/fc-features/
    ├── pad-ufes-20/
    └── tiny-imagenet/
models/cache/
├── huggingface/
└── torch/
outputs/
├── mimic-cxr/
├── abide/
├── pad-ufes-20/
└── tiny-imagenet/
```

Raw files are never edited. Generated MIMIC reports, feature arrays, dataset
caches, checkpoints, and metrics are written to their processed/output roots.
All paths saved by MIMIC preprocessing are relative to the declared raw JPG
root, so moving the whole workspace does not invalidate its image index.

## Moving the workspace elsewhere

The default is convenient but not required. Set these variables before any
setup, download, prepare, or run command:

```bash
export MEDMAP_DATA_ROOT=/mnt/medmap/data
export MEDMAP_MODEL_CACHE=/mnt/medmap/model-cache
export MEDMAP_OUTPUT_ROOT=/mnt/medmap/outputs
bash scripts/setup_workspace.sh
```

Every helper and launcher uses the same three values. Do not edit scripts to
change paths.

## MIMIC-CXR

MIMIC-CXR cannot be downloaded until the user completes PhysioNet
credentialing and accepts both data use agreements:

1. Create a [PhysioNet account](https://physionet.org/register/).
2. Complete the required CITI "Data or Specimens Only Research" training.
3. Request access and sign the agreement for
   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/).
4. Separately request access and sign the agreement for
   [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
5. Accept the model terms of the backbone you will fine-tune on Hugging Face,
   [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-pt) and/or
   [MedGemma 4B](https://huggingface.co/google/medgemma-4b-pt) (selected with
   `MIMIC_BACKBONE=gemma|medgemma`), then run `hf auth login` if the model is
   not already cached.

After approval:

```bash
PHYSIONET_USER=your_username bash scripts/download_mimic_cxr.sh
bash scripts/prepare_mimic_cxr.sh
bash examples/run_mimic_non_private.sh
```

The downloader asks for the PhysioNet password interactively and never puts it
in a command, file, or environment variable. By default it downloads only the
benchmark shards `p10 p11 p12`.

`prepare_mimic_cxr.sh` performs both required deterministic preprocessing
stages:

1. `heuristic_based_img_selector.py`-equivalent selection: one image per study
   with `PA > AP > LATERAL > OTHER`, written to
   `data/processed/mimic-cxr/best-image-index/`.
2. The requested `ex_v1.py` behavior (implemented here as the `_mmp` pipeline):
   remove Findings/Impression and mask CheXpert/NegBio label mentions, written
   to `data/processed/mimic-cxr/reports-mmp/`. Raw reports remain untouched.

Examples with overrides:

```bash
# Text-only non-private ablation
bash examples/run_mimic_non_private.sh \
  --no-include_img --include_txt --max_length 226 --learning_rate 1e-4

# Feature-DP (default private mechanism)
TARGET_EPSILON=8 bash examples/run_mimic_private.sh --seed 100

# Full DP-SGD
MIMIC_PRIVACY_MODE=full_dp TARGET_EPSILON=8 \
  bash examples/run_mimic_private.sh --learning_rate 1e-3
```

See [mimic_cxr.md](mimic_cxr.md) for algorithm and parameter details.

## ABIDE I

ABIDE I is downloaded by nilearn from the Preprocessed Connectomes Project.
The selected C-PAC derivatives are large (approximately 100+ GB) and feature
extraction takes several hours:

```bash
bash scripts/prepare_abide.sh
```

This writes raw PCP content to `data/raw/abide/` and reusable 741-dimensional
functional-connectivity features to `data/processed/abide/fc-features/`.
Run one arm at a time:

```bash
# Baseline (run first for each split)
SPLIT=non_iid METHOD=lbfgs bash examples/run_abide_classic.sh

# Full DP and Feature-DP
SPLIT=non_iid METHOD=full_dp TARGET_EPSILON=8 \
  bash examples/run_abide_classic.sh --iterations 3000
SPLIT=non_iid METHOD=feat_dp TARGET_EPSILON=8 \
  bash examples/run_abide_classic.sh --iterations 3000

# Aggregate a completed split
python -m scripts.summarize_abide \
  --data-dir data/raw/abide \
  --output-root outputs/abide \
  --split non_iid --stages all
```

See [abide.md](abide.md) for split, privacy, and licensing details.

## PAD-UFES-20

PAD-UFES-20 is public under CC BY 4.0. One helper downloads the official
Mendeley archive, validates every metadata image, downloads/caches the
EfficientNet-B3 weights when needed, and creates the reusable image-feature
cache:

```bash
bash scripts/prepare_pad_ufes_20.sh
```

Run one arm and pass any `scripts.train_pad` option after the launcher:

```bash
# Non-private image-only (default)
bash examples/run_pad.sh --epochs 100

# Non-private image + metadata
MODE=sgd bash examples/run_pad.sh --use_metadata --hidden_dim 256

# Full DP or Feature-DP
MODE=dpsgd TARGET_EPSILON=8 bash examples/run_pad.sh
MODE=feature_dpsgd TARGET_EPSILON=8 bash examples/run_pad.sh --C_meta 0.5
```

See [pad_ufes.md](pad_ufes.md) for preprocessing and mechanism details.

## Tiny ImageNet / ImageNet-Captions

This task requires three sources. ImageNet/ILSVRC access is conditional on
accepting its non-commercial research terms and the Kaggle competition rules.
Configure the Kaggle CLI first (`~/.kaggle/kaggle.json` must have mode 600),
then run:

```bash
bash scripts/download_tiny_imagenet.sh
bash examples/run_imagenet_non_private.sh --validate-only
bash examples/run_imagenet_non_private.sh
```

If ILSVRC was obtained through another approved channel, place either the
competition ZIP or extracted `ILSVRC/` tree under
`data/raw/tiny-imagenet/ilsvrc/`, then run:

```bash
bash scripts/download_tiny_imagenet.sh --skip-ilsvrc
bash examples/run_imagenet_non_private.sh --validate-only
```

The validation command checks the source tree/ZIP, exactly 200 unique WNIDs,
the caption/image join, and packaged split bounds/overlap without loading
SmolVLM. Training examples:

```bash
bash examples/run_imagenet_non_private.sh \
  --epochs 5 --batch-size 16 --learning-rate 1e-4
IMAGENET_PRIVACY_MODE=feature_dp TARGET_EPSILON=4 \
  bash examples/run_imagenet_private.sh --gradient-checkpointing
IMAGENET_PRIVACY_MODE=full_dp TARGET_EPSILON=4 \
  bash examples/run_imagenet_private.sh
```

See [imagenet.md](imagenet.md) for source terms and all CLI parameters.

## Common launcher behavior

- Launchers may be called from any directory; they locate the repository.
- Values after a launcher are forwarded unchanged to the Python CLI, so
  explicit user options override launcher defaults.
- W&B is disabled by default. Enable it deliberately with
  `--wandb-mode online` (Tiny ImageNet), `--wandb_mode online` (MIMIC), or
  the documented backend-specific setting.
- Downloaded data, model caches, and outputs are ignored by Git. Never add
  restricted MIMIC data or generated reports to a submission archive.
