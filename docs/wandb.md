# Weights & Biases (W&B) setup

All deep-learning entry points (`train_mimic.py`, `train_imagenet.py`,
`train_abide_hf.py`) integrate with [Weights & Biases](https://wandb.ai) for
experiment tracking: losses and metrics per step/epoch, GPU/CPU memory
stats, per-label ROC curves (MIMIC), mirrored validation/test evaluation,
and prediction dumps. The classic ABIDE backend and the PAD pipeline write
plain CSV/JSON outputs instead and do not use W&B.

## Setup

```bash
pip install wandb          # already included in requirements.txt
wandb login                # paste your API key from https://wandb.ai/authorize
```

Optionally set defaults once per shell:

```bash
export WANDB_ENTITY=your-team-or-username
```

## Per-run configuration

| Entry point | How to configure |
| --- | --- |
| `scripts/train_mimic.py` | CLI: `--wandb_project` (default `medmap-mimic-<backbone>`), `--wandb_entity`, `--wandb_mode {online,offline,disabled}`, `--wandb_tags`, `--wandb_watch_model`; default mode is `disabled`. |
| `scripts/train_imagenet.py` | CLI: `--wandb-project`, `--wandb-entity`, `--wandb-mode {online,offline,disabled}`, `--run-name`; default mode is `disabled`. |
| `scripts/train_abide_hf.py` | Legacy env vars: `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, `RUN_NAME`. |

Run names encode the experiment configuration (mechanism, epsilon, clip
norms, learning rate, seed, timestamp), e.g.
`MIMIC-CXR-Gemma34B-FeatureDP-eps-4.0-clip-1.0-public-clip-69.0-lr-0.002-seed-41-...`
or `MIMIC-CXR-MedGemma4B-FeatureDP-eps-4.0-clip-1.0-public-clip-50.0-lr-0.002-seed-42-...`,
so sweeps stay organized without extra bookkeeping.

## Running without W&B

Every experiment runs fine without an account:

```bash
# MIMIC
python -m scripts.train_mimic ... --wandb_mode disabled

# ImageNet / ABIDE HF
bash examples/run_imagenet_non_private.sh --wandb-mode disabled
WANDB_MODE=disabled bash examples/run_abide_hf_private.sh
```

With W&B disabled, the W&B-specific callbacks are skipped and all metrics
still go to stdout and to the JSON/CSV files in the output directory.

## Offline clusters

Use `WANDB_MODE=offline` for the ABIDE HF backend,
`--wandb_mode offline` for MIMIC, or `--wandb-mode offline` for Tiny ImageNet
on air-gapped machines; sync later with:

```bash
wandb sync wandb/offline-run-*
```
