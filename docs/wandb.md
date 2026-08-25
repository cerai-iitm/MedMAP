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
| `scripts/train_mimic.py` | `--wandb_project` (default `medmap-mimic-medgemma`), `--wandb_entity`, `--wandb_mode {online,offline,disabled}`, `--wandb_tags a,b,c`, `--wandb_watch_model`, `--gpu_log_every_n_steps` |
| `scripts/train_imagenet.py` | env vars: `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, `RUN_NAME` (run name is auto-generated from the setting if unset) |
| `scripts/train_abide_hf.py` | env vars: `WANDB_PROJECT` (default `medmap-abide-linear`), `WANDB_ENTITY`, `WANDB_MODE`, `RUN_NAME` |

Run names encode the experiment configuration (mechanism, epsilon, clip
norms, learning rate, seed, timestamp), e.g.
`MIMIC-CXR-Medgemma-FeatureDP-eps-4.0-clip-1.0-public-clip-50.0-lr-0.002-seed-42-...`,
so sweeps stay organized without extra bookkeeping.

## Running without W&B

Every experiment runs fine without an account:

```bash
# MIMIC
python -m scripts.train_mimic ... --wandb_mode disabled

# ImageNet / ABIDE HF
WANDB_MODE=disabled bash examples/run_imagenet_non_private.sh
WANDB_MODE=disabled bash examples/run_abide_hf_private.sh
```

With W&B disabled, the W&B-specific callbacks are skipped and all metrics
still go to stdout and to the JSON/CSV files in the output directory.

## Offline clusters

Use `WANDB_MODE=offline` (or `--wandb_mode offline`) on air-gapped machines;
sync later with:

```bash
wandb sync wandb/offline-run-*
```
