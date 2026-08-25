# PAD-UFES-20: skin-lesion images + clinical metadata

Six-class skin-lesion diagnosis (ACK, BCC, MEL, NEV, SCC, SEK) from
smartphone lesion photos (public modality) and the patient's clinical
questionnaire (private modality: age, gender, family background,
smoking/drinking, Fitzpatrick skin type, cancer history, lesion diameters).

Contents: [Download](#1-download) · [Preprocessing](#2-preprocessing) ·
[Training](#3-training) · [Reproducibility notes](#4-reproducibility-notes) ·
[Parameter reference](#5-parameter-reference) · [License & citations](#6-license--citations)

---

## 1. Download

PAD-UFES-20 is openly available from Mendeley Data under **CC BY 4.0**
(DOI: [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1), ~3.3 GB).

Easiest: let the training script download and lay it out for you on the
first run:

```bash
python -m scripts.train_pad --mode sgd --lr 1e-3 --use_metadata \
    --data_root /data/PAD-UFES-20 --download
```

This fetches the official archive, extracts `imgs_part_1..3`, and flattens
all images into `<data_root>/all_images/`. Manual download works too — the
expected layout is simply:

```
/data/PAD-UFES-20/
├── metadata.csv          # 2,298 lesion records, 26 columns
└── all_images/           # all 2,298 images (flat, named by img_id)
```

## 2. Preprocessing

Everything is built into the pipeline (`medmap/pad/`):

- **Images** (`features.py`): a frozen, ImageNet-pretrained EfficientNet-B3
  (300x300 input, `features`+`avgpool`, eval mode) produces a 1536-d
  embedding per image. Extraction runs once and is cached to
  `<data_root>/features_cache_patient_ohe.npz`; every later run (seeds,
  epsilons, modes) reuses the cache.
- **Metadata** (`preprocessing.py`): age is bucketed into 10 clinical bins
  and lesion diameters into 7; every feature column is canonicalized
  (missing/empty → "Unknown", booleans normalized, "BRASIL" → "BRAZIL") and
  **one-hot encoded**. `patient_id`, `lesion_id` and `biopsed` are excluded
  from model inputs; only the target `diagnostic` is label encoded.
- **Split** (`preprocessing.py`): `StratifiedGroupKFold(10)` with
  `groups=patient_id` — fold 0 = test, fold 1 = val, folds 2-9 = train
  (~80/10/10), **patient-disjoint** with a hard leakage check. The split is
  fixed by `--split_seed` (default 101), independent of the run seed.

## 3. Training

Three modes on the same MLP head (`medmap/models/pad_mlp.py`:
Linear(input, 512) → ReLU → Linear(512, 6)):

```bash
DATA=/data/PAD-UFES-20

# Non-private image-only ("no meta": metadata block zeroed, same input dim)
python -m scripts.train_pad --mode sgd --lr 1e-3 --data_root $DATA

# Non-private image + metadata
python -m scripts.train_pad --mode sgd --lr 1e-3 --data_root $DATA --use_metadata

# DP-SGD (full gradient clipped to --max_grad_norm, default 1.0)
python -m scripts.train_pad --mode dpsgd --lr 1e-3 --epsilon 8 --data_root $DATA

# Feature DP-SGD (image vs. metadata gradient components clipped separately:
# --C_base 50.0 / --C_meta 1.0; noise scales with C_meta only)
python -m scripts.train_pad --mode feature_dpsgd --lr 1e-3 --epsilon 8 --data_root $DATA
```

`examples/run_pad.sh` runs the full grid (both non-private arms + DP/FDP
over ε ∈ {1, 2, 4, 8, 16, 32, 64}).

Mechanism details (`medmap/pad/privacy.py`): per-sample gradients are
computed functionally with `torch.func.vmap(grad(...))` over full-batch
steps (microbatches of 16 for memory). In `feature_dpsgd` the image-only
gradient component is obtained by re-running each example with the metadata
columns zeroed; the residual (full − base) is the metadata component. The
noise multiplier is calibrated to (ε, δ = 1/N_train^1.1) with the
dp-accounting RDP accountant over `--epochs` steps.

Each run prints a machine-parsable line
(`RESULT mode=... lr=... epsilon=... seed=... test_loss=... test_acc=...`),
full per-class test metrics, and appends per-subgroup test accuracies (age
bins, gender, family background, smoking/drinking, Fitzpatrick, cancer
history) to `--subgroup_csv` (default `subgroup_results.csv`).

## 4. Reproducibility notes

- `--split_seed 101` fixes the patient-level split for every reported run;
  `--seed` varies only model init, shuffling, and DP noise.
- The DP modes force `--use_metadata` (the base/meta gradient decomposition
  is meaningless with a zeroed metadata block).
- Feature extraction is deterministic (frozen backbone, eval mode), so the
  cache is reproducible from a fresh download.

## 5. Parameter reference

### `scripts/train_pad.py`

**Mode & privacy:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--mode` | required | `sgd` (non-private Adam), `dpsgd` (DP-SGD on the full gradient), `feature_dpsgd` (Feature DP-SGD: image vs. metadata components clipped separately). |
| `--epsilon` | — | Target privacy budget ε; required for the two DP modes, ignored for `sgd`. |
| `--delta` | `1/N_train^1.1` | Target δ for the accountant. |
| `--max_grad_norm` | 1.0 | Clip norm C for `dpsgd`. |
| `--C_base` | 50.0 | Image-component clip norm for `feature_dpsgd` (large = nearly unclipped). |
| `--C_meta` | 1.0 | Metadata-component clip norm for `feature_dpsgd`; the noise scale is `sigma * C_meta`, so only the metadata pays the privacy cost. |
| `--use_metadata` | off | Include the real one-hot metadata block (`sgd` only; the DP modes force it on — their gradient decomposition needs real metadata). Without it the metadata block is zeros of the same width. |

**Data & caching:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--data_root` | required | PAD-UFES-20 root (`metadata.csv` + `all_images/`). |
| `--download` | off | Download + extract the official Mendeley archive if images are missing. |
| `--img_dir` / `--meta_path` | `<data_root>/all_images` / `<data_root>/metadata.csv` | Explicit overrides. |
| `--cache_path` | `<data_root>/features_cache_patient_ohe.npz` | EfficientNet feature cache, shared across all runs/seeds. |

**Training & reporting:**

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--lr` | required | Adam learning rate. |
| `--epochs` | 100 | Full-batch training steps (also the accountant's composition length). |
| `--hidden_dim` | 512 | MLP hidden width. |
| `--seed` | 0 | Run seed: model init, shuffling, DP noise — but NOT the split. |
| `--split_seed` | 101 | Fixes the patient-level 80/10/10 split independently of `--seed`, so every run in a sweep sees the identical split. |
| `--subgroup_csv` | `subgroup_results.csv` | Long-format CSV that per-subgroup test accuracies are appended to. |
| `--setting_tag` | `<mode>_eps<ε>` | Label stored with the subgroup rows to identify the run. |
| `--gpu` | — | Sets `CUDA_VISIBLE_DEVICES` before torch loads. |

---

## 6. License & citations

**Data license:** Creative Commons **CC BY 4.0** (attribution).
**Backbone weights:** torchvision EfficientNet-B3, ImageNet-1k weights
(BSD-3-Clause torchvision; ImageNet terms apply to the pretraining data).

```bibtex
@article{pacheco2020padufes,
  title   = {PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones},
  author  = {Pacheco, Andre G C and Lima, Gustavo R and Salomao, Amanda S and Krohling, Breno and Biral, Igor P and de Angelo, Gabriel G and Alves Jr, F{\'a}bio C R and Esgario, Jos{\'e} G M and Simora, Alana C and Castro, Pedro B C and others},
  journal = {Data in Brief}, volume = {32}, pages = {106221}, year = {2020}
}

@inproceedings{tan2019efficientnet,
  title     = {EfficientNet: Rethinking model scaling for convolutional neural networks},
  author    = {Tan, Mingxing and Le, Quoc},
  booktitle = {International Conference on Machine Learning (ICML)}, pages = {6105--6114}, year = {2019}
}
```
