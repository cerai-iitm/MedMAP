# ABIDE I: fMRI functional connectivity + phenotypes (logistic regression)

Binary autism vs. control classification on ABIDE I. The fMRI functional
connectivity features are the public modality; the phenotypic record (age,
sex, IQ scores, handedness, medication status, ...) is the private modality.

Two interchangeable backends implement the same task:

| Backend | Entry point | What it is |
| --- | --- | --- |
| **classic** | `scripts/train_abide_classic.py` | Faithful packaging of the original research scripts: sklearn `LogisticRegression` (L-BFGS) baseline + hand-written NumPy/Numba full-batch DP-GD and Feature-DP, with fixed IID or site-disjoint (non-IID) splits, per-arm hyperparameter tuning, and rich per-seed outputs. Produces the reported results. |
| **hf** | `scripts/train_abide_hf.py` | The same logistic-regression model implemented as a Hugging Face `PreTrainedModel` (single `nn.Linear`), trained with AdamW through the shared trainer stack and the modified dp-transformers fork — the same DP machinery as MedGemma/SmolVLM, and a template for swapping in any model. |

Contents: [Download](#1-download) · [Feature extraction](#2-feature-extraction) ·
[Classic backend](#3-classic-backend) · [HF backend](#4-hf-backend) ·
[Reproducibility notes](#5-reproducibility-notes) · [License & citations](#6-license--citations)

---

## 1. Download

ABIDE I preprocessed data is openly available (no registration) from the
[Preprocessed Connectomes Project (PCP)](http://preprocessed-connectomes-project.org/abide/)
— see also the [ABIDE homepage](https://fcon_1000.projects.nitrc.org/indi/abide/)
and the [phenotypic legend (PDF)](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf).

The benchmark uses the **C-PAC pipeline**, `func_preproc` derivative, with
no quality filtering, band-pass filtering, or global signal regression —
exactly what nilearn's `fetch_abide_pcp` downloads. Choose a root directory
(`$ABIDE_DATA_DIR` below); the raw derivatives are ~100+ GB:

```bash
export ABIDE_DATA_DIR=/data/abide/ABIDE_PCP_full

python - <<'EOF'
import os
from nilearn import datasets
datasets.fetch_abide_pcp(
    data_dir=os.environ["ABIDE_DATA_DIR"],
    pipeline="cpac",
    derivatives=["func_preproc"],
    quality_checked=False,
    band_pass_filtering=False,
    global_signal_regression=False,
)
EOF
```

This also downloads the phenotypic CSV to
`$ABIDE_DATA_DIR/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv`.

## 2. Feature extraction

The experiments consume a small precomputed feature directory
(`$ABIDE_DATA_DIR/fc_features/`, ~12 MB), not the raw NIfTIs. Generate it
once (several hours; MSDL atlas → 39 regions → 741 vectorized correlation
features per subject):

```bash
python -m scripts.extract_abide_features --data-dir "$ABIDE_DATA_DIR"
# writes fc_features/{X_conn.npy (1035x741), y.npy, meta.json}
```

Phenotypes are preprocessed at train time (shared logic in both backends):
the 11 recommended sensitive columns (from the packaged
`recommended_feature_lists.json`) are scaled to [-1, 1] with **fixed domain
bounds** (e.g. age 5.13–64, IQ 49–149; missing → 0) and categoricals are
one-hot encoded with an all-zero row for missing values; all fitting happens
on the training split only. `SITE_ID` is never used as an input feature.

## 3. Classic backend

Run the non-private baseline first — it **creates and persists the fixed
split** and tunes the global regularization constant C (validation accuracy
over C ∈ {1e-4 ... 10}) that the DP arms reuse:

```bash
# IID:      stratified random 70/10/20 (seeds 81-90, tuning seed 81)
# non-IID:  SITE_ID-disjoint via StratifiedGroupKFold — train/val/test share
#           no acquisition sites (seeds 21-30, tuning seed 21)
python -m scripts.train_abide_classic --split non_iid --method lbfgs \
    --data-dir "$ABIDE_DATA_DIR"

# Full DP-GD and Feature-DP, one target epsilon per invocation:
for EPS in 1 2 4 8 16 32 64; do
  python -m scripts.train_abide_classic --split non_iid --method full_dp \
      --data-dir "$ABIDE_DATA_DIR" --epsilon $EPS
  python -m scripts.train_abide_classic --split non_iid --method feat_dp \
      --data-dir "$ABIDE_DATA_DIR" --epsilon $EPS
done

# Summary tables + privacy-utility plots (+ subgroup/LaTeX tables for non_iid)
python -m scripts.summarize_abide --data-dir "$ABIDE_DATA_DIR" --split non_iid --stages all
```

`examples/run_abide_classic.sh` wraps the whole sequence. Results land in

```
<output-root>/{iid,non_iid}_fixed_split/
├── non_priv/                 # baseline + persisted fixed split + global C
├── full_dp/e<EPS>/seed<N>/   # per-seed DP-GD outputs + accounting JSONs
├── feat_dp/e<EPS>/seed<N>/
└── *_all_eps_vertical_summary.csv, privacy_utility_plots/, subgroup_metrics/
```

Method details (all defaults reproduce the reported runs):

- **full_dp**: full-batch gradient descent; each example's *entire* gradient
  is clipped to norm 1.0, Gaussian noise is added to the sum, and the L2
  term (from the tuned C) is applied after the noise. The learning rate is
  tuned once on the tuning seed over a grid with automatic endpoint
  expansion; the other 9 seeds reuse it (only the noise RNG changes).
- **feat_dp** (Feature-DP): the model keeps separate fMRI and phenotype
  weight blocks; per example, the *public* (fMRI-only) gradient is
  accumulated exactly, while only the *private phenotype residual* gradient
  is clipped (grid default 0.6455 IID / 0.3228 non-IID — quantiles of the
  observed residual-norm distribution) and noised.
- **Accounting**: RDP (Google dp-accounting) with δ = n^-1.1 and the noise
  multiplier solved by Brent root-finding for the target ε; σ is calibrated
  separately for the tuning model (n_train) and the final refit
  (n_train+val).
- Numba JIT accelerates the kernels when available; a pure-NumPy fallback
  runs automatically otherwise.
- Overrides: `--seeds`, `--tuning-seed`, `--iterations`, `--lr-grid`,
  `--clip-grid`, `--tuned-c`, `--output-root`, `--fetch-raw`.

## 4. HF backend

Environment-variable driven (same conventions as the SmolVLM entry point):

```bash
export ABIDE_DATA_DIR=/data/abide/ABIDE_PCP_full

# Non-private modality ablations (fmri_only / sensitive_only / fmri+sensitive)
bash examples/run_abide_hf_non_private.sh

# Full DP-SGD and Feature-DP through the dp-transformers fork
TARGET_EPS=8.0 bash examples/run_abide_hf_private.sh
```

Key variables: `SEED` (81), `EPOCHS` (100), `BATCH` (64), `LR` (1e-3),
`USE_DP`, `FEATURE_DP` (1 = Feature-DP), `TARGET_EPS`, `CLIP_NORM` (1.0),
`PUBLIC_CLIP` (50.0), `INCLUDE_FMRI` / `INCLUDE_SENSITIVE`,
`WANDB_MODE=disabled` to turn W&B off. The split here is the stratified IID
70/10/20 split used by the original MedMAP implementation.

## 5. Reproducibility notes

- Fixed splits are created once from the tuning seed and persisted
  (`fixed_train_idx.npy`, audit JSONs, and full CSV dumps under
  `non_priv/`); the DP arms deterministically re-derive the identical split.
- The ε sweep is one target per invocation (matching how the original
  experiments were run); each run records its achieved ε and σ in
  `dp_accounting_*.json` files.
- The non-IID protocol is the harder, more realistic setting: test subjects
  come from acquisition sites never seen during training.
- Classic defaults per split: iterations 2000 (IID) / 3000 (non-IID),
  C 0.1 / 0.3, Feature-DP lr 0.08 / 0.16.

## 6. License & citations

**Data terms:** ABIDE I is shared for non-commercial research under a
Creative Commons **BY-NC-SA** license (attribution, non-commercial,
share-alike); users must not attempt to re-identify participants. See the
[ABIDE usage notes](https://fcon_1000.projects.nitrc.org/indi/abide/).

```bibtex
@article{dimartino2014abide,
  title   = {The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism},
  author  = {Di Martino, Adriana and Yan, Chao-Gan and Li, Qingyang and Denio, Erin and Castellanos, Francisco X and Alaerts, Kaat and Anderson, Jeffrey S and Assaf, Michal and Bookheimer, Susan Y and Dapretto, Mirella and others},
  journal = {Molecular Psychiatry}, volume = {19}, number = {6}, pages = {659--667}, year = {2014}
}

@article{craddock2013pcp,
  title   = {The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives},
  author  = {Craddock, Cameron and Benhajali, Yassine and Chu, Carlton and Chouinard, Francois and Evans, Alan and Jakab, Andr{\'a}s and Khundrakpam, Budhachandra Singh and Lewis, John David and Li, Qingyang and Milham, Michael and others},
  journal = {Frontiers in Neuroinformatics}, volume = {7}, year = {2013}
}

% MSDL atlas used for time-series extraction
@inproceedings{varoquaux2011msdl,
  title     = {Multi-subject dictionary learning to segment an atlas of brain spontaneous activity},
  author    = {Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Pedregosa, Fabian and Michel, Vincent and Thirion, Bertrand},
  booktitle = {Information Processing in Medical Imaging (IPMI)}, pages = {562--573}, year = {2011}
}

% Tooling
@article{abraham2014nilearn,
  title   = {Machine learning for neuroimaging with scikit-learn},
  author  = {Abraham, Alexandre and Pedregosa, Fabian and Eickenberg, Michael and Gervais, Philippe and Mueller, Andreas and Kossaifi, Jean and Gramfort, Alexandre and Thirion, Bertrand and Varoquaux, Ga{\"e}l},
  journal = {Frontiers in Neuroinformatics}, volume = {8}, year = {2014}
}
```
