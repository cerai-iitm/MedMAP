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
[Reproducibility notes](#5-reproducibility-notes) · [Parameter reference](#6-parameter-reference) · [License & citations](#7-license--citations)

---

## 1. Download

ABIDE I preprocessed data is openly available (no registration) from the
[Preprocessed Connectomes Project (PCP)](http://preprocessed-connectomes-project.org/abide/)
— see also the [ABIDE homepage](https://fcon_1000.projects.nitrc.org/indi/abide/)
and the [phenotypic legend (PDF)](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf).

The benchmark uses the **C-PAC pipeline**, `func_preproc` derivative, with
no quality filtering, band-pass filtering, or global signal regression. The
canonical helper both downloads through nilearn and extracts reusable
functional-connectivity features:

```bash
bash scripts/setup_workspace.sh
bash scripts/prepare_abide.sh
```

The raw derivatives are approximately 100+ GB and extraction takes several
hours. Raw PCP files and the phenotype CSV are written to
`data/raw/abide/`; the latter is expected at
`data/raw/abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv`.

## 2. Feature extraction

The experiments consume a small precomputed feature directory
(`data/processed/abide/fc-features/`, about 12 MB), not raw NIfTIs. The same
`prepare_abide.sh` command writes `X_conn.npy`, `y.npy`, and `meta.json` there
(MSDL atlas → 39 regions → 741 vectorized correlation features per subject).
Pass extractor overrides directly, for example:

```bash
bash scripts/prepare_abide.sh --t-r 2.0
```

Phenotypes are preprocessed at train time (shared logic in both backends):
the 11 recommended sensitive columns (from the packaged
`recommended_feature_lists.json`) are scaled to [-1, 1] with **fixed domain
bounds** (e.g. age 5.13–64, IQ 49–149; missing → 0) and categoricals are
one-hot encoded with an all-zero row for missing values; all fitting happens
on the training split only. `SITE_ID` is never used as an input feature.

## 3. Classic backend

Run the non-private baseline first for the selected protocol, then invoke one
privacy arm per epsilon. The launcher supplies all canonical paths and forwards
additional CLI options:

```bash
# IID: stratified random protocol; non-IID: SITE_ID-disjoint protocol
SPLIT=non_iid METHOD=lbfgs bash examples/run_abide_classic.sh

METHOD=full_dp TARGET_EPSILON=8 \
  bash examples/run_abide_classic.sh --iterations 3000
METHOD=feat_dp TARGET_EPSILON=8 \
  bash examples/run_abide_classic.sh --iterations 3000

python -m scripts.summarize_abide \
  --data-dir data/raw/abide \
  --output-root outputs/abide \
  --split non_iid --stages all
```

Repeat the two private commands with other `TARGET_EPSILON` values for a
sweep. Results land in:

```text
outputs/abide/{iid,non_iid}_fixed_split/
├── non_priv/
├── full_dp/e<EPS>/seed<N>/
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

- The baseline persists its selected split and tuned C under `non_priv/`.
  Current DP modules deterministically reconstruct their protocol from the
  tuning seed; if the baseline selects a C other than the documented default,
  pass it explicitly with `--tuned-c` so all arms use that value.
- The ε sweep is one target per invocation (matching how the original
  experiments were run); each run records its achieved ε and σ in
  `dp_accounting_*.json` files.
- The non-IID protocol is the harder, more realistic setting: test subjects
  come from acquisition sites never seen during training.
- Classic defaults per split: iterations 2000 (IID) / 3000 (non-IID),
  C 0.1 / 0.3, Feature-DP lr 0.08 / 0.16.

## 6. Parameter reference

### `scripts/extract_abide_features.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | required | ABIDE PCP root; raw data is downloaded here if missing. |
| `--out-dir` | `<data-dir>/fc_features` | Where `X_conn.npy` / `y.npy` / `meta.json` are written. |
| `--t-r` | 2.0 | Repetition time passed to the masker. |

### `scripts/train_abide_classic.py` (classic backend)

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--split` | required | `iid` (stratified random 70/10/20) or `non_iid` (SITE_ID-disjoint: test sites unseen in training). |
| `--method` | required | `lbfgs` (non-private baseline; also creates the fixed split and tunes C — run first), `full_dp` (full-batch DP gradient descent), `feat_dp` (Feature-DP). |
| `--data-dir` | required | ABIDE PCP root (contains `ABIDE_pcp/` and `fc_features/`). |
| `--output-root` | `<data-dir>/outputs` | Root of the results tree. |
| `--fc-dir` | `<data-dir>/fc_features` | Location of the precomputed FC features. |
| `--phenotypic-csv` | `<data-dir>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv` | Phenotypic records. |
| `--feature-json` | data dir, then packaged copy | `recommended_feature_lists.json` defining the 11 sensitive phenotype columns. |
| `--epsilon` | 64 iid / 1 non_iid (script defaults) | Target ε for `full_dp` / `feat_dp`; run one ε per invocation to sweep. |
| `--iterations` | 2000 iid / 3000 non_iid | Full-batch gradient-descent steps (also the accountant's composition length). |
| `--seeds` | 81–90 iid / 21–30 non_iid | Comma-separated final-run seeds (only the DP noise varies across them; the split is fixed). |
| `--tuning-seed` | 81 iid / 21 non_iid | Seed used to create the fixed split and tune hyperparameters once. |
| `--lr-grid` | method defaults | Learning-rate grid searched on the tuning seed (full_dp auto-expands at grid endpoints). |
| `--clip-grid` | 0.6455 iid / 0.3228 non_iid | Private clip-norm grid for `feat_dp` (values are quantiles of the observed private-residual gradient norms). |
| `--tuned-c` | 0.1 iid / 0.3 non_iid | Logistic-regression C used by the DP arms (originally selected by the `lbfgs` arm's validation grid search). |
| `--fetch-raw` | off | Run nilearn's `fetch_abide_pcp` first (downloads raw derivatives if missing). |

### `scripts/summarize_abide.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--data-dir` / `--output-root` | required / `<data-dir>/outputs` | Locates the results tree to aggregate. |
| `--split` | required | Which tree to summarize (`iid` or `non_iid`). |
| `--stages` | `all` | Any of `tables` (cross-ε summary CSVs), `plots` (privacy-utility curves), `subgroups` (per-subgroup metric tables, non-IID), `latex` (LaTeX table export, non-IID). |

### `scripts/train_abide_hf.py` (HF backend — environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `ABIDE_DATA_DIR` | required | ABIDE PCP root. |
| `ABIDE_FC_DIR` | `<ABIDE_DATA_DIR>/fc_features` | FC feature directory. |
| `FEATURE_JSON` / `PHENOTYPIC_CSV` | packaged / default location | Overrides for the sensitive-feature list and the phenotypic CSV. |
| `INCLUDE_FMRI` / `INCLUDE_SENSITIVE` | 1 / 1 | Modality ablations (at least one must be on; DP forces both on). |
| `INCLUDE_SITE_ID_AS_FEATURE` | 0 | Optionally adds SITE_ID as an input feature (off in all reported runs). |
| `USE_DP` | 0 | Switch differential privacy on. |
| `FEATURE_DP` | 0 | With `USE_DP=1`: 1 = Feature-DP (fMRI public, phenotypes private), 0 = full DP-SGD. |
| `TARGET_EPS` / `CLIP_NORM` / `PUBLIC_CLIP` | 8.0 / 1.0 / 50.0 | Privacy budget, private clip norm, public clip norm (Feature-DP). |
| `SEED` | 81 | Run seed. |
| `EPOCHS` / `BATCH` / `GRAD_ACCUM` | 100 / 64 / 1 | Optimization schedule (under DP the effective batch is `BATCH*GRAD_ACCUM`). |
| `LR` / `WEIGHT_DECAY` / `WARMUP_RATIO` | 1e-3 / 1e-4 / 0.0 | AdamW settings. |
| `EVAL_STRATEGY`, `EVAL_STEPS`, `SAVE_STRATEGY`, `SAVE_STEPS`, `LOGGING_STRATEGY`, `LOGGING_STEPS` | epoch / 500 / epoch / 500 / steps / 10 | Evaluation, checkpointing and logging cadence. |
| `OUT_DIR` / `RUN_NAME` | auto-generated from the setting | Output directory and W&B run name. |
| `WANDB_MODE` | online | `disabled` runs without W&B. |

---

## 7. License & citations

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
