from __future__ import annotations
# Dataset creation
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

abide = datasets.fetch_abide_pcp(
    data_dir="/storage/common/datasets/ABIDE_PCP_full",
    pipeline="cpac",
    derivatives=["func_preproc"],
    quality_checked=False,
    band_pass_filtering=False,
    global_signal_regression=False,
)

# atlas = datasets.fetch_atlas_msdl(data_dir="/storage/common/datasets/ABIDE_PCP")

# masker = NiftiMapsMasker(
#     maps_img=atlas.maps,
#     standardize="zscore_sample",
#     detrend=True,
#     low_pass=0.1,
#     high_pass=0.01,
#     t_r=2.0, # replace with subject/site-specific TR if you handle it explicitly
# )

# time_series = [masker.fit_transform(img) for img in abide.func_preproc]

# conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
# X_conn = conn.fit_transform(time_series)

# y = (abide.phenotypic["DX_GROUP"] == 1).astype(int) # check coding in your local version
import numpy as np
from pathlib import Path
import json

save_dir = Path("/storage/common/datasets/ABIDE_PCP_full/fc_features")
save_dir.mkdir(parents=True, exist_ok=True)

X_conn = np.load(save_dir / "X_conn.npy")
y = np.load(save_dir / "y.npy")

with open(save_dir / "meta.json", "r") as f:
    meta = json.load(f)

print("X_conn shape:", X_conn.shape)
print("y shape:", y.shape)
print("Metadata:", meta)
# Print min and max values in X_conn to check for any anomalies
print("X_conn min value:", np.min(X_conn))
print("X_conn max value:", np.max(X_conn))
abide.phenotypic

# Metrics FDP Multiseed per Target Epsilon
#!/usr/bin/env python3
"""
Feature-DP logistic regression on ABIDE fMRI + phenotypic features.

Run:
    python abide_feature_dp_logreg_two_modes_lr_clip_tuning.py

This script keeps the same dataset loading, cleaning, feature selection,
domain scaling, preprocessing, IID train/validation/test splitting, plotting,
and subgroup reporting style as the previous ABIDE logistic-regression scripts.

Main change:
    Instead of full DP-SGD over all features, this implements Feature DP:

        public features  = fMRI FC features
        private features = selected phenotypic features

Feature-DP update:
    public gradient:
        grad_pub = grad f(theta; x_public, private=0, y)

    private residual gradient:
        grad_priv = grad f(theta; x_public, x_private, y)
                    - grad f(theta; x_public, private=0, y)

    Only the private residual gradient is clipped and noised.

Final training policy:
    1. Tune learning rate and private clip norm exactly once using LRC_SEED.
    2. Reuse the selected learning rate and private clip norm for every seed
       in SEED_LIST.
    3. For each seed, refit the final model on train + validation using the
       fixed hyperparameters and evaluate once on the held-out test set.

Feature combinations:
    1. Feature DP without anat & func

Important:
    X_conn must already exist in the Python session, or set X_CONN_PATH below.
"""



import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize as opt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


# =========================================================
# Optional acceleration imports
# =========================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False


# =========================================================
# Dataset / seed configuration
# =========================================================
class SeedConfig:
    """Seed-related configuration.

    LRC_SEED is used exactly once for hyperparameter tuning over
    (learning_rate, private_clip_norm). The selected pair is then reused for
    every seed in SEED_LIST. LRC_SEED does not need to be present in SEED_LIST.
    """

    SEED_LIST = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    LRC_SEED = 81

SEED_LIST = [int(s) for s in SeedConfig.SEED_LIST]
LRC_SEED = int(SeedConfig.LRC_SEED)

# Backward-compatible alias. All multi-seed code passes the seed explicitly.
SEED = LRC_SEED

TARGET_EPSILON = 64.0
ITERATIONS = 2000

# Private residual gradient is always clipped and noised.
# This value is kept as a backward-compatible fallback.
# The actual private clip norm used by the final model is selected from
# PRIVATE_CLIP_NORM_GRID during validation tuning.
CLIP_NORM_PRIVATE = 1.0

# Fixed private clip-norm grid requested for this experiment.
# The CSV-based quantile loader is kept for backward compatibility, but by
# default the code uses this fixed grid.
USE_FIXED_PRIVATE_CLIP_NORM_GRID = True
PRIVATE_CLIP_NORM_GRID = [
    0.6455,
]

# Backward-compatible quantile-grid settings. Used only if
# USE_FIXED_PRIVATE_CLIP_NORM_GRID = False.
PRIVATE_CLIP_NORM_QUANTILES_FOR_GRID = [10, 25, 50, 75]
FEATURE_NORM_QUANTILE_CSV_PATH = Path("/storage/common/datasets/ABIDE_PCP_full/outputs/feature_norm_analysis/feature_norm_quantiles_1_to_100.csv")
PRIVATE_CLIP_NORM_GRID_FALLBACK = [CLIP_NORM_PRIVATE]

# Logistic-regression regularization hyperparameter set by user.
# Objective:
#     mean_i BCE_i + (lambda / 2) * (||w_public||_2^2 + ||w_private||_2^2)
# where:
#     lambda = 1 / (tuned_C * n_train)
# Separate LogisticRegression C values for the two Feature-DP experiments.
TUNED_C_WithOut_AnatFunc = 0.1
# TUNED_C_With_AnatFunc = 1
TUNED_FIT_INTERCEPT = True

# Learning-rate grid for hyperparameter configuration search.
LEARNING_RATES_INITIAL = [0.08]
MAX_ENDPOINT_EXPANSIONS = 0
ENDPOINT_EXPANSION_FACTOR = 2.0

TARGET_COL = "DX_GROUP"

EPSILON_OUTPUT_DIR = Path(
    f"/storage/common/datasets/ABIDE_PCP_full/outputs/iid_fixed_split_test/feat_dp/e{int(TARGET_EPSILON)}"
)


def get_seed_output_dir(seed: int) -> Path:
    return EPSILON_OUTPUT_DIR / f"seed{int(seed)}"


# Backward-compatible alias. Runtime code passes seed-specific output dirs explicitly.
OUTPUT_DIR = get_seed_output_dir(SEED)
EPSILON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path("/mnt/data/Phenotypic_V1_0b_preprocessed1.csv")

FEATURE_JSON_CANDIDATES = [
    Path("/mnt/data/recommended_feature_lists.json"),
    Path("/storage/common/datasets/ABIDE_PCP_full/abide_feature_selection_outputs/recommended_feature_lists.json"),
]

# If X_conn already exists in the Python session, it will be used.
# Otherwise set this to a .npy or .npz file containing X_conn.
X_CONN_PATH: Optional[Path] = None

USE_SITE_ID_AS_FEATURE = False
FC_IS_CORRELATION = True

# Overall split sizes:
# train = 70%, validation = 10%, test = 20%
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20

# User-selectable preprocessing for phenotype_other_numeric_cols.
# Allowed:
#   "standardize_zero_fill"
#   "minmax_minus1_1_zero_fill"
#   "zero_fill_only"
PHENOTYPE_OTHER_NUMERIC_MODE = "minmax_minus1_1_zero_fill"

# Optional fixed-split source.
#
# When False:
#     The code creates exactly one train/validation/test split using LRC_SEED
#     through the same StratifiedShuffleSplit logic used by the non-private
#     reference script, and reuses that fixed split for every seed in SEED_LIST.
#
# When True:
#     The code loads the already-dumped non-private fixed split CSVs below and
#     recovers the corresponding row indices in df_model_raw / X_conn_model.
#     The Feature-DP preprocessing/training code then uses those exact samples.
USE_PREPROCESS_DUMP = False
PREPROCESS_DUMP_DIR = Path(
    "/storage/common/datasets/ABIDE_PCP_full/outputs/iid_fixed_split/non_priv"
)
PREPROCESS_DUMP_TRAIN_CSV = PREPROCESS_DUMP_DIR / "train_fixed.csv"
PREPROCESS_DUMP_VAL_CSV = PREPROCESS_DUMP_DIR / "val_fixed.csv"
PREPROCESS_DUMP_TEST_CSV = PREPROCESS_DUMP_DIR / "test_fixed.csv"
PREPROCESS_DUMP_TRAIN_VAL_CSV = PREPROCESS_DUMP_DIR / "train_val_fixed.csv"

# Candidate columns used only to identify rows when USE_PREPROCESS_DUMP=True.
# The loader first tries stable ABIDE identifiers such as SUB_ID. If they are
# absent, it tries explicit saved model-index columns. If those are absent too,
# it can recover rows from dumped fc_***** columns by rounded keys.
DUMP_IDENTIFIER_COLUMN_CANDIDATES = [
    "SUB_ID",
    "subject_id",
    "SUBJECT_ID",
    "participant_id",
    "participant",
    "FILE_ID",
    "file_id",
]
DUMP_INDEX_COLUMN_CANDIDATES = [
    "row_index_after_valid_mask",
    "df_model_index",
    "df_model_raw_index",
    "model_index",
    "model_idx",
    "global_index",
    "global_idx",
    "orig_index",
    "original_index",
    "__index__",
    "row_index",
    "row_idx",
    "index",
    "Unnamed: 0",
]
DUMP_FC_KEY_MAX_COLUMNS = 64
DUMP_FC_KEY_ROUND_DECIMALS = 10


# =========================================================
# Feature-DP configuration
# =========================================================
DELTA_POWER = -1.1  # delta = n ** DELTA_POWER

# If None, noise multiplier is solved using RDPAccountant.
NOISE_MULTIPLIER: Optional[float] = None

# Public-gradient clipping can be disabled by setting this to None.
CLIP_NORM_PUBLIC: Optional[float] = None

FULL_BATCH = True

# Keep same L2 logic as previous code.
# For Feature DP, use add_after_noise.
L2_REG_HANDLING = "add_after_noise"

SELECTION_METRIC = "accuracy"  # {"accuracy", "balanced_accuracy", "roc_auc", "log_loss"}
SELECTION_TIE_TOL = 1e-12

RECORD_EVERY_N_STEPS = 1
SHOW_PLOTS = True


class RuntimeConfig:
    """Runtime options that apply across all seeds."""

    # Numba accelerates the repeated full-batch Feature-DP update loop.
    USE_NUMBA_JIT = True

    # Parallelization distributes independent hyperparameter configurations,
    # where each configuration is one (learning_rate, clip_norm_private) pair.
    PARALLELIZE_HYPERPARAMETER_CONFIGS = True
    N_PARALLEL_JOBS = 128
    PARALLEL_BACKEND = "loky"


USE_NUMBA_JIT = bool(RuntimeConfig.USE_NUMBA_JIT)
PARALLELIZE_HYPERPARAMETER_CONFIGS = bool(RuntimeConfig.PARALLELIZE_HYPERPARAMETER_CONFIGS)
N_PARALLEL_JOBS = int(RuntimeConfig.N_PARALLEL_JOBS)
PARALLEL_BACKEND = str(RuntimeConfig.PARALLEL_BACKEND)


# =========================================================
# Feature-DP feature-combination experiments
# =========================================================
EXPERIMENTS = [
    {
        "experiment_key": "feature_dp_without_anat_func",
        "experiment_name": "Feature DP without anat & func",
        "tuned_C": TUNED_C_WithOut_AnatFunc,
        "feature_group_keys": [
            "recommended_initial_numeric_or_binary",
            "recommended_initial_categorical",
            "optional_categorical_with_caution",
        ],
    },
    # {
    #     "experiment_key": "feature_dp_with_anat_func",
    #     "experiment_name": "Feature DP with anat & func",
    #     "tuned_C": TUNED_C_With_AnatFunc,
    #     "feature_group_keys": [
    #         "recommended_initial_numeric_or_binary",
    #         "recommended_initial_categorical",
    #         "optional_categorical_with_caution",
    #         "anat_func_numeric_or_binary",
    #     ],
    # },
]


# =========================================================
# Constants copied from reference preprocessing workflow
# =========================================================
ALLOWED_FEATURE_GROUP_KEYS = {
    "recommended_initial_numeric_or_binary",
    "anat_func_numeric_or_binary",
    "recommended_initial_categorical",
    "optional_categorical_with_caution",
    "drop_leakage",
}

ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES = {
    "standardize_zero_fill",
    "minmax_minus1_1_zero_fill",
    "zero_fill_only",
}

DOMAIN_FIXED_BOUNDS = {
    "AGE_AT_SCAN": (5.13, 64.0),
    "SEX": (1.0, 2.0),
    "HANDEDNESS_SCORES": (-100.0, 100.0),
    "CURRENT_MED_STATUS": (0.0, 1.0),
    "EYE_STATUS_AT_SCAN": (1.0, 2.0),
}

IQ_GLOBAL_BOUNDS = {
    "FIQ": (49.0, 149.0),
    "VIQ": (45.0, 156.0),
    "PIQ": (53.0, 149.0),
}

DOMAIN_SCALED_NUMERIC_COLS = set(DOMAIN_FIXED_BOUNDS) | set(IQ_GLOBAL_BOUNDS)


# =========================================================
# Subgroup reporting configuration
# =========================================================
SUBGROUP_EVAL_FEATURES = [
    "SITE_ID",
    "AGE_AT_SCAN",
    "EYE_STATUS_AT_SCAN",
    "SEX",
    "FIQ",
    "PIQ",
    "VIQ",
    "CURRENT_MED_STATUS",
    "HANDEDNESS_CATEGORY",
    "FIQ_TEST_TYPE",
    "PIQ_TEST_TYPE",
    "VIQ_TEST_TYPE",
]

SENSIBLE_NUMERIC_BINS = {
    "AGE_AT_SCAN": {
        "bins": [-np.inf, 10, 15, 20, 30, 40, np.inf],
        "labels": ["<10", "10-14", "15-19", "20-29", "30-39", "40+"],
    },
    "FIQ": {
        "bins": [-np.inf, 85, 100, 115, 130, np.inf],
        "labels": ["<85", "85-99", "100-114", "115-129", "130+"],
    },
    "PIQ": {
        "bins": [-np.inf, 85, 100, 115, 130, np.inf],
        "labels": ["<85", "85-99", "100-114", "115-129", "130+"],
    },
    "VIQ": {
        "bins": [-np.inf, 85, 100, 115, 130, np.inf],
        "labels": ["<85", "85-99", "100-114", "115-129", "130+"],
    },
}

MISSING_GROUP_LABEL = "Missing"

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sns.set_theme(style="whitegrid")


# =========================================================
# Dataclasses
# =========================================================
@dataclass
class PreparedGlobalData:
    df: pd.DataFrame
    df_model_raw: pd.DataFrame
    df_model: pd.DataFrame
    X_conn_model: np.ndarray
    y: np.ndarray
    feature_groups: dict
    feature_json_path: Path
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    split_name: str


@dataclass
class FeatureBlockData:
    public_df: pd.DataFrame
    private_df: pd.DataFrame
    fc_cols: list[str]
    phenotype_domain_scaled_cols: list[str]
    phenotype_other_numeric_cols: list[str]
    phenotype_categorical_cols: list[str]
    selected_phenotype_cols: list[str]
    missing_cols: list[str]


# =========================================================
# Custom transformers: same preprocessing logic as reference
# =========================================================
class ZeroFillTransformer(BaseEstimator, TransformerMixin):
    """
    Replace NaN / +/-inf with 0.0.
    Useful after deterministic domain scaling or for already-bounded FC features.
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class StandardizeThenZeroFill(BaseEstimator, TransformerMixin):
    """
    Standardize numeric features using train-set mean/std computed on observed values,
    then replace missing / non-finite transformed entries with 0.
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)

        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)

        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        self.scale_ = np.where(
            np.isfinite(self.scale_) & (self.scale_ > 0),
            self.scale_,
            1.0,
        )

        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = (X_arr - self.mean_) / self.scale_
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class MinMaxMinusOneOneThenZeroFill(BaseEstimator, TransformerMixin):
    """
    Min-max scale each numeric feature to [-1, 1] using train-set observed values,
    then replace missing / non-finite transformed entries with 0.
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)

        self.data_min_ = np.nanmin(X_arr, axis=0)
        self.data_max_ = np.nanmax(X_arr, axis=0)

        self.data_min_ = np.where(np.isfinite(self.data_min_), self.data_min_, 0.0)
        self.data_max_ = np.where(np.isfinite(self.data_max_), self.data_max_, 0.0)
        self.range_ = self.data_max_ - self.data_min_
        self.valid_range_ = np.isfinite(self.range_) & (self.range_ > 0)

        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = np.zeros_like(X_arr, dtype=np.float64)

        valid_cols = self.valid_range_
        if np.any(valid_cols):
            X_scaled[:, valid_cols] = (
                2.0
                * (X_arr[:, valid_cols] - self.data_min_[valid_cols])
                / self.range_[valid_cols]
                - 1.0
            )

        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class MissingAsAllZeroOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode observed categories only.
    Missing values and unseen categories map to an all-zero block.
    """

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = [str(c) for c in X_df.columns]
        self.categories_ = []

        for j in range(X_df.shape[1]):
            s = X_df.iloc[:, j]
            observed = s[~pd.isna(s)]
            cats = pd.Index(observed.astype(str)).unique().tolist()
            cats = sorted(cats)
            self.categories_.append(cats)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        n = len(X_df)
        blocks = []

        for j, cats in enumerate(self.categories_):
            block = np.zeros((n, len(cats)), dtype=np.float64)
            if len(cats) == 0:
                blocks.append(block)
                continue

            s = X_df.iloc[:, j]
            valid_mask = ~pd.isna(s)
            if valid_mask.any():
                values = s.astype(str).to_numpy()
                valid = valid_mask.to_numpy()

                for k, cat in enumerate(cats):
                    block[valid & (values == cat), k] = 1.0

            blocks.append(block)

        if not blocks:
            return np.zeros((n, 0), dtype=np.float64)

        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)],
            )

        names = []
        for feat, cats in zip(input_features, self.categories_):
            names.extend([f"{feat}_{cat}" for cat in cats])

        return np.asarray(names, dtype=object)


# =========================================================
# Loading and cleaning helpers: same as reference logic
# =========================================================
def load_abide_dataframe(csv_path: Path) -> pd.DataFrame:
    try:
        df = abide.phenotypic.copy()  # noqa: F821
        print("Using abide.phenotypic from current session.")
    except Exception:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataframe from CSV: {csv_path}")

    return df


def load_fc_matrix(x_conn_path: Optional[Path]) -> np.ndarray:
    try:
        X_conn_np = np.asarray(X_conn, dtype=np.float64)  # noqa: F821
        print("Using X_conn from current session.")
        return X_conn_np
    except Exception:
        pass

    if x_conn_path is None:
        raise ValueError(
            "X_conn is not available in the current Python session. "
            "Please set X_CONN_PATH to a .npy or .npz file."
        )

    if x_conn_path.suffix == ".npy":
        X_conn_np = np.load(x_conn_path)
    elif x_conn_path.suffix == ".npz":
        payload = np.load(x_conn_path)
        if "X_conn" in payload:
            X_conn_np = payload["X_conn"]
        else:
            first_key = list(payload.keys())[0]
            X_conn_np = payload[first_key]
    else:
        raise ValueError("X_CONN_PATH must point to a .npy or .npz file.")

    print(f"Loaded X_conn from file: {x_conn_path}")
    return np.asarray(X_conn_np, dtype=np.float64)


def clean_abide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_map = {
        -9999: np.nan,
        "-9999": np.nan,
        "": np.nan,
        " ": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "n/a": np.nan,
        "`": np.nan,
    }

    df = df.replace(missing_map)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def resolve_feature_json_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find recommended_feature_lists.json in any expected location."
    )


def load_feature_groups(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def normalize_handedness_category(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()

    if s in {"", "NAN", "NONE", "-9999", "`"}:
        return np.nan
    if s in {"R", "RIGHT", "RIGHT_HANDED", "1"}:
        return "R"
    if s in {"L", "LEFT", "LEFT_HANDED", "2"}:
        return "L"
    if s in {"AMBI", "AMBIDEXTROUS", "MIXED", "L->R", "3"}:
        return "Ambi"

    return s


def normalize_test_type(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()

    if s in {"", "NAN", "NONE", "-9999", "`"}:
        return np.nan

    s = s.replace(" ", "_").replace("-", "_")

    alias_map = {
        "RAVENS": "RAVEN",
        "PPVT": "PPVT",
        "WISC4": "WISC_IV",
        "WISC_IV_FULL": "WISC_IV_FULL",
        "WISC_III_DUTCH": "WISC_III_DUTCH",
        "WASI_II": "WASI_II",
        "WAIS_III": "WAIS_III",
        "WAIS_IV": "WAIS_IV",
        "WAIS_IV_NL": "WAIS_IV_NL",
        "WISC_III": "WISC_III",
        "WISC_IV": "WISC_IV",
        "WISC_V": "WISC_V",
        "WPPSI_III": "WPPSI_III",
        "DAS_II_SA": "DAS_II_SA",
        "KBITS_2": "KBIT_2",
    }

    return alias_map.get(s, s)


# =========================================================
# Feature-list helpers
# =========================================================
def build_feature_lists_from_selected_keys(
    df: pd.DataFrame,
    feature_groups: dict,
    selected_group_keys: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    selected_group_keys = list(dict.fromkeys(selected_group_keys))

    invalid_keys = sorted(set(selected_group_keys) - ALLOWED_FEATURE_GROUP_KEYS)
    if invalid_keys:
        raise ValueError(f"Invalid feature group keys: {invalid_keys}")

    numeric_group_keys = {
        "recommended_initial_numeric_or_binary",
        "anat_func_numeric_or_binary",
        "drop_leakage",
    }

    categorical_group_keys = {
        "recommended_initial_categorical",
        "optional_categorical_with_caution",
    }

    all_selected = []
    numeric_cols = []
    categorical_cols = []

    for key in selected_group_keys:
        cols = feature_groups.get(key, [])

        for c in cols:
            if c not in all_selected:
                all_selected.append(c)

        if key in numeric_group_keys:
            for c in cols:
                if c not in numeric_cols:
                    numeric_cols.append(c)

        if key in categorical_group_keys:
            for c in cols:
                if c not in categorical_cols:
                    categorical_cols.append(c)

    all_selected = [c for c in all_selected if c != TARGET_COL]
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    categorical_cols = [c for c in categorical_cols if c != TARGET_COL]

    missing_columns = [c for c in all_selected if c not in df.columns]
    selected_columns = [c for c in all_selected if c in df.columns]
    numeric_feature_columns = [c for c in numeric_cols if c in df.columns]
    categorical_columns = [c for c in categorical_cols if c in df.columns]

    if not USE_SITE_ID_AS_FEATURE:
        selected_columns = [c for c in selected_columns if c != "SITE_ID"]
        numeric_feature_columns = [c for c in numeric_feature_columns if c != "SITE_ID"]
        categorical_columns = [c for c in categorical_columns if c != "SITE_ID"]

    return selected_columns, numeric_feature_columns, categorical_columns, missing_columns


def get_all_numeric_candidate_columns(feature_groups: dict) -> list[str]:
    numeric_group_keys = [
        "recommended_initial_numeric_or_binary",
        "anat_func_numeric_or_binary",
        "drop_leakage",
    ]

    cols = []
    for key in numeric_group_keys:
        for c in feature_groups.get(key, []):
            if c != TARGET_COL and c not in cols:
                cols.append(c)

    return cols


# =========================================================
# Domain scaling helpers
# =========================================================
def scale_to_minus1_plus1(series: pd.Series, lo: float, hi: float) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)

    if hi < lo:
        raise ValueError(f"Invalid bounds: {(lo, hi)}")

    if hi == lo:
        out.loc[series.notna()] = 0.0
        return out

    mask = series.notna()
    out.loc[mask] = 2.0 * (series.loc[mask] - lo) / (hi - lo) - 1.0

    return out.clip(-1.0, 1.0)


def apply_domain_scaling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, (lo, hi) in DOMAIN_FIXED_BOUNDS.items():
        if col in df.columns:
            df[col] = scale_to_minus1_plus1(df[col], lo, hi)

    for score_col, (lo, hi) in IQ_GLOBAL_BOUNDS.items():
        if score_col in df.columns:
            df[score_col] = scale_to_minus1_plus1(df[score_col], lo, hi)

    return df


# =========================================================
# Public/private preprocessing helpers
# =========================================================
def build_other_numeric_transformer(mode: str) -> Pipeline:
    if mode not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError(
            f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={mode!r}. "
            f"Allowed values: {sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES)}"
        )

    if mode == "standardize_zero_fill":
        return Pipeline(
            steps=[
                ("scale_then_zero_fill", StandardizeThenZeroFill()),
            ]
        )

    if mode == "minmax_minus1_1_zero_fill":
        return Pipeline(
            steps=[
                ("scale_then_zero_fill", MinMaxMinusOneOneThenZeroFill()),
            ]
        )

    return Pipeline(
        steps=[
            ("zero_fill", ZeroFillTransformer()),
        ]
    )


def make_public_preprocessor(fc_cols: list[str]) -> Optional[ColumnTransformer]:
    if not fc_cols:
        return None

    if FC_IS_CORRELATION:
        fc_pipe = Pipeline(
            steps=[
                ("zero_fill", ZeroFillTransformer()),
            ]
        )
    else:
        fc_pipe = Pipeline(
            steps=[
                ("scale_then_zero_fill", StandardizeThenZeroFill()),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("fc", fc_pipe, fc_cols),
        ],
        remainder="drop",
    )


def make_private_preprocessor(
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
) -> Optional[ColumnTransformer]:
    transformers = []

    if phenotype_domain_scaled_cols:
        phen_domain_pipe = Pipeline(
            steps=[
                ("zero_fill", ZeroFillTransformer()),
            ]
        )
        transformers.append(
            ("phen_domain", phen_domain_pipe, phenotype_domain_scaled_cols)
        )

    if phenotype_other_numeric_cols:
        phen_other_num_pipe = build_other_numeric_transformer(
            PHENOTYPE_OTHER_NUMERIC_MODE
        )
        transformers.append(
            ("phen_other_num", phen_other_num_pipe, phenotype_other_numeric_cols)
        )

    if phenotype_categorical_cols:
        phen_cat_pipe = Pipeline(
            steps=[
                ("onehot_missing_zero", MissingAsAllZeroOneHotEncoder()),
            ]
        )
        transformers.append(
            ("phen_cat", phen_cat_pipe, phenotype_categorical_cols)
        )

    if not transformers:
        return None

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def fit_transform_public_private(
    *,
    public_train_df: pd.DataFrame,
    private_train_df: pd.DataFrame,
    public_val_df: Optional[pd.DataFrame],
    private_val_df: Optional[pd.DataFrame],
    public_test_df: Optional[pd.DataFrame],
    private_test_df: Optional[pd.DataFrame],
    fc_cols: list[str],
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
):
    public_preprocessor = make_public_preprocessor(fc_cols)

    private_preprocessor = make_private_preprocessor(
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
    )

    if public_preprocessor is not None:
        X_pub_train = np.asarray(
            public_preprocessor.fit_transform(public_train_df),
            dtype=np.float64,
        )
        public_feature_names = public_preprocessor.get_feature_names_out().tolist()
    else:
        X_pub_train = np.zeros((len(public_train_df), 0), dtype=np.float64)
        public_feature_names = []

    if private_preprocessor is not None:
        X_priv_train = np.asarray(
            private_preprocessor.fit_transform(private_train_df),
            dtype=np.float64,
        )
        private_feature_names = private_preprocessor.get_feature_names_out().tolist()
    else:
        X_priv_train = np.zeros((len(private_train_df), 0), dtype=np.float64)
        private_feature_names = []

    def transform_public(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if df is None:
            return None
        if public_preprocessor is None:
            return np.zeros((len(df), 0), dtype=np.float64)
        return np.asarray(public_preprocessor.transform(df), dtype=np.float64)

    def transform_private(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if df is None:
            return None
        if private_preprocessor is None:
            return np.zeros((len(df), 0), dtype=np.float64)
        return np.asarray(private_preprocessor.transform(df), dtype=np.float64)

    X_pub_val = transform_public(public_val_df)
    X_priv_val = transform_private(private_val_df)
    X_pub_test = transform_public(public_test_df)
    X_priv_test = transform_private(private_test_df)

    return {
        "public_preprocessor": public_preprocessor,
        "private_preprocessor": private_preprocessor,
        "X_pub_train": X_pub_train,
        "X_priv_train": X_priv_train,
        "X_pub_val": X_pub_val,
        "X_priv_val": X_priv_val,
        "X_pub_test": X_pub_test,
        "X_priv_test": X_priv_test,
        "public_feature_names": public_feature_names,
        "private_feature_names": private_feature_names,
    }


def build_feature_blocks_for_experiment(
    *,
    df: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    feature_groups: dict,
    selected_group_keys: list[str],
) -> FeatureBlockData:
    (
        selected_phenotype_cols,
        phenotype_numeric_cols,
        phenotype_categorical_cols,
        missing_cols,
    ) = build_feature_lists_from_selected_keys(
        df=df,
        feature_groups=feature_groups,
        selected_group_keys=selected_group_keys,
    )

    phenotype_domain_scaled_cols = [
        c
        for c in phenotype_numeric_cols
        if c in DOMAIN_SCALED_NUMERIC_COLS and c in df_model.columns
    ]

    phenotype_other_numeric_cols = [
        c
        for c in phenotype_numeric_cols
        if c not in DOMAIN_SCALED_NUMERIC_COLS and c in df_model.columns
    ]

    n_samples, n_fc_features = X_conn_model.shape
    fc_cols = [f"fc_{i:05d}" for i in range(n_fc_features)]

    public_df = pd.DataFrame(
        X_conn_model,
        columns=fc_cols,
        index=df_model.index,
    )

    if selected_phenotype_cols:
        private_df = df_model[selected_phenotype_cols].copy()
    else:
        private_df = pd.DataFrame(index=df_model.index)

    print(
        f"Public fMRI block shape: {public_df.shape}; "
        f"Private phenotype block shape: {private_df.shape}"
    )

    return FeatureBlockData(
        public_df=public_df.reset_index(drop=True),
        private_df=private_df.reset_index(drop=True),
        fc_cols=fc_cols,
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        selected_phenotype_cols=selected_phenotype_cols,
        missing_cols=missing_cols,
    )


# =========================================================
# IID train/val/test split
# =========================================================
def make_train_val_test_split(
    y: np.ndarray,
    seed: int,
    validation_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Pure IID stratified splitting: no site-aware grouping is used.
    """

    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")

    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    n = len(y)

    test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_val_idx, test_idx = next(test_splitter.split(np.zeros(n), y))
    y_train_val = y[train_val_idx]

    val_fraction_within_train_val = validation_size / (1.0 - test_size)

    val_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction_within_train_val,
        random_state=seed + 1,
    )

    train_rel_idx, val_rel_idx = next(
        val_splitter.split(np.zeros(len(train_val_idx)), y_train_val)
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    split_name = (
        "StratifiedShuffleSplit train/validation/test "
        f"(IID, no SITE_ID grouping): "
        f"train={1.0 - validation_size - test_size:.2f}, "
        f"validation={validation_size:.2f}, test={test_size:.2f}"
    )

    return train_idx, val_idx, test_idx, split_name



# =========================================================
# Fixed split / preprocess-dump helpers
# =========================================================
def _as_clean_int_index(values: pd.Series, *, n_total: int) -> Optional[np.ndarray]:
    """Return values as a unique integer index array if they look like global row ids."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return None

    arr_float = numeric.to_numpy(dtype=np.float64)
    arr_int = arr_float.astype(np.int64)

    if not np.allclose(arr_float, arr_int):
        return None
    if len(np.unique(arr_int)) != len(arr_int):
        return None
    if np.any(arr_int < 0) or np.any(arr_int >= int(n_total)):
        return None

    # If the column is exactly 0, 1, 2, ... within this dumped split, it is
    # usually just a local CSV index saved after reset_index(drop=True), not the
    # original df_model row index. Reject it to avoid silently using wrong rows.
    if np.array_equal(arr_int, np.arange(len(arr_int), dtype=np.int64)):
        return None

    return arr_int.astype(int)


def _indices_from_identifier_column(
    *,
    dump_df: pd.DataFrame,
    df_model_raw: pd.DataFrame,
    split_label: str,
) -> Optional[np.ndarray]:
    """Resolve dump rows to df_model_raw rows using stable identifier columns."""
    for col in DUMP_IDENTIFIER_COLUMN_CANDIDATES:
        if col not in dump_df.columns or col not in df_model_raw.columns:
            continue

        source_values = df_model_raw[col].astype("string").str.strip()
        if source_values.duplicated().any():
            # We need one-to-one IDs for exact split reconstruction.
            continue

        source_map = {
            str(value): int(idx)
            for idx, value in source_values.items()
            if value is not pd.NA and str(value) != "<NA>"
        }

        dump_values = dump_df[col].astype("string").str.strip()
        out = []
        ok = True
        for value in dump_values:
            key = str(value)
            if key == "<NA>" or key not in source_map:
                ok = False
                break
            out.append(source_map[key])

        if ok:
            arr = np.asarray(out, dtype=int)
            print(
                f"Recovered {split_label} indices from dump identifier column {col!r}."
            )
            return arr

    return None


def _indices_from_explicit_index_column(
    *,
    dump_df: pd.DataFrame,
    n_total: int,
    split_label: str,
) -> Optional[np.ndarray]:
    """Resolve dump rows from an explicit saved global/model index column."""
    for col in DUMP_INDEX_COLUMN_CANDIDATES:
        if col not in dump_df.columns:
            continue

        arr = _as_clean_int_index(dump_df[col], n_total=n_total)
        if arr is not None:
            print(
                f"Recovered {split_label} indices from dump index column {col!r}."
            )
            return arr

    return None


def _select_fc_key_columns(dump_df: pd.DataFrame, n_fc_features: int) -> list[str]:
    fc_cols = [f"fc_{i:05d}" for i in range(int(n_fc_features))]
    overlap = [c for c in fc_cols if c in dump_df.columns]

    if not overlap:
        return []

    if len(overlap) <= DUMP_FC_KEY_MAX_COLUMNS:
        return overlap

    positions = np.linspace(
        0,
        len(overlap) - 1,
        int(DUMP_FC_KEY_MAX_COLUMNS),
        dtype=int,
    )
    return [overlap[int(i)] for i in positions]


def _rounded_tuple_keys_from_frame(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    if not cols:
        return []

    numeric_df = df[cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    arr = numeric_df.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.round(arr, decimals=int(DUMP_FC_KEY_ROUND_DECIMALS))
    return [tuple(row.tolist()) for row in arr]


def _indices_from_fc_columns(
    *,
    dump_df: pd.DataFrame,
    X_conn_model: np.ndarray,
    split_label: str,
) -> Optional[np.ndarray]:
    """Resolve dump rows by matching rounded fMRI FC columns, if present."""
    n_total, n_fc_features = X_conn_model.shape
    key_cols = _select_fc_key_columns(dump_df, n_fc_features=n_fc_features)

    if not key_cols:
        return None

    source_fc = pd.DataFrame(
        X_conn_model,
        columns=[f"fc_{i:05d}" for i in range(int(n_fc_features))],
    )

    source_keys = _rounded_tuple_keys_from_frame(source_fc, key_cols)
    key_to_idx: dict[tuple, int] = {}
    duplicate_keys = set()

    for idx, key in enumerate(source_keys):
        if key in key_to_idx:
            duplicate_keys.add(key)
        else:
            key_to_idx[key] = int(idx)

    if duplicate_keys:
        for key in duplicate_keys:
            key_to_idx.pop(key, None)

    dump_keys = _rounded_tuple_keys_from_frame(dump_df, key_cols)
    out = []
    for key in dump_keys:
        if key not in key_to_idx:
            return None
        out.append(key_to_idx[key])

    arr = np.asarray(out, dtype=int)
    if len(np.unique(arr)) != len(arr):
        return None

    print(
        f"Recovered {split_label} indices by matching {len(key_cols)} rounded fc_* columns."
    )
    return arr


def _resolve_dump_split_indices(
    *,
    dump_df: pd.DataFrame,
    df_model_raw: pd.DataFrame,
    X_conn_model: np.ndarray,
    split_label: str,
) -> np.ndarray:
    """Resolve one dumped split CSV to df_model_raw/X_conn_model row indices."""
    n_total = len(df_model_raw)

    for resolver in [
        lambda: _indices_from_identifier_column(
            dump_df=dump_df,
            df_model_raw=df_model_raw,
            split_label=split_label,
        ),
        lambda: _indices_from_explicit_index_column(
            dump_df=dump_df,
            n_total=n_total,
            split_label=split_label,
        ),
        lambda: _indices_from_fc_columns(
            dump_df=dump_df,
            X_conn_model=X_conn_model,
            split_label=split_label,
        ),
    ]:
        arr = resolver()
        if arr is not None:
            return np.asarray(arr, dtype=int)

    raise ValueError(
        f"Could not recover row indices for {split_label!r} from the preprocess dump. "
        "Please ensure the dumped CSV contains either a stable identifier column "
        f"from {DUMP_IDENTIFIER_COLUMN_CANDIDATES}, an explicit global/model index "
        f"column from {DUMP_INDEX_COLUMN_CANDIDATES}, or fc_***** columns that match "
        "X_conn_model."
    )



def _load_fixed_split_indices_from_npy_if_available(
    *,
    n_total: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load fixed split indices directly from non-private .npy artifacts if present.

    The IID fixed-split non-private script writes both simple names
    (train_idx.npy, val_idx.npy, test_idx.npy) and fixed_* names. These are
    already in df_model_raw / X_conn_model index space after valid-mask
    filtering, so they are more reliable than reconstructing indices from
    floating-point feature columns.
    """
    candidate_sets = [
        (
            PREPROCESS_DUMP_DIR / "fixed_train_idx.npy",
            PREPROCESS_DUMP_DIR / "fixed_val_idx.npy",
            PREPROCESS_DUMP_DIR / "fixed_test_idx.npy",
        ),
        (
            PREPROCESS_DUMP_DIR / "train_idx.npy",
            PREPROCESS_DUMP_DIR / "val_idx.npy",
            PREPROCESS_DUMP_DIR / "test_idx.npy",
        ),
    ]

    for train_path, val_path, test_path in candidate_sets:
        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            continue

        train_idx = np.asarray(np.load(train_path), dtype=int).reshape(-1)
        val_idx = np.asarray(np.load(val_path), dtype=int).reshape(-1)
        test_idx = np.asarray(np.load(test_path), dtype=int).reshape(-1)

        all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
        if len(np.unique(all_idx)) != len(all_idx):
            raise ValueError(
                "Loaded fixed split .npy files are not disjoint across "
                "train/validation/test."
            )
        if np.any(all_idx < 0) or np.any(all_idx >= int(n_total)):
            raise ValueError(
                "Loaded fixed split .npy files contain row indices outside "
                f"[0, {int(n_total) - 1}]."
            )

        split_name = (
            "Fixed IID train/validation/test split loaded from non-private "
            f".npy index artifacts in {PREPROCESS_DUMP_DIR}. "
            "Feature-DP reuses these exact row indices for every run seed."
        )
        print(
            "Recovered fixed split indices directly from .npy artifacts:\n"
            f"  train: {train_path}\n"
            f"  validation: {val_path}\n"
            f"  test: {test_path}"
        )
        return train_idx, val_idx, test_idx, split_name

    return None
def load_fixed_split_indices_from_preprocess_dump(
    *,
    df_model_raw: pd.DataFrame,
    X_conn_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load fixed split indices from the non-private dumped artifacts.

    Priority order:
      1. Direct .npy index arrays written by the non-private fixed-split script.
      2. CSV column row_index_after_valid_mask, when available.
      3. Other stable id/index columns or fc_* matching as a fallback.
    """
    npy_result = _load_fixed_split_indices_from_npy_if_available(
        n_total=len(df_model_raw),
    )
    if npy_result is not None:
        return npy_result

    split_paths = {
        "train": PREPROCESS_DUMP_TRAIN_CSV,
        "validation": PREPROCESS_DUMP_VAL_CSV,
        "test": PREPROCESS_DUMP_TEST_CSV,
    }

    for label, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"USE_PREPROCESS_DUMP=True, but the {label} dump does not exist: {path}"
            )

    dump_train = pd.read_csv(PREPROCESS_DUMP_TRAIN_CSV)
    dump_val = pd.read_csv(PREPROCESS_DUMP_VAL_CSV)
    dump_test = pd.read_csv(PREPROCESS_DUMP_TEST_CSV)

    train_idx = _resolve_dump_split_indices(
        dump_df=dump_train,
        df_model_raw=df_model_raw,
        X_conn_model=X_conn_model,
        split_label="train",
    )
    val_idx = _resolve_dump_split_indices(
        dump_df=dump_val,
        df_model_raw=df_model_raw,
        X_conn_model=X_conn_model,
        split_label="validation",
    )
    test_idx = _resolve_dump_split_indices(
        dump_df=dump_test,
        df_model_raw=df_model_raw,
        X_conn_model=X_conn_model,
        split_label="test",
    )

    all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError(
            "The fixed preprocess dump split is not disjoint: at least one row "
            "appears in more than one of train/validation/test."
        )

    if PREPROCESS_DUMP_TRAIN_VAL_CSV.exists():
        dump_train_val = pd.read_csv(PREPROCESS_DUMP_TRAIN_VAL_CSV)
        train_val_idx_dump = _resolve_dump_split_indices(
            dump_df=dump_train_val,
            df_model_raw=df_model_raw,
            X_conn_model=X_conn_model,
            split_label="train_validation",
        )
        expected = np.concatenate([train_idx, val_idx], axis=0)

        if set(train_val_idx_dump.tolist()) != set(expected.tolist()):
            raise ValueError(
                "train_val_fixed.csv does not contain exactly the union of "
                "train_fixed.csv and val_fixed.csv."
            )

    split_name = (
        "Fixed train/validation/test split loaded from non-private preprocess "
        f"dump CSVs in {PREPROCESS_DUMP_DIR}. "
        "Feature-DP reuses these exact row indices for every run seed."
    )

    return train_idx, val_idx, test_idx, split_name


def save_prepared_global_arrays(
    *,
    global_data: PreparedGlobalData,
    output_dir: Path,
) -> None:
    """Save the same global arrays/split files that prepare_global_data saves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_conn_model.npy", global_data.X_conn_model)
    np.save(output_dir / "y_binary.npy", global_data.y)
    np.save(output_dir / "train_idx.npy", global_data.train_idx)
    np.save(output_dir / "val_idx.npy", global_data.val_idx)
    np.save(output_dir / "test_idx.npy", global_data.test_idx)



# =========================================================
# DP accounting
# =========================================================
def _make_rdp_accountant(orders):
    try:
        from dp_accounting import rdp

        return rdp.RdpAccountant(orders)
    except Exception:
        from dp_accounting.rdp import rdp_privacy_accountant

        return rdp_privacy_accountant.RdpAccountant(orders)


def _compose_poisson_gaussian_event(
    accountant,
    sampling: float,
    noise_multiplier: float,
    step: int,
) -> None:
    from dp_accounting import dp_event as event

    base_event = event.PoissonSampledDpEvent(
        sampling,
        event.GaussianDpEvent(noise_multiplier),
    )

    try:
        dpevent = event.SelfComposedDpEvent(base_event, step)
        accountant.compose(dpevent)
    except Exception:
        accountant.compose(base_event, count=step)


def epsilon_for_noise_multiplier(
    *,
    noise_multiplier: float,
    target_delta: float,
    sampling: float,
    step: int,
) -> float:
    if noise_multiplier <= 0:
        return math.inf

    RDP_ORDERS = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )

    accountant = _make_rdp_accountant(RDP_ORDERS)

    _compose_poisson_gaussian_event(
        accountant=accountant,
        sampling=sampling,
        noise_multiplier=noise_multiplier,
        step=step,
    )

    return float(accountant.get_epsilon(target_delta))


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sampling: float,
    step: int,
) -> float:
    """
    Uses the RDP accountant logic requested by the user.
    """

    RDP_ORDERS = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )

    def objective(noise_multiplier):
        from dp_accounting import dp_event as event
        from dp_accounting import rdp

        accountant = rdp.RdpAccountant(RDP_ORDERS)
        dpevent = event.SelfComposedDpEvent(
            event.PoissonSampledDpEvent(
                sampling,
                event.GaussianDpEvent(noise_multiplier),
            ),
            step,
        )
        accountant.compose(dpevent)
        eps = accountant.get_epsilon(target_delta)
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    return float(optimal_noise)


# =========================================================
# Metrics
# =========================================================
def safe_scalar_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return np.nan


def flatten_classification_report_dict(report: dict) -> dict:
    flat = {}

    for outer_key, outer_val in report.items():
        if isinstance(outer_val, dict):
            for inner_key, inner_val in outer_val.items():
                if isinstance(inner_val, (int, float, np.integer, np.floating)):
                    flat[f"{outer_key}_{inner_key}"] = float(inner_val)
        elif isinstance(outer_val, (int, float, np.integer, np.floating)):
            flat[outer_key] = float(outer_val)

    return flat


def compute_prediction_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    unique_classes = np.unique(y_true)

    out = {
        "n_total": int(len(y_true)),
        "n_control": int(np.sum(y_true == 0)),
        "n_autism": int(np.sum(y_true == 1)),
        "autism_prevalence": float(np.mean(y_true == 1)) if len(y_true) else np.nan,
        "n_unique_classes": int(len(unique_classes)),
        "predicted_autism_rate": float(np.mean(y_pred == 1)) if len(y_pred) else np.nan,
        "accuracy": safe_scalar_metric(accuracy_score, y_true, y_pred),
        "log_loss": safe_scalar_metric(
            log_loss,
            y_true,
            np.clip(y_proba, 1e-12, 1.0 - 1e-12),
            labels=[0, 1],
        ),
    }

    if len(unique_classes) == 2:
        out["balanced_accuracy"] = safe_scalar_metric(
            balanced_accuracy_score,
            y_true,
            y_pred,
        )
        out["roc_auc"] = safe_scalar_metric(
            roc_auc_score,
            y_true,
            y_proba,
        )
        out["average_precision"] = safe_scalar_metric(
            average_precision_score,
            y_true,
            y_proba,
        )
    else:
        out["balanced_accuracy"] = np.nan
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["control", "autism"],
        output_dict=True,
        zero_division=0,
    )

    out.update(flatten_classification_report_dict(report))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["cm_true_control_pred_control"] = int(cm[0, 0])
    out["cm_true_control_pred_autism"] = int(cm[0, 1])
    out["cm_true_autism_pred_control"] = int(cm[1, 0])
    out["cm_true_autism_pred_autism"] = int(cm[1, 1])

    return out


def selection_score_from_row(row: dict, metric: str, prefix: str = "val") -> float:
    key = f"{prefix}_{metric}"

    if key not in row:
        raise KeyError(f"Selection metric {key!r} not found.")

    value = float(row[key])

    if metric == "log_loss":
        return -value

    return value


def load_private_clip_norm_grid(
    *,
    quantile_csv_path: Optional[Path],
    quantiles: list[int],
    fallback_grid: list[float],
) -> list[float]:
    """
    Build the private clipping grid.

    By default, this uses the fixed PRIVATE_CLIP_NORM_GRID requested for this
    experiment. The CSV/quantile path is retained for backward compatibility
    when USE_FIXED_PRIVATE_CLIP_NORM_GRID is set to False.
    """
    if USE_FIXED_PRIVATE_CLIP_NORM_GRID:
        grid = sorted(
            set(
                float(x)
                for x in PRIVATE_CLIP_NORM_GRID
                if np.isfinite(float(x)) and float(x) > 0
            )
        )
        if not grid:
            raise ValueError("PRIVATE_CLIP_NORM_GRID is empty after filtering.")
        print("\nUsing fixed private clip-norm grid:")
        print("Clip norms:", grid)
        return grid

    if quantile_csv_path is None or not quantile_csv_path.exists():
        print(
            "\nPrivate clip-norm quantile CSV not found. "
            f"Using fallback grid: {fallback_grid}"
        )
        return sorted(set(float(x) for x in fallback_grid if float(x) > 0))

    qdf = pd.read_csv(quantile_csv_path)

    required_cols = {"quantile_percent", "norm_value"}
    missing_cols = required_cols - set(qdf.columns)
    if missing_cols:
        raise ValueError(
            f"Private clip-norm CSV is missing columns: {sorted(missing_cols)}. "
            f"Expected columns: {sorted(required_cols)}"
        )

    qdf = qdf.copy()
    qdf["quantile_percent"] = pd.to_numeric(qdf["quantile_percent"], errors="coerce")
    qdf["norm_value"] = pd.to_numeric(qdf["norm_value"], errors="coerce")
    qdf = qdf.dropna(subset=["quantile_percent", "norm_value"])
    qdf = qdf.loc[qdf["norm_value"] > 0].copy()

    if qdf.empty:
        raise ValueError("Private clip-norm CSV does not contain any positive norm values.")

    grid = []
    for q in quantiles:
        exact = qdf.loc[qdf["quantile_percent"] == q, "norm_value"]

        if not exact.empty:
            value = float(exact.iloc[0])
        else:
            nearest_idx = (qdf["quantile_percent"] - q).abs().idxmin()
            value = float(qdf.loc[nearest_idx, "norm_value"])

        grid.append(value)

    grid = sorted(set(float(x) for x in grid if np.isfinite(x) and float(x) > 0))

    if not grid:
        raise ValueError("Private clip-norm grid is empty after reading quantile CSV.")

    # Add clip norm = 1 to the grid if it's not already present, as a reasonable default.
    if 1.0 not in grid:
        grid.append(1.0)
        grid = sorted(set(grid))

    print("\nPrivate clip-norm grid loaded from feature-norm quantiles:")
    print("Quantiles:", quantiles)
    print("Clip norms:", grid)

    return grid


def print_metric_block(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            print(f"{k:>36s}: {v:.6f}")




# =========================================================
# Optional Numba kernels for the Feature-DP update loop
# =========================================================
JIT_HISTORY_COLUMNS = [
    "step",
    "learning_rate",
    "batch_size",
    "public_clipping_enabled",
    "clip_norm_public",
    "clip_norm_private",
    "mean_public_grad_norm_before_clip",
    "max_public_grad_norm_before_clip",
    "fraction_public_clipped",
    "mean_private_residual_grad_norm_before_clip",
    "max_private_residual_grad_norm_before_clip",
    "fraction_private_residual_clipped",
    "total_grad_w_public_norm",
    "total_grad_w_private_norm",
    "total_grad_b_abs",
    "w_public_l2_norm",
    "w_private_l2_norm",
    "train_accuracy",
    "train_log_loss",
    "train_objective",
    "train_data_loss",
    "train_l2_reg_loss",
    "train_l2_reg_strength",
    "train_w_public_l2_norm",
    "train_w_private_l2_norm",
    "val_accuracy",
    "val_log_loss",
    "val_objective",
    "val_data_loss",
    "val_l2_reg_loss",
    "val_l2_reg_strength",
    "val_w_public_l2_norm",
    "val_w_private_l2_norm",
    "test_accuracy",
    "test_log_loss",
    "test_objective",
    "test_data_loss",
    "test_l2_reg_loss",
    "test_l2_reg_strength",
    "test_w_public_l2_norm",
    "test_w_private_l2_norm",
]


def _count_recorded_steps(iterations: int, record_every_n_steps: int) -> int:
    count = 1  # step 0
    for step in range(1, int(iterations) + 1):
        if step == 1 or step == int(iterations) or step % int(record_every_n_steps) == 0:
            count += 1
    return count


def generate_feature_dp_noise_arrays(
    *,
    random_state: int,
    iterations: int,
    n_public_features: int,
    n_private_features: int,
    noise_multiplier: float,
    clip_norm_private: float,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Gaussian noise outside Numba using the same NumPy Generator path
    used by the reference Python implementation.

    Per step, the draw order is:
        1. public-weight part of private residual noise
        2. private-weight part of private residual noise
        3. intercept residual noise, if enabled
    """
    rng = np.random.default_rng(int(random_state))
    noise_std = float(noise_multiplier) * float(clip_norm_private)

    noise_w_pub = np.empty((int(iterations), int(n_public_features)), dtype=np.float64)
    noise_w_priv = np.empty((int(iterations), int(n_private_features)), dtype=np.float64)
    noise_b = np.zeros(int(iterations), dtype=np.float64)

    for step in range(int(iterations)):
        noise_w_pub[step, :] = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=int(n_public_features),
        )
        noise_w_priv[step, :] = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=int(n_private_features),
        )
        if fit_intercept:
            noise_b[step] = float(rng.normal(loc=0.0, scale=noise_std))

    return noise_w_pub, noise_w_priv, noise_b


if NUMBA_AVAILABLE:
    @njit
    def _jit_sigmoid_scalar(z: float) -> float:
        if z > 50.0:
            z = 50.0
        elif z < -50.0:
            z = -50.0
        return 1.0 / (1.0 + math.exp(-z))


    @njit
    def _jit_weight_l2_sq(w: np.ndarray) -> float:
        out = 0.0
        for j in range(w.shape[0]):
            out += w[j] * w[j]
        return out


    @njit
    def _jit_eval_basic_metrics(
        X_pub: np.ndarray,
        X_priv: np.ndarray,
        y: np.ndarray,
        w_pub: np.ndarray,
        w_priv: np.ndarray,
        b: float,
        fit_intercept: bool,
        l2_reg_strength: float,
    ) -> tuple:
        n = y.shape[0]
        if n == 0:
            return (
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                l2_reg_strength,
                math.sqrt(_jit_weight_l2_sq(w_pub)),
                math.sqrt(_jit_weight_l2_sq(w_priv)),
            )

        correct = 0
        loss_sum = 0.0

        for i in range(n):
            logit = 0.0

            for j in range(w_pub.shape[0]):
                logit += X_pub[i, j] * w_pub[j]

            for j in range(w_priv.shape[0]):
                logit += X_priv[i, j] * w_priv[j]

            if fit_intercept:
                logit += b

            p = _jit_sigmoid_scalar(logit)

            if p >= 0.5:
                pred = 1
            else:
                pred = 0

            if pred == int(y[i]):
                correct += 1

            if p < 1e-12:
                p_loss = 1e-12
            elif p > 1.0 - 1e-12:
                p_loss = 1.0 - 1e-12
            else:
                p_loss = p

            loss_sum += -(
                y[i] * math.log(p_loss)
                + (1.0 - y[i]) * math.log(1.0 - p_loss)
            )

        acc = correct / n
        data_loss = loss_sum / n
        pub_sq = _jit_weight_l2_sq(w_pub)
        priv_sq = _jit_weight_l2_sq(w_priv)
        reg_loss = 0.5 * l2_reg_strength * (pub_sq + priv_sq)
        objective = data_loss + reg_loss

        return (
            acc,
            data_loss,
            objective,
            data_loss,
            reg_loss,
            l2_reg_strength,
            math.sqrt(pub_sq),
            math.sqrt(priv_sq),
        )


    @njit
    def _jit_write_history_row(
        history: np.ndarray,
        row_idx: int,
        step: int,
        learning_rate: float,
        batch_size: int,
        public_clipping_enabled: bool,
        clip_norm_public_value: float,
        clip_norm_private: float,
        mean_public_norm: float,
        max_public_norm: float,
        fraction_public_clipped: float,
        mean_private_norm: float,
        max_private_norm: float,
        fraction_private_clipped: float,
        total_grad_w_public_norm: float,
        total_grad_w_private_norm: float,
        total_grad_b_abs: float,
        w_pub: np.ndarray,
        w_priv: np.ndarray,
        b: float,
        fit_intercept: bool,
        l2_reg_strength: float,
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        has_val: bool,
        X_pub_val: np.ndarray,
        X_priv_val: np.ndarray,
        y_val: np.ndarray,
        has_test: bool,
        X_pub_test: np.ndarray,
        X_priv_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        pub_norm = math.sqrt(_jit_weight_l2_sq(w_pub))
        priv_norm = math.sqrt(_jit_weight_l2_sq(w_priv))

        history[row_idx, 0] = step
        history[row_idx, 1] = learning_rate
        history[row_idx, 2] = batch_size
        history[row_idx, 3] = 1.0 if public_clipping_enabled else 0.0
        history[row_idx, 4] = clip_norm_public_value
        history[row_idx, 5] = clip_norm_private
        history[row_idx, 6] = mean_public_norm
        history[row_idx, 7] = max_public_norm
        history[row_idx, 8] = fraction_public_clipped
        history[row_idx, 9] = mean_private_norm
        history[row_idx, 10] = max_private_norm
        history[row_idx, 11] = fraction_private_clipped
        history[row_idx, 12] = total_grad_w_public_norm
        history[row_idx, 13] = total_grad_w_private_norm
        history[row_idx, 14] = total_grad_b_abs
        history[row_idx, 15] = pub_norm
        history[row_idx, 16] = priv_norm

        train_metrics = _jit_eval_basic_metrics(
            X_pub_train,
            X_priv_train,
            y_train,
            w_pub,
            w_priv,
            b,
            fit_intercept,
            l2_reg_strength,
        )
        for k in range(8):
            history[row_idx, 17 + k] = train_metrics[k]

        if has_val:
            val_metrics = _jit_eval_basic_metrics(
                X_pub_val,
                X_priv_val,
                y_val,
                w_pub,
                w_priv,
                b,
                fit_intercept,
                l2_reg_strength,
            )
            for k in range(8):
                history[row_idx, 25 + k] = val_metrics[k]

        if has_test:
            test_metrics = _jit_eval_basic_metrics(
                X_pub_test,
                X_priv_test,
                y_test,
                w_pub,
                w_priv,
                b,
                fit_intercept,
                l2_reg_strength,
            )
            for k in range(8):
                history[row_idx, 33 + k] = test_metrics[k]


    @njit
    def _jit_train_feature_dp_full_batch(
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        has_val: bool,
        X_pub_val: np.ndarray,
        X_priv_val: np.ndarray,
        y_val: np.ndarray,
        has_test: bool,
        X_pub_test: np.ndarray,
        X_priv_test: np.ndarray,
        y_test: np.ndarray,
        noise_w_pub: np.ndarray,
        noise_w_priv: np.ndarray,
        noise_b: np.ndarray,
        learning_rate: float,
        clip_norm_public_enabled: bool,
        clip_norm_public_value: float,
        clip_norm_private: float,
        iterations: int,
        C: float,
        fit_intercept: bool,
        record_every_n_steps: int,
        stability_nu: float,
        history: np.ndarray,
    ) -> tuple:
        n_train = y_train.shape[0]
        n_pub = X_pub_train.shape[1]
        n_priv = X_priv_train.shape[1]

        w_pub = np.zeros(n_pub, dtype=np.float64)
        w_priv = np.zeros(n_priv, dtype=np.float64)
        b = 0.0

        l2_reg_strength = 1.0 / (C * n_train)

        sum_pub_w_pub = np.zeros(n_pub, dtype=np.float64)
        sum_priv_w_pub = np.zeros(n_pub, dtype=np.float64)
        sum_priv_w_priv = np.zeros(n_priv, dtype=np.float64)

        row_idx = 0

        _jit_write_history_row(
            history,
            row_idx,
            0,
            learning_rate,
            n_train,
            clip_norm_public_enabled,
            clip_norm_public_value,
            clip_norm_private,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            w_pub,
            w_priv,
            b,
            fit_intercept,
            l2_reg_strength,
            X_pub_train,
            X_priv_train,
            y_train,
            has_val,
            X_pub_val,
            X_priv_val,
            y_val,
            has_test,
            X_pub_test,
            X_priv_test,
            y_test,
        )
        row_idx += 1

        for step in range(1, iterations + 1):
            for j in range(n_pub):
                sum_pub_w_pub[j] = 0.0
                sum_priv_w_pub[j] = 0.0

            for j in range(n_priv):
                sum_priv_w_priv[j] = 0.0

            sum_pub_b = 0.0
            sum_priv_b = 0.0

            sum_public_norm = 0.0
            max_public_norm = 0.0
            count_public_clipped = 0

            sum_private_norm = 0.0
            max_private_norm = 0.0
            count_private_clipped = 0

            for i in range(n_train):
                logit_pub = 0.0
                for j in range(n_pub):
                    logit_pub += X_pub_train[i, j] * w_pub[j]

                if fit_intercept:
                    logit_pub += b

                p_pub = _jit_sigmoid_scalar(logit_pub)
                err_pub = p_pub - y_train[i]

                logit_full = logit_pub
                for j in range(n_priv):
                    logit_full += X_priv_train[i, j] * w_priv[j]

                p_full = _jit_sigmoid_scalar(logit_full)
                err_full = p_full - y_train[i]
                err_resid = err_full - err_pub

                public_norm_sq = 0.0
                for j in range(n_pub):
                    g = err_pub * X_pub_train[i, j]
                    public_norm_sq += g * g

                if fit_intercept:
                    public_norm_sq += err_pub * err_pub

                public_norm = math.sqrt(public_norm_sq)
                sum_public_norm += public_norm
                if public_norm > max_public_norm:
                    max_public_norm = public_norm

                if clip_norm_public_enabled:
                    public_scale = clip_norm_public_value / (public_norm + stability_nu)
                    if public_scale > 1.0:
                        public_scale = 1.0
                    else:
                        count_public_clipped += 1
                else:
                    public_scale = 1.0

                for j in range(n_pub):
                    sum_pub_w_pub[j] += public_scale * err_pub * X_pub_train[i, j]

                if fit_intercept:
                    sum_pub_b += public_scale * err_pub

                private_norm_sq = 0.0
                for j in range(n_pub):
                    g = err_resid * X_pub_train[i, j]
                    private_norm_sq += g * g

                for j in range(n_priv):
                    g = err_full * X_priv_train[i, j]
                    private_norm_sq += g * g

                if fit_intercept:
                    private_norm_sq += err_resid * err_resid

                private_norm = math.sqrt(private_norm_sq)
                sum_private_norm += private_norm
                if private_norm > max_private_norm:
                    max_private_norm = private_norm

                private_scale = clip_norm_private / (private_norm + stability_nu)
                if private_scale > 1.0:
                    private_scale = 1.0
                else:
                    count_private_clipped += 1

                for j in range(n_pub):
                    sum_priv_w_pub[j] += private_scale * err_resid * X_pub_train[i, j]

                for j in range(n_priv):
                    sum_priv_w_priv[j] += private_scale * err_full * X_priv_train[i, j]

                if fit_intercept:
                    sum_priv_b += private_scale * err_resid

            total_grad_w_public_norm_sq = 0.0
            total_grad_w_private_norm_sq = 0.0

            for j in range(n_pub):
                grad_pub = sum_pub_w_pub[j] / n_train
                grad_priv = (sum_priv_w_pub[j] + noise_w_pub[step - 1, j]) / n_train
                total_grad = grad_pub + grad_priv + l2_reg_strength * w_pub[j]
                total_grad_w_public_norm_sq += total_grad * total_grad
                w_pub[j] = w_pub[j] - learning_rate * total_grad

            for j in range(n_priv):
                grad_priv = (sum_priv_w_priv[j] + noise_w_priv[step - 1, j]) / n_train
                total_grad = grad_priv + l2_reg_strength * w_priv[j]
                total_grad_w_private_norm_sq += total_grad * total_grad
                w_priv[j] = w_priv[j] - learning_rate * total_grad

            total_grad_b = 0.0
            if fit_intercept:
                grad_pub_b = sum_pub_b / n_train
                grad_priv_b = (sum_priv_b + noise_b[step - 1]) / n_train
                total_grad_b = grad_pub_b + grad_priv_b
                b = b - learning_rate * total_grad_b
            else:
                b = 0.0

            if step == 1 or step == iterations or step % record_every_n_steps == 0:
                _jit_write_history_row(
                    history,
                    row_idx,
                    step,
                    learning_rate,
                    n_train,
                    clip_norm_public_enabled,
                    clip_norm_public_value,
                    clip_norm_private,
                    sum_public_norm / n_train,
                    max_public_norm,
                    count_public_clipped / n_train,
                    sum_private_norm / n_train,
                    max_private_norm,
                    count_private_clipped / n_train,
                    math.sqrt(total_grad_w_public_norm_sq),
                    math.sqrt(total_grad_w_private_norm_sq),
                    abs(total_grad_b),
                    w_pub,
                    w_priv,
                    b,
                    fit_intercept,
                    l2_reg_strength,
                    X_pub_train,
                    X_priv_train,
                    y_train,
                    has_val,
                    X_pub_val,
                    X_priv_val,
                    y_val,
                    has_test,
                    X_pub_test,
                    X_priv_test,
                    y_test,
                )
                row_idx += 1

        return w_pub, w_priv, b, l2_reg_strength


# =========================================================
# Feature-DP logistic regression
# =========================================================
class FeatureDPLogisticRegressionBinary:
    """
    Binary logistic regression with public and private feature blocks.

    Public features:
        fMRI FC features

    Private features:
        selected phenotypic features

    Model:
        logit_full = X_pub @ w_pub + X_priv @ w_priv + b
        logit_pub  = X_pub @ w_pub + b

    Gradients:
        public gradient:
            grad_pub = grad loss(logit_pub)

        private residual gradient:
            grad_priv = grad loss(logit_full) - grad loss(logit_pub)

    Only private residual gradients are clipped and noised.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        clip_norm_public: Optional[float],
        clip_norm_private: float,
        noise_multiplier: float,
        iterations: int,
        C: float,
        fit_intercept: bool,
        l2_reg_handling: str,
        random_state: int,
        record_every_n_steps: int = 1,
        stability_nu: float = 1e-6,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if clip_norm_public is not None and clip_norm_public <= 0:
            raise ValueError("clip_norm_public must be positive or None.")
        if clip_norm_private <= 0:
            raise ValueError("clip_norm_private must be positive.")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if C <= 0:
            raise ValueError("C must be positive.")
        if l2_reg_handling != "add_after_noise":
            raise ValueError("Feature DP implementation expects L2_REG_HANDLING='add_after_noise'.")

        self.learning_rate = float(learning_rate)
        self.clip_norm_public = None if clip_norm_public is None else float(clip_norm_public)
        self.clip_norm_private = float(clip_norm_private)
        self.noise_multiplier = float(noise_multiplier)
        self.iterations = int(iterations)
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.l2_reg_handling = str(l2_reg_handling)
        self.random_state = int(random_state)
        self.record_every_n_steps = max(1, int(record_every_n_steps))
        self.stability_nu = float(stability_nu)

        self.w_pub_: Optional[np.ndarray] = None
        self.w_priv_: Optional[np.ndarray] = None
        self.b_: Optional[float] = None
        self.l2_reg_strength_: Optional[float] = None
        self.step_history_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _check_initialized(self) -> None:
        if self.w_pub_ is None or self.w_priv_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted.")

    def _initialize_parameters(self, n_public_features: int, n_private_features: int) -> None:
        self.w_pub_ = np.zeros(n_public_features, dtype=np.float64)
        self.w_priv_ = np.zeros(n_private_features, dtype=np.float64)
        self.b_ = 0.0

    def predict_proba_1d(self, X_pub: np.ndarray, X_priv: np.ndarray) -> np.ndarray:
        self._check_initialized()

        logits = X_pub @ self.w_pub_ + X_priv @ self.w_priv_
        if self.fit_intercept:
            logits = logits + self.b_

        return self.sigmoid(logits)

    def predict(self, X_pub: np.ndarray, X_priv: np.ndarray) -> np.ndarray:
        return (self.predict_proba_1d(X_pub, X_priv) >= 0.5).astype(int)

    def objective_loss(
        self,
        X_pub: np.ndarray,
        X_priv: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        self._check_initialized()

        if self.l2_reg_strength_ is None:
            raise RuntimeError("l2_reg_strength_ is not initialized.")

        p = self.predict_proba_1d(X_pub, X_priv)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        y = np.asarray(y, dtype=np.float64)
        data_loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        reg_loss = 0.5 * self.l2_reg_strength_ * (
            float(np.dot(self.w_pub_, self.w_pub_))
            + float(np.dot(self.w_priv_, self.w_priv_))
        )

        objective = float(data_loss + reg_loss)

        return objective, float(data_loss), float(reg_loss)

    def evaluate_dataset(
        self,
        X_pub: np.ndarray,
        X_priv: np.ndarray,
        y: np.ndarray,
        *,
        prefix: str,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        y_proba = self.predict_proba_1d(X_pub, X_priv)
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_prediction_metrics_from_arrays(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        objective, data_loss, reg_loss = self.objective_loss(X_pub, X_priv, y)

        metrics["objective"] = objective
        metrics["data_loss"] = data_loss
        metrics["l2_reg_loss"] = reg_loss
        metrics["l2_reg_strength"] = float(self.l2_reg_strength_)
        metrics["w_public_l2_norm"] = float(np.linalg.norm(self.w_pub_))
        metrics["w_private_l2_norm"] = float(np.linalg.norm(self.w_priv_))

        prefixed = {f"{prefix}_{k}": v for k, v in metrics.items()}

        return prefixed, y_pred, y_proba

    def compute_per_example_full_gradients(
        self,
        X_pub: np.ndarray,
        X_priv: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_initialized()

        logits = X_pub @ self.w_pub_ + X_priv @ self.w_priv_
        if self.fit_intercept:
            logits = logits + self.b_

        p = self.sigmoid(logits)
        errors = p - y

        grad_w_pub = errors[:, None] * X_pub
        grad_w_priv = errors[:, None] * X_priv
        grad_b = errors.copy() if self.fit_intercept else np.zeros_like(errors)

        return grad_w_pub, grad_w_priv, grad_b

    def compute_per_example_public_gradients(
        self,
        X_pub: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_initialized()

        logits = X_pub @ self.w_pub_
        if self.fit_intercept:
            logits = logits + self.b_

        p = self.sigmoid(logits)
        errors = p - y

        grad_w_pub = errors[:, None] * X_pub
        grad_w_priv = np.zeros((len(X_pub), len(self.w_priv_)), dtype=np.float64)
        grad_b = errors.copy() if self.fit_intercept else np.zeros_like(errors)

        return grad_w_pub, grad_w_priv, grad_b

    def compute_per_example_private_residual_gradients(
        self,
        X_pub: np.ndarray,
        X_priv: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full = self.compute_per_example_full_gradients(X_pub, X_priv, y)
        pub = self.compute_per_example_public_gradients(X_pub, y)

        return (
            full[0] - pub[0],
            full[1] - pub[1],
            full[2] - pub[2],
        )

    @staticmethod
    def per_example_gradient_l2_norms(
        grads: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        grad_w_pub, grad_w_priv, grad_b = grads

        sq = (
            np.sum(grad_w_pub * grad_w_pub, axis=1)
            + np.sum(grad_w_priv * grad_w_priv, axis=1)
            + grad_b * grad_b
        )

        return np.sqrt(sq)

    def clip_per_example_grads(
        self,
        grads: tuple[np.ndarray, np.ndarray, np.ndarray],
        clip_norm: Optional[float],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        grad_w_pub, grad_w_priv, grad_b = grads

        norms = self.per_example_gradient_l2_norms(grads)

        if clip_norm is None:
            scales = np.ones_like(norms)
            return (
                grad_w_pub.copy(),
                grad_w_priv.copy(),
                grad_b.copy(),
            ), norms, scales

        scales = np.minimum(1.0, clip_norm / (norms + self.stability_nu))

        clipped = (
            grad_w_pub * scales[:, None],
            grad_w_priv * scales[:, None],
            grad_b * scales,
        )

        return clipped, norms, scales

    @staticmethod
    def sum_per_example_grads(
        grads: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return (
            np.sum(grads[0], axis=0),
            np.sum(grads[1], axis=0),
            float(np.sum(grads[2])),
        )

    def feature_dp_full_batch_step(
        self,
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        rng: np.random.Generator,
    ) -> dict:
        self._check_initialized()

        if self.l2_reg_strength_ is None:
            raise RuntimeError("l2_reg_strength_ is not initialized.")

        n_train = len(y_train)

        # Public gradient: computed using public-only logit.
        public_grads = self.compute_per_example_public_gradients(
            X_pub_train,
            y_train,
        )

        clipped_public_grads, public_norms, public_scales = self.clip_per_example_grads(
            public_grads,
            self.clip_norm_public,
        )

        sum_pub_w_pub, sum_pub_w_priv, sum_pub_b = self.sum_per_example_grads(
            clipped_public_grads
        )

        # Private residual gradient: full gradient - public gradient.
        private_residual_grads = self.compute_per_example_private_residual_gradients(
            X_pub_train,
            X_priv_train,
            y_train,
        )

        clipped_private_grads, private_norms, private_scales = self.clip_per_example_grads(
            private_residual_grads,
            self.clip_norm_private,
        )

        sum_priv_w_pub, sum_priv_w_priv, sum_priv_b = self.sum_per_example_grads(
            clipped_private_grads
        )

        # Noise is added only to the private residual gradient.
        noise_std = self.noise_multiplier * self.clip_norm_private

        noisy_sum_priv_w_pub = sum_priv_w_pub + rng.normal(
            loc=0.0,
            scale=noise_std,
            size=sum_priv_w_pub.shape,
        )

        noisy_sum_priv_w_priv = sum_priv_w_priv + rng.normal(
            loc=0.0,
            scale=noise_std,
            size=sum_priv_w_priv.shape,
        )

        noisy_sum_priv_b = sum_priv_b
        if self.fit_intercept:
            noisy_sum_priv_b = noisy_sum_priv_b + float(
                rng.normal(loc=0.0, scale=noise_std)
            )

        # Average gradients.
        grad_pub_w_pub = sum_pub_w_pub / n_train
        grad_pub_w_priv = sum_pub_w_priv / n_train
        grad_pub_b = sum_pub_b / n_train

        grad_priv_w_pub = noisy_sum_priv_w_pub / n_train
        grad_priv_w_priv = noisy_sum_priv_w_priv / n_train
        grad_priv_b = noisy_sum_priv_b / n_train

        total_grad_w_pub = grad_pub_w_pub + grad_priv_w_pub
        total_grad_w_priv = grad_pub_w_priv + grad_priv_w_priv
        total_grad_b = grad_pub_b + grad_priv_b

        # Deterministic L2 gradient after clipping/noising.
        # Bias is not regularized.
        if self.l2_reg_handling == "add_after_noise":
            total_grad_w_pub = total_grad_w_pub + self.l2_reg_strength_ * self.w_pub_
            total_grad_w_priv = total_grad_w_priv + self.l2_reg_strength_ * self.w_priv_

        self.w_pub_ = self.w_pub_ - self.learning_rate * total_grad_w_pub
        self.w_priv_ = self.w_priv_ - self.learning_rate * total_grad_w_priv

        if self.fit_intercept:
            self.b_ = float(self.b_ - self.learning_rate * total_grad_b)
        else:
            self.b_ = 0.0

        return {
            "batch_size": int(n_train),
            "public_clipping_enabled": bool(self.clip_norm_public is not None),
            "clip_norm_public": np.nan if self.clip_norm_public is None else float(self.clip_norm_public),
            "clip_norm_private": float(self.clip_norm_private),
            "mean_public_grad_norm_before_clip": float(np.mean(public_norms)),
            "max_public_grad_norm_before_clip": float(np.max(public_norms)),
            "fraction_public_clipped": float(np.mean(public_scales < 1.0)),
            "mean_private_residual_grad_norm_before_clip": float(np.mean(private_norms)),
            "max_private_residual_grad_norm_before_clip": float(np.max(private_norms)),
            "fraction_private_residual_clipped": float(np.mean(private_scales < 1.0)),
            "total_grad_w_public_norm": float(np.linalg.norm(total_grad_w_pub)),
            "total_grad_w_private_norm": float(np.linalg.norm(total_grad_w_priv)),
            "total_grad_b_abs": float(abs(total_grad_b)),
            "w_public_l2_norm": float(np.linalg.norm(self.w_pub_)),
            "w_private_l2_norm": float(np.linalg.norm(self.w_priv_)),
        }

    def _make_history_row(
        self,
        *,
        step: int,
        update_stats: dict,
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        X_pub_val: Optional[np.ndarray],
        X_priv_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        X_pub_test: Optional[np.ndarray],
        X_priv_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
    ) -> dict:
        row = {
            "step": int(step),
            "learning_rate": self.learning_rate,
            **update_stats,
        }

        train_metrics, _, _ = self.evaluate_dataset(
            X_pub_train,
            X_priv_train,
            y_train,
            prefix="train",
        )
        row.update(train_metrics)

        if X_pub_val is not None and X_priv_val is not None and y_val is not None:
            val_metrics, _, _ = self.evaluate_dataset(
                X_pub_val,
                X_priv_val,
                y_val,
                prefix="val",
            )
            row.update(val_metrics)

        if X_pub_test is not None and X_priv_test is not None and y_test is not None:
            test_metrics, _, _ = self.evaluate_dataset(
                X_pub_test,
                X_priv_test,
                y_test,
                prefix="test",
            )
            row.update(test_metrics)

        return row

    def _fit_numba(
        self,
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_pub_val: Optional[np.ndarray] = None,
        X_priv_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_pub_test: Optional[np.ndarray] = None,
        X_priv_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "FeatureDPLogisticRegressionBinary":
        """
        Numba-accelerated version of the same full-batch Feature-DP update.

        The Python reference path is left intact. This path pre-generates the
        Gaussian DP noise using numpy.default_rng, then passes those arrays into
        the JIT kernel. This keeps the DP mechanism and draw order aligned with
        the reference implementation while avoiding Python-level loops inside
        the repeated update.
        """
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is not available, so _fit_numba cannot run.")

        X_pub_train = np.asarray(X_pub_train, dtype=np.float64)
        X_priv_train = np.asarray(X_priv_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if len(X_pub_train) != len(X_priv_train) or len(X_pub_train) != len(y_train):
            raise ValueError("X_pub_train, X_priv_train, and y_train must have same row count.")

        n_train = len(y_train)
        n_public_features = X_pub_train.shape[1]
        n_private_features = X_priv_train.shape[1]

        has_val = X_pub_val is not None and X_priv_val is not None and y_val is not None
        has_test = X_pub_test is not None and X_priv_test is not None and y_test is not None

        if has_val:
            X_pub_val_arr = np.asarray(X_pub_val, dtype=np.float64)
            X_priv_val_arr = np.asarray(X_priv_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=np.float64).reshape(-1)
        else:
            X_pub_val_arr = np.zeros((0, n_public_features), dtype=np.float64)
            X_priv_val_arr = np.zeros((0, n_private_features), dtype=np.float64)
            y_val_arr = np.zeros(0, dtype=np.float64)

        if has_test:
            X_pub_test_arr = np.asarray(X_pub_test, dtype=np.float64)
            X_priv_test_arr = np.asarray(X_priv_test, dtype=np.float64)
            y_test_arr = np.asarray(y_test, dtype=np.float64).reshape(-1)
        else:
            X_pub_test_arr = np.zeros((0, n_public_features), dtype=np.float64)
            X_priv_test_arr = np.zeros((0, n_private_features), dtype=np.float64)
            y_test_arr = np.zeros(0, dtype=np.float64)

        noise_w_pub, noise_w_priv, noise_b = generate_feature_dp_noise_arrays(
            random_state=self.random_state,
            iterations=self.iterations,
            n_public_features=n_public_features,
            n_private_features=n_private_features,
            noise_multiplier=self.noise_multiplier,
            clip_norm_private=self.clip_norm_private,
            fit_intercept=self.fit_intercept,
        )

        n_records = _count_recorded_steps(
            iterations=self.iterations,
            record_every_n_steps=self.record_every_n_steps,
        )
        history_arr = np.full(
            (n_records, len(JIT_HISTORY_COLUMNS)),
            np.nan,
            dtype=np.float64,
        )

        clip_norm_public_enabled = self.clip_norm_public is not None
        clip_norm_public_value = (
            np.nan if self.clip_norm_public is None else float(self.clip_norm_public)
        )

        w_pub, w_priv, b, l2_reg_strength = _jit_train_feature_dp_full_batch(
            X_pub_train,
            X_priv_train,
            y_train,
            bool(has_val),
            X_pub_val_arr,
            X_priv_val_arr,
            y_val_arr,
            bool(has_test),
            X_pub_test_arr,
            X_priv_test_arr,
            y_test_arr,
            noise_w_pub,
            noise_w_priv,
            noise_b,
            self.learning_rate,
            bool(clip_norm_public_enabled),
            float(clip_norm_public_value) if clip_norm_public_enabled else np.nan,
            self.clip_norm_private,
            self.iterations,
            self.C,
            self.fit_intercept,
            self.record_every_n_steps,
            self.stability_nu,
            history_arr,
        )

        self.w_pub_ = np.asarray(w_pub, dtype=np.float64)
        self.w_priv_ = np.asarray(w_priv, dtype=np.float64)
        self.b_ = float(b)
        self.l2_reg_strength_ = float(l2_reg_strength)

        self.step_history_ = pd.DataFrame(history_arr, columns=JIT_HISTORY_COLUMNS)

        # Fill the last history row with the full sklearn-based metrics used by
        # selection_score_from_row. This preserves the original selection/reporting
        # behavior while letting the heavy update loop run under Numba.
        final_idx = self.step_history_.index[-1]

        train_metrics, _, _ = self.evaluate_dataset(
            X_pub_train,
            X_priv_train,
            y_train,
            prefix="train",
        )
        for key, value in train_metrics.items():
            self.step_history_.loc[final_idx, key] = value

        if has_val:
            val_metrics, _, _ = self.evaluate_dataset(
                X_pub_val_arr,
                X_priv_val_arr,
                y_val_arr,
                prefix="val",
            )
            for key, value in val_metrics.items():
                self.step_history_.loc[final_idx, key] = value

        if has_test:
            test_metrics, _, _ = self.evaluate_dataset(
                X_pub_test_arr,
                X_priv_test_arr,
                y_test_arr,
                prefix="test",
            )
            for key, value in test_metrics.items():
                self.step_history_.loc[final_idx, key] = value

        return self


    def fit(
        self,
        X_pub_train: np.ndarray,
        X_priv_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_pub_val: Optional[np.ndarray] = None,
        X_priv_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_pub_test: Optional[np.ndarray] = None,
        X_priv_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "FeatureDPLogisticRegressionBinary":
        if USE_NUMBA_JIT and NUMBA_AVAILABLE:
            return self._fit_numba(
                X_pub_train,
                X_priv_train,
                y_train,
                X_pub_val=X_pub_val,
                X_priv_val=X_priv_val,
                y_val=y_val,
                X_pub_test=X_pub_test,
                X_priv_test=X_priv_test,
                y_test=y_test,
            )

        X_pub_train = np.asarray(X_pub_train, dtype=np.float64)
        X_priv_train = np.asarray(X_priv_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if len(X_pub_train) != len(X_priv_train) or len(X_pub_train) != len(y_train):
            raise ValueError("X_pub_train, X_priv_train, and y_train must have same row count.")

        n_train = len(y_train)

        self._initialize_parameters(
            n_public_features=X_pub_train.shape[1],
            n_private_features=X_priv_train.shape[1],
        )

        self.l2_reg_strength_ = 1.0 / (self.C * n_train)

        rng = np.random.default_rng(self.random_state)

        rows = []

        zero_stats = {
            "batch_size": int(n_train),
            "public_clipping_enabled": bool(self.clip_norm_public is not None),
            "clip_norm_public": np.nan if self.clip_norm_public is None else float(self.clip_norm_public),
            "clip_norm_private": float(self.clip_norm_private),
            "mean_public_grad_norm_before_clip": np.nan,
            "max_public_grad_norm_before_clip": np.nan,
            "fraction_public_clipped": np.nan,
            "mean_private_residual_grad_norm_before_clip": np.nan,
            "max_private_residual_grad_norm_before_clip": np.nan,
            "fraction_private_residual_clipped": np.nan,
            "total_grad_w_public_norm": np.nan,
            "total_grad_w_private_norm": np.nan,
            "total_grad_b_abs": np.nan,
            "w_public_l2_norm": float(np.linalg.norm(self.w_pub_)),
            "w_private_l2_norm": float(np.linalg.norm(self.w_priv_)),
        }

        rows.append(
            self._make_history_row(
                step=0,
                update_stats=zero_stats,
                X_pub_train=X_pub_train,
                X_priv_train=X_priv_train,
                y_train=y_train,
                X_pub_val=X_pub_val,
                X_priv_val=X_priv_val,
                y_val=y_val,
                X_pub_test=X_pub_test,
                X_priv_test=X_priv_test,
                y_test=y_test,
            )
        )

        for step in range(1, self.iterations + 1):
            update_stats = self.feature_dp_full_batch_step(
                X_pub_train,
                X_priv_train,
                y_train,
                rng,
            )

            if step == 1 or step == self.iterations or step % self.record_every_n_steps == 0:
                rows.append(
                    self._make_history_row(
                        step=step,
                        update_stats=update_stats,
                        X_pub_train=X_pub_train,
                        X_priv_train=X_priv_train,
                        y_train=y_train,
                        X_pub_val=X_pub_val,
                        X_priv_val=X_priv_val,
                        y_val=y_val,
                        X_pub_test=X_pub_test,
                        X_priv_test=X_priv_test,
                        y_test=y_test,
                    )
                )

        self.step_history_ = pd.DataFrame(rows)

        return self


# =========================================================
# Training helpers
# =========================================================
def make_hyperparameter_config_seed(
    base_seed: int,
    lr: float,
    *,
    clip_norm_private: Optional[float] = None,
    final: bool = False,
) -> int:
    key = f"lr={lr:.16g}"
    if clip_norm_private is not None:
        key += f"|clip_private={clip_norm_private:.16g}"

    # Stable across processes/machines, unlike Python's built-in hash().
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    config_hash_key = int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)

    offset = 123457 if final else 0
    return int((base_seed * 1000003 + config_hash_key + offset) % (2**32 - 1))


def train_one_hyperparameter_config(
    *,
    learning_rate: float,
    clip_norm_private: float,
    X_pub_train: np.ndarray,
    X_priv_train: np.ndarray,
    y_train: np.ndarray,
    X_pub_val: np.ndarray,
    X_priv_val: np.ndarray,
    y_val: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
) -> tuple[dict, pd.DataFrame, FeatureDPLogisticRegressionBinary]:
    model = FeatureDPLogisticRegressionBinary(
        learning_rate=learning_rate,
        clip_norm_public=CLIP_NORM_PUBLIC,
        clip_norm_private=clip_norm_private,
        noise_multiplier=noise_multiplier,
        iterations=ITERATIONS,
        C=tuned_C,
        fit_intercept=TUNED_FIT_INTERCEPT,
        l2_reg_handling=L2_REG_HANDLING,
        random_state=make_hyperparameter_config_seed(
            seed,
            learning_rate,
            clip_norm_private=clip_norm_private,
            final=False,
        ),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
    )

    # During hyperparameter tuning, use only train and validation.
    # Do not pass the test set here.
    model.fit(
        X_pub_train,
        X_priv_train,
        y_train,
        X_pub_val=X_pub_val,
        X_priv_val=X_priv_val,
        y_val=y_val,
    )

    history = model.step_history_.copy()
    last_row = history.iloc[-1].to_dict()

    final_metrics = {
        "learning_rate": learning_rate,
        "clip_norm_private": clip_norm_private,
        "phase": "selection_train_model",
        **last_row,
    }

    final_metrics["selection_score"] = selection_score_from_row(
        final_metrics,
        metric=SELECTION_METRIC,
        prefix="val",
    )

    return final_metrics, history, model


def evaluate_hyperparameter_grid_with_lr_expansion(
    *,
    initial_learning_rates: list[float],
    private_clip_norm_grid: list[float],
    X_pub_train: np.ndarray,
    X_priv_train: np.ndarray,
    y_train: np.ndarray,
    X_pub_val: np.ndarray,
    X_priv_val: np.ndarray,
    y_val: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
    out_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[tuple[float, float], pd.DataFrame],
    dict[tuple[float, float], FeatureDPLogisticRegressionBinary],
]:
    tried_config_rows: dict[tuple[float, float], dict] = {}
    config_histories: dict[tuple[float, float], pd.DataFrame] = {}
    config_models: dict[tuple[float, float], FeatureDPLogisticRegressionBinary] = {}

    current_learning_rates = sorted(set(float(x) for x in initial_learning_rates))
    private_clip_norm_grid = sorted(set(float(x) for x in private_clip_norm_grid))

    for expansion_round in range(MAX_ENDPOINT_EXPANSIONS + 1):
        print(f"\nHyperparameter grid round {expansion_round + 1}:")
        print("Learning rates:", current_learning_rates)
        print("Private clip norms:", private_clip_norm_grid)

        hyperparameter_configs_to_run = []
        for lr in current_learning_rates:
            for clip_norm_private in private_clip_norm_grid:
                hyperparameter_config = (float(lr), float(clip_norm_private))
                if hyperparameter_config not in tried_config_rows:
                    hyperparameter_configs_to_run.append(hyperparameter_config)

        def _run_single_hyperparameter_config(hyperparameter_config: tuple[float, float]):
            lr, clip_norm_private = hyperparameter_config
            print(
                f"Training config: lr={lr:g}, "
                f"clip_norm_private={clip_norm_private:g}"
            )

            row, hist, model = train_one_hyperparameter_config(
                learning_rate=lr,
                clip_norm_private=clip_norm_private,
                X_pub_train=X_pub_train,
                X_priv_train=X_priv_train,
                y_train=y_train,
                X_pub_val=X_pub_val,
                X_priv_val=X_priv_val,
                y_val=y_val,
                noise_multiplier=noise_multiplier,
                seed=seed,
                tuned_C=tuned_C,
            )

            return hyperparameter_config, row, hist, model

        if hyperparameter_configs_to_run:
            if PARALLELIZE_HYPERPARAMETER_CONFIGS and JOBLIB_AVAILABLE and len(hyperparameter_configs_to_run) > 1:
                print(
                    f"Parallel hyperparameter configuration testing enabled: "
                    f"n_jobs={N_PARALLEL_JOBS}, backend={PARALLEL_BACKEND}."
                )
                results = Parallel(
                    n_jobs=N_PARALLEL_JOBS,
                    backend=PARALLEL_BACKEND,
                )(
                    delayed(_run_single_hyperparameter_config)(hyperparameter_config)
                    for hyperparameter_config in hyperparameter_configs_to_run
                )
            elif PARALLELIZE_HYPERPARAMETER_CONFIGS and not JOBLIB_AVAILABLE:
                print("PARALLELIZE_HYPERPARAMETER_CONFIGS=True, but joblib is unavailable. Falling back to sequential.")
                results = [_run_single_hyperparameter_config(hyperparameter_config) for hyperparameter_config in hyperparameter_configs_to_run]
            else:
                results = [_run_single_hyperparameter_config(hyperparameter_config) for hyperparameter_config in hyperparameter_configs_to_run]
        else:
            results = []

        for hyperparameter_config, row, hist, model in results:
            lr, clip_norm_private = hyperparameter_config

            tried_config_rows[hyperparameter_config] = row
            config_histories[hyperparameter_config] = hist
            config_models[hyperparameter_config] = model

            hist.to_csv(
                out_dir / f"step_history_lr_{lr:g}_clippriv_{clip_norm_private:g}.csv",
                index=False,
            )

            print(
                f"  lr={lr:g} | "
                f"clip_priv={clip_norm_private:g} | "
                f"val_acc={row['val_accuracy']:.6f} | "
                f"val_bal_acc={row['val_balanced_accuracy']:.6f} | "
                f"val_log_loss={row['val_log_loss']:.6f}"
            )

        result_df = pd.DataFrame(
            [tried_config_rows[key] for key in sorted(tried_config_rows)]
        ).reset_index(drop=True)

        best_idx = int(result_df["selection_score"].idxmax())
        best_lr = float(result_df.loc[best_idx, "learning_rate"])

        ordered_lrs = sorted(set(current_learning_rates))

        if len(ordered_lrs) >= 2 and best_lr == ordered_lrs[0]:
            new_lr = ordered_lrs[0] / ENDPOINT_EXPANSION_FACTOR
            if new_lr not in ordered_lrs:
                current_learning_rates = sorted(set(ordered_lrs + [new_lr]))
                print(f"Best LR is lower endpoint. Expanding downward by /{ENDPOINT_EXPANSION_FACTOR:g}: {new_lr:g}")
                continue

        if len(ordered_lrs) >= 2 and best_lr == ordered_lrs[-1]:
            new_lr = ordered_lrs[-1] * ENDPOINT_EXPANSION_FACTOR
            if new_lr not in ordered_lrs:
                current_learning_rates = sorted(set(ordered_lrs + [new_lr]))
                print(f"Best LR is upper endpoint. Expanding upward by *{ENDPOINT_EXPANSION_FACTOR:g}: {new_lr:g}")
                continue

        break

    result_df = pd.DataFrame(
        [tried_config_rows[key] for key in sorted(tried_config_rows)]
    ).reset_index(drop=True)

    return result_df, config_histories, config_models


def train_final_refit_model(
    *,
    best_lr: float,
    best_clip_norm_private: float,
    X_pub_train_val: np.ndarray,
    X_priv_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_pub_test: np.ndarray,
    X_priv_test: np.ndarray,
    y_test: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
) -> tuple[FeatureDPLogisticRegressionBinary, pd.DataFrame, dict, dict, np.ndarray, np.ndarray]:
    final_model = FeatureDPLogisticRegressionBinary(
        learning_rate=best_lr,
        clip_norm_public=CLIP_NORM_PUBLIC,
        clip_norm_private=best_clip_norm_private,
        noise_multiplier=noise_multiplier,
        iterations=ITERATIONS,
        C=tuned_C,
        fit_intercept=TUNED_FIT_INTERCEPT,
        l2_reg_handling=L2_REG_HANDLING,
        random_state=make_hyperparameter_config_seed(
            seed,
            best_lr,
            clip_norm_private=best_clip_norm_private,
            final=True,
        ),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
    )

    final_model.fit(
        X_pub_train_val,
        X_priv_train_val,
        y_train_val,
        X_pub_test=X_pub_test,
        X_priv_test=X_priv_test,
        y_test=y_test,
    )

    final_history = final_model.step_history_.copy()

    train_val_metrics, _, _ = final_model.evaluate_dataset(
        X_pub_train_val,
        X_priv_train_val,
        y_train_val,
        prefix="trainval",
    )

    test_metrics, y_test_pred, y_test_proba = final_model.evaluate_dataset(
        X_pub_test,
        X_priv_test,
        y_test,
        prefix="test",
    )

    return (
        final_model,
        final_history,
        train_val_metrics,
        test_metrics,
        y_test_pred,
        y_test_proba,
    )


# =========================================================
# Subgroup metrics
# =========================================================
def build_reporting_group_series(df_meta: pd.DataFrame, feature_name: str) -> pd.Series:
    if feature_name not in df_meta.columns:
        return pd.Series(
            MISSING_GROUP_LABEL,
            index=df_meta.index,
            dtype="string",
            name=feature_name,
        )

    s = df_meta[feature_name].copy()

    if feature_name in {"SITE_ID", "FIQ_TEST_TYPE", "PIQ_TEST_TYPE", "VIQ_TEST_TYPE"}:
        s = s.map(
            normalize_test_type
            if feature_name.endswith("_TEST_TYPE")
            else lambda x: np.nan if pd.isna(x) else str(x).strip().upper()
        )
        s = pd.Series(s, index=df_meta.index, name=feature_name, dtype="object")
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)

        return s.astype("string")

    if feature_name == "HANDEDNESS_CATEGORY":
        s = s.map(normalize_handedness_category)
        s = pd.Series(s, index=df_meta.index, name=feature_name, dtype="object")
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)

        mapping = {
            "R": "Right",
            "L": "Left",
            "Ambi": "Mixed/Ambi",
            "MIXED": "Mixed/Ambi",
            "L->R": "Mixed/Ambi",
        }

        s = s.map(lambda x: mapping.get(x, x))

        return s.astype("string")

    if feature_name == "SEX":
        s = pd.to_numeric(s, errors="coerce")
        s = s.map({1.0: "Male", 2.0: "Female"})
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)

        return s.astype("string")

    if feature_name == "EYE_STATUS_AT_SCAN":
        s = pd.to_numeric(s, errors="coerce")
        s = s.map({1.0: "Eyes Open", 2.0: "Eyes Closed"})
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)

        return s.astype("string")

    if feature_name == "CURRENT_MED_STATUS":
        s = pd.to_numeric(s, errors="coerce")
        s = s.map({0.0: "No Current Meds", 1.0: "On Current Meds"})
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)

        return s.astype("string")

    if feature_name in SENSIBLE_NUMERIC_BINS:
        spec = SENSIBLE_NUMERIC_BINS[feature_name]

        s_num = pd.to_numeric(s, errors="coerce")
        s_binned = pd.cut(
            s_num,
            bins=spec["bins"],
            labels=spec["labels"],
            right=False,
            include_lowest=True,
        ).astype("object")

        s_binned = pd.Series(
            s_binned,
            index=df_meta.index,
            name=feature_name,
            dtype="object",
        )

        if feature_name in {"FIQ", "PIQ", "VIQ"}:
            s_binned = s_binned.where(
                ~pd.isna(s_binned),
                f"{feature_name} {MISSING_GROUP_LABEL}",
            )

        return s_binned.astype("string")

    s = s.astype("string")
    s = s.fillna(MISSING_GROUP_LABEL)
    s = s.replace(
        {
            "": MISSING_GROUP_LABEL,
            "nan": MISSING_GROUP_LABEL,
            "None": MISSING_GROUP_LABEL,
        }
    )

    return s


def get_group_order_for_feature(feature_name: str, group_series: pd.Series) -> list[str]:
    values = pd.Series(group_series, dtype="string").dropna().astype(str).tolist()
    values_set = set(values)

    if feature_name in {"AGE_AT_SCAN", "FIQ", "PIQ", "VIQ"}:
        ordered = list(SENSIBLE_NUMERIC_BINS[feature_name]["labels"])

        if feature_name in {"FIQ", "PIQ", "VIQ"}:
            ordered.append(f"{feature_name} {MISSING_GROUP_LABEL}")

        return [x for x in ordered if x in values_set]

    return sorted(values_set)


def compute_subgroup_metrics(
    *,
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: list[str],
    experiment_key: str,
    experiment_name: str,
    split_name: str,
    best_lr: float,
    best_clip_norm_private: float,
    target_epsilon: float,
    achieved_epsilon: float,
    delta: float,
    noise_multiplier: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    rows = []

    for feature_name in feature_names:
        group_series = build_reporting_group_series(df_meta, feature_name)
        group_order = get_group_order_for_feature(feature_name, group_series)

        for group_value in group_order:
            mask = group_series.eq(group_value).to_numpy()

            if not np.any(mask):
                continue

            metrics = compute_prediction_metrics_from_arrays(
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                y_proba=y_proba[mask],
            )

            rows.append(
                {
                    "experiment_key": experiment_key,
                    "experiment_name": experiment_name,
                    "split": split_name,
                    "best_lr": best_lr,
                    "best_clip_norm_private": best_clip_norm_private,
                    "target_epsilon": target_epsilon,
                    "achieved_epsilon": achieved_epsilon,
                    "delta": delta,
                    "noise_multiplier": noise_multiplier,
                    "feature": feature_name,
                    "group": str(group_value),
                    "n_samples": metrics["n_total"],
                    "n_control": metrics["n_control"],
                    "n_autism": metrics["n_autism"],
                    "autism_prevalence": metrics["autism_prevalence"],
                    "predicted_autism_rate": metrics["predicted_autism_rate"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "log_loss": metrics["log_loss"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "n_unique_classes": metrics["n_unique_classes"],
                    "control_precision": metrics.get("control_precision", np.nan),
                    "control_recall": metrics.get("control_recall", np.nan),
                    "control_f1-score": metrics.get("control_f1-score", np.nan),
                    "autism_precision": metrics.get("autism_precision", np.nan),
                    "autism_recall": metrics.get("autism_recall", np.nan),
                    "autism_f1-score": metrics.get("autism_f1-score", np.nan),
                    "cm_true_control_pred_control": metrics["cm_true_control_pred_control"],
                    "cm_true_control_pred_autism": metrics["cm_true_control_pred_autism"],
                    "cm_true_autism_pred_control": metrics["cm_true_autism_pred_control"],
                    "cm_true_autism_pred_autism": metrics["cm_true_autism_pred_autism"],
                }
            )

    combined_df = pd.DataFrame(rows)

    if combined_df.empty:
        return combined_df

    feature_order = [f for f in SUBGROUP_EVAL_FEATURES if f in feature_names]

    combined_df["feature"] = pd.Categorical(
        combined_df["feature"],
        categories=feature_order,
        ordered=True,
    )

    combined_df = combined_df.sort_values(
        ["experiment_key", "split", "feature"],
        kind="stable",
    ).reset_index(drop=True)

    combined_df["feature"] = combined_df["feature"].astype(str)

    return combined_df


# =========================================================
# Plot helpers
# =========================================================
def save_or_show(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_step_metric_overlay(
    config_histories: dict,
    *,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False

    def sort_key(item):
        hyperparameter_config, _ = item
        if isinstance(hyperparameter_config, tuple):
            return hyperparameter_config
        return (float(hyperparameter_config), -1.0)

    for hyperparameter_config, hist in sorted(config_histories.items(), key=sort_key):
        if hist is None or hist.empty or metric_col not in hist.columns:
            continue

        if isinstance(hyperparameter_config, tuple):
            lr, clip_priv = hyperparameter_config
            label = f"lr={lr:g}, Cpriv={clip_priv:g}"
        else:
            label = f"lr={float(hyperparameter_config):g}"

        plt.plot(hist["step"], hist[metric_col], label=label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Feature-DP step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(out_path)


def save_all_step_plots(config_histories: dict[tuple[float, float], pd.DataFrame], out_dir: Path) -> None:
    specs = [
        ("train_accuracy", "Train accuracy", "Train accuracy vs step"),
        ("train_log_loss", "Train log loss", "Train log loss vs step"),
        ("val_accuracy", "Validation accuracy", "Validation accuracy vs step"),
        ("val_log_loss", "Validation log loss", "Validation log loss vs step"),
        ("train_objective", "Train objective", "Train objective vs step"),
        ("val_objective", "Validation objective", "Validation objective vs step"),
    ]

    for metric_col, ylabel, title in specs:
        save_step_metric_overlay(
            config_histories,
            metric_col=metric_col,
            ylabel=ylabel,
            title=title,
            out_path=out_dir / f"{metric_col}_vs_step_all_configs.png",
        )


def save_metric_vs_lr_plots(selection_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = selection_df.sort_values(["clip_norm_private", "learning_rate"]).copy()

    def plot_train_val(metric: str, ylabel: str, title: str, filename: str) -> None:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"

        if train_col not in plot_df.columns or val_col not in plot_df.columns:
            return

        plt.figure(figsize=(10, 6))

        for clip_norm_private, group_df in plot_df.groupby("clip_norm_private"):
            group_df = group_df.sort_values("learning_rate")

            plt.semilogx(
                group_df["learning_rate"],
                group_df[train_col],
                marker="o",
                linestyle="--",
                label=f"Train | Cpriv={clip_norm_private:g}",
            )

            plt.semilogx(
                group_df["learning_rate"],
                group_df[val_col],
                marker="o",
                label=f"Val | Cpriv={clip_norm_private:g}",
            )

        plt.xlabel("Learning rate")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        save_or_show(out_dir / filename)

    plot_train_val(
        "accuracy",
        "Accuracy",
        "Final train/validation accuracy vs LR and private clip norm",
        "accuracy_vs_lr_and_private_clip_norm.png",
    )
    plot_train_val(
        "log_loss",
        "Log loss",
        "Final train/validation log loss vs LR and private clip norm",
        "log_loss_vs_lr_and_private_clip_norm.png",
    )
    plot_train_val(
        "balanced_accuracy",
        "Balanced accuracy",
        "Final train/validation balanced accuracy vs LR and private clip norm",
        "balanced_accuracy_vs_lr_and_private_clip_norm.png",
    )
    plot_train_val(
        "roc_auc",
        "AuROC",
        "Final train/validation AuROC vs LR and private clip norm",
        "roc_auc_vs_lr_and_private_clip_norm.png",
    )


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    split_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Control (0)", "Autism (1)"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"{split_label} confusion matrix")
    save_or_show(out_dir / f"{split_label.lower()}_confusion_matrix.png")


def save_roc_pr_plots(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    split_label: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{split_label} ROC curve")
    plt.legend()
    save_or_show(out_dir / f"{split_label.lower()}_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"Average precision = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_label} precision-recall curve")
    plt.legend()
    save_or_show(out_dir / f"{split_label.lower()}_pr_curve.png")


def save_calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    split_label: str,
) -> None:
    plt.figure(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_true,
        y_proba,
        n_bins=10,
        strategy="quantile",
    )
    plt.title(f"{split_label} calibration curve")
    save_or_show(out_dir / f"{split_label.lower()}_calibration_curve.png")


def save_probability_histogram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    split_label: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(y_proba[y_true == 0], bins=20, alpha=0.6, label="Control (0)")
    plt.hist(y_proba[y_true == 1], bins=20, alpha=0.6, label="Autism (1)")
    plt.xlabel("Predicted probability of Autism")
    plt.ylabel("Count")
    plt.title(f"{split_label} predicted-probability distribution by class")
    plt.legend()
    save_or_show(out_dir / f"{split_label.lower()}_probability_histogram.png")


def save_coefficients(
    public_feature_names: list[str],
    private_feature_names: list[str],
    model: FeatureDPLogisticRegressionBinary,
    out_dir: Path,
    top_k: int = 20,
) -> pd.DataFrame:
    if model.w_pub_ is None or model.w_priv_ is None:
        raise RuntimeError("Model is not fitted.")

    public_df = pd.DataFrame(
        {
            "feature": public_feature_names,
            "coefficient": model.w_pub_,
            "abs_coefficient": np.abs(model.w_pub_),
            "modality": "public_fmri",
        }
    )

    private_df = pd.DataFrame(
        {
            "feature": private_feature_names,
            "coefficient": model.w_priv_,
            "abs_coefficient": np.abs(model.w_priv_),
            "modality": "private_phenotype",
        }
    )

    coef_df = pd.concat([public_df, private_df], axis=0, ignore_index=True)
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(out_dir / "final_feature_dp_coefficients_all.csv", index=False)

    top_pos = coef_df.sort_values("coefficient", ascending=False).head(top_k)
    top_neg = coef_df.sort_values("coefficient", ascending=True).head(top_k)
    plot_df = pd.concat([top_neg, top_pos], axis=0)

    plt.figure(figsize=(13, 9))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title("Most influential Feature-DP logistic-regression coefficients")
    save_or_show(out_dir / "final_feature_dp_top_coefficients.png")

    modality_importance = (
        coef_df.groupby("modality", as_index=False)["abs_coefficient"]
        .sum()
        .sort_values("abs_coefficient", ascending=False)
    )
    modality_importance.to_csv(
        out_dir / "final_feature_dp_coefficient_modality_importance.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.bar(modality_importance["modality"], modality_importance["abs_coefficient"])
    plt.xlabel("Feature block")
    plt.ylabel("Sum of absolute coefficients")
    plt.title("Feature-DP coefficient importance by feature block")
    save_or_show(out_dir / "final_feature_dp_coefficient_modality_importance.png")

    return coef_df


# =========================================================
# Global data preparation
# =========================================================
def prepare_global_data(*, seed: int, output_dir: Path) -> PreparedGlobalData:
    """Prepare data once and create/load the fixed train/validation/test split.

    The data loading, cleaning, valid-mask filtering, domain scaling, and
    feature-list preparation are kept aligned with the non-private reference
    workflow.

    Split policy:
        - USE_PREPROCESS_DUMP=False:
            Create exactly one IID stratified train/validation/test split using
            this function's seed argument. In main(), this is called only for
            LRC_SEED and the resulting indices are reused for every run seed.
        - USE_PREPROCESS_DUMP=True:
            Recover train/validation/test indices from the dumped non-private
            fixed-split CSVs and reuse those exact samples for every run seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if PHENOTYPE_OTHER_NUMERIC_MODE not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError(
            f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={PHENOTYPE_OTHER_NUMERIC_MODE!r}. "
            f"Allowed values: {sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES)}"
        )

    df_raw = load_abide_dataframe(CSV_PATH)
    df = clean_abide_dataframe(df_raw)

    if "HANDEDNESS_CATEGORY" in df.columns:
        df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].map(
            normalize_handedness_category
        )

    if "SITE_ID" in df.columns:
        df["SITE_ID"] = (
            df["SITE_ID"]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-9999": pd.NA})
        )

    feature_json_path = resolve_feature_json_path(FEATURE_JSON_CANDIDATES)
    feature_groups = load_feature_groups(feature_json_path)

    print("\nFeature JSON path used:")
    print(feature_json_path)

    all_numeric_candidate_cols = get_all_numeric_candidate_columns(feature_groups)

    for col in [TARGET_COL] + all_numeric_candidate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = df[TARGET_COL].isin([1, 2]).to_numpy()

    X_conn_np = load_fc_matrix(X_CONN_PATH)
    X_conn_np = np.asarray(X_conn_np, dtype=np.float64)
    X_conn_np = np.where(np.isfinite(X_conn_np), X_conn_np, np.nan)

    print("\nX_conn shape:", X_conn_np.shape)
    print("Phenotypic dataframe shape:", df.shape)

    if X_conn_np.shape[0] != len(df):
        raise ValueError(
            f"Row mismatch: X_conn has {X_conn_np.shape[0]} rows "
            f"but phenotypic dataframe has {len(df)} rows. "
            "Please ensure X_conn is aligned with abide.phenotypic."
        )

    df_model_raw = df.loc[valid_mask].copy().reset_index(drop=True)
    X_conn_model = X_conn_np[valid_mask]

    df_model = apply_domain_scaling(df_model_raw.copy())

    y = (df_model_raw[TARGET_COL] == 1).astype(int).to_numpy()

    print("\nAfter valid-mask filtering:")
    print("df_model shape:", df_model.shape)
    print("X_conn_model shape:", X_conn_model.shape)
    print("Binary class counts (0=Control, 1=Autism):")
    print(pd.Series(y).value_counts().sort_index())

    if USE_PREPROCESS_DUMP:
        train_idx, val_idx, test_idx, split_name = load_fixed_split_indices_from_preprocess_dump(
            df_model_raw=df_model_raw,
            X_conn_model=X_conn_model,
        )
        split_seed_description = "loaded_from_preprocess_dump"
    else:
        train_idx, val_idx, test_idx, split_name = make_train_val_test_split(
            y=y,
            seed=int(seed),
            validation_size=VALIDATION_SIZE,
            test_size=TEST_SIZE,
        )
        split_name = (
            split_name
            + f"; fixed once with LRC_SEED={int(seed)} and reused for all run seeds"
        )
        split_seed_description = str(int(seed))

    print("\nFixed global split strategy:")
    print(split_name)
    print(f"Split seed/source : {split_seed_description}")
    print(f"Train size        : {len(train_idx)}")
    print(f"Validation size   : {len(val_idx)}")
    print(f"Test size         : {len(test_idx)}")
    print(f"Train class counts      : {dict(pd.Series(y[train_idx]).value_counts().sort_index())}")
    print(f"Validation class counts : {dict(pd.Series(y[val_idx]).value_counts().sort_index())}")
    print(f"Test class counts       : {dict(pd.Series(y[test_idx]).value_counts().sort_index())}")

    np.save(output_dir / "X_conn_model.npy", X_conn_model)
    np.save(output_dir / "y_binary.npy", y)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    return PreparedGlobalData(
        df=df,
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        feature_groups=feature_groups,
        feature_json_path=feature_json_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_name=split_name,
    )

# =========================================================
# One experiment runner
# =========================================================
def run_single_experiment(
    *,
    global_data: PreparedGlobalData,
    experiment_key: str,
    experiment_name: str,
    selected_group_keys: list[str],
    tuned_C: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_dir = OUTPUT_DIR / experiment_key
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 120)
    print(f"Running experiment: {experiment_name}")
    print(f"Using tuned_C: {tuned_C}")
    print("#" * 120)

    feature_blocks = build_feature_blocks_for_experiment(
        df=global_data.df,
        df_model=global_data.df_model,
        X_conn_model=global_data.X_conn_model,
        feature_groups=global_data.feature_groups,
        selected_group_keys=selected_group_keys,
    )

    print("\nSelected feature group keys:")
    print(selected_group_keys)

    print("\nPrivate phenotype columns:")
    print(feature_blocks.selected_phenotype_cols)

    print("\nPhenotype domain-scaled numeric columns:")
    print(feature_blocks.phenotype_domain_scaled_cols)

    print("\nPhenotype other numeric columns:")
    print(feature_blocks.phenotype_other_numeric_cols)

    print("\nPhenotype categorical columns:")
    print(feature_blocks.phenotype_categorical_cols)

    print("\nRequested JSON columns absent from dataframe:")
    print(feature_blocks.missing_cols if feature_blocks.missing_cols else "None")

    train_idx = global_data.train_idx
    val_idx = global_data.val_idx
    test_idx = global_data.test_idx

    public_train_df = feature_blocks.public_df.iloc[train_idx].reset_index(drop=True)
    public_val_df = feature_blocks.public_df.iloc[val_idx].reset_index(drop=True)
    public_test_df = feature_blocks.public_df.iloc[test_idx].reset_index(drop=True)

    private_train_df = feature_blocks.private_df.iloc[train_idx].reset_index(drop=True)
    private_val_df = feature_blocks.private_df.iloc[val_idx].reset_index(drop=True)
    private_test_df = feature_blocks.private_df.iloc[test_idx].reset_index(drop=True)

    y_train = global_data.y[train_idx]
    y_val = global_data.y[val_idx]
    y_test = global_data.y[test_idx]

    public_train_val_df = pd.concat([public_train_df, public_val_df], axis=0).reset_index(drop=True)
    private_train_val_df = pd.concat([private_train_df, private_val_df], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    transformed = fit_transform_public_private(
        public_train_df=public_train_df,
        private_train_df=private_train_df,
        public_val_df=public_val_df,
        private_val_df=private_val_df,
        public_test_df=public_test_df,
        private_test_df=private_test_df,
        fc_cols=feature_blocks.fc_cols,
        phenotype_domain_scaled_cols=feature_blocks.phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=feature_blocks.phenotype_other_numeric_cols,
        phenotype_categorical_cols=feature_blocks.phenotype_categorical_cols,
    )

    X_pub_train = transformed["X_pub_train"]
    X_priv_train = transformed["X_priv_train"]
    X_pub_val = transformed["X_pub_val"]
    X_priv_val = transformed["X_priv_val"]
    X_pub_test = transformed["X_pub_test"]
    X_priv_test = transformed["X_priv_test"]

    private_clip_norm_grid = load_private_clip_norm_grid(
        quantile_csv_path=FEATURE_NORM_QUANTILE_CSV_PATH,
        quantiles=PRIVATE_CLIP_NORM_QUANTILES_FOR_GRID,
        fallback_grid=PRIVATE_CLIP_NORM_GRID_FALLBACK,
    )

    with open(experiment_dir / "private_clip_norm_grid_from_feature_norm_quantiles.json", "w") as f:
        json.dump(
            {
                "feature_norm_quantile_csv_path": str(FEATURE_NORM_QUANTILE_CSV_PATH),
                "use_fixed_private_clip_norm_grid": USE_FIXED_PRIVATE_CLIP_NORM_GRID,
                "private_clip_norm_quantiles_for_grid": PRIVATE_CLIP_NORM_QUANTILES_FOR_GRID,
                "private_clip_norm_grid": private_clip_norm_grid,
            },
            f,
            indent=2,
        )

    # DP accounting for selection models trained on train split.
    n_train = len(y_train)
    selection_delta = float(n_train ** DELTA_POWER)
    selection_sampling = 1.0

    if NOISE_MULTIPLIER is None:
        selection_noise_multiplier = get_noise_multiplier(
            target_epsilon=TARGET_EPSILON,
            target_delta=selection_delta,
            sampling=selection_sampling,
            step=ITERATIONS,
        )
    else:
        selection_noise_multiplier = float(NOISE_MULTIPLIER)

    selection_achieved_epsilon = epsilon_for_noise_multiplier(
        noise_multiplier=selection_noise_multiplier,
        target_delta=selection_delta,
        sampling=selection_sampling,
        step=ITERATIONS,
    )

    selection_dp_summary = {
        "phase": "selection_train_model",
        "mechanism": "Feature DP",
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": selection_achieved_epsilon,
        "delta": selection_delta,
        "noise_multiplier": selection_noise_multiplier,
        "clip_norm_public": None if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
        "clip_norm_private_grid": private_clip_norm_grid,
        "iterations": ITERATIONS,
        "sampling": selection_sampling,
        "n_train_for_accounting": n_train,
        "tuned_C": tuned_C,
        "full_batch": True,
        "public_features": "fMRI FC",
        "private_features": "phenotype",
    }

    with open(experiment_dir / "feature_dp_accounting_selection_train_model.json", "w") as f:
        json.dump(selection_dp_summary, f, indent=2)

    print("\nSelection Feature-DP accounting summary:")
    print(json.dumps(selection_dp_summary, indent=2))

    selection_df, config_histories, config_models = evaluate_hyperparameter_grid_with_lr_expansion(
        initial_learning_rates=LEARNING_RATES_INITIAL,
        private_clip_norm_grid=private_clip_norm_grid,
        X_pub_train=X_pub_train,
        X_priv_train=X_priv_train,
        y_train=y_train,
        X_pub_val=X_pub_val,
        X_priv_val=X_priv_val,
        y_val=y_val,
        noise_multiplier=selection_noise_multiplier,
        seed=SEED,
        tuned_C=tuned_C,
        out_dir=experiment_dir,
    )

    selection_df = selection_df.sort_values(
        ["clip_norm_private", "learning_rate"]
    ).reset_index(drop=True)
    selection_df.insert(0, "experiment_key", experiment_key)
    selection_df.insert(1, "experiment_name", experiment_name)
    selection_df.insert(2, "target_epsilon", TARGET_EPSILON)
    selection_df.insert(3, "achieved_epsilon", selection_achieved_epsilon)
    selection_df.insert(4, "delta", selection_delta)
    selection_df.insert(5, "noise_multiplier", selection_noise_multiplier)
    selection_df.insert(6, "TUNED_C", tuned_C)
    selection_df["clip_norm_public"] = np.nan if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC)

    selection_df.to_csv(
        experiment_dir / "feature_dp_lr_private_clip_selection_metrics_all_configs.csv",
        index=False,
    )

    # Backward-compatible filename.
    selection_df.to_csv(
        experiment_dir / "feature_dp_lr_selection_metrics_all_learning_rates.csv",
        index=False,
    )

    save_all_step_plots(config_histories, experiment_dir)
    save_metric_vs_lr_plots(selection_df, experiment_dir)

    best_score = float(selection_df["selection_score"].max())
    tied_configs = selection_df.loc[
        selection_df["selection_score"] >= best_score - SELECTION_TIE_TOL
    ].copy()
    tied_configs = tied_configs.sort_values(
        ["clip_norm_private", "learning_rate"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_selection_row = tied_configs.iloc[0].to_dict()
    best_lr = float(best_selection_row["learning_rate"])
    best_clip_norm_private = float(best_selection_row["clip_norm_private"])

    with open(experiment_dir / "best_hyperparameter_selection_summary.json", "w") as f:
        json.dump(best_selection_row, f, indent=2)

    # Backward-compatible filename.
    with open(experiment_dir / "best_lr_selection_summary.json", "w") as f:
        json.dump(best_selection_row, f, indent=2)

    print("\nBest hyperparameters selected using validation split:")
    print("Best LR:", best_lr)
    print("Best private clip norm:", best_clip_norm_private)
    print("Best selection score:", best_score)

    best_selection_model = config_models[(best_lr, best_clip_norm_private)]

    _, y_val_pred, y_val_proba = best_selection_model.evaluate_dataset(
        X_pub_val,
        X_priv_val,
        y_val,
        prefix="val",
    )

    save_confusion_matrix_plot(y_val, y_val_pred, experiment_dir, "validation_selected_model")
    save_roc_pr_plots(y_val, y_val_proba, experiment_dir, "validation_selected_model")
    save_calibration_plot(y_val, y_val_proba, experiment_dir, "validation_selected_model")
    save_probability_histogram(y_val, y_val_proba, experiment_dir, "validation_selected_model")

    # Final refit on train + validation using best LR and best private clip norm.
    transformed_final = fit_transform_public_private(
        public_train_df=public_train_val_df,
        private_train_df=private_train_val_df,
        public_val_df=None,
        private_val_df=None,
        public_test_df=public_test_df,
        private_test_df=private_test_df,
        fc_cols=feature_blocks.fc_cols,
        phenotype_domain_scaled_cols=feature_blocks.phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=feature_blocks.phenotype_other_numeric_cols,
        phenotype_categorical_cols=feature_blocks.phenotype_categorical_cols,
    )

    X_pub_train_val = transformed_final["X_pub_train"]
    X_priv_train_val = transformed_final["X_priv_train"]
    X_pub_test_final = transformed_final["X_pub_test"]
    X_priv_test_final = transformed_final["X_priv_test"]

    n_train_val = len(y_train_val)
    final_delta = float(n_train_val ** DELTA_POWER)
    final_sampling = 1.0

    if NOISE_MULTIPLIER is None:
        final_noise_multiplier = get_noise_multiplier(
            target_epsilon=TARGET_EPSILON,
            target_delta=final_delta,
            sampling=final_sampling,
            step=ITERATIONS,
        )
    else:
        final_noise_multiplier = float(NOISE_MULTIPLIER)

    final_achieved_epsilon = epsilon_for_noise_multiplier(
        noise_multiplier=final_noise_multiplier,
        target_delta=final_delta,
        sampling=final_sampling,
        step=ITERATIONS,
    )

    final_dp_summary = {
        "phase": "final_refit_train_validation_model",
        "mechanism": "Feature DP",
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": final_achieved_epsilon,
        "delta": final_delta,
        "noise_multiplier": final_noise_multiplier,
        "clip_norm_public": None if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
        "clip_norm_private": best_clip_norm_private,
        "iterations": ITERATIONS,
        "sampling": final_sampling,
        "n_train_for_accounting": n_train_val,
        "tuned_C": tuned_C,
        "full_batch": True,
        "public_features": "fMRI FC",
        "private_features": "phenotype",
    }

    with open(experiment_dir / "feature_dp_accounting_final_refit_train_validation_model.json", "w") as f:
        json.dump(final_dp_summary, f, indent=2)

    print("\nFinal-refit Feature-DP accounting summary:")
    print(json.dumps(final_dp_summary, indent=2))

    (
        final_model,
        final_history,
        train_val_metrics,
        final_test_metrics,
        y_test_pred_final,
        y_test_proba_final,
    ) = train_final_refit_model(
        best_lr=best_lr,
        best_clip_norm_private=best_clip_norm_private,
        X_pub_train_val=X_pub_train_val,
        X_priv_train_val=X_priv_train_val,
        y_train_val=y_train_val,
        X_pub_test=X_pub_test_final,
        X_priv_test=X_priv_test_final,
        y_test=y_test,
        noise_multiplier=final_noise_multiplier,
        seed=SEED,
        tuned_C=tuned_C,
    )

    final_history.to_csv(experiment_dir / "final_refit_trainval_step_history.csv", index=False)

    final_history_key = (best_lr, best_clip_norm_private)

    save_step_metric_overlay(
        {final_history_key: final_history},
        metric_col="train_accuracy",
        ylabel="Train+validation accuracy",
        title="Final refit train+validation accuracy vs step",
        out_path=experiment_dir / "final_refit_trainval_accuracy_vs_step.png",
    )

    save_step_metric_overlay(
        {final_history_key: final_history},
        metric_col="train_log_loss",
        ylabel="Train+validation log loss",
        title="Final refit train+validation log loss vs step",
        out_path=experiment_dir / "final_refit_trainval_log_loss_vs_step.png",
    )

    save_step_metric_overlay(
        {final_history_key: final_history},
        metric_col="test_accuracy",
        ylabel="Test accuracy",
        title="Final refit test accuracy vs step",
        out_path=experiment_dir / "final_refit_test_accuracy_vs_step.png",
    )

    save_step_metric_overlay(
        {final_history_key: final_history},
        metric_col="test_log_loss",
        ylabel="Test log loss",
        title="Final refit test log loss vs step",
        out_path=experiment_dir / "final_refit_test_log_loss_vs_step.png",
    )

    save_confusion_matrix_plot(y_test, y_test_pred_final, experiment_dir, "test_final_model")
    save_roc_pr_plots(y_test, y_test_proba_final, experiment_dir, "test_final_model")
    save_calibration_plot(y_test, y_test_proba_final, experiment_dir, "test_final_model")
    save_probability_histogram(y_test, y_test_proba_final, experiment_dir, "test_final_model")

    save_coefficients(
        public_feature_names=transformed_final["public_feature_names"],
        private_feature_names=transformed_final["private_feature_names"],
        model=final_model,
        out_dir=experiment_dir,
        top_k=20,
    )

    print_metric_block(
        f"{experiment_name} | Validation metrics for selected train-split model",
        {
            "accuracy": best_selection_row["val_accuracy"],
            "balanced_accuracy": best_selection_row["val_balanced_accuracy"],
            "log_loss": best_selection_row["val_log_loss"],
            "roc_auc": best_selection_row["val_roc_auc"],
            "average_precision": best_selection_row["val_average_precision"],
        },
    )

    print_metric_block(
        f"{experiment_name} | Test metrics for final train+validation refit model",
        {
            "accuracy": final_test_metrics["test_accuracy"],
            "balanced_accuracy": final_test_metrics["test_balanced_accuracy"],
            "log_loss": final_test_metrics["test_log_loss"],
            "roc_auc": final_test_metrics["test_roc_auc"],
            "average_precision": final_test_metrics["test_average_precision"],
        },
    )

    validation_report = classification_report(
        y_val,
        y_val_pred,
        target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
        digits=4,
    )

    test_report = classification_report(
        y_test,
        y_test_pred_final,
        target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
        digits=4,
    )

    (experiment_dir / "validation_selected_model_classification_report.txt").write_text(
        validation_report
    )
    (experiment_dir / "test_final_model_classification_report.txt").write_text(
        test_report
    )

    pd.DataFrame(
        confusion_matrix(y_val, y_val_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "validation_selected_model_confusion_matrix.csv")

    pd.DataFrame(
        confusion_matrix(y_test, y_test_pred_final, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "test_final_model_confusion_matrix.csv")

    subgroup_eval_features_present = [
        c for c in SUBGROUP_EVAL_FEATURES if c in global_data.df_model_raw.columns
    ]

    df_subgroup_test = (
        global_data.df_model_raw.iloc[test_idx][subgroup_eval_features_present]
        .reset_index(drop=True)
    )

    test_subgroup_metrics = compute_subgroup_metrics(
        df_meta=df_subgroup_test,
        y_true=y_test,
        y_pred=y_test_pred_final,
        y_proba=y_test_proba_final,
        feature_names=subgroup_eval_features_present,
        experiment_key=experiment_key,
        experiment_name=experiment_name,
        split_name="test_final_model",
        best_lr=best_lr,
        best_clip_norm_private=best_clip_norm_private,
        target_epsilon=TARGET_EPSILON,
        achieved_epsilon=final_achieved_epsilon,
        delta=final_delta,
        noise_multiplier=final_noise_multiplier,
    )

    if test_subgroup_metrics is not None and not test_subgroup_metrics.empty:
        test_subgroup_metrics.insert(0, "seed", int(SEED))
        test_subgroup_metrics.to_csv(
            experiment_dir / "test_final_model_subgroup_metrics.csv",
            index=False,
        )

    summary_rows = []

    summary_rows.append(
        {
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "validation_selected_model",
            "model_training_data": "train",
            "best_lr": best_lr,
            "best_clip_norm_private": best_clip_norm_private,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": selection_achieved_epsilon,
            "delta": selection_delta,
            "noise_multiplier": selection_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm_public": np.nan if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
            "clip_norm_private": best_clip_norm_private,
            "iterations": ITERATIONS,
            "n_public_fc_features": len(feature_blocks.fc_cols),
            "n_private_phenotype_raw_columns": len(feature_blocks.selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": best_selection_row["val_accuracy"],
            "balanced_accuracy": best_selection_row["val_balanced_accuracy"],
            "log_loss": best_selection_row["val_log_loss"],
            "roc_auc": best_selection_row["val_roc_auc"],
            "average_precision": best_selection_row["val_average_precision"],
        }
    )

    summary_rows.append(
        {
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "train_validation_final_model",
            "model_training_data": "train+validation",
            "best_lr": best_lr,
            "best_clip_norm_private": best_clip_norm_private,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": final_achieved_epsilon,
            "delta": final_delta,
            "noise_multiplier": final_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm_public": np.nan if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
            "clip_norm_private": best_clip_norm_private,
            "iterations": ITERATIONS,
            "n_public_fc_features": len(feature_blocks.fc_cols),
            "n_private_phenotype_raw_columns": len(feature_blocks.selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": train_val_metrics["trainval_accuracy"],
            "balanced_accuracy": train_val_metrics["trainval_balanced_accuracy"],
            "log_loss": train_val_metrics["trainval_log_loss"],
            "roc_auc": train_val_metrics["trainval_roc_auc"],
            "average_precision": train_val_metrics["trainval_average_precision"],
        }
    )

    summary_rows.append(
        {
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "test_final_model",
            "model_training_data": "train+validation",
            "best_lr": best_lr,
            "best_clip_norm_private": best_clip_norm_private,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": final_achieved_epsilon,
            "delta": final_delta,
            "noise_multiplier": final_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm_public": np.nan if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
            "clip_norm_private": best_clip_norm_private,
            "iterations": ITERATIONS,
            "n_public_fc_features": len(feature_blocks.fc_cols),
            "n_private_phenotype_raw_columns": len(feature_blocks.selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": final_test_metrics["test_accuracy"],
            "balanced_accuracy": final_test_metrics["test_balanced_accuracy"],
            "log_loss": final_test_metrics["test_log_loss"],
            "roc_auc": final_test_metrics["test_roc_auc"],
            "average_precision": final_test_metrics["test_average_precision"],
        }
    )

    metrics_summary_df = pd.DataFrame(summary_rows)
    metrics_summary_df.insert(0, "seed", int(SEED))
    metrics_summary_df.to_csv(
        experiment_dir / "feature_dp_final_metrics_summary.csv",
        index=False,
    )

    run_feature_summary = {
        "experiment_key": experiment_key,
        "experiment_name": experiment_name,
        "mechanism": "Feature DP",
        "public_features": "fMRI FC",
        "private_features": "selected phenotype features",
        "selected_feature_group_keys": selected_group_keys,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "selected_private_phenotype_columns_present_in_dataframe": feature_blocks.selected_phenotype_cols,
        "phenotype_domain_scaled_numeric_columns": feature_blocks.phenotype_domain_scaled_cols,
        "phenotype_other_numeric_columns": feature_blocks.phenotype_other_numeric_cols,
        "phenotype_categorical_columns": feature_blocks.phenotype_categorical_cols,
        "fc_feature_count": len(feature_blocks.fc_cols),
        "requested_json_columns_absent_from_dataframe": feature_blocks.missing_cols,
        "n_total_raw_columns_before_encoding": len(feature_blocks.selected_phenotype_cols) + len(feature_blocks.fc_cols),
        "subgroup_eval_features_requested": SUBGROUP_EVAL_FEATURES,
        "subgroup_eval_features_present": subgroup_eval_features_present,
        "sensible_numeric_bins": SENSIBLE_NUMERIC_BINS,
        "split_strategy": global_data.split_name,
        "site_aware_splitting_used": False,
        "best_lr_selected_on_validation": best_lr,
        "best_clip_norm_private_selected_on_validation": best_clip_norm_private,
        "use_fixed_private_clip_norm_grid": USE_FIXED_PRIVATE_CLIP_NORM_GRID,
        "private_clip_norm_quantiles_for_grid": PRIVATE_CLIP_NORM_QUANTILES_FOR_GRID,
        "private_clip_norm_grid": private_clip_norm_grid,
        "TUNED_C": tuned_C,
        "l2_reg_handling": L2_REG_HANDLING,
        "clip_norm_public": None if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
        "clip_norm_private": best_clip_norm_private,
        "missing_numeric_policy": "replace NaN/inf with 0 after preprocessing",
        "missing_categorical_policy": "missing/unseen categories mapped to all-zero one-hot block",
    }

    with open(experiment_dir / "run_feature_summary.json", "w") as f:
        json.dump(run_feature_summary, f, indent=2)

    print("\nSaved experiment outputs in:")
    print(experiment_dir)

    return metrics_summary_df, test_subgroup_metrics


# =========================================================
# Multi-seed orchestration and aggregation
# =========================================================
def aggregate_numeric_columns_mean_std_sem(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate numeric metric columns as mean/std/sem/n within groups."""
    if df is None or df.empty:
        return pd.DataFrame()

    group_cols = [c for c in group_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]

    if not metric_cols:
        return pd.DataFrame()

    rows = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False, sort=False)
    else:
        grouped = [((), df)]

    for group_key, group_df in grouped:
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {col: val for col, val in zip(group_cols, group_key)}
        else:
            row = {}

        for metric in metric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            n = int(len(values))

            row[f"{metric}_n"] = n
            if n == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_sem"] = np.nan
            else:
                mean_value = float(values.mean())
                std_value = float(values.std(ddof=1)) if n > 1 else np.nan
                sem_value = float(std_value / np.sqrt(n)) if n > 1 else np.nan

                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_std"] = std_value
                row[f"{metric}_sem"] = sem_value

        rows.append(row)

    return pd.DataFrame(rows)


def write_seed_level_outputs(
    *,
    seed: int,
    output_dir: Path,
    global_data: PreparedGlobalData,
    metrics_frames: list[pd.DataFrame],
    subgroup_frames: list[pd.DataFrame],
    run_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write the same seed-level combined CSVs as the original single-seed main."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_df = pd.concat(metrics_frames, axis=0, ignore_index=True)
    all_metrics_df.to_csv(
        output_dir / "all_modes_feature_dp_final_metrics_summary.csv",
        index=False,
    )

    final_validation_test_metrics = all_metrics_df.loc[
        all_metrics_df["split"].isin(["validation_selected_model", "test_final_model"])
    ].reset_index(drop=True)

    final_validation_test_metrics.to_csv(
        output_dir / "all_modes_feature_dp_final_validation_test_metrics.csv",
        index=False,
    )

    test_columns = [
        "seed",
        "experiment_key",
        "experiment_name",
        "best_lr",
        "best_clip_norm_private",
        "target_epsilon",
        "achieved_epsilon",
        "delta",
        "noise_multiplier",
        "TUNED_C",
        "clip_norm_public",
        "clip_norm_private",
        "iterations",
        "n_public_fc_features",
        "n_private_phenotype_raw_columns",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
    ]
    test_columns = [c for c in test_columns if c in all_metrics_df.columns]

    test_comparison = all_metrics_df.loc[
        all_metrics_df["split"] == "test_final_model",
        test_columns,
    ].reset_index(drop=True)

    test_comparison.to_csv(
        output_dir / "all_modes_feature_dp_test_metrics_comparison.csv",
        index=False,
    )

    if subgroup_frames:
        all_subgroup_metrics_df = pd.concat(
            subgroup_frames,
            axis=0,
            ignore_index=True,
        )
    else:
        all_subgroup_metrics_df = pd.DataFrame()

    all_subgroup_metrics_df.to_csv(
        output_dir / "all_modes_feature_dp_test_subgroup_metrics_summary.csv",
        index=False,
    )

    run_summary = {
        "seed": int(seed),
        "run_role": run_role,
        "target_col": TARGET_COL,
        "csv_path": str(CSV_PATH),
        "feature_json_path_used": str(global_data.feature_json_path),
        "output_dir": str(output_dir),
        "epsilon_output_dir": str(EPSILON_OUTPUT_DIR),
        "experiments": EXPERIMENTS,
        "mechanism": "Feature DP",
        "public_features": "fMRI FC",
        "private_features": "selected phenotype features",
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "split_strategy": global_data.split_name,
        "site_aware_splitting_used": False,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "target_epsilon": TARGET_EPSILON,
        "delta_power": DELTA_POWER,
        "clip_norm_public": None if CLIP_NORM_PUBLIC is None else float(CLIP_NORM_PUBLIC),
        "clip_norm_private_default_fallback": CLIP_NORM_PRIVATE,
        "feature_norm_quantile_csv_path": str(FEATURE_NORM_QUANTILE_CSV_PATH),
        "private_clip_norm_quantiles_for_grid": PRIVATE_CLIP_NORM_QUANTILES_FOR_GRID,
        "private_clip_norm_grid_fallback": PRIVATE_CLIP_NORM_GRID_FALLBACK,
        "iterations": ITERATIONS,
        "full_batch": True,
        "TUNED_C_WithOut_AnatFunc": TUNED_C_WithOut_AnatFunc,
        "TUNED_FIT_INTERCEPT": TUNED_FIT_INTERCEPT,
        "L2_REG_HANDLING": L2_REG_HANDLING,
        "initial_learning_rates": LEARNING_RATES_INITIAL,
        "max_endpoint_expansions": MAX_ENDPOINT_EXPANSIONS,
        "endpoint_expansion_factor": ENDPOINT_EXPANSION_FACTOR,
        "use_fixed_private_clip_norm_grid": USE_FIXED_PRIVATE_CLIP_NORM_GRID,
        "private_clip_norm_grid": PRIVATE_CLIP_NORM_GRID,
        "use_numba_jit": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
        "parallelize_hyperparameter_configs": PARALLELIZE_HYPERPARAMETER_CONFIGS,
        "joblib_available": JOBLIB_AVAILABLE,
        "n_parallel_jobs": N_PARALLEL_JOBS,
        "parallel_backend": PARALLEL_BACKEND,
        "selection_metric": SELECTION_METRIC,
        "sensible_numeric_bins": SENSIBLE_NUMERIC_BINS,
        "n_total_after_valid_mask": int(len(global_data.y)),
        "n_train": int(len(global_data.train_idx)),
        "n_validation": int(len(global_data.val_idx)),
        "n_test": int(len(global_data.test_idx)),
        "class_counts_total": {
            str(k): int(v)
            for k, v in pd.Series(global_data.y).value_counts().sort_index().to_dict().items()
        },
        "final_model_training_policy": (
            "Learning rate and private clip norm are tuned exactly once using "
            "LRC_SEED. The selected hyperparameters are reused for each seed in "
            "SEED_LIST. For each seed, the final model is refit on train+validation "
            "and evaluated on the held-out test set."
        ),
    }

    with open(output_dir / "run_summary_all_modes_feature_dp.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\n" + "=" * 120)
    print(f"Final Feature-DP test metrics comparison for seed {int(seed)}")
    print("=" * 120)
    print(test_comparison.to_string(index=False))

    print("\nSaved seed-level combined outputs in:")
    print(output_dir)

    return all_metrics_df, all_subgroup_metrics_df, test_comparison


def extract_fixed_hyperparameters_from_tuning(
    tuning_metrics_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Extract one selected (best_lr, best_clip_norm_private) pair per experiment."""
    test_rows = tuning_metrics_df.loc[
        tuning_metrics_df["split"] == "test_final_model"
    ].copy()

    if test_rows.empty:
        raise ValueError("Could not extract hyperparameters: no test_final_model rows found.")

    fixed = {}
    for _, row in test_rows.iterrows():
        fixed[str(row["experiment_key"])] = {
            "best_lr": float(row["best_lr"]),
            "best_clip_norm_private": float(row["best_clip_norm_private"]),
        }

    return fixed


def run_all_experiments_for_seed(
    *,
    seed: int,
    run_role: str,
    fixed_hyperparameters_by_experiment: Optional[dict[str, dict[str, float]]] = None,
    fixed_global_data: Optional[PreparedGlobalData] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreparedGlobalData]:
    """Run all active experiments for one run seed using a fixed data split.

    The run seed controls only the Feature-DP stochastic training path, i.e.,
    the DP-noise streams created through make_hyperparameter_config_seed().
    The train/validation/test samples are fixed by fixed_global_data.

    When fixed_global_data is None, this function prepares the data and creates
    or loads the split. In main(), that happens only for the LRC_SEED tuning run.
    Every subsequent seed receives the same PreparedGlobalData object, so the
    split is exactly identical across all seeds.

    When fixed_hyperparameters_by_experiment is provided, no broad LR/private
    clip search is performed. Instead, the global grids are temporarily reduced
    to the selected (LR, private clip norm) pair for the experiment, so the
    original run_single_experiment logic is reused without changing the core
    Feature-DP training implementation or output filenames.
    """
    global SEED, OUTPUT_DIR, LEARNING_RATES_INITIAL, PRIVATE_CLIP_NORM_GRID, USE_FIXED_PRIVATE_CLIP_NORM_GRID

    seed = int(seed)
    SEED = seed
    OUTPUT_DIR = get_seed_output_dir(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"Running Feature-DP pipeline for seed={seed} | role={run_role}")
    print("=" * 120)

    if fixed_global_data is None:
        # This should be the single LRC_SEED call in main().
        global_data = prepare_global_data(seed=int(LRC_SEED), output_dir=OUTPUT_DIR)
    else:
        global_data = fixed_global_data
        save_prepared_global_arrays(global_data=global_data, output_dir=OUTPUT_DIR)
        print("\nReusing fixed train/validation/test split created by LRC_SEED.")
        print(global_data.split_name)
        print(f"Train size      : {len(global_data.train_idx)}")
        print(f"Validation size : {len(global_data.val_idx)}")
        print(f"Test size       : {len(global_data.test_idx)}")

    metrics_frames = []
    subgroup_frames = []

    original_learning_rates = list(LEARNING_RATES_INITIAL)
    original_private_clip_grid = list(PRIVATE_CLIP_NORM_GRID)
    original_use_fixed_private_clip_grid = bool(USE_FIXED_PRIVATE_CLIP_NORM_GRID)

    for exp in EXPERIMENTS:
        exp_key = exp["experiment_key"]

        using_fixed_hyperparameters = fixed_hyperparameters_by_experiment is not None
        if using_fixed_hyperparameters:
            if exp_key not in fixed_hyperparameters_by_experiment:
                raise KeyError(
                    f"No fixed hyperparameters available for experiment {exp_key!r}."
                )

            selected = fixed_hyperparameters_by_experiment[exp_key]
            LEARNING_RATES_INITIAL = [float(selected["best_lr"])]
            PRIVATE_CLIP_NORM_GRID = [float(selected["best_clip_norm_private"])]
            USE_FIXED_PRIVATE_CLIP_NORM_GRID = True

            print(
                f"\nReusing LRC-selected hyperparameters for {exp_key}: "
                f"best_lr={LEARNING_RATES_INITIAL[0]}, "
                f"best_clip_norm_private={PRIVATE_CLIP_NORM_GRID[0]}"
            )
        else:
            LEARNING_RATES_INITIAL = list(original_learning_rates)
            PRIVATE_CLIP_NORM_GRID = list(original_private_clip_grid)
            USE_FIXED_PRIVATE_CLIP_NORM_GRID = bool(original_use_fixed_private_clip_grid)

        try:
            metrics_df, subgroup_df = run_single_experiment(
                global_data=global_data,
                experiment_key=exp_key,
                experiment_name=exp["experiment_name"],
                selected_group_keys=exp["feature_group_keys"],
                tuned_C=float(exp["tuned_C"]),
            )
        finally:
            LEARNING_RATES_INITIAL = list(original_learning_rates)
            PRIVATE_CLIP_NORM_GRID = list(original_private_clip_grid)
            USE_FIXED_PRIVATE_CLIP_NORM_GRID = bool(original_use_fixed_private_clip_grid)

        metrics_frames.append(metrics_df)

        if subgroup_df is not None and not subgroup_df.empty:
            subgroup_frames.append(subgroup_df)

    all_metrics_df, all_subgroup_metrics_df, test_comparison = write_seed_level_outputs(
        seed=seed,
        output_dir=OUTPUT_DIR,
        global_data=global_data,
        metrics_frames=metrics_frames,
        subgroup_frames=subgroup_frames,
        run_role=run_role,
    )

    return all_metrics_df, all_subgroup_metrics_df, test_comparison, global_data

# =========================================================
# Main
# =========================================================
def main() -> None:
    print(f"Numba requested: {USE_NUMBA_JIT}; numba available: {NUMBA_AVAILABLE}")
    print(
        f"Parallel hyperparameter trials requested: "
        f"{PARALLELIZE_HYPERPARAMETER_CONFIGS}; joblib available: {JOBLIB_AVAILABLE}"
    )
    print(f"SEED_LIST: {SEED_LIST}")
    print(f"LRC_SEED for one-time LR/private-clip tuning: {LRC_SEED}")
    print(f"Fixed split seed/source: {'preprocess dump CSVs' if USE_PREPROCESS_DUMP else f'LRC_SEED={int(LRC_SEED)}'}")
    print(f"USE_PREPROCESS_DUMP: {USE_PREPROCESS_DUMP}")
    if USE_PREPROCESS_DUMP:
        print(f"Preprocess dump directory: {PREPROCESS_DUMP_DIR}")
    print(f"Epsilon-level output directory: {EPSILON_OUTPUT_DIR}")

    EPSILON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. Tune LR + private clip norm exactly once on LRC_SEED.
    # -----------------------------------------------------
    lrc_metrics_df, lrc_subgroup_df, _, lrc_global_data = run_all_experiments_for_seed(
        seed=LRC_SEED,
        run_role="lrc_hyperparameter_tuning_seed",
        fixed_hyperparameters_by_experiment=None,
        fixed_global_data=None,
    )

    fixed_hyperparameters_by_experiment = extract_fixed_hyperparameters_from_tuning(
        lrc_metrics_df
    )

    lrc_hyperparameter_summary = {
        "lrc_seed": int(LRC_SEED),
        "seed_list": [int(s) for s in SEED_LIST],
        "fixed_hyperparameters_by_experiment": fixed_hyperparameters_by_experiment,
        "fixed_split_strategy": lrc_global_data.split_name,
        "use_preprocess_dump": bool(USE_PREPROCESS_DUMP),
        "preprocess_dump_dir": str(PREPROCESS_DUMP_DIR) if USE_PREPROCESS_DUMP else None,
        "n_train": int(len(lrc_global_data.train_idx)),
        "n_validation": int(len(lrc_global_data.val_idx)),
        "n_test": int(len(lrc_global_data.test_idx)),
        "selection_metric": SELECTION_METRIC,
        "selection_tie_rule": (
            "Configurations within SELECTION_TIE_TOL of best validation score are "
            "sorted by clip_norm_private ascending, then learning_rate ascending."
        ),
        "policy": (
            "The selected best_lr and best_clip_norm_private from LRC_SEED are reused "
            "for every seed in SEED_LIST."
        ),
    }

    with open(EPSILON_OUTPUT_DIR / "lrc_fixed_hyperparameter_summary.json", "w") as f:
        json.dump(lrc_hyperparameter_summary, f, indent=2)

    print("\n" + "=" * 120)
    print("Fixed hyperparameters selected from LRC_SEED")
    print("=" * 120)
    print(json.dumps(lrc_hyperparameter_summary, indent=2))

    # -----------------------------------------------------
    # 2. Run all requested seeds using the fixed LR/private clip pair.
    # -----------------------------------------------------
    all_seed_metrics = []
    all_seed_subgroup_metrics = []
    all_seed_test_comparisons = []
    per_seed_summaries = []

    if int(LRC_SEED) in {int(s) for s in SEED_LIST}:
        # The LRC run already used the selected hyperparameters and produced
        # complete seed-level outputs, so we include it in the multi-seed mean.
        all_seed_metrics.append(lrc_metrics_df)
        if lrc_subgroup_df is not None and not lrc_subgroup_df.empty:
            all_seed_subgroup_metrics.append(lrc_subgroup_df)

        lrc_test_comparison = lrc_metrics_df.loc[
            lrc_metrics_df["split"] == "test_final_model"
        ].reset_index(drop=True)
        all_seed_test_comparisons.append(lrc_test_comparison)

        per_seed_summaries.append(
            {
                "seed": int(LRC_SEED),
                "role": "included_lrc_hyperparameter_tuning_seed",
                "output_dir": str(get_seed_output_dir(LRC_SEED)),
            }
        )

    for seed in SEED_LIST:
        seed = int(seed)
        if seed == int(LRC_SEED):
            continue

        seed_metrics_df, seed_subgroup_df, seed_test_comparison, _ = run_all_experiments_for_seed(
            seed=seed,
            run_role="fixed_lrc_hyperparameter_evaluation_seed",
            fixed_hyperparameters_by_experiment=fixed_hyperparameters_by_experiment,
            fixed_global_data=lrc_global_data,
        )

        all_seed_metrics.append(seed_metrics_df)
        all_seed_test_comparisons.append(seed_test_comparison)

        if seed_subgroup_df is not None and not seed_subgroup_df.empty:
            all_seed_subgroup_metrics.append(seed_subgroup_df)

        per_seed_summaries.append(
            {
                "seed": seed,
                "role": "fixed_lrc_hyperparameter_evaluation_seed",
                "output_dir": str(get_seed_output_dir(seed)),
            }
        )

    if not all_seed_metrics:
        raise RuntimeError("No SEED_LIST results were collected.")

    all_seeds_metrics_df = pd.concat(all_seed_metrics, axis=0, ignore_index=True)
    all_seeds_metrics_df.to_csv(
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_final_metrics_summary.csv",
        index=False,
    )

    all_seeds_test_comparison = all_seeds_metrics_df.loc[
        all_seeds_metrics_df["split"] == "test_final_model"
    ].reset_index(drop=True)
    all_seeds_test_comparison.to_csv(
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_comparison.csv",
        index=False,
    )

    test_metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
    ]
    test_group_cols = [
        "experiment_key",
        "experiment_name",
        "split",
        "target_epsilon",
        "best_lr",
        "best_clip_norm_private",
        "TUNED_C",
        "clip_norm_public",
        "clip_norm_private",
        "iterations",
        "n_public_fc_features",
        "n_private_phenotype_raw_columns",
    ]
    test_metrics_mean_std_sem = aggregate_numeric_columns_mean_std_sem(
        all_seeds_test_comparison,
        group_cols=test_group_cols,
        metric_cols=test_metric_cols,
    )
    test_metrics_mean_std_sem.to_csv(
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_mean_std_sem.csv",
        index=False,
    )

    if all_seed_subgroup_metrics:
        all_seeds_subgroup_df = pd.concat(
            all_seed_subgroup_metrics,
            axis=0,
            ignore_index=True,
        )
    else:
        all_seeds_subgroup_df = pd.DataFrame()

    all_seeds_subgroup_df.to_csv(
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_summary.csv",
        index=False,
    )

    subgroup_metric_cols = [
        "n_samples",
        "n_control",
        "n_autism",
        "autism_prevalence",
        "predicted_autism_rate",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
        "n_unique_classes",
        "control_precision",
        "control_recall",
        "control_f1-score",
        "autism_precision",
        "autism_recall",
        "autism_f1-score",
        "cm_true_control_pred_control",
        "cm_true_control_pred_autism",
        "cm_true_autism_pred_control",
        "cm_true_autism_pred_autism",
    ]
    subgroup_group_cols = [
        "experiment_key",
        "experiment_name",
        "split",
        "target_epsilon",
        "best_lr",
        "best_clip_norm_private",
        "feature",
        "group",
    ]
    subgroup_metrics_mean_std_sem = aggregate_numeric_columns_mean_std_sem(
        all_seeds_subgroup_df,
        group_cols=subgroup_group_cols,
        metric_cols=subgroup_metric_cols,
    )
    subgroup_metrics_mean_std_sem.to_csv(
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_mean_std_sem.csv",
        index=False,
    )

    multiseed_run_summary = {
        "seed_list": [int(s) for s in SEED_LIST],
        "lrc_seed": int(LRC_SEED),
        "target_epsilon": TARGET_EPSILON,
        "epsilon_output_dir": str(EPSILON_OUTPUT_DIR),
        "fixed_hyperparameters_by_experiment": fixed_hyperparameters_by_experiment,
        "fixed_split_strategy": lrc_global_data.split_name,
        "use_preprocess_dump": bool(USE_PREPROCESS_DUMP),
        "preprocess_dump_dir": str(PREPROCESS_DUMP_DIR) if USE_PREPROCESS_DUMP else None,
        "n_train": int(len(lrc_global_data.train_idx)),
        "n_validation": int(len(lrc_global_data.val_idx)),
        "n_test": int(len(lrc_global_data.test_idx)),
        "lrc_hyperparameter_summary_path": str(
            EPSILON_OUTPUT_DIR / "lrc_fixed_hyperparameter_summary.json"
        ),
        "experiments": EXPERIMENTS,
        "mechanism": "Feature DP",
        "public_features": "fMRI FC",
        "private_features": "selected phenotype features",
        "selection_metric": SELECTION_METRIC,
        "selection_tie_tol": SELECTION_TIE_TOL,
        "initial_learning_rates_for_lrc_tuning": LEARNING_RATES_INITIAL,
        "private_clip_norm_grid": PRIVATE_CLIP_NORM_GRID,
        "use_numba_jit": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
        "parallelize_hyperparameter_configs": PARALLELIZE_HYPERPARAMETER_CONFIGS,
        "joblib_available": JOBLIB_AVAILABLE,
        "n_parallel_jobs": N_PARALLEL_JOBS,
        "parallel_backend": PARALLEL_BACKEND,
        "per_seed_summaries": per_seed_summaries,
        "outputs": {
            "all_seed_metrics": str(
                EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_final_metrics_summary.csv"
            ),
            "all_seed_test_metrics": str(
                EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_comparison.csv"
            ),
            "test_metrics_mean_std_sem": str(
                EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_mean_std_sem.csv"
            ),
            "all_seed_subgroup_metrics": str(
                EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_summary.csv"
            ),
            "subgroup_metrics_mean_std_sem": str(
                EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_mean_std_sem.csv"
            ),
        },
    }

    with open(EPSILON_OUTPUT_DIR / "run_summary_multiseed_feature_dp.json", "w") as f:
        json.dump(multiseed_run_summary, f, indent=2)

    print("\n" + "=" * 120)
    print("Multi-seed Feature-DP test metrics mean/std/sem")
    print("=" * 120)
    print(test_metrics_mean_std_sem.to_string(index=False))

    print("\nSaved epsilon-level outputs in:")
    print(EPSILON_OUTPUT_DIR)

    print("\nMost important epsilon-level output files:")
    for fp in [
        EPSILON_OUTPUT_DIR / "lrc_fixed_hyperparameter_summary.json",
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_final_metrics_summary.csv",
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_comparison.csv",
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_metrics_mean_std_sem.csv",
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_summary.csv",
        EPSILON_OUTPUT_DIR / "all_seeds_feature_dp_test_subgroup_metrics_mean_std_sem.csv",
        EPSILON_OUTPUT_DIR / "run_summary_multiseed_feature_dp.json",
    ]:
        print(fp)


if __name__ == "__main__":
    main()