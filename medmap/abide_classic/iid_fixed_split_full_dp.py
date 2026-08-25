from __future__ import annotations
# ---------------------------------------------------------------------------
# Dataset creation (shared loader; see medmap/abide_classic/data.py).
# Configure via environment variables before running (docs/abide.md):
#   ABIDE_DATA_DIR (required), ABIDE_FC_DIR, ABIDE_OUTPUT_ROOT,
#   PHENOTYPIC_CSV, FEATURE_JSON, ABIDE_FETCH_RAW
# ---------------------------------------------------------------------------
from medmap.abide_classic import data as _abide_data

_ABIDE_DATA_ROOT, _ABIDE_OUTPUT_ROOT = _abide_data.resolve_roots()
_D = str(_ABIDE_DATA_ROOT)
_O = str(_ABIDE_OUTPUT_ROOT)
X_conn, y, meta, abide = _abide_data.load_abide_inputs(_ABIDE_DATA_ROOT)

# Metrics for Full DP Multiseeded
#!/usr/bin/env python3
"""
Manual full-batch DP-SGD / DP-GD logistic regression on ABIDE fMRI + phenotypic features.

Updated version:
    1. Keeps data loading, cleaning, feature selection, domain scaling,
       preprocessing, DP accounting, DP-GD logic, Numba-JIT kernel, optional
       joblib learning-rate parallelism, and train/validation/test learning
       policy unchanged from the attached IID full-DP reference code.
    2. Creates the IID train/validation/test split exactly once using LR_SEED
       and reuses the exact same samples for every seed in SEED_LIST.
    3. LR_SEED does not need to be present in SEED_LIST. The LR tuning role
       remains unchanged: tune LR once on LR_SEED, then reuse the selected LR
       for all reported seeds.
    4. Adds USE_PREPROCESS_DUMP. When enabled, the script reads the fixed
       prepared CSVs dumped by the IID fixed-split non-private pipeline instead
       of rebuilding the raw ABIDE + X_conn preparation stage.
"""

import json
import math
import os
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

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrapper(fn):
            return fn
        return wrapper

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


# =========================================================
# Dataset / seed / execution configuration
# =========================================================
class SeedConfig:
    """
    Seed-level configuration.

    LR_SEED is used exactly once for learning-rate tuning. The selected
    best LR is then reused for every seed in SEED_LIST.

    LR_SEED does not need to be present in SEED_LIST. If it is present,
    the LR-tuning run is also used as that seed's normal evaluation run
    for LR_TUNING_EXPERIMENT_KEY.
    """

    SEED_LIST = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    LR_SEED = 81
    LR_TUNING_EXPERIMENT_KEY = "full_dp_without_anat_func"


class ExperimentConfig:
    """Experiment-level configuration shared by all seeds."""

    ITERATIONS = 2000

    # Logistic-regression sklearn-style regularization parameter set by user.
    # Objective:
    #     mean_i BCE_i + (lambda / 2) ||w||_2^2
    # where:
    #     lambda = 1 / (tuned_C * n_train)
    TUNED_C_WithOut_AnatFunc = 0.1
    # TUNED_C_With_AnatFunc = 0.1
    TUNED_FIT_INTERCEPT = True
    TARGET_EPSILON = 64.0

    # Learning-rate grid used only for LR_SEED tuning.
    LEARNING_RATES_INITIAL = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    MAX_ENDPOINT_EXPANSIONS = 3
    ENDPOINT_EXPANSION_FACTOR = 2.0

    TARGET_COL = "DX_GROUP"

    CSV_PATH = _abide_data.phenotypic_csv_path(_ABIDE_DATA_ROOT)

    FEATURE_JSON_CANDIDATES = _abide_data.feature_json_candidates(_ABIDE_DATA_ROOT)

    # If X_conn already exists in the Python session, it will be used.
    # Otherwise set this to a .npy or .npz file containing X_conn.
    X_CONN_PATH: Optional[Path] = None

    USE_SITE_ID_AS_FEATURE = False
    FC_IS_CORRELATION = True

    # Overall split sizes: train = 80%, validation = 10%, test = 10%
    VALIDATION_SIZE = 0.10
    TEST_SIZE = 0.20

    # User-selectable preprocessing for phenotype_other_numeric_cols.
    # Allowed: "standardize_zero_fill", "minmax_minus1_1_zero_fill", "zero_fill_only"
    PHENOTYPE_OTHER_NUMERIC_MODE = "minmax_minus1_1_zero_fill"


class DPConfig:
    """DP-GD / DP-SGD configuration shared by all seeds."""

    DELTA_POWER = -1.1  # delta = n ** DELTA_POWER

    # If None, noise multiplier is solved using RDPAccountant.
    NOISE_MULTIPLIER: Optional[float] = None

    CLIP_NORM = 1.0
    FULL_BATCH = True

    # L2 regularization handling:
    #   "add_after_noise": privatize clipped data gradients, then add deterministic lambda*w.
    #   "inside_per_example": add lambda*w to each per-example gradient before clipping.
    L2_REG_HANDLING = "add_after_noise"  # {"add_after_noise", "inside_per_example"}

    SELECTION_METRIC = "accuracy"  # {"accuracy", "balanced_accuracy", "roc_auc", "log_loss"}
    SELECTION_TIE_TOL = 1e-12

    RECORD_EVERY_N_STEPS = 1
    SHOW_PLOTS = False


class ExecutionConfig:
    """Speed / execution configuration shared by all seeds."""

    USE_NUMBA_JIT = True
    PARALLELIZE_LEARNING_RATES = True
    LR_PARALLEL_N_JOBS = min(64, os.cpu_count() or 1)
    LR_PARALLEL_BACKEND = "loky"  # "loky" is safer; "threading" avoids copies but can oversubscribe CPU.


# =========================================================
# Optional prepared-data shortcut
# =========================================================
# When True, the script reads the fixed split CSVs dumped by the IID
# fixed-split non-private pipeline instead of rebuilding the raw ABIDE +
# X_conn preparation stage. The DP-GD preprocessing inside each experiment
# remains unchanged: the preprocessor is fit on train only and then applied
# to validation/test, exactly as in the reference code.
USE_PREPROCESS_DUMP = False
PRINT_PREPROCESS_DUMP_COLUMN_NAMES = False

PREPROCESS_DUMP_DIR = Path(f"{_O}/iid_fixed_split/non_priv")
PREPROCESS_DUMP_TRAIN_CSV = PREPROCESS_DUMP_DIR / "train_fixed.csv"
PREPROCESS_DUMP_VAL_CSV = PREPROCESS_DUMP_DIR / "val_fixed.csv"
PREPROCESS_DUMP_TEST_CSV = PREPROCESS_DUMP_DIR / "test_fixed.csv"
PREPROCESS_DUMP_TRAIN_VAL_CSV = PREPROCESS_DUMP_DIR / "train_val_fixed.csv"


# Preserve the original global names used throughout the code.
SEED_LIST = [int(s) for s in SeedConfig.SEED_LIST]
LR_SEED = int(SeedConfig.LR_SEED)

# Optional environment overrides (the defaults above reproduce the paper runs).
SEED_LIST = _abide_data.env_override_int_list("ABIDE_SEEDS", SEED_LIST)
LR_SEED = _abide_data.env_override_int("ABIDE_TUNING_SEED", LR_SEED)
LR_TUNING_EXPERIMENT_KEY = str(SeedConfig.LR_TUNING_EXPERIMENT_KEY)

ITERATIONS = ExperimentConfig.ITERATIONS
TUNED_C_WithOut_AnatFunc = ExperimentConfig.TUNED_C_WithOut_AnatFunc
# TUNED_C_With_AnatFunc = ExperimentConfig.TUNED_C_With_AnatFunc
TUNED_FIT_INTERCEPT = ExperimentConfig.TUNED_FIT_INTERCEPT
TARGET_EPSILON = ExperimentConfig.TARGET_EPSILON
LEARNING_RATES_INITIAL = ExperimentConfig.LEARNING_RATES_INITIAL

# Optional environment overrides (the defaults above reproduce the paper runs).
TARGET_EPSILON = _abide_data.env_override_float("ABIDE_TARGET_EPSILON", TARGET_EPSILON)
ITERATIONS = _abide_data.env_override_int("ABIDE_ITERATIONS", ITERATIONS)
TUNED_C_WithOut_AnatFunc = _abide_data.env_override_float("ABIDE_TUNED_C", TUNED_C_WithOut_AnatFunc)
LEARNING_RATES_INITIAL = _abide_data.env_override_float_list("ABIDE_LR_GRID", LEARNING_RATES_INITIAL)
MAX_ENDPOINT_EXPANSIONS = ExperimentConfig.MAX_ENDPOINT_EXPANSIONS
ENDPOINT_EXPANSION_FACTOR = ExperimentConfig.ENDPOINT_EXPANSION_FACTOR
TARGET_COL = ExperimentConfig.TARGET_COL
CSV_PATH = ExperimentConfig.CSV_PATH
FEATURE_JSON_CANDIDATES = ExperimentConfig.FEATURE_JSON_CANDIDATES
X_CONN_PATH = ExperimentConfig.X_CONN_PATH
USE_SITE_ID_AS_FEATURE = ExperimentConfig.USE_SITE_ID_AS_FEATURE
FC_IS_CORRELATION = ExperimentConfig.FC_IS_CORRELATION
VALIDATION_SIZE = ExperimentConfig.VALIDATION_SIZE
TEST_SIZE = ExperimentConfig.TEST_SIZE
PHENOTYPE_OTHER_NUMERIC_MODE = ExperimentConfig.PHENOTYPE_OTHER_NUMERIC_MODE

DELTA_POWER = DPConfig.DELTA_POWER
NOISE_MULTIPLIER = DPConfig.NOISE_MULTIPLIER
CLIP_NORM = DPConfig.CLIP_NORM
FULL_BATCH = DPConfig.FULL_BATCH
L2_REG_HANDLING = DPConfig.L2_REG_HANDLING
SELECTION_METRIC = DPConfig.SELECTION_METRIC
SELECTION_TIE_TOL = DPConfig.SELECTION_TIE_TOL
RECORD_EVERY_N_STEPS = DPConfig.RECORD_EVERY_N_STEPS
SHOW_PLOTS = DPConfig.SHOW_PLOTS

USE_NUMBA_JIT = ExecutionConfig.USE_NUMBA_JIT
PARALLELIZE_LEARNING_RATES = ExecutionConfig.PARALLELIZE_LEARNING_RATES
LR_PARALLEL_N_JOBS = ExecutionConfig.LR_PARALLEL_N_JOBS
LR_PARALLEL_BACKEND = ExecutionConfig.LR_PARALLEL_BACKEND

OUTPUT_DIR = Path(f"{_O}/iid_fixed_split/full_dp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON_OUTPUT_DIR = Path(
    f"{_O}/iid_fixed_split/full_dp/e{int(TARGET_EPSILON)}"
)
EPSILON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_seed_output_dir(seed: int) -> Path:
    out_dir = EPSILON_OUTPUT_DIR / f"seed{int(seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# =========================================================
# Two full-DP feature-combination experiments
# =========================================================
EXPERIMENTS = [
    {
        "experiment_key": "full_dp_without_anat_func",
        "experiment_name": "full DP without anat & func",
        "tuned_C": TUNED_C_WithOut_AnatFunc,
        "feature_group_keys": [
            "recommended_initial_numeric_or_binary",
            "recommended_initial_categorical",
            "optional_categorical_with_caution",
        ],
    },
    # {
    #     "experiment_key": "full_dp_with_anat_func",
    #     "experiment_name": "full DP with anat & func",
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
@dataclass(frozen=True)
class Params:
    w: np.ndarray
    b: float


@dataclass
class PreparedGlobalData:
    seed: int
    output_dir: Path
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
    split_audit: dict
    preprocess_dump_used: bool = False
    preprocess_dump_dir: Optional[Path] = None


# =========================================================
# Custom transformers: same preprocessing logic as reference
# =========================================================
class ZeroFillTransformer(BaseEstimator, TransformerMixin):
    """Replace NaN / +/-inf with 0.0."""

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
    """Standardize numeric features on observed train values, then zero-fill."""

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        self.scale_ = np.where(np.isfinite(self.scale_) & (self.scale_ > 0), self.scale_, 1.0)
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
    """Min-max scale to [-1, 1] using train observed values, then zero-fill."""

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
                2.0 * (X_arr[:, valid_cols] - self.data_min_[valid_cols]) / self.range_[valid_cols] - 1.0
            )
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class MissingAsAllZeroOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode observed categories only; missing/unseen map to all-zero."""

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = [str(c) for c in X_df.columns]
        self.categories_ = []
        for j in range(X_df.shape[1]):
            s = X_df.iloc[:, j]
            observed = s[~pd.isna(s)]
            cats = pd.Index(observed.astype(str)).unique().tolist()
            self.categories_.append(sorted(cats))
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
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
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
            "X_conn is not available in the current Python session. Please set X_CONN_PATH to a .npy or .npz file."
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
    raise FileNotFoundError("Could not find recommended_feature_lists.json in any expected location.")


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
# Combine FC + phenotype and preprocessing helpers
# =========================================================
def build_combined_dataframe(
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    phenotype_selected_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    n_samples, n_fc_features = X_conn_model.shape
    fc_colnames = [f"fc_{i:05d}" for i in range(n_fc_features)]
    fc_df = pd.DataFrame(X_conn_model, columns=fc_colnames, index=df_model.index)
    if phenotype_selected_cols:
        phen_df = df_model[phenotype_selected_cols].copy()
        X_combined = pd.concat([fc_df, phen_df], axis=1)
    else:
        X_combined = fc_df.copy()
    print(
        f"Combined dataframe shape: {X_combined.shape} "
        f"(FC: {n_fc_features} cols, phenotype: {len(phenotype_selected_cols)} cols)"
    )
    return X_combined, fc_colnames


def build_other_numeric_transformer(mode: str) -> Pipeline:
    if mode not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError(
            f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={mode!r}. "
            f"Allowed values: {sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES)}"
        )
    if mode == "standardize_zero_fill":
        return Pipeline(steps=[("scale_then_zero_fill", StandardizeThenZeroFill())])
    if mode == "minmax_minus1_1_zero_fill":
        return Pipeline(steps=[("scale_then_zero_fill", MinMaxMinusOneOneThenZeroFill())])
    return Pipeline(steps=[("zero_fill", ZeroFillTransformer())])


def make_preprocessor(
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
    fc_cols: list[str],
    fc_is_correlation: bool = True,
    phenotype_other_numeric_mode: str = "standardize_zero_fill",
) -> ColumnTransformer:
    transformers = []
    if fc_cols:
        if fc_is_correlation:
            fc_pipe = Pipeline(steps=[("zero_fill", ZeroFillTransformer())])
        else:
            fc_pipe = Pipeline(steps=[("scale_then_zero_fill", StandardizeThenZeroFill())])
        transformers.append(("fc", fc_pipe, fc_cols))

    if phenotype_domain_scaled_cols:
        phen_domain_pipe = Pipeline(steps=[("zero_fill", ZeroFillTransformer())])
        transformers.append(("phen_domain", phen_domain_pipe, phenotype_domain_scaled_cols))

    if phenotype_other_numeric_cols:
        phen_other_num_pipe = build_other_numeric_transformer(phenotype_other_numeric_mode)
        transformers.append(("phen_other_num", phen_other_num_pipe, phenotype_other_numeric_cols))

    if phenotype_categorical_cols:
        phen_cat_pipe = Pipeline(steps=[("onehot_missing_zero", MissingAsAllZeroOneHotEncoder())])
        transformers.append(("phen_cat", phen_cat_pipe, phenotype_categorical_cols))

    if not transformers:
        raise ValueError("No feature blocks were provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_transform_preprocessor(
    *,
    X_train_df: pd.DataFrame,
    X_val_df: Optional[pd.DataFrame],
    X_test_df: Optional[pd.DataFrame],
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
    fc_cols: list[str],
) -> tuple[ColumnTransformer, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    preprocessor = make_preprocessor(
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
        fc_is_correlation=FC_IS_CORRELATION,
        phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
    )
    X_train_np = np.asarray(preprocessor.fit_transform(X_train_df), dtype=np.float64)
    X_val_np = None
    if X_val_df is not None:
        X_val_np = np.asarray(preprocessor.transform(X_val_df), dtype=np.float64)
    X_test_np = None
    if X_test_df is not None:
        X_test_np = np.asarray(preprocessor.transform(X_test_df), dtype=np.float64)
    return preprocessor, X_train_np, X_val_np, X_test_np


# =========================================================
# IID train/val/test split
# =========================================================
def _class_count_dict(y_split: np.ndarray) -> dict:
    return {
        str(k): int(v)
        for k, v in pd.Series(y_split).value_counts().sort_index().to_dict().items()
    }


def build_split_audit(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Audit the fixed IID split for sizes, class counts, and index leakage."""
    n = len(y)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    all_assigned = train_set | val_set | test_set

    audit = {
        "n_total": int(n),
        "train": {
            "n_samples": int(len(train_idx)),
            "fraction": float(len(train_idx) / n),
            "class_counts": _class_count_dict(y[train_idx]),
        },
        "validation": {
            "n_samples": int(len(val_idx)),
            "fraction": float(len(val_idx) / n),
            "class_counts": _class_count_dict(y[val_idx]),
        },
        "test": {
            "n_samples": int(len(test_idx)),
            "fraction": float(len(test_idx) / n),
            "class_counts": _class_count_dict(y[test_idx]),
        },
        "index_audit": {
            "train_val_overlap_count": int(len(train_set & val_set)),
            "train_test_overlap_count": int(len(train_set & test_set)),
            "validation_test_overlap_count": int(len(val_set & test_set)),
            "n_unique_assigned_indices": int(len(all_assigned)),
            "all_indices_assigned_exactly_once": bool(len(all_assigned) == n),
        },
    }

    if not audit["index_audit"]["all_indices_assigned_exactly_once"]:
        raise RuntimeError(
            "Fixed split does not assign every sample exactly once. "
            f"Index audit: {audit['index_audit']}"
        )

    overlap_keys = [
        "train_val_overlap_count",
        "train_test_overlap_count",
        "validation_test_overlap_count",
    ]
    if any(audit["index_audit"][k] != 0 for k in overlap_keys):
        raise RuntimeError(f"Index leakage detected across splits: {audit['index_audit']}")

    return audit


def make_train_val_test_split(
    y: np.ndarray,
    seed: int,
    validation_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """Pure IID stratified splitting: no site-aware grouping is used.

    This function is called only once using LR_SEED in the fixed-split workflow.
    The returned train_idx, val_idx, and test_idx are then reused for all seeds.
    """
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    n = len(y)
    test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(test_splitter.split(np.zeros(n), y))
    y_train_val = y[train_val_idx]

    val_fraction_within_train_val = validation_size / (1.0 - test_size)
    val_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction_within_train_val,
        random_state=seed + 1,
    )
    train_rel_idx, val_rel_idx = next(val_splitter.split(np.zeros(len(train_val_idx)), y_train_val))
    train_idx = np.asarray(train_val_idx[train_rel_idx], dtype=int)
    val_idx = np.asarray(train_val_idx[val_rel_idx], dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    split_name = (
        "Fixed StratifiedShuffleSplit train/validation/test "
        "(IID, no SITE_ID grouping): "
        f"created_once_with_LR_SEED={seed}; "
        f"train={1.0 - validation_size - test_size:.2f}, "
        f"validation={validation_size:.2f}, test={test_size:.2f}"
    )
    split_audit = build_split_audit(y, train_idx, val_idx, test_idx)
    split_audit["fixed_split_created_with_seed"] = int(seed)
    split_audit["site_aware_splitting_used"] = False
    split_audit["split_strategy"] = split_name
    return train_idx, val_idx, test_idx, split_name, split_audit


# =========================================================
# DP accounting: unchanged
# =========================================================
def _make_rdp_accountant(orders):
    try:
        from dp_accounting import rdp
        return rdp.RdpAccountant(orders)
    except Exception:
        from dp_accounting.rdp import rdp_privacy_accountant
        return rdp_privacy_accountant.RdpAccountant(orders)


def _compose_poisson_gaussian_event(accountant, sampling: float, noise_multiplier: float, step: int):
    from dp_accounting import dp_event as event
    base_event = event.PoissonSampledDpEvent(sampling, event.GaussianDpEvent(noise_multiplier))
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
    """Uses the same RDP accountant logic as the reference code."""

    def objective(noise_multiplier):
        eps = epsilon_for_noise_multiplier(
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            sampling=sampling,
            step=step,
        )
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    return float(optimal_noise)


# =========================================================
# Manual logistic regression utilities
# =========================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba_from_params(
    X: np.ndarray,
    params: Params,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    logits = X @ params.w
    if fit_intercept:
        logits = logits + params.b
    return sigmoid(logits)


def predict_from_params(
    X: np.ndarray,
    params: Params,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    proba = predict_proba_from_params(X, params, fit_intercept=fit_intercept)
    return (proba >= 0.5).astype(int)


def objective_loss(
    X: np.ndarray,
    y: np.ndarray,
    params: Params,
    *,
    l2_reg_strength: float,
    fit_intercept: bool,
) -> float:
    p = predict_proba_from_params(X, params, fit_intercept=fit_intercept)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    data_loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    reg_loss = 0.5 * l2_reg_strength * float(np.dot(params.w, params.w))
    return float(data_loss + reg_loss)


def compute_per_example_gradients(
    X: np.ndarray,
    y: np.ndarray,
    params: Params,
    *,
    l2_reg_strength: float,
    fit_intercept: bool,
    include_l2_in_per_example_grad: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy implementation retained for clarity/fallback compatibility."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = predict_proba_from_params(X, params, fit_intercept=fit_intercept)
    errors = p - y
    grad_w = errors[:, None] * X
    if include_l2_in_per_example_grad:
        grad_w = grad_w + l2_reg_strength * params.w[None, :]
    grad_b = errors.copy() if fit_intercept else np.zeros_like(errors)
    return grad_w, grad_b


def compute_per_example_gradient_l2_norms(
    grad_w_examples: np.ndarray,
    grad_b_examples: np.ndarray,
) -> np.ndarray:
    squared = np.sum(grad_w_examples ** 2, axis=1) + grad_b_examples ** 2
    return np.sqrt(squared)


def clip_per_example_gradients(
    grad_w_examples: np.ndarray,
    grad_b_examples: np.ndarray,
    *,
    clip_norm: float,
    nu: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norms = compute_per_example_gradient_l2_norms(grad_w_examples, grad_b_examples)
    scales = np.minimum(1.0, clip_norm / (norms + nu))
    clipped_grad_w = grad_w_examples * scales[:, None]
    clipped_grad_b = grad_b_examples * scales
    return clipped_grad_w, clipped_grad_b, norms, scales


def full_batch_dp_gd_update(
    params: Params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    learning_rate: float,
    l2_reg_strength: float,
    fit_intercept: bool,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    l2_reg_handling: str,
) -> tuple[Params, dict]:
    """Reference Python/NumPy update retained as a fallback when Numba is disabled."""
    if l2_reg_handling not in {"add_after_noise", "inside_per_example"}:
        raise ValueError("L2_REG_HANDLING must be one of {'add_after_noise', 'inside_per_example'}.")
    n = X_train.shape[0]
    include_l2_inside = l2_reg_handling == "inside_per_example"
    grad_w_examples, grad_b_examples = compute_per_example_gradients(
        X_train,
        y_train,
        params,
        l2_reg_strength=l2_reg_strength,
        fit_intercept=fit_intercept,
        include_l2_in_per_example_grad=include_l2_inside,
    )
    clipped_grad_w, clipped_grad_b, norms, scales = clip_per_example_gradients(
        grad_w_examples,
        grad_b_examples,
        clip_norm=clip_norm,
    )
    summed_grad_w = np.sum(clipped_grad_w, axis=0)
    summed_grad_b = float(np.sum(clipped_grad_b))
    noise_std = noise_multiplier * clip_norm
    noisy_summed_grad_w = summed_grad_w + rng.normal(loc=0.0, scale=noise_std, size=summed_grad_w.shape)
    noisy_summed_grad_b = summed_grad_b
    if fit_intercept:
        noisy_summed_grad_b = noisy_summed_grad_b + float(rng.normal(loc=0.0, scale=noise_std))
    avg_grad_w = noisy_summed_grad_w / n
    avg_grad_b = noisy_summed_grad_b / n if fit_intercept else 0.0
    if l2_reg_handling == "add_after_noise":
        avg_grad_w = avg_grad_w + l2_reg_strength * params.w
    new_w = params.w - learning_rate * avg_grad_w
    new_b = params.b - learning_rate * avg_grad_b if fit_intercept else 0.0
    update_stats = {
        "batch_size": int(n),
        "mean_unclipped_grad_norm": float(np.mean(norms)),
        "max_unclipped_grad_norm": float(np.max(norms)),
        "fraction_clipped": float(np.mean(scales < 1.0)),
        "avg_noisy_grad_w_norm": float(np.linalg.norm(avg_grad_w)),
        "avg_noisy_grad_b_abs": float(abs(avg_grad_b)),
    }
    return Params(w=new_w, b=float(new_b)), update_stats


# =========================================================
# Numba-JIT DP-GD training kernel
# =========================================================
HISTORY_COLUMNS = [
    "step",
    "learning_rate",
    "batch_size",
    "mean_unclipped_grad_norm",
    "max_unclipped_grad_norm",
    "fraction_clipped",
    "avg_noisy_grad_w_norm",
    "avg_noisy_grad_b_abs",
    "train_accuracy",
    "train_log_loss",
    "train_objective",
    "val_accuracy",
    "val_log_loss",
    "val_objective",
    "test_accuracy",
    "test_log_loss",
    "test_objective",
]


@njit(cache=True)
def _jit_sigmoid_scalar(z: float) -> float:
    if z > 50.0:
        z = 50.0
    elif z < -50.0:
        z = -50.0
    return 1.0 / (1.0 + math.exp(-z))


@njit(cache=True)
def _jit_basic_metrics(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    fit_intercept_int: int,
    l2_reg_strength: float,
) -> tuple[float, float, float]:
    n = X.shape[0]
    d = X.shape[1]
    if n == 0:
        return math.nan, math.nan, math.nan

    correct = 0
    data_loss = 0.0
    for i in range(n):
        z = 0.0
        for j in range(d):
            z += X[i, j] * w[j]
        if fit_intercept_int == 1:
            z += b
        p = _jit_sigmoid_scalar(z)
        if p < 1e-12:
            p_clip = 1e-12
        elif p > 1.0 - 1e-12:
            p_clip = 1.0 - 1e-12
        else:
            p_clip = p
        if (p >= 0.5 and y[i] >= 0.5) or (p < 0.5 and y[i] < 0.5):
            correct += 1
        data_loss += -(y[i] * math.log(p_clip) + (1.0 - y[i]) * math.log(1.0 - p_clip))

    w_norm_sq = 0.0
    for j in range(d):
        w_norm_sq += w[j] * w[j]
    acc = correct / n
    log_loss_value = data_loss / n
    obj = log_loss_value + 0.5 * l2_reg_strength * w_norm_sq
    return acc, log_loss_value, obj


@njit(cache=True)
def _jit_fill_history_row(
    history: np.ndarray,
    rec_idx: int,
    step: int,
    learning_rate: float,
    batch_size: int,
    mean_unclipped_grad_norm: float,
    max_unclipped_grad_norm: float,
    fraction_clipped: float,
    avg_noisy_grad_w_norm: float,
    avg_noisy_grad_b_abs: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    has_val: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    has_test: int,
    w: np.ndarray,
    b: float,
    fit_intercept_int: int,
    l2_reg_strength: float,
) -> None:
    train_acc, train_log_loss, train_obj = _jit_basic_metrics(
        X_train, y_train, w, b, fit_intercept_int, l2_reg_strength
    )
    if has_val == 1:
        val_acc, val_log_loss, val_obj = _jit_basic_metrics(
            X_val, y_val, w, b, fit_intercept_int, l2_reg_strength
        )
    else:
        val_acc, val_log_loss, val_obj = math.nan, math.nan, math.nan

    if has_test == 1:
        test_acc, test_log_loss, test_obj = _jit_basic_metrics(
            X_test, y_test, w, b, fit_intercept_int, l2_reg_strength
        )
    else:
        test_acc, test_log_loss, test_obj = math.nan, math.nan, math.nan

    history[rec_idx, 0] = step
    history[rec_idx, 1] = learning_rate
    history[rec_idx, 2] = batch_size
    history[rec_idx, 3] = mean_unclipped_grad_norm
    history[rec_idx, 4] = max_unclipped_grad_norm
    history[rec_idx, 5] = fraction_clipped
    history[rec_idx, 6] = avg_noisy_grad_w_norm
    history[rec_idx, 7] = avg_noisy_grad_b_abs
    history[rec_idx, 8] = train_acc
    history[rec_idx, 9] = train_log_loss
    history[rec_idx, 10] = train_obj
    history[rec_idx, 11] = val_acc
    history[rec_idx, 12] = val_log_loss
    history[rec_idx, 13] = val_obj
    history[rec_idx, 14] = test_acc
    history[rec_idx, 15] = test_log_loss
    history[rec_idx, 16] = test_obj


@njit(cache=True)
def _jit_train_full_batch_dp_gd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    has_val: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    has_test: int,
    learning_rate: float,
    l2_reg_strength: float,
    fit_intercept_int: int,
    iterations: int,
    clip_norm: float,
    noise_multiplier: float,
    l2_mode: int,
    random_state: int,
    record_every_n_steps: int,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """
    Fused full-batch DP-GD loop.

    l2_mode:
        0 -> add_after_noise
        1 -> inside_per_example

    This computes the same DP update as the reference code but avoids materializing
    the full n_train x n_features per-example gradient matrix at every step.
    """
    n = X_train.shape[0]
    d = X_train.shape[1]
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    history = np.empty((iterations + 1, 17), dtype=np.float64)
    history[:, :] = np.nan

    np.random.seed(random_state)

    rec_idx = 0
    _jit_fill_history_row(
        history,
        rec_idx,
        0,
        learning_rate,
        n,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        X_train,
        y_train,
        X_val,
        y_val,
        has_val,
        X_test,
        y_test,
        has_test,
        w,
        b,
        fit_intercept_int,
        l2_reg_strength,
    )
    rec_idx += 1

    sum_grad_w = np.zeros(d, dtype=np.float64)
    noise_std = noise_multiplier * clip_norm

    for step in range(1, iterations + 1):
        for j in range(d):
            sum_grad_w[j] = 0.0
        sum_grad_b = 0.0
        total_norm = 0.0
        max_norm = 0.0
        clipped_count = 0

        # Per-example gradient, clipping, and summation.
        for i in range(n):
            z = 0.0
            for j in range(d):
                z += X_train[i, j] * w[j]
            if fit_intercept_int == 1:
                z += b
            p = _jit_sigmoid_scalar(z)
            err = p - y_train[i]

            norm_sq = 0.0
            if fit_intercept_int == 1:
                norm_sq += err * err
            for j in range(d):
                gj = err * X_train[i, j]
                if l2_mode == 1:
                    gj += l2_reg_strength * w[j]
                norm_sq += gj * gj

            norm = math.sqrt(norm_sq)
            total_norm += norm
            if norm > max_norm:
                max_norm = norm

            scale = 1.0
            denom = norm + 1e-6
            if clip_norm / denom < 1.0:
                scale = clip_norm / denom
                clipped_count += 1

            for j in range(d):
                gj = err * X_train[i, j]
                if l2_mode == 1:
                    gj += l2_reg_strength * w[j]
                sum_grad_w[j] += gj * scale
            if fit_intercept_int == 1:
                sum_grad_b += err * scale

        avg_grad_w_norm_sq = 0.0
        for j in range(d):
            noisy_sum = sum_grad_w[j] + np.random.normal(0.0, noise_std)
            avg_grad = noisy_sum / n
            if l2_mode == 0:
                avg_grad += l2_reg_strength * w[j]
            w[j] -= learning_rate * avg_grad
            avg_grad_w_norm_sq += avg_grad * avg_grad

        avg_grad_b = 0.0
        if fit_intercept_int == 1:
            noisy_sum_b = sum_grad_b + np.random.normal(0.0, noise_std)
            avg_grad_b = noisy_sum_b / n
            b -= learning_rate * avg_grad_b

        should_record = False
        if step == 1 or step == iterations or step % record_every_n_steps == 0:
            should_record = True

        if should_record:
            _jit_fill_history_row(
                history,
                rec_idx,
                step,
                learning_rate,
                n,
                total_norm / n,
                max_norm,
                clipped_count / n,
                math.sqrt(avg_grad_w_norm_sq),
                abs(avg_grad_b),
                X_train,
                y_train,
                X_val,
                y_val,
                has_val,
                X_test,
                y_test,
                has_test,
                w,
                b,
                fit_intercept_int,
                l2_reg_strength,
            )
            rec_idx += 1

    return w, b, history, rec_idx


def _python_train_full_batch_dp_gd(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    learning_rate: float,
    l2_reg_strength: float,
    fit_intercept: bool,
    iterations: int,
    clip_norm: float,
    noise_multiplier: float,
    l2_reg_handling: str,
    random_state: int,
    record_every_n_steps: int,
) -> tuple[Params, pd.DataFrame]:
    """Original Python update loop fallback."""
    n_train, n_features = X_train.shape
    params = Params(w=np.zeros(n_features, dtype=np.float64), b=0.0)
    rng = np.random.default_rng(random_state)
    rows = []

    def make_row(step: int, update_stats: dict) -> dict:
        row = {"step": int(step), "learning_rate": learning_rate, **update_stats}
        train_metrics, _, _ = evaluate_params(
            X_train,
            y_train,
            params,
            l2_reg_strength=l2_reg_strength,
            fit_intercept=fit_intercept,
            prefix="train",
        )
        row.update(train_metrics)
        if X_val is not None and y_val is not None:
            val_metrics, _, _ = evaluate_params(
                X_val,
                y_val,
                params,
                l2_reg_strength=l2_reg_strength,
                fit_intercept=fit_intercept,
                prefix="val",
            )
            row.update(val_metrics)
        if X_test is not None and y_test is not None:
            test_metrics, _, _ = evaluate_params(
                X_test,
                y_test,
                params,
                l2_reg_strength=l2_reg_strength,
                fit_intercept=fit_intercept,
                prefix="test",
            )
            row.update(test_metrics)
        return row

    rows.append(
        make_row(
            0,
            {
                "batch_size": int(n_train),
                "mean_unclipped_grad_norm": np.nan,
                "max_unclipped_grad_norm": np.nan,
                "fraction_clipped": np.nan,
                "avg_noisy_grad_w_norm": np.nan,
                "avg_noisy_grad_b_abs": np.nan,
            },
        )
    )

    for step in range(1, iterations + 1):
        params, update_stats = full_batch_dp_gd_update(
            params,
            X_train,
            y_train,
            learning_rate=learning_rate,
            l2_reg_strength=l2_reg_strength,
            fit_intercept=fit_intercept,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            rng=rng,
            l2_reg_handling=l2_reg_handling,
        )
        if step == 1 or step == iterations or step % record_every_n_steps == 0:
            rows.append(make_row(step, update_stats))

    return params, pd.DataFrame(rows)


def _run_training_loop(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    learning_rate: float,
    l2_reg_strength: float,
    fit_intercept: bool,
    iterations: int,
    clip_norm: float,
    noise_multiplier: float,
    l2_reg_handling: str,
    random_state: int,
    record_every_n_steps: int,
) -> tuple[Params, pd.DataFrame]:
    if l2_reg_handling not in {"add_after_noise", "inside_per_example"}:
        raise ValueError("L2_REG_HANDLING must be one of {'add_after_noise', 'inside_per_example'}.")

    if USE_NUMBA_JIT and NUMBA_AVAILABLE:
        n_features = X_train.shape[1]
        X_val_in = X_val if X_val is not None else np.zeros((0, n_features), dtype=np.float64)
        y_val_in = y_val if y_val is not None else np.zeros(0, dtype=np.float64)
        X_test_in = X_test if X_test is not None else np.zeros((0, n_features), dtype=np.float64)
        y_test_in = y_test if y_test is not None else np.zeros(0, dtype=np.float64)

        l2_mode = 0 if l2_reg_handling == "add_after_noise" else 1
        safe_seed = int(random_state % (2**31 - 1))
        w, b, history_arr, n_rows = _jit_train_full_batch_dp_gd(
            np.asarray(X_train, dtype=np.float64),
            np.asarray(y_train, dtype=np.float64),
            np.asarray(X_val_in, dtype=np.float64),
            np.asarray(y_val_in, dtype=np.float64),
            1 if X_val is not None and y_val is not None else 0,
            np.asarray(X_test_in, dtype=np.float64),
            np.asarray(y_test_in, dtype=np.float64),
            1 if X_test is not None and y_test is not None else 0,
            float(learning_rate),
            float(l2_reg_strength),
            1 if fit_intercept else 0,
            int(iterations),
            float(clip_norm),
            float(noise_multiplier),
            int(l2_mode),
            int(safe_seed),
            int(max(1, record_every_n_steps)),
        )
        history_df = pd.DataFrame(history_arr[:n_rows].copy(), columns=HISTORY_COLUMNS)
        history_df["step"] = history_df["step"].astype(int)
        history_df["batch_size"] = history_df["batch_size"].astype(int)
        return Params(w=np.asarray(w, dtype=np.float64), b=float(b)), history_df

    if USE_NUMBA_JIT and not NUMBA_AVAILABLE:
        warnings.warn("USE_NUMBA_JIT=True but numba is not installed. Falling back to Python/NumPy loop.")

    return _python_train_full_batch_dp_gd(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        learning_rate=learning_rate,
        l2_reg_strength=l2_reg_strength,
        fit_intercept=fit_intercept,
        iterations=iterations,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        l2_reg_handling=l2_reg_handling,
        random_state=random_state,
        record_every_n_steps=record_every_n_steps,
    )


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
        out["balanced_accuracy"] = safe_scalar_metric(balanced_accuracy_score, y_true, y_pred)
        out["roc_auc"] = safe_scalar_metric(roc_auc_score, y_true, y_proba)
        out["average_precision"] = safe_scalar_metric(average_precision_score, y_true, y_proba)
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


def evaluate_params(
    X: np.ndarray,
    y: np.ndarray,
    params: Params,
    *,
    l2_reg_strength: float,
    fit_intercept: bool,
    prefix: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_proba = predict_proba_from_params(X, params, fit_intercept=fit_intercept)
    y_pred = (y_proba >= 0.5).astype(int)
    base_metrics = compute_prediction_metrics_from_arrays(y_true=y, y_pred=y_pred, y_proba=y_proba)
    base_metrics["objective"] = objective_loss(
        X,
        y,
        params,
        l2_reg_strength=l2_reg_strength,
        fit_intercept=fit_intercept,
    )
    metrics = {f"{prefix}_{k}": v for k, v in base_metrics.items()}
    return metrics, y_pred, y_proba


def selection_score_from_row(row: dict, metric: str, prefix: str = "val") -> float:
    key = f"{prefix}_{metric}"
    if key not in row:
        raise KeyError(f"Selection metric {key!r} not found.")
    value = float(row[key])
    return -value if metric == "log_loss" else value


def print_metric_block(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            print(f"{k:>36s}: {v:.6f}")


# =========================================================
# Manual full-batch DP logistic regression class
# =========================================================
class ManualFullBatchDPLogisticRegression:
    def __init__(
        self,
        *,
        C: float,
        fit_intercept: bool,
        learning_rate: float,
        iterations: int,
        clip_norm: float,
        noise_multiplier: float,
        l2_reg_handling: str,
        random_state: int,
        record_every_n_steps: int = 1,
    ) -> None:
        if C <= 0:
            raise ValueError("C must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive.")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")

        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.l2_reg_handling = str(l2_reg_handling)
        self.random_state = int(random_state)
        self.record_every_n_steps = max(1, int(record_every_n_steps))

        self.params_: Optional[Params] = None
        self.l2_reg_strength_: Optional[float] = None
        self.step_history_: pd.DataFrame = pd.DataFrame()
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> "ManualFullBatchDPLogisticRegression":
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of rows.")

        n_train, n_features = X_train.shape
        self.n_features_in_ = n_features
        self.l2_reg_strength_ = 1.0 / (self.C * n_train)

        self.params_, self.step_history_ = _run_training_loop(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            learning_rate=self.learning_rate,
            l2_reg_strength=self.l2_reg_strength_,
            fit_intercept=self.fit_intercept,
            iterations=self.iterations,
            clip_norm=self.clip_norm,
            noise_multiplier=self.noise_multiplier,
            l2_reg_handling=self.l2_reg_handling,
            random_state=self.random_state,
            record_every_n_steps=self.record_every_n_steps,
        )
        return self

    def predict_proba_1d(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Model is not fitted.")
        return predict_proba_from_params(
            np.asarray(X, dtype=np.float64),
            self.params_,
            fit_intercept=self.fit_intercept,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_1d(X) >= 0.5).astype(int)

    def metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        prefix: str,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        if self.params_ is None or self.l2_reg_strength_ is None:
            raise RuntimeError("Model is not fitted.")
        return evaluate_params(
            X,
            y,
            self.params_,
            l2_reg_strength=self.l2_reg_strength_,
            fit_intercept=self.fit_intercept,
            prefix=prefix,
        )


# =========================================================
# Training helpers
# =========================================================
import hashlib

def stable_lr_key(lr: float) -> int:
    payload = f"{float(lr):.16g}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


def make_lr_seed(base_seed: int, lr: float, *, final: bool = False) -> int:
    lr_key = stable_lr_key(lr)
    offset = 123457 if final else 0
    return int((int(base_seed) * 1000003 + lr_key + offset) % (2**32 - 1))


def train_one_learning_rate(
    *,
    learning_rate: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
) -> tuple[dict, pd.DataFrame, ManualFullBatchDPLogisticRegression]:
    model = ManualFullBatchDPLogisticRegression(
        C=tuned_C,
        fit_intercept=TUNED_FIT_INTERCEPT,
        learning_rate=learning_rate,
        iterations=ITERATIONS,
        clip_norm=CLIP_NORM,
        noise_multiplier=noise_multiplier,
        l2_reg_handling=L2_REG_HANDLING,
        random_state=make_lr_seed(seed, learning_rate, final=False),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    history = model.step_history_.copy()
    last_row = history.iloc[-1].to_dict()

    train_metrics, _, _ = model.metrics(X_train, y_train, prefix="train")
    val_metrics, _, _ = model.metrics(X_val, y_val, prefix="val")

    final_metrics = {
        "learning_rate": learning_rate,
        "phase": "selection_train_model",
        **last_row,
        **train_metrics,
        **val_metrics,
    }
    final_metrics["selection_score"] = selection_score_from_row(
        final_metrics,
        metric=SELECTION_METRIC,
        prefix="val",
    )
    return final_metrics, history, model


def _run_lr_trial_for_parallel(
    learning_rate: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
):
    return learning_rate, train_one_learning_rate(
        learning_rate=learning_rate,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        noise_multiplier=noise_multiplier,
        seed=seed,
        tuned_C=tuned_C,
    )


def evaluate_learning_rate_grid_with_expansion(
    *,
    initial_lrs: list[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, ManualFullBatchDPLogisticRegression]]:
    tried_rows: dict[float, dict] = {}
    histories: dict[float, pd.DataFrame] = {}
    models: dict[float, ManualFullBatchDPLogisticRegression] = {}
    current_lrs = sorted(set(float(x) for x in initial_lrs))

    for expansion_round in range(MAX_ENDPOINT_EXPANSIONS + 1):
        print(f"\nLearning-rate grid round {expansion_round + 1}: {current_lrs}")
        lrs_to_run = [lr for lr in current_lrs if lr not in tried_rows]

        if PARALLELIZE_LEARNING_RATES and JOBLIB_AVAILABLE and len(lrs_to_run) > 1:
            print(
                f"Parallel LR testing enabled: n_jobs={LR_PARALLEL_N_JOBS}, "
                f"backend={LR_PARALLEL_BACKEND}."
            )
            results = Parallel(n_jobs=LR_PARALLEL_N_JOBS, backend=LR_PARALLEL_BACKEND)(
                delayed(_run_lr_trial_for_parallel)(
                    lr,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    noise_multiplier,
                    seed,
                    tuned_C,
                )
                for lr in lrs_to_run
            )
        else:
            if PARALLELIZE_LEARNING_RATES and not JOBLIB_AVAILABLE:
                print("joblib is unavailable; running LR trials sequentially.")
            results = []
            for lr in lrs_to_run:
                print(f"Training learning rate: {lr:g}")
                results.append(
                    _run_lr_trial_for_parallel(
                        lr,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        noise_multiplier,
                        seed,
                        tuned_C,
                    )
                )

        for lr, (row, hist, model) in results:
            tried_rows[lr] = row
            histories[lr] = hist
            models[lr] = model
            hist.to_csv(out_dir / f"step_history_lr_{lr:g}.csv", index=False)
            print(
                f"  lr={lr:g} | "
                f"val_acc={row['val_accuracy']:.6f} | "
                f"val_bal_acc={row['val_balanced_accuracy']:.6f} | "
                f"val_log_loss={row['val_log_loss']:.6f}"
            )

        result_df = pd.DataFrame([tried_rows[lr] for lr in sorted(tried_rows)]).reset_index(drop=True)
        best_idx = int(result_df["selection_score"].idxmax())
        best_lr = float(result_df.loc[best_idx, "learning_rate"])
        ordered = sorted(set(current_lrs))

        if len(ordered) >= 2 and best_lr == ordered[0]:
            new_lr = ordered[0] / ENDPOINT_EXPANSION_FACTOR
            if new_lr not in ordered:
                current_lrs = sorted(set(ordered + [new_lr]))
                print(f"Best LR is lower endpoint. Expanding downward by /{ENDPOINT_EXPANSION_FACTOR:g}: {new_lr:g}")
                continue

        if len(ordered) >= 2 and best_lr == ordered[-1]:
            new_lr = ordered[-1] * ENDPOINT_EXPANSION_FACTOR
            if new_lr not in ordered:
                current_lrs = sorted(set(ordered + [new_lr]))
                print(f"Best LR is upper endpoint. Expanding upward by x{ENDPOINT_EXPANSION_FACTOR:g}: {new_lr:g}")
                continue

        break

    result_df = pd.DataFrame([tried_rows[lr] for lr in sorted(tried_rows)]).reset_index(drop=True)
    return result_df, histories, models


def train_final_refit_model(
    *,
    best_lr: float,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_multiplier: float,
    seed: int,
    tuned_C: float,
) -> tuple[ManualFullBatchDPLogisticRegression, pd.DataFrame, dict, dict, np.ndarray, np.ndarray]:
    final_model = ManualFullBatchDPLogisticRegression(
        C=tuned_C,
        fit_intercept=TUNED_FIT_INTERCEPT,
        learning_rate=best_lr,
        iterations=ITERATIONS,
        clip_norm=CLIP_NORM,
        noise_multiplier=noise_multiplier,
        l2_reg_handling=L2_REG_HANDLING,
        random_state=make_lr_seed(seed, best_lr, final=True),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
    )
    final_model.fit(X_train_val, y_train_val, X_test=X_test, y_test=y_test)
    final_history = final_model.step_history_.copy()
    train_val_metrics, _, _ = final_model.metrics(X_train_val, y_train_val, prefix="trainval")
    test_metrics, y_test_pred, y_test_proba = final_model.metrics(X_test, y_test, prefix="test")
    return final_model, final_history, train_val_metrics, test_metrics, y_test_pred, y_test_proba


# =========================================================
# Subgroup metrics
# =========================================================
def build_reporting_group_series(df_meta: pd.DataFrame, feature_name: str) -> pd.Series:
    if feature_name not in df_meta.columns:
        return pd.Series(MISSING_GROUP_LABEL, index=df_meta.index, dtype="string", name=feature_name)

    s = df_meta[feature_name].copy()
    if feature_name in {"SITE_ID", "FIQ_TEST_TYPE", "PIQ_TEST_TYPE", "VIQ_TEST_TYPE"}:
        s = s.map(
            normalize_test_type if feature_name.endswith("_TEST_TYPE") else lambda x: np.nan if pd.isna(x) else str(x).strip().upper()
        )
        s = pd.Series(s, index=df_meta.index, name=feature_name, dtype="object")
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)
        return s.astype("string")

    if feature_name == "HANDEDNESS_CATEGORY":
        s = s.map(normalize_handedness_category)
        s = pd.Series(s, index=df_meta.index, name=feature_name, dtype="object")
        s = s.where(~pd.isna(s), MISSING_GROUP_LABEL)
        mapping = {"R": "Right", "L": "Left", "Ambi": "Mixed/Ambi", "MIXED": "Mixed/Ambi", "L->R": "Mixed/Ambi"}
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
        s_binned = pd.cut(s_num, bins=spec["bins"], labels=spec["labels"], right=False, include_lowest=True).astype("object")
        s_binned = pd.Series(s_binned, index=df_meta.index, name=feature_name, dtype="object")
        if feature_name in {"FIQ", "PIQ", "VIQ"}:
            s_binned = s_binned.where(~pd.isna(s_binned), f"{feature_name} {MISSING_GROUP_LABEL}")
        return s_binned.astype("string")

    s = s.astype("string")
    s = s.fillna(MISSING_GROUP_LABEL)
    s = s.replace({"": MISSING_GROUP_LABEL, "nan": MISSING_GROUP_LABEL, "None": MISSING_GROUP_LABEL})
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
            metrics = compute_prediction_metrics_from_arrays(y_true=y_true[mask], y_pred=y_pred[mask], y_proba=y_proba[mask])
            rows.append(
                {
                    "experiment_key": experiment_key,
                    "experiment_name": experiment_name,
                    "split": split_name,
                    "best_lr": best_lr,
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
    combined_df["feature"] = pd.Categorical(combined_df["feature"], categories=feature_order, ordered=True)
    combined_df = combined_df.sort_values(["experiment_key", "split", "feature"], kind="stable").reset_index(drop=True)
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
    histories: dict[float, pd.DataFrame],
    *,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False

    for lr, hist in sorted(histories.items()):
        if hist is None or hist.empty or metric_col not in hist.columns:
            continue

        x = pd.to_numeric(hist["step"], errors="coerce")
        y = pd.to_numeric(hist[metric_col], errors="coerce")

        finite_mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())

        if not np.any(finite_mask):
            continue

        plt.plot(
            x.to_numpy()[finite_mask],
            y.to_numpy()[finite_mask],
            label=f"lr={lr:g}",
        )
        plotted = True

    if not plotted:
        plt.close()
        print(f"Skipping plot because no finite values were found for: {metric_col}")
        return

    plt.xlabel("DP-GD step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(out_path)



def save_all_step_plots(histories: dict[float, pd.DataFrame], out_dir: Path) -> None:
    """
    These histories are from LR selection models trained on train and evaluated
    on validation only. Therefore, do not plot test_* metrics here.
    Test metrics are plotted later only for the final train+validation refit model.
    """
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
            histories,
            metric_col=metric_col,
            ylabel=ylabel,
            title=title,
            out_path=out_dir / f"{metric_col}_vs_step_all_lrs.png",
        )


def save_metric_vs_lr_plots(selection_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = selection_df.sort_values("learning_rate").copy()
    plot_specs = [
        ("accuracy", "Accuracy", "Selection accuracy vs learning rate"),
        ("log_loss", "Log loss", "Selection log loss vs learning rate"),
        ("balanced_accuracy", "Balanced accuracy", "Selection balanced accuracy vs learning rate"),
        ("roc_auc", "AuROC", "Selection AuROC vs learning rate"),
    ]
    for metric, ylabel, title in plot_specs:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in plot_df.columns or val_col not in plot_df.columns:
            continue
        plt.figure(figsize=(9, 6))
        plt.semilogx(plot_df["learning_rate"], plot_df[train_col], marker="o", label="Train")
        plt.semilogx(plot_df["learning_rate"], plot_df[val_col], marker="o", label="Validation")
        plt.xlabel("Learning rate")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        save_or_show(out_dir / f"{metric}_vs_learning_rate.png")


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, split_label: str) -> None:
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


def save_roc_pr_plots(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path, split_label: str) -> None:
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


def save_calibration_plot(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path, split_label: str) -> None:
    plt.figure(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.title(f"{split_label} calibration curve")
    save_or_show(out_dir / f"{split_label.lower()}_calibration_curve.png")


def save_probability_histogram(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path, split_label: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(y_proba[y_true == 0], bins=20, alpha=0.6, label="Control (0)")
    plt.hist(y_proba[y_true == 1], bins=20, alpha=0.6, label="Autism (1)")
    plt.xlabel("Predicted probability of Autism")
    plt.ylabel("Count")
    plt.title(f"{split_label} predicted-probability distribution by class")
    plt.legend()
    save_or_show(out_dir / f"{split_label.lower()}_probability_histogram.png")


def save_coefficients(
    preprocessor: ColumnTransformer,
    model: ManualFullBatchDPLogisticRegression,
    out_dir: Path,
    top_k: int = 20,
) -> pd.DataFrame:
    if model.params_ is None:
        raise RuntimeError("Model is not fitted.")
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(model.params_.w.shape[0])], dtype=object)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.params_.w,
            "abs_coefficient": np.abs(model.params_.w),
        }
    )

    def infer_modality(name: str) -> str:
        if name.startswith("fc__"):
            return "fc"
        if name.startswith("phen_domain__"):
            return "phen_domain"
        if name.startswith("phen_other_num__"):
            return "phen_other_num"
        if name.startswith("phen_cat__"):
            return "phen_cat"
        return "other"

    coef_df["modality"] = coef_df["feature"].map(infer_modality)
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(out_dir / "final_dp_gd_coefficients_all.csv", index=False)

    top_pos = coef_df.sort_values("coefficient", ascending=False).head(top_k)
    top_neg = coef_df.sort_values("coefficient", ascending=True).head(top_k)
    plot_df = pd.concat([top_neg, top_pos], axis=0)
    plt.figure(figsize=(13, 9))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title("Most influential DP-GD logistic-regression coefficients")
    save_or_show(out_dir / "final_dp_gd_top_coefficients.png")

    modality_importance = coef_df.groupby("modality", as_index=False)["abs_coefficient"].sum().sort_values(
        "abs_coefficient", ascending=False
    )
    modality_importance.to_csv(out_dir / "final_dp_gd_coefficient_modality_importance.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.bar(modality_importance["modality"], modality_importance["abs_coefficient"])
    plt.xlabel("Feature block")
    plt.ylabel("Sum of absolute coefficients")
    plt.title("DP-GD coefficient importance by feature block")
    save_or_show(out_dir / "final_dp_gd_coefficient_modality_importance.png")
    return coef_df


# =========================================================
# Fixed-split / preprocess-dump helpers
# =========================================================
def _json_clean_for_split(obj):
    if isinstance(obj, dict):
        return {str(k): _json_clean_for_split(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean_for_split(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean_for_split(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json_file(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_clean_for_split(data), f, indent=2)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required preprocess dump CSV not found: {path}")
    return pd.read_csv(path)


def _strip_prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    out = df[cols].copy()
    out.columns = [str(c)[len(prefix):] for c in cols]
    return out


def _restore_metadata_columns_from_dump(all_df: pd.DataFrame) -> pd.DataFrame:
    """Build df_model_raw-like metadata from the common non-private dump."""
    meta = pd.DataFrame(index=all_df.index)

    if "DX_GROUP_original" in all_df.columns:
        meta[TARGET_COL] = pd.to_numeric(all_df["DX_GROUP_original"], errors="coerce")
    elif "y_binary" in all_df.columns:
        y_tmp = pd.to_numeric(all_df["y_binary"], errors="coerce")
        meta[TARGET_COL] = np.where(y_tmp == 1, 1, 2)

    if "SITE_ID" in all_df.columns:
        meta["SITE_ID"] = all_df["SITE_ID"].astype("string")

    for feature_name in SUBGROUP_EVAL_FEATURES:
        if feature_name in meta.columns:
            continue
        meta_col = f"meta__{feature_name}"
        phen_col = f"phen__{feature_name}"
        if meta_col in all_df.columns:
            meta[feature_name] = all_df[meta_col]
        elif phen_col in all_df.columns:
            meta[feature_name] = all_df[phen_col]
        elif feature_name in all_df.columns:
            meta[feature_name] = all_df[feature_name]

    return meta


def load_global_data_from_preprocess_dump(*, seed: int, output_dir: Path) -> PreparedGlobalData:
    """Load fixed IID prepared data dumped by the non-private pipeline.

    The root CSVs are the common prepared tables with fc_* columns and phen__*
    columns. This reconstructs the objects expected by run_single_experiment.
    """
    seed = int(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nUsing fixed prepared CSV dump from non-private IID pipeline:")
    print(PREPROCESS_DUMP_DIR)

    train_df = _read_required_csv(PREPROCESS_DUMP_TRAIN_CSV)
    val_df = _read_required_csv(PREPROCESS_DUMP_VAL_CSV)
    test_df = _read_required_csv(PREPROCESS_DUMP_TEST_CSV)
    train_val_df = _read_required_csv(PREPROCESS_DUMP_TRAIN_VAL_CSV)

    if PRINT_PREPROCESS_DUMP_COLUMN_NAMES:
        print("\nPreprocess dump train columns:")
        print(train_df.columns.tolist())

    required_cols = {"row_index_after_valid_mask", "split", "y_binary"}
    for split_name_for_error, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing_required = sorted(required_cols - set(split_df.columns))
        if missing_required:
            raise ValueError(
                f"{split_name_for_error}_fixed.csv is missing required columns: {missing_required}"
            )

    train_idx = train_df["row_index_after_valid_mask"].astype(int).to_numpy()
    val_idx = val_df["row_index_after_valid_mask"].astype(int).to_numpy()
    test_idx = test_df["row_index_after_valid_mask"].astype(int).to_numpy()
    train_val_idx_from_dump = train_val_df["row_index_after_valid_mask"].astype(int).to_numpy()

    expected_train_val_set = set(train_idx.tolist()) | set(val_idx.tolist())
    observed_train_val_set = set(train_val_idx_from_dump.tolist())
    if expected_train_val_set != observed_train_val_set:
        raise ValueError(
            "train_val_fixed.csv does not match the union of train_fixed.csv and val_fixed.csv."
        )

    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    all_df["row_index_after_valid_mask"] = all_df["row_index_after_valid_mask"].astype(int)
    all_df = all_df.sort_values("row_index_after_valid_mask").reset_index(drop=True)

    expected = np.arange(len(all_df), dtype=int)
    observed = all_df["row_index_after_valid_mask"].to_numpy(dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError(
            "Preprocess dump row_index_after_valid_mask must cover 0..n-1 exactly. "
            "Please check that train/val/test_fixed.csv came from the same fixed split."
        )

    y = all_df["y_binary"].astype(int).to_numpy()
    fc_cols = sorted([c for c in all_df.columns if str(c).startswith("fc_")])
    if not fc_cols:
        raise ValueError("No fc_* columns found in preprocess dump CSVs.")

    X_conn_model = all_df[fc_cols].to_numpy(dtype=np.float64)
    X_conn_model = np.where(np.isfinite(X_conn_model), X_conn_model, np.nan)

    df_model = _strip_prefix_columns(all_df, "phen__")
    if TARGET_COL not in df_model.columns:
        df_model[TARGET_COL] = np.where(y == 1, 1, 2)
    if "SITE_ID" in all_df.columns and "SITE_ID" not in df_model.columns:
        df_model["SITE_ID"] = all_df["SITE_ID"].astype("string")

    df_model_raw = _restore_metadata_columns_from_dump(all_df)
    if TARGET_COL not in df_model_raw.columns:
        df_model_raw[TARGET_COL] = np.where(y == 1, 1, 2)

    df = df_model.copy()

    feature_json_path = resolve_feature_json_path(FEATURE_JSON_CANDIDATES)
    feature_groups = load_feature_groups(feature_json_path)
    print("\nFeature JSON path used:")
    print(feature_json_path)

    split_name = (
        "Fixed IID train/validation/test split loaded from non-private "
        f"preprocess dump at {PREPROCESS_DUMP_DIR}; expected LR_SEED={LR_SEED}"
    )
    split_audit = build_split_audit(y, train_idx, val_idx, test_idx)
    split_audit["fixed_split_created_with_seed"] = int(LR_SEED)
    split_audit["site_aware_splitting_used"] = False
    split_audit["split_strategy"] = split_name
    split_audit["preprocess_dump_used"] = True
    split_audit["preprocess_dump_dir"] = str(PREPROCESS_DUMP_DIR)

    summary_json = PREPROCESS_DUMP_DIR / "fixed_prepared_data_summary.json"
    if summary_json.exists():
        try:
            with open(summary_json, "r") as f:
                dump_summary = json.load(f)
            split_audit["source_fixed_prepared_data_summary"] = dump_summary
            source_seed = dump_summary.get("fixed_split_created_with_seed", None)
            if source_seed is not None and int(source_seed) != int(LR_SEED):
                print(
                    "WARNING: preprocess dump reports fixed_split_created_with_seed="
                    f"{source_seed}, but this script has LR_SEED={LR_SEED}. "
                    "The loaded dump split will still be used exactly as-is."
                )
        except Exception as exc:
            print(f"Could not read fixed_prepared_data_summary.json: {exc}")

    print("\nAfter loading fixed preprocess dump:")
    print("df_model shape:", df_model.shape)
    print("X_conn_model shape:", X_conn_model.shape)
    print("Binary class counts (0=Control, 1=Autism):")
    print(pd.Series(y).value_counts().sort_index())
    print("\nFixed split loaded from dump:")
    print(f"Train size      : {len(train_idx)}")
    print(f"Validation size : {len(val_idx)}")
    print(f"Test size       : {len(test_idx)}")

    np.save(output_dir / "X_conn_model.npy", X_conn_model)
    np.save(output_dir / "y_binary.npy", y)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    return PreparedGlobalData(
        seed=seed,
        output_dir=output_dir,
        df=df.reset_index(drop=True),
        df_model_raw=df_model_raw.reset_index(drop=True),
        df_model=df_model.reset_index(drop=True),
        X_conn_model=X_conn_model,
        y=y,
        feature_groups=feature_groups,
        feature_json_path=feature_json_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_name=split_name,
        split_audit=split_audit,
        preprocess_dump_used=True,
        preprocess_dump_dir=PREPROCESS_DUMP_DIR,
    )


def save_fixed_split_artifacts(data: PreparedGlobalData) -> None:
    """Save the fixed split once at OUTPUT_DIR and EPSILON_OUTPUT_DIR for reuse/auditing."""
    train_val_idx = np.concatenate([data.train_idx, data.val_idx])
    for root in [OUTPUT_DIR, EPSILON_OUTPUT_DIR]:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "train_idx.npy", data.train_idx)
        np.save(root / "val_idx.npy", data.val_idx)
        np.save(root / "test_idx.npy", data.test_idx)
        np.save(root / "fixed_train_idx.npy", data.train_idx)
        np.save(root / "fixed_val_idx.npy", data.val_idx)
        np.save(root / "fixed_test_idx.npy", data.test_idx)
        np.save(root / "fixed_train_val_idx.npy", train_val_idx)
        np.save(root / "y_binary.npy", data.y)
        np.save(root / "seed_list.npy", np.asarray(SEED_LIST, dtype=int))
        np.save(root / "lr_seed.npy", np.asarray([LR_SEED], dtype=int))

        rows = []
        for split_label, idx_arr in [
            ("train", data.train_idx),
            ("validation", data.val_idx),
            ("test", data.test_idx),
        ]:
            for idx in idx_arr:
                rows.append(
                    {
                        "row_index_after_valid_mask": int(idx),
                        "split": split_label,
                        "y_binary": int(data.y[int(idx)]),
                    }
                )
        split_df = pd.DataFrame(rows).sort_values("row_index_after_valid_mask").reset_index(drop=True)
        split_df.to_csv(root / "fixed_split_assignment.csv", index=False)
        split_df.to_csv(root / "fixed_split_indices.csv", index=False)

        save_json_file(data.split_audit, root / "fixed_split_audit.json")
        save_json_file(
            {
                "fixed_split_created_with_seed": int(LR_SEED),
                "seed_list": [int(s) for s in SEED_LIST],
                "lr_seed": int(LR_SEED),
                "lr_seed_in_seed_list": bool(int(LR_SEED) in [int(s) for s in SEED_LIST]),
                "site_aware_splitting_used": False,
                "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
                "validation_size": VALIDATION_SIZE,
                "test_size": TEST_SIZE,
                "n_total_after_valid_mask": int(len(data.y)),
                "n_train": int(len(data.train_idx)),
                "n_validation": int(len(data.val_idx)),
                "n_test": int(len(data.test_idx)),
                "split_strategy": data.split_name,
                "split_audit": data.split_audit,
                "use_preprocess_dump": bool(data.preprocess_dump_used),
                "preprocess_dump_dir": str(data.preprocess_dump_dir) if data.preprocess_dump_dir is not None else None,
                "policy": (
                    "The IID train/validation/test split is created or loaded exactly once "
                    "using LR_SEED, then reused unchanged for every seed in SEED_LIST."
                ),
            },
            root / "fixed_split_summary.json",
        )


def save_prepared_arrays_for_output_dir(data: PreparedGlobalData, output_dir: Path) -> None:
    """Preserve the original per-seed array filenames while reusing fixed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_conn_model.npy", data.X_conn_model)
    np.save(output_dir / "y_binary.npy", data.y)
    np.save(output_dir / "train_idx.npy", data.train_idx)
    np.save(output_dir / "val_idx.npy", data.val_idx)
    np.save(output_dir / "test_idx.npy", data.test_idx)


def clone_prepared_global_data_for_seed(
    fixed_data: PreparedGlobalData,
    *,
    seed: int,
    output_dir: Path,
) -> PreparedGlobalData:
    """Reuse the exact fixed data and split, changing only seed/output_dir."""
    cloned = PreparedGlobalData(
        seed=int(seed),
        output_dir=output_dir,
        df=fixed_data.df,
        df_model_raw=fixed_data.df_model_raw,
        df_model=fixed_data.df_model,
        X_conn_model=fixed_data.X_conn_model,
        y=fixed_data.y,
        feature_groups=fixed_data.feature_groups,
        feature_json_path=fixed_data.feature_json_path,
        train_idx=fixed_data.train_idx.copy(),
        val_idx=fixed_data.val_idx.copy(),
        test_idx=fixed_data.test_idx.copy(),
        split_name=fixed_data.split_name,
        split_audit=fixed_data.split_audit,
        preprocess_dump_used=fixed_data.preprocess_dump_used,
        preprocess_dump_dir=fixed_data.preprocess_dump_dir,
    )
    save_prepared_arrays_for_output_dir(cloned, output_dir)
    return cloned


# =========================================================
# Global data preparation
# =========================================================
def prepare_global_data(*, seed: int, output_dir: Path) -> PreparedGlobalData:
    """
    Prepare ABIDE data once for the fixed-split workflow.

    This function is called for LR_SEED only. It either rebuilds the original
    raw ABIDE + X_conn preparation pipeline, or, when USE_PREPROCESS_DUMP=True,
    loads the fixed prepared CSV dump from the non-private pipeline. Other
    SEED_LIST runs reuse the returned data/split through clone_prepared_global_data_for_seed.
    """
    seed = int(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_PREPROCESS_DUMP:
        return load_global_data_from_preprocess_dump(seed=seed, output_dir=output_dir)

    if PHENOTYPE_OTHER_NUMERIC_MODE not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError(
            f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={PHENOTYPE_OTHER_NUMERIC_MODE!r}. "
            f"Allowed values: {sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES)}"
        )

    df_raw = load_abide_dataframe(CSV_PATH)
    df = clean_abide_dataframe(df_raw)

    if "HANDEDNESS_CATEGORY" in df.columns:
        df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].map(normalize_handedness_category)

    if "SITE_ID" in df.columns:
        df["SITE_ID"] = (
            df["SITE_ID"].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-9999": pd.NA})
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
            f"but phenotypic dataframe has {len(df)} rows. Please ensure X_conn is aligned with abide.phenotypic."
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

    train_idx, val_idx, test_idx, split_name, split_audit = make_train_val_test_split(
        y=y,
        seed=seed,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
    )

    print("\nGlobal split strategy used for all experiments:")
    print(split_name)
    print(f"Fixed split seed: {seed}")
    print(f"Train size      : {len(train_idx)}")
    print(f"Validation size : {len(val_idx)}")
    print(f"Test size       : {len(test_idx)}")

    np.save(output_dir / "X_conn_model.npy", X_conn_model)
    np.save(output_dir / "y_binary.npy", y)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    return PreparedGlobalData(
        seed=seed,
        output_dir=output_dir,
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
        split_audit=split_audit,
        preprocess_dump_used=False,
        preprocess_dump_dir=None,
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
    tune_learning_rate: bool,
    fixed_best_lr: Optional[float] = None,
    seed_role: str = "seed_eval",
    lr_source_seed: Optional[int] = None,
    lr_source_experiment_key: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Run one experiment for one seed.

    If tune_learning_rate=True, this performs the original LR-grid tuning on
    this seed and returns the selected best LR.

    If tune_learning_rate=False, this does not search over the LR grid. It uses
    fixed_best_lr, trains the selected train-split model with that LR, then refits
    on train+validation and evaluates on test. DP-GD logic and DP accounting are
    unchanged.
    """
    if not tune_learning_rate and fixed_best_lr is None:
        raise ValueError("fixed_best_lr must be provided when tune_learning_rate=False.")

    seed = int(global_data.seed)
    experiment_dir = global_data.output_dir / experiment_key
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 120)
    print(f"Running experiment: {experiment_name}")
    print(f"Seed: {seed} | seed_role: {seed_role}")
    print(f"Using tuned_C: {tuned_C}")
    print("#" * 120)

    selected_phenotype_cols, phenotype_numeric_cols, phenotype_categorical_cols, missing_cols = build_feature_lists_from_selected_keys(
        df=global_data.df,
        feature_groups=global_data.feature_groups,
        selected_group_keys=selected_group_keys,
    )

    phenotype_domain_scaled_cols = [
        c for c in phenotype_numeric_cols if c in DOMAIN_SCALED_NUMERIC_COLS and c in global_data.df_model.columns
    ]
    phenotype_other_numeric_cols = [
        c for c in phenotype_numeric_cols if c not in DOMAIN_SCALED_NUMERIC_COLS and c in global_data.df_model.columns
    ]

    print("\nSelected feature group keys:")
    print(selected_group_keys)
    print("\nSelected phenotype columns:")
    print(selected_phenotype_cols)
    print("\nPhenotype numeric columns:")
    print(phenotype_numeric_cols)
    print("\nPhenotype categorical columns:")
    print(phenotype_categorical_cols)
    print("\nRequested JSON columns absent from dataframe:")
    print(missing_cols if missing_cols else "None")
    print("\nPhenotype numeric columns with deterministic domain scaling + zero fill:")
    print(phenotype_domain_scaled_cols)
    print("\nPhenotype other numeric columns:")
    print(phenotype_other_numeric_cols)

    X_df, fc_cols = build_combined_dataframe(
        df_model=global_data.df_model,
        X_conn_model=global_data.X_conn_model,
        phenotype_selected_cols=selected_phenotype_cols,
    )

    train_idx = global_data.train_idx
    val_idx = global_data.val_idx
    test_idx = global_data.test_idx

    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)

    y_train = global_data.y[train_idx]
    y_val = global_data.y[val_idx]
    y_test = global_data.y[test_idx]

    X_train_val_df = pd.concat([X_train_df, X_val_df], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    selection_preprocessor, X_train, X_val, X_test = fit_transform_preprocessor(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
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

    lr_selection_mode = "tuned_on_lr_seed" if tune_learning_rate else "fixed_from_lr_seed"

    selection_dp_summary = {
        "seed": seed,
        "seed_role": seed_role,
        "phase": "selection_train_model",
        "lr_selection_mode": lr_selection_mode,
        "lr_source_seed": lr_source_seed,
        "lr_source_experiment_key": lr_source_experiment_key,
        "fixed_best_lr": None if fixed_best_lr is None else float(fixed_best_lr),
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": selection_achieved_epsilon,
        "delta": selection_delta,
        "noise_multiplier": selection_noise_multiplier,
        "clip_norm": CLIP_NORM,
        "iterations": ITERATIONS,
        "sampling": selection_sampling,
        "n_train_for_accounting": n_train,
        "tuned_C": tuned_C,
        "full_batch": True,
        "numba_jit_requested": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
    }
    with open(experiment_dir / "dp_accounting_selection_train_model.json", "w") as f:
        json.dump(selection_dp_summary, f, indent=2)
    print("\nSelection DP accounting summary:")
    print(json.dumps(selection_dp_summary, indent=2))

    if tune_learning_rate:
        selection_df, histories, models = evaluate_learning_rate_grid_with_expansion(
            initial_lrs=LEARNING_RATES_INITIAL,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            noise_multiplier=selection_noise_multiplier,
            seed=seed,
            tuned_C=tuned_C,
            out_dir=experiment_dir,
        )
    else:
        best_lr_fixed = float(fixed_best_lr)
        print("\nSkipping LR-grid tuning for this seed.")
        print(f"Using fixed best LR from LR_SEED={lr_source_seed}: {best_lr_fixed:g}")
        row, hist, model = train_one_learning_rate(
            learning_rate=best_lr_fixed,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            noise_multiplier=selection_noise_multiplier,
            seed=seed,
            tuned_C=tuned_C,
        )
        histories = {best_lr_fixed: hist}
        models = {best_lr_fixed: model}
        selection_df = pd.DataFrame([row]).reset_index(drop=True)
        hist.to_csv(experiment_dir / f"step_history_lr_{best_lr_fixed:g}.csv", index=False)
        print(
            f"  lr={best_lr_fixed:g} | "
            f"val_acc={row['val_accuracy']:.6f} | "
            f"val_bal_acc={row['val_balanced_accuracy']:.6f} | "
            f"val_log_loss={row['val_log_loss']:.6f}"
        )

    selection_df = selection_df.sort_values("learning_rate").reset_index(drop=True)
    selection_df.insert(0, "seed", seed)
    selection_df.insert(1, "seed_role", seed_role)
    selection_df.insert(2, "experiment_key", experiment_key)
    selection_df.insert(3, "experiment_name", experiment_name)
    selection_df.insert(4, "lr_selection_mode", lr_selection_mode)
    selection_df.insert(5, "lr_source_seed", lr_source_seed)
    selection_df.insert(6, "lr_source_experiment_key", lr_source_experiment_key)
    selection_df.insert(7, "target_epsilon", TARGET_EPSILON)
    selection_df.insert(8, "achieved_epsilon", selection_achieved_epsilon)
    selection_df.insert(9, "delta", selection_delta)
    selection_df.insert(10, "noise_multiplier", selection_noise_multiplier)
    selection_df.insert(11, "TUNED_C", tuned_C)
    selection_df.to_csv(experiment_dir / "dp_gd_lr_selection_metrics_all_learning_rates.csv", index=False)

    save_all_step_plots(histories, experiment_dir)
    save_metric_vs_lr_plots(selection_df, experiment_dir)

    best_score = float(selection_df["selection_score"].max())
    tied_lrs = sorted(
        selection_df.loc[
            selection_df["selection_score"] >= best_score - SELECTION_TIE_TOL,
            "learning_rate",
        ].astype(float).tolist()
    )
    best_lr = tied_lrs[0]
    best_selection_row = selection_df.loc[selection_df["learning_rate"].astype(float) == best_lr].iloc[0].to_dict()
    with open(experiment_dir / "best_lr_selection_summary.json", "w") as f:
        json.dump(best_selection_row, f, indent=2)

    print("\nBest LR used for this seed/experiment:")
    print(best_lr)
    print("Best selection score:", best_score)

    best_selection_model = models[best_lr]
    _, y_val_pred, y_val_proba = best_selection_model.metrics(X_val, y_val, prefix="val")
    save_confusion_matrix_plot(y_val, y_val_pred, experiment_dir, "validation_selected_model")
    save_roc_pr_plots(y_val, y_val_proba, experiment_dir, "validation_selected_model")
    save_calibration_plot(y_val, y_val_proba, experiment_dir, "validation_selected_model")
    save_probability_histogram(y_val, y_val_proba, experiment_dir, "validation_selected_model")

    # Final refit on train + validation using best learning rate.
    final_preprocessor, X_train_val, _, X_test_final = fit_transform_preprocessor(
        X_train_df=X_train_val_df,
        X_val_df=None,
        X_test_df=X_test_df,
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
    )

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
        "seed": seed,
        "seed_role": seed_role,
        "phase": "final_refit_train_validation_model",
        "lr_selection_mode": lr_selection_mode,
        "lr_source_seed": lr_source_seed,
        "lr_source_experiment_key": lr_source_experiment_key,
        "best_lr": best_lr,
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": final_achieved_epsilon,
        "delta": final_delta,
        "noise_multiplier": final_noise_multiplier,
        "clip_norm": CLIP_NORM,
        "iterations": ITERATIONS,
        "sampling": final_sampling,
        "n_train_for_accounting": n_train_val,
        "tuned_C": tuned_C,
        "full_batch": True,
        "numba_jit_requested": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
    }
    with open(experiment_dir / "dp_accounting_final_refit_train_validation_model.json", "w") as f:
        json.dump(final_dp_summary, f, indent=2)
    print("\nFinal-refit DP accounting summary:")
    print(json.dumps(final_dp_summary, indent=2))

    final_model, final_history, train_val_metrics, final_test_metrics, y_test_pred_final, y_test_proba_final = train_final_refit_model(
        best_lr=best_lr,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test_final,
        y_test=y_test,
        noise_multiplier=final_noise_multiplier,
        seed=seed,
        tuned_C=tuned_C,
    )
    final_history.to_csv(experiment_dir / "final_refit_trainval_step_history.csv", index=False)

    save_step_metric_overlay(
        {best_lr: final_history},
        metric_col="train_accuracy",
        ylabel="Train+validation accuracy",
        title="Final refit train+validation accuracy vs step",
        out_path=experiment_dir / "final_refit_trainval_accuracy_vs_step.png",
    )
    save_step_metric_overlay(
        {best_lr: final_history},
        metric_col="train_log_loss",
        ylabel="Train+validation log loss",
        title="Final refit train+validation log loss vs step",
        out_path=experiment_dir / "final_refit_trainval_log_loss_vs_step.png",
    )
    save_step_metric_overlay(
        {best_lr: final_history},
        metric_col="test_accuracy",
        ylabel="Test accuracy",
        title="Final refit test accuracy vs step",
        out_path=experiment_dir / "final_refit_test_accuracy_vs_step.png",
    )
    save_step_metric_overlay(
        {best_lr: final_history},
        metric_col="test_log_loss",
        ylabel="Test log loss",
        title="Final refit test log loss vs step",
        out_path=experiment_dir / "final_refit_test_log_loss_vs_step.png",
    )

    save_confusion_matrix_plot(y_test, y_test_pred_final, experiment_dir, "test_final_model")
    save_roc_pr_plots(y_test, y_test_proba_final, experiment_dir, "test_final_model")
    save_calibration_plot(y_test, y_test_proba_final, experiment_dir, "test_final_model")
    save_probability_histogram(y_test, y_test_proba_final, experiment_dir, "test_final_model")
    save_coefficients(final_preprocessor, final_model, experiment_dir, top_k=20)

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
    (experiment_dir / "validation_selected_model_classification_report.txt").write_text(validation_report)
    (experiment_dir / "test_final_model_classification_report.txt").write_text(test_report)

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

    subgroup_eval_features_present = [c for c in SUBGROUP_EVAL_FEATURES if c in global_data.df_model_raw.columns]
    df_subgroup_test = global_data.df_model_raw.iloc[test_idx][subgroup_eval_features_present].reset_index(drop=True)
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
        target_epsilon=TARGET_EPSILON,
        achieved_epsilon=final_achieved_epsilon,
        delta=final_delta,
        noise_multiplier=final_noise_multiplier,
    )
    if test_subgroup_metrics is not None and not test_subgroup_metrics.empty:
        test_subgroup_metrics.insert(0, "seed", seed)
        test_subgroup_metrics.insert(1, "seed_role", seed_role)
        test_subgroup_metrics.insert(2, "lr_selection_mode", lr_selection_mode)
        test_subgroup_metrics.insert(3, "lr_source_seed", lr_source_seed)
        test_subgroup_metrics.insert(4, "lr_source_experiment_key", lr_source_experiment_key)

    summary_rows = [
        {
            "seed": seed,
            "seed_role": seed_role,
            "lr_selection_mode": lr_selection_mode,
            "lr_source_seed": lr_source_seed,
            "lr_source_experiment_key": lr_source_experiment_key,
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "validation_selected_model",
            "model_training_data": "train",
            "best_lr": best_lr,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": selection_achieved_epsilon,
            "delta": selection_delta,
            "noise_multiplier": selection_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm": CLIP_NORM,
            "iterations": ITERATIONS,
            "n_fc_features": len(fc_cols),
            "n_selected_phenotype_raw_columns": len(selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": best_selection_row["val_accuracy"],
            "balanced_accuracy": best_selection_row["val_balanced_accuracy"],
            "log_loss": best_selection_row["val_log_loss"],
            "roc_auc": best_selection_row["val_roc_auc"],
            "average_precision": best_selection_row["val_average_precision"],
        },
        {
            "seed": seed,
            "seed_role": seed_role,
            "lr_selection_mode": lr_selection_mode,
            "lr_source_seed": lr_source_seed,
            "lr_source_experiment_key": lr_source_experiment_key,
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "train_validation_final_model",
            "model_training_data": "train+validation",
            "best_lr": best_lr,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": final_achieved_epsilon,
            "delta": final_delta,
            "noise_multiplier": final_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm": CLIP_NORM,
            "iterations": ITERATIONS,
            "n_fc_features": len(fc_cols),
            "n_selected_phenotype_raw_columns": len(selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": train_val_metrics["trainval_accuracy"],
            "balanced_accuracy": train_val_metrics["trainval_balanced_accuracy"],
            "log_loss": train_val_metrics["trainval_log_loss"],
            "roc_auc": train_val_metrics["trainval_roc_auc"],
            "average_precision": train_val_metrics["trainval_average_precision"],
        },
        {
            "seed": seed,
            "seed_role": seed_role,
            "lr_selection_mode": lr_selection_mode,
            "lr_source_seed": lr_source_seed,
            "lr_source_experiment_key": lr_source_experiment_key,
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "split": "test_final_model",
            "model_training_data": "train+validation",
            "best_lr": best_lr,
            "target_epsilon": TARGET_EPSILON,
            "achieved_epsilon": final_achieved_epsilon,
            "delta": final_delta,
            "noise_multiplier": final_noise_multiplier,
            "TUNED_C": tuned_C,
            "clip_norm": CLIP_NORM,
            "iterations": ITERATIONS,
            "n_fc_features": len(fc_cols),
            "n_selected_phenotype_raw_columns": len(selected_phenotype_cols),
            "selected_feature_group_keys": json.dumps(selected_group_keys),
            "accuracy": final_test_metrics["test_accuracy"],
            "balanced_accuracy": final_test_metrics["test_balanced_accuracy"],
            "log_loss": final_test_metrics["test_log_loss"],
            "roc_auc": final_test_metrics["test_roc_auc"],
            "average_precision": final_test_metrics["test_average_precision"],
        },
    ]

    metrics_summary_df = pd.DataFrame(summary_rows)
    metrics_summary_df.to_csv(experiment_dir / "dp_gd_final_metrics_summary.csv", index=False)

    run_feature_summary = {
        "seed": seed,
        "seed_role": seed_role,
        "lr_selection_mode": lr_selection_mode,
        "lr_source_seed": lr_source_seed,
        "lr_source_experiment_key": lr_source_experiment_key,
        "experiment_key": experiment_key,
        "experiment_name": experiment_name,
        "selected_feature_group_keys": selected_group_keys,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "selected_phenotype_columns_present_in_dataframe": selected_phenotype_cols,
        "phenotype_domain_scaled_numeric_columns": phenotype_domain_scaled_cols,
        "phenotype_other_numeric_columns": phenotype_other_numeric_cols,
        "phenotype_categorical_columns": phenotype_categorical_cols,
        "fc_feature_count": len(fc_cols),
        "requested_json_columns_absent_from_dataframe": missing_cols,
        "n_total_raw_columns_before_encoding": len(selected_phenotype_cols) + len(fc_cols),
        "subgroup_eval_features_requested": SUBGROUP_EVAL_FEATURES,
        "subgroup_eval_features_present": subgroup_eval_features_present,
        "sensible_numeric_bins": SENSIBLE_NUMERIC_BINS,
        "split_strategy": global_data.split_name,
        "fixed_split_created_with_seed": LR_SEED,
        "split_reused_for_all_seeds": True,
        "preprocess_dump_used": bool(global_data.preprocess_dump_used),
        "preprocess_dump_dir": str(global_data.preprocess_dump_dir) if global_data.preprocess_dump_dir is not None else None,
        "site_aware_splitting_used": False,
        "best_lr_used": best_lr,
        "TUNED_C": tuned_C,
        "l2_reg_handling": L2_REG_HANDLING,
        "use_numba_jit": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
        "parallelize_learning_rates": PARALLELIZE_LEARNING_RATES,
        "endpoint_expansion_factor": ENDPOINT_EXPANSION_FACTOR,
        "missing_numeric_policy": "replace NaN/inf with 0 after preprocessing",
        "missing_categorical_policy": "missing/unseen categories mapped to all-zero one-hot block",
    }
    with open(experiment_dir / "run_feature_summary.json", "w") as f:
        json.dump(run_feature_summary, f, indent=2)

    print("\nSaved experiment outputs in:")
    print(experiment_dir)
    return metrics_summary_df, test_subgroup_metrics, float(best_lr)


# =========================================================
# Main
# =========================================================
def get_experiment_by_key(experiment_key: str) -> dict:
    matches = [exp for exp in EXPERIMENTS if exp["experiment_key"] == experiment_key]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one experiment with key={experiment_key!r}, found {len(matches)}."
        )
    return matches[0]


def compute_mean_std_sem(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Compute mean, sample std, and standard error across seeds."""
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n_seeds"] = int(group[seed_col].nunique()) if seed_col in group.columns else int(len(group))

        for metric_col in metric_cols:
            if metric_col not in group.columns:
                continue
            values = pd.to_numeric(group[metric_col], errors="coerce").dropna()
            row[f"{metric_col}_n"] = int(len(values))
            row[f"{metric_col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric_col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric_col}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_seed_combined_outputs(
    *,
    output_dir: Path,
    seed: int,
    global_data: PreparedGlobalData,
    all_metrics: list[pd.DataFrame],
    all_subgroup_metrics: list[pd.DataFrame],
    global_best_lr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save the same combined outputs as the original single-seed main, under one seed folder."""
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    else:
        all_metrics_df = pd.DataFrame()

    all_metrics_df.to_csv(output_dir / "all_modes_dp_gd_final_metrics_summary.csv", index=False)

    if not all_metrics_df.empty:
        final_validation_test_metrics = all_metrics_df.loc[
            all_metrics_df["split"].isin(["validation_selected_model", "test_final_model"])
        ].reset_index(drop=True)
    else:
        final_validation_test_metrics = pd.DataFrame()

    final_validation_test_metrics.to_csv(
        output_dir / "all_modes_dp_gd_final_validation_test_metrics.csv",
        index=False,
    )

    test_cols = [
        "seed",
        "seed_role",
        "lr_selection_mode",
        "lr_source_seed",
        "lr_source_experiment_key",
        "experiment_key",
        "experiment_name",
        "best_lr",
        "target_epsilon",
        "achieved_epsilon",
        "delta",
        "noise_multiplier",
        "TUNED_C",
        "clip_norm",
        "iterations",
        "n_fc_features",
        "n_selected_phenotype_raw_columns",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
    ]

    if not all_metrics_df.empty:
        available_test_cols = [c for c in test_cols if c in all_metrics_df.columns]
        test_comparison = all_metrics_df.loc[
            all_metrics_df["split"] == "test_final_model",
            available_test_cols,
        ].reset_index(drop=True)
    else:
        test_comparison = pd.DataFrame()

    test_comparison.to_csv(output_dir / "all_modes_dp_gd_test_metrics_comparison.csv", index=False)

    if all_subgroup_metrics:
        all_subgroup_metrics_df = pd.concat(all_subgroup_metrics, axis=0, ignore_index=True)
    else:
        all_subgroup_metrics_df = pd.DataFrame()

    all_subgroup_metrics_df.to_csv(
        output_dir / "all_modes_dp_gd_test_subgroup_metrics_summary.csv",
        index=False,
    )

    run_summary = {
        "seed": int(seed),
        "seed_list": SEED_LIST,
        "lr_seed": LR_SEED,
        "global_best_lr_from_lr_seed": global_best_lr,
        "target_col": TARGET_COL,
        "csv_path": str(CSV_PATH),
        "feature_json_path_used": str(global_data.feature_json_path),
        "output_dir": str(output_dir),
        "experiments": EXPERIMENTS,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "split_strategy": global_data.split_name,
        "fixed_split_created_with_seed": LR_SEED,
        "split_reused_for_all_seeds": True,
        "preprocess_dump_used": bool(global_data.preprocess_dump_used),
        "preprocess_dump_dir": str(global_data.preprocess_dump_dir) if global_data.preprocess_dump_dir is not None else None,
        "site_aware_splitting_used": False,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "target_epsilon": TARGET_EPSILON,
        "delta_power": DELTA_POWER,
        "clip_norm": CLIP_NORM,
        "iterations": ITERATIONS,
        "full_batch": True,
        "TUNED_C_WithOut_AnatFunc": TUNED_C_WithOut_AnatFunc,
        "TUNED_FIT_INTERCEPT": TUNED_FIT_INTERCEPT,
        "L2_REG_HANDLING": L2_REG_HANDLING,
        "learning_rates_initial": LEARNING_RATES_INITIAL,
        "max_endpoint_expansions": MAX_ENDPOINT_EXPANSIONS,
        "endpoint_expansion_factor": ENDPOINT_EXPANSION_FACTOR,
        "selection_metric": SELECTION_METRIC,
        "use_numba_jit": USE_NUMBA_JIT,
        "numba_available": NUMBA_AVAILABLE,
        "parallelize_learning_rates": PARALLELIZE_LEARNING_RATES,
        "lr_parallel_n_jobs": LR_PARALLEL_N_JOBS,
        "lr_parallel_backend": LR_PARALLEL_BACKEND,
        "sensible_numeric_bins": SENSIBLE_NUMERIC_BINS,
        "n_total_after_valid_mask": int(len(global_data.y)),
        "n_train": int(len(global_data.train_idx)),
        "n_validation": int(len(global_data.val_idx)),
        "n_test": int(len(global_data.test_idx)),
        "class_counts_total": {
            str(k): int(v) for k, v in pd.Series(global_data.y).value_counts().sort_index().to_dict().items()
        },
    }
    with open(output_dir / "run_summary_all_modes_dp_gd.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\n" + "=" * 120)
    print(f"Final DP-GD test metrics comparison for seed={seed}")
    print("=" * 120)
    if not test_comparison.empty:
        print(test_comparison.to_string(index=False))
    else:
        print("No test comparison rows available.")
    print("\nSaved seed outputs in:")
    print(output_dir)

    return all_metrics_df, test_comparison, all_subgroup_metrics_df


def save_cross_seed_outputs(
    *,
    epsilon_output_dir: Path,
    all_seed_metrics: list[pd.DataFrame],
    all_seed_subgroup_metrics: list[pd.DataFrame],
    global_best_lr: float,
    lr_tuning_summary: dict,
) -> None:
    """Save raw cross-seed outputs plus mean/std/SEM summaries under e{epsilon}."""
    epsilon_output_dir.mkdir(parents=True, exist_ok=True)

    if all_seed_metrics:
        all_seed_metrics_df = pd.concat(all_seed_metrics, axis=0, ignore_index=True)
    else:
        all_seed_metrics_df = pd.DataFrame()

    all_seed_metrics_df.to_csv(
        epsilon_output_dir / "all_seeds_dp_gd_final_metrics_summary.csv",
        index=False,
    )

    test_metrics_df = all_seed_metrics_df.loc[
        all_seed_metrics_df["split"] == "test_final_model"
    ].reset_index(drop=True) if not all_seed_metrics_df.empty else pd.DataFrame()

    test_metrics_df.to_csv(
        epsilon_output_dir / "all_seeds_dp_gd_test_metrics_comparison.csv",
        index=False,
    )

    test_metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
    ]
    test_mean_sem_df = compute_mean_std_sem(
        test_metrics_df,
        group_cols=["experiment_key", "experiment_name", "target_epsilon", "TUNED_C", "best_lr"],
        metric_cols=test_metric_cols,
    )
    test_mean_sem_df.to_csv(
        epsilon_output_dir / "all_seeds_dp_gd_test_metrics_mean_sem.csv",
        index=False,
    )

    if all_seed_subgroup_metrics:
        all_seed_subgroup_metrics_df = pd.concat(all_seed_subgroup_metrics, axis=0, ignore_index=True)
    else:
        all_seed_subgroup_metrics_df = pd.DataFrame()

    all_seed_subgroup_metrics_df.to_csv(
        epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_summary.csv",
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
    subgroup_mean_sem_df = compute_mean_std_sem(
        all_seed_subgroup_metrics_df,
        group_cols=["experiment_key", "experiment_name", "split", "feature", "group", "target_epsilon", "best_lr"],
        metric_cols=subgroup_metric_cols,
    )
    subgroup_mean_sem_df.to_csv(
        epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_mean_sem.csv",
        index=False,
    )

    with open(epsilon_output_dir / "global_lr_selection_summary.json", "w") as f:
        json.dump(lr_tuning_summary, f, indent=2)

    cross_seed_run_summary = {
        "seed_list": SEED_LIST,
        "lr_seed": LR_SEED,
        "lr_tuning_experiment_key": LR_TUNING_EXPERIMENT_KEY,
        "global_best_lr": global_best_lr,
        "target_epsilon": TARGET_EPSILON,
        "output_dir": str(OUTPUT_DIR),
        "epsilon_output_dir": str(epsilon_output_dir),
        "fixed_split_created_with_seed": LR_SEED,
        "split_reused_for_all_seeds": True,
        "use_preprocess_dump": USE_PREPROCESS_DUMP,
        "preprocess_dump_dir": str(PREPROCESS_DUMP_DIR) if USE_PREPROCESS_DUMP else None,
        "test_metrics_raw_csv": str(epsilon_output_dir / "all_seeds_dp_gd_test_metrics_comparison.csv"),
        "test_metrics_mean_sem_csv": str(epsilon_output_dir / "all_seeds_dp_gd_test_metrics_mean_sem.csv"),
        "subgroup_metrics_raw_csv": str(epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_summary.csv"),
        "subgroup_metrics_mean_sem_csv": str(epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_mean_sem.csv"),
        "use_numba_jit": USE_NUMBA_JIT,
        "parallelize_learning_rates": PARALLELIZE_LEARNING_RATES,
    }
    with open(epsilon_output_dir / "run_summary_all_seeds_dp_gd.json", "w") as f:
        json.dump(cross_seed_run_summary, f, indent=2)

    print("\n" + "=" * 120)
    print("Cross-seed test metrics mean/std/SEM")
    print("=" * 120)
    if not test_mean_sem_df.empty:
        print(test_mean_sem_df.to_string(index=False))
    else:
        print("No cross-seed test summary available.")

    print("\nSaved cross-seed outputs in:")
    print(epsilon_output_dir)


def main() -> None:
    if USE_NUMBA_JIT:
        print(f"Numba requested: {USE_NUMBA_JIT}; numba available: {NUMBA_AVAILABLE}")
        if not NUMBA_AVAILABLE:
            print("Proceeding with Python/NumPy fallback because numba is unavailable.")
    if PARALLELIZE_LEARNING_RATES:
        print(f"Parallel LR trials requested: joblib available={JOBLIB_AVAILABLE}")

    if len(SEED_LIST) == 0:
        raise ValueError("SEED_LIST must contain at least one seed.")

    lr_tuning_exp = get_experiment_by_key(LR_TUNING_EXPERIMENT_KEY)

    print("\n" + "=" * 120)
    print("GLOBAL LEARNING-RATE TUNING")
    print("=" * 120)
    print(f"LR_SEED: {LR_SEED}")
    print(f"LR tuning experiment: {LR_TUNING_EXPERIMENT_KEY}")
    print(f"SEED_LIST for final reporting: {SEED_LIST}")

    # ------------------------------------------------------------------
    # 1. Tune learning rate exactly once using LR_SEED.
    # ------------------------------------------------------------------
    lr_seed_output_dir = get_seed_output_dir(LR_SEED)
    lr_global_data = prepare_global_data(seed=LR_SEED, output_dir=lr_seed_output_dir)
    save_fixed_split_artifacts(lr_global_data)

    lr_seed_metrics_for_aggregation = []
    lr_seed_subgroups_for_aggregation = []

    lr_metrics_df, lr_subgroup_df, global_best_lr = run_single_experiment(
        global_data=lr_global_data,
        experiment_key=lr_tuning_exp["experiment_key"],
        experiment_name=lr_tuning_exp["experiment_name"],
        selected_group_keys=lr_tuning_exp["feature_group_keys"],
        tuned_C=float(lr_tuning_exp["tuned_C"]),
        tune_learning_rate=True,
        fixed_best_lr=None,
        seed_role="lr_tuning_seed" if LR_SEED not in SEED_LIST else "lr_tuning_seed_and_eval_seed",
        lr_source_seed=LR_SEED,
        lr_source_experiment_key=LR_TUNING_EXPERIMENT_KEY,
    )

    lr_tuning_summary = {
        "lr_seed": LR_SEED,
        "seed_list": SEED_LIST,
        "lr_tuning_experiment_key": LR_TUNING_EXPERIMENT_KEY,
        "lr_tuning_experiment_name": lr_tuning_exp["experiment_name"],
        "global_best_lr": global_best_lr,
        "target_epsilon": TARGET_EPSILON,
        "learning_rates_initial": LEARNING_RATES_INITIAL,
        "max_endpoint_expansions": MAX_ENDPOINT_EXPANSIONS,
        "endpoint_expansion_factor": ENDPOINT_EXPANSION_FACTOR,
        "selection_metric": SELECTION_METRIC,
        "selection_tie_tol": SELECTION_TIE_TOL,
        "fixed_split_created_with_seed": LR_SEED,
        "lr_seed_in_seed_list": bool(LR_SEED in SEED_LIST),
        "use_preprocess_dump": USE_PREPROCESS_DUMP,
        "preprocess_dump_dir": str(PREPROCESS_DUMP_DIR) if USE_PREPROCESS_DUMP else None,
        "policy": (
            "The IID train/validation/test split is created or loaded exactly once using LR_SEED. "
            "Learning rate is tuned exactly once on LR_SEED using that fixed validation split. "
            "The resulting global_best_lr and the exact same fixed split are reused for every seed in SEED_LIST."
        ),
    }
    with open(EPSILON_OUTPUT_DIR / "global_lr_selection_summary.json", "w") as f:
        json.dump(lr_tuning_summary, f, indent=2)

    # If LR_SEED is itself part of SEED_LIST, its tuned run should count as that seed's
    # evaluation for the LR-tuning experiment. Other experiments, if any, are run below
    # using the fixed global LR.
    if LR_SEED in SEED_LIST:
        lr_seed_metrics_for_aggregation.append(lr_metrics_df)
        if lr_subgroup_df is not None and not lr_subgroup_df.empty:
            lr_seed_subgroups_for_aggregation.append(lr_subgroup_df)

        for exp in EXPERIMENTS:
            if exp["experiment_key"] == LR_TUNING_EXPERIMENT_KEY:
                continue
            m_df, sg_df, _ = run_single_experiment(
                global_data=lr_global_data,
                experiment_key=exp["experiment_key"],
                experiment_name=exp["experiment_name"],
                selected_group_keys=exp["feature_group_keys"],
                tuned_C=float(exp["tuned_C"]),
                tune_learning_rate=False,
                fixed_best_lr=global_best_lr,
                seed_role="eval_seed_fixed_lr",
                lr_source_seed=LR_SEED,
                lr_source_experiment_key=LR_TUNING_EXPERIMENT_KEY,
            )
            lr_seed_metrics_for_aggregation.append(m_df)
            if sg_df is not None and not sg_df.empty:
                lr_seed_subgroups_for_aggregation.append(sg_df)

    # Save LR_SEED combined outputs. If LR_SEED is not in SEED_LIST this contains only
    # the tuning experiment; it is not included in cross-seed mean/SEM.
    save_seed_combined_outputs(
        output_dir=lr_seed_output_dir,
        seed=LR_SEED,
        global_data=lr_global_data,
        all_metrics=lr_seed_metrics_for_aggregation if LR_SEED in SEED_LIST else [lr_metrics_df],
        all_subgroup_metrics=lr_seed_subgroups_for_aggregation if LR_SEED in SEED_LIST else ([lr_subgroup_df] if lr_subgroup_df is not None and not lr_subgroup_df.empty else []),
        global_best_lr=global_best_lr,
    )

    # ------------------------------------------------------------------
    # 2. Run all requested evaluation seeds using fixed global_best_lr.
    # ------------------------------------------------------------------
    all_seed_metrics = []
    all_seed_subgroup_metrics = []

    if LR_SEED in SEED_LIST:
        all_seed_metrics.extend(lr_seed_metrics_for_aggregation)
        all_seed_subgroup_metrics.extend(lr_seed_subgroups_for_aggregation)

    for seed in SEED_LIST:
        seed = int(seed)
        if seed == LR_SEED:
            continue

        print("\n" + "=" * 120)
        print(f"RUNNING FIXED-LR EVALUATION SEED: {seed}")
        print("=" * 120)

        seed_output_dir = get_seed_output_dir(seed)
        global_data = clone_prepared_global_data_for_seed(
            lr_global_data,
            seed=seed,
            output_dir=seed_output_dir,
        )

        seed_metrics = []
        seed_subgroups = []

        for exp in EXPERIMENTS:
            metrics_df, subgroup_df, _ = run_single_experiment(
                global_data=global_data,
                experiment_key=exp["experiment_key"],
                experiment_name=exp["experiment_name"],
                selected_group_keys=exp["feature_group_keys"],
                tuned_C=float(exp["tuned_C"]),
                tune_learning_rate=False,
                fixed_best_lr=global_best_lr,
                seed_role="eval_seed_fixed_lr",
                lr_source_seed=LR_SEED,
                lr_source_experiment_key=LR_TUNING_EXPERIMENT_KEY,
            )
            seed_metrics.append(metrics_df)
            all_seed_metrics.append(metrics_df)

            if subgroup_df is not None and not subgroup_df.empty:
                seed_subgroups.append(subgroup_df)
                all_seed_subgroup_metrics.append(subgroup_df)

        save_seed_combined_outputs(
            output_dir=seed_output_dir,
            seed=seed,
            global_data=global_data,
            all_metrics=seed_metrics,
            all_subgroup_metrics=seed_subgroups,
            global_best_lr=global_best_lr,
        )

    # ------------------------------------------------------------------
    # 3. Save cross-seed raw outputs and mean/std/SEM under e{epsilon}.
    # ------------------------------------------------------------------
    save_cross_seed_outputs(
        epsilon_output_dir=EPSILON_OUTPUT_DIR,
        all_seed_metrics=all_seed_metrics,
        all_seed_subgroup_metrics=all_seed_subgroup_metrics,
        global_best_lr=global_best_lr,
        lr_tuning_summary=lr_tuning_summary,
    )


if __name__ == "__main__":
    main()