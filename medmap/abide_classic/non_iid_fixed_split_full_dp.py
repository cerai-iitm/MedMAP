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

# NON-IID Metrics for Full DP Multiseeded
#!/usr/bin/env python3
"""
Non-IID SITE-aware fixed-split full-batch DP-GD logistic regression for ABIDE.

Core policy preserved from the reference script:
- Same data loading / cleaning / feature selection / scaling / preprocessing.
- Same manual full-batch DP-GD update: per-example gradients -> clipping -> sum
  -> Gaussian noise -> average -> optional L2 -> GD update.
- Same Numba JIT path with Python/NumPy fallback.
- Same optional joblib parallelization for learning-rate trials.
- LR is tuned exactly once on LR_SEED and reused for every seed in SEED_LIST.
- Selected train model is chosen on validation; final model is refit on
  train+validation and evaluated on test.

Required fixed-split policy:
- LR_SEED does not need to be present in SEED_LIST.
- The train/validation/test split is created exactly once using LR_SEED.
- The exact same train_idx, val_idx, and test_idx are reused for every seed in
  SEED_LIST.
- SITE_AWARE_SPLIT=True and USE_SITE_ID_AS_FEATURE=False.
- SITE_ID is used for grouping/splitting and subgroup reporting only, not as a
  model input feature.

Optional prepared-data shortcut:
- If USE_PREPROCESS_DUMP=True, the script reads the fixed split CSVs dumped by
  the non-private fixed-split pipeline instead of rebuilding the raw ABIDE +
  X_conn preparation stage.
"""

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as opt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
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
# Configuration
# =========================================================
class SeedConfig:
    SEED_LIST = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    LR_SEED = 21
    LR_TUNING_EXPERIMENT_KEY = "full_dp_without_anat_func"


class ExperimentConfig:
    ITERATIONS = 3000
    TUNED_C_WithOut_AnatFunc = 0.3
    # TUNED_C_With_AnatFunc = 0.1
    TUNED_FIT_INTERCEPT = True
    TARGET_EPSILON = 8.0
    LEARNING_RATES_INITIAL = [
        0.08,
        0.16, 0.32, 0.64, 1.28, 2.56,
    ]
    MAX_ENDPOINT_EXPANSIONS = 0
    ENDPOINT_EXPANSION_FACTOR = 2.0
    TARGET_COL = "DX_GROUP"
    CSV_PATH = _abide_data.phenotypic_csv_path(_ABIDE_DATA_ROOT)
    FEATURE_JSON_CANDIDATES = _abide_data.feature_json_candidates(_ABIDE_DATA_ROOT)
    X_CONN_PATH: Optional[Path] = None
    USE_SITE_ID_AS_FEATURE = False
    SITE_AWARE_SPLIT = True
    SITE_AWARE_N_SPLITS = 10
    FC_IS_CORRELATION = True
    VALIDATION_SIZE = 0.10
    TEST_SIZE = 0.20
    PHENOTYPE_OTHER_NUMERIC_MODE = "minmax_minus1_1_zero_fill"


class DPConfig:
    DELTA_POWER = -1.1
    NOISE_MULTIPLIER: Optional[float] = None
    CLIP_NORM = 1.0
    FULL_BATCH = True
    L2_REG_HANDLING = "add_after_noise"  # {"add_after_noise", "inside_per_example"}
    SELECTION_METRIC = "accuracy"
    SELECTION_TIE_TOL = 1e-12
    RECORD_EVERY_N_STEPS = 1
    SHOW_PLOTS = False


class ExecutionConfig:
    USE_NUMBA_JIT = True
    PARALLELIZE_LEARNING_RATES = True
    LR_PARALLEL_N_JOBS = min(64, os.cpu_count() or 1)
    LR_PARALLEL_BACKEND = "loky"

USE_PREPROCESS_DUMP = False
PRINT_PREPROCESS_DUMP_COLUMN_NAMES = False

SEED_LIST = [int(s) for s in SeedConfig.SEED_LIST]
LR_SEED = int(SeedConfig.LR_SEED)

# Optional environment overrides (the defaults above reproduce the paper runs).
SEED_LIST = _abide_data.env_override_int_list("ABIDE_SEEDS", SEED_LIST)
LR_SEED = _abide_data.env_override_int("ABIDE_TUNING_SEED", LR_SEED)
LR_TUNING_EXPERIMENT_KEY = str(SeedConfig.LR_TUNING_EXPERIMENT_KEY)
ITERATIONS = ExperimentConfig.ITERATIONS
TUNED_C_WithOut_AnatFunc = ExperimentConfig.TUNED_C_WithOut_AnatFunc
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
SITE_AWARE_SPLIT = ExperimentConfig.SITE_AWARE_SPLIT
SITE_AWARE_N_SPLITS = ExperimentConfig.SITE_AWARE_N_SPLITS
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

# New fixed-split output root requested by the user.
OUTPUT_DIR = Path(f"{_O}/non_iid_fixed_split/full_dp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON_OUTPUT_DIR = OUTPUT_DIR / f"e{int(TARGET_EPSILON)}"
EPSILON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Optional prepared-data shortcut.
# -------------------------------------------------------------------------
# When True, the script reads these fixed split CSVs produced by the non-private
# fixed-split pipeline and skips raw ABIDE CSV/X_conn loading, cleaning,
# valid-label filtering, domain scaling, and SITE-aware split construction.
# The downstream DP-GD feature preprocessing inside run_single_experiment is
# intentionally kept unchanged.
PREPROCESS_DUMP_DIR = Path(f"{_O}/non_iid_fixed_split/non_priv")
PREPROCESS_DUMP_TRAIN_CSV = PREPROCESS_DUMP_DIR / "train_fixed.csv"
PREPROCESS_DUMP_VAL_CSV = PREPROCESS_DUMP_DIR / "val_fixed.csv"
PREPROCESS_DUMP_TEST_CSV = PREPROCESS_DUMP_DIR / "test_fixed.csv"
PREPROCESS_DUMP_TRAIN_VAL_CSV = PREPROCESS_DUMP_DIR / "train_val_fixed.csv"


def get_seed_output_dir(seed: int) -> Path:
    out = EPSILON_OUTPUT_DIR / f"seed{int(seed)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


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
]

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
# JSON / serialization helpers
# =========================================================
def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_clean(data), f, indent=2)


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
    groups_all: Optional[pd.Series]
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
# Preprocessing transformers
# =========================================================
class ZeroFillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.nan_to_num(
            np.asarray(X, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class StandardizeThenZeroFill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        self.scale_ = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(
            (X - self.mean_) / self.scale_,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class MinMaxMinusOneOneThenZeroFill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        self.data_min_ = np.where(np.isfinite(data_min), data_min, 0.0)
        self.data_max_ = np.where(np.isfinite(data_max), data_max, 0.0)
        self.range_ = self.data_max_ - self.data_min_
        self.valid_range_ = np.isfinite(self.range_) & (self.range_ > 0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X, dtype=np.float64)
        valid_cols = self.valid_range_
        if np.any(valid_cols):
            out[:, valid_cols] = (
                2.0
                * (X[:, valid_cols] - self.data_min_[valid_cols])
                / self.range_[valid_cols]
                - 1.0
            )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class MissingAsAllZeroOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = [str(c) for c in X_df.columns]
        self.categories_ = []
        for j in range(X_df.shape[1]):
            observed = X_df.iloc[:, j][~pd.isna(X_df.iloc[:, j])]
            cats = sorted(pd.Index(observed.astype(str)).unique().tolist())
            self.categories_.append(cats)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        n = len(X_df)
        blocks = []
        for j, cats in enumerate(self.categories_):
            block = np.zeros((n, len(cats)), dtype=np.float64)
            if cats:
                s = X_df.iloc[:, j]
                valid = (~pd.isna(s)).to_numpy()
                vals = s.astype(str).to_numpy()
                for k, cat in enumerate(cats):
                    block[valid & (vals == cat), k] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

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
# Loading / feature helpers
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
            "X_conn is not available. Set X_CONN_PATH to a .npy/.npz file "
            "or define X_conn in the session."
        )
    if x_conn_path.suffix == ".npy":
        arr = np.load(x_conn_path)
    elif x_conn_path.suffix == ".npz":
        payload = np.load(x_conn_path)
        arr = payload["X_conn"] if "X_conn" in payload else payload[list(payload.keys())[0]]
    else:
        raise ValueError("X_CONN_PATH must point to a .npy or .npz file.")
    return np.asarray(arr, dtype=np.float64)


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
    return df.replace([np.inf, -np.inf], np.nan)


def get_site_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    if "SITE_ID" not in df.columns:
        return None
    return (
        df["SITE_ID"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-9999": pd.NA})
        .fillna("__MISSING_SITE__")
        .astype(str)
        .reset_index(drop=True)
    )


def resolve_feature_json_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find recommended_feature_lists.json.")


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
    return {"RAVENS": "RAVEN", "WISC4": "WISC_IV", "KBITS_2": "KBIT_2"}.get(s, s)


def build_feature_lists_from_selected_keys(
    df: pd.DataFrame,
    feature_groups: dict,
    selected_group_keys: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    selected_group_keys = list(dict.fromkeys(selected_group_keys))
    invalid = sorted(set(selected_group_keys) - ALLOWED_FEATURE_GROUP_KEYS)
    if invalid:
        raise ValueError(f"Invalid feature group keys: {invalid}")

    numeric_keys = {
        "recommended_initial_numeric_or_binary",
        "anat_func_numeric_or_binary",
        "drop_leakage",
    }
    categorical_keys = {
        "recommended_initial_categorical",
        "optional_categorical_with_caution",
    }
    all_selected, numeric_cols, cat_cols = [], [], []

    for key in selected_group_keys:
        for c in feature_groups.get(key, []):
            if c not in all_selected:
                all_selected.append(c)
            if key in numeric_keys and c not in numeric_cols:
                numeric_cols.append(c)
            if key in categorical_keys and c not in cat_cols:
                cat_cols.append(c)

    all_selected = [c for c in all_selected if c != TARGET_COL]
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    missing = [c for c in all_selected if c not in df.columns]
    selected = [c for c in all_selected if c in df.columns]
    numeric = [c for c in numeric_cols if c in df.columns]
    cat = [c for c in cat_cols if c in df.columns]

    if not USE_SITE_ID_AS_FEATURE:
        selected = [c for c in selected if c != "SITE_ID"]
        numeric = [c for c in numeric if c != "SITE_ID"]
        cat = [c for c in cat if c != "SITE_ID"]

    return selected, numeric, cat, missing


def get_all_numeric_candidate_columns(feature_groups: dict) -> list[str]:
    cols = []
    for key in [
        "recommended_initial_numeric_or_binary",
        "anat_func_numeric_or_binary",
        "drop_leakage",
    ]:
        for c in feature_groups.get(key, []):
            if c != TARGET_COL and c not in cols:
                cols.append(c)
    return cols


def scale_to_minus1_plus1(series: pd.Series, lo: float, hi: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if hi < lo:
        raise ValueError(f"Invalid bounds: {(lo, hi)}")
    if hi == lo:
        out.loc[s.notna()] = 0.0
        return out
    mask = s.notna()
    out.loc[mask] = 2.0 * (s.loc[mask] - lo) / (hi - lo) - 1.0
    return out.clip(-1.0, 1.0)


def apply_domain_scaling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, bounds in DOMAIN_FIXED_BOUNDS.items():
        if col in df.columns:
            df[col] = scale_to_minus1_plus1(df[col], *bounds)
    for col, bounds in IQ_GLOBAL_BOUNDS.items():
        if col in df.columns:
            df[col] = scale_to_minus1_plus1(df[col], *bounds)
    return df


def build_combined_dataframe(
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    phenotype_selected_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    _, d = X_conn_model.shape
    fc_cols = [f"fc_{i:05d}" for i in range(d)]
    fc_df = pd.DataFrame(X_conn_model, columns=fc_cols, index=df_model.index)
    X = (
        pd.concat([fc_df, df_model[phenotype_selected_cols].copy()], axis=1)
        if phenotype_selected_cols
        else fc_df.copy()
    )
    print(
        f"Combined dataframe shape: {X.shape} "
        f"(FC: {d} cols, phenotype: {len(phenotype_selected_cols)} cols)"
    )
    return X, fc_cols


def build_other_numeric_transformer(mode: str) -> Pipeline:
    if mode == "standardize_zero_fill":
        return Pipeline([("scale_then_zero_fill", StandardizeThenZeroFill())])
    if mode == "minmax_minus1_1_zero_fill":
        return Pipeline([("scale_then_zero_fill", MinMaxMinusOneOneThenZeroFill())])
    if mode == "zero_fill_only":
        return Pipeline([("zero_fill", ZeroFillTransformer())])
    raise ValueError(f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={mode!r}")


def make_preprocessor(domain_cols, other_num_cols, cat_cols, fc_cols) -> ColumnTransformer:
    transformers = []

    if fc_cols:
        fc_pipe = (
            Pipeline([("zero_fill", ZeroFillTransformer())])
            if FC_IS_CORRELATION
            else Pipeline([("scale_then_zero_fill", StandardizeThenZeroFill())])
        )
        transformers.append(("fc", fc_pipe, fc_cols))

    if domain_cols:
        transformers.append(
            ("phen_domain", Pipeline([("zero_fill", ZeroFillTransformer())]), domain_cols)
        )

    if other_num_cols:
        transformers.append(
            ("phen_other_num", build_other_numeric_transformer(PHENOTYPE_OTHER_NUMERIC_MODE), other_num_cols)
        )

    if cat_cols:
        transformers.append(
            ("phen_cat", Pipeline([("onehot_missing_zero", MissingAsAllZeroOneHotEncoder())]), cat_cols)
        )

    if not transformers:
        raise ValueError("No feature blocks were provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_transform_preprocessor(X_train_df, X_val_df, X_test_df, domain_cols, other_num_cols, cat_cols, fc_cols):
    pre = make_preprocessor(domain_cols, other_num_cols, cat_cols, fc_cols)
    Xtr = np.asarray(pre.fit_transform(X_train_df), dtype=np.float64)
    Xv = None if X_val_df is None else np.asarray(pre.transform(X_val_df), dtype=np.float64)
    Xt = None if X_test_df is None else np.asarray(pre.transform(X_test_df), dtype=np.float64)
    return pre, Xtr, Xv, Xt


# =========================================================
# SITE-aware splitting
# =========================================================
def _class_count_dict(y_split):
    return {
        str(k): int(v)
        for k, v in pd.Series(y_split).value_counts().sort_index().to_dict().items()
    }


def _site_count_dict(groups_split):
    if groups_split is None:
        return {}
    return {
        str(k): int(v)
        for k, v in groups_split.value_counts().sort_index().to_dict().items()
    }


def validate_site_disjointness(groups: pd.Series, train_idx, val_idx, test_idx) -> dict:
    tr = set(groups.iloc[train_idx].astype(str))
    va = set(groups.iloc[val_idx].astype(str))
    te = set(groups.iloc[test_idx].astype(str))
    audit = {
        "train_unique_site_count": len(tr),
        "validation_unique_site_count": len(va),
        "test_unique_site_count": len(te),
        "train_val_site_overlap": sorted(tr & va),
        "train_test_site_overlap": sorted(tr & te),
        "validation_test_site_overlap": sorted(va & te),
    }
    audit["train_val_site_overlap_count"] = len(audit["train_val_site_overlap"])
    audit["train_test_site_overlap_count"] = len(audit["train_test_site_overlap"])
    audit["validation_test_site_overlap_count"] = len(audit["validation_test_site_overlap"])
    if (
        audit["train_val_site_overlap"]
        or audit["train_test_site_overlap"]
        or audit["validation_test_site_overlap"]
    ):
        raise RuntimeError("SITE_ID leakage detected: " + json.dumps(audit, indent=2))
    return audit


def build_split_audit(y, train_idx, val_idx, test_idx, groups=None) -> dict:
    audit = {
        "n_total": int(len(y)),
        "train": {
            "n_samples": int(len(train_idx)),
            "fraction": float(len(train_idx) / len(y)),
            "class_counts": _class_count_dict(y[train_idx]),
        },
        "validation": {
            "n_samples": int(len(val_idx)),
            "fraction": float(len(val_idx) / len(y)),
            "class_counts": _class_count_dict(y[val_idx]),
        },
        "test": {
            "n_samples": int(len(test_idx)),
            "fraction": float(len(test_idx) / len(y)),
            "class_counts": _class_count_dict(y[test_idx]),
        },
    }
    if groups is not None:
        audit["site_disjointness"] = validate_site_disjointness(
            groups,
            train_idx,
            val_idx,
            test_idx,
        )
        audit["train"]["site_counts"] = _site_count_dict(groups.iloc[train_idx])
        audit["validation"]["site_counts"] = _site_count_dict(groups.iloc[val_idx])
        audit["test"]["site_counts"] = _site_count_dict(groups.iloc[test_idx])
    return audit


def _site_aware_split_score(y, train_idx, val_idx, test_idx, train_size, validation_size, test_size) -> float:
    n = len(y)
    overall = float(np.mean(y == 1))
    score = 0.0
    for idx, target in [
        (train_idx, train_size),
        (val_idx, validation_size),
        (test_idx, test_size),
    ]:
        score += 5.0 * abs(len(idx) / n - target)
        if len(idx) == 0:
            score += 1e6
            continue
        score += abs(float(np.mean(y[idx] == 1)) - overall)
        if len(np.unique(y[idx])) < 2:
            score += 1e4
    return float(score)


def make_site_aware_train_val_test_split(y, groups, seed, validation_size, test_size, n_splits):
    if groups is None:
        raise ValueError("SITE-aware split requested but SITE_ID groups are unavailable.")
    if len(groups) != len(y):
        raise ValueError("groups and y length mismatch")
    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique SITE_ID groups; found {groups.nunique()}."
        )

    n_val_folds = max(1, int(round(validation_size * n_splits)))
    n_test_folds = max(1, int(round(test_size * n_splits)))
    if n_val_folds + n_test_folds >= n_splits:
        raise ValueError("SITE_AWARE_N_SPLITS too small for requested split sizes.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        np.asarray(idx, dtype=int)
        for _, idx in splitter.split(np.zeros(len(y)), y, groups=groups)
    ]

    train_size = 1.0 - validation_size - test_size
    best = (math.inf, None)
    fold_ids = list(range(n_splits))

    for val_ids in combinations(fold_ids, n_val_folds):
        remain = [k for k in fold_ids if k not in val_ids]
        for test_ids in combinations(remain, n_test_folds):
            used = set(val_ids) | set(test_ids)
            train_ids = [k for k in fold_ids if k not in used]
            val_idx = np.concatenate([folds[k] for k in val_ids])
            test_idx = np.concatenate([folds[k] for k in test_ids])
            train_idx = np.concatenate([folds[k] for k in train_ids])
            score = _site_aware_split_score(
                y,
                train_idx,
                val_idx,
                test_idx,
                train_size,
                validation_size,
                test_size,
            )
            if score < best[0]:
                best = (
                    score,
                    (
                        np.sort(train_idx),
                        np.sort(val_idx),
                        np.sort(test_idx),
                        tuple(train_ids),
                        tuple(val_ids),
                        tuple(test_ids),
                    ),
                )

    if best[1] is None:
        raise RuntimeError("Could not create SITE-aware split.")

    train_idx, val_idx, test_idx, train_ids, val_ids, test_ids = best[1]
    split_name = (
        "StratifiedGroupKFold SITE_ID-aware train/validation/test split "
        f"(Non-IID fixed split, SITE_ID-disjoint): target train={train_size:.2f}, "
        f"validation={validation_size:.2f}, test={test_size:.2f}; "
        f"n_splits={n_splits}, validation_folds={n_val_folds}, test_folds={n_test_folds}; "
        f"created_once_with_seed={seed}"
    )
    audit = build_split_audit(y, train_idx, val_idx, test_idx, groups)
    audit.update(
        {
            "site_aware_n_splits": int(n_splits),
            "selected_train_fold_ids": list(train_ids),
            "selected_validation_fold_ids": list(val_ids),
            "selected_test_fold_ids": list(test_ids),
            "split_selection_score": float(best[0]),
            "fixed_split_created_with_seed": int(seed),
        }
    )
    return train_idx, val_idx, test_idx, split_name, audit


def make_train_val_test_split(y, seed, validation_size, test_size, groups=None, site_aware=True, n_splits=10):
    if site_aware:
        return make_site_aware_train_val_test_split(
            y,
            groups,
            seed,
            validation_size,
            test_size,
            n_splits,
        )

    n = len(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(np.zeros(n), y))
    val_frac = validation_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    train_rel, val_rel = next(sss2.split(np.zeros(len(train_val_idx)), y[train_val_idx]))
    train_idx = np.sort(train_val_idx[train_rel])
    val_idx = np.sort(train_val_idx[val_rel])
    test_idx = np.sort(test_idx)
    split_name = (
        f"StratifiedShuffleSplit IID fixed split: train={1-validation_size-test_size:.2f}, "
        f"validation={validation_size:.2f}, test={test_size:.2f}; created_once_with_seed={seed}"
    )
    return train_idx, val_idx, test_idx, split_name, build_split_audit(
        y,
        train_idx,
        val_idx,
        test_idx,
    )


# =========================================================
# Fixed-split persistence helpers
# =========================================================
def save_prepared_arrays_for_output_dir(data: PreparedGlobalData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_val_idx = np.sort(np.concatenate([data.train_idx, data.val_idx]))

    np.save(output_dir / "X_conn_model.npy", data.X_conn_model)
    np.save(output_dir / "y_binary.npy", data.y)

    # Save only explicit fixed-split index filenames.
    np.save(output_dir / "fixed_train_idx.npy", data.train_idx)
    np.save(output_dir / "fixed_val_idx.npy", data.val_idx)
    np.save(output_dir / "fixed_test_idx.npy", data.test_idx)
    np.save(output_dir / "fixed_train_val_idx.npy", train_val_idx)

    save_json(data.split_audit, output_dir / "split_audit.json")
    if data.groups_all is not None:
        data.groups_all.to_csv(output_dir / "site_groups_after_valid_mask.csv", index=False)


def save_fixed_split_artifacts(data: PreparedGlobalData) -> None:
    """Save the one fixed split at the root OUTPUT_DIR and epsilon directory."""
    train_val_idx = np.sort(np.concatenate([data.train_idx, data.val_idx]))

    for root in [OUTPUT_DIR, EPSILON_OUTPUT_DIR]:
        root.mkdir(parents=True, exist_ok=True)

        # Explicit fixed-split filenames for reuse by other scripts.
        np.save(root / "fixed_train_idx.npy", data.train_idx)
        np.save(root / "fixed_val_idx.npy", data.val_idx)
        np.save(root / "fixed_test_idx.npy", data.test_idx)
        np.save(root / "fixed_train_val_idx.npy", train_val_idx)

        np.save(root / "X_conn_model.npy", data.X_conn_model)
        np.save(root / "y_binary.npy", data.y)
        np.save(root / "seed_list.npy", np.asarray(SEED_LIST, dtype=int))
        np.save(root / "lr_seed.npy", np.asarray([LR_SEED], dtype=int))

        if data.groups_all is not None:
            data.groups_all.to_csv(root / "site_groups_after_valid_mask.csv", index=False)

        split_rows = []
        for split_name, idx in [
            ("train", data.train_idx),
            ("validation", data.val_idx),
            ("test", data.test_idx),
        ]:
            for row_idx in idx:
                split_rows.append({"row_index_after_valid_mask": int(row_idx), "split": split_name})
        split_df = pd.DataFrame(split_rows).sort_values("row_index_after_valid_mask")
        split_df.to_csv(root / "fixed_split_assignment.csv", index=False)
        split_df.to_csv(root / "fixed_split_indices.csv", index=False)

        save_json(data.split_audit, root / "fixed_split_audit.json")
        save_json(
            {
                "fixed_split_created_with_seed": int(LR_SEED),
                "seed_list": [int(s) for s in SEED_LIST],
                "lr_seed": int(LR_SEED),
                "lr_seed_in_seed_list": bool(LR_SEED in SEED_LIST),
                "site_aware_splitting_used": SITE_AWARE_SPLIT,
                "site_aware_n_splits": SITE_AWARE_N_SPLITS,
                "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
                "validation_size": VALIDATION_SIZE,
                "test_size": TEST_SIZE,
                "n_total_after_valid_mask": int(len(data.y)),
                "n_train": int(len(data.train_idx)),
                "n_validation": int(len(data.val_idx)),
                "n_test": int(len(data.test_idx)),
                "split_strategy": data.split_name,
                "split_audit": data.split_audit,
                "policy": (
                    "Train/validation/test split created exactly once using LR_SEED, "
                    "unless USE_PREPROCESS_DUMP=True, in which case the fixed split "
                    "is read from the non-private dumped CSVs. The exact same indices "
                    "are reused for every seed in SEED_LIST."
                ),
                "use_preprocess_dump": bool(data.preprocess_dump_used),
                "preprocess_dump_dir": str(data.preprocess_dump_dir) if data.preprocess_dump_dir is not None else None,
            },
            root / "fixed_split_summary.json",
        )


def clone_prepared_global_data_for_seed(
    fixed_data: PreparedGlobalData,
    *,
    seed: int,
    output_dir: Path,
) -> PreparedGlobalData:
    cloned = PreparedGlobalData(
        seed=int(seed),
        output_dir=output_dir,
        df=fixed_data.df,
        df_model_raw=fixed_data.df_model_raw,
        df_model=fixed_data.df_model,
        X_conn_model=fixed_data.X_conn_model,
        y=fixed_data.y,
        groups_all=fixed_data.groups_all,
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
# DP accounting
# =========================================================
def _make_rdp_accountant(orders):
    try:
        from dp_accounting import rdp

        return rdp.RdpAccountant(orders)
    except Exception:
        from dp_accounting.rdp import rdp_privacy_accountant

        return rdp_privacy_accountant.RdpAccountant(orders)


def _compose_poisson_gaussian_event(accountant, sampling, noise_multiplier, step):
    from dp_accounting import dp_event as event

    base_event = event.PoissonSampledDpEvent(
        sampling,
        event.GaussianDpEvent(noise_multiplier),
    )
    try:
        accountant.compose(event.SelfComposedDpEvent(base_event, step))
    except Exception:
        accountant.compose(base_event, count=step)


def epsilon_for_noise_multiplier(*, noise_multiplier, target_delta, sampling, step) -> float:
    if noise_multiplier <= 0:
        return math.inf
    orders = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
    orders += list(range(5, 64)) + [128, 256, 512]
    accountant = _make_rdp_accountant(orders)
    _compose_poisson_gaussian_event(accountant, sampling, noise_multiplier, step)
    return float(accountant.get_epsilon(target_delta))


def get_noise_multiplier(target_epsilon, target_delta, sampling, step) -> float:
    def objective(sigma):
        return (
            epsilon_for_noise_multiplier(
                noise_multiplier=sigma,
                target_delta=target_delta,
                sampling=sampling,
                step=step,
            )
            - target_epsilon
        )

    return float(opt.brentq(objective, 1e-6, 1000))


# =========================================================
# DP-GD logistic regression
# =========================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def predict_proba_from_params(X, params: Params, *, fit_intercept: bool):
    z = X @ params.w
    if fit_intercept:
        z = z + params.b
    return sigmoid(z)


def objective_loss(X, y, params: Params, *, l2_reg_strength: float, fit_intercept: bool):
    p = np.clip(
        predict_proba_from_params(X, params, fit_intercept=fit_intercept),
        1e-12,
        1 - 1e-12,
    )
    return float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        + 0.5 * l2_reg_strength * np.dot(params.w, params.w)
    )


def safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return np.nan


def compute_prediction_metrics_from_arrays(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    unique = np.unique(y_true)

    out = {
        "n_total": int(len(y_true)),
        "n_control": int(np.sum(y_true == 0)),
        "n_autism": int(np.sum(y_true == 1)),
        "autism_prevalence": float(np.mean(y_true == 1)) if len(y_true) else np.nan,
        "n_unique_classes": int(len(unique)),
        "predicted_autism_rate": float(np.mean(y_pred == 1)) if len(y_pred) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "log_loss": safe_metric(
            log_loss,
            y_true,
            np.clip(y_proba, 1e-12, 1 - 1e-12),
            labels=[0, 1],
        ),
    }
    if len(unique) == 2:
        out.update(
            {
                "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
                "roc_auc": safe_metric(roc_auc_score, y_true, y_proba),
                "average_precision": safe_metric(average_precision_score, y_true, y_proba),
            }
        )
    else:
        out.update(
            {"balanced_accuracy": np.nan, "roc_auc": np.nan, "average_precision": np.nan}
        )

    rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["control", "autism"],
        output_dict=True,
        zero_division=0,
    )
    for ok, ov in rep.items():
        if isinstance(ov, dict):
            for ik, iv in ov.items():
                if isinstance(iv, (int, float, np.integer, np.floating)):
                    out[f"{ok}_{ik}"] = float(iv)
        elif isinstance(ov, (int, float, np.integer, np.floating)):
            out[ok] = float(ov)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out.update(
        {
            "cm_true_control_pred_control": int(cm[0, 0]),
            "cm_true_control_pred_autism": int(cm[0, 1]),
            "cm_true_autism_pred_control": int(cm[1, 0]),
            "cm_true_autism_pred_autism": int(cm[1, 1]),
        }
    )
    return out


def print_metric_block(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            print(f"{k:>36s}: {v:.6f}")
        else:
            print(f"{k:>36s}: {v}")


def evaluate_params(X, y, params: Params, *, l2_reg_strength: float, fit_intercept: bool, prefix: str):
    p = predict_proba_from_params(X, params, fit_intercept=fit_intercept)
    pred = (p >= 0.5).astype(int)
    m = compute_prediction_metrics_from_arrays(y, pred, p)
    m["objective"] = objective_loss(
        X,
        y,
        params,
        l2_reg_strength=l2_reg_strength,
        fit_intercept=fit_intercept,
    )
    return {f"{prefix}_{k}": v for k, v in m.items()}, pred, p


@njit(cache=True)
def _jit_sigmoid_scalar(z):
    if z > 50.0:
        z = 50.0
    elif z < -50.0:
        z = -50.0
    return 1.0 / (1.0 + math.exp(-z))


@njit(cache=True)
def _jit_basic_metrics(X, y, w, b, fit_intercept_int, l2_reg_strength):
    n, d = X.shape
    if n == 0:
        return math.nan, math.nan, math.nan
    correct = 0
    loss = 0.0
    for i in range(n):
        z = 0.0
        for j in range(d):
            z += X[i, j] * w[j]
        if fit_intercept_int == 1:
            z += b
        p = _jit_sigmoid_scalar(z)
        pc = min(max(p, 1e-12), 1.0 - 1e-12)
        if (p >= 0.5 and y[i] >= 0.5) or (p < 0.5 and y[i] < 0.5):
            correct += 1
        loss += -(y[i] * math.log(pc) + (1.0 - y[i]) * math.log(1.0 - pc))
    wnorm = 0.0
    for j in range(d):
        wnorm += w[j] * w[j]
    ll = loss / n
    return correct / n, ll, ll + 0.5 * l2_reg_strength * wnorm


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
def _jit_train_full_batch_dp_gd(
    X_train,
    y_train,
    X_val,
    y_val,
    has_val,
    X_test,
    y_test,
    has_test,
    lr,
    l2,
    fit_intercept_int,
    iterations,
    clip_norm,
    sigma,
    l2_mode,
    random_state,
    record_every,
):
    n, d = X_train.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    hist = np.empty((iterations + 1, 17), dtype=np.float64)
    hist[:, :] = np.nan
    np.random.seed(random_state)
    rec = 0

    for step in range(0, iterations + 1):
        mean_norm = math.nan
        max_norm = math.nan
        frac_clip = math.nan
        avg_w_norm = math.nan
        avg_b_abs = math.nan

        if step > 0:
            sum_w = np.zeros(d, dtype=np.float64)
            sum_b = 0.0
            total_norm = 0.0
            max_norm = 0.0
            clipped = 0

            for i in range(n):
                z = 0.0
                for j in range(d):
                    z += X_train[i, j] * w[j]
                if fit_intercept_int == 1:
                    z += b
                err = _jit_sigmoid_scalar(z) - y_train[i]

                norm_sq = err * err if fit_intercept_int == 1 else 0.0
                for j in range(d):
                    gj = err * X_train[i, j]
                    if l2_mode == 1:
                        gj += l2 * w[j]
                    norm_sq += gj * gj

                norm = math.sqrt(norm_sq)
                total_norm += norm
                max_norm = max(max_norm, norm)
                scale = 1.0
                if clip_norm / (norm + 1e-6) < 1.0:
                    scale = clip_norm / (norm + 1e-6)
                    clipped += 1

                for j in range(d):
                    gj = err * X_train[i, j]
                    if l2_mode == 1:
                        gj += l2 * w[j]
                    sum_w[j] += gj * scale
                if fit_intercept_int == 1:
                    sum_b += err * scale

            grad_norm_sq = 0.0
            noise_std = sigma * clip_norm
            for j in range(d):
                g = (sum_w[j] + np.random.normal(0.0, noise_std)) / n
                if l2_mode == 0:
                    g += l2 * w[j]
                w[j] -= lr * g
                grad_norm_sq += g * g

            gb = 0.0
            if fit_intercept_int == 1:
                gb = (sum_b + np.random.normal(0.0, noise_std)) / n
                b -= lr * gb

            mean_norm = total_norm / n
            frac_clip = clipped / n
            avg_w_norm = math.sqrt(grad_norm_sq)
            avg_b_abs = abs(gb)

        if step == 0 or step == 1 or step == iterations or step % record_every == 0:
            tr = _jit_basic_metrics(X_train, y_train, w, b, fit_intercept_int, l2)
            va = (
                _jit_basic_metrics(X_val, y_val, w, b, fit_intercept_int, l2)
                if has_val == 1
                else (math.nan, math.nan, math.nan)
            )
            te = (
                _jit_basic_metrics(X_test, y_test, w, b, fit_intercept_int, l2)
                if has_test == 1
                else (math.nan, math.nan, math.nan)
            )
            hist[rec, 0] = step
            hist[rec, 1] = lr
            hist[rec, 2] = n
            hist[rec, 3] = mean_norm
            hist[rec, 4] = max_norm
            hist[rec, 5] = frac_clip
            hist[rec, 6] = avg_w_norm
            hist[rec, 7] = avg_b_abs
            hist[rec, 8] = tr[0]
            hist[rec, 9] = tr[1]
            hist[rec, 10] = tr[2]
            hist[rec, 11] = va[0]
            hist[rec, 12] = va[1]
            hist[rec, 13] = va[2]
            hist[rec, 14] = te[0]
            hist[rec, 15] = te[1]
            hist[rec, 16] = te[2]
            rec += 1

    return w, b, hist, rec


class ManualFullBatchDPLogisticRegression:
    def __init__(
        self,
        *,
        C,
        fit_intercept,
        learning_rate,
        iterations,
        clip_norm,
        noise_multiplier,
        l2_reg_handling,
        random_state,
        record_every_n_steps=1,
    ):
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.l2_reg_handling = str(l2_reg_handling)
        self.random_state = int(random_state)
        self.record_every_n_steps = max(1, int(record_every_n_steps))
        self.params_ = None
        self.l2_reg_strength_ = None
        self.step_history_ = pd.DataFrame()

    def fit(self, X_train, y_train, *, X_val=None, y_val=None, X_test=None, y_test=None):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        n, d = X_train.shape
        self.l2_reg_strength_ = 1.0 / (self.C * n)

        if USE_NUMBA_JIT and NUMBA_AVAILABLE:
            Xv = X_val if X_val is not None else np.zeros((0, d), dtype=np.float64)
            yv = y_val if y_val is not None else np.zeros(0, dtype=np.float64)
            Xt = X_test if X_test is not None else np.zeros((0, d), dtype=np.float64)
            yt = y_test if y_test is not None else np.zeros(0, dtype=np.float64)
            mode = 0 if self.l2_reg_handling == "add_after_noise" else 1
            w, b, hist, nrows = _jit_train_full_batch_dp_gd(
                X_train,
                y_train,
                np.asarray(Xv, dtype=np.float64),
                np.asarray(yv, dtype=np.float64),
                int(X_val is not None),
                np.asarray(Xt, dtype=np.float64),
                np.asarray(yt, dtype=np.float64),
                int(X_test is not None),
                self.learning_rate,
                self.l2_reg_strength_,
                int(self.fit_intercept),
                self.iterations,
                self.clip_norm,
                self.noise_multiplier,
                mode,
                int(self.random_state % (2**31 - 1)),
                self.record_every_n_steps,
            )
            self.params_ = Params(np.asarray(w, dtype=np.float64), float(b))
            self.step_history_ = pd.DataFrame(hist[:nrows].copy(), columns=HISTORY_COLUMNS)
            self.step_history_["step"] = self.step_history_["step"].astype(int)
            self.step_history_["batch_size"] = self.step_history_["batch_size"].astype(int)
        else:
            self.params_ = self._python_fit(X_train, y_train, X_val, y_val, X_test, y_test)
        return self

    def _python_fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        n, d = X_train.shape
        params = Params(np.zeros(d), 0.0)
        rng = np.random.default_rng(self.random_state)
        rows = []

        for step in range(self.iterations + 1):
            if step > 0:
                p = predict_proba_from_params(X_train, params, fit_intercept=self.fit_intercept)
                err = p - y_train
                gw = err[:, None] * X_train
                gb = err.copy() if self.fit_intercept else np.zeros_like(err)
                if self.l2_reg_handling == "inside_per_example":
                    gw += self.l2_reg_strength_ * params.w[None, :]
                norms = np.sqrt(np.sum(gw * gw, axis=1) + gb * gb)
                scale = np.minimum(1.0, self.clip_norm / (norms + 1e-6))
                sw = np.sum(gw * scale[:, None], axis=0)
                sw += rng.normal(0.0, self.noise_multiplier * self.clip_norm, d)
                sb = float(np.sum(gb * scale))
                if self.fit_intercept:
                    sb += float(rng.normal(0.0, self.noise_multiplier * self.clip_norm))
                avgw = sw / n
                avgb = sb / n if self.fit_intercept else 0.0
                if self.l2_reg_handling == "add_after_noise":
                    avgw += self.l2_reg_strength_ * params.w
                params = Params(
                    params.w - self.learning_rate * avgw,
                    float(params.b - self.learning_rate * avgb if self.fit_intercept else 0.0),
                )

            if step == 0 or step == 1 or step == self.iterations or step % self.record_every_n_steps == 0:
                row = {"step": step, "learning_rate": self.learning_rate, "batch_size": n}
                row.update(
                    evaluate_params(
                        X_train,
                        y_train,
                        params,
                        l2_reg_strength=self.l2_reg_strength_,
                        fit_intercept=self.fit_intercept,
                        prefix="train",
                    )[0]
                )
                if X_val is not None:
                    row.update(
                        evaluate_params(
                            X_val,
                            y_val,
                            params,
                            l2_reg_strength=self.l2_reg_strength_,
                            fit_intercept=self.fit_intercept,
                            prefix="val",
                        )[0]
                    )
                if X_test is not None:
                    row.update(
                        evaluate_params(
                            X_test,
                            y_test,
                            params,
                            l2_reg_strength=self.l2_reg_strength_,
                            fit_intercept=self.fit_intercept,
                            prefix="test",
                        )[0]
                    )
                rows.append(row)

        self.step_history_ = pd.DataFrame(rows)
        return params

    def metrics(self, X, y, *, prefix):
        if self.params_ is None or self.l2_reg_strength_ is None:
            raise RuntimeError("Model not fitted")
        return evaluate_params(
            np.asarray(X, dtype=np.float64),
            y,
            self.params_,
            l2_reg_strength=self.l2_reg_strength_,
            fit_intercept=self.fit_intercept,
            prefix=prefix,
        )


# =========================================================
# Training helpers
# =========================================================
def stable_lr_key(lr: float) -> int:
    digest = hashlib.blake2b(f"{float(lr):.16g}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**31 - 1)


def make_lr_seed(base_seed: int, lr: float, *, final: bool = False) -> int:
    return int(
        (int(base_seed) * 1000003 + stable_lr_key(lr) + (123457 if final else 0))
        % (2**32 - 1)
    )


def selection_score_from_row(row: dict, metric: str, prefix: str = "val") -> float:
    val = float(row[f"{prefix}_{metric}"])
    return -val if metric == "log_loss" else val


def train_one_learning_rate(*, learning_rate, X_train, y_train, X_val, y_val, noise_multiplier, seed, tuned_C):
    model = ManualFullBatchDPLogisticRegression(
        C=tuned_C,
        fit_intercept=TUNED_FIT_INTERCEPT,
        learning_rate=learning_rate,
        iterations=ITERATIONS,
        clip_norm=CLIP_NORM,
        noise_multiplier=noise_multiplier,
        l2_reg_handling=L2_REG_HANDLING,
        random_state=make_lr_seed(seed, learning_rate),
        record_every_n_steps=RECORD_EVERY_N_STEPS,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    hist = model.step_history_.copy()
    last = hist.iloc[-1].to_dict()
    tr, _, _ = model.metrics(X_train, y_train, prefix="train")
    va, _, _ = model.metrics(X_val, y_val, prefix="val")
    row = {"learning_rate": learning_rate, "phase": "selection_train_model", **last, **tr, **va}
    row["selection_score"] = selection_score_from_row(row, SELECTION_METRIC, prefix="val")
    return row, hist, model


def _run_lr_trial_for_parallel(lr, X_train, y_train, X_val, y_val, noise_multiplier, seed, tuned_C):
    return lr, train_one_learning_rate(
        learning_rate=lr,
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
    initial_lrs,
    X_train,
    y_train,
    X_val,
    y_val,
    noise_multiplier,
    seed,
    tuned_C,
    out_dir,
):
    tried, histories, models = {}, {}, {}
    current = sorted(set(float(x) for x in initial_lrs))

    for round_id in range(MAX_ENDPOINT_EXPANSIONS + 1):
        print(f"\nLearning-rate grid round {round_id + 1}: {current}")
        todo = [lr for lr in current if lr not in tried]

        if PARALLELIZE_LEARNING_RATES and JOBLIB_AVAILABLE and len(todo) > 1:
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
                for lr in todo
            )
        else:
            results = [
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
                for lr in todo
            ]

        for lr, (row, hist, model) in results:
            tried[lr] = row
            histories[lr] = hist
            models[lr] = model
            hist.to_csv(out_dir / f"step_history_lr_{lr:g}.csv", index=False)
            print(
                f"  lr={lr:g} | val_acc={row['val_accuracy']:.6f} | "
                f"val_bal_acc={row['val_balanced_accuracy']:.6f} | "
                f"val_log_loss={row['val_log_loss']:.6f}"
            )

        df = pd.DataFrame([tried[lr] for lr in sorted(tried)]).reset_index(drop=True)
        best_lr = float(df.loc[int(df["selection_score"].idxmax()), "learning_rate"])
        ordered = sorted(set(current))
        if best_lr == ordered[0]:
            new = ordered[0] / ENDPOINT_EXPANSION_FACTOR
            if new not in ordered:
                current = sorted(set(ordered + [new]))
                continue
        if best_lr == ordered[-1]:
            new = ordered[-1] * ENDPOINT_EXPANSION_FACTOR
            if new not in ordered:
                current = sorted(set(ordered + [new]))
                continue
        break

    return (
        pd.DataFrame([tried[lr] for lr in sorted(tried)]).reset_index(drop=True),
        histories,
        models,
    )


def train_final_refit_model(*, best_lr, X_train_val, y_train_val, X_test, y_test, noise_multiplier, seed, tuned_C):
    model = ManualFullBatchDPLogisticRegression(
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
    model.fit(X_train_val, y_train_val, X_test=X_test, y_test=y_test)
    trainval_m, ytv_pred, ytv_proba = model.metrics(X_train_val, y_train_val, prefix="trainval")
    test_m, yt_pred, yt_proba = model.metrics(X_test, y_test, prefix="test")
    return model, model.step_history_.copy(), trainval_m, test_m, ytv_pred, ytv_proba, yt_pred, yt_proba


# =========================================================
# Subgroup helpers
# =========================================================
def build_reporting_group_series(df_meta: pd.DataFrame, feature_name: str) -> pd.Series:
    if feature_name not in df_meta.columns:
        return pd.Series(MISSING_GROUP_LABEL, index=df_meta.index, dtype="string", name=feature_name)

    s = df_meta[feature_name].copy()

    if feature_name in {"SITE_ID", "FIQ_TEST_TYPE", "PIQ_TEST_TYPE", "VIQ_TEST_TYPE"}:
        s = s.map(
            normalize_test_type
            if feature_name.endswith("_TEST_TYPE")
            else lambda x: np.nan if pd.isna(x) else str(x).strip().upper()
        )
        return (
            pd.Series(s, index=df_meta.index, dtype="object")
            .where(~pd.isna(s), MISSING_GROUP_LABEL)
            .astype("string")
        )

    if feature_name == "HANDEDNESS_CATEGORY":
        s = s.map(normalize_handedness_category).where(lambda z: ~pd.isna(z), MISSING_GROUP_LABEL)
        return s.map(
            lambda x: {
                "R": "Right",
                "L": "Left",
                "Ambi": "Mixed/Ambi",
                "MIXED": "Mixed/Ambi",
                "L->R": "Mixed/Ambi",
            }.get(x, x)
        ).astype("string")

    if feature_name == "SEX":
        return (
            pd.to_numeric(s, errors="coerce")
            .map({1.0: "Male", 2.0: "Female"})
            .fillna(MISSING_GROUP_LABEL)
            .astype("string")
        )

    if feature_name == "EYE_STATUS_AT_SCAN":
        return (
            pd.to_numeric(s, errors="coerce")
            .map({1.0: "Eyes Open", 2.0: "Eyes Closed"})
            .fillna(MISSING_GROUP_LABEL)
            .astype("string")
        )

    if feature_name == "CURRENT_MED_STATUS":
        return (
            pd.to_numeric(s, errors="coerce")
            .map({0.0: "No Current Meds", 1.0: "On Current Meds"})
            .fillna(MISSING_GROUP_LABEL)
            .astype("string")
        )

    if feature_name in SENSIBLE_NUMERIC_BINS:
        spec = SENSIBLE_NUMERIC_BINS[feature_name]
        binned = pd.cut(
            pd.to_numeric(s, errors="coerce"),
            bins=spec["bins"],
            labels=spec["labels"],
            right=False,
            include_lowest=True,
        ).astype("object")
        if feature_name in {"FIQ", "PIQ", "VIQ"}:
            binned = pd.Series(binned, index=df_meta.index).where(
                ~pd.isna(binned),
                f"{feature_name} {MISSING_GROUP_LABEL}",
            )
        return pd.Series(binned, index=df_meta.index).astype("string")

    return (
        s.astype("string")
        .fillna(MISSING_GROUP_LABEL)
        .replace({"": MISSING_GROUP_LABEL, "nan": MISSING_GROUP_LABEL, "None": MISSING_GROUP_LABEL})
    )


def get_group_order_for_feature(feature_name: str, group_series: pd.Series) -> list[str]:
    vals = set(pd.Series(group_series, dtype="string").dropna().astype(str).tolist())
    if feature_name in {"AGE_AT_SCAN", "FIQ", "PIQ", "VIQ"}:
        order = list(SENSIBLE_NUMERIC_BINS[feature_name]["labels"])
        if feature_name in {"FIQ", "PIQ", "VIQ"}:
            order.append(f"{feature_name} {MISSING_GROUP_LABEL}")
        return [x for x in order if x in vals]
    return sorted(vals)


def compute_subgroup_metrics(
    *,
    df_meta,
    y_true,
    y_pred,
    y_proba,
    feature_names,
    experiment_key,
    experiment_name,
    split_name,
    best_lr,
    target_epsilon,
    achieved_epsilon,
    delta,
    noise_multiplier,
):
    rows = []
    for feat in feature_names:
        gs = build_reporting_group_series(df_meta, feat)
        for gv in get_group_order_for_feature(feat, gs):
            mask = gs.eq(gv).to_numpy()
            if not np.any(mask):
                continue
            m = compute_prediction_metrics_from_arrays(y_true[mask], y_pred[mask], y_proba[mask])
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
                    "feature": feat,
                    "group": str(gv),
                    "n_samples": m["n_total"],
                    "n_control": m["n_control"],
                    "n_autism": m["n_autism"],
                    "autism_prevalence": m["autism_prevalence"],
                    "predicted_autism_rate": m["predicted_autism_rate"],
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "log_loss": m["log_loss"],
                    "roc_auc": m["roc_auc"],
                    "average_precision": m["average_precision"],
                    "n_unique_classes": m["n_unique_classes"],
                    "control_precision": m.get("control_precision", np.nan),
                    "control_recall": m.get("control_recall", np.nan),
                    "control_f1-score": m.get("control_f1-score", np.nan),
                    "autism_precision": m.get("autism_precision", np.nan),
                    "autism_recall": m.get("autism_recall", np.nan),
                    "autism_f1-score": m.get("autism_f1-score", np.nan),
                    "cm_true_control_pred_control": m["cm_true_control_pred_control"],
                    "cm_true_control_pred_autism": m["cm_true_control_pred_autism"],
                    "cm_true_autism_pred_control": m["cm_true_autism_pred_control"],
                    "cm_true_autism_pred_autism": m["cm_true_autism_pred_autism"],
                }
            )
    return pd.DataFrame(rows)


def save_basic_plots(histories, selection_df, out_dir):
    try:
        for col in ["train_accuracy", "val_accuracy", "train_log_loss", "val_log_loss"]:
            plt.figure(figsize=(8, 5))
            for lr, hist in sorted(histories.items()):
                if col in hist.columns:
                    plt.plot(hist["step"], hist[col], label=f"lr={lr:g}")
            plt.xlabel("DP-GD step")
            plt.ylabel(col)
            plt.title(f"{col} vs step")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{col}_vs_step_all_lrs.png", dpi=200)
            plt.close()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")



# =========================================================
# Prepared dump loading helpers
# =========================================================
def _require_preprocess_dump_files() -> None:
    required = [PREPROCESS_DUMP_TRAIN_CSV, PREPROCESS_DUMP_VAL_CSV, PREPROCESS_DUMP_TEST_CSV, PREPROCESS_DUMP_TRAIN_VAL_CSV]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("USE_PREPROCESS_DUMP=True, but these required CSV files are missing:\n  - " + "\n  - ".join(missing))


def _load_fixed_split_csv(path: Path, expected_split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["row_index_after_valid_mask", "y_binary"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain {col}. Please regenerate the non-private fixed split dump.")
    df = df.copy()
    df["row_index_after_valid_mask"] = pd.to_numeric(df["row_index_after_valid_mask"], errors="raise").astype(int)
    df["y_binary"] = pd.to_numeric(df["y_binary"], errors="raise").astype(int)
    if "split" not in df.columns:
        df.insert(1, "split", expected_split)
    else:
        df["split"] = expected_split
    return df


def _print_column_names_for_split_df(*, split_name: str, path: Path, df: pd.DataFrame) -> None:
    """Print every column name in a loaded fixed-split CSV."""
    print("\n" + "=" * 120)
    print(f"COLUMN NAMES FOR PREPROCESS DUMP SPLIT: {split_name}")
    print("=" * 120)
    print(f"File: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:05d}] {col}")


def _print_preprocess_dump_column_names(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
) -> None:
    """Print all column names for each fixed split CSV loaded from PREPROCESS_DUMP_DIR."""
    if not PRINT_PREPROCESS_DUMP_COLUMN_NAMES:
        return

    _print_column_names_for_split_df(
        split_name="train",
        path=PREPROCESS_DUMP_TRAIN_CSV,
        df=train_df,
    )
    _print_column_names_for_split_df(
        split_name="validation",
        path=PREPROCESS_DUMP_VAL_CSV,
        df=val_df,
    )
    _print_column_names_for_split_df(
        split_name="test",
        path=PREPROCESS_DUMP_TEST_CSV,
        df=test_df,
    )
    _print_column_names_for_split_df(
        split_name="train_validation",
        path=PREPROCESS_DUMP_TRAIN_VAL_CSV,
        df=train_val_df,
    )


def _read_preprocess_dump_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_preprocess_dump_files()
    train_df = _load_fixed_split_csv(PREPROCESS_DUMP_TRAIN_CSV, "train")
    val_df = _load_fixed_split_csv(PREPROCESS_DUMP_VAL_CSV, "validation")
    test_df = _load_fixed_split_csv(PREPROCESS_DUMP_TEST_CSV, "test")
    train_val_df = pd.read_csv(PREPROCESS_DUMP_TRAIN_VAL_CSV)
    for col in ["row_index_after_valid_mask", "y_binary"]:
        if col not in train_val_df.columns:
            raise ValueError(f"{PREPROCESS_DUMP_TRAIN_VAL_CSV} must contain {col}.")
    train_val_df = train_val_df.copy()
    train_val_df["row_index_after_valid_mask"] = pd.to_numeric(train_val_df["row_index_after_valid_mask"], errors="raise").astype(int)
    train_val_df["y_binary"] = pd.to_numeric(train_val_df["y_binary"], errors="raise").astype(int)
    _print_preprocess_dump_column_names(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_val_df=train_val_df,
    )
    return train_df, val_df, test_df, train_val_df


def _combine_dumped_splits_in_model_index_space(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    duplicated = combined["row_index_after_valid_mask"].duplicated(keep=False)
    if duplicated.any():
        dup = combined.loc[duplicated, "row_index_after_valid_mask"].sort_values().unique().tolist()
        raise ValueError(f"Duplicate row_index_after_valid_mask values in fixed CSV dump: {dup[:20]}")
    combined = combined.sort_values("row_index_after_valid_mask").reset_index(drop=True)
    index_to_position = {int(idx): int(pos) for pos, idx in enumerate(combined["row_index_after_valid_mask"].to_numpy())}
    train_idx = np.asarray([index_to_position[int(idx)] for idx in train_df["row_index_after_valid_mask"].to_numpy()], dtype=int)
    val_idx = np.asarray([index_to_position[int(idx)] for idx in val_df["row_index_after_valid_mask"].to_numpy()], dtype=int)
    test_idx = np.asarray([index_to_position[int(idx)] for idx in test_df["row_index_after_valid_mask"].to_numpy()], dtype=int)
    return combined, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _validate_train_val_dump_consistency(train_df: pd.DataFrame, val_df: pd.DataFrame, train_val_df: pd.DataFrame) -> None:
    expected = sorted(set(train_df["row_index_after_valid_mask"].astype(int)).union(set(val_df["row_index_after_valid_mask"].astype(int))))
    observed = sorted(set(train_val_df["row_index_after_valid_mask"].astype(int)))
    if expected != observed:
        raise ValueError("train_val_fixed.csv is not equal to train_fixed.csv + val_fixed.csv in row_index_after_valid_mask space. Please regenerate the dump.")


def _build_frames_from_preprocess_dump(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Optional[pd.Series]]:
    fc_cols = sorted([c for c in combined.columns if c.startswith("fc_")], key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 and x.split("_")[1].isdigit() else x)
    if not fc_cols:
        raise ValueError("The fixed CSV dump does not contain fc_ columns. The DP-GD code needs fMRI connectivity features.")
    X_conn_model = combined[fc_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X_conn_model = np.where(np.isfinite(X_conn_model), X_conn_model, np.nan)
    y = pd.to_numeric(combined["y_binary"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_binary in the fixed CSV dump must contain only 0/1 labels.")

    phen_cols = [c for c in combined.columns if c.startswith("phen__")]
    df_model = pd.DataFrame(index=np.arange(len(combined)))
    for col in phen_cols:
        df_model[col[len("phen__"):]] = combined[col].to_numpy()
    df_model[TARGET_COL] = np.where(y == 1, 1, 2)
    if "SITE_ID" in combined.columns:
        df_model["SITE_ID"] = combined["SITE_ID"].astype("string").to_numpy()

    df_model_raw = pd.DataFrame(index=np.arange(len(combined)))
    if f"{TARGET_COL}_original" in combined.columns:
        df_model_raw[TARGET_COL] = pd.to_numeric(combined[f"{TARGET_COL}_original"], errors="coerce").to_numpy()
    else:
        df_model_raw[TARGET_COL] = np.where(y == 1, 1, 2)
    if "SITE_ID" in combined.columns:
        df_model_raw["SITE_ID"] = combined["SITE_ID"].astype("string").to_numpy()
    for col in combined.columns:
        if col.startswith("meta__"):
            df_model_raw[col[len("meta__"):]] = combined[col].to_numpy()
    for col in df_model.columns:
        if col not in df_model_raw.columns:
            df_model_raw[col] = df_model[col].to_numpy()

    groups_all = get_site_groups(df_model_raw)
    df = df_model.copy()
    for col in df_model_raw.columns:
        if col not in df.columns:
            df[col] = df_model_raw[col].to_numpy()
    return df, df_model_raw, df_model, X_conn_model, y, groups_all


def _load_split_summary_from_dump() -> tuple[Optional[str], Optional[dict]]:
    summary_path = PREPROCESS_DUMP_DIR / "fixed_split_summary.json"
    audit_path = PREPROCESS_DUMP_DIR / "fixed_split_audit.json"
    split_name = None
    audit = None
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        split_name = summary.get("split_strategy") or summary.get("source_split_strategy")
        audit = summary.get("split_audit") or summary.get("source_split_audit")
    if audit is None and audit_path.exists():
        with open(audit_path, "r") as f:
            audit = json.load(f)
    return split_name, audit


def load_preprocess_dump_as_global_data(*, seed: int, output_dir: Path) -> PreparedGlobalData:
    """Load fixed split CSVs dumped by the non-private fixed-split pipeline."""
    if USE_SITE_ID_AS_FEATURE:
        raise ValueError("USE_SITE_ID_AS_FEATURE must remain False for SITE-aware Non-IID experiments.")
    feature_json_path = resolve_feature_json_path(FEATURE_JSON_CANDIDATES)
    feature_groups = load_feature_groups(feature_json_path)
    print("\nUSE_PREPROCESS_DUMP=True")
    print("Reading prepared fixed split CSVs from:")
    print(PREPROCESS_DUMP_DIR)
    print("\nFeature JSON path used:")
    print(feature_json_path)

    train_df, val_df, test_df, train_val_df = _read_preprocess_dump_splits()
    _validate_train_val_dump_consistency(train_df, val_df, train_val_df)
    combined, train_idx, val_idx, test_idx = _combine_dumped_splits_in_model_index_space(train_df, val_df, test_df)
    df, df_model_raw, df_model, X_conn_model, y, groups_all = _build_frames_from_preprocess_dump(combined)
    if SITE_AWARE_SPLIT and groups_all is None:
        raise ValueError("SITE_AWARE_SPLIT=True but SITE_ID is missing from the fixed CSV dump.")

    loaded_split_name, loaded_audit = _load_split_summary_from_dump()
    split_name = loaded_split_name or f"Fixed train/validation/test split loaded from non-private prepared CSV dump at {PREPROCESS_DUMP_DIR}"
    audit = build_split_audit(y, train_idx, val_idx, test_idx, groups_all)
    if isinstance(loaded_audit, dict):
        audit["loaded_fixed_split_audit_from_dump"] = loaded_audit
    audit["use_preprocess_dump"] = True
    audit["preprocess_dump_dir"] = str(PREPROCESS_DUMP_DIR)

    print("\nPrepared dump loaded:")
    print("df_model_raw shape:", df_model_raw.shape)
    print("df_model shape:", df_model.shape)
    print("X_conn_model shape:", X_conn_model.shape)
    print("Binary class counts (0=Control, 1=Autism):")
    print(pd.Series(y).value_counts().sort_index())
    print("\nFixed split loaded from dump:")
    print(f"Train size      : {len(train_idx)}")
    print(f"Validation size : {len(val_idx)}")
    print(f"Test size       : {len(test_idx)}")
    print(f"Train class counts      : {_class_count_dict(y[train_idx])}")
    print(f"Validation class counts : {_class_count_dict(y[val_idx])}")
    print(f"Test class counts       : {_class_count_dict(y[test_idx])}")

    data = PreparedGlobalData(
        seed=int(seed), output_dir=output_dir, df=df, df_model_raw=df_model_raw,
        df_model=df_model, X_conn_model=X_conn_model, y=y, groups_all=groups_all,
        feature_groups=feature_groups, feature_json_path=feature_json_path,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        split_name=split_name, split_audit=audit,
        preprocess_dump_used=True, preprocess_dump_dir=PREPROCESS_DUMP_DIR,
    )
    save_prepared_arrays_for_output_dir(data, output_dir)
    return data


# =========================================================
# Main experiment routines
# =========================================================
def prepare_global_data(*, seed: int, output_dir: Path) -> PreparedGlobalData:
    if PHENOTYPE_OTHER_NUMERIC_MODE not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError("Invalid PHENOTYPE_OTHER_NUMERIC_MODE")
    if USE_SITE_ID_AS_FEATURE:
        raise ValueError("USE_SITE_ID_AS_FEATURE must remain False for SITE-aware Non-IID experiments.")

    df = clean_abide_dataframe(load_abide_dataframe(CSV_PATH))

    if "HANDEDNESS_CATEGORY" in df.columns:
        df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].map(normalize_handedness_category)

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

    for col in [TARGET_COL] + get_all_numeric_candidate_columns(feature_groups):
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
            f"Row mismatch: X_conn rows={X_conn_np.shape[0]}, df rows={len(df)}"
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

    groups_all = get_site_groups(df_model_raw)
    if groups_all is not None:
        print("\nSITE_ID summary after valid-mask filtering:")
        print(f"Number of unique SITE_ID groups: {groups_all.nunique()}")
        print("Top SITE_ID counts:")
        print(groups_all.value_counts().head(20).to_string())

    if SITE_AWARE_SPLIT and groups_all is None:
        raise ValueError("SITE_AWARE_SPLIT=True but SITE_ID column is missing.")

    train_idx, val_idx, test_idx, split_name, audit = make_train_val_test_split(
        y,
        seed,
        VALIDATION_SIZE,
        TEST_SIZE,
        groups=groups_all,
        site_aware=SITE_AWARE_SPLIT,
        n_splits=SITE_AWARE_N_SPLITS,
    )

    print("\nFixed split strategy used for all experiments and all seeds:")
    print(split_name)
    print(f"Split creation seed : {seed}")
    print(f"Train size          : {len(train_idx)}")
    print(f"Validation size     : {len(val_idx)}")
    print(f"Test size           : {len(test_idx)}")
    print(f"Train class counts      : {_class_count_dict(y[train_idx])}")
    print(f"Validation class counts : {_class_count_dict(y[val_idx])}")
    print(f"Test class counts       : {_class_count_dict(y[test_idx])}")

    if groups_all is not None:
        site_audit = audit.get("site_disjointness", {})
        print("\nSITE-aware split audit:")
        print(f"Train unique SITE_ID count      : {site_audit.get('train_unique_site_count')}")
        print(f"Validation unique SITE_ID count : {site_audit.get('validation_unique_site_count')}")
        print(f"Test unique SITE_ID count       : {site_audit.get('test_unique_site_count')}")
        print(f"Train/validation SITE_ID overlap count : {site_audit.get('train_val_site_overlap_count')}")
        print(f"Train/test SITE_ID overlap count       : {site_audit.get('train_test_site_overlap_count')}")
        print(f"Validation/test SITE_ID overlap count  : {site_audit.get('validation_test_site_overlap_count')}")

    data = PreparedGlobalData(
        seed=int(seed),
        output_dir=output_dir,
        df=df,
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        groups_all=groups_all,
        feature_groups=feature_groups,
        feature_json_path=feature_json_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_name=split_name,
        split_audit=audit,
        preprocess_dump_used=False,
        preprocess_dump_dir=None,
    )
    save_prepared_arrays_for_output_dir(data, output_dir)
    return data


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
):
    if not tune_learning_rate and fixed_best_lr is None:
        raise ValueError("fixed_best_lr required when tune_learning_rate=False")

    seed = int(global_data.seed)
    out_dir = global_data.output_dir / experiment_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 120)
    print(f"Running experiment: {experiment_name}")
    print(f"Seed: {seed} | seed_role: {seed_role}")
    print(f"Using tuned_C: {tuned_C}")
    print("#" * 120)

    selected_cols, numeric_cols, cat_cols, missing_cols = build_feature_lists_from_selected_keys(
        global_data.df,
        global_data.feature_groups,
        selected_group_keys,
    )
    domain_cols = [
        c
        for c in numeric_cols
        if c in DOMAIN_SCALED_NUMERIC_COLS and c in global_data.df_model.columns
    ]
    other_num_cols = [
        c
        for c in numeric_cols
        if c not in DOMAIN_SCALED_NUMERIC_COLS and c in global_data.df_model.columns
    ]

    print("\nSelected feature group keys:")
    print(selected_group_keys)
    print("\nSelected phenotype columns:")
    print(selected_cols)
    print("\nPhenotype numeric columns:")
    print(numeric_cols)
    print("\nPhenotype categorical columns:")
    print(cat_cols)
    print("\nRequested JSON columns absent from dataframe:")
    print(missing_cols if missing_cols else "None")
    print("\nPhenotype numeric columns with deterministic domain scaling + zero fill:")
    print(domain_cols)
    print("\nPhenotype other numeric columns:")
    print(other_num_cols)
    print("Processing mode for phenotype_other_numeric_cols:", PHENOTYPE_OTHER_NUMERIC_MODE)
    print("\nPhenotype categorical columns with missing/unseen -> all-zero one-hot block:")
    print(cat_cols)

    X_df, fc_cols = build_combined_dataframe(
        global_data.df_model,
        global_data.X_conn_model,
        selected_cols,
    )

    train_idx, val_idx, test_idx = (
        global_data.train_idx,
        global_data.val_idx,
        global_data.test_idx,
    )
    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
    y_train = global_data.y[train_idx]
    y_val = global_data.y[val_idx]
    y_test = global_data.y[test_idx]
    X_train_val_df = pd.concat([X_train_df, X_val_df], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    print("=" * 120)
    print("Train data column names:")
    print(X_train_df.columns.tolist())
    print("\nValidation data column names:")
    print(X_val_df.columns.tolist())
    print("\nTest data column names:")
    print(X_test_df.columns.tolist())
    print("=" * 120)

    print("\nSplit strategy:")
    print(global_data.split_name)
    print(f"Train size      : {len(X_train_df)}")
    print(f"Validation size : {len(X_val_df)}")
    print(f"Test size       : {len(X_test_df)}")
    print(f"Train class counts      : {_class_count_dict(y_train)}")
    print(f"Validation class counts : {_class_count_dict(y_val)}")
    print(f"Test class counts       : {_class_count_dict(y_test)}")

    if global_data.groups_all is not None:
        site_audit = global_data.split_audit.get("site_disjointness", {})
        print("\nSITE_ID-disjoint fixed split check:")
        print(f"Train unique SITE_ID count      : {site_audit.get('train_unique_site_count')}")
        print(f"Validation unique SITE_ID count : {site_audit.get('validation_unique_site_count')}")
        print(f"Test unique SITE_ID count       : {site_audit.get('test_unique_site_count')}")
        print(f"Train/validation overlap count  : {site_audit.get('train_val_site_overlap_count')}")
        print(f"Train/test overlap count        : {site_audit.get('train_test_site_overlap_count')}")
        print(f"Validation/test overlap count   : {site_audit.get('validation_test_site_overlap_count')}")

    _, X_train, X_val, X_test = fit_transform_preprocessor(
        X_train_df,
        X_val_df,
        X_test_df,
        domain_cols,
        other_num_cols,
        cat_cols,
        fc_cols,
    )

    n_train = len(y_train)
    selection_delta = float(n_train ** DELTA_POWER)
    sampling = 1.0
    selection_noise = (
        float(NOISE_MULTIPLIER)
        if NOISE_MULTIPLIER is not None
        else get_noise_multiplier(TARGET_EPSILON, selection_delta, sampling, ITERATIONS)
    )
    selection_eps = epsilon_for_noise_multiplier(
        noise_multiplier=selection_noise,
        target_delta=selection_delta,
        sampling=sampling,
        step=ITERATIONS,
    )
    lr_mode = "tuned_on_lr_seed" if tune_learning_rate else "fixed_from_lr_seed"
    selection_dp_summary = {
        "seed": seed,
        "phase": "selection_train_model",
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": selection_eps,
        "delta": selection_delta,
        "noise_multiplier": selection_noise,
        "clip_norm": CLIP_NORM,
        "iterations": ITERATIONS,
        "sampling": sampling,
        "n_train_for_accounting": n_train,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "site_aware_n_splits": SITE_AWARE_N_SPLITS,
        "fixed_split_created_with_seed": LR_SEED,
    }
    save_json(selection_dp_summary, out_dir / "dp_accounting_selection_train_model.json")
    print("\nSelection DP accounting summary:")
    print(json.dumps(json_clean(selection_dp_summary), indent=2))

    if tune_learning_rate:
        print("\nLR-selection policy:")
        print(
            "Running full learning-rate grid tuning only for the source "
            f"seed/experiment (LR_SEED={LR_SEED}, experiment_key={LR_TUNING_EXPERIMENT_KEY})."
        )
        selection_df, histories, models = evaluate_learning_rate_grid_with_expansion(
            initial_lrs=LEARNING_RATES_INITIAL,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            noise_multiplier=selection_noise,
            seed=seed,
            tuned_C=tuned_C,
            out_dir=out_dir,
        )
    else:
        lr = float(fixed_best_lr)
        print("\nSkipping LR-grid tuning for this seed.")
        print(f"Using fixed best LR from LR_SEED={lr_source_seed}: {lr:g}")
        row, hist, model = train_one_learning_rate(
            learning_rate=lr,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            noise_multiplier=selection_noise,
            seed=seed,
            tuned_C=tuned_C,
        )
        selection_df, histories, models = pd.DataFrame([row]), {lr: hist}, {lr: model}
        hist.to_csv(out_dir / f"step_history_lr_{lr:g}.csv", index=False)
        print(
            f"  lr={lr:g} | val_acc={row['val_accuracy']:.6f} | "
            f"val_bal_acc={row['val_balanced_accuracy']:.6f} | "
            f"val_log_loss={row['val_log_loss']:.6f}"
        )

    selection_df = selection_df.sort_values("learning_rate").reset_index(drop=True)
    metadata_cols = [
        ("seed", seed),
        ("seed_role", seed_role),
        ("experiment_key", experiment_key),
        ("experiment_name", experiment_name),
        ("lr_selection_mode", lr_mode),
        ("lr_source_seed", lr_source_seed),
        ("lr_source_experiment_key", lr_source_experiment_key),
        ("target_epsilon", TARGET_EPSILON),
        ("achieved_epsilon", selection_eps),
        ("delta", selection_delta),
        ("noise_multiplier", selection_noise),
        ("TUNED_C", tuned_C),
    ]
    for i, (col, val) in enumerate(metadata_cols):
        selection_df.insert(i, col, val)
    selection_df.to_csv(out_dir / "dp_gd_lr_selection_metrics_all_learning_rates.csv", index=False)
    save_basic_plots(histories, selection_df, out_dir)

    best_score = float(selection_df["selection_score"].max())
    best_lr = sorted(
        selection_df.loc[
            selection_df["selection_score"] >= best_score - SELECTION_TIE_TOL,
            "learning_rate",
        ].astype(float)
    )[0]
    best_row = selection_df.loc[
        selection_df["learning_rate"].astype(float) == best_lr
    ].iloc[0].to_dict()
    save_json(best_row, out_dir / "best_lr_selection_summary.json")
    print("\nBest LR used for this seed/experiment:")
    print(best_lr)
    print("Best selection score:", best_score)

    best_model = models[best_lr]
    train_m, y_train_pred, y_train_proba = best_model.metrics(X_train, y_train, prefix="train")
    val_m, y_val_pred, y_val_proba = best_model.metrics(X_val, y_val, prefix="val")

    _, X_train_val, _, X_test_final = fit_transform_preprocessor(
        X_train_val_df,
        None,
        X_test_df,
        domain_cols,
        other_num_cols,
        cat_cols,
        fc_cols,
    )
    n_train_val = len(y_train_val)
    final_delta = float(n_train_val ** DELTA_POWER)
    final_noise = (
        float(NOISE_MULTIPLIER)
        if NOISE_MULTIPLIER is not None
        else get_noise_multiplier(TARGET_EPSILON, final_delta, sampling, ITERATIONS)
    )
    final_eps = epsilon_for_noise_multiplier(
        noise_multiplier=final_noise,
        target_delta=final_delta,
        sampling=sampling,
        step=ITERATIONS,
    )
    final_dp_summary = {
        "seed": seed,
        "phase": "final_refit_train_validation_model",
        "best_lr": best_lr,
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": final_eps,
        "delta": final_delta,
        "noise_multiplier": final_noise,
        "clip_norm": CLIP_NORM,
        "iterations": ITERATIONS,
        "sampling": sampling,
        "n_train_for_accounting": n_train_val,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "site_aware_n_splits": SITE_AWARE_N_SPLITS,
        "fixed_split_created_with_seed": LR_SEED,
    }
    save_json(final_dp_summary, out_dir / "dp_accounting_final_refit_train_validation_model.json")
    print("\nFinal-refit DP accounting summary:")
    print(json.dumps(json_clean(final_dp_summary), indent=2))

    (
        final_model,
        final_hist,
        trainval_m,
        test_m,
        y_trainval_pred,
        y_trainval_proba,
        y_test_pred,
        y_test_proba,
    ) = train_final_refit_model(
        best_lr=best_lr,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test_final,
        y_test=y_test,
        noise_multiplier=final_noise,
        seed=seed,
        tuned_C=tuned_C,
    )
    final_hist.to_csv(out_dir / "final_refit_trainval_step_history.csv", index=False)

    print_metric_block(
        f"{experiment_name} | TRAIN metrics for selected train-split model",
        {
            "accuracy": train_m["train_accuracy"],
            "balanced_accuracy": train_m["train_balanced_accuracy"],
            "log_loss": train_m["train_log_loss"],
            "roc_auc": train_m["train_roc_auc"],
            "average_precision": train_m["train_average_precision"],
        },
    )
    print_metric_block(
        f"{experiment_name} | VALIDATION metrics for selected train-split model",
        {
            "accuracy": val_m["val_accuracy"],
            "balanced_accuracy": val_m["val_balanced_accuracy"],
            "log_loss": val_m["val_log_loss"],
            "roc_auc": val_m["val_roc_auc"],
            "average_precision": val_m["val_average_precision"],
        },
    )
    print_metric_block(
        f"{experiment_name} | TRAIN+VALIDATION metrics for final refit model",
        {
            "accuracy": trainval_m["trainval_accuracy"],
            "balanced_accuracy": trainval_m["trainval_balanced_accuracy"],
            "log_loss": trainval_m["trainval_log_loss"],
            "roc_auc": trainval_m["trainval_roc_auc"],
            "average_precision": trainval_m["trainval_average_precision"],
        },
    )
    print_metric_block(
        f"{experiment_name} | TEST metrics for final train+validation refit model",
        {
            "accuracy": test_m["test_accuracy"],
            "balanced_accuracy": test_m["test_balanced_accuracy"],
            "log_loss": test_m["test_log_loss"],
            "roc_auc": test_m["test_roc_auc"],
            "average_precision": test_m["test_average_precision"],
        },
    )

    print("\nValidation classification report:")
    print(
        classification_report(
            y_val,
            y_val_pred,
            target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
            digits=4,
        )
    )
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
            digits=4,
        )
    )

    (out_dir / "validation_selected_model_classification_report.txt").write_text(
        classification_report(
            y_val,
            y_val_pred,
            target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
            digits=4,
        )
    )
    (out_dir / "test_final_model_classification_report.txt").write_text(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
            digits=4,
        )
    )
    pd.DataFrame(
        confusion_matrix(y_val, y_val_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(out_dir / "validation_selected_model_confusion_matrix.csv")
    pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(out_dir / "test_final_model_confusion_matrix.csv")

    present = [c for c in SUBGROUP_EVAL_FEATURES if c in global_data.df_model_raw.columns]
    meta_train = global_data.df_model_raw.iloc[train_idx][present].reset_index(drop=True)
    meta_val = global_data.df_model_raw.iloc[val_idx][present].reset_index(drop=True)
    meta_trainval = pd.concat([meta_train, meta_val], ignore_index=True)
    meta_test = global_data.df_model_raw.iloc[test_idx][present].reset_index(drop=True)

    sg_frames = []
    for split, meta, yy, yp, ypr, eps, delt, noi in [
        ("train_selected_model", meta_train, y_train, y_train_pred, y_train_proba, selection_eps, selection_delta, selection_noise),
        ("validation_selected_model", meta_val, y_val, y_val_pred, y_val_proba, selection_eps, selection_delta, selection_noise),
        ("train_validation_final_model", meta_trainval, y_train_val, y_trainval_pred, y_trainval_proba, final_eps, final_delta, final_noise),
        ("test_final_model", meta_test, y_test, y_test_pred, y_test_proba, final_eps, final_delta, final_noise),
    ]:
        sg = compute_subgroup_metrics(
            df_meta=meta,
            y_true=yy,
            y_pred=yp,
            y_proba=ypr,
            feature_names=present,
            experiment_key=experiment_key,
            experiment_name=experiment_name,
            split_name=split,
            best_lr=best_lr,
            target_epsilon=TARGET_EPSILON,
            achieved_epsilon=eps,
            delta=delt,
            noise_multiplier=noi,
        )
        if not sg.empty:
            sg_frames.append(sg)

    subgroup_df = pd.concat(sg_frames, ignore_index=True) if sg_frames else pd.DataFrame()
    if not subgroup_df.empty:
        subgroup_df.insert(0, "seed", seed)
        subgroup_df.insert(1, "seed_role", seed_role)
        subgroup_df.insert(2, "lr_selection_mode", lr_mode)
        subgroup_df.insert(3, "lr_source_seed", lr_source_seed)
        subgroup_df.insert(4, "lr_source_experiment_key", lr_source_experiment_key)

    subgroup_df.to_csv(out_dir / "dp_gd_subgroup_metrics_all_splits.csv", index=False)
    if not subgroup_df.empty:
        subgroup_df.loc[subgroup_df["split"] == "test_final_model"].to_csv(
            out_dir / "dp_gd_test_subgroup_metrics.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(out_dir / "dp_gd_test_subgroup_metrics.csv", index=False)

    test_subgroup_df = (
        subgroup_df.loc[subgroup_df["split"] == "test_final_model"]
        if not subgroup_df.empty
        else pd.DataFrame()
    )
    if not test_subgroup_df.empty:
        print("\n" + "=" * 120)
        print(f"TEST subgroup metrics for: {experiment_name}")
        print("=" * 120)
        cols_to_print = [
            "feature",
            "group",
            "n_samples",
            "n_control",
            "n_autism",
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "roc_auc",
            "average_precision",
            "n_unique_classes",
        ]
        for feat in SUBGROUP_EVAL_FEATURES:
            feat_df = test_subgroup_df.loc[test_subgroup_df["feature"] == feat]
            if feat_df.empty:
                continue
            print("\n" + "-" * 120)
            print(f"TEST subgroup metrics for {feat}")
            print("-" * 120)
            print(feat_df[cols_to_print].to_string(index=False))

    metric_rows = [
        ("train_selected_model", "train", selection_eps, selection_delta, selection_noise, train_m, "train"),
        ("validation_selected_model", "train", selection_eps, selection_delta, selection_noise, val_m, "val"),
        ("train_validation_final_model", "train+validation", final_eps, final_delta, final_noise, trainval_m, "trainval"),
        ("test_final_model", "train+validation", final_eps, final_delta, final_noise, test_m, "test"),
    ]
    rows = []
    for split, data_name, eps, delt, noi, m, prefix in metric_rows:
        rows.append(
            {
                "seed": seed,
                "seed_role": seed_role,
                "lr_selection_mode": lr_mode,
                "lr_source_seed": lr_source_seed,
                "lr_source_experiment_key": lr_source_experiment_key,
                "experiment_key": experiment_key,
                "experiment_name": experiment_name,
                "split": split,
                "model_training_data": data_name,
                "best_lr": best_lr,
                "target_epsilon": TARGET_EPSILON,
                "achieved_epsilon": eps,
                "delta": delt,
                "noise_multiplier": noi,
                "TUNED_C": tuned_C,
                "clip_norm": CLIP_NORM,
                "iterations": ITERATIONS,
                "n_fc_features": len(fc_cols),
                "n_selected_phenotype_raw_columns": len(selected_cols),
                "selected_feature_group_keys": json.dumps(selected_group_keys),
                "accuracy": m[f"{prefix}_accuracy"],
                "balanced_accuracy": m[f"{prefix}_balanced_accuracy"],
                "log_loss": m[f"{prefix}_log_loss"],
                "roc_auc": m[f"{prefix}_roc_auc"],
                "average_precision": m[f"{prefix}_average_precision"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "dp_gd_final_metrics_summary.csv", index=False)

    save_json(
        {
            "seed": seed,
            "seed_role": seed_role,
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "selected_feature_group_keys": selected_group_keys,
            "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
            "site_aware_splitting_used": SITE_AWARE_SPLIT,
            "site_aware_n_splits": SITE_AWARE_N_SPLITS,
            "fixed_split_created_with_seed": LR_SEED,
            "use_preprocess_dump": bool(global_data.preprocess_dump_used),
            "preprocess_dump_dir": str(global_data.preprocess_dump_dir) if global_data.preprocess_dump_dir is not None else None,
            "split_strategy": global_data.split_name,
            "split_audit": global_data.split_audit,
            "best_lr_used": best_lr,
            "TUNED_C": tuned_C,
            "use_numba_jit": USE_NUMBA_JIT,
            "parallelize_learning_rates": PARALLELIZE_LEARNING_RATES,
            "selected_phenotype_columns_present_in_dataframe": selected_cols,
            "phenotype_domain_scaled_numeric_columns": domain_cols,
            "phenotype_other_numeric_columns": other_num_cols,
            "phenotype_categorical_columns": cat_cols,
            "fc_feature_count": len(fc_cols),
            "requested_json_columns_absent_from_dataframe": missing_cols,
        },
        out_dir / "run_feature_summary.json",
    )

    print("\nSaved experiment outputs in:")
    print(out_dir)
    print("\nMost important output files:")
    for fp in [
        out_dir / "dp_gd_lr_selection_metrics_all_learning_rates.csv",
        out_dir / "best_lr_selection_summary.json",
        out_dir / "dp_gd_final_metrics_summary.csv",
        out_dir / "dp_gd_test_subgroup_metrics.csv",
        out_dir / "dp_accounting_selection_train_model.json",
        out_dir / "dp_accounting_final_refit_train_validation_model.json",
        out_dir / "run_feature_summary.json",
    ]:
        print(fp)

    return metrics_df, subgroup_df, float(best_lr)


def get_experiment_by_key(key: str) -> dict:
    matches = [e for e in EXPERIMENTS if e["experiment_key"] == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one experiment with key {key}, found {len(matches)}")
    return matches[0]


def compute_mean_std_sem(df: pd.DataFrame, *, group_cols: list[str], metric_cols: list[str], seed_col: str = "seed") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(group_cols, key)}
        row["n_seeds"] = int(g[seed_col].nunique()) if seed_col in g.columns else int(len(g))
        for m in metric_cols:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            n = len(vals)
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            row[f"{m}_n"] = int(n)
            row[f"{m}_mean"] = float(vals.mean()) if n else np.nan
            row[f"{m}_std"] = std
            row[f"{m}_sem"] = float(std / math.sqrt(n)) if n > 1 else np.nan
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
):
    mdf = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    sdf = pd.concat(all_subgroup_metrics, ignore_index=True) if all_subgroup_metrics else pd.DataFrame()

    mdf.to_csv(output_dir / "all_modes_dp_gd_final_metrics_summary.csv", index=False)
    if not mdf.empty:
        mdf.loc[mdf["split"] == "test_final_model"].to_csv(
            output_dir / "all_modes_dp_gd_test_metrics_comparison.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(output_dir / "all_modes_dp_gd_test_metrics_comparison.csv", index=False)

    sdf.to_csv(output_dir / "all_modes_dp_gd_subgroup_metrics_all_splits_summary.csv", index=False)
    if not sdf.empty:
        sdf.loc[sdf["split"] == "test_final_model"].to_csv(
            output_dir / "all_modes_dp_gd_test_subgroup_metrics_summary.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(output_dir / "all_modes_dp_gd_test_subgroup_metrics_summary.csv", index=False)

    save_json(
        {
            "seed": seed,
            "seed_list": SEED_LIST,
            "lr_seed": LR_SEED,
            "lr_seed_in_seed_list": bool(LR_SEED in SEED_LIST),
            "global_best_lr_from_lr_seed": global_best_lr,
            "output_dir": str(output_dir),
            "split_strategy": global_data.split_name,
            "split_audit": global_data.split_audit,
            "fixed_split_created_with_seed": LR_SEED,
            "site_aware_splitting_used": SITE_AWARE_SPLIT,
            "site_aware_n_splits": SITE_AWARE_N_SPLITS,
            "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
            "target_epsilon": TARGET_EPSILON,
            "clip_norm": CLIP_NORM,
            "iterations": ITERATIONS,
            "n_total_after_valid_mask": len(global_data.y),
            "n_train": len(global_data.train_idx),
            "n_validation": len(global_data.val_idx),
            "n_test": len(global_data.test_idx),
        },
        output_dir / "run_summary_all_modes_dp_gd.json",
    )

    print("\n" + "=" * 120)
    print(f"Final DP-GD test metrics comparison for seed={seed}")
    print("=" * 120)
    test_comparison = mdf.loc[mdf["split"] == "test_final_model"].reset_index(drop=True) if not mdf.empty else pd.DataFrame()
    print(test_comparison.to_string(index=False) if not test_comparison.empty else "No test comparison rows available.")
    print("\nSaved seed outputs in:")
    print(output_dir)
    return mdf, sdf


def save_cross_seed_outputs(
    *,
    epsilon_output_dir: Path,
    all_seed_metrics: list[pd.DataFrame],
    all_seed_subgroup_metrics: list[pd.DataFrame],
    global_best_lr: float,
    lr_tuning_summary: dict,
):
    mdf = pd.concat(all_seed_metrics, ignore_index=True) if all_seed_metrics else pd.DataFrame()
    sdf = pd.concat(all_seed_subgroup_metrics, ignore_index=True) if all_seed_subgroup_metrics else pd.DataFrame()

    mdf.to_csv(epsilon_output_dir / "all_seeds_dp_gd_final_metrics_summary.csv", index=False)
    test_mdf = mdf.loc[mdf["split"] == "test_final_model"].reset_index(drop=True) if not mdf.empty else pd.DataFrame()
    test_mdf.to_csv(epsilon_output_dir / "all_seeds_dp_gd_test_metrics_comparison.csv", index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "log_loss", "roc_auc", "average_precision"]
    compute_mean_std_sem(
        mdf,
        group_cols=["experiment_key", "experiment_name", "split", "target_epsilon", "TUNED_C", "best_lr"],
        metric_cols=metric_cols,
    ).to_csv(epsilon_output_dir / "all_seeds_dp_gd_all_splits_metrics_mean_std_sem_n.csv", index=False)
    compute_mean_std_sem(
        test_mdf,
        group_cols=["experiment_key", "experiment_name", "target_epsilon", "TUNED_C", "best_lr"],
        metric_cols=metric_cols,
    ).to_csv(epsilon_output_dir / "all_seeds_dp_gd_test_metrics_mean_std_sem_n.csv", index=False)

    sdf.to_csv(epsilon_output_dir / "all_seeds_dp_gd_subgroup_metrics_all_splits_summary.csv", index=False)
    test_sdf = sdf.loc[sdf["split"] == "test_final_model"].reset_index(drop=True) if not sdf.empty else pd.DataFrame()
    test_sdf.to_csv(epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_summary.csv", index=False)

    sg_cols = [
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
    compute_mean_std_sem(
        sdf,
        group_cols=["experiment_key", "experiment_name", "split", "feature", "group", "target_epsilon", "best_lr"],
        metric_cols=sg_cols,
    ).to_csv(epsilon_output_dir / "all_seeds_dp_gd_subgroup_metrics_all_splits_mean_std_sem_n.csv", index=False)
    compute_mean_std_sem(
        test_sdf,
        group_cols=["experiment_key", "experiment_name", "split", "feature", "group", "target_epsilon", "best_lr"],
        metric_cols=sg_cols,
    ).to_csv(epsilon_output_dir / "all_seeds_dp_gd_test_subgroup_metrics_mean_std_sem_n.csv", index=False)

    save_json(lr_tuning_summary, epsilon_output_dir / "global_lr_selection_summary.json")
    save_json(
        {
            "seed_list": SEED_LIST,
            "lr_seed": LR_SEED,
            "lr_seed_in_seed_list": bool(LR_SEED in SEED_LIST),
            "lr_tuning_experiment_key": LR_TUNING_EXPERIMENT_KEY,
            "global_best_lr": global_best_lr,
            "target_epsilon": TARGET_EPSILON,
            "output_dir": str(OUTPUT_DIR),
            "epsilon_output_dir": str(epsilon_output_dir),
            "site_aware_splitting_used": SITE_AWARE_SPLIT,
            "site_aware_n_splits": SITE_AWARE_N_SPLITS,
            "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
            "fixed_split_created_with_seed": LR_SEED,
            "use_numba_jit": USE_NUMBA_JIT,
            "parallelize_learning_rates": PARALLELIZE_LEARNING_RATES,
        },
        epsilon_output_dir / "run_summary_all_seeds_dp_gd.json",
    )

    print("\n" + "=" * 120)
    print("Cross-seed test metrics mean/std/SEM")
    print("=" * 120)
    test_mean_sem_df = compute_mean_std_sem(
        test_mdf,
        group_cols=["experiment_key", "experiment_name", "target_epsilon", "TUNED_C", "best_lr"],
        metric_cols=metric_cols,
    )
    print(test_mean_sem_df.to_string(index=False) if not test_mean_sem_df.empty else "No cross-seed test summary available.")
    print("\nSaved cross-seed outputs in:")
    print(epsilon_output_dir)


# =========================================================
# Main
# =========================================================
def main():
    if len(SEED_LIST) == 0:
        raise ValueError("SEED_LIST must contain at least one seed.")
    if USE_SITE_ID_AS_FEATURE:
        raise ValueError("USE_SITE_ID_AS_FEATURE must be False.")

    if USE_NUMBA_JIT:
        print(f"Numba requested: {USE_NUMBA_JIT}; numba available: {NUMBA_AVAILABLE}")
        if not NUMBA_AVAILABLE:
            print("Proceeding with Python/NumPy fallback because numba is unavailable.")
    if PARALLELIZE_LEARNING_RATES:
        print(f"Parallel LR trials requested: joblib available={JOBLIB_AVAILABLE}")

    lr_exp = get_experiment_by_key(LR_TUNING_EXPERIMENT_KEY)

    print("\n" + "=" * 120)
    print("GLOBAL LEARNING-RATE TUNING WITH FIXED SPLIT")
    print("=" * 120)
    print(f"LR_SEED: {LR_SEED}")
    print(f"LR_SEED in SEED_LIST: {LR_SEED in SEED_LIST}")
    print(f"LR tuning experiment: {LR_TUNING_EXPERIMENT_KEY}")
    print(f"SEED_LIST for final reporting: {SEED_LIST}")
    print(f"SITE_AWARE_SPLIT: {SITE_AWARE_SPLIT}")
    print(f"SITE_AWARE_N_SPLITS: {SITE_AWARE_N_SPLITS}")
    print(f"USE_SITE_ID_AS_FEATURE: {USE_SITE_ID_AS_FEATURE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"EPSILON_OUTPUT_DIR: {EPSILON_OUTPUT_DIR}")
    print(f"USE_PREPROCESS_DUMP: {USE_PREPROCESS_DUMP}")
    print(f"PREPROCESS_DUMP_DIR: {PREPROCESS_DUMP_DIR}")

    # Create/load the fixed split exactly once.
    # If USE_PREPROCESS_DUMP=True, this loads the already dumped fixed CSVs from
    # the non-private pipeline instead of rebuilding raw ABIDE/X_conn preparation.
    lr_seed_dir = get_seed_output_dir(LR_SEED)
    if USE_PREPROCESS_DUMP:
        fixed_data = load_preprocess_dump_as_global_data(seed=LR_SEED, output_dir=lr_seed_dir)
    else:
        fixed_data = prepare_global_data(seed=LR_SEED, output_dir=lr_seed_dir)
    save_fixed_split_artifacts(fixed_data)

    # Tune LR once on LR_SEED using the fixed split.
    lr_metrics, lr_subgroups, global_best_lr = run_single_experiment(
        global_data=fixed_data,
        experiment_key=lr_exp["experiment_key"],
        experiment_name=lr_exp["experiment_name"],
        selected_group_keys=lr_exp["feature_group_keys"],
        tuned_C=float(lr_exp["tuned_C"]),
        tune_learning_rate=True,
        seed_role="lr_tuning_seed_and_eval_seed" if LR_SEED in SEED_LIST else "lr_tuning_seed",
        lr_source_seed=LR_SEED,
        lr_source_experiment_key=LR_TUNING_EXPERIMENT_KEY,
    )

    lr_summary = {
        "lr_seed": LR_SEED,
        "lr_seed_in_seed_list": bool(LR_SEED in SEED_LIST),
        "seed_list": SEED_LIST,
        "lr_tuning_experiment_key": LR_TUNING_EXPERIMENT_KEY,
        "global_best_lr": global_best_lr,
        "target_epsilon": TARGET_EPSILON,
        "learning_rates_initial": LEARNING_RATES_INITIAL,
        "max_endpoint_expansions": MAX_ENDPOINT_EXPANSIONS,
        "endpoint_expansion_factor": ENDPOINT_EXPANSION_FACTOR,
        "selection_metric": SELECTION_METRIC,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "site_aware_n_splits": SITE_AWARE_N_SPLITS,
        "fixed_split_created_with_seed": LR_SEED,
        "fixed_split_output_dir": str(OUTPUT_DIR),
        "epsilon_output_dir": str(EPSILON_OUTPUT_DIR),
        "use_preprocess_dump": bool(fixed_data.preprocess_dump_used),
        "preprocess_dump_dir": str(fixed_data.preprocess_dump_dir) if fixed_data.preprocess_dump_dir is not None else None,
        "policy": (
            "LR tuned once on LR_SEED validation split and reused for every seed. "
            "The train/validation/test split is created once using LR_SEED and reused for every seed in SEED_LIST. "
            "LR_SEED does not need to be present in SEED_LIST."
        ),
    }

    all_seed_metrics = []
    all_seed_subgroups = []
    lr_seed_metrics = []
    lr_seed_subgroups = []

    if LR_SEED in SEED_LIST:
        lr_seed_metrics.append(lr_metrics)
        all_seed_metrics.append(lr_metrics)
        if not lr_subgroups.empty:
            lr_seed_subgroups.append(lr_subgroups)
            all_seed_subgroups.append(lr_subgroups)

        for exp in EXPERIMENTS:
            if exp["experiment_key"] == LR_TUNING_EXPERIMENT_KEY:
                continue
            m, sg, _ = run_single_experiment(
                global_data=fixed_data,
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
            lr_seed_metrics.append(m)
            all_seed_metrics.append(m)
            if not sg.empty:
                lr_seed_subgroups.append(sg)
                all_seed_subgroups.append(sg)

    save_seed_combined_outputs(
        output_dir=lr_seed_dir,
        seed=LR_SEED,
        global_data=fixed_data,
        all_metrics=lr_seed_metrics if LR_SEED in SEED_LIST else [lr_metrics],
        all_subgroup_metrics=(
            lr_seed_subgroups
            if LR_SEED in SEED_LIST
            else ([lr_subgroups] if not lr_subgroups.empty else [])
        ),
        global_best_lr=global_best_lr,
    )

    # Evaluation seeds use the fixed split from LR_SEED, but each seed controls
    # the DP noise stream through make_lr_seed(seed, lr) and final=True variants.
    for seed in SEED_LIST:
        seed = int(seed)
        if seed == LR_SEED:
            continue

        print("\n" + "=" * 120)
        print(f"RUNNING FIXED-SPLIT FIXED-LR EVALUATION SEED: {seed}")
        print("=" * 120)

        seed_dir = get_seed_output_dir(seed)
        gd = clone_prepared_global_data_for_seed(
            fixed_data,
            seed=seed,
            output_dir=seed_dir,
        )
        seed_metrics = []
        seed_subgroups = []

        for exp in EXPERIMENTS:
            m, sg, _ = run_single_experiment(
                global_data=gd,
                experiment_key=exp["experiment_key"],
                experiment_name=exp["experiment_name"],
                selected_group_keys=exp["feature_group_keys"],
                tuned_C=float(exp["tuned_C"]),
                tune_learning_rate=False,
                fixed_best_lr=global_best_lr,
                seed_role="eval_seed_fixed_lr_fixed_split",
                lr_source_seed=LR_SEED,
                lr_source_experiment_key=LR_TUNING_EXPERIMENT_KEY,
            )
            seed_metrics.append(m)
            all_seed_metrics.append(m)
            if not sg.empty:
                seed_subgroups.append(sg)
                all_seed_subgroups.append(sg)

        save_seed_combined_outputs(
            output_dir=seed_dir,
            seed=seed,
            global_data=gd,
            all_metrics=seed_metrics,
            all_subgroup_metrics=seed_subgroups,
            global_best_lr=global_best_lr,
        )

    save_cross_seed_outputs(
        epsilon_output_dir=EPSILON_OUTPUT_DIR,
        all_seed_metrics=all_seed_metrics,
        all_seed_subgroup_metrics=all_seed_subgroups,
        global_best_lr=global_best_lr,
        lr_tuning_summary=lr_summary,
    )

    print("\nFixed split artifacts were saved under:")
    print(OUTPUT_DIR)
    print(EPSILON_OUTPUT_DIR)


if __name__ == "__main__":
    main()