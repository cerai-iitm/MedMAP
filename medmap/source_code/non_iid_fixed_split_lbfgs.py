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

# Non-IID Metrics for Public (Logistic Reg L-BFGS)
from pathlib import Path
from typing import Optional
from itertools import combinations
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


# =========================================================
# Configuration
# =========================================================
RANDOM_SEEDS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
GLOBAL_C_TUNING_SEED = 21
GLOBAL_C_TUNING_EXPERIMENT_KEY = "fmri_plus_sensitive_without_anat_func"

MAX_ITER = 500
TARGET_COL = "DX_GROUP"

# Non-IID SITE_ID-aware output directory.
OUTPUT_DIR = Path("/storage/common/datasets/ABIDE_PCP_full/outputs/non_iid_fixed_split/non_priv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path("/mnt/data/Phenotypic_V1_0b_preprocessed1.csv")

FEATURE_JSON_CANDIDATES = [
    Path("/mnt/data/recommended_feature_lists.json"),
    Path("/storage/common/datasets/ABIDE_PCP_full/abide_feature_selection_outputs/recommended_feature_lists.json"),
]

ALLOWED_FEATURE_GROUP_KEYS = {
    "recommended_initial_numeric_or_binary",
    "anat_func_numeric_or_binary",
    "recommended_initial_categorical",
    "optional_categorical_with_caution",
    "drop_leakage",
}

# SITE_ID is NOT used as an input feature. It is used only for grouping/splitting
# and for subgroup reporting.
USE_SITE_ID_AS_FEATURE = False
SITE_AWARE_SPLIT = True
SITE_AWARE_N_SPLITS = 10

# Kept same as the IID code: train ~= 70%, validation ~= 10%, test ~= 20%.
# With SITE-aware grouping, exact fractions may not be possible because complete
# SITE_ID groups must stay together. The split scorer chooses the closest valid
# SITE_ID-disjoint split among StratifiedGroupKFold folds.
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20

LEARNING_CURVE_FRACTIONS = np.linspace(0.2, 1.0, 8)
LEARNING_CURVE_REPEATS = 5

FC_IS_CORRELATION = True

PHENOTYPE_OTHER_NUMERIC_MODE = "minmax_minus1_1_zero_fill"

C_GRID = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0]


# =========================================================
# Global C-selection policy
# =========================================================
# Choose which seed should be used for:
#   1. creating the single fixed train/validation/test split, and
#   2. the one-time C hyperparameter tuning on the source experiment.
#
# This seed does NOT need to be present in RANDOM_SEEDS.
# If it is present, its source tuning run is also included as that seed's
# experiment result. If it is absent, it is used only for split creation and
# C selection, while final reported multi-seed results are computed over
# RANDOM_SEEDS only.

C_SELECTION_MODE_TUNE_SOURCE = "tune_source_seed_sensitive_without_anat_func"
C_SELECTION_MODE_REUSE_GLOBAL = "reuse_global_C_from_source_seed"

# Preserve the original IID code's final training policy:
# tune/select C using train/validation, then refit a fresh final model on
# train+validation and evaluate once on test.
REFIT_FINAL_ON_TRAIN_VAL = True


# =========================================================
# Two feature-combination experiments
# Removed: fmri_plus_sensitive_with_anat_func
# =========================================================
EXPERIMENTS = [
    {
        "experiment_key": "fmri_only",
        "experiment_name": "fMRI only",
        "feature_group_keys": [],
    },
    {
        "experiment_key": "fmri_plus_sensitive_without_anat_func",
        "experiment_name": "fMRI + sensitive info without anat & func",
        "feature_group_keys": [
            "recommended_initial_numeric_or_binary",
            "recommended_initial_categorical",
            "optional_categorical_with_caution",
        ],
    },
]


# =========================================================
# Subgroup / per-feature evaluation configuration
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
sns.set_theme(style="whitegrid")


# =========================================================
# Domain-knowledge bounds for phenotype scaling
# =========================================================
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

ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES = {
    "standardize_zero_fill",
    "minmax_minus1_1_zero_fill",
    "zero_fill_only",
}


# =========================================================
# JSON helper
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


# =========================================================
# Custom transformers: no imputer, zero-fill after scaling
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

    If a feature is constant or has invalid bounds on the train split,
    observed values map to 0.
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
# Helpers: loading and cleaning
# =========================================================
def load_abide_dataframe(csv_path: Path) -> pd.DataFrame:
    try:
        df = abide.phenotypic.copy()  # noqa: F821
        print("Using abide.phenotypic from current session.")
    except Exception:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataframe from CSV: {csv_path}")

    return df


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


def get_site_groups(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return cleaned SITE_ID groups for SITE-aware Non-IID splitting.

    SITE_ID is not used as an input feature when USE_SITE_ID_AS_FEATURE=False.
    It is only used as a grouping variable and subgroup-reporting variable.
    """
    if "SITE_ID" not in df.columns:
        return None

    groups = (
        df["SITE_ID"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-9999": pd.NA})
        .fillna("__MISSING_SITE__")
    )

    return groups.astype(str)


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


# =========================================================
# Helpers: normalization of raw categorical values
# =========================================================
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
# Helpers: phenotype feature selection from JSON keys
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
# Helpers: deterministic domain scaling
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
# Helpers: combine FC + phenotype
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
    print(f"Sample of combined dataframe:\n{X_combined.head()}")

    return X_combined, fc_colnames


# =========================================================
# Helpers: preprocessing + pipeline
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

        transformers.append(("fc", fc_pipe, fc_cols))

    if phenotype_domain_scaled_cols:
        phen_domain_pipe = Pipeline(
            steps=[
                ("zero_fill", ZeroFillTransformer()),
            ]
        )
        transformers.append(("phen_domain", phen_domain_pipe, phenotype_domain_scaled_cols))

    if phenotype_other_numeric_cols:
        phen_other_num_pipe = build_other_numeric_transformer(phenotype_other_numeric_mode)
        transformers.append(("phen_other_num", phen_other_num_pipe, phenotype_other_numeric_cols))

    if phenotype_categorical_cols:
        phen_cat_pipe = Pipeline(
            steps=[
                ("onehot_missing_zero", MissingAsAllZeroOneHotEncoder()),
            ]
        )
        transformers.append(("phen_cat", phen_cat_pipe, phenotype_categorical_cols))

    if not transformers:
        raise ValueError("No feature blocks were provided.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def make_logistic_pipeline(
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
    fc_cols: list[str],
    seed: int,
    C: float,
    fc_is_correlation: bool = True,
    phenotype_other_numeric_mode: str = "standardize_zero_fill",
) -> Pipeline:
    preprocessor = make_preprocessor(
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
        fc_is_correlation=fc_is_correlation,
        phenotype_other_numeric_mode=phenotype_other_numeric_mode,
    )

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=C,
        class_weight=None,
        fit_intercept=True,
        tol=1e-4,
        max_iter=MAX_ITER,
        random_state=seed,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe


# =========================================================
# Helpers: Non-IID SITE-aware train/validation/test split
# =========================================================
def _class_count_dict(y_split: np.ndarray) -> dict:
    return {
        str(k): int(v)
        for k, v in pd.Series(y_split).value_counts().sort_index().to_dict().items()
    }


def _site_count_dict(groups_split: Optional[pd.Series]) -> dict:
    if groups_split is None:
        return {}

    return {
        str(k): int(v)
        for k, v in groups_split.value_counts().sort_index().to_dict().items()
    }


def validate_site_disjointness(
    groups: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    train_sites = set(groups.iloc[train_idx].astype(str).tolist())
    val_sites = set(groups.iloc[val_idx].astype(str).tolist())
    test_sites = set(groups.iloc[test_idx].astype(str).tolist())

    train_val_overlap = sorted(train_sites & val_sites)
    train_test_overlap = sorted(train_sites & test_sites)
    val_test_overlap = sorted(val_sites & test_sites)

    audit = {
        "train_unique_site_count": int(len(train_sites)),
        "validation_unique_site_count": int(len(val_sites)),
        "test_unique_site_count": int(len(test_sites)),
        "train_val_site_overlap_count": int(len(train_val_overlap)),
        "train_test_site_overlap_count": int(len(train_test_overlap)),
        "validation_test_site_overlap_count": int(len(val_test_overlap)),
        "train_val_site_overlap": train_val_overlap,
        "train_test_site_overlap": train_test_overlap,
        "validation_test_site_overlap": val_test_overlap,
    }

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "SITE_ID leakage detected across train/validation/test splits. "
            f"Overlap audit:\n{json.dumps(audit, indent=2)}"
        )

    return audit


def build_split_audit(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: Optional[pd.Series] = None,
) -> dict:
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
            groups=groups,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        audit["train"]["site_counts"] = _site_count_dict(groups.iloc[train_idx])
        audit["validation"]["site_counts"] = _site_count_dict(groups.iloc[val_idx])
        audit["test"]["site_counts"] = _site_count_dict(groups.iloc[test_idx])

    return audit


def _site_aware_split_score(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    train_size: float,
    validation_size: float,
    test_size: float,
) -> float:
    """
    Lower is better.

    The score balances:
      1. closeness to target train/val/test sizes,
      2. closeness to overall class prevalence,
      3. avoiding single-class validation/test splits.
    """
    n = len(y)
    overall_pos_rate = float(np.mean(y == 1))

    split_specs = [
        ("train", train_idx, train_size),
        ("validation", val_idx, validation_size),
        ("test", test_idx, test_size),
    ]

    score = 0.0

    for split_name, idx, target_frac in split_specs:
        actual_frac = len(idx) / n
        score += 5.0 * abs(actual_frac - target_frac)

        if len(idx) == 0:
            score += 1e6
            continue

        pos_rate = float(np.mean(y[idx] == 1))
        score += abs(pos_rate - overall_pos_rate)

        # Validation/test must contain both classes for ROC-AUC and meaningful tuning.
        # Train should also contain both classes for logistic regression.
        if len(np.unique(y[idx])) < 2:
            score += 1e4

        # Extra penalty for one-sample validation/test sites if they happen.
        if split_name in {"validation", "test"} and len(idx) < 2:
            score += 1e4

    return float(score)


def _num_folds_for_fraction(fraction: float, n_splits: int, name: str) -> int:
    raw = fraction * n_splits
    rounded = int(round(raw))

    if not np.isclose(raw, rounded, atol=1e-8):
        raise ValueError(
            f"{name}={fraction} is not compatible with SITE_AWARE_N_SPLITS={n_splits}. "
            f"fraction * n_splits = {raw}, which is not an integer. "
            "Use fractions like 0.10, 0.20 with n_splits=10."
        )

    if rounded <= 0:
        raise ValueError(f"{name} must correspond to at least one fold.")

    return rounded


def make_site_aware_train_val_test_split(
    y: np.ndarray,
    groups: pd.Series,
    seed: int,
    validation_size: float,
    test_size: float,
    n_splits: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Non-IID SITE-aware split.

    Generalized fold-combination logic:
      - Build n_splits StratifiedGroupKFold folds.
      - Choose validation_size * n_splits folds as validation.
      - Choose test_size * n_splits disjoint folds as test.
      - Remaining folds become train.

    For the current config VALIDATION_SIZE=0.10, TEST_SIZE=0.20,
    SITE_AWARE_N_SPLITS=10, this gives approximately 70/10/20.

    This guarantees that the same SITE_ID does not appear in more than one split.
    Exact fractions may not hold because complete SITE_ID groups are indivisible.
    """
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")

    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    if groups is None:
        raise ValueError("SITE-aware split requested, but groups=None.")

    if len(groups) != len(y):
        raise ValueError(
            f"groups length mismatch: len(groups)={len(groups)}, len(y)={len(y)}"
        )

    n_unique_sites = int(groups.nunique())
    if n_unique_sites < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique SITE_ID groups for SITE-aware splitting. "
            f"Found only {n_unique_sites} unique groups."
        )

    val_fold_count = _num_folds_for_fraction(validation_size, n_splits, "validation_size")
    test_fold_count = _num_folds_for_fraction(test_size, n_splits, "test_size")

    if val_fold_count + test_fold_count >= n_splits:
        raise ValueError(
            "validation folds + test folds must be less than n_splits so that train has folds."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    X_dummy = np.zeros(len(y))
    folds = []
    for _, fold_idx in splitter.split(X_dummy, y, groups=groups):
        folds.append(np.asarray(fold_idx, dtype=int))

    train_size = 1.0 - validation_size - test_size

    best_score = math.inf
    best_tuple = None

    fold_ids = list(range(n_splits))

    for val_fold_ids in combinations(fold_ids, val_fold_count):
        val_fold_set = set(val_fold_ids)
        remaining_after_val = [k for k in fold_ids if k not in val_fold_set]

        for test_fold_ids in combinations(remaining_after_val, test_fold_count):
            test_fold_set = set(test_fold_ids)
            train_fold_ids = [k for k in fold_ids if k not in val_fold_set and k not in test_fold_set]

            val_idx = np.concatenate([folds[k] for k in val_fold_ids])
            test_idx = np.concatenate([folds[k] for k in test_fold_ids])
            train_idx = np.concatenate([folds[k] for k in train_fold_ids])

            score = _site_aware_split_score(
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
            )

            if score < best_score:
                best_score = score
                best_tuple = (
                    np.sort(train_idx),
                    np.sort(val_idx),
                    np.sort(test_idx),
                    tuple(int(k) for k in val_fold_ids),
                    tuple(int(k) for k in test_fold_ids),
                    tuple(int(k) for k in train_fold_ids),
                )

    if best_tuple is None:
        raise RuntimeError("Could not create SITE-aware train/validation/test split.")

    train_idx, val_idx, test_idx, val_fold_ids, test_fold_ids, train_fold_ids = best_tuple

    split_name = (
        "StratifiedGroupKFold SITE_ID-aware train/validation/test split "
        "(Non-IID, SITE_ID-disjoint): "
        f"target train={train_size:.2f}, validation={validation_size:.2f}, test={test_size:.2f}; "
        f"n_splits={n_splits}"
    )

    split_audit = build_split_audit(
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        groups=groups,
    )

    split_audit["selected_train_fold_ids"] = list(train_fold_ids)
    split_audit["selected_validation_fold_ids"] = list(val_fold_ids)
    split_audit["selected_test_fold_ids"] = list(test_fold_ids)
    split_audit["split_selection_score"] = float(best_score)
    split_audit["site_aware_n_splits"] = int(n_splits)

    return train_idx, val_idx, test_idx, split_name, split_audit


def make_iid_train_val_test_split(
    y: np.ndarray,
    seed: int,
    validation_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Pure IID stratified splitting. Kept as fallback only.
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

    train_idx = np.sort(train_val_idx[train_rel_idx])
    val_idx = np.sort(train_val_idx[val_rel_idx])
    test_idx = np.sort(test_idx)

    split_name = (
        "StratifiedShuffleSplit train/validation/test "
        f"(IID, no SITE_ID grouping): "
        f"train={1.0 - validation_size - test_size:.2f}, "
        f"validation={validation_size:.2f}, test={test_size:.2f}"
    )

    split_audit = build_split_audit(
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        groups=None,
    )

    return train_idx, val_idx, test_idx, split_name, split_audit


def make_train_val_test_split(
    y: np.ndarray,
    seed: int,
    validation_size: float,
    test_size: float,
    groups: Optional[pd.Series] = None,
    site_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    if site_aware:
        return make_site_aware_train_val_test_split(
            y=y,
            groups=groups,
            seed=seed,
            validation_size=validation_size,
            test_size=test_size,
            n_splits=SITE_AWARE_N_SPLITS,
        )

    return make_iid_train_val_test_split(
        y=y,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
    )


# =========================================================
# Helpers: metrics
# =========================================================
def get_positive_class_probability(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(X)
    class_index = list(estimator.named_steps["model"].classes_).index(1)
    return proba[:, class_index]


def evaluate_binary_classifier(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_pred = estimator.predict(X)
    y_proba = get_positive_class_probability(estimator, X)

    metrics = compute_prediction_metrics_from_arrays(y, y_pred, y_proba)
    compact_metrics = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "log_loss": metrics["log_loss"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
    }

    return compact_metrics, y_pred, y_proba


def print_metric_block(title: str, metrics: dict) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for k, v in metrics.items():
        if pd.isna(v):
            print(f"{k:>32s}: NaN")
        else:
            print(f"{k:>32s}: {v:.6f}")


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
        "log_loss": safe_scalar_metric(log_loss, y_true, y_proba, labels=[0, 1]),
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


def print_classification_report_safe(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    try:
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Control (DX_GROUP=2)", "Autism (DX_GROUP=1)"],
                digits=4,
                zero_division=0,
            )
        )
    except Exception as exc:
        print(f"Could not compute classification report: {exc}")


# =========================================================
# Helpers: C tuning / fixed-C fitting
# =========================================================
def make_train_val_result_row(
    *,
    C: float,
    train_metrics: dict,
    val_metrics: dict,
) -> dict:
    row = {
        "C": float(C),
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": None,
        "fit_intercept": True,
        "tol": 1e-4,
        "max_iter": MAX_ITER,
    }

    for k, v in train_metrics.items():
        row[f"train_{k}"] = v

    for k, v in val_metrics.items():
        row[f"validation_{k}"] = v

    return row


def tune_c_on_validation(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
    fc_cols: list[str],
    seed: int,
    c_grid: list[float],
    fc_is_correlation: bool,
    phenotype_other_numeric_mode: str,
) -> tuple[float, Pipeline, pd.DataFrame]:
    """
    Full C-grid tuning.

    In this workflow, this is called only once for the selected global tuning seed
    and source experiment. The selected C is reused for all other seeds/experiments.
    """
    rows = []
    fitted_models = {}

    for C in c_grid:
        estimator = make_logistic_pipeline(
            phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
            phenotype_other_numeric_cols=phenotype_other_numeric_cols,
            phenotype_categorical_cols=phenotype_categorical_cols,
            fc_cols=fc_cols,
            seed=seed,
            C=C,
            fc_is_correlation=fc_is_correlation,
            phenotype_other_numeric_mode=phenotype_other_numeric_mode,
        )

        estimator.fit(X_train, y_train)

        train_metrics, _, _ = evaluate_binary_classifier(estimator, X_train, y_train)
        val_metrics, _, _ = evaluate_binary_classifier(estimator, X_val, y_val)

        fitted_models[float(C)] = estimator

        rows.append(
            make_train_val_result_row(
                C=float(C),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        )

    tuning_df = pd.DataFrame(rows)

    tuning_df = tuning_df.sort_values(
        by=[
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_roc_auc",
            "validation_log_loss",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    tuning_df["rank_validation_accuracy"] = np.arange(1, len(tuning_df) + 1)

    best_C = float(tuning_df.loc[0, "C"])
    best_model = fitted_models[best_C]

    return best_C, best_model, tuning_df


def fit_fixed_c_on_train_and_evaluate_validation(
    *,
    fixed_C: float,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    phenotype_domain_scaled_cols: list[str],
    phenotype_other_numeric_cols: list[str],
    phenotype_categorical_cols: list[str],
    fc_cols: list[str],
    seed: int,
    fc_is_correlation: bool,
    phenotype_other_numeric_mode: str,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Train exactly one model using a previously selected global C.

    This preserves the train/validation output structure but does not perform
    another C-grid search.
    """
    estimator = make_logistic_pipeline(
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
        seed=seed,
        C=float(fixed_C),
        fc_is_correlation=fc_is_correlation,
        phenotype_other_numeric_mode=phenotype_other_numeric_mode,
    )

    estimator.fit(X_train, y_train)

    train_metrics, _, _ = evaluate_binary_classifier(estimator, X_train, y_train)
    val_metrics, _, _ = evaluate_binary_classifier(estimator, X_val, y_val)

    row = make_train_val_result_row(
        C=float(fixed_C),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    tuning_like_df = pd.DataFrame([row])
    tuning_like_df["rank_validation_accuracy"] = 1

    return estimator, tuning_like_df


# =========================================================
# Helpers: subgroup reporting
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
    best_C: float,
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
                    "best_C": best_C,
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


def print_subgroup_metric_tables(combined_df: pd.DataFrame, experiment_name: str) -> None:
    if combined_df is None or combined_df.empty:
        print("\nNo subgroup metrics available to print.")
        return

    print("\n" + "=" * 120)
    print(f"Subgroup metrics for: {experiment_name}")
    print("=" * 120)

    for feature_name in SUBGROUP_EVAL_FEATURES:
        feature_df = combined_df.loc[combined_df["feature"] == feature_name]

        if feature_df.empty:
            continue

        print("\n" + "-" * 120)
        print(f"Subgroup metrics for {feature_name}")
        print("-" * 120)

        columns_to_print = [
            "split",
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

        print(feature_df[columns_to_print].to_string(index=False))


# =========================================================
# Helpers: learning curves
# =========================================================
def build_holdout_learning_curves(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    fractions: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    n_train_total = len(X_train)

    for frac in fractions:
        for rep in range(repeats):
            if np.isclose(frac, 1.0):
                subset_idx = np.arange(n_train_total)
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=frac,
                    random_state=seed + 1000 * rep + int(frac * 100),
                )
                subset_idx, _ = next(
                    sss.split(np.zeros(n_train_total), y_train)
                )

            X_sub = X_train.iloc[subset_idx].copy()
            y_sub = y_train[subset_idx].copy()

            est = clone(estimator)
            est.fit(X_sub, y_sub)

            train_metrics, _, _ = evaluate_binary_classifier(est, X_sub, y_sub)
            val_metrics, _, _ = evaluate_binary_classifier(est, X_val, y_val)

            rows.append(
                {
                    "fraction": float(frac),
                    "n_samples": int(len(X_sub)),
                    "repeat": int(rep),
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "train_log_loss": train_metrics["log_loss"],
                    "val_log_loss": val_metrics["log_loss"],
                }
            )

    return pd.DataFrame(rows)


def aggregate_learning_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        curve_df.groupby(["fraction", "n_samples"], as_index=False)
        .agg(
            train_accuracy_mean=("train_accuracy", "mean"),
            train_accuracy_std=("train_accuracy", "std"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            train_log_loss_mean=("train_log_loss", "mean"),
            train_log_loss_std=("train_log_loss", "std"),
            val_log_loss_mean=("val_log_loss", "mean"),
            val_log_loss_std=("val_log_loss", "std"),
        )
        .sort_values("n_samples")
        .reset_index(drop=True)
    )

    return agg


# =========================================================
# Plot helpers
# =========================================================
def save_c_tuning_plot(tuning_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = tuning_df.sort_values("C").copy()

    plt.figure(figsize=(8, 6))
    plt.semilogx(plot_df["C"], plot_df["train_accuracy"], marker="o", label="Train accuracy")
    plt.semilogx(plot_df["C"], plot_df["validation_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Train vs validation accuracy for C tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "c_tuning_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.semilogx(plot_df["C"], plot_df["train_log_loss"], marker="o", label="Train log loss")
    plt.semilogx(plot_df["C"], plot_df["validation_log_loss"], marker="o", label="Validation log loss")
    plt.xlabel("C")
    plt.ylabel("Log loss")
    plt.title("Train vs validation log loss for C tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "c_tuning_log_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    top = tuning_df.nsmallest(15, "rank_validation_accuracy").copy()
    top["label"] = top.apply(
        lambda row: (
            f"lbfgs | l2 | C={row['C']} | "
            f"fit_int=True | tol=1e-4 | max_iter={MAX_ITER}"
        ),
        axis=1,
    )
    top = top.sort_values("validation_accuracy", ascending=True)

    plt.figure(figsize=(13, 8))
    plt.barh(top["label"], top["validation_accuracy"])
    plt.xlabel("Validation accuracy")
    plt.ylabel("Configuration")
    plt.title("Top C configurations by validation accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "top_c_configurations_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


def save_learning_curve_plots(curve_agg: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(
        curve_agg["n_samples"],
        curve_agg["train_accuracy_mean"],
        marker="o",
        label="Train accuracy",
    )
    plt.plot(
        curve_agg["n_samples"],
        curve_agg["val_accuracy_mean"],
        marker="o",
        label="Validation accuracy",
    )
    plt.fill_between(
        curve_agg["n_samples"],
        curve_agg["train_accuracy_mean"] - curve_agg["train_accuracy_std"].fillna(0),
        curve_agg["train_accuracy_mean"] + curve_agg["train_accuracy_std"].fillna(0),
        alpha=0.2,
    )
    plt.fill_between(
        curve_agg["n_samples"],
        curve_agg["val_accuracy_mean"] - curve_agg["val_accuracy_std"].fillna(0),
        curve_agg["val_accuracy_mean"] + curve_agg["val_accuracy_std"].fillna(0),
        alpha=0.2,
    )
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Train vs validation accuracy curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(
        curve_agg["n_samples"],
        curve_agg["train_log_loss_mean"],
        marker="o",
        label="Train log loss",
    )
    plt.plot(
        curve_agg["n_samples"],
        curve_agg["val_log_loss_mean"],
        marker="o",
        label="Validation log loss",
    )
    plt.fill_between(
        curve_agg["n_samples"],
        curve_agg["train_log_loss_mean"] - curve_agg["train_log_loss_std"].fillna(0),
        curve_agg["train_log_loss_mean"] + curve_agg["train_log_loss_std"].fillna(0),
        alpha=0.2,
    )
    plt.fill_between(
        curve_agg["n_samples"],
        curve_agg["val_log_loss_mean"] - curve_agg["val_log_loss_std"].fillna(0),
        curve_agg["val_log_loss_mean"] + curve_agg["val_log_loss_std"].fillna(0),
        alpha=0.2,
    )
    plt.xlabel("Number of training samples")
    plt.ylabel("Log loss")
    plt.title("Train vs validation loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_log_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


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
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{split_label.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


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
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{split_label.lower()}_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"Average precision = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_label} precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{split_label.lower()}_pr_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


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
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{split_label.lower()}_calibration_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


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
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{split_label.lower()}_probability_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def save_top_coefficients_plot(
    best_model: Pipeline,
    out_dir: Path,
    top_k: int = 20,
) -> pd.DataFrame:
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    coefs = best_model.named_steps["model"].coef_.ravel()

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
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
    coef_df.to_csv(out_dir / "model_coefficients_all.csv", index=False)

    top_pos = coef_df.sort_values("coefficient", ascending=False).head(top_k)
    top_neg = coef_df.sort_values("coefficient", ascending=True).head(top_k)
    plot_df = pd.concat([top_neg, top_pos], axis=0)

    plt.figure(figsize=(13, 9))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title("Most influential logistic-regression coefficients")
    plt.tight_layout()
    plt.savefig(out_dir / "top_logistic_coefficients.png", dpi=300, bbox_inches="tight")
    plt.show()

    modality_importance = (
        coef_df.groupby("modality", as_index=False)["abs_coefficient"]
        .sum()
        .sort_values("abs_coefficient", ascending=False)
    )

    modality_importance.to_csv(out_dir / "coefficient_modality_importance.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(modality_importance["modality"], modality_importance["abs_coefficient"])
    plt.xlabel("Feature block")
    plt.ylabel("Sum of absolute coefficients")
    plt.title("Coefficient importance by feature block")
    plt.tight_layout()
    plt.savefig(
        out_dir / "coefficient_modality_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return coef_df


# =========================================================
# Experiment utilities
# =========================================================
def get_experiment_by_key(experiments: list[dict], experiment_key: str) -> dict:
    matches = [exp for exp in experiments if exp["experiment_key"] == experiment_key]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one experiment with key={experiment_key!r}, "
            f"found {len(matches)}."
        )

    return matches[0]


def append_result(
    *,
    metrics_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    all_metrics: list[pd.DataFrame],
    all_subgroup_metrics: list[pd.DataFrame],
) -> None:
    all_metrics.append(metrics_df)

    if subgroup_df is not None and not subgroup_df.empty:
        all_subgroup_metrics.append(subgroup_df)


# =========================================================
# Helpers: aggregate metrics across seeds
# =========================================================
def aggregate_metrics_across_seeds(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
    seed_col: str = "seed",
) -> pd.DataFrame:
    """
    Aggregate metrics across seeds with mean, sample standard deviation,
    standard error of the mean, and the number of non-missing seed values.

    For each metric m, the output columns are:
        m_mean, m_std, m_sem, m_n

    Notes:
        - std uses sample std, i.e. ddof=1.
        - sem = std / sqrt(n), where n is the non-missing count for that metric.
        - If n <= 1, std/sem are NaN.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(group_cols))

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing group columns for aggregation: {missing_group_cols}")

    available_metric_cols = [c for c in metric_cols if c in df.columns]
    if not available_metric_cols:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    work_df = df.copy()

    for col in available_metric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    rows = []

    for group_values, group_df in work_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {col: value for col, value in zip(group_cols, group_values)}

        if seed_col in group_df.columns:
            row["n_seeds"] = int(group_df[seed_col].nunique())

        for metric in available_metric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            n = int(values.shape[0])
            mean = float(values.mean()) if n > 0 else np.nan
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            sem = float(std / np.sqrt(n)) if n > 1 else np.nan

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_n"] = n

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# =========================================================
# One experiment runner
# =========================================================
def run_single_experiment(
    *,
    experiment_key: str,
    experiment_name: str,
    selected_group_keys: list[str],
    df: pd.DataFrame,
    df_model_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    y: np.ndarray,
    feature_groups: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    output_root: Path,
    seed: int,
    c_selection_mode: str,
    fixed_C: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    experiment_dir = output_root / experiment_key
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if c_selection_mode not in {
        C_SELECTION_MODE_TUNE_SOURCE,
        C_SELECTION_MODE_REUSE_GLOBAL,
    }:
        raise ValueError(f"Invalid c_selection_mode={c_selection_mode!r}")

    if c_selection_mode == C_SELECTION_MODE_REUSE_GLOBAL and fixed_C is None:
        raise ValueError("fixed_C must be provided when reusing global C.")

    print("\n" + "#" * 120)
    print(f"Running experiment: {experiment_name}")
    print("#" * 120)

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

    print("\nPhenotype numeric columns with deterministic domain scaling + zero fill:")
    print(phenotype_domain_scaled_cols)

    print("\nPhenotype other numeric columns:")
    print(phenotype_other_numeric_cols)
    print("Processing mode for phenotype_other_numeric_cols:", PHENOTYPE_OTHER_NUMERIC_MODE)

    print("\nPhenotype categorical columns with missing/unseen -> all-zero one-hot block:")
    print(phenotype_categorical_cols)

    X, fc_cols = build_combined_dataframe(
        df_model=df_model,
        X_conn_model=X_conn_model,
        phenotype_selected_cols=selected_phenotype_cols,
    )

    print("\nCombined feature dataframe shape:", X.shape)
    print("Number of FC features:", len(fc_cols))
    print("Number of selected phenotype raw columns:", len(selected_phenotype_cols))

    train_val_idx = np.concatenate([train_idx, val_idx])

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    X_train_val = X.iloc[train_val_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    y_train_val = y[train_val_idx]

    subgroup_eval_features_present = [
        c for c in SUBGROUP_EVAL_FEATURES if c in df_model_raw.columns
    ]

    df_subgroup_train = (
        df_model_raw.iloc[train_idx][subgroup_eval_features_present]
        .reset_index(drop=True)
    )
    df_subgroup_val = (
        df_model_raw.iloc[val_idx][subgroup_eval_features_present]
        .reset_index(drop=True)
    )
    df_subgroup_test = (
        df_model_raw.iloc[test_idx][subgroup_eval_features_present]
        .reset_index(drop=True)
    )
    df_subgroup_train_val = (
        df_model_raw.iloc[train_val_idx][subgroup_eval_features_present]
        .reset_index(drop=True)
    )

    print("\nSplit strategy:")
    print(split_name)
    print(f"Train size      : {len(X_train)}")
    print(f"Validation size : {len(X_val)}")
    print(f"Test size       : {len(X_test)}")
    print(f"Train class counts      : {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"Validation class counts : {dict(pd.Series(y_val).value_counts().sort_index())}")
    print(f"Test class counts       : {dict(pd.Series(y_test).value_counts().sort_index())}")

    if c_selection_mode == C_SELECTION_MODE_TUNE_SOURCE:
        print("\nC-selection policy:")
        print(
            "Running full C-grid tuning only for the source experiment "
            f"(seed={seed}, experiment_key={experiment_key})."
        )

        best_C, selected_model, tuning_df = tune_c_on_validation(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
            phenotype_other_numeric_cols=phenotype_other_numeric_cols,
            phenotype_categorical_cols=phenotype_categorical_cols,
            fc_cols=fc_cols,
            seed=seed,
            c_grid=C_GRID,
            fc_is_correlation=FC_IS_CORRELATION,
            phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
        )

        c_source_description = (
            f"selected by full validation tuning on seed={seed}, "
            f"experiment_key={experiment_key}"
        )

    else:
        best_C = float(fixed_C)

        print("\nC-selection policy:")
        print(
            "Reusing global fixed C selected from "
            f"seed={GLOBAL_C_TUNING_SEED}, "
            f"experiment_key={GLOBAL_C_TUNING_EXPERIMENT_KEY}."
        )
        print(f"Global fixed C being used: {best_C}")

        selected_model, tuning_df = fit_fixed_c_on_train_and_evaluate_validation(
            fixed_C=best_C,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
            phenotype_other_numeric_cols=phenotype_other_numeric_cols,
            phenotype_categorical_cols=phenotype_categorical_cols,
            fc_cols=fc_cols,
            seed=seed,
            fc_is_correlation=FC_IS_CORRELATION,
            phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
        )

        c_source_description = (
            f"reused from seed={GLOBAL_C_TUNING_SEED}, "
            f"experiment_key={GLOBAL_C_TUNING_EXPERIMENT_KEY}"
        )

    tuning_df.insert(0, "seed", int(seed))
    tuning_df.insert(1, "experiment_key", experiment_key)
    tuning_df.insert(2, "experiment_name", experiment_name)
    tuning_df.insert(3, "c_selection_mode", c_selection_mode)
    tuning_df.insert(4, "c_source_description", c_source_description)

    tuning_df.to_csv(experiment_dir / "c_tuning_results_train_val.csv", index=False)

    print("\nBest C used for this experiment:")
    print(best_C)

    best_val_row = tuning_df.iloc[0].to_dict()
    save_json(best_val_row, experiment_dir / "best_validation_tuning_summary.json")

    # Evaluate selected model trained only on train split.
    train_metrics, y_train_pred, y_train_proba = evaluate_binary_classifier(
        selected_model,
        X_train,
        y_train,
    )

    val_metrics, y_val_pred, y_val_proba = evaluate_binary_classifier(
        selected_model,
        X_val,
        y_val,
    )

    print_metric_block(
        f"{experiment_name} | TRAIN metrics for selected model",
        train_metrics,
    )
    print_metric_block(
        f"{experiment_name} | VALIDATION metrics for selected model",
        val_metrics,
    )

    print_classification_report_safe(
        y_val,
        y_val_pred,
        "Validation classification report:",
    )

    # Final model policy: preserve the original IID code's train+validation refit.
    if REFIT_FINAL_ON_TRAIN_VAL:
        final_model = make_logistic_pipeline(
            phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
            phenotype_other_numeric_cols=phenotype_other_numeric_cols,
            phenotype_categorical_cols=phenotype_categorical_cols,
            fc_cols=fc_cols,
            seed=seed,
            C=best_C,
            fc_is_correlation=FC_IS_CORRELATION,
            phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
        )

        final_model.fit(X_train_val, y_train_val)
        test_model = final_model
        final_training_policy = "final_refit_on_train_plus_validation"
    else:
        final_model = selected_model
        test_model = selected_model
        final_training_policy = "test_selected_model_trained_on_train_only"

    train_val_metrics, y_train_val_pred, y_train_val_proba = evaluate_binary_classifier(
        final_model,
        X_train_val,
        y_train_val,
    )

    test_metrics, y_test_pred, y_test_proba = evaluate_binary_classifier(
        test_model,
        X_test,
        y_test,
    )

    print_metric_block(
        f"{experiment_name} | TRAIN+VALIDATION metrics for final model",
        train_val_metrics,
    )
    print_metric_block(
        f"{experiment_name} | TEST metrics for final model",
        test_metrics,
    )

    print_classification_report_safe(
        y_test,
        y_test_pred,
        "Test classification report:",
    )

    # Save metrics.
    save_json(train_metrics, experiment_dir / "selected_model_train_metrics.json")
    save_json(val_metrics, experiment_dir / "selected_model_validation_metrics.json")
    save_json(train_val_metrics, experiment_dir / "final_model_train_val_metrics.json")
    save_json(test_metrics, experiment_dir / "final_model_test_metrics.json")

    # Save confusion matrices.
    pd.DataFrame(
        confusion_matrix(y_train, y_train_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "selected_model_train_confusion_matrix.csv")

    pd.DataFrame(
        confusion_matrix(y_val, y_val_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "selected_model_validation_confusion_matrix.csv")

    pd.DataFrame(
        confusion_matrix(y_train_val, y_train_val_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "final_model_train_val_confusion_matrix.csv")

    pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=[0, 1]),
        index=["true_control", "true_autism"],
        columns=["pred_control", "pred_autism"],
    ).to_csv(experiment_dir / "final_model_test_confusion_matrix.csv")

    # Learning curves use the chosen C and the original train/validation split.
    learning_curve_estimator = make_logistic_pipeline(
        phenotype_domain_scaled_cols=phenotype_domain_scaled_cols,
        phenotype_other_numeric_cols=phenotype_other_numeric_cols,
        phenotype_categorical_cols=phenotype_categorical_cols,
        fc_cols=fc_cols,
        seed=seed,
        C=best_C,
        fc_is_correlation=FC_IS_CORRELATION,
        phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
    )

    curve_df = build_holdout_learning_curves(
        estimator=learning_curve_estimator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fractions=LEARNING_CURVE_FRACTIONS,
        repeats=LEARNING_CURVE_REPEATS,
        seed=seed,
    )

    curve_df.to_csv(experiment_dir / "holdout_learning_curve_raw.csv", index=False)

    curve_agg = aggregate_learning_curves(curve_df)
    curve_agg.to_csv(experiment_dir / "holdout_learning_curve_summary.csv", index=False)

    # Plots.
    # Kept same as original: plotting helpers exist, calls remain commented.
    # Uncomment if you want plot files generated.
    # save_c_tuning_plot(tuning_df, experiment_dir)
    # save_learning_curve_plots(curve_agg, experiment_dir)

    # save_confusion_matrix_plot(
    #     y_true=y_val,
    #     y_pred=y_val_pred,
    #     out_dir=experiment_dir,
    #     split_label="validation",
    # )
    # save_roc_pr_plots(
    #     y_true=y_val,
    #     y_proba=y_val_proba,
    #     out_dir=experiment_dir,
    #     split_label="validation",
    # )
    # save_calibration_plot(
    #     y_true=y_val,
    #     y_proba=y_val_proba,
    #     out_dir=experiment_dir,
    #     split_label="validation",
    # )
    # save_probability_histogram(
    #     y_true=y_val,
    #     y_proba=y_val_proba,
    #     out_dir=experiment_dir,
    #     split_label="validation",
    # )

    # save_confusion_matrix_plot(
    #     y_true=y_test,
    #     y_pred=y_test_pred,
    #     out_dir=experiment_dir,
    #     split_label="test",
    # )
    # save_roc_pr_plots(
    #     y_true=y_test,
    #     y_proba=y_test_proba,
    #     out_dir=experiment_dir,
    #     split_label="test",
    # )
    # save_calibration_plot(
    #     y_true=y_test,
    #     y_proba=y_test_proba,
    #     out_dir=experiment_dir,
    #     split_label="test",
    # )
    # save_probability_histogram(
    #     y_true=y_test,
    #     y_proba=y_test_proba,
    #     out_dir=experiment_dir,
    #     split_label="test",
    # )

    # coef_df = save_top_coefficients_plot(
    #     best_model=final_model,
    #     out_dir=experiment_dir,
    #     top_k=20,
    # )

    # Subgroup metrics for every split.
    subgroup_frames = []

    for split_label, df_meta, yy, yy_pred, yy_proba in [
        ("train_selected_model", df_subgroup_train, y_train, y_train_pred, y_train_proba),
        ("validation_selected_model", df_subgroup_val, y_val, y_val_pred, y_val_proba),
        (
            "train_validation_final_model",
            df_subgroup_train_val,
            y_train_val,
            y_train_val_pred,
            y_train_val_proba,
        ),
        ("test_final_model", df_subgroup_test, y_test, y_test_pred, y_test_proba),
    ]:
        split_subgroup_df = compute_subgroup_metrics(
            df_meta=df_meta,
            y_true=yy,
            y_pred=yy_pred,
            y_proba=yy_proba,
            feature_names=subgroup_eval_features_present,
            experiment_key=experiment_key,
            experiment_name=experiment_name,
            split_name=split_label,
            best_C=best_C,
        )
        if split_subgroup_df is not None and not split_subgroup_df.empty:
            split_subgroup_df.insert(0, "seed", int(seed))
            subgroup_frames.append(split_subgroup_df)

    all_split_subgroup_metrics = (
        pd.concat(subgroup_frames, axis=0, ignore_index=True)
        if subgroup_frames
        else pd.DataFrame()
    )

    test_subgroup_metrics = (
        all_split_subgroup_metrics.loc[
            all_split_subgroup_metrics["split"] == "test_final_model"
        ].reset_index(drop=True)
        if not all_split_subgroup_metrics.empty
        else pd.DataFrame()
    )

    all_split_subgroup_metrics.to_csv(
        experiment_dir / "all_splits_subgroup_metrics.csv",
        index=False,
    )
    test_subgroup_metrics.to_csv(
        experiment_dir / "test_subgroup_metrics.csv",
        index=False,
    )

    print_subgroup_metric_tables(test_subgroup_metrics, experiment_name)

    run_feature_summary = {
        "seed": int(seed),
        "experiment_key": experiment_key,
        "experiment_name": experiment_name,
        "selected_feature_group_keys": selected_group_keys,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "non_iid_definition": (
            "SITE_ID-disjoint train/validation/test split. "
            "The same SITE_ID is not allowed to appear in more than one split."
        ),
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "allowed_phenotype_other_numeric_modes": sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES),
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
        "split_strategy": split_name,
        "best_C_used": best_C,
        "c_selection_mode": c_selection_mode,
        "c_source_description": c_source_description,
        "global_c_tuning_seed": GLOBAL_C_TUNING_SEED,
        "global_c_tuning_experiment_key": GLOBAL_C_TUNING_EXPERIMENT_KEY,
        "final_training_policy": final_training_policy,
        "missing_numeric_policy": "replace NaN/inf with 0 after preprocessing",
        "missing_categorical_policy": "missing/unseen categories mapped to all-zero one-hot block",
    }

    save_json(run_feature_summary, experiment_dir / "run_feature_summary.json")

    metric_rows = []

    for split_label, metrics in [
        ("train_selected_model", train_metrics),
        ("validation_selected_model", val_metrics),
        ("train_validation_final_model", train_val_metrics),
        ("test_final_model", test_metrics),
    ]:
        metric_rows.append(
            {
                "seed": int(seed),
                "experiment_key": experiment_key,
                "experiment_name": experiment_name,
                "split": split_label,
                "best_C": best_C,
                "c_selection_mode": c_selection_mode,
                "c_source_description": c_source_description,
                "final_training_policy": final_training_policy,
                "n_fc_features": len(fc_cols),
                "n_selected_phenotype_raw_columns": len(selected_phenotype_cols),
                "selected_feature_group_keys": json.dumps(selected_group_keys),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(experiment_dir / "metrics_summary_table.csv", index=False)

    test_metrics_df = metrics_df.loc[
        metrics_df["split"] == "test_final_model"
    ].reset_index(drop=True)
    test_metrics_df.to_csv(experiment_dir / "test_metrics_summary_table.csv", index=False)

    np.save(experiment_dir / "y_train.npy", y_train)
    np.save(experiment_dir / "y_val.npy", y_val)
    np.save(experiment_dir / "y_test.npy", y_test)
    np.save(experiment_dir / "y_test_pred.npy", y_test_pred)
    np.save(experiment_dir / "y_test_proba.npy", y_test_proba)

    print("\nSaved experiment outputs in:")
    print(experiment_dir)

    print("\nMost important output files:")
    for fp in [
        experiment_dir / "c_tuning_results_train_val.csv",
        experiment_dir / "best_validation_tuning_summary.json",
        experiment_dir / "metrics_summary_table.csv",
        experiment_dir / "test_metrics_summary_table.csv",
        experiment_dir / "all_splits_subgroup_metrics.csv",
        experiment_dir / "test_subgroup_metrics.csv",
        experiment_dir / "final_model_test_metrics.json",
        experiment_dir / "holdout_learning_curve_summary.csv",
    ]:
        print(fp)

    return metrics_df, all_split_subgroup_metrics, best_C


# =========================================================
# Main workflow
# =========================================================
def save_seed_level_outputs(
    *,
    seed_output_dir: Path,
    seed_metrics: list[pd.DataFrame],
    seed_subgroup_metrics: list[pd.DataFrame],
) -> None:
    if seed_metrics:
        seed_all_metrics_df = pd.concat(seed_metrics, axis=0, ignore_index=True)
    else:
        seed_all_metrics_df = pd.DataFrame()

    seed_all_metrics_df.to_csv(seed_output_dir / "all_modes_metrics_summary_seed.csv", index=False)

    seed_test_metrics_df = (
        seed_all_metrics_df.loc[seed_all_metrics_df["split"] == "test_final_model"].reset_index(drop=True)
        if not seed_all_metrics_df.empty
        else pd.DataFrame()
    )
    seed_test_metrics_df.to_csv(seed_output_dir / "all_modes_test_metrics_comparison_seed.csv", index=False)

    if seed_subgroup_metrics:
        seed_all_subgroup_df = pd.concat(seed_subgroup_metrics, axis=0, ignore_index=True)
    else:
        seed_all_subgroup_df = pd.DataFrame()

    seed_all_subgroup_df.to_csv(
        seed_output_dir / "all_modes_all_splits_subgroup_metrics_seed.csv",
        index=False,
    )

    seed_test_subgroup_df = (
        seed_all_subgroup_df.loc[seed_all_subgroup_df["split"] == "test_final_model"].reset_index(drop=True)
        if not seed_all_subgroup_df.empty
        else pd.DataFrame()
    )
    seed_test_subgroup_df.to_csv(
        seed_output_dir / "all_modes_test_subgroup_metrics_seed.csv",
        index=False,
    )


def save_indices_for_seed_dir(
    *,
    seed_output_dir: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_audit: dict,
) -> None:
    """
    Save the train/validation/test indices inside a seed-specific directory.

    In this fixed-split workflow, these files are identical for all experiment
    seeds. They are still written in each seed folder to preserve the previous
    output structure.
    """
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(seed_output_dir / "train_idx.npy", train_idx)
    np.save(seed_output_dir / "val_idx.npy", val_idx)
    np.save(seed_output_dir / "test_idx.npy", test_idx)

    save_json(split_audit, seed_output_dir / "split_audit.json")



def _split_index_map_from_arrays(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    return {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
        "train_validation": np.sort(np.concatenate([train_idx, val_idx])),
    }


def _base_metadata_frame_for_fixed_csvs(
    *,
    df_model_raw: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[pd.Series],
) -> pd.DataFrame:
    n = len(y)
    meta = pd.DataFrame(
        {
            "row_index_after_valid_mask": np.arange(n, dtype=int),
            "y_binary": np.asarray(y, dtype=int),
        }
    )

    if TARGET_COL in df_model_raw.columns:
        meta[f"{TARGET_COL}_original"] = df_model_raw[TARGET_COL].to_numpy()

    if groups is not None:
        meta["SITE_ID"] = groups.astype(str).to_numpy()
    elif "SITE_ID" in df_model_raw.columns:
        meta["SITE_ID"] = df_model_raw["SITE_ID"].astype("string").astype(str).to_numpy()

    for col in SUBGROUP_EVAL_FEATURES:
        if col in meta.columns:
            continue
        if col in df_model_raw.columns:
            meta[f"meta__{col}"] = df_model_raw[col].to_numpy()

    return meta


def build_common_fixed_prepared_dataframe(
    *,
    df_model_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    y: np.ndarray,
    groups: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Build one common prepared table after ABIDE cleaning, valid-label filtering,
    deterministic domain scaling, and fMRI NaN/inf zero filling.

    This common table is intentionally not one-hot encoded because it is shared
    across all experiment definitions. Fully numeric experiment-specific tables
    are saved separately in fixed_processed_by_experiment/<experiment_key>/.
    """
    n_samples, n_fc_features = X_conn_model.shape
    fc_cols = [f"fc_{i:05d}" for i in range(n_fc_features)]
    fc_df = pd.DataFrame(
        np.nan_to_num(
            np.asarray(X_conn_model, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        columns=fc_cols,
    )

    phenotype_df = df_model.reset_index(drop=True).copy()
    phenotype_df = phenotype_df.drop(columns=[TARGET_COL], errors="ignore")
    if not USE_SITE_ID_AS_FEATURE:
        phenotype_df = phenotype_df.drop(columns=["SITE_ID"], errors="ignore")
    phenotype_df = phenotype_df.add_prefix("phen__")

    prepared_df = pd.concat(
        [
            _base_metadata_frame_for_fixed_csvs(
                df_model_raw=df_model_raw,
                y=y,
                groups=groups,
            ),
            fc_df,
            phenotype_df,
        ],
        axis=1,
    )
    return prepared_df


def save_common_fixed_prepared_split_csvs(
    *,
    output_dir: Path,
    df_model_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    y: np.ndarray,
    groups: Optional[pd.Series],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_seed: int,
) -> None:
    """
    Save common prepared fixed split CSVs at OUTPUT_DIR:
      - all_fixed.csv
      - train_fixed.csv
      - val_fixed.csv
      - test_fixed.csv
      - train_val_fixed.csv
      - fixed_prepared_data_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_df = build_common_fixed_prepared_dataframe(
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        groups=groups,
    )
    split_map = _split_index_map_from_arrays(train_idx, val_idx, test_idx)

    split_label_by_idx = pd.Series("unassigned", index=np.arange(len(y)), dtype="object")
    split_label_by_idx.iloc[split_map["train"]] = "train"
    split_label_by_idx.iloc[split_map["validation"]] = "validation"
    split_label_by_idx.iloc[split_map["test"]] = "test"

    all_df = prepared_df.copy()
    all_df.insert(1, "split", split_label_by_idx.to_numpy())
    all_df.to_csv(output_dir / "all_fixed.csv", index=False)

    file_map = {
        "train": "train_fixed.csv",
        "validation": "val_fixed.csv",
        "test": "test_fixed.csv",
        "train_validation": "train_val_fixed.csv",
    }
    for split_name, filename in file_map.items():
        idx = split_map[split_name]
        split_df = prepared_df.iloc[idx].copy().reset_index(drop=True)
        split_df.insert(1, "split", split_name)
        split_df.to_csv(output_dir / filename, index=False)

    save_json(
        {
            "fixed_split_created_with_seed": int(split_seed),
            "target_column": "y_binary",
            "index_space": (
                "row_index_after_valid_mask refers to df_model_raw / df_model / "
                "X_conn_model after valid-mask filtering and reset_index(drop=True)."
            ),
            "csv_policy": (
                "These common CSV files contain the prepared base table after "
                "ABIDE loading, cleaning, valid-label filtering, deterministic "
                "domain scaling, and fMRI zero-filling. They avoid rebuilding "
                "the base data preparation pipeline in downstream jobs."
            ),
            "site_id_policy": (
                "SITE_ID is included only as metadata. It is not included as a "
                "phen__ model-input column when USE_SITE_ID_AS_FEATURE=False."
            ),
            "n_total": int(len(y)),
            "n_train": int(len(split_map["train"])),
            "n_validation": int(len(split_map["validation"])),
            "n_test": int(len(split_map["test"])),
            "n_train_validation": int(len(split_map["train_validation"])),
            "n_fc_features": int(X_conn_model.shape[1]),
            "n_phenotype_columns": int(
                len([c for c in prepared_df.columns if c.startswith("phen__")])
            ),
            "files": {
                "all": "all_fixed.csv",
                "train": "train_fixed.csv",
                "validation": "val_fixed.csv",
                "test": "test_fixed.csv",
                "train_validation": "train_val_fixed.csv",
            },
        },
        output_dir / "fixed_prepared_data_summary.json",
    )


def _make_experiment_feature_blocks_for_fixed_csvs(
    *,
    exp: dict,
    df: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    feature_groups: dict,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str], list[str]]:
    selected_cols, numeric_cols, cat_cols, missing_cols = build_feature_lists_from_selected_keys(
        df=df,
        feature_groups=feature_groups,
        selected_group_keys=exp["feature_group_keys"],
    )
    domain_cols = [
        c for c in numeric_cols
        if c in DOMAIN_SCALED_NUMERIC_COLS and c in df_model.columns
    ]
    other_num_cols = [
        c for c in numeric_cols
        if c not in DOMAIN_SCALED_NUMERIC_COLS and c in df_model.columns
    ]
    X_df, fc_cols = build_combined_dataframe(
        df_model=df_model,
        X_conn_model=X_conn_model,
        phenotype_selected_cols=selected_cols,
    )
    return X_df, fc_cols, domain_cols, other_num_cols, cat_cols, selected_cols, missing_cols


def _processed_split_dataframe_for_fixed_csvs(
    *,
    X_processed: np.ndarray,
    y_split: np.ndarray,
    original_indices: np.ndarray,
    feature_names: np.ndarray,
    groups: Optional[pd.Series],
    split_name: str,
) -> pd.DataFrame:
    feature_df = pd.DataFrame(
        np.asarray(X_processed, dtype=np.float64),
        columns=[str(c) for c in feature_names],
    )
    meta = pd.DataFrame(
        {
            "row_index_after_valid_mask": np.asarray(original_indices, dtype=int),
            "split": split_name,
            "y_binary": np.asarray(y_split, dtype=int),
        }
    )
    if groups is not None:
        meta["SITE_ID"] = groups.iloc[original_indices].astype(str).to_numpy()
    return pd.concat([meta.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def save_experiment_ready_fixed_split_csvs(
    *,
    output_dir: Path,
    df: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    y: np.ndarray,
    groups: Optional[pd.Series],
    feature_groups: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_seed: int,
) -> None:
    """
    Save fully numeric, experiment-specific, model-ready fixed split CSVs.

    The preprocessor is fit only on the fixed train split and then applied to
    fixed validation/test. These files can be consumed directly by downstream
    training jobs such as DP-GD without rerunning the feature engineering code.
    """
    base_dir = output_dir / "fixed_processed_by_experiment"
    base_dir.mkdir(parents=True, exist_ok=True)
    split_map = _split_index_map_from_arrays(train_idx, val_idx, test_idx)

    for exp in EXPERIMENTS:
        exp_dir = base_dir / exp["experiment_key"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        (
            X_df,
            fc_cols,
            domain_cols,
            other_num_cols,
            cat_cols,
            selected_cols,
            missing_cols,
        ) = _make_experiment_feature_blocks_for_fixed_csvs(
            exp=exp,
            df=df,
            df_model=df_model,
            X_conn_model=X_conn_model,
            feature_groups=feature_groups,
        )

        tr = split_map["train"]
        va = split_map["validation"]
        te = split_map["test"]

        X_train_df = X_df.iloc[tr].reset_index(drop=True)
        X_val_df = X_df.iloc[va].reset_index(drop=True)
        X_test_df = X_df.iloc[te].reset_index(drop=True)

        preprocessor = make_preprocessor(
            phenotype_domain_scaled_cols=domain_cols,
            phenotype_other_numeric_cols=other_num_cols,
            phenotype_categorical_cols=cat_cols,
            fc_cols=fc_cols,
            fc_is_correlation=FC_IS_CORRELATION,
            phenotype_other_numeric_mode=PHENOTYPE_OTHER_NUMERIC_MODE,
        )
        X_train = np.asarray(preprocessor.fit_transform(X_train_df), dtype=np.float64)
        X_val = np.asarray(preprocessor.transform(X_val_df), dtype=np.float64)
        X_test = np.asarray(preprocessor.transform(X_test_df), dtype=np.float64)
        feature_names = preprocessor.get_feature_names_out()

        train_df = _processed_split_dataframe_for_fixed_csvs(
            X_processed=X_train,
            y_split=y[tr],
            original_indices=tr,
            feature_names=feature_names,
            groups=groups,
            split_name="train",
        )
        val_df = _processed_split_dataframe_for_fixed_csvs(
            X_processed=X_val,
            y_split=y[va],
            original_indices=va,
            feature_names=feature_names,
            groups=groups,
            split_name="validation",
        )
        test_df = _processed_split_dataframe_for_fixed_csvs(
            X_processed=X_test,
            y_split=y[te],
            original_indices=te,
            feature_names=feature_names,
            groups=groups,
            split_name="test",
        )
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        train_val_df["split"] = "train_validation"
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        train_df.to_csv(exp_dir / "train_fixed.csv", index=False)
        val_df.to_csv(exp_dir / "val_fixed.csv", index=False)
        test_df.to_csv(exp_dir / "test_fixed.csv", index=False)
        train_val_df.to_csv(exp_dir / "train_val_fixed.csv", index=False)
        all_df.to_csv(exp_dir / "all_fixed.csv", index=False)

        save_json(
            {
                "experiment_key": exp["experiment_key"],
                "experiment_name": exp["experiment_name"],
                "fixed_split_created_with_seed": int(split_seed),
                "target_column": "y_binary",
                "metadata_columns": [
                    c for c in ["row_index_after_valid_mask", "split", "y_binary", "SITE_ID"]
                    if c in train_df.columns
                ],
                "n_train": int(len(train_df)),
                "n_validation": int(len(val_df)),
                "n_test": int(len(test_df)),
                "n_train_validation": int(len(train_val_df)),
                "n_features_after_preprocessing": int(len(feature_names)),
                "selected_feature_group_keys": exp["feature_group_keys"],
                "selected_phenotype_columns_present_in_dataframe": selected_cols,
                "phenotype_domain_scaled_numeric_columns": domain_cols,
                "phenotype_other_numeric_columns": other_num_cols,
                "phenotype_categorical_columns": cat_cols,
                "fc_feature_count": int(len(fc_cols)),
                "requested_json_columns_absent_from_dataframe": missing_cols,
                "preprocessing_fit_policy": (
                    "The preprocessor is fit only on fixed train and then "
                    "applied to fixed validation/test to avoid leakage."
                ),
                "files": {
                    "train": "train_fixed.csv",
                    "validation": "val_fixed.csv",
                    "test": "test_fixed.csv",
                    "train_validation": "train_val_fixed.csv",
                    "all": "all_fixed.csv",
                },
            },
            exp_dir / "fixed_processed_schema.json",
        )

def save_fixed_split_outputs(
    *,
    output_dir: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    groups: Optional[pd.Series],
    split_name: str,
    split_audit: dict,
    split_seed: int,
    validation_size: float,
    test_size: float,
    df: pd.DataFrame,
    df_model_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    X_conn_model: np.ndarray,
    feature_groups: dict,
) -> None:
    """
    Save the single fixed split at OUTPUT_DIR so that other scripts can reuse
    exactly the same train/validation/test samples.

    Files saved:
        train_idx.npy / val_idx.npy / test_idx.npy
        fixed_train_idx.npy / fixed_val_idx.npy / fixed_test_idx.npy
        fixed_train_val_idx.npy
        fixed_split_indices.csv
        fixed_split_assignment.csv
        fixed_split_audit.json
        fixed_split_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))

    # # Backward/simple filenames for direct reuse.
    # np.save(output_dir / "train_idx.npy", train_idx)
    # np.save(output_dir / "val_idx.npy", val_idx)
    # np.save(output_dir / "test_idx.npy", test_idx)

    # Explicit fixed-split filenames.
    np.save(output_dir / "fixed_train_idx.npy", train_idx)
    np.save(output_dir / "fixed_val_idx.npy", val_idx)
    np.save(output_dir / "fixed_test_idx.npy", test_idx)
    np.save(output_dir / "fixed_train_val_idx.npy", train_val_idx)

    index_rows = []
    for split_label, idx_arr in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        for idx in idx_arr:
            row = {
                "index": int(idx),
                "split": split_label,
                "y_binary": int(y[int(idx)]),
            }
            if groups is not None:
                row["SITE_ID"] = str(groups.iloc[int(idx)])
            index_rows.append(row)

    fixed_split_indices_df = pd.DataFrame(index_rows)
    fixed_split_indices_df.to_csv(output_dir / "fixed_split_indices.csv", index=False)

    assignment = pd.DataFrame(
        {
            "index": np.arange(len(y), dtype=int),
            "split": "unassigned",
            "y_binary": np.asarray(y, dtype=int),
        }
    )
    assignment.loc[train_idx, "split"] = "train"
    assignment.loc[val_idx, "split"] = "validation"
    assignment.loc[test_idx, "split"] = "test"

    if groups is not None:
        assignment["SITE_ID"] = groups.astype(str).to_numpy()

    assignment.to_csv(output_dir / "fixed_split_assignment.csv", index=False)

    split_summary = {
        "split_seed": int(split_seed),
        "split_strategy": split_name,
        "validation_size": float(validation_size),
        "test_size": float(test_size),
        "train_size_target": float(1.0 - validation_size - test_size),
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "fixed_split_policy": (
            "The train/validation/test split is created exactly once using "
            "GLOBAL_C_TUNING_SEED and then reused unchanged for every seed in "
            "RANDOM_SEEDS."
        ),
        "index_space": (
            "Indices refer to df_model_raw / df_model / X_conn_model after "
            "valid-mask filtering and reset_index(drop=True)."
        ),
        "files": {
            "train_idx": "train_idx.npy",
            "val_idx": "val_idx.npy",
            "test_idx": "test_idx.npy",
            "fixed_train_idx": "fixed_train_idx.npy",
            "fixed_val_idx": "fixed_val_idx.npy",
            "fixed_test_idx": "fixed_test_idx.npy",
            "fixed_train_val_idx": "fixed_train_val_idx.npy",
            "fixed_split_indices": "fixed_split_indices.csv",
            "fixed_split_assignment": "fixed_split_assignment.csv",
            "fixed_split_audit": "fixed_split_audit.json",
        },
        "split_audit": split_audit,
    }

    save_common_fixed_prepared_split_csvs(
        output_dir=output_dir,
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        groups=groups,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_seed=split_seed,
    )

    save_experiment_ready_fixed_split_csvs(
        output_dir=output_dir,
        df=df,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        groups=groups,
        feature_groups=feature_groups,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_seed=split_seed,
    )

    split_summary["fixed_prepared_csv_files"] = {
        "common_all": "all_fixed.csv",
        "common_train": "train_fixed.csv",
        "common_validation": "val_fixed.csv",
        "common_test": "test_fixed.csv",
        "common_train_validation": "train_val_fixed.csv",
        "experiment_ready_base_dir": "fixed_processed_by_experiment/<experiment_key>/",
    }

    save_json(split_audit, output_dir / "fixed_split_audit.json")
    save_json(split_summary, output_dir / "fixed_split_summary.json")


def build_fixed_split_seed_summary(
    *,
    seed: int,
    split_name: str,
    split_audit: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
) -> dict:
    train_class_counts = {
        str(k): int(v)
        for k, v in pd.Series(y[train_idx]).value_counts().sort_index().to_dict().items()
    }
    val_class_counts = {
        str(k): int(v)
        for k, v in pd.Series(y[val_idx]).value_counts().sort_index().to_dict().items()
    }
    test_class_counts = {
        str(k): int(v)
        for k, v in pd.Series(y[test_idx]).value_counts().sort_index().to_dict().items()
    }

    return {
        "seed": int(seed),
        "split_strategy": split_name,
        "split_created_once_with_seed": int(GLOBAL_C_TUNING_SEED),
        "split_reused_for_this_seed": True,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_class_counts": train_class_counts,
        "validation_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "split_audit": split_audit,
    }


def main() -> None:
    if PHENOTYPE_OTHER_NUMERIC_MODE not in ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES:
        raise ValueError(
            f"Invalid PHENOTYPE_OTHER_NUMERIC_MODE={PHENOTYPE_OTHER_NUMERIC_MODE!r}. "
            f"Allowed values: {sorted(ALLOWED_PHENOTYPE_OTHER_NUMERIC_MODES)}"
        )

    if USE_SITE_ID_AS_FEATURE:
        raise ValueError(
            "This Non-IID experiment requires USE_SITE_ID_AS_FEATURE=False. "
            "SITE_ID should only be used for grouping and reporting, not as a model input."
        )

    if not RANDOM_SEEDS:
        raise ValueError("RANDOM_SEEDS must contain at least one experiment seed.")

    source_experiment = get_experiment_by_key(
        EXPERIMENTS,
        GLOBAL_C_TUNING_EXPERIMENT_KEY,
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

    # X_conn must already exist and must be aligned with the phenotypic dataframe.
    X_conn_np = np.asarray(X_conn, dtype=np.float64)  # noqa: F821
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

    groups_all = get_site_groups(df_model_raw)
    if SITE_AWARE_SPLIT and groups_all is None:
        raise ValueError(
            "SITE_AWARE_SPLIT=True but SITE_ID column is not available. "
            "Cannot create Non-IID SITE-aware split."
        )

    print("\nAfter valid-mask filtering:")
    print("df_model shape:", df_model.shape)
    print("X_conn_model shape:", X_conn_model.shape)
    print("Binary class counts (0=Control, 1=Autism):")
    print(pd.Series(y).value_counts().sort_index())

    if groups_all is not None:
        print("\nSITE_ID counts after valid-mask filtering:")
        print(groups_all.value_counts().sort_index())

    # These are independent of experiment seed, so save once at the root output directory.
    np.save(OUTPUT_DIR / "X_conn_model.npy", X_conn_model)
    np.save(OUTPUT_DIR / "y_binary.npy", y)
    np.save(OUTPUT_DIR / "random_seeds.npy", np.asarray(RANDOM_SEEDS, dtype=int))
    np.save(OUTPUT_DIR / "global_c_tuning_seed.npy", np.asarray([GLOBAL_C_TUNING_SEED], dtype=int))

    if groups_all is not None:
        groups_all.to_csv(OUTPUT_DIR / "site_groups_after_valid_mask.csv", index=False)

    # -------------------------------------------------------------------------
    # Fixed split:
    # Create the train/validation/test split exactly once using
    # GLOBAL_C_TUNING_SEED, then reuse it for every seed in RANDOM_SEEDS.
    # -------------------------------------------------------------------------
    train_idx, val_idx, test_idx, split_name, split_audit = make_train_val_test_split(
        y=y,
        seed=int(GLOBAL_C_TUNING_SEED),
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        groups=groups_all,
        site_aware=SITE_AWARE_SPLIT,
    )

    print("\n" + "=" * 120)
    print("FIXED TRAIN/VALIDATION/TEST SPLIT CREATED ONCE")
    print("=" * 120)
    print(f"Split seed      : {GLOBAL_C_TUNING_SEED}")
    print(split_name)
    print(f"Train size      : {len(train_idx)}")
    print(f"Validation size : {len(val_idx)}")
    print(f"Test size       : {len(test_idx)}")
    print("\nFixed split audit:")
    print(json.dumps(json_clean(split_audit), indent=2))

    save_fixed_split_outputs(
        output_dir=OUTPUT_DIR,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        y=y,
        groups=groups_all,
        split_name=split_name,
        split_audit=split_audit,
        split_seed=int(GLOBAL_C_TUNING_SEED),
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        df=df,
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        feature_groups=feature_groups,
    )

    # -------------------------------------------------------------------------
    # One-time global C tuning:
    # Always run the source experiment once using GLOBAL_C_TUNING_SEED and the
    # fixed split. GLOBAL_C_TUNING_SEED may or may not be in RANDOM_SEEDS.
    # -------------------------------------------------------------------------
    global_best_C: Optional[float] = None
    global_c_selection_summary: Optional[dict] = None
    global_tuning_metrics_df: Optional[pd.DataFrame] = None
    global_tuning_subgroup_df: Optional[pd.DataFrame] = None

    tuning_seed_output_dir = OUTPUT_DIR / f"seed_{int(GLOBAL_C_TUNING_SEED)}"
    save_indices_for_seed_dir(
        seed_output_dir=tuning_seed_output_dir,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_audit=split_audit,
    )

    print("\n" + "=" * 120)
    print("RUNNING ONE-TIME GLOBAL C TUNING")
    print("=" * 120)
    print(f"Global C tuning seed       : {GLOBAL_C_TUNING_SEED}")
    print(f"Global C tuning experiment : {GLOBAL_C_TUNING_EXPERIMENT_KEY}")

    global_tuning_metrics_df, global_tuning_subgroup_df, tuned_C = run_single_experiment(
        experiment_key=source_experiment["experiment_key"],
        experiment_name=source_experiment["experiment_name"],
        selected_group_keys=source_experiment["feature_group_keys"],
        df=df,
        df_model_raw=df_model_raw,
        df_model=df_model,
        X_conn_model=X_conn_model,
        y=y,
        feature_groups=feature_groups,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_name=split_name,
        output_root=tuning_seed_output_dir,
        seed=int(GLOBAL_C_TUNING_SEED),
        c_selection_mode=C_SELECTION_MODE_TUNE_SOURCE,
        fixed_C=None,
    )

    global_best_C = float(tuned_C)

    global_c_selection_summary = {
        "global_best_C": global_best_C,
        "global_c_tuning_seed": int(GLOBAL_C_TUNING_SEED),
        "global_c_tuning_seed_is_in_random_seeds": bool(int(GLOBAL_C_TUNING_SEED) in [int(s) for s in RANDOM_SEEDS]),
        "global_c_tuning_experiment_key": GLOBAL_C_TUNING_EXPERIMENT_KEY,
        "global_c_tuning_experiment_name": source_experiment["experiment_name"],
        "c_grid": C_GRID,
        "selection_rule": (
            "Sort by validation_accuracy descending, "
            "then validation_balanced_accuracy descending, "
            "then validation_roc_auc descending, "
            "then validation_log_loss ascending."
        ),
        "reuse_policy": (
            "The selected C from this source run is reused for all experiments "
            "for every seed in RANDOM_SEEDS. No further C-grid tuning is performed."
        ),
        "fixed_split_policy": (
            "The train/validation/test split is created once using "
            "GLOBAL_C_TUNING_SEED and then reused unchanged for every experiment seed."
        ),
        "source_split_strategy": split_name,
        "source_split_audit": split_audit,
    }

    save_json(
        global_c_selection_summary,
        OUTPUT_DIR / "global_fixed_C_selection_summary.json",
    )
    np.save(OUTPUT_DIR / "global_fixed_C.npy", np.asarray([global_best_C]))

    print("\n" + "=" * 120)
    print("GLOBAL C SELECTED")
    print("=" * 120)
    print(f"Source seed       : {GLOBAL_C_TUNING_SEED}")
    print(f"Source experiment : {GLOBAL_C_TUNING_EXPERIMENT_KEY}")
    print(f"Global best C     : {global_best_C}")

    random_seed_set = {int(s) for s in RANDOM_SEEDS}

    if int(GLOBAL_C_TUNING_SEED) not in random_seed_set:
        save_seed_level_outputs(
            seed_output_dir=tuning_seed_output_dir,
            seed_metrics=[global_tuning_metrics_df],
            seed_subgroup_metrics=[
                global_tuning_subgroup_df
            ] if global_tuning_subgroup_df is not None and not global_tuning_subgroup_df.empty else [],
        )

    all_metrics: list[pd.DataFrame] = []
    all_subgroup_metrics: list[pd.DataFrame] = []
    per_seed_split_summaries = []

    # -------------------------------------------------------------------------
    # Final repeated experiments:
    # For every seed in RANDOM_SEEDS, reuse the fixed split and global C.
    # If GLOBAL_C_TUNING_SEED is in RANDOM_SEEDS, reuse the already-computed
    # source experiment result for that seed and run only the remaining modes.
    # -------------------------------------------------------------------------
    for seed in [int(s) for s in RANDOM_SEEDS]:
        print("\n" + "=" * 120)
        print(f"Running full experiment for seed = {seed}")
        print("=" * 120)
        print(
            "Using the fixed train/validation/test split created with "
            f"GLOBAL_C_TUNING_SEED={GLOBAL_C_TUNING_SEED}."
        )

        seed_output_dir = OUTPUT_DIR / f"seed_{seed}"
        save_indices_for_seed_dir(
            seed_output_dir=seed_output_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            split_audit=split_audit,
        )

        print("\nFixed split strategy reused for this seed:")
        print(split_name)
        print(f"Train size      : {len(train_idx)}")
        print(f"Validation size : {len(val_idx)}")
        print(f"Test size       : {len(test_idx)}")

        per_seed_split_summaries.append(
            build_fixed_split_seed_summary(
                seed=seed,
                split_name=split_name,
                split_audit=split_audit,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                y=y,
            )
        )

        seed_metrics: list[pd.DataFrame] = []
        seed_subgroup_metrics: list[pd.DataFrame] = []

        if seed == int(GLOBAL_C_TUNING_SEED):
            # Source experiment has already been run with full C-grid tuning.
            append_result(
                metrics_df=global_tuning_metrics_df,
                subgroup_df=global_tuning_subgroup_df,
                all_metrics=all_metrics,
                all_subgroup_metrics=all_subgroup_metrics,
            )
            seed_metrics.append(global_tuning_metrics_df)
            if global_tuning_subgroup_df is not None and not global_tuning_subgroup_df.empty:
                seed_subgroup_metrics.append(global_tuning_subgroup_df)

            experiments_to_run = [
                exp for exp in EXPERIMENTS
                if exp["experiment_key"] != GLOBAL_C_TUNING_EXPERIMENT_KEY
            ]
        else:
            experiments_to_run = EXPERIMENTS

        for exp in experiments_to_run:
            metrics_df, subgroup_df, _ = run_single_experiment(
                experiment_key=exp["experiment_key"],
                experiment_name=exp["experiment_name"],
                selected_group_keys=exp["feature_group_keys"],
                df=df,
                df_model_raw=df_model_raw,
                df_model=df_model,
                X_conn_model=X_conn_model,
                y=y,
                feature_groups=feature_groups,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                split_name=split_name,
                output_root=seed_output_dir,
                seed=seed,
                c_selection_mode=C_SELECTION_MODE_REUSE_GLOBAL,
                fixed_C=global_best_C,
            )

            append_result(
                metrics_df=metrics_df,
                subgroup_df=subgroup_df,
                all_metrics=all_metrics,
                all_subgroup_metrics=all_subgroup_metrics,
            )
            seed_metrics.append(metrics_df)
            if subgroup_df is not None and not subgroup_df.empty:
                seed_subgroup_metrics.append(subgroup_df)

        save_seed_level_outputs(
            seed_output_dir=seed_output_dir,
            seed_metrics=seed_metrics,
            seed_subgroup_metrics=seed_subgroup_metrics,
        )

    all_metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    all_metrics_df.to_csv(OUTPUT_DIR / "all_modes_metrics_summary.csv", index=False)

    test_metrics_comparison = all_metrics_df.loc[
        all_metrics_df["split"] == "test_final_model",
        [
            "seed",
            "experiment_key",
            "experiment_name",
            "best_C",
            "c_selection_mode",
            "c_source_description",
            "final_training_policy",
            "n_fc_features",
            "n_selected_phenotype_raw_columns",
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "roc_auc",
            "average_precision",
        ],
    ].reset_index(drop=True)

    test_metrics_comparison.to_csv(
        OUTPUT_DIR / "all_modes_test_metrics_comparison.csv",
        index=False,
    )

    # Aggregate all split metrics across RANDOM_SEEDS for each mode/experiment/split.
    metric_cols_for_seed_aggregation = [
        "best_C",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "average_precision",
    ]

    all_splits_metrics_across_seeds = aggregate_metrics_across_seeds(
        all_metrics_df,
        group_cols=["experiment_key", "experiment_name", "split"],
        metric_cols=metric_cols_for_seed_aggregation,
        seed_col="seed",
    )

    all_splits_metrics_across_seeds.to_csv(
        OUTPUT_DIR / "all_modes_all_splits_metrics_across_seeds_mean_std_sem_n.csv",
        index=False,
    )

    # Aggregate final test metrics across RANDOM_SEEDS for each mode/experiment.
    test_metrics_across_seeds = aggregate_metrics_across_seeds(
        test_metrics_comparison,
        group_cols=["experiment_key", "experiment_name"],
        metric_cols=metric_cols_for_seed_aggregation,
        seed_col="seed",
    )

    test_metrics_across_seeds.to_csv(
        OUTPUT_DIR / "all_modes_test_metrics_across_seeds_mean_std_sem_n.csv",
        index=False,
    )

    # Backward-compatible filename: same table, now with *_sem and *_n columns too.
    test_metrics_across_seeds.to_csv(
        OUTPUT_DIR / "all_modes_test_metrics_across_seeds_mean_std.csv",
        index=False,
    )

    if all_subgroup_metrics:
        all_subgroup_metrics_df = pd.concat(
            all_subgroup_metrics,
            axis=0,
            ignore_index=True,
        )
    else:
        all_subgroup_metrics_df = pd.DataFrame()

    all_subgroup_metrics_df.to_csv(
        OUTPUT_DIR / "all_modes_all_splits_subgroup_metrics_summary.csv",
        index=False,
    )

    test_subgroup_metrics_df = (
        all_subgroup_metrics_df.loc[
            all_subgroup_metrics_df["split"] == "test_final_model"
        ].reset_index(drop=True)
        if not all_subgroup_metrics_df.empty
        else pd.DataFrame()
    )

    test_subgroup_metrics_df.to_csv(
        OUTPUT_DIR / "all_modes_test_subgroup_metrics_summary.csv",
        index=False,
    )

    subgroup_metric_cols_for_seed_aggregation = [
        "best_C",
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

    all_splits_subgroup_metrics_across_seeds = pd.DataFrame()
    if not all_subgroup_metrics_df.empty:
        all_splits_subgroup_metrics_across_seeds = aggregate_metrics_across_seeds(
            all_subgroup_metrics_df,
            group_cols=[
                "experiment_key",
                "experiment_name",
                "split",
                "feature",
                "group",
            ],
            metric_cols=subgroup_metric_cols_for_seed_aggregation,
            seed_col="seed",
        )

    all_splits_subgroup_metrics_across_seeds.to_csv(
        OUTPUT_DIR / "all_modes_all_splits_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
        index=False,
    )

    test_subgroup_metrics_across_seeds = pd.DataFrame()
    if not test_subgroup_metrics_df.empty:
        test_subgroup_metrics_across_seeds = aggregate_metrics_across_seeds(
            test_subgroup_metrics_df,
            group_cols=[
                "experiment_key",
                "experiment_name",
                "split",
                "feature",
                "group",
            ],
            metric_cols=subgroup_metric_cols_for_seed_aggregation,
            seed_col="seed",
        )

    test_subgroup_metrics_across_seeds.to_csv(
        OUTPUT_DIR / "all_modes_test_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
        index=False,
    )

    run_summary = {
        "random_seeds": [int(s) for s in RANDOM_SEEDS],
        "target_col": TARGET_COL,
        "csv_path": str(CSV_PATH),
        "feature_json_path_used": str(feature_json_path),
        "output_dir": str(OUTPUT_DIR),
        "experiments": EXPERIMENTS,
        "removed_experiment": "fmri_plus_sensitive_with_anat_func",
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "site_aware_splitting_used": SITE_AWARE_SPLIT,
        "site_aware_n_splits": SITE_AWARE_N_SPLITS,
        "non_iid_definition": (
            "SITE_ID-disjoint train/validation/test split using StratifiedGroupKFold. "
            "The same SITE_ID is not allowed to appear in more than one split."
        ),
        "fixed_split_policy": (
            f"The split is created exactly once with GLOBAL_C_TUNING_SEED={GLOBAL_C_TUNING_SEED} "
            "and reused unchanged for every seed in RANDOM_SEEDS."
        ),
        "fixed_split_files": {
            "train_idx": "train_idx.npy",
            "val_idx": "val_idx.npy",
            "test_idx": "test_idx.npy",
            "fixed_train_idx": "fixed_train_idx.npy",
            "fixed_val_idx": "fixed_val_idx.npy",
            "fixed_test_idx": "fixed_test_idx.npy",
            "fixed_train_val_idx": "fixed_train_val_idx.npy",
            "fixed_split_indices": "fixed_split_indices.csv",
            "fixed_split_assignment": "fixed_split_assignment.csv",
            "fixed_split_audit": "fixed_split_audit.json",
            "fixed_split_summary": "fixed_split_summary.json",
            "common_train_fixed_csv": "train_fixed.csv",
            "common_val_fixed_csv": "val_fixed.csv",
            "common_test_fixed_csv": "test_fixed.csv",
            "experiment_ready_csv_dir": "fixed_processed_by_experiment/<experiment_key>/",
        },
        "use_site_id_as_feature": USE_SITE_ID_AS_FEATURE,
        "fc_is_correlation": FC_IS_CORRELATION,
        "phenotype_other_numeric_mode": PHENOTYPE_OTHER_NUMERIC_MODE,
        "c_grid": C_GRID,
        "global_best_C": global_best_C,
        "global_c_tuning_seed": int(GLOBAL_C_TUNING_SEED),
        "global_c_tuning_seed_is_in_random_seeds": bool(int(GLOBAL_C_TUNING_SEED) in random_seed_set),
        "global_c_tuning_experiment_key": GLOBAL_C_TUNING_EXPERIMENT_KEY,
        "global_c_selection_summary": global_c_selection_summary,
        "sensible_numeric_bins": SENSIBLE_NUMERIC_BINS,
        "n_total_after_valid_mask": int(len(y)),
        "class_counts_total": {
            str(k): int(v)
            for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()
        },
        "fixed_split_summary": {
            "split_strategy": split_name,
            "split_audit": split_audit,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "per_seed_split_summaries": per_seed_split_summaries,
        "hyperparameter_policy": (
            f"C is tuned exactly once: seed={GLOBAL_C_TUNING_SEED}, "
            f"experiment={GLOBAL_C_TUNING_EXPERIMENT_KEY}. The selected C is then reused "
            "for all experiments across RANDOM_SEEDS. No further C-grid tuning is performed."
        ),
        "final_model_training_policy": (
            "For every seed and experiment, the selected/reused C is used to train a model "
            "on the fixed train split and evaluate on the fixed validation split. Then a fresh "
            "final model is refit on fixed train+validation using the same C and evaluated "
            "once on the fixed held-out SITE_ID-disjoint test set."
            if REFIT_FINAL_ON_TRAIN_VAL
            else "For every seed and experiment, the selected/reused C is used to train a model on the fixed train split; the selected train-only model is evaluated on fixed validation and fixed test."
        ),
        "reported_outputs": {
            "fixed_train_idx": "train_idx.npy / fixed_train_idx.npy",
            "fixed_val_idx": "val_idx.npy / fixed_val_idx.npy",
            "fixed_test_idx": "test_idx.npy / fixed_test_idx.npy",
            "per_seed_test_metrics": "seed_<seed>/all_modes_test_metrics_comparison_seed.csv",
            "per_seed_test_subgroup_metrics": "seed_<seed>/all_modes_test_subgroup_metrics_seed.csv",
            "overall_test_metrics": "all_modes_test_metrics_comparison.csv",
            "overall_test_metrics_aggregate": "all_modes_test_metrics_across_seeds_mean_std_sem_n.csv",
            "overall_test_subgroup_metrics": "all_modes_test_subgroup_metrics_summary.csv",
            "overall_test_subgroup_metrics_aggregate": "all_modes_test_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
            "overall_all_split_metrics_aggregate": "all_modes_all_splits_metrics_across_seeds_mean_std_sem_n.csv",
            "overall_all_split_subgroup_metrics_aggregate": "all_modes_all_splits_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
        },
    }

    save_json(run_summary, OUTPUT_DIR / "run_summary_all_modes.json")

    print("\n" + "=" * 120)
    print("Final test metrics comparison across all modes and all RANDOM_SEEDS")
    print("=" * 120)
    print(test_metrics_comparison.to_string(index=False))

    print("\n" + "=" * 120)
    print("Final test metrics mean/std/sem/n across RANDOM_SEEDS")
    print("=" * 120)
    print(test_metrics_across_seeds.to_string(index=False))

    print("\nSaved combined outputs in:")
    print(OUTPUT_DIR)

    print("\nMost important fixed split files:")
    for fp in [
        OUTPUT_DIR / "train_idx.npy",
        OUTPUT_DIR / "val_idx.npy",
        OUTPUT_DIR / "test_idx.npy",
        OUTPUT_DIR / "fixed_train_idx.npy",
        OUTPUT_DIR / "fixed_val_idx.npy",
        OUTPUT_DIR / "fixed_test_idx.npy",
        OUTPUT_DIR / "fixed_train_val_idx.npy",
        OUTPUT_DIR / "fixed_split_indices.csv",
        OUTPUT_DIR / "fixed_split_assignment.csv",
        OUTPUT_DIR / "fixed_split_audit.json",
        OUTPUT_DIR / "fixed_split_summary.json",
        OUTPUT_DIR / "train_fixed.csv",
        OUTPUT_DIR / "val_fixed.csv",
        OUTPUT_DIR / "test_fixed.csv",
        OUTPUT_DIR / "train_val_fixed.csv",
        OUTPUT_DIR / "fixed_prepared_data_summary.json",
        OUTPUT_DIR / "fixed_processed_by_experiment",
    ]:
        print(fp)

    print("\nMost important combined output files:")
    for fp in [
        OUTPUT_DIR / "global_fixed_C_selection_summary.json",
        OUTPUT_DIR / "global_fixed_C.npy",
        OUTPUT_DIR / "all_modes_metrics_summary.csv",
        OUTPUT_DIR / "all_modes_test_metrics_comparison.csv",
        OUTPUT_DIR / "all_modes_test_metrics_across_seeds_mean_std_sem_n.csv",
        OUTPUT_DIR / "all_modes_test_metrics_across_seeds_mean_std.csv",
        OUTPUT_DIR / "all_modes_all_splits_metrics_across_seeds_mean_std_sem_n.csv",
        OUTPUT_DIR / "all_modes_test_subgroup_metrics_summary.csv",
        OUTPUT_DIR / "all_modes_test_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
        OUTPUT_DIR / "all_modes_all_splits_subgroup_metrics_summary.csv",
        OUTPUT_DIR / "all_modes_all_splits_subgroup_metrics_across_seeds_mean_std_sem_n.csv",
        OUTPUT_DIR / "run_summary_all_modes.json",
    ]:
        print(fp)

    if int(GLOBAL_C_TUNING_SEED) not in random_seed_set:
        print("\nGlobal C tuning output was saved separately under:")
        print(OUTPUT_DIR / f"seed_{int(GLOBAL_C_TUNING_SEED)}")

    print("\nSeed-specific outputs are saved under:")
    for seed in RANDOM_SEEDS:
        print(OUTPUT_DIR / f"seed_{int(seed)}")


if __name__ == "__main__":
    main()