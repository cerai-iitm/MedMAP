# medmap/datasets/abide/hf_pipeline.py
#
# ABIDE I data pipeline for the Hugging Face backend: loads the precomputed
# fMRI functional-connectivity features (X_conn.npy) plus the phenotypic CSV,
# preprocesses the sensitive phenotype features (fixed-domain scaling to
# [-1, 1], missing-as-all-zero one-hot categoricals; fit on train only), and
# builds train/val/test HF Datasets with a stratified IID split.
# See docs/abide.md for how to produce the inputs.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


TARGET_COL = "DX_GROUP"

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


@dataclass
class ABIDEDatasetBundle:
    train_ds: Dataset
    val_ds: Dataset
    test_ds: Dataset
    metadata: dict
    preprocessor: "ABIDESensitivePreprocessor"


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


def load_abide_phenotypic(
    data_dir: str | Path,
    phenotypic_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Prefer nilearn's fetch_abide_pcp so the row order matches the downloaded
    func_preproc files. Fall back to a local phenotypic CSV when provided.
    """
    try:
        from nilearn import datasets as nilearn_datasets

        abide = nilearn_datasets.fetch_abide_pcp(
            data_dir=str(data_dir),
            pipeline="cpac",
            derivatives=["func_preproc"],
            quality_checked=False,
            band_pass_filtering=False,
            global_signal_regression=False,
        )
        return abide.phenotypic.copy()
    except Exception as exc:
        if phenotypic_csv is None:
            raise RuntimeError(
                "Could not fetch ABIDE via nilearn and no phenotypic_csv was provided."
            ) from exc

        return pd.read_csv(phenotypic_csv)


def load_feature_groups(feature_json_path: str | Path) -> dict:
    with open(feature_json_path, "r") as f:
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


def build_sensitive_feature_columns(
    df: pd.DataFrame,
    feature_groups: dict,
    include_site_id_as_feature: bool = False,
    feature_group_keys: Optional[list[str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Default sensitive groups follow your non-private ABIDE setting:
      - recommended_initial_numeric_or_binary
      - recommended_initial_categorical
      - optional_categorical_with_caution

    By default SITE_ID is removed as an input feature.
    """

    if feature_group_keys is None:
        feature_group_keys = [
            "recommended_initial_numeric_or_binary",
            "recommended_initial_categorical",
            "optional_categorical_with_caution",
        ]

    numeric_group_keys = {
        "recommended_initial_numeric_or_binary",
        "anat_func_numeric_or_binary",
    }

    categorical_group_keys = {
        "recommended_initial_categorical",
        "optional_categorical_with_caution",
    }

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for key in feature_group_keys:
        cols = feature_groups.get(key, [])

        if key in numeric_group_keys:
            for c in cols:
                if c != TARGET_COL and c in df.columns and c not in numeric_cols:
                    numeric_cols.append(c)

        if key in categorical_group_keys:
            for c in cols:
                if c != TARGET_COL and c in df.columns and c not in categorical_cols:
                    categorical_cols.append(c)

    if not include_site_id_as_feature:
        numeric_cols = [c for c in numeric_cols if c != "SITE_ID"]
        categorical_cols = [c for c in categorical_cols if c != "SITE_ID"]

    selected_cols = numeric_cols + categorical_cols
    return selected_cols, numeric_cols, categorical_cols


class ABIDESensitivePreprocessor:
    """
    Train-split-fitted preprocessor for sensitive / phenotype features.

    Numeric behavior:
      - fixed-domain columns are scaled to [-1, 1] using fixed domain bounds;
      - other numeric columns are min-max scaled to [-1, 1] using train data;
      - missing/non-finite values are replaced with 0 after scaling.

    Categorical behavior:
      - categories are learned only from train split;
      - missing and unseen values map to an all-zero block.

    This matches the spirit of your sklearn baseline while producing a fully
    numeric matrix for HF Trainer.
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        other_numeric_mode: str = "minmax_minus1_1_zero_fill",
    ):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.other_numeric_mode = other_numeric_mode

        self.domain_cols = [c for c in self.numeric_cols if c in DOMAIN_SCALED_NUMERIC_COLS]
        self.other_numeric_cols = [c for c in self.numeric_cols if c not in DOMAIN_SCALED_NUMERIC_COLS]

        self.other_min_: dict[str, float] = {}
        self.other_max_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def _fixed_bounds(self, col: str) -> tuple[float, float]:
        if col in DOMAIN_FIXED_BOUNDS:
            return DOMAIN_FIXED_BOUNDS[col]
        if col in IQ_GLOBAL_BOUNDS:
            return IQ_GLOBAL_BOUNDS[col]
        raise KeyError(col)

    def _scale_fixed_domain(self, s: pd.Series, col: str) -> np.ndarray:
        lo, hi = self._fixed_bounds(col)
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

        if hi == lo:
            out = np.zeros_like(x, dtype=np.float64)
        else:
            out = 2.0 * (x - lo) / (hi - lo) - 1.0
            out = np.clip(out, -1.0, 1.0)

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_categorical_series(self, s: pd.Series, col: str) -> pd.Series:
        if col == "HANDEDNESS_CATEGORY":
            return s.map(normalize_handedness_category)

        if col in {"FIQ_TEST_TYPE", "PIQ_TEST_TYPE", "VIQ_TEST_TYPE"}:
            return s.map(normalize_test_type)

        out = s.copy()
        out = out.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
        out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-9999": np.nan})
        return out

    def fit(self, df_train: pd.DataFrame) -> "ABIDESensitivePreprocessor":
        df_train = df_train.copy()

        for col in self.other_numeric_cols:
            x = pd.to_numeric(df_train[col], errors="coerce").to_numpy(dtype=np.float64)

            if np.all(~np.isfinite(x)):
                self.other_min_[col] = 0.0
                self.other_max_[col] = 0.0
            else:
                self.other_min_[col] = float(np.nanmin(x))
                self.other_max_[col] = float(np.nanmax(x))

        for col in self.categorical_cols:
            s = self._normalize_categorical_series(df_train[col], col)
            observed = s[~pd.isna(s)].astype(str)
            self.categories_[col] = sorted(pd.Index(observed).unique().tolist())

        feature_names: list[str] = []
        feature_names.extend(self.domain_cols)
        feature_names.extend(self.other_numeric_cols)

        for col in self.categorical_cols:
            for cat in self.categories_[col]:
                feature_names.append(f"{col}={cat}")

        self.feature_names_ = feature_names
        return self

    def _transform_other_numeric_col(self, s: pd.Series, col: str) -> np.ndarray:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

        if self.other_numeric_mode == "zero_fill_only":
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        lo = self.other_min_[col]
        hi = self.other_max_[col]
        denom = hi - lo

        if not np.isfinite(denom) or denom <= 0:
            out = np.zeros_like(x, dtype=np.float64)
        else:
            out = 2.0 * (x - lo) / denom - 1.0
            out = np.clip(out, -1.0, 1.0)

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []

        for col in self.domain_cols:
            blocks.append(self._scale_fixed_domain(df[col], col).reshape(-1, 1))

        for col in self.other_numeric_cols:
            blocks.append(self._transform_other_numeric_col(df[col], col).reshape(-1, 1))

        for col in self.categorical_cols:
            cats = self.categories_[col]
            s = self._normalize_categorical_series(df[col], col)

            block = np.zeros((len(df), len(cats)), dtype=np.float32)

            if len(cats) > 0:
                valid = (~pd.isna(s)).to_numpy(dtype=bool)

                s_safe = s.where(~pd.isna(s), "__MISSING__").astype(str)
                values = s_safe.to_numpy(dtype=object)

                for j, cat in enumerate(cats):
                    match = valid & (values == str(cat))
                    block[match, j] = 1.0

            blocks.append(block)

        if not blocks:
            return np.zeros((len(df), 0), dtype=np.float32)

        out = np.hstack(blocks).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        self.fit(df_train)
        return self.transform(df_train)

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_)


def make_iid_train_val_test_split(
    y: np.ndarray,
    seed: int,
    validation_size: float = 0.10,
    test_size: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")

    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    n = len(y)

    splitter_1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_val_idx, test_idx = next(splitter_1.split(np.zeros(n), y))

    val_fraction_within_train_val = validation_size / (1.0 - test_size)
    y_train_val = y[train_val_idx]

    splitter_2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction_within_train_val,
        random_state=seed + 1,
    )
    train_rel_idx, val_rel_idx = next(splitter_2.split(np.zeros(len(train_val_idx)), y_train_val))

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def _make_hf_dataset(
    fmri: np.ndarray,
    sensitive: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
) -> Dataset:
    indices = np.asarray(indices, dtype=int)

    data = {
        "fmri_features": fmri[indices].astype(np.float32).tolist(),
        "sensitive_features": sensitive[indices].astype(np.float32).tolist(),
        "labels": y[indices].astype(np.int64).tolist(),
        "row_index": indices.astype(np.int64).tolist(),
    }

    return Dataset.from_dict(data)


def build_abide_linear_hf_datasets(
    *,
    data_dir: str | Path,
    fc_dir: Optional[str | Path] = None,
    feature_json_path: Optional[str | Path] = None,
    phenotypic_csv: Optional[str | Path] = None,
    seed: int = 81,
    validation_size: float = 0.10,
    test_size: float = 0.20,
    include_site_id_as_feature: bool = False,
    sensitive_feature_group_keys: Optional[list[str]] = None,
) -> ABIDEDatasetBundle:
    data_dir = Path(data_dir)
    fc_dir = Path(fc_dir) if fc_dir is not None else data_dir / "fc_features"
    if feature_json_path is None:
        # Packaged copy of the recommended sensitive-feature groups.
        feature_json_path = Path(__file__).resolve().parent / "resources" / "recommended_feature_lists.json"

    df = load_abide_phenotypic(data_dir=data_dir, phenotypic_csv=phenotypic_csv)
    df = clean_abide_dataframe(df).reset_index(drop=True)

    x_conn_path = fc_dir / "X_conn.npy"
    y_path = fc_dir / "y.npy"
    meta_path = fc_dir / "meta.json"

    if not x_conn_path.exists():
        raise FileNotFoundError(f"Missing fMRI connectivity features: {x_conn_path}")

    X_conn = np.load(x_conn_path)

    if y_path.exists():
        y = np.load(y_path)
    else:
        if TARGET_COL not in df.columns:
            raise ValueError(
                f"{TARGET_COL} not found in phenotype dataframe and {y_path} does not exist."
            )
        y = (pd.to_numeric(df[TARGET_COL], errors="coerce") == 1).astype(np.int64).to_numpy()

    if len(df) != len(X_conn) or len(y) != len(X_conn):
        raise ValueError(
            "ABIDE row alignment mismatch. Expected len(df) == len(X_conn) == len(y), got "
            f"len(df)={len(df)}, len(X_conn)={len(X_conn)}, len(y)={len(y)}."
        )

    y = np.asarray(y).astype(np.int64)

    # Convert ABIDE raw labels if needed: DX_GROUP 1 -> autism, 2 -> control.
    unique_y = set(np.unique(y).tolist())
    if unique_y <= {1, 2}:
        y = (y == 1).astype(np.int64)

    valid = np.isfinite(y)
    X_conn = X_conn[valid]
    y = y[valid].astype(np.int64)
    df = df.loc[valid].reset_index(drop=True)

    X_fmri = np.nan_to_num(
        X_conn.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    feature_groups = load_feature_groups(feature_json_path)

    selected_cols, numeric_cols, categorical_cols = build_sensitive_feature_columns(
        df=df,
        feature_groups=feature_groups,
        include_site_id_as_feature=include_site_id_as_feature,
        feature_group_keys=sensitive_feature_group_keys,
    )

    train_idx, val_idx, test_idx = make_iid_train_val_test_split(
        y=y,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
    )

    preprocessor = ABIDESensitivePreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        other_numeric_mode="minmax_minus1_1_zero_fill",
    )
    preprocessor.fit(df.iloc[train_idx])

    X_sensitive = preprocessor.transform(df)

    train_ds = _make_hf_dataset(X_fmri, X_sensitive, y, train_idx)
    val_ds = _make_hf_dataset(X_fmri, X_sensitive, y, val_idx)
    test_ds = _make_hf_dataset(X_fmri, X_sensitive, y, test_idx)

    metadata = {
        "n_total": int(len(y)),
        "n_train": int(len(train_ds)),
        "n_validation": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "fmri_dim": int(X_fmri.shape[1]),
        "sensitive_dim": int(X_sensitive.shape[1]),
        "num_labels": 2,
        "target": TARGET_COL,
        "label_mapping": {
            "1": "autism",
            "0": "control",
        },
        "selected_sensitive_columns": selected_cols,
        "numeric_sensitive_columns": numeric_cols,
        "categorical_sensitive_columns": categorical_cols,
        "sensitive_feature_names_after_preprocessing": preprocessor.get_feature_names_out(),
        "split": {
            "type": "iid_stratified",
            "seed": int(seed),
            "validation_size": float(validation_size),
            "test_size": float(test_size),
        },
        "files": {
            "X_conn": str(x_conn_path),
            "y": str(y_path) if y_path.exists() else None,
            "meta": str(meta_path) if meta_path.exists() else None,
            "feature_json": str(feature_json_path),
        },
    }

    return ABIDEDatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        metadata=metadata,
        preprocessor=preprocessor,
    )