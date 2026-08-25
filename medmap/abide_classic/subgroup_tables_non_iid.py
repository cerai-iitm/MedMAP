#!/usr/bin/env python3

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ported from "8. summarizer_fixed.ipynb". Output root is resolved from
# ABIDE_OUTPUT_ROOT / ABIDE_DATA_DIR (see medmap/abide_classic/data.py).
# ---------------------------------------------------------------------------
from medmap.abide_classic import data as _abide_data

_ABIDE_DATA_ROOT, _ABIDE_OUTPUT_ROOT = _abide_data.resolve_roots()
_O = str(_ABIDE_OUTPUT_ROOT)

from pathlib import Path
from typing import Iterable, Optional
import re
import shutil
import warnings

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

NON_IID_ROOT = Path(f"{_O}/non_iid_fixed_split")

NON_PRIV_CSV = (
    NON_IID_ROOT
    / "non_priv"
    / "all_modes_test_subgroup_metrics_across_seeds_mean_std_sem_n.csv"
)

FULL_DP_ROOT = NON_IID_ROOT / "full_dp"
FEATURE_DP_ROOT = NON_IID_ROOT / "feat_dp"

FULL_DP_CSV_NAME = "all_seeds_dp_gd_test_subgroup_metrics_mean_std_sem_n.csv"
FEATURE_DP_CSV_NAME = "all_seeds_feature_dp_test_subgroup_metrics_mean_std_sem_n.csv"

OUTPUT_DIR = NON_IID_ROOT / "subgroup_metrics"

ACCURACY_TABLE_DIR = OUTPUT_DIR / "per_feature_accuracy_table"
ROC_AUC_TABLE_DIR = OUTPUT_DIR / "per_feature_roc_auc_table"
LONG_TABLE_DIR = OUTPUT_DIR / "per_feature_long_tables"

EPSILONS = [1, 2, 4, 8, 16, 32, 64]

TEST_SPLIT_NAME = "test_final_model"

NON_PRIVATE_EXPERIMENT_KEYS = [
    "fmri_only",
    "fmri_plus_sensitive_without_anat_func",
]

FULL_DP_EXPERIMENT_KEY = "full_dp_without_anat_func"
FEATURE_DP_EXPERIMENT_KEY = "feature_dp_without_anat_func"

EXPECTED_EXPERIMENT_ORDER = [
    "fmri_only",
    "fmri_plus_sensitive_without_anat_func",
    "full_dp_without_anat_func",
    "feature_dp_without_anat_func",
]

METRICS_TO_MAKE_WIDE_TABLES = {
    "accuracy": ACCURACY_TABLE_DIR,
    "roc_auc": ROC_AUC_TABLE_DIR,
}

CLEAN_OLD_UNUSED_OUTPUT_FOLDERS = True
CLEAN_TARGET_OUTPUT_FOLDERS = True

OLD_UNUSED_FOLDERS = [
    OUTPUT_DIR / "per_feature_tables",
    OUTPUT_DIR / "per_feature_accuracy_mean_tables",
]


# =========================================================
# Basic helpers
# =========================================================

def safe_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def epsilon_label(eps) -> str:
    if pd.isna(eps):
        return "non_private"

    eps_str = str(eps).strip()

    if eps_str.lower() in {"non_private", "none", "nan"}:
        return "non_private"

    try:
        eps_float = float(eps_str)
        if eps_float.is_integer():
            return str(int(eps_float))
        return str(eps_float)
    except Exception:
        return eps_str


def epsilon_sort_value(eps) -> float:
    eps = epsilon_label(eps)

    if eps == "non_private":
        return -1.0

    try:
        return float(eps)
    except Exception:
        return 1e18


def require_columns(df: pd.DataFrame, required: Iterable[str], csv_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"CSV path: {csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    return pd.read_csv(path)


def filter_test_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        return df.copy()

    return df[df["split"].astype(str) == TEST_SPLIT_NAME].copy()


def epsilon_dir_candidates(root: Path, eps: int | float) -> list[Path]:
    eps_str = epsilon_label(eps)

    candidates = [root / f"e{eps_str}"]

    try:
        eps_float = float(eps)
        if eps_float.is_integer():
            candidates.append(root / f"e{int(eps_float)}")
    except Exception:
        pass

    unique = []
    seen = set()

    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


# =========================================================
# Cleanup
# =========================================================

def cleanup_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_OLD_UNUSED_OUTPUT_FOLDERS:
        for folder in OLD_UNUSED_FOLDERS:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Removed old unused folder: {folder}")

    if CLEAN_TARGET_OUTPUT_FOLDERS:
        for folder in [
            ACCURACY_TABLE_DIR,
            ROC_AUC_TABLE_DIR,
            LONG_TABLE_DIR,
        ]:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Removed old target folder: {folder}")

    ACCURACY_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ROC_AUC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LONG_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Loading and normalization
# =========================================================

def normalize_subgroup_df(
    df: pd.DataFrame,
    *,
    source_csv: Path,
    model_family: str,
    forced_target_epsilon: Optional[int | float | str],
) -> pd.DataFrame:
    require_columns(
        df,
        required=[
            "experiment_key",
            "feature",
            "group",
        ],
        csv_path=source_csv,
    )

    df = filter_test_split(df)

    if forced_target_epsilon is None:
        if "target_epsilon" not in df.columns:
            df["target_epsilon"] = "non_private"
        else:
            df["target_epsilon"] = df["target_epsilon"].fillna("non_private")
    else:
        df["target_epsilon"] = epsilon_label(forced_target_epsilon)

    df["target_epsilon"] = df["target_epsilon"].apply(epsilon_label)

    df["experiment_key"] = df["experiment_key"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["group"] = df["group"].astype(str)

    df["model_family"] = model_family
    df["source_csv"] = str(source_csv)

    return df


def load_non_private_metrics() -> pd.DataFrame:
    df = read_csv_checked(NON_PRIV_CSV)

    df = normalize_subgroup_df(
        df,
        source_csv=NON_PRIV_CSV,
        model_family="non_private",
        forced_target_epsilon=None,
    )

    df = df[df["experiment_key"].isin(NON_PRIVATE_EXPERIMENT_KEYS)].copy()

    return df


def load_private_metrics_by_epsilon(
    *,
    root: Path,
    csv_name: str,
    epsilons: list[int | float],
    expected_experiment_key: str,
    model_family: str,
) -> pd.DataFrame:
    frames = []
    missing = []

    for eps in epsilons:
        selected_csv = None

        for eps_dir in epsilon_dir_candidates(root, eps):
            candidate = eps_dir / csv_name

            if candidate.exists():
                selected_csv = candidate
                break

        if selected_csv is None:
            missing.append(str(root / f"e{epsilon_label(eps)}" / csv_name))
            continue

        df = read_csv_checked(selected_csv)

        df = normalize_subgroup_df(
            df,
            source_csv=selected_csv,
            model_family=model_family,
            forced_target_epsilon=eps,
        )

        df = df[df["experiment_key"].astype(str) == expected_experiment_key].copy()

        frames.append(df)

    if missing:
        warnings.warn(
            "Some private CSVs were not found and will be skipped:\n"
            + "\n".join(missing)
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def add_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    exp_order = {
        exp_key: idx
        for idx, exp_key in enumerate(EXPECTED_EXPERIMENT_ORDER)
    }

    df["_experiment_order"] = (
        df["experiment_key"]
        .map(exp_order)
        .fillna(9999)
        .astype(int)
    )

    df["_epsilon_order"] = df["target_epsilon"].apply(epsilon_sort_value)

    return df


def load_all_metrics() -> pd.DataFrame:
    frames = []

    frames.append(load_non_private_metrics())

    full_dp_df = load_private_metrics_by_epsilon(
        root=FULL_DP_ROOT,
        csv_name=FULL_DP_CSV_NAME,
        epsilons=EPSILONS,
        expected_experiment_key=FULL_DP_EXPERIMENT_KEY,
        model_family="full_dp",
    )

    if not full_dp_df.empty:
        frames.append(full_dp_df)

    feature_dp_df = load_private_metrics_by_epsilon(
        root=FEATURE_DP_ROOT,
        csv_name=FEATURE_DP_CSV_NAME,
        epsilons=EPSILONS,
        expected_experiment_key=FEATURE_DP_EXPERIMENT_KEY,
        model_family="feature_dp",
    )

    if not feature_dp_df.empty:
        frames.append(feature_dp_df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = add_sort_columns(combined_df)

    combined_df = combined_df.sort_values(
        by=[
            "feature",
            "group",
            "_experiment_order",
            "_epsilon_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return combined_df


# =========================================================
# Column inference
# =========================================================

def infer_metric_value_columns(df: pd.DataFrame) -> list[str]:
    identifier_cols = {
        "experiment_key",
        "experiment_name",
        "split",
        "feature",
        "group",
        "target_epsilon",
        "source_csv",
        "model_family",
        "_experiment_order",
        "_epsilon_order",
        "best_lr",
        "best_clip_norm_private",
    }

    value_cols = []

    for col in df.columns:
        if col in identifier_cols:
            continue

        if (
            col.endswith("_mean")
            or col.endswith("_std")
            or col.endswith("_sem")
            or col.endswith("_n")
            or col == "n_seeds"
        ):
            value_cols.append(col)

    priority_cols = [
        "n_seeds",
        "n_samples_mean",
        "n_samples_std",
        "n_samples_sem",
        "n_samples_n",
        "n_control_mean",
        "n_control_std",
        "n_control_sem",
        "n_control_n",
        "n_autism_mean",
        "n_autism_std",
        "n_autism_sem",
        "n_autism_n",
        "autism_prevalence_mean",
        "autism_prevalence_std",
        "autism_prevalence_sem",
        "autism_prevalence_n",
        "predicted_autism_rate_mean",
        "predicted_autism_rate_std",
        "predicted_autism_rate_sem",
        "predicted_autism_rate_n",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_sem",
        "accuracy_n",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_sem",
        "balanced_accuracy_n",
        "log_loss_mean",
        "log_loss_std",
        "log_loss_sem",
        "log_loss_n",
        "roc_auc_mean",
        "roc_auc_std",
        "roc_auc_sem",
        "roc_auc_n",
        "average_precision_mean",
        "average_precision_std",
        "average_precision_sem",
        "average_precision_n",
        "control_precision_mean",
        "control_precision_std",
        "control_precision_sem",
        "control_precision_n",
        "control_recall_mean",
        "control_recall_std",
        "control_recall_sem",
        "control_recall_n",
        "control_f1-score_mean",
        "control_f1-score_std",
        "control_f1-score_sem",
        "control_f1-score_n",
        "autism_precision_mean",
        "autism_precision_std",
        "autism_precision_sem",
        "autism_precision_n",
        "autism_recall_mean",
        "autism_recall_std",
        "autism_recall_sem",
        "autism_recall_n",
        "autism_f1-score_mean",
        "autism_f1-score_std",
        "autism_f1-score_sem",
        "autism_f1-score_n",
        "cm_true_control_pred_control_mean",
        "cm_true_control_pred_control_std",
        "cm_true_control_pred_control_sem",
        "cm_true_control_pred_control_n",
        "cm_true_control_pred_autism_mean",
        "cm_true_control_pred_autism_std",
        "cm_true_control_pred_autism_sem",
        "cm_true_control_pred_autism_n",
        "cm_true_autism_pred_control_mean",
        "cm_true_autism_pred_control_std",
        "cm_true_autism_pred_control_sem",
        "cm_true_autism_pred_control_n",
        "cm_true_autism_pred_autism_mean",
        "cm_true_autism_pred_autism_std",
        "cm_true_autism_pred_autism_sem",
        "cm_true_autism_pred_autism_n",
    ]

    ordered = [c for c in priority_cols if c in value_cols]
    remaining = [c for c in value_cols if c not in ordered]

    return ordered + remaining


def metric_stat_columns(metric_name: str) -> list[str]:
    return [
        f"{metric_name}_mean",
        f"{metric_name}_std",
        f"{metric_name}_sem",
        f"{metric_name}_n",
    ]


def stat_suffix(metric_col: str, metric_name: str) -> str:
    prefix = f"{metric_name}_"

    if not metric_col.startswith(prefix):
        return metric_col

    return metric_col[len(prefix):]


# =========================================================
# Ordering logic
# =========================================================

def get_expected_row_index() -> pd.MultiIndex:
    rows = []

    rows.append(("fmri_only", "non_private"))
    rows.append(("fmri_plus_sensitive_without_anat_func", "non_private"))

    for eps in EPSILONS:
        rows.append(("full_dp_without_anat_func", epsilon_label(eps)))

    for eps in EPSILONS:
        rows.append(("feature_dp_without_anat_func", epsilon_label(eps)))

    return pd.MultiIndex.from_tuples(
        rows,
        names=[
            "experiment_key",
            "target_epsilon",
        ],
    )


def get_group_order(feature_df: pd.DataFrame) -> list[str]:
    feature = str(feature_df["feature"].iloc[0])

    observed = list(pd.unique(feature_df["group"].astype(str)))

    known_orders = {
        "AGE_AT_SCAN": [
            "<10",
            "10-14",
            "15-19",
            "20-29",
            "30-39",
            "40+",
        ],
        "EYE_STATUS_AT_SCAN": [
            "Eyes Closed",
            "Eyes Open",
            "Missing",
        ],
        "SEX": [
            "Female",
            "Male",
            "Missing",
        ],
        "FIQ": [
            "<85",
            "85-99",
            "100-114",
            "115-129",
            "130+",
            "FIQ Missing",
            "Missing",
        ],
        "PIQ": [
            "<85",
            "85-99",
            "100-114",
            "115-129",
            "130+",
            "PIQ Missing",
            "Missing",
        ],
        "VIQ": [
            "<85",
            "85-99",
            "100-114",
            "115-129",
            "130+",
            "VIQ Missing",
            "Missing",
        ],
        "CURRENT_MED_STATUS": [
            "Missing",
            "No Current Meds",
            "On Current Meds",
        ],
        "HANDEDNESS_CATEGORY": [
            "Left",
            "Missing",
            "Mixed/Ambi",
            "Right",
        ],
    }

    if feature in known_orders:
        ordered = [g for g in known_orders[feature] if g in observed]
        remaining = [g for g in observed if g not in ordered]
        return ordered + remaining

    return observed


# =========================================================
# Wide metric tables
# =========================================================

def make_feature_metric_wide_table(
    feature_df: pd.DataFrame,
    *,
    metric_name: str,
) -> pd.DataFrame:
    feature_df = feature_df.copy()

    required_stat_cols = metric_stat_columns(metric_name)

    missing_metric_cols = [
        c for c in required_stat_cols
        if c not in feature_df.columns
    ]

    if missing_metric_cols:
        raise ValueError(
            f"Cannot create {metric_name} table for feature "
            f"{feature_df['feature'].iloc[0]} because these columns are missing: "
            f"{missing_metric_cols}"
        )

    require_columns(
        feature_df,
        required=[
            "experiment_key",
            "target_epsilon",
            "feature",
            "group",
            *required_stat_cols,
        ],
        csv_path=Path("<combined dataframe>"),
    )

    group_order = get_group_order(feature_df)
    expected_index = get_expected_row_index()

    wide = feature_df.pivot_table(
        index=[
            "experiment_key",
            "target_epsilon",
        ],
        columns="group",
        values=required_stat_cols,
        aggfunc="first",
    )

    wide = wide.reindex(expected_index)

    flat_columns = []

    for metric_col, group in wide.columns:
        suffix = stat_suffix(metric_col, metric_name)
        flat_columns.append(f"{group}_{suffix}")

    wide.columns = flat_columns

    ordered_cols = []

    for group in group_order:
        for metric_col in required_stat_cols:
            suffix = stat_suffix(metric_col, metric_name)
            col = f"{group}_{suffix}"

            if col not in wide.columns:
                wide[col] = np.nan

            ordered_cols.append(col)

    remaining_cols = [c for c in wide.columns if c not in ordered_cols]

    wide = wide[ordered_cols + remaining_cols]
    wide = wide.reset_index()

    return wide


def save_feature_metric_wide_tables(
    combined_df: pd.DataFrame,
    *,
    metric_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    features = list(pd.unique(combined_df["feature"].astype(str)))

    for feature in features:
        feature_df = combined_df[combined_df["feature"].astype(str) == feature].copy()

        try:
            metric_wide = make_feature_metric_wide_table(
                feature_df,
                metric_name=metric_name,
            )
        except ValueError as exc:
            warnings.warn(str(exc))
            continue

        feature_safe = safe_filename(feature)

        output_path = (
            output_dir
            / f"{feature_safe}_subgroup_table_{metric_name}_wide.csv"
        )

        metric_wide.to_csv(output_path, index=False)

        manifest_rows.append(
            {
                "feature": feature,
                "table_type": f"{metric_name}_wide",
                "metric": metric_name,
                "path": str(output_path),
                "n_rows": len(metric_wide),
                "n_columns": len(metric_wide.columns),
            }
        )

    return pd.DataFrame(manifest_rows)


# =========================================================
# Per-feature long tables
# =========================================================

def make_feature_long_table(
    feature_df: pd.DataFrame,
    *,
    value_columns: list[str],
) -> pd.DataFrame:
    keep_cols = [
        "experiment_key",
        "target_epsilon",
        "feature",
        "group",
        "model_family",
        "experiment_name",
        "split",
        "source_csv",
    ]

    keep_cols = [c for c in keep_cols if c in feature_df.columns]

    metric_cols = [c for c in value_columns if c in feature_df.columns]

    out = feature_df[
        keep_cols
        + metric_cols
        + [
            "_experiment_order",
            "_epsilon_order",
        ]
    ].copy()

    out = out.sort_values(
        by=[
            "_experiment_order",
            "_epsilon_order",
            "feature",
            "group",
        ],
        kind="mergesort",
    )

    out = out.drop(
        columns=[
            "_experiment_order",
            "_epsilon_order",
        ],
        errors="ignore",
    )

    return out


def save_feature_long_tables(
    combined_df: pd.DataFrame,
    *,
    value_columns: list[str],
) -> pd.DataFrame:
    LONG_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    features = list(pd.unique(combined_df["feature"].astype(str)))

    for feature in features:
        feature_df = combined_df[combined_df["feature"].astype(str) == feature].copy()

        long_df = make_feature_long_table(
            feature_df,
            value_columns=value_columns,
        )

        feature_safe = safe_filename(feature)

        output_path = LONG_TABLE_DIR / f"{feature_safe}_subgroup_table_long.csv"

        long_df.to_csv(output_path, index=False)

        manifest_rows.append(
            {
                "feature": feature,
                "table_type": "long",
                "metric": "all_available_metrics",
                "path": str(output_path),
                "n_rows": len(long_df),
                "n_columns": len(long_df.columns),
            }
        )

    return pd.DataFrame(manifest_rows)


# =========================================================
# Combined vertical table
# =========================================================

def save_combined_vertical_table(
    combined_df: pd.DataFrame,
    *,
    value_columns: list[str],
) -> Path:
    keep_cols = [
        "experiment_key",
        "target_epsilon",
        "feature",
        "group",
        "model_family",
        "experiment_name",
        "split",
        "source_csv",
    ]

    keep_cols = [c for c in keep_cols if c in combined_df.columns]
    metric_cols = [c for c in value_columns if c in combined_df.columns]

    out = combined_df[
        keep_cols
        + metric_cols
        + [
            "_experiment_order",
            "_epsilon_order",
        ]
    ].copy()

    out = out.sort_values(
        by=[
            "feature",
            "group",
            "_experiment_order",
            "_epsilon_order",
        ],
        kind="mergesort",
    )

    out = out.drop(
        columns=[
            "_experiment_order",
            "_epsilon_order",
        ],
        errors="ignore",
    )

    output_path = OUTPUT_DIR / "all_models_all_eps_subgroup_metrics_vertical.csv"
    out.to_csv(output_path, index=False)

    return output_path


# =========================================================
# Manifest
# =========================================================

def save_manifest(manifest_frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [
        df for df in manifest_frames
        if df is not None and not df.empty
    ]

    if non_empty:
        manifest_df = pd.concat(non_empty, ignore_index=True)
    else:
        manifest_df = pd.DataFrame(
            columns=[
                "feature",
                "table_type",
                "metric",
                "path",
                "n_rows",
                "n_columns",
            ]
        )

    manifest_path = OUTPUT_DIR / "subgroup_tables_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    return manifest_df


# =========================================================
# Summary
# =========================================================

def print_summary(
    *,
    combined_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    vertical_path: Path,
) -> None:
    print("\nSaved combined vertical table:")
    print(f"  {vertical_path}")

    print("\nSaved per-feature accuracy tables:")
    print(f"  {ACCURACY_TABLE_DIR}")

    print("\nSaved per-feature ROC-AUC tables:")
    print(f"  {ROC_AUC_TABLE_DIR}")

    print("\nSaved per-feature long tables:")
    print(f"  {LONG_TABLE_DIR}")

    print("\nSaved manifest:")
    print(f"  {OUTPUT_DIR / 'subgroup_tables_manifest.csv'}")

    print("\nSummary:")
    print(f"  Combined rows        : {len(combined_df)}")
    print(f"  Combined columns     : {len(combined_df.columns)}")
    print(f"  Number of features   : {combined_df['feature'].nunique()}")
    print(f"  Number of output CSVs: {len(manifest_df)}")

    print("\nExperiment keys and epsilons found:")

    for exp_key in EXPECTED_EXPERIMENT_ORDER:
        sub = combined_df[combined_df["experiment_key"] == exp_key]

        if sub.empty:
            print(f"  {exp_key}: NOT FOUND")
            continue

        eps_found = sorted(
            sub["target_epsilon"].dropna().unique(),
            key=epsilon_sort_value,
        )

        print(f"  {exp_key}: {eps_found}")


# =========================================================
# Main
# =========================================================

def main() -> None:
    cleanup_outputs()

    combined_df = load_all_metrics()

    value_columns = infer_metric_value_columns(combined_df)

    vertical_path = save_combined_vertical_table(
        combined_df,
        value_columns=value_columns,
    )

    manifest_frames = []

    for metric_name, metric_output_dir in METRICS_TO_MAKE_WIDE_TABLES.items():
        metric_manifest = save_feature_metric_wide_tables(
            combined_df,
            metric_name=metric_name,
            output_dir=metric_output_dir,
        )

        manifest_frames.append(metric_manifest)

    long_manifest = save_feature_long_tables(
        combined_df,
        value_columns=value_columns,
    )

    manifest_frames.append(long_manifest)

    manifest_df = save_manifest(manifest_frames)

    print_summary(
        combined_df=combined_df,
        manifest_df=manifest_df,
        vertical_path=vertical_path,
    )


if __name__ == "__main__":
    main()