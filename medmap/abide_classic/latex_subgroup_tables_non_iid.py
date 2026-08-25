#!/usr/bin/env python3
"""
Generate LaTeX subgroup tables from per-feature long CSVs.

Input:
    <ABIDE_OUTPUT_ROOT>/non_iid_fixed_split/subgroup_metrics/per_feature_long_tables/
        {feature}_subgroup_table_long.csv

Outputs:
    latex_tables/accuracy/{feature}_accuracy_table.tex
    latex_tables/roc_auc/{feature}_roc_auc_table.tex

Options:
    TRANSPOSE = False
        Rows    = feature groups
        Columns = fMRI, fMRI+sensitive, epsilon Full-DP/Feature-DP

    TRANSPOSE = True
        Columns = feature groups
        Rows    = fMRI, fMRI+sensitive, epsilon Full-DP/Feature-DP

    USE_PERCENTAGE = True
        Values are multiplied by 100 before printing.

    USE_PERCENTAGE = False
        Values are printed in default decimal form.

    ADD_PERCENT_SIGN = True
        Adds \\% after mean and std values.

    ADD_PERCENT_SIGN = False
        Does not add percentage sign.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ported from "8. summarizer_fixed.ipynb". Output root is resolved from
# ABIDE_OUTPUT_ROOT / ABIDE_DATA_DIR (see medmap/abide_classic/data.py).
# ---------------------------------------------------------------------------
from medmap.abide_classic import data as _abide_data

_ABIDE_DATA_ROOT, _ABIDE_OUTPUT_ROOT = _abide_data.resolve_roots()
_O = str(_ABIDE_OUTPUT_ROOT)

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------
TRANSPOSE = False

USE_PERCENTAGE = True
ADD_PERCENT_SIGN = False

DECIMALS_DECIMAL = 3
DECIMALS_PERCENTAGE = 1

MISSING_CELL = "--"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(
    f"{_O}/non_iid_fixed_split/subgroup_metrics"
)

INPUT_DIR = BASE_DIR / "per_feature_long_tables"

ACCURACY_OUT_DIR = BASE_DIR / "latex_tables" / "accuracy"
ROC_AUC_OUT_DIR = BASE_DIR / "latex_tables" / "roc_auc"


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------
FEATURES = [
    "AGE_AT_SCAN",
    "CURRENT_MED_STATUS",
    "EYE_STATUS_AT_SCAN",
    "HANDEDNESS_CATEGORY",
    "SEX",
    "SITE_ID",
    "FIQ",
    "FIQ_TEST_TYPE",
    "PIQ",
    "PIQ_TEST_TYPE",
    "VIQ",
    "VIQ_TEST_TYPE",
]


# ---------------------------------------------------------------------
# Preferred group ordering
# ---------------------------------------------------------------------
GROUP_ORDER = {
    "AGE_AT_SCAN": ["<10", "10-14", "15-19", "20-29", "30-39", "40+"],

    "FIQ": ["<85", "85-99", "100-114", "115-129", "130+", "FIQ Missing", "Missing"],
    "PIQ": ["<85", "85-99", "100-114", "115-129", "130+", "PIQ Missing", "Missing"],
    "VIQ": ["<85", "85-99", "100-114", "115-129", "130+", "VIQ Missing", "Missing"],

    "CURRENT_MED_STATUS": ["No Current Meds", "On Current Meds", "Missing"],
    "EYE_STATUS_AT_SCAN": ["Eyes Open", "Eyes Closed", "Missing"],
    "HANDEDNESS_CATEGORY": ["Right", "Left", "Mixed/Ambi", "Missing"],
    "SEX": ["Male", "Female", "Missing"],
}


# ---------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------
LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(x: object) -> str:
    if pd.isna(x):
        return ""

    s = str(x)
    return "".join(LATEX_SPECIAL_CHARS.get(ch, ch) for ch in s)


def fmt_epsilon(eps: object) -> str:
    try:
        val = float(eps)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    except Exception:
        return latex_escape(eps)


def natural_key(s: object):
    s = str(s)
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", s)
    ]


def is_missing_like(x: object) -> bool:
    s = str(x).strip().lower()
    return s in {"missing", "nan", "none", ""} or s.endswith(" missing")


def group_sort_key(feature: str, group: object):
    group_s = str(group)

    explicit_order = GROUP_ORDER.get(feature, [])
    if group_s in explicit_order:
        return (0, explicit_order.index(group_s), "")

    if is_missing_like(group_s):
        return (2, 10_000, natural_key(group_s))

    m_lt = re.match(r"^\s*<\s*(-?\d+(?:\.\d+)?)\s*$", group_s)
    if m_lt:
        return (1, float(m_lt.group(1)) - 1e-6, group_s)

    m_range = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$",
        group_s,
    )
    if m_range:
        return (1, float(m_range.group(1)), group_s)

    m_plus = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*\+\s*$", group_s)
    if m_plus:
        return (1, float(m_plus.group(1)), group_s)

    return (1, 1_000_000, natural_key(group_s))


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def col_as_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series(default, index=df.index, dtype=str)


def target_epsilon_as_float(x: object) -> Optional[float]:
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    if s in {"non_private", "non-private", "np", "none", "nan", ""}:
        return None

    try:
        return float(s)
    except ValueError:
        return None


def unique_numeric_epsilons(df: pd.DataFrame) -> list[float]:
    eps = []

    for x in df["target_epsilon"].unique():
        val = target_epsilon_as_float(x)
        if val is not None and np.isfinite(val):
            eps.append(float(val))

    return sorted(set(eps))


def sorted_groups(df: pd.DataFrame, feature: str) -> list[str]:
    return sorted(
        df["group"].dropna().astype(str).unique(),
        key=lambda g: group_sort_key(feature, g),
    )


def first_match(df: pd.DataFrame, mask: pd.Series) -> Optional[pd.Series]:
    sub = df.loc[mask].copy()

    if sub.empty:
        return None

    return sub.iloc[0]


def get_nonprivate_row(
    df: pd.DataFrame,
    group: object,
    kind: str,
) -> Optional[pd.Series]:
    group_mask = df["group"].astype(str) == str(group)
    non_private_mask = df["target_epsilon"].apply(target_epsilon_as_float).isna()

    experiment_key = col_as_str(df, "experiment_key").str.lower()
    experiment_name = col_as_str(df, "experiment_name").str.lower()

    if kind == "fmri_only":
        kind_mask = (
            (experiment_key == "fmri_only")
            | (experiment_name == "fmri only")
        )

    elif kind == "fmri_sensitive":
        kind_mask = (
            (experiment_key == "fmri_plus_sensitive_without_anat_func")
            | experiment_name.str.contains("fmri \\+ sensitive", regex=True, na=False)
        )

    else:
        raise ValueError(f"Unknown non-private kind: {kind}")

    return first_match(df, group_mask & non_private_mask & kind_mask)


def get_dp_row(
    df: pd.DataFrame,
    group: object,
    epsilon: float,
    model_family: str,
) -> Optional[pd.Series]:
    eps_float = df["target_epsilon"].apply(target_epsilon_as_float)

    group_mask = df["group"].astype(str) == str(group)
    eps_mask = eps_float.notna() & np.isclose(eps_float.astype(float), float(epsilon))

    model_family_col = col_as_str(df, "model_family").str.lower()
    experiment_key = col_as_str(df, "experiment_key").str.lower()

    family_mask = model_family_col == model_family.lower()

    if model_family.lower() == "full_dp":
        exp_key_mask = experiment_key.str.contains("full_dp", na=False)
    elif model_family.lower() == "feature_dp":
        exp_key_mask = experiment_key.str.contains("feature_dp", na=False)
    else:
        exp_key_mask = pd.Series(False, index=df.index)

    return first_match(df, group_mask & eps_mask & (family_mask | exp_key_mask))


# ---------------------------------------------------------------------
# Metric formatting
# ---------------------------------------------------------------------
def metric_multiplier() -> float:
    return 100.0 if USE_PERCENTAGE else 1.0


def metric_decimals() -> int:
    return DECIMALS_PERCENTAGE if USE_PERCENTAGE else DECIMALS_DECIMAL


def format_number(x: float) -> str:
    decimals = metric_decimals()
    suffix = r"\%" if ADD_PERCENT_SIGN else ""
    return rf"{x:.{decimals}f}{suffix}"


def format_metric_cell(
    row: Optional[pd.Series],
    metric_mean_col: str,
    metric_std_col: str,
) -> str:
    """
    Format like:
        0.679_{± 0.007}

    In LaTeX:
        $0.679_{\scriptscriptstyle \pm 0.007}$

    Here the standard deviation is shown as a smaller subscript.
    """
    if row is None:
        return MISSING_CELL

    mean = pd.to_numeric(row.get(metric_mean_col, np.nan), errors="coerce")
    std = pd.to_numeric(row.get(metric_std_col, np.nan), errors="coerce")

    if pd.isna(mean) or not np.isfinite(mean):
        return MISSING_CELL

    mult = metric_multiplier()

    mean_val = float(mean) * mult
    mean_str = format_number(mean_val)

    if pd.isna(std) or not np.isfinite(std):
        return rf"${mean_str}$"

    std_val = float(std) * mult
    std_str = format_number(std_val)

    return rf"${mean_str}_{{\scriptscriptstyle {std_str}}}$"


# ---------------------------------------------------------------------
# Row/column specification helpers
# ---------------------------------------------------------------------
def method_rows(epsilons: list[float]) -> list[dict]:
    rows = [
        {
            "label": "fMRI",
            "kind": "non_private",
            "nonprivate_kind": "fmri_only",
            "epsilon": None,
            "model_family": None,
        },
        {
            "label": r"fMRI+sensitive info ($\epsilon=\infty$)",
            "kind": "non_private",
            "nonprivate_kind": "fmri_sensitive",
            "epsilon": None,
            "model_family": None,
        },
    ]

    for eps in epsilons:
        rows.append(
            {
                "label": rf"$\epsilon={fmt_epsilon(eps)}$ Full DP",
                "kind": "dp",
                "nonprivate_kind": None,
                "epsilon": eps,
                "model_family": "full_dp",
            }
        )
        rows.append(
            {
                "label": rf"$\epsilon={fmt_epsilon(eps)}$ Feature DP",
                "kind": "dp",
                "nonprivate_kind": None,
                "epsilon": eps,
                "model_family": "feature_dp",
            }
        )

    return rows


def get_cell_for_method(
    df: pd.DataFrame,
    group: str,
    method: dict,
    mean_col: str,
    std_col: str,
) -> str:
    if method["kind"] == "non_private":
        row = get_nonprivate_row(df, group, method["nonprivate_kind"])

    elif method["kind"] == "dp":
        row = get_dp_row(
            df=df,
            group=group,
            epsilon=method["epsilon"],
            model_family=method["model_family"],
        )

    else:
        raise ValueError(f"Unknown method kind: {method['kind']}")

    return format_metric_cell(row, mean_col, std_col)


# ---------------------------------------------------------------------
# Normal table
# ---------------------------------------------------------------------
def make_normal_latex_table(
    df: pd.DataFrame,
    feature: str,
    metric: str,
    metric_display_name: str,
    out_tex_path: Path,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    epsilons = unique_numeric_epsilons(df)
    groups = sorted_groups(df, feature)

    n_cols = 1 + 2 + 2 * len(epsilons)
    col_spec = "l" + "c" * (n_cols - 1)

    header_1_cells = [
        latex_escape(feature),
        "fMRI",
        r"fMRI+sensitive info ($\epsilon=\infty$)",
    ]

    for eps in epsilons:
        header_1_cells.append(
            rf"\multicolumn{{2}}{{c}}{{$\epsilon={fmt_epsilon(eps)}$}}"
        )

    header_2_cells = ["", "", ""]
    for _ in epsilons:
        header_2_cells.extend(["Full DP", "Feature DP"])

    cmidrules = []
    for j, _ in enumerate(epsilons):
        start_col = 4 + 2 * j
        end_col = start_col + 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")

    body_lines = []

    for group in groups:
        row_cells = [latex_escape(group)]

        row_cells.append(
            format_metric_cell(
                get_nonprivate_row(df, group, "fmri_only"),
                mean_col,
                std_col,
            )
        )

        row_cells.append(
            format_metric_cell(
                get_nonprivate_row(df, group, "fmri_sensitive"),
                mean_col,
                std_col,
            )
        )

        for eps in epsilons:
            row_cells.append(
                format_metric_cell(
                    get_dp_row(df, group, eps, "full_dp"),
                    mean_col,
                    std_col,
                )
            )
            row_cells.append(
                format_metric_cell(
                    get_dp_row(df, group, eps, "feature_dp"),
                    mean_col,
                    std_col,
                )
            )

        body_lines.append(" & ".join(row_cells) + r" \\")

    write_latex_table(
        out_tex_path=out_tex_path,
        col_spec=col_spec,
        header_lines=[
            " & ".join(header_1_cells) + r" \\",
            "".join(cmidrules),
            " & ".join(header_2_cells) + r" \\",
        ],
        body_lines=body_lines,
        feature=feature,
        metric=metric,
        metric_display_name=metric_display_name,
        transposed=False,
    )


# ---------------------------------------------------------------------
# Transposed table
# ---------------------------------------------------------------------
def make_transposed_latex_table(
    df: pd.DataFrame,
    feature: str,
    metric: str,
    metric_display_name: str,
    out_tex_path: Path,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    epsilons = unique_numeric_epsilons(df)
    groups = sorted_groups(df, feature)
    methods = method_rows(epsilons)

    n_cols = 1 + len(groups)
    col_spec = "l" + "c" * (n_cols - 1)

    header_cells = [latex_escape(feature)] + [latex_escape(g) for g in groups]
    header_lines = [" & ".join(header_cells) + r" \\"]

    body_lines = []

    for method in methods:
        row_cells = [method["label"]]

        for group in groups:
            row_cells.append(
                get_cell_for_method(
                    df=df,
                    group=group,
                    method=method,
                    mean_col=mean_col,
                    std_col=std_col,
                )
            )

        body_lines.append(" & ".join(row_cells) + r" \\")

    write_latex_table(
        out_tex_path=out_tex_path,
        col_spec=col_spec,
        header_lines=header_lines,
        body_lines=body_lines,
        feature=feature,
        metric=metric,
        metric_display_name=metric_display_name,
        transposed=True,
    )


# ---------------------------------------------------------------------
# Generic LaTeX writer
# ---------------------------------------------------------------------
def write_latex_table(
    out_tex_path: Path,
    col_spec: str,
    header_lines: list[str],
    body_lines: list[str],
    feature: str,
    metric: str,
    metric_display_name: str,
    transposed: bool,
) -> None:
    orientation_text = "transposed " if transposed else ""

    value_unit = "percentage points" if USE_PERCENTAGE else "decimal form"

    caption = (
        rf"{orientation_text}{metric_display_name} subgroup results for "
        rf"{latex_escape(feature)}. Entries are mean with standard deviation "
        rf"across seeds, reported in {value_unit}."
    )

    label_suffix = "_transposed" if transposed else ""
    label = rf"tab:{feature.lower()}_{metric}{label_suffix}"

    table_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_lines,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]

    out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text("\n".join(table_lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main table builder
# ---------------------------------------------------------------------
def make_latex_table(
    df: pd.DataFrame,
    feature: str,
    metric: str,
    metric_display_name: str,
    out_tex_path: Path,
    transpose: bool = TRANSPOSE,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    required_cols = {"target_epsilon", "group", mean_col, std_col}
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise ValueError(
            f"{out_tex_path.name}: missing required columns: {sorted(missing_cols)}"
        )

    if transpose:
        make_transposed_latex_table(
            df=df,
            feature=feature,
            metric=metric,
            metric_display_name=metric_display_name,
            out_tex_path=out_tex_path,
        )
    else:
        make_normal_latex_table(
            df=df,
            feature=feature,
            metric=metric,
            metric_display_name=metric_display_name,
            out_tex_path=out_tex_path,
        )


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def read_feature_csv(feature: str) -> pd.DataFrame:
    csv_path = INPUT_DIR / f"{feature}_subgroup_table_long.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "feature" not in df.columns:
        df["feature"] = feature

    return df


def generate_all_tables(features: Iterable[str] = FEATURES) -> None:
    ACCURACY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROC_AUC_OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    skipped = []

    suffix = "_transposed" if TRANSPOSE else ""

    for feature in features:
        try:
            df = read_feature_csv(feature)

            accuracy_tex_path = ACCURACY_OUT_DIR / f"{feature}_accuracy_table{suffix}.tex"
            roc_auc_tex_path = ROC_AUC_OUT_DIR / f"{feature}_roc_auc_table{suffix}.tex"

            make_latex_table(
                df=df,
                feature=feature,
                metric="accuracy",
                metric_display_name="Accuracy",
                out_tex_path=accuracy_tex_path,
                transpose=TRANSPOSE,
            )

            make_latex_table(
                df=df,
                feature=feature,
                metric="roc_auc",
                metric_display_name="ROC-AUC",
                out_tex_path=roc_auc_tex_path,
                transpose=TRANSPOSE,
            )

            generated.extend([accuracy_tex_path, roc_auc_tex_path])

            print(f"[OK] {feature}")
            print(f"     accuracy -> {accuracy_tex_path}")
            print(f"     roc_auc  -> {roc_auc_tex_path}")

        except Exception as e:
            skipped.append((feature, str(e)))
            print(f"[SKIP] {feature}: {e}")

    print("\nDone.")
    print(f"TRANSPOSE = {TRANSPOSE}")
    print(f"USE_PERCENTAGE = {USE_PERCENTAGE}")
    print(f"ADD_PERCENT_SIGN = {ADD_PERCENT_SIGN}")
    print(f"Generated {len(generated)} .tex files.")

    if skipped:
        print("\nSkipped features:")
        for feature, reason in skipped:
            print(f"  - {feature}: {reason}")


if __name__ == "__main__":
    generate_all_tables()