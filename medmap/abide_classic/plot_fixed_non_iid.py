"""Privacy-utility plots for the non-IID fixed-split summary CSV.

See scripts/summarize_abide.py for the CLI wrapper.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Ported from "8. summarizer_fixed.ipynb". Output root is resolved from
# ABIDE_OUTPUT_ROOT / ABIDE_DATA_DIR (see medmap/abide_classic/data.py).
# ---------------------------------------------------------------------------
from medmap.abide_classic import data as _abide_data

_ABIDE_DATA_ROOT, _ABIDE_OUTPUT_ROOT = _abide_data.resolve_roots()
_O = str(_ABIDE_OUTPUT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = Path(
    f"{_O}/non_iid_fixed_split/non_iid_all_eps_vertical_summary.csv"
)

OUT_DIR = CSV_PATH.parent / "privacy_utility_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert metrics from fraction to percentage.
# Example: 0.68 -> 68
SCALE_TO_PERCENT = True

FIGSIZE = (5.2, 4.6)
DPI = 300

# Legend order requested
LEGEND_ORDER = [
    "Feature DP",
    "Full DP",
    "fMRI + sensitive info",
    "fMRI only",
]

STYLE = {
    "Feature DP": {
        "color": "tab:red",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 2.5,
        "markersize": 8,
        "shade_alpha": 0.18,
    },
    "Full DP": {
        "color": "tab:green",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 2.5,
        "markersize": 8,
        "shade_alpha": 0.18,
    },
    "fMRI + sensitive info": {
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "fMRI only": {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
    },
}


# =========================================================
# HELPERS
# =========================================================
def clean_experiment_name(name: str) -> str:
    name = str(name).strip()

    mapping = {
        "full DP": "Full DP",
        # "Full DP": "Full DP",
        "Feature DP": "Feature DP",
        "fMRI only": "fMRI only",
        "fMRI + sensitive info": "fMRI + sensitive info",
    }

    return mapping.get(name, name)


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "target_epsilon",
        "experiment_name",
        "accuracy",
        "accuracy_std",
        "accuracy_sem",
        "roc_auc",
        "roc_auc_std",
        "roc_auc_sem",
    }

    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.copy()
    df["experiment_name"] = df["experiment_name"].map(clean_experiment_name)
    df["target_epsilon"] = pd.to_numeric(df["target_epsilon"], errors="coerce")

    metric_cols = [
        "accuracy",
        "accuracy_std",
        "accuracy_sem",
        "roc_auc",
        "roc_auc_std",
        "roc_auc_sem",
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_epsilon", "experiment_name"])

    if SCALE_TO_PERCENT:
        for col in metric_cols:
            df[col] = 100.0 * df[col]

    return df


def get_baseline_row(df: pd.DataFrame, experiment_name: str) -> pd.Series:
    sub = df[df["experiment_name"] == experiment_name].copy()

    if sub.empty:
        raise ValueError(f"Could not find baseline experiment: {experiment_name}")

    # Non-private baselines are repeated for every epsilon.
    # Take the first one after sorting.
    return sub.sort_values("target_epsilon").iloc[0]


def get_curve_df(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    sub = df[df["experiment_name"] == experiment_name].copy()

    if sub.empty:
        raise ValueError(f"Could not find experiment: {experiment_name}")

    return sub.sort_values("target_epsilon")


def format_epsilon_ticks(eps_values: np.ndarray) -> list[str]:
    labels = []

    for x in eps_values:
        if float(x).is_integer():
            labels.append(str(int(x)))
        else:
            labels.append(str(x))

    return labels


def add_horizontal_baseline(
    ax,
    *,
    eps_min: float,
    eps_max: float,
    value: float,
    label: str,
) -> None:
    """
    Plain horizontal baseline line only.
    No std/sem shading for non-private baselines.
    """
    style = STYLE[label]

    ax.hlines(
        y=value,
        xmin=eps_min,
        xmax=eps_max,
        label=label,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
    )


def add_dp_curve_with_shaded_error(
    ax,
    curve_df: pd.DataFrame,
    *,
    metric_col: str,
    error_col: str,
    label: str,
) -> None:
    """
    Plot DP curve with shaded uncertainty region.

    Shaded region = metric ± std/sem depending on error_col.
    """
    style = STYLE[label]

    x = curve_df["target_epsilon"].to_numpy(dtype=float)
    y = curve_df[metric_col].to_numpy(dtype=float)
    err = curve_df[error_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    err = err[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    err = err[order]

    ax.plot(
        x,
        y,
        label=label,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
    )

    valid_err = np.isfinite(err) & (err >= 0)

    if valid_err.any():
        y_lower = y - err
        y_upper = y + err

        ax.fill_between(
            x,
            y_lower,
            y_upper,
            color=style["color"],
            alpha=style["shade_alpha"],
            linewidth=0,
        )


def enforce_legend_order(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ordered_handles = []
    ordered_labels = []

    for label in LEGEND_ORDER:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="best",
        fontsize=10,
        frameon=True,
    )


def apply_reference_like_x_axis(ax, eps_values: np.ndarray) -> None:
    """
    Makes x-axis spacing similar to the attached reference plot.

    Equal spacing is given to multiplicative jumps:
        1 -> 2 -> 4 -> 8 -> 16

    This is a log2 x-axis.
    """
    eps_values = np.sort(np.asarray(eps_values, dtype=float))

    if np.any(eps_values <= 0):
        raise ValueError(
            "All epsilon values must be positive for log-scaled x-axis."
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_values)
    ax.set_xticklabels(format_epsilon_ticks(eps_values))

    # Padding similar to the reference figure.
    ax.set_xlim(
        float(eps_values.min()) / np.sqrt(2),
        float(eps_values.max()) * np.sqrt(2),
    )


def plot_privacy_utility(
    df: pd.DataFrame,
    *,
    metric_col: str,
    error_col: str,
    metric_label: str,
    error_label: str,
    out_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    eps_values = np.sort(df["target_epsilon"].dropna().unique().astype(float))

    if len(eps_values) == 0:
        raise ValueError("No valid epsilon values found.")

    eps_min = float(eps_values.min())
    eps_max = float(eps_values.max())

    # -----------------------------------------------------
    # 1. Non-private baselines as plain horizontal lines
    # -----------------------------------------------------
    fmri_sensitive = get_baseline_row(df, "fMRI + sensitive info")
    fmri_only = get_baseline_row(df, "fMRI only")

    add_horizontal_baseline(
        ax,
        eps_min=eps_min,
        eps_max=eps_max,
        value=float(fmri_sensitive[metric_col]),
        label="fMRI + sensitive info",
    )

    add_horizontal_baseline(
        ax,
        eps_min=eps_min,
        eps_max=eps_max,
        value=float(fmri_only[metric_col]),
        label="fMRI only",
    )

    # -----------------------------------------------------
    # 2. DP methods with shaded std/sem bands
    # -----------------------------------------------------
    full_dp_df = get_curve_df(df, "Full DP")
    feature_dp_df = get_curve_df(df, "Feature DP")

    add_dp_curve_with_shaded_error(
        ax,
        full_dp_df,
        metric_col=metric_col,
        error_col=error_col,
        label="Full DP",
    )

    add_dp_curve_with_shaded_error(
        ax,
        feature_dp_df,
        metric_col=metric_col,
        error_col=error_col,
        label="Feature DP",
    )

    # -----------------------------------------------------
    # Axis formatting
    # -----------------------------------------------------
    apply_reference_like_x_axis(ax, eps_values)

    ax.set_xlabel(r"$\epsilon$", fontsize=13)
    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(
        f"Non-IID {metric_label} vs " + r"$\epsilon$" + f" with {error_label}",
        fontsize=13,
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.45)

    enforce_legend_order(ax)

    fig.tight_layout()

    png_path = OUT_DIR / f"{out_name}.png"
    pdf_path = OUT_DIR / f"{out_name}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.show()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    df = load_summary(CSV_PATH)

    suffix = "percent" if SCALE_TO_PERCENT else "fraction"

    # 1. Accuracy vs epsilon with std shading
    plot_privacy_utility(
        df,
        metric_col="accuracy",
        error_col="accuracy_std",
        metric_label="Test Accuracy (%)" if SCALE_TO_PERCENT else "Test Accuracy",
        error_label="standard deviation",
        out_name=f"abide_non_iid_accuracy_privacy_utility_with_std_shaded_log2x_{suffix}",
    )

    # 2. Accuracy vs epsilon with sem shading
    plot_privacy_utility(
        df,
        metric_col="accuracy",
        error_col="accuracy_sem",
        metric_label="Test Accuracy (%)" if SCALE_TO_PERCENT else "Test Accuracy",
        error_label="standard error",
        out_name=f"abide_non_iid_accuracy_privacy_utility_with_sem_shaded_log2x_{suffix}",
    )

    # 3. AuROC vs epsilon with std shading
    plot_privacy_utility(
        df,
        metric_col="roc_auc",
        error_col="roc_auc_std",
        metric_label="Test AuROC (%)" if SCALE_TO_PERCENT else "Test AuROC",
        error_label="standard deviation",
        out_name=f"abide_non_iid_auroc_privacy_utility_with_std_shaded_log2x_{suffix}",
    )

    # 4. AuROC vs epsilon with sem shading
    plot_privacy_utility(
        df,
        metric_col="roc_auc",
        error_col="roc_auc_sem",
        metric_label="Test AuROC (%)" if SCALE_TO_PERCENT else "Test AuROC",
        error_label="standard error",
        out_name=f"abide_non_iid_auroc_privacy_utility_with_sem_shaded_log2x_{suffix}",
    )

    print(f"\nAll plots saved inside:\n{OUT_DIR}")


if __name__ == "__main__":
    main()