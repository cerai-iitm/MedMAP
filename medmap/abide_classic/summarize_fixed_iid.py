"""Aggregate IID fixed-split results into vertical/wide summary CSVs.

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
import re
import math
import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================
IID_DIR = Path(f"{_O}/iid_fixed_split")

NON_PRIV_DIR = IID_DIR / "non_priv"
FULL_DP_DIR = IID_DIR / "full_dp"
FEAT_DP_DIR = IID_DIR / "feat_dp"

NON_PRIV_FILE = "all_modes_test_metrics_across_seeds_mean_std.csv"
FULL_DP_FILE = "all_seeds_dp_gd_test_metrics_mean_sem.csv"
FEAT_DP_FILE = "all_seeds_feature_dp_test_metrics_mean_std_sem.csv"

OUT_VERTICAL_ALL_EPS = IID_DIR / "iid_all_eps_vertical_summary.csv"
OUT_WIDE_ALL_EPS = IID_DIR / "iid_all_eps_wide_summary.csv"

# This is only a label for full DP, because your full-DP summary CSV
# does not seem to store the clipping value directly.
FULL_DP_CLIP_LABEL = "Cpub=1"

# Experiment ordering in output
ORDERED_EXPERIMENTS = [
    "fMRI only",
    "fMRI + sensitive info",
    "full DP",
    "Feature DP",
]


# =========================================================
# HELPERS
# =========================================================
def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x, ndigits: int = 4) -> str:
    """
    Compact formatting for labels like C=0.1, LR=0.04, Cpriv=0.0807.
    """
    x = safe_float(x)
    if not np.isfinite(x):
        return ""
    s = f"{x:.{ndigits}g}"
    return s


def get_metric(row: pd.Series, metric: str, stat: str):
    """
    Example:
        get_metric(row, "accuracy", "mean") -> row["accuracy_mean"]
        get_metric(row, "roc_auc", "sem") -> row["roc_auc_sem"]
    """
    col = f"{metric}_{stat}"
    return safe_float(row[col]) if col in row.index else np.nan


def get_n(row: pd.Series, metric: str) -> int | None:
    col = f"{metric}_n"
    if col not in row.index or pd.isna(row[col]):
        return None
    return int(row[col])


def infer_seed_summary(*dirs: Path, fallback_n: int | None = None) -> str:
    """
    Infers seed range from folders like:
        seed_41, seed_42, ...
        seed41, seed42, ...
    """
    seeds = []

    for d in dirs:
        if not d.exists():
            continue

        for p in d.glob("seed*"):
            m = re.search(r"seed_?(\d+)", p.name)
            if m:
                seeds.append(int(m.group(1)))

    seeds = sorted(set(seeds))

    if seeds:
        if seeds == list(range(min(seeds), max(seeds) + 1)):
            return f"Average over seeds = {min(seeds)} to {max(seeds)}"
        return "Average over seeds = " + ",".join(map(str, seeds))

    if fallback_n is not None:
        return f"Average over seeds, n={fallback_n}"

    return "Average over seeds"


def parse_epsilon_from_dirname(name: str):
    """
    Parses:
        e1      -> 1
        e8      -> 8
        e0.5    -> 0.5
        eps1    -> 1
    """
    m = re.search(r"(?:e|eps)([0-9]+(?:\.[0-9]+)?)", name.lower())
    if not m:
        return None

    val = float(m.group(1))
    if val.is_integer():
        return int(val)
    return val


def discover_epsilon_dirs() -> dict:
    """
    Finds epsilon directories under full_dp and feat_dp.
    Returns:
        {
            eps_value: {
                "eps_label": "e1",
                "full_dp_dir": Path(...),
                "feat_dp_dir": Path(...),
            },
            ...
        }
    """
    found = {}

    for parent, key in [(FULL_DP_DIR, "full_dp_dir"), (FEAT_DP_DIR, "feat_dp_dir")]:
        if not parent.exists():
            continue

        for d in parent.iterdir():
            if not d.is_dir():
                continue

            eps = parse_epsilon_from_dirname(d.name)
            if eps is None:
                continue

            found.setdefault(eps, {"eps_label": d.name})
            found[eps][key] = d

    return dict(sorted(found.items(), key=lambda kv: float(kv[0])))


def row_common(
    *,
    epsilon,
    iterations,
    seed_summary,
    experiment_name,
    clip,
    tuned_params,
    source_type,
    source_path,
    input_row: pd.Series,
) -> dict:
    """
    Builds one clean output row.
    """
    return {
        "target_epsilon": epsilon,
        "iterations": iterations,
        "seed": seed_summary,
        "experiment_name": experiment_name,
        "clip": clip,
        "tuned_params": tuned_params,

        "accuracy": get_metric(input_row, "accuracy", "mean"),
        "accuracy_std": get_metric(input_row, "accuracy", "std"),
        "accuracy_sem": get_metric(input_row, "accuracy", "sem"),
        "accuracy_n": get_n(input_row, "accuracy"),

        "balanced_accuracy": get_metric(input_row, "balanced_accuracy", "mean"),
        "balanced_accuracy_std": get_metric(input_row, "balanced_accuracy", "std"),
        "balanced_accuracy_sem": get_metric(input_row, "balanced_accuracy", "sem"),
        "balanced_accuracy_n": get_n(input_row, "balanced_accuracy"),

        "log_loss": get_metric(input_row, "log_loss", "mean"),
        "log_loss_std": get_metric(input_row, "log_loss", "std"),
        "log_loss_sem": get_metric(input_row, "log_loss", "sem"),
        "log_loss_n": get_n(input_row, "log_loss"),

        "roc_auc": get_metric(input_row, "roc_auc", "mean"),
        "roc_auc_std": get_metric(input_row, "roc_auc", "std"),
        "roc_auc_sem": get_metric(input_row, "roc_auc", "sem"),
        "roc_auc_n": get_n(input_row, "roc_auc"),

        "average_precision": get_metric(input_row, "average_precision", "mean"),
        "average_precision_std": get_metric(input_row, "average_precision", "std"),
        "average_precision_sem": get_metric(input_row, "average_precision", "sem"),
        "average_precision_n": get_n(input_row, "average_precision"),

        "source_type": source_type,
        "source_path": str(source_path),
    }


def get_non_private_rows(epsilon, seed_summary: str) -> list[dict]:
    path = NON_PRIV_DIR / NON_PRIV_FILE
    df = read_csv_checked(path)

    rows = []

    mapping = {
        "fmri_only": "fMRI only",
        "fmri_plus_sensitive_without_anat_func": "fMRI + sensitive info",
    }

    for exp_key, clean_name in mapping.items():
        sub = df[df["experiment_key"] == exp_key].copy()

        if sub.empty:
            print(f"WARNING: non-private experiment not found: {exp_key}")
            continue

        r = sub.iloc[0]

        best_c = r["best_C_mean"] if "best_C_mean" in r.index else np.nan
        tuned_params = f"C={fmt_num(best_c)}" if pd.notna(best_c) else ""

        rows.append(
            row_common(
                epsilon=epsilon,
                iterations=np.nan,
                seed_summary=seed_summary,
                experiment_name=clean_name,
                clip="None",
                tuned_params=tuned_params,
                source_type="non_private",
                source_path=path,
                input_row=r,
            )
        )

    return rows


def get_full_dp_row(epsilon, eps_dir: Path, seed_summary: str) -> dict | None:
    path = eps_dir / FULL_DP_FILE
    if not path.exists():
        print(f"WARNING: full-DP file missing: {path}")
        return None

    df = read_csv_checked(path)

    if "experiment_key" in df.columns:
        sub = df[df["experiment_key"].astype(str).str.contains("full_dp", case=False)]
        if not sub.empty:
            r = sub.iloc[0]
        else:
            r = df.iloc[0]
    else:
        r = df.iloc[0]

    lr = r["best_lr"] if "best_lr" in r.index else np.nan
    tuned_params = f"LR={fmt_num(lr)}" if pd.notna(lr) else ""

    return row_common(
        epsilon=epsilon,
        iterations=np.nan,
        seed_summary=seed_summary,
        experiment_name="full DP",
        clip=FULL_DP_CLIP_LABEL,
        tuned_params=tuned_params,
        source_type="full_dp",
        source_path=path,
        input_row=r,
    )


def get_feature_dp_row(epsilon, eps_dir: Path, seed_summary: str) -> dict | None:
    path = eps_dir / FEAT_DP_FILE
    if not path.exists():
        print(f"WARNING: Feature-DP file missing: {path}")
        return None

    df = read_csv_checked(path)

    if "split" in df.columns:
        sub = df[df["split"].astype(str).str.contains("test", case=False)]
        r = sub.iloc[0] if not sub.empty else df.iloc[0]
    else:
        r = df.iloc[0]

    lr = r["best_lr"] if "best_lr" in r.index else np.nan

    cpriv = np.nan
    for col in ["best_clip_norm_private", "clip_norm_private"]:
        if col in r.index and pd.notna(r[col]):
            cpriv = r[col]
            break

    iterations = r["iterations"] if "iterations" in r.index else np.nan

    tuned_params = f"LR={fmt_num(lr)}" if pd.notna(lr) else ""
    clip = f"Cpriv={fmt_num(cpriv)}" if pd.notna(cpriv) else ""

    return row_common(
        epsilon=epsilon,
        iterations=iterations,
        seed_summary=seed_summary,
        experiment_name="Feature DP",
        clip=clip,
        tuned_params=tuned_params,
        source_type="feature_dp",
        source_path=path,
        input_row=r,
    )


def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: i for i, name in enumerate(ORDERED_EXPERIMENTS)}
    df = df.copy()
    df["_order"] = df["experiment_name"].map(order_map).fillna(999)
    df = df.sort_values(["target_epsilon", "_order"]).drop(columns=["_order"])
    return df


def make_screenshot_style_wide(df_eps: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the important columns similar to your screenshot.
    CSV cannot store merged cells like Excel, so epsilon/iteration/seed are columns.
    """
    cols = [
        "target_epsilon",
        "iterations",
        "seed",
        "experiment_name",
        "clip",
        "tuned_params",
        "accuracy",
        "accuracy_sem",
        "roc_auc",
        "roc_auc_sem",
    ]

    out = df_eps[cols].copy()

    # Round for clean report-style CSV
    for c in ["accuracy", "accuracy_sem", "roc_auc", "roc_auc_sem"]:
        out[c] = out[c].astype(float).round(4)

    return out


def make_wider_one_row_per_epsilon(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extra combined wide format:
    one row per epsilon, columns like:
        fMRI_only_accuracy, fMRI_only_roc_auc, full_DP_accuracy, ...
    """
    metric_cols = [
        "accuracy",
        "accuracy_sem",
        "roc_auc",
        "roc_auc_sem",
        "balanced_accuracy",
        "balanced_accuracy_sem",
        "log_loss",
        "log_loss_sem",
        "average_precision",
        "average_precision_sem",
    ]

    rows = []

    for eps, g in df_all.groupby("target_epsilon", dropna=False):
        row = {"target_epsilon": eps}

        for _, r in g.iterrows():
            prefix = (
                str(r["experiment_name"])
                .replace(" ", "_")
                .replace("+", "plus")
                .replace("-", "_")
                .replace("__", "_")
            )

            row[f"{prefix}_clip"] = r["clip"]
            row[f"{prefix}_tuned_params"] = r["tuned_params"]

            for m in metric_cols:
                row[f"{prefix}_{m}"] = r[m]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("target_epsilon")


# =========================================================
# MAIN
# =========================================================
def main():
    IID_DIR.mkdir(parents=True, exist_ok=True)

    eps_info = discover_epsilon_dirs()

    if not eps_info:
        raise RuntimeError(
            f"No epsilon directories found under:\n"
            f"  {FULL_DP_DIR}\n"
            f"  {FEAT_DP_DIR}"
        )

    all_rows = []

    for eps, info in eps_info.items():
        eps_label = info["eps_label"]
        full_dir = info.get("full_dp_dir")
        feat_dir = info.get("feat_dp_dir")

        fallback_n = None

        # Try to infer n_seeds from whichever file is available
        for maybe_file in [
            NON_PRIV_DIR / NON_PRIV_FILE,
            full_dir / FULL_DP_FILE if full_dir is not None else None,
            feat_dir / FEAT_DP_FILE if feat_dir is not None else None,
        ]:
            if maybe_file is not None and maybe_file.exists():
                tmp = pd.read_csv(maybe_file, nrows=1)
                if "n_seeds" in tmp.columns:
                    fallback_n = int(tmp["n_seeds"].iloc[0])
                    break

        seed_summary = infer_seed_summary(
            NON_PRIV_DIR,
            full_dir if full_dir is not None else Path("__missing__"),
            feat_dir if feat_dir is not None else Path("__missing__"),
            fallback_n=fallback_n,
        )

        rows = []

        # Non-private rows are repeated for every epsilon report.
        rows.extend(get_non_private_rows(epsilon=eps, seed_summary=seed_summary))

        if full_dir is not None:
            r = get_full_dp_row(epsilon=eps, eps_dir=full_dir, seed_summary=seed_summary)
            if r is not None:
                rows.append(r)

        if feat_dir is not None:
            r = get_feature_dp_row(epsilon=eps, eps_dir=feat_dir, seed_summary=seed_summary)
            if r is not None:
                rows.append(r)

        df_eps = order_rows(pd.DataFrame(rows))

        # Save screenshot-style wide CSV for this epsilon
        wide_eps = make_screenshot_style_wide(df_eps)

        out_wide_eps = IID_DIR / f"iid_wide_summary_{eps_label}.csv"
        wide_eps.to_csv(out_wide_eps, index=False)

        print(f"Saved wide CSV for epsilon={eps}: {out_wide_eps}")

        all_rows.extend(rows)

    # Vertical combined CSV for all epsilons
    df_all = order_rows(pd.DataFrame(all_rows))
    df_all.to_csv(OUT_VERTICAL_ALL_EPS, index=False)
    print(f"\nSaved vertical all-eps CSV: {OUT_VERTICAL_ALL_EPS}")

    # Extra: one-row-per-epsilon wide CSV
    df_wide_all = make_wider_one_row_per_epsilon(df_all)
    df_wide_all.to_csv(OUT_WIDE_ALL_EPS, index=False)
    print(f"Saved combined wide all-eps CSV: {OUT_WIDE_ALL_EPS}")

    print("\nDone.")


if __name__ == "__main__":
    main()