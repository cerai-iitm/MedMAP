"""Aggregate non-IID fixed-split results into vertical/wide summary CSVs.

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
import json
import re
import math
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(f"{_O}/non_iid_fixed_split")

NON_PRIV_DIR = BASE_DIR / "non_priv"
FULL_DP_DIR = BASE_DIR / "full_dp"
FEAT_DP_DIR = BASE_DIR / "feat_dp"

NON_PRIV_FILE = "all_modes_test_metrics_across_seeds_mean_std_sem_n.csv"
FULL_DP_FILE = "all_seeds_dp_gd_test_metrics_mean_std_sem_n.csv"
FEAT_DP_FILE = "all_seeds_feature_dp_test_metrics_mean_std_sem_n.csv"

# Output files
OUT_VERTICAL_ALL_EPS = BASE_DIR / "non_iid_all_eps_vertical_summary.csv"
OUT_WIDE_ALL_EPS = BASE_DIR / "non_iid_all_eps_wide_summary.csv"

# If you want the table label exactly like your screenshot, keep this.
# Set to None if you want the script to infer seed folders automatically.
SEED_LIST_OVERRIDE = list(range(41, 51))

# If iteration count is not found in CSV/JSON, manually specify here.
# Example:
# ITERATIONS_BY_EPS = {1: 5000, 2: 2000, 4: 2000}
ITERATIONS_BY_EPS: dict[float | int, int | float] = {}

# Full-DP clipping label. Your screenshot shows Cpub=1.
FULL_DP_CLIP_LABEL = "Cpub=1"

ORDERED_EXPERIMENTS = [
    "fMRI only",
    "fMRI + sensitive info",
    "full DP",
    "Feature DP",
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "roc_auc",
    "average_precision",
]

STATS = ["mean", "std", "sem", "n"]


# =========================================================
# GENERIC HELPERS
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


def fmt_num(x, ndigits: int = 6) -> str:
    x = safe_float(x)
    if not np.isfinite(x):
        return ""
    return f"{x:.{ndigits}g}"


def parse_epsilon_from_dirname(name: str):
    """
    Parses:
        e1   -> 1
        e2   -> 2
        e16  -> 16
        eps8 -> 8
    """
    m = re.search(r"(?:e|eps)([0-9]+(?:\.[0-9]+)?)", name.lower())
    if not m:
        return None

    val = float(m.group(1))
    return int(val) if val.is_integer() else val


def eps_sort_key(eps):
    return float(eps)


def compress_seed_list(seeds: list[int]) -> str:
    """
    [41,42,43,44,45,46,47,48,49,50] -> '41 to 50'
    [13,41,42,43,89] -> '13, 41 to 43, 89'
    """
    if not seeds:
        return ""

    seeds = sorted(set(int(s) for s in seeds))
    ranges = []

    start = prev = seeds[0]

    for s in seeds[1:]:
        if s == prev + 1:
            prev = s
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start} to {prev}")
            start = prev = s

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start} to {prev}")

    return ", ".join(ranges)


def infer_seed_summary(*dirs: Path, fallback_n: int | None = None) -> str:
    if SEED_LIST_OVERRIDE is not None:
        return f"Average over seeds = {compress_seed_list(SEED_LIST_OVERRIDE)}"

    seeds = []

    for d in dirs:
        if d is None or not d.exists():
            continue

        for p in d.glob("seed*"):
            m = re.search(r"seed_?(\d+)", p.name)
            if m:
                seeds.append(int(m.group(1)))

    seeds = sorted(set(seeds))

    if seeds:
        return f"Average over seeds = {compress_seed_list(seeds)}"

    if fallback_n is not None:
        return f"Average over seeds, n={fallback_n}"

    return "Average over seeds"


def discover_epsilon_dirs() -> dict:
    """
    Finds e1, e2, e4, ... under full_dp and feat_dp.

    Returns:
        {
            1: {
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

    return dict(sorted(found.items(), key=lambda kv: eps_sort_key(kv[0])))


def get_first_existing_value(row: pd.Series, candidates: list[str], default=np.nan):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


def get_stat(row: pd.Series, metric: str, stat: str):
    """
    Supports both:
        accuracy_mean, accuracy_std, accuracy_sem, accuracy_n
    and:
        accuracy
    for mean.
    """
    if stat == "mean":
        for c in [f"{metric}_mean", metric]:
            if c in row.index and pd.notna(row[c]):
                return safe_float(row[c])

    col = f"{metric}_{stat}"

    if col in row.index and pd.notna(row[col]):
        if stat == "n":
            return int(row[col])
        return safe_float(row[col])

    return np.nan


def build_metric_columns(row: pd.Series) -> dict:
    out = {}

    for metric in METRICS:
        for stat in STATS:
            colname = f"{metric}_{stat}" if stat != "mean" else metric
            out[colname] = get_stat(row, metric, stat)

    return out


def get_fallback_n_from_row(row: pd.Series) -> int | None:
    for metric in METRICS:
        col = f"{metric}_n"
        if col in row.index and pd.notna(row[col]):
            return int(row[col])
    return None


# =========================================================
# ROW SELECTION HELPERS
# =========================================================
def normalize_text(x) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def select_row_by_candidates(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    preferred_cols: list[str] | None = None,
) -> pd.Series | None:
    """
    Tries exact matching first, then substring matching over text columns.
    """
    if df.empty:
        return None

    candidates_norm = [normalize_text(c) for c in candidates]

    if preferred_cols is None:
        preferred_cols = [
            "experiment_name",
            "experiment_key",
            "mode",
            "method",
            "setting",
            "split",
        ]

    text_cols = [
        c
        for c in preferred_cols
        if c in df.columns
    ]

    extra_text_cols = [
        c
        for c in df.columns
        if df[c].dtype == object and c not in text_cols
    ]

    text_cols = text_cols + extra_text_cols

    # Exact match
    for col in text_cols:
        s = df[col].astype(str).map(normalize_text)
        for cand in candidates_norm:
            mask = s == cand
            if mask.any():
                return df.loc[mask].iloc[0]

    # Contains match
    for col in text_cols:
        s = df[col].astype(str).map(normalize_text)
        for cand in candidates_norm:
            mask = s.str.contains(cand, regex=False, na=False)
            if mask.any():
                return df.loc[mask].iloc[0]

    return None


# =========================================================
# ITERATION INFERENCE
# =========================================================
def find_key_recursive(obj: Any, candidate_keys: set[str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in candidate_keys:
                return v

        for v in obj.values():
            found = find_key_recursive(v, candidate_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for v in obj:
            found = find_key_recursive(v, candidate_keys)
            if found is not None:
                return found

    return None


def infer_iterations_from_csv(path: Path) -> float:
    if not path.exists():
        return np.nan

    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return np.nan

    candidate_cols = [
        "iterations",
        "num_iterations",
        "n_iterations",
        "max_iterations",
        "max_iter",
        "T",
        "steps",
    ]

    for c in candidate_cols:
        if c in df.columns and pd.notna(df[c].iloc[0]):
            return safe_float(df[c].iloc[0])

    return np.nan


def infer_iterations_from_jsons(eps_dir: Path) -> float:
    if eps_dir is None or not eps_dir.exists():
        return np.nan

    candidate_keys = {
        "iterations",
        "num_iterations",
        "n_iterations",
        "max_iterations",
        "max_iter",
        "t",
        "steps",
    }

    json_files = list(eps_dir.glob("*.json"))

    for fp in json_files:
        try:
            with open(fp, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        val = find_key_recursive(payload, candidate_keys)
        if val is not None:
            val = safe_float(val)
            if np.isfinite(val):
                return val

    return np.nan


def infer_iterations_for_eps(eps, full_dir: Path | None, feat_dir: Path | None) -> float:
    if eps in ITERATIONS_BY_EPS:
        return ITERATIONS_BY_EPS[eps]

    eps_float = float(eps)
    if eps_float in ITERATIONS_BY_EPS:
        return ITERATIONS_BY_EPS[eps_float]

    candidate_files = []

    if full_dir is not None:
        candidate_files.append(full_dir / FULL_DP_FILE)

    if feat_dir is not None:
        candidate_files.append(feat_dir / FEAT_DP_FILE)

    for fp in candidate_files:
        val = infer_iterations_from_csv(fp)
        if np.isfinite(val):
            return val

    for d in [full_dir, feat_dir]:
        val = infer_iterations_from_jsons(d)
        if np.isfinite(val):
            return val

    return np.nan


# =========================================================
# OUTPUT ROW BUILDERS
# =========================================================
def make_output_row(
    *,
    target_epsilon,
    iterations,
    seed_summary: str,
    experiment_name: str,
    clip: str,
    tuned_params: str,
    source_type: str,
    source_path: Path,
    input_row: pd.Series,
) -> dict:
    row = {
        "target_epsilon": target_epsilon,
        "iterations": iterations,
        "seed": seed_summary,
        "experiment_name": experiment_name,
        "clip": clip,
        "tuned_params": tuned_params,
        "source_type": source_type,
        "source_path": str(source_path),
    }

    row.update(build_metric_columns(input_row))

    return row


def get_non_private_rows(
    *,
    epsilon,
    iterations,
    seed_summary: str,
) -> list[dict]:
    path = NON_PRIV_DIR / NON_PRIV_FILE
    df = read_csv_checked(path)

    rows = []

    fmri_only_candidates = [
        "fMRI only",
        "fmri_only",
        "fmri",
        "fmri only",
    ]

    fmri_sensitive_candidates = [
        "fMRI + sensitive info",
        "fmri_plus_sensitive_info",
        "fmri_plus_sensitive_without_anat_func",
        "fmri sensitive",
        "fmri+sensitive",
        "fmri_plus_sensitive",
    ]

    specs = [
        ("fMRI only", fmri_only_candidates),
        ("fMRI + sensitive info", fmri_sensitive_candidates),
    ]

    for clean_name, candidates in specs:
        r = select_row_by_candidates(df, candidates)

        if r is None:
            print(f"WARNING: could not find non-private row for: {clean_name}")
            continue

        c_val = get_first_existing_value(
            r,
            [
                "best_C_mean",
                "best_C",
                "C_mean",
                "C",
                "model__C_mean",
                "model__C",
            ],
        )

        tuned_params = f"C={fmt_num(c_val)}" if pd.notna(c_val) else ""

        rows.append(
            make_output_row(
                target_epsilon=epsilon,
                iterations=iterations,
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


def get_full_dp_row(
    *,
    epsilon,
    iterations,
    eps_dir: Path,
    seed_summary: str,
) -> dict | None:
    path = eps_dir / FULL_DP_FILE

    if not path.exists():
        print(f"WARNING: full-DP file missing: {path}")
        return None

    df = read_csv_checked(path)

    r = select_row_by_candidates(
        df,
        [
            "full DP",
            "full_dp",
            "dp_gd",
            "full dp",
        ],
    )

    if r is None:
        r = df.iloc[0]

    lr = get_first_existing_value(
        r,
        [
            "best_lr_mean",
            "best_lr",
            "learning_rate_mean",
            "learning_rate",
            "lr_mean",
            "lr",
            "selected_lr",
            "tuned_lr",
        ],
    )

    tuned_params = f"LR={fmt_num(lr)}" if pd.notna(lr) else ""

    return make_output_row(
        target_epsilon=epsilon,
        iterations=iterations,
        seed_summary=seed_summary,
        experiment_name="full DP",
        clip=FULL_DP_CLIP_LABEL,
        tuned_params=tuned_params,
        source_type="full_dp",
        source_path=path,
        input_row=r,
    )


def get_feature_dp_row(
    *,
    epsilon,
    iterations,
    eps_dir: Path,
    seed_summary: str,
) -> dict | None:
    path = eps_dir / FEAT_DP_FILE

    if not path.exists():
        print(f"WARNING: Feature-DP file missing: {path}")
        return None

    df = read_csv_checked(path)

    r = select_row_by_candidates(
        df,
        [
            "Feature DP",
            "feature_dp",
            "feat_dp",
            "feature dp",
        ],
    )

    if r is None:
        # Usually this file contains only one aggregate row.
        r = df.iloc[0]

    lr = get_first_existing_value(
        r,
        [
            "best_lr_mean",
            "best_lr",
            "learning_rate_mean",
            "learning_rate",
            "lr_mean",
            "lr",
            "selected_lr",
            "tuned_lr",
        ],
    )

    cpriv = get_first_existing_value(
        r,
        [
            "best_clip_norm_private_mean",
            "best_clip_norm_private",
            "clip_norm_private_mean",
            "clip_norm_private",
            "private_clip_norm_mean",
            "private_clip_norm",
            "best_private_clip_norm_mean",
            "best_private_clip_norm",
            "Cpriv_mean",
            "Cpriv",
        ],
    )

    tuned_params = f"LR={fmt_num(lr)}" if pd.notna(lr) else ""
    clip = f"Cpriv={fmt_num(cpriv)}" if pd.notna(cpriv) else ""

    return make_output_row(
        target_epsilon=epsilon,
        iterations=iterations,
        seed_summary=seed_summary,
        experiment_name="Feature DP",
        clip=clip,
        tuned_params=tuned_params,
        source_type="feature_dp",
        source_path=path,
        input_row=r,
    )


# =========================================================
# OUTPUT FORMATTING
# =========================================================
def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: i for i, name in enumerate(ORDERED_EXPERIMENTS)}

    df = df.copy()
    df["_order"] = df["experiment_name"].map(order_map).fillna(999)
    df = df.sort_values(["target_epsilon", "_order"]).drop(columns=["_order"])

    return df


def make_screenshot_style_wide(df_eps: pd.DataFrame) -> pd.DataFrame:
    """
    Per-epsilon CSV close to your screenshot.

    Includes std/sem/n also, since your new input files contain mean/std/sem/n.
    """
    cols = [
        "target_epsilon",
        "iterations",
        "seed",
        "experiment_name",
        "clip",
        "tuned_params",

        "accuracy",
        "accuracy_std",
        "accuracy_sem",
        "accuracy_n",

        "roc_auc",
        "roc_auc_std",
        "roc_auc_sem",
        "roc_auc_n",
    ]

    available = [c for c in cols if c in df_eps.columns]
    out = df_eps[available].copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns

    for c in numeric_cols:
        if c.endswith("_n"):
            continue
        out[c] = out[c].round(6)

    return out


def make_wide_one_row_per_epsilon(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extra wide format:
        one row per epsilon,
        columns like:
            Feature_DP_accuracy,
            Feature_DP_accuracy_sem,
            full_DP_roc_auc,
            ...
    """
    metric_cols = []

    for metric in METRICS:
        metric_cols.append(metric)
        metric_cols.append(f"{metric}_std")
        metric_cols.append(f"{metric}_sem")
        metric_cols.append(f"{metric}_n")

    rows = []

    for eps, g in df_all.groupby("target_epsilon", dropna=False):
        row = {"target_epsilon": eps}

        if "iterations" in g.columns:
            row["iterations"] = g["iterations"].dropna().iloc[0] if g["iterations"].notna().any() else np.nan

        if "seed" in g.columns:
            row["seed"] = g["seed"].iloc[0]

        for _, r in g.iterrows():
            prefix = (
                str(r["experiment_name"])
                .replace("+", "plus")
                .replace(" ", "_")
                .replace("-", "_")
                .replace("__", "_")
            )

            row[f"{prefix}_clip"] = r.get("clip", "")
            row[f"{prefix}_tuned_params"] = r.get("tuned_params", "")

            for m in metric_cols:
                if m in r.index:
                    row[f"{prefix}_{m}"] = r[m]

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("target_epsilon")

    numeric_cols = out.select_dtypes(include=[np.number]).columns

    for c in numeric_cols:
        if c.endswith("_n") or c in {"target_epsilon", "iterations"}:
            continue
        out[c] = out[c].round(6)

    return out


def get_fallback_n_from_available_files(
    full_dir: Path | None,
    feat_dir: Path | None,
) -> int | None:
    candidate_files = [
        NON_PRIV_DIR / NON_PRIV_FILE,
    ]

    if full_dir is not None:
        candidate_files.append(full_dir / FULL_DP_FILE)

    if feat_dir is not None:
        candidate_files.append(feat_dir / FEAT_DP_FILE)

    for fp in candidate_files:
        if not fp.exists():
            continue

        try:
            df = pd.read_csv(fp, nrows=1)
        except Exception:
            continue

        if df.empty:
            continue

        n = get_fallback_n_from_row(df.iloc[0])
        if n is not None:
            return n

    return None


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    eps_info = discover_epsilon_dirs()

    if not eps_info:
        raise RuntimeError(
            "No epsilon directories found under:\n"
            f"  {FULL_DP_DIR}\n"
            f"  {FEAT_DP_DIR}"
        )

    all_rows = []

    print("\nDiscovered epsilon folders:")
    for eps, info in eps_info.items():
        print(f"  epsilon={eps}: {info}")

    for eps, info in eps_info.items():
        eps_label = info["eps_label"]
        full_dir = info.get("full_dp_dir")
        feat_dir = info.get("feat_dp_dir")

        iterations = infer_iterations_for_eps(eps, full_dir, feat_dir)

        fallback_n = get_fallback_n_from_available_files(full_dir, feat_dir)

        seed_summary = infer_seed_summary(
            NON_PRIV_DIR,
            full_dir if full_dir is not None else Path("__missing__"),
            feat_dir if feat_dir is not None else Path("__missing__"),
            fallback_n=fallback_n,
        )

        rows = []

        rows.extend(
            get_non_private_rows(
                epsilon=eps,
                iterations=iterations,
                seed_summary=seed_summary,
            )
        )

        if full_dir is not None:
            r = get_full_dp_row(
                epsilon=eps,
                iterations=iterations,
                eps_dir=full_dir,
                seed_summary=seed_summary,
            )
            if r is not None:
                rows.append(r)

        if feat_dir is not None:
            r = get_feature_dp_row(
                epsilon=eps,
                iterations=iterations,
                eps_dir=feat_dir,
                seed_summary=seed_summary,
            )
            if r is not None:
                rows.append(r)

        df_eps = order_rows(pd.DataFrame(rows))

        out_wide_eps = BASE_DIR / f"non_iid_wide_summary_{eps_label}.csv"
        wide_eps = make_screenshot_style_wide(df_eps)
        wide_eps.to_csv(out_wide_eps, index=False)

        print(f"\nSaved per-epsilon wide CSV for epsilon={eps}:")
        print(out_wide_eps)

        all_rows.extend(rows)

    df_all = order_rows(pd.DataFrame(all_rows))

    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if c.endswith("_n") or c in {"target_epsilon", "iterations"}:
            continue
        df_all[c] = df_all[c].round(6)

    df_all.to_csv(OUT_VERTICAL_ALL_EPS, index=False)

    print("\nSaved vertical all-eps CSV:")
    print(OUT_VERTICAL_ALL_EPS)

    df_wide_all = make_wide_one_row_per_epsilon(df_all)
    df_wide_all.to_csv(OUT_WIDE_ALL_EPS, index=False)

    print("\nSaved combined wide all-eps CSV:")
    print(OUT_WIDE_ALL_EPS)

    print("\nDone.")


if __name__ == "__main__":
    main()