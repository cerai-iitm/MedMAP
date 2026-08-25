"""Shared data loading for the ABIDE classic backend.

Every classic experiment module (sklearn LBFGS baseline, NumPy/Numba
full-batch DP-GD, and Feature-DP) starts from the same inputs:

- ``X_conn.npy`` / ``y.npy`` / ``meta.json``: precomputed functional
  connectivity features (MSDL atlas, vectorized correlations) produced by
  ``scripts/extract_abide_features.py``.
- the ABIDE I phenotypic CSV (``Phenotypic_V1_0b_preprocessed1.csv``),
  downloaded by nilearn's ``fetch_abide_pcp`` into
  ``<ABIDE_DATA_DIR>/ABIDE_pcp/``.

Configuration is via environment variables so that the experiment modules
(which mirror the original research scripts) stay byte-comparable to the
originals:

    ABIDE_DATA_DIR     (required) ABIDE PCP root directory.
    ABIDE_FC_DIR       (optional) default: <ABIDE_DATA_DIR>/fc_features
    ABIDE_OUTPUT_ROOT  (optional) default: <ABIDE_DATA_DIR>/outputs
    PHENOTYPIC_CSV     (optional) default:
                       <ABIDE_DATA_DIR>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv
    ABIDE_FETCH_RAW    (optional, "1" to enable) run nilearn's
                       fetch_abide_pcp exactly like the original scripts did
                       (downloads raw derivatives if missing). Off by default;
                       only the FC features + phenotypic CSV are needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _require_env_dir(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} must be set (see docs/abide.md). Example:\n"
            f"    export {name}=/data/abide/ABIDE_PCP_full"
        )
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{name}={path} does not exist or is not a directory.")
    return path


def resolve_roots() -> Tuple[Path, Path]:
    """Return (data_root, output_root) from the environment."""
    data_root = _require_env_dir("ABIDE_DATA_DIR")
    output_root = Path(os.environ.get("ABIDE_OUTPUT_ROOT", str(data_root / "outputs")))
    output_root.mkdir(parents=True, exist_ok=True)
    return data_root, output_root


def fc_features_dir(data_root: Path) -> Path:
    return Path(os.environ.get("ABIDE_FC_DIR", str(data_root / "fc_features")))


def phenotypic_csv_path(data_root: Path) -> Path:
    return Path(
        os.environ.get(
            "PHENOTYPIC_CSV",
            str(data_root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"),
        )
    )


def feature_json_candidates(data_root: Path) -> list[Path]:
    """Candidate locations for recommended_feature_lists.json.

    Order: explicit FEATURE_JSON env var, the conventional location inside the
    data root, then the copy packaged with medmap (always present).
    """
    packaged = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "abide"
        / "resources"
        / "recommended_feature_lists.json"
    )
    candidates: list[Path] = []
    env_value = os.environ.get("FEATURE_JSON", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(data_root / "abide_feature_selection_outputs" / "recommended_feature_lists.json")
    candidates.append(packaged)
    return candidates


def maybe_fetch_abide(data_root: Path):
    """Optionally run nilearn's fetch_abide_pcp exactly like the original
    scripts' preamble (ABIDE_FETCH_RAW=1). Returns the nilearn Bunch or None.
    """
    if os.environ.get("ABIDE_FETCH_RAW", "0").strip() not in {"1", "true", "True"}:
        return None
    from nilearn import datasets as nilearn_datasets

    return nilearn_datasets.fetch_abide_pcp(
        data_dir=str(data_root),
        pipeline="cpac",
        derivatives=["func_preproc"],
        quality_checked=False,
        band_pass_filtering=False,
        global_signal_regression=False,
    )


def load_abide_inputs(
    data_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, object]:
    """Load (X_conn, y, meta, abide) — the module-level globals the original
    experiment scripts created in their dataset-creation preamble.

    ``abide`` is the nilearn Bunch when ABIDE_FETCH_RAW=1, else None (the
    phenotypic CSV must then exist; see phenotypic_csv_path()).
    """
    if data_root is None:
        data_root, _ = resolve_roots()

    abide = maybe_fetch_abide(data_root)

    save_dir = fc_features_dir(data_root)
    x_conn_path = save_dir / "X_conn.npy"
    if not x_conn_path.exists():
        raise FileNotFoundError(
            f"Missing {x_conn_path}. Generate the FC features first:\n"
            "    python -m scripts.extract_abide_features --data-dir <ABIDE_DATA_DIR>\n"
            "(see docs/abide.md)"
        )

    X_conn = np.load(x_conn_path)
    y = np.load(save_dir / "y.npy")

    with open(save_dir / "meta.json", "r") as f:
        meta = json.load(f)

    print("X_conn shape:", X_conn.shape)
    print("y shape:", y.shape)
    print("Metadata:", meta)
    # Print min and max values in X_conn to check for any anomalies
    print("X_conn min value:", np.min(X_conn))
    print("X_conn max value:", np.max(X_conn))

    return X_conn, y, meta, abide


def env_override_float(env_name: str, default: float) -> float:
    value = os.environ.get(env_name, "").strip()
    return float(value) if value else float(default)


def env_override_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name, "").strip()
    return int(value) if value else int(default)


def env_override_int_list(env_name: str, default: list[int]) -> list[int]:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return list(default)
    return [int(x) for x in value.replace(",", " ").split()]


def env_override_float_list(env_name: str, default: list[float]) -> list[float]:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return list(default)
    return [float(x) for x in value.replace(",", " ").split()]
