#!/usr/bin/env python3
"""Extract ABIDE I functional-connectivity features (X_conn.npy / y.npy).

Reproduces the feature-extraction stage used by every ABIDE experiment:

1. Download / locate the ABIDE Preprocessed Connectomes Project (PCP) data
   with nilearn's ``fetch_abide_pcp`` (C-PAC pipeline, ``func_preproc``
   derivative, no quality filtering, no band-pass filtering, no global
   signal regression).
2. Extract per-subject time series with the MSDL probabilistic atlas
   (39 regions) using ``NiftiMapsMasker`` (z-score standardization, detrend,
   low_pass=0.1 Hz, high_pass=0.01 Hz, t_r=2.0 s).
3. Compute vectorized correlation connectivity with
   ``ConnectivityMeasure(kind="correlation", vectorize=True,
   discard_diagonal=True)`` -> 39*38/2 = 741 features per subject.
4. Save ``X_conn.npy`` (n_subjects x 741), ``y.npy``
   (DX_GROUP == 1 -> autism = 1, control = 0) and ``meta.json`` into
   ``<data-dir>/fc_features/``.

The full raw download is ~100+ GB and extraction takes several hours; the
result is a 12 MB feature directory that every downstream experiment reuses.

Usage:

    python -m scripts.extract_abide_features --data-dir /data/abide/ABIDE_PCP_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._bootstrap import setup_project_paths

setup_project_paths()

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True,
                    help="ABIDE PCP root; raw data is downloaded here if missing.")
    ap.add_argument("--out-dir", default=None,
                    help="Feature output directory (default: <data-dir>/fc_features).")
    ap.add_argument("--t-r", type=float, default=2.0,
                    help="Repetition time passed to the masker (default 2.0; replace "
                         "with subject/site-specific TR if you handle it explicitly).")
    args = ap.parse_args()

    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiMapsMasker

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "fc_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Fetching ABIDE PCP (downloads raw derivatives if missing)...")
    abide = datasets.fetch_abide_pcp(
        data_dir=str(data_dir),
        pipeline="cpac",
        derivatives=["func_preproc"],
        quality_checked=False,
        band_pass_filtering=False,
        global_signal_regression=False,
    )

    print("[2/4] Extracting MSDL time series...")
    atlas = datasets.fetch_atlas_msdl(data_dir=str(data_dir))

    masker = NiftiMapsMasker(
        maps_img=atlas.maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=args.t_r,
    )

    time_series = []
    n_total = len(abide.func_preproc)
    for i, img in enumerate(abide.func_preproc, start=1):
        time_series.append(masker.fit_transform(img))
        if i == 1 or i % 25 == 0 or i == n_total:
            print(f"    {i}/{n_total} subjects")

    print("[3/4] Computing vectorized correlation connectivity...")
    conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    X_conn = conn.fit_transform(time_series)

    # DX_GROUP coding in ABIDE I: 1 = autism, 2 = control.
    y = (abide.phenotypic["DX_GROUP"] == 1).astype(int).to_numpy()

    print("[4/4] Saving features...")
    np.save(out_dir / "X_conn.npy", X_conn)
    np.save(out_dir / "y.npy", y)
    meta = {
        "X_conn_shape": list(X_conn.shape),
        "y_shape": list(y.shape),
        "dtype_X_conn": str(X_conn.dtype),
        "dtype_y": str(y.dtype),
        "description": "Functional connectivity features from ABIDE",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved X_conn {X_conn.shape}, y {y.shape} to {out_dir}")


if __name__ == "__main__":
    main()
