#!/usr/bin/env python3
"""ABIDE I logistic regression — classic backend (sklearn / NumPy).

Faithful, modular packaging of the original fixed-split research scripts:

    --method lbfgs    non-private sklearn LogisticRegression (L-BFGS).
                      Also CREATES AND PERSISTS the fixed split and tunes the
                      global C on the tuning seed — run this arm first.
    --method full_dp  full-batch DP gradient descent (per-example clipping of
                      the ENTIRE gradient + Gaussian noise; NumPy/Numba).
    --method feat_dp  Feature-DP: only the private phenotype residual gradient
                      is clipped and noised; the fMRI component stays exact.

    --split iid       stratified random 70/10/20 split (seeds 81-90).
    --split non_iid   SITE_ID-disjoint split via StratifiedGroupKFold
                      (train/val/test share no acquisition sites; seeds 21-30).

Privacy accounting: RDP (Google dp-accounting) with delta = n^-1.1 and the
noise multiplier solved by Brent root-finding for the target epsilon.
The epsilon sweep is run one target at a time, e.g.:

    for EPS in 1 2 4 8 16 32 64; do
        python -m scripts.train_abide_classic --split iid --method full_dp \
            --data-dir /data/abide/ABIDE_PCP_full --epsilon $EPS
    done

All defaults reproduce the reported runs. Results land in
<output-root>/{iid,non_iid}_fixed_split/{non_priv,full_dp,feat_dp}[/e<eps>]/...
Aggregate with scripts/summarize_abide.py afterwards. See docs/abide.md.
"""

from __future__ import annotations

import argparse
import importlib
import os

from scripts._bootstrap import setup_project_paths

setup_project_paths()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=["iid", "non_iid"])
    ap.add_argument("--method", required=True, choices=["lbfgs", "full_dp", "feat_dp"])

    # Paths
    ap.add_argument("--data-dir", required=True,
                    help="ABIDE PCP root (contains ABIDE_pcp/ and fc_features/).")
    ap.add_argument("--output-root", default=None,
                    help="Default: <data-dir>/outputs")
    ap.add_argument("--fc-dir", default=None,
                    help="Default: <data-dir>/fc_features (X_conn.npy, y.npy, meta.json).")
    ap.add_argument("--phenotypic-csv", default=None,
                    help="Default: <data-dir>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--feature-json", default=None,
                    help="recommended_feature_lists.json (default: data dir, then the packaged copy).")
    ap.add_argument("--fetch-raw", action="store_true",
                    help="Run nilearn fetch_abide_pcp first (downloads raw derivatives if missing).")

    # Experiment overrides (defaults inside the modules reproduce the paper runs)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Target epsilon (full_dp / feat_dp only).")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Full-batch GD iterations (default: 2000 iid / 3000 non_iid).")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated final-run seeds (default: 81-90 iid / 21-30 non_iid).")
    ap.add_argument("--tuning-seed", type=int, default=None,
                    help="Hyperparameter-tuning + fixed-split seed (default: 81 iid / 21 non_iid).")
    ap.add_argument("--lr-grid", default=None,
                    help="Comma-separated learning-rate grid override.")
    ap.add_argument("--clip-grid", default=None,
                    help="Comma-separated private clip-norm grid override (feat_dp only).")
    ap.add_argument("--tuned-c", type=float, default=None,
                    help="Logistic-regression C used by the DP arms "
                         "(default: 0.1 iid / 0.3 non_iid, tuned by the lbfgs arm).")

    args = ap.parse_args()

    if args.method == "lbfgs" and args.epsilon is not None:
        ap.error("--epsilon only applies to full_dp / feat_dp.")

    env = {
        "ABIDE_DATA_DIR": args.data_dir,
        "ABIDE_OUTPUT_ROOT": args.output_root,
        "ABIDE_FC_DIR": args.fc_dir,
        "PHENOTYPIC_CSV": args.phenotypic_csv,
        "FEATURE_JSON": args.feature_json,
        "ABIDE_FETCH_RAW": "1" if args.fetch_raw else None,
        "ABIDE_TARGET_EPSILON": args.epsilon,
        "ABIDE_ITERATIONS": args.iterations,
        "ABIDE_SEEDS": args.seeds,
        "ABIDE_TUNING_SEED": args.tuning_seed,
        "ABIDE_LR_GRID": args.lr_grid,
        "ABIDE_CLIP_GRID": args.clip_grid,
        "ABIDE_TUNED_C": args.tuned_c,
    }
    for key, value in env.items():
        if value is not None:
            os.environ[key] = str(value)

    module_name = f"medmap.abide_classic.{args.split}_fixed_split_{args.method}"
    print(f"[train_abide_classic] running {module_name}")
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
