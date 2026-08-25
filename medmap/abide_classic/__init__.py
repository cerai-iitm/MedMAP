"""ABIDE classic backend: faithful, path-parametrized packaging of the
original fixed-split research scripts.

Experiment modules (dispatched by scripts/train_abide_classic.py):

    {iid,non_iid}_fixed_split_lbfgs     non-private sklearn LogisticRegression
                                        baseline; also creates + persists the
                                        fixed split and tunes the global C.
    {iid,non_iid}_fixed_split_full_dp   full-batch DP gradient descent
                                        (per-example clipping of the entire
                                        gradient + Gaussian noise; Numba JIT
                                        with a pure-NumPy fallback).
    {iid,non_iid}_fixed_split_feat_dp   Feature-DP: only the private phenotype
                                        residual gradient is clipped/noised.

Summary modules (dispatched by scripts/summarize_abide.py):

    summarize_fixed_{iid,non_iid}       cross-epsilon summary CSVs
    plot_fixed_{iid,non_iid}            privacy-utility plots
    subgroup_tables_non_iid             per-subgroup metric tables
    latex_subgroup_tables_non_iid       LaTeX table export

Shared configuration and data loading live in ``data`` (environment
variables: ABIDE_DATA_DIR, ABIDE_OUTPUT_ROOT, ABIDE_FC_DIR, PHENOTYPIC_CSV,
FEATURE_JSON, ABIDE_FETCH_RAW, plus the ABIDE_* experiment overrides).

NOTE: the experiment/summary modules intentionally run their workload at
import time (they are packaged research scripts); import them only via the
scripts/ dispatchers, and import ``medmap.abide_classic.data`` freely.
"""
