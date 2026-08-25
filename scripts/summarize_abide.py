#!/usr/bin/env python3
"""Aggregate ABIDE classic-backend results into summary tables and plots.

Runs the summary stages ported from the original summarizer notebook against
the fixed-split output tree produced by scripts/train_abide_classic.py:

    <output-root>/{iid,non_iid}_fixed_split/{non_priv,full_dp,feat_dp}/...

Stages (each stage is an importable module in medmap/abide_classic/):

    tables     per-split vertical + wide summary CSVs across all epsilons
    plots      privacy-utility curves (accuracy/AUROC/... vs epsilon)
    subgroups  per-subgroup accuracy/AUROC tables (non-IID tree)
    latex      LaTeX subgroup tables from the long CSVs (non-IID tree)

Examples:

    python -m scripts.summarize_abide --data-dir /data/abide/ABIDE_PCP_full \
        --split iid --stages tables plots
    python -m scripts.summarize_abide --data-dir /data/abide/ABIDE_PCP_full \
        --split non_iid --stages all
"""

from __future__ import annotations

import argparse
import importlib
import os

from scripts._bootstrap import setup_project_paths

setup_project_paths()

STAGE_MODULES = {
    ("iid", "tables"): ["medmap.abide_classic.summarize_fixed_iid"],
    ("iid", "plots"): ["medmap.abide_classic.plot_fixed_iid"],
    ("non_iid", "tables"): ["medmap.abide_classic.summarize_fixed_non_iid"],
    ("non_iid", "plots"): ["medmap.abide_classic.plot_fixed_non_iid"],
    ("non_iid", "subgroups"): ["medmap.abide_classic.subgroup_tables_non_iid"],
    ("non_iid", "latex"): ["medmap.abide_classic.latex_subgroup_tables_non_iid"],
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True, help="ABIDE PCP root.")
    ap.add_argument("--output-root", default=None, help="Default: <data-dir>/outputs")
    ap.add_argument("--split", required=True, choices=["iid", "non_iid"])
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["tables", "plots", "subgroups", "latex", "all"],
    )
    args = ap.parse_args()

    os.environ["ABIDE_DATA_DIR"] = args.data_dir
    if args.output_root:
        os.environ["ABIDE_OUTPUT_ROOT"] = args.output_root

    if "all" in args.stages:
        stages = ["tables", "plots"] + (["subgroups", "latex"] if args.split == "non_iid" else [])
    else:
        stages = args.stages

    for stage in stages:
        key = (args.split, stage)
        if key not in STAGE_MODULES:
            print(f"[summarize_abide] Skipping stage '{stage}' (not available for split '{args.split}').")
            continue
        for module_name in STAGE_MODULES[key]:
            print(f"[summarize_abide] Running {module_name} ...")
            # The stage modules execute on import (they are ported notebook
            # cells); force a fresh run each invocation.
            importlib.import_module(module_name)
        print(f"[summarize_abide] Stage '{stage}' done.")


if __name__ == "__main__":
    main()
