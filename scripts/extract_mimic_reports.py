#!/usr/bin/env python3
"""CLI wrapper: generate leakage-filtered MIMIC-CXR reports (s<id>_mmp.txt).

Usage (from the repository root):

    # Read-only validation first
    python -m scripts.extract_mimic_reports --shard p10 --dry-run --limit 100 \
        --reports-root /data/mimic/physionet.org/files/mimic-cxr/2.1.0/files

    # Full processing
    python -m scripts.extract_mimic_reports --shard p10 p11 p12 \
        --reports-root /data/mimic/physionet.org/files/mimic-cxr/2.1.0/files

See medmap/datasets/mimic_cxr/report_extraction.py and docs/mimic_cxr.md.
"""

from scripts._bootstrap import setup_project_paths

setup_project_paths()

from medmap.datasets.mimic_cxr.report_extraction import build_parser, run_many  # noqa: E402

if __name__ == "__main__":
    run_many(build_parser().parse_args())
