#!/usr/bin/env python3
"""CLI wrapper: select one representative MIMIC-CXR image per study.

Usage (from the repository root):

    python -m scripts.select_mimic_images p10 p11 p12 \
        --dataset-root /data/mimic/physionet.org/files/mimic-cxr-jpg/2.1.0 \
        --output-root  /data/mimic/best_img_index

See medmap/datasets/mimic_cxr/image_selection.py and docs/mimic_cxr.md.
"""

from scripts._bootstrap import setup_project_paths

setup_project_paths()

from medmap.datasets.mimic_cxr.image_selection import main  # noqa: E402

if __name__ == "__main__":
    main()
