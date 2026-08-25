"""MIMIC-CXR dataset support: image selection, leakage-filtered report
extraction, and the cached HF dataset builder.

Preprocessing pipeline (see docs/mimic_cxr.md for the full walkthrough):

1. ``image_selection``    — pick one representative JPG per study
                            (PA > AP > LATERAL > other; metadata only).
2. ``report_extraction``  — write leakage-filtered ``s<study_id>_mmp.txt``
                            reports (Findings/Impression removed, CheXpert/
                            NegBio mentions masked) + per-shard audit CSV.
3. ``builder``            — merge best-image index + filtered reports +
                            CheXpert labels, split with the official
                            PhysioNet split CSV, cache to disk.
"""

from medmap.datasets.mimic_cxr.report_extraction import (  # noqa: F401
    AUDIT_CSV_NAME,
    REPORT_SUFFIX,
)

# NOTE: the dataset builder (medmap.datasets.mimic_cxr.builder) requires the
# heavyweight pandas/datasets stack and is intentionally NOT imported here, so
# that the preprocessing CLIs (image selection, report extraction) work in a
# minimal environment. Import it explicitly:
#   from medmap.datasets.mimic_cxr.builder import load_or_build_cached_mimic_splits
