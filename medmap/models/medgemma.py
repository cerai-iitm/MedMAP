"""Backward-compatible shim.

The MIMIC-CXR classifier now lives in :mod:`medmap.models.gemma` and serves
both ``google/medgemma-4b-pt`` and ``google/gemma-3-4b-pt``. Existing imports
of ``MedGemmaForMultiLabelClassification`` / ``setup_medgemma_model`` keep
working through this module.
"""

from medmap.models.gemma import (  # noqa: F401
    BACKBONE_PRESETS,
    BackbonePreset,
    DEFAULT_BACKBONE,
    GemmaForMultiLabelClassification,
    MedGemmaForMultiLabelClassification,
    resolve_backbone,
    setup_gemma_model,
    setup_medgemma_model,
)

__all__ = [
    "BACKBONE_PRESETS",
    "BackbonePreset",
    "DEFAULT_BACKBONE",
    "GemmaForMultiLabelClassification",
    "MedGemmaForMultiLabelClassification",
    "resolve_backbone",
    "setup_gemma_model",
    "setup_medgemma_model",
]
