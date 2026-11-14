"""Backbone presets for MIMIC-MedMAP (dependency-free; safe to import for --help).

The paper reports MIMIC-MedMAP with two Gemma 3 family backbones:

- ``gemma``    -- ``google/gemma-3-4b-pt`` (general-purpose VLM; main results)
- ``medgemma`` -- ``google/medgemma-4b-pt`` (medically adapted reference model)

:data:`BACKBONE_PRESETS` records the per-backbone configuration used for the
reported runs (model id, delta convention, sequence length, and the selected
private hyperparameters) so launchers and ``scripts/train_mimic.py`` can fill
in defaults by backbone name while still accepting explicit overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BackbonePreset:
    """Reported configuration of one MIMIC-MedMAP backbone.

    ``delta_power`` follows the convention ``target_delta = n_train ** delta_power``.
    The learning rates and the public clipping norm are the values selected
    for the reported runs (see the paper, Appendix E) and are used as launcher
    defaults; they are not enforced by the training script.
    """

    key: str
    model_name: str
    display_name: str
    delta_power: float
    max_length: int
    non_private_lr: float
    dp_lr: float
    feature_dp_lr: float
    clip_norm: float
    public_clip_norm: float
    notes: str = ""


BACKBONE_PRESETS: Dict[str, BackbonePreset] = {
    "medgemma": BackbonePreset(
        key="medgemma",
        model_name="google/medgemma-4b-pt",
        display_name="MedGemma 4B",
        # The MedGemma runs used the dp-transformers default delta = 1/n.
        delta_power=-1.0,
        max_length=482,
        non_private_lr=5e-5,
        dp_lr=5e-3,
        feature_dp_lr=2e-3,
        clip_norm=1.0,
        public_clip_norm=50.0,
        notes="Medically adapted Gemma 3 reference model (pretraining data include MIMIC-CXR).",
    ),
    "gemma": BackbonePreset(
        key="gemma",
        model_name="google/gemma-3-4b-pt",
        display_name="Gemma 3 4B",
        # The Gemma 3 runs used delta = n ** -1.1 (same convention as ABIDE/PAD).
        delta_power=-1.1,
        max_length=482,
        non_private_lr=5e-5,
        dp_lr=5e-3,
        feature_dp_lr=2e-3,
        clip_norm=1.0,
        public_clip_norm=69.0,
        notes="General-purpose Gemma 3 base checkpoint; main MIMIC-MedMAP results.",
    ),
}

DEFAULT_BACKBONE = "medgemma"


def resolve_backbone(key: str) -> BackbonePreset:
    """Return the preset for ``key`` (case-insensitive) or raise ``ValueError``."""
    normalized = str(key).strip().lower()
    try:
        return BACKBONE_PRESETS[normalized]
    except KeyError:
        choices = ", ".join(sorted(BACKBONE_PRESETS))
        raise ValueError(f"Unknown MIMIC backbone {key!r}. Choose one of: {choices}.") from None
