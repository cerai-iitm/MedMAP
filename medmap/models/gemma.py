"""Gemma-3 family backbones (MedGemma 4B and Gemma 3 4B) for MIMIC-CXR.

Both backbones share the Gemma 3 architecture, processor, and prompt layout
(``<start_of_image>`` block followed by the report text), so one classifier
wrapper serves both. The paper reports MIMIC-MedMAP with two backbones:

- ``gemma``    -- ``google/gemma-3-4b-pt`` (general-purpose VLM; main results)
- ``medgemma`` -- ``google/medgemma-4b-pt`` (medically adapted reference model)

The per-backbone presets (model id, delta convention, sequence length, selected
private hyperparameters) live in :mod:`medmap.models.mimic_backbones`, which
has no torch dependency, and are re-exported here.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

from medmap.models.mimic_backbones import (  # noqa: F401  (re-exported)
    BACKBONE_PRESETS,
    DEFAULT_BACKBONE,
    BackbonePreset,
    resolve_backbone,
)

try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except Exception:  # pragma: no cover - peft is an optional dependency
    PEFT_AVAILABLE = False


class GemmaForMultiLabelClassification(nn.Module):
    """
    Multi-label classifier on top of a Gemma 3 family backbone (MedGemma or Gemma 3).

    The classifier operates on mean-pooled final hidden states. Image and text
    tokens are pooled separately; the fused representation averages the two
    when both are present. Because image tokens never attend to text tokens,
    the image-only pooled vector of a fused forward equals what an image-only
    forward would produce, so one backward pass on ``text_loss - image_loss``
    yields the protected-modality gradient residual exactly.

    Modes
    -----
    loss_mode="standard":
        Non-private training:
        - logits = classifier(fused pooled representation)
        - loss = masked BCE over all finite labels

    loss_mode="decomposed":
        Private / Feature-DP training:
        - logits_text = classifier(fused pooled representation)
        - logits_img  = classifier(image-only representation)
        - public=True:
              loss = image_loss
              logits = logits_img
        - public=False, feature_dp=True:
              loss = text_loss - image_loss
              logits = logits_text
        - public=False, feature_dp=False:
              loss = text_loss
              logits = logits_text
    """

    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

        hidden = self.config.text_config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        self.image_token_id = getattr(self.config, "image_token_index", None)
        self.pad_token_id = getattr(self.config, "pad_token_id", 0)

    # ---------------------------------------------------------
    # Gradient checkpointing passthrough
    # ---------------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        for obj in [self.backbone, getattr(self.backbone, "model", None)]:
            if obj is None:
                continue

            if hasattr(obj, "gradient_checkpointing_enable"):
                try:
                    obj.gradient_checkpointing_enable(**kwargs)
                except TypeError:
                    obj.gradient_checkpointing_enable()
                return

    def gradient_checkpointing_disable(self):
        for obj in [self.backbone, getattr(self.backbone, "model", None)]:
            if obj is None:
                continue

            if hasattr(obj, "gradient_checkpointing_disable"):
                obj.gradient_checkpointing_disable()
                return

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        public: Optional[bool] = False,
        # The DP trainer passes feature_dp explicitly; the default keeps the
        # decomposed mode equivalent to Feature-DP when called directly.
        feature_dp: Optional[bool] = True,
        # "standard"   -> non-private loss
        # "decomposed" -> DP / Feature-DP loss
        loss_mode: str = "standard",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if loss_mode not in {"standard", "decomposed"}:
            raise ValueError(
                f"Unknown loss_mode={loss_mode}. "
                "Use loss_mode='standard' or loss_mode='decomposed'."
            )

        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            use_cache=False,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state

        # ---------------------------------------------------------
        # Build valid / image / text masks
        # ---------------------------------------------------------
        valid = attention_mask.bool() & (input_ids != self.pad_token_id)

        if self.image_token_id is None:
            img_mask = torch.zeros_like(valid)
        else:
            img_mask = valid & (input_ids == self.image_token_id)

        txt_mask = valid & (~img_mask)

        # ---------------------------------------------------------
        # Pool image and text representations separately
        # ---------------------------------------------------------
        img_vec = self._masked_mean(last_hidden, img_mask)
        txt_vec = self._masked_mean(last_hidden, txt_mask)

        has_img = img_mask.any(dim=1)
        has_txt = txt_mask.any(dim=1)

        # ---------------------------------------------------------
        # Fused representation
        # - both image and text: average
        # - image only: image vector
        # - text only: text vector
        # - neither: remains zero
        # ---------------------------------------------------------
        pooled = torch.zeros_like(txt_vec)

        both = has_img & has_txt
        only_img = has_img & (~has_txt)
        only_txt = has_txt & (~has_img)

        pooled[both] = 0.5 * (img_vec[both] + txt_vec[both])
        pooled[only_img] = img_vec[only_img]
        pooled[only_txt] = txt_vec[only_txt]

        # ---------------------------------------------------------
        # Logits
        # ---------------------------------------------------------
        logits_text = self.classifier(pooled)
        logits_img = self.classifier(img_vec)

        if loss_mode == "standard":
            logits = logits_text
        else:
            logits = logits_img if public else logits_text

        out = {"logits": logits}

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        loss = None

        if labels is not None:
            mask = torch.isfinite(labels)
            safe_labels = torch.nan_to_num(labels, nan=0.0)

            if loss_mode == "standard":
                per_elem = F.binary_cross_entropy_with_logits(
                    logits,
                    safe_labels,
                    reduction="none",
                )
                loss = (per_elem * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

            else:
                # TEXT LOSS (fused representation)
                text_loss = torch.tensor(0.0, device=logits.device)
                if has_txt.any():
                    per_elem_text = F.binary_cross_entropy_with_logits(
                        logits_text[has_txt],
                        safe_labels[has_txt],
                        reduction="none",
                    )
                    mask_text = mask[has_txt]
                    text_loss = (per_elem_text * mask_text.float()).sum() / mask_text.float().sum().clamp_min(1.0)

                # IMAGE LOSS (image-only representation)
                image_loss = torch.tensor(0.0, device=logits.device)
                if has_img.any():
                    per_elem_img = F.binary_cross_entropy_with_logits(
                        logits_img[has_img],
                        safe_labels[has_img],
                        reduction="none",
                    )
                    mask_img = mask[has_img]
                    image_loss = (per_elem_img * mask_img.float()).sum() / mask_img.float().sum().clamp_min(1.0)

                if public:
                    w_text, w_image = 0.0, 1.0
                elif feature_dp:
                    w_text, w_image = 1.0, -1.0
                else:
                    w_text, w_image = 1.0, 0.0

                loss = w_text * text_loss + w_image * image_loss

            out["loss"] = loss

        elif loss_mode == "decomposed":
            # The DP trainer expects a "loss" key even when labels are absent.
            out["loss"] = loss

        return out


# Backward-compatible alias (the class was previously MedGemma-specific).
MedGemmaForMultiLabelClassification = GemmaForMultiLabelClassification


def setup_gemma_model(
    model_name: str,
    num_labels: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a Gemma 3 family backbone (MedGemma or Gemma 3) and attach the head.

    ``model_name`` is any Hugging Face id of a Gemma 3 architecture checkpoint,
    typically ``BACKBONE_PRESETS[key].model_name``.
    """
    backbone = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch_dtype)
    backbone.config.use_cache = False
    model = GemmaForMultiLabelClassification(backbone, num_labels=num_labels)

    if use_lora and PEFT_AVAILABLE:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2",
        ]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    elif use_lora:
        raise RuntimeError("--use_lora requested but the 'peft' package is not installed.")
    return model


# Backward-compatible alias.
setup_medgemma_model = setup_gemma_model
