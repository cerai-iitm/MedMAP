import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import AutoModelForImageTextToText

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

class MedGemmaForMultiLabelClassification(nn.Module):
    """
    Unified MedGemma multi-label classifier.

    Modes
    -----
    loss_mode="standard":
        Matches Code 1, non-private training:
        - compute fused image/text pooled representation
        - logits = classifier(pooled)
        - loss = masked BCE over all finite labels

    loss_mode="decomposed":
        Matches Code 2, private / selective-DP training:
        - logits_text = classifier(fused pooled representation)
        - logits_img  = classifier(image-only representation)
        - public=True:
              loss = image_loss
              logits = logits_img
        - public=False, selective_dp=True:
              loss = text_loss - image_loss
              logits = logits_text
        - public=False, selective_dp=False:
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

        # IMPORTANT:
        # Code 2 had selective_dp=True by default.
        # So we keep True here for exact Code 2 behavior when loss_mode="decomposed".
        selective_dp: Optional[bool] = True,

        # "standard"   -> Code 1 behavior
        # "decomposed" -> Code 2 behavior
        loss_mode: str = "standard",

        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        print("HERE LOSS MODE IS:", loss_mode)
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
        # Same as Code 1 and Code 2:
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
            # Code 1 behavior
            logits = logits_text
        else:
            # Code 2 behavior
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
                # -------------------------------------------------
                # Code 1 loss:
                # masked BCE over all finite labels
                # -------------------------------------------------
                per_elem = F.binary_cross_entropy_with_logits(
                    logits,
                    safe_labels,
                    reduction="none",
                )

                loss = (
                    per_elem * mask.float()
                ).sum() / mask.float().sum().clamp_min(1.0)

            else:
                # -------------------------------------------------
                # Code 2 decomposed loss
                # -------------------------------------------------

                # TEXT LOSS
                text_loss = torch.tensor(0.0, device=logits.device)

                if has_txt.any():
                    per_elem_text = F.binary_cross_entropy_with_logits(
                        logits_text[has_txt],
                        safe_labels[has_txt],
                        reduction="none",
                    )

                    mask_text = mask[has_txt]

                    text_loss = (
                        per_elem_text * mask_text.float()
                    ).sum() / mask_text.float().sum().clamp_min(1.0)

                # IMAGE LOSS
                image_loss = torch.tensor(0.0, device=logits.device)

                if has_img.any():
                    per_elem_img = F.binary_cross_entropy_with_logits(
                        logits_img[has_img],
                        safe_labels[has_img],
                        reduction="none",
                    )

                    mask_img = mask[has_img]

                    image_loss = (
                        per_elem_img * mask_img.float()
                    ).sum() / mask_img.float().sum().clamp_min(1.0)

                # Same weighting logic as Code 2
                if public:
                    w_text, w_image = 0.0, 1.0
                else:
                    if selective_dp:
                        w_text, w_image = 1.0, -1.0
                    else:
                        w_text, w_image = 1.0, 0.0

                loss = w_text * text_loss + w_image * image_loss

            out["loss"] = loss

        elif loss_mode == "decomposed":
            # This matches Code 2 more closely, where out["loss"] exists
            # even when labels is None.
            out["loss"] = loss

        return out

def setup_medgemma_model(model_name: str, num_labels: int, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    backbone = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    backbone.config.use_cache = False
    model = MedGemmaForMultiLabelClassification(backbone, num_labels=num_labels)
    
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
    return model