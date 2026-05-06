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
    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.classifier = nn.Linear(self.config.text_config.hidden_size, num_labels)
        self.image_token_id = getattr(self.config, "image_token_index", None)
        self.pad_token_id = getattr(self.config, "pad_token_id", 0)

    def gradient_checkpointing_enable(self, **kwargs):
        for obj in [self.backbone, getattr(self.backbone, "model", None)]:
            if hasattr(obj, "gradient_checkpointing_enable"):
                try: obj.gradient_checkpointing_enable(**kwargs)
                except TypeError: obj.gradient_checkpointing_enable()

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, 
                pixel_values: Optional[torch.FloatTensor] = None, labels: Optional[torch.FloatTensor] = None,
                public: Optional[bool] = False, selective_dp: Optional[bool] = False, 
                loss_mode: str = "standard", **kwargs) -> Dict[str, torch.Tensor]:
        
        outputs = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, use_cache=False, return_dict=True)
        last_hidden = outputs.last_hidden_state
        
        valid = attention_mask.bool() & (input_ids != self.pad_token_id)
        img_mask = valid & (input_ids == self.image_token_id) if self.image_token_id is not None else torch.zeros_like(valid)
        txt_mask = valid & (~img_mask)

        img_vec, txt_vec = self._masked_mean(last_hidden, img_mask), self._masked_mean(last_hidden, txt_mask)
        has_img, has_txt = img_mask.any(dim=1), txt_mask.any(dim=1)

        pooled = torch.zeros_like(txt_vec)
        both = has_img & has_txt
        pooled[both] = 0.5 * (img_vec[both] + txt_vec[both])
        pooled[has_img & ~has_txt] = img_vec[has_img & ~has_txt]
        pooled[has_txt & ~has_img] = txt_vec[has_txt & ~has_img]

        logits_text = self.classifier(pooled)
        logits_img = self.classifier(img_vec)
        
        logits = logits_text if loss_mode == "standard" else (logits_img if public else logits_text)
        out = {"logits": logits}
        
        if labels is not None:
            mask = torch.isfinite(labels)
            safe_labels = torch.nan_to_num(labels, nan=0.0)
            
            if loss_mode == "standard":
                per_elem = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
                out["loss"] = (per_elem * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            else:
                text_loss = torch.tensor(0.0, device=logits.device)
                if has_txt.any():
                    per_elem_text = F.binary_cross_entropy_with_logits(logits_text[has_txt], safe_labels[has_txt], reduction="none")
                    text_loss = (per_elem_text * mask[has_txt].float()).sum() / mask[has_txt].float().sum().clamp_min(1.0)
                    
                image_loss = torch.tensor(0.0, device=logits.device)
                if has_img.any():
                    per_elem_img = F.binary_cross_entropy_with_logits(logits_img[has_img], safe_labels[has_img], reduction="none")
                    image_loss = (per_elem_img * mask[has_img].float()).sum() / mask[has_img].float().sum().clamp_min(1.0)

                w_text, w_image = (0.0, 1.0) if public else ((1.0, -1.0) if selective_dp else (1.0, 0.0))
                out["loss"] = w_text * text_loss + w_image * image_loss

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
    return model, backbone