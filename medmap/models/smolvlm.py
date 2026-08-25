import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.smolvlm import SmolVLMConfig, SmolVLMModel, SmolVLMPreTrainedModel

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    TaskType = None

class SmolVLMForImageTextClassification(SmolVLMPreTrainedModel):
    """
    Unified SmolVLM classifier.

    Modes:
    ------
    loss_mode="standard":
        Behaves like the normal heterogeneous classifier:
        - text-only: pool text tokens
        - image-only: pool image tokens
        - image+text: average text_pool and img_pool
        - loss = CE(logits, labels) over all rows

    loss_mode="decomposed":
        Behaves like the Feature-DP/public-private classifier:
        - computes logits_text from fused embedding
        - computes logits_image from image-only embedding
        - public=True        -> image_loss
        - feature_dp=True  -> text_loss - image_loss
        - otherwise          -> text_loss
    """

    def __init__(self, config: SmolVLMConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = SmolVLMModel(config)

        hidden = config.text_config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        self.post_init()

    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        # DP / public-private controls
        public: Optional[bool] = False,
        feature_dp: Optional[bool] = False,

        # IMPORTANT:
        # "standard"   -> attached normal fused classifier behavior
        # "decomposed" -> Feature-DP/public-private behavior
        loss_mode: str = "standard",

        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,

        has_text_flags: Optional[torch.BoolTensor] = None,
        has_image_flags: Optional[torch.BoolTensor] = None,
        text_valid_mask: Optional[torch.BoolTensor] = None,

        **kwargs,
    ) -> SequenceClassifierOutput:

        if loss_mode not in {"standard", "decomposed"}:
            raise ValueError(
                f"Unknown loss_mode={loss_mode}. "
                "Use loss_mode='standard' or loss_mode='decomposed'."
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        last_hidden = outputs.last_hidden_state
        B, T, H = last_hidden.shape
        device = last_hidden.device

        # ---------------------------------------------------------
        # Case 1: no input_ids
        # This matches the attached normal classifier fallback.
        # ---------------------------------------------------------
        if input_ids is None:
            pooled = last_hidden.mean(dim=1)

            logits = self.classifier(pooled)

            loss = None
            if labels is not None:
                labels = labels.to(device)
                loss = F.cross_entropy(logits.float(), labels.view(-1))

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # ---------------------------------------------------------
        # Case 2: normal token-aware pooling
        # ---------------------------------------------------------
        if getattr(self.config, "image_token_id", None) is None:
            raise ValueError(
                "config.image_token_id is None. "
                "Set it from the tokenizer id of '<image>'."
            )

        # Use boolean masks. This keeps the logic close to your DP version.
        if attention_mask is not None:
            am = attention_mask.to(device=device).bool()
        else:
            am = torch.ones((B, T), device=device, dtype=torch.bool)

        input_ids = input_ids.to(device)

        is_image = (input_ids == self.config.image_token_id) & am
        is_text = (input_ids != self.config.image_token_id) & am

        # Respect collator image flags.
        # This prevents dummy image tokens from contributing to image pooling.
        if has_image_flags is not None:
            has_image_flags = has_image_flags.to(device).bool()
            is_image = is_image & has_image_flags.unsqueeze(1)

        # Respect text_valid_mask.
        # This can remove wrapper/template tokens from text pooling.
        if text_valid_mask is not None:
            is_text = is_text & text_valid_mask.to(device).bool()

        # ---------------------------------------------------------
        # Text pooling
        # ---------------------------------------------------------
        text_sum = (last_hidden * is_text.unsqueeze(-1)).sum(dim=1)
        text_den = is_text.sum(dim=1).clamp(min=1).unsqueeze(-1)
        text_pool = text_sum / text_den

        # ---------------------------------------------------------
        # Image pooling
        # ---------------------------------------------------------
        img_sum = (last_hidden * is_image.unsqueeze(-1)).sum(dim=1)
        img_den = is_image.sum(dim=1).clamp(min=1).unsqueeze(-1)
        img_pool = img_sum / img_den

        has_image = is_image.sum(dim=1) > 0

        if has_text_flags is not None:
            has_text = has_text_flags.to(device).bool()
        else:
            has_text = is_text.sum(dim=1) > 0

        # ---------------------------------------------------------
        # Fused pooled embedding
        # Same as attached normal classifier:
        # - both image and text: average
        # - text only: text_pool
        # - image only: img_pool
        # ---------------------------------------------------------
        pooled = torch.where(
            has_text.unsqueeze(-1) & has_image.unsqueeze(-1),
            0.5 * (text_pool + img_pool),
            torch.where(
                has_text.unsqueeze(-1),
                text_pool,
                img_pool,
            ),
        )

        # Fallback if neither modality is present.
        # This exists in the attached normal classifier.
        empty_rows = (~has_text) & (~has_image)
        if empty_rows.any():
            pooled[empty_rows] = last_hidden[empty_rows].mean(dim=1)

        # ---------------------------------------------------------
        # Logits
        # ---------------------------------------------------------
        logits_text = self.classifier(pooled)
        logits_image = self.classifier(img_pool)

        if loss_mode == "standard":
            # Exactly the normal fused-classifier behavior.
            logits = logits_text
        else:
            # Feature-DP/public-private behavior.
            logits = logits_image if public else logits_text

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        loss = None
        if labels is not None:
            labels = labels.to(device)

            if loss_mode == "standard":
                # Matches attached normal classifier:
                # CE over all rows.
                loss = F.cross_entropy(logits.float(), labels.view(-1))

            else:
                # Matches your Feature-DP/public-private version:
                # modality-specific losses.
                if has_image.any():
                    image_loss = F.cross_entropy(
                        logits_image[has_image].float(),
                        labels[has_image],
                        reduction="mean",
                    )
                else:
                    image_loss = torch.tensor(0.0, device=device)

                if has_text.any():
                    text_loss = F.cross_entropy(
                        logits_text[has_text].float(),
                        labels[has_text],
                        reduction="mean",
                    )
                else:
                    text_loss = torch.tensor(0.0, device=device)

                if public:
                    w_text, w_image = 0.0, 1.0
                else:
                    if feature_dp:
                        w_text, w_image = 1.0, -1.0
                    else:
                        w_text, w_image = 1.0, 0.0

                loss = w_text * text_loss + w_image * image_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def setup_model(model_name: str, config: SmolVLMConfig, num_classes: int, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    model = SmolVLMForImageTextClassification.from_pretrained(
        model_name,
        config=config,
        num_labels=num_classes,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.use_cache = False

    if not use_lora:
        return model
    if not PEFT_AVAILABLE:
        print("[WARN] PEFT / LoRA not available; continuing without LoRA.")
        return model

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none")
    try:
        if TaskType is not None:
            lora_cfg.task_type = TaskType.SEQ_CLS
    except Exception:
        pass
    
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model