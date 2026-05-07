# medmap/models/linear_abide.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LinearABIDEConfig(PretrainedConfig):
    """
    HuggingFace config for a simple linear ABIDE classifier.

    Public modality:
        fMRI connectivity features.

    Private / sensitive modality:
        selected phenotype features.

    The model is intentionally linear:
        logits = W [x_fmri, x_sensitive] + b
    """

    model_type = "linear_abide"

    def __init__(
        self,
        fmri_dim: int = 0,
        sensitive_dim: int = 0,
        num_labels: int = 2,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fmri_dim = int(fmri_dim)
        self.sensitive_dim = int(sensitive_dim)
        self.num_labels = int(num_labels)
        self.initializer_range = float(initializer_range)


class LinearABIDEForSequenceClassification(PreTrainedModel):
    """
    Linear HuggingFace-compatible ABIDE classifier.

    loss_mode="standard":
        Non-private training/evaluation.
        Uses the active modalities controlled by has_fmri_flags and
        has_sensitive_flags.

    loss_mode="decomposed":
        DP / Feature-DP training.
        public=True:
            CE(model(fMRI, zero-sensitive), y)

        selective_dp=True:
            CE(model(fMRI, sensitive), y)
            -
            CE(model(fMRI, zero-sensitive), y)

        selective_dp=False:
            CE(model(fMRI, sensitive), y)

    This forward contract is designed to be compatible with your
    AccLoggingMixin + OpacusDPTrainer wrapper.
    """

    config_class = LinearABIDEConfig

    def __init__(self, config: LinearABIDEConfig):
        super().__init__(config)

        input_dim = config.fmri_dim + config.sensitive_dim
        if input_dim <= 0:
            raise ValueError(
                "LinearABIDE requires fmri_dim + sensitive_dim > 0. "
                f"Got fmri_dim={config.fmri_dim}, sensitive_dim={config.sensitive_dim}."
            )

        self.classifier = nn.Linear(input_dim, config.num_labels)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _ensure_tensor(
        self,
        x: Optional[torch.Tensor],
        batch_size: int,
        dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        if x is None:
            return torch.zeros(batch_size, dim, dtype=torch.float32, device=device)
        return x.to(device=device, dtype=torch.float32)

    def _apply_flag(
        self,
        x: torch.Tensor,
        flag: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if flag is None:
            return x
        flag = flag.to(device=x.device).bool().float().view(-1, 1)
        return x * flag

    def _concat(self, fmri: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        return torch.cat([fmri, sensitive], dim=-1)

    def forward(
        self,
        fmri_features: Optional[torch.FloatTensor] = None,
        sensitive_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        # DPTrainer / Feature-DP controls.
        public: Optional[bool] = False,
        selective_dp: Optional[bool] = False,
        loss_mode: str = "standard",

        # Modality presence flags from the collator.
        has_fmri_flags: Optional[torch.BoolTensor] = None,
        has_sensitive_flags: Optional[torch.BoolTensor] = None,

        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if loss_mode not in {"standard", "decomposed"}:
            raise ValueError(
                f"Unknown loss_mode={loss_mode!r}. "
                "Use loss_mode='standard' or loss_mode='decomposed'."
            )

        device = self.classifier.weight.device

        if fmri_features is not None:
            batch_size = fmri_features.shape[0]
        elif sensitive_features is not None:
            batch_size = sensitive_features.shape[0]
        elif labels is not None:
            batch_size = labels.shape[0]
        else:
            raise ValueError("At least one of fmri_features, sensitive_features, or labels is required.")

        fmri = self._ensure_tensor(
            fmri_features,
            batch_size=batch_size,
            dim=self.config.fmri_dim,
            device=device,
        )
        sensitive = self._ensure_tensor(
            sensitive_features,
            batch_size=batch_size,
            dim=self.config.sensitive_dim,
            device=device,
        )

        # Apply modality-selection flags.
        # For example:
        #   INCLUDE_FMRI=False      -> fMRI block is zeroed.
        #   INCLUDE_SENSITIVE=False -> sensitive block is zeroed.
        fmri_active = self._apply_flag(fmri, has_fmri_flags)
        sensitive_active = self._apply_flag(sensitive, has_sensitive_flags)

        zeros_sensitive = torch.zeros_like(sensitive_active)

        # Public-only view: x_pub = (fMRI, zero-sensitive).
        x_public = self._concat(fmri_active, zeros_sensitive)

        # Full view: x = (fMRI, sensitive).
        x_full = self._concat(fmri_active, sensitive_active)

        logits_public = self.classifier(x_public)
        logits_full = self.classifier(x_full)

        if loss_mode == "standard":
            logits = logits_full
        else:
            logits = logits_public if public else logits_full

        loss = None
        if labels is not None:
            labels = labels.to(device=device).long().view(-1)

            if loss_mode == "standard":
                loss = F.cross_entropy(logits_full.float(), labels, reduction="mean")
            else:
                public_loss = F.cross_entropy(logits_public.float(), labels, reduction="mean")
                full_loss = F.cross_entropy(logits_full.float(), labels, reduction="mean")

                if public:
                    loss = public_loss
                else:
                    if selective_dp:
                        loss = full_loss - public_loss
                    else:
                        loss = full_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def setup_linear_abide_model(
    fmri_dim: int,
    sensitive_dim: int,
    num_labels: int = 2,
) -> LinearABIDEForSequenceClassification:
    config = LinearABIDEConfig(
        fmri_dim=fmri_dim,
        sensitive_dim=sensitive_dim,
        num_labels=num_labels,
    )
    model = LinearABIDEForSequenceClassification(config)
    return model