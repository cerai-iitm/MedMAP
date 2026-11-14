import torch
import torch.nn as nn
from transformers import Trainer
from dp_transformers.dp_utils import OpacusDPTrainer


class AccLoggingMixin:
    def __init__(self, *args, loss_mode: str = "standard", **kwargs):
        super().__init__(*args, **kwargs)

        if loss_mode not in {"standard", "decomposed"}:
            raise ValueError(
                f"Invalid loss_mode={loss_mode}. "
                "Use 'standard' or 'decomposed'."
            )

        self.loss_mode = loss_mode
        self._tr_correct = 0
        self._tr_total = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        public=False,
        feature_dp=False,
        **kwargs,
    ):
        labels_for_acc = inputs.get("labels", None)

        if (
            getattr(self, "label_smoother", None) is not None
            or getattr(self, "compute_loss_func", None) is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        forward_kwargs = dict(
            public=public,
            feature_dp=feature_dp,
            loss_mode=self.loss_mode,
            **kwargs,
        )

        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            forward_kwargs["num_items_in_batch"] = num_items_in_batch

        outputs = model(**inputs, **forward_kwargs)

        if getattr(self, "compute_loss_func", None) is not None:
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        elif labels is not None and getattr(self, "label_smoother", None) is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss; keys: "
                    f"{','.join(outputs.keys())}. Inputs were {','.join(inputs.keys())}."
                )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            getattr(self.args, "average_tokens_across_devices", False)
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        try:
            logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
            # Skip the public (image-only) pass under Feature-DP so the logged
            # train accuracy reflects the full-modality forward only.
            if labels_for_acc is not None and logits is not None and not public:
                with torch.no_grad():
                    labels_for_acc = labels_for_acc.to(logits.device)
                    if labels_for_acc.ndim == 2 and labels_for_acc.is_floating_point():
                        # Multi-label (MIMIC): NaN-safe micro accuracy over
                        # known label positions, threshold 0.5 on the sigmoid.
                        preds = torch.sigmoid(logits) >= 0.5
                        known = torch.isfinite(labels_for_acc)
                        targets = labels_for_acc >= 0.5
                        self._tr_correct += int(((preds == targets) & known).sum().item())
                        self._tr_total += int(known.sum().item())
                    else:
                        # Single-label (ABIDE / ImageNet): argmax accuracy.
                        preds = logits.argmax(dim=-1)
                        if labels_for_acc.shape != preds.shape:
                            labels_for_acc = labels_for_acc.view_as(preds)
                        self._tr_correct += int((preds == labels_for_acc).sum().item())
                        self._tr_total += int(labels_for_acc.numel())
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        prev_last_logged = self._globalstep_last_logged

        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

        if self._globalstep_last_logged > prev_last_logged:
            corr_t = torch.tensor(
                self._tr_correct,
                dtype=torch.long,
                device=self.args.device,
            )
            tot_t = torch.tensor(
                self._tr_total,
                dtype=torch.long,
                device=self.args.device,
            )

            corr = self.accelerator.gather(corr_t).sum().item()
            tot = self.accelerator.gather(tot_t).sum().item()

            if tot > 0:
                acc = round(corr / float(tot), 4)
                self.log({"accuracy": acc}, start_time)

            self._tr_correct = 0
            self._tr_total = 0
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Safely unties weights before saving to avoid HF Serialization errors."""
        output_dir = output_dir or self.args.output_dir
        
        unwrapped = self.model
        for attr in ("module", "base_model", "model"):
            unwrapped = getattr(unwrapped, attr, unwrapped)
        backbone = getattr(unwrapped, "backbone", unwrapped)
        
        try:
            lm_head = backbone.lm_head
            embed_w = backbone.model.language_model.embed_tokens.weight
            tied = lm_head.weight.data_ptr() == embed_w.data_ptr()
        except AttributeError:
            tied = False
        
        if tied:
            lm_head.weight = nn.Parameter(lm_head.weight.clone())
        
        try:
            super().save_model(output_dir, _internal_call)
        finally:
            if tied:
                lm_head.weight = backbone.model.language_model.embed_tokens.weight


class StandardTrainer(AccLoggingMixin, Trainer):
    """Standard HuggingFace Trainer with accuracy logging."""
    pass


class DPTrainer(AccLoggingMixin, OpacusDPTrainer):
    """Opacus DP Trainer with accuracy logging."""
    pass


def get_trainer(use_dp: bool, **kwargs):
    if use_dp:
        kwargs.setdefault("loss_mode", "decomposed")
        return DPTrainer(**kwargs)

    kwargs.setdefault("loss_mode", "standard")
    return StandardTrainer(**kwargs)