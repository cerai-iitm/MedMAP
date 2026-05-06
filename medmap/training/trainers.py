import torch
from transformers import Trainer
from dp_transformers.dp_utils import OpacusDPTrainer

class AccLoggingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tr_correct = 0
        self._tr_total = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, public=False, selective_dp=False, **kwargs):
        labels_for_acc = inputs.get("labels", None)
        
        if (getattr(self, "label_smoother", None) is not None or getattr(self, "compute_loss_func", None) is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Forward
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            outputs = model(**inputs, public=public, selective_dp=selective_dp, num_items_in_batch=num_items_in_batch, **kwargs)
        else:
            outputs = model(**inputs, public=public, selective_dp=selective_dp, **kwargs)

        # Compute loss
        if getattr(self, "compute_loss_func", None) is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif labels is not None and getattr(self, "label_smoother", None) is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # Standard: model should have put 'loss' in outputs
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss; keys: "
                    f"{','.join(outputs.keys())}. Inputs were {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if getattr(self.args, "average_tokens_across_devices", False) and (self.model_accepts_loss_kwargs or self.compute_loss_func) and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        #  Accumulate accuracy counters for this forward window (no grad) 
        try:
            if labels_for_acc is not None and hasattr(outputs, "logits"):
                with torch.no_grad():
                    preds = outputs.logits.argmax(dim=-1)
                    if labels_for_acc.shape != preds.shape:
                        labels_for_acc = labels_for_acc.view_as(preds)
                    self._tr_correct += int((preds == labels_for_acc).sum().item())
                    self._tr_total   += int(labels_for_acc.numel())
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        prev_last_logged = self._globalstep_last_logged
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)

        if self._globalstep_last_logged > prev_last_logged:
            corr_t = torch.tensor(self._tr_correct, dtype=torch.long, device=self.args.device)
            tot_t  = torch.tensor(self._tr_total,   dtype=torch.long, device=self.args.device)
            corr = self.accelerator.gather(corr_t).sum().item()
            tot  = self.accelerator.gather(tot_t).sum().item()

            if tot > 0:
                acc = round(corr / float(tot), 4)
                self.log({"accuracy": acc}, start_time)

            self._tr_correct = 0
            self._tr_total = 0

class StandardTrainer(AccLoggingMixin, Trainer):
    """Standard HuggingFace Trainer with accuracy logging."""
    pass

class DPTrainer(AccLoggingMixin, OpacusDPTrainer):
    """Opacus DP Trainer with accuracy logging."""
    pass

def get_trainer(use_dp: bool, **kwargs):
    if use_dp:
        return DPTrainer(**kwargs)
    return StandardTrainer(**kwargs)