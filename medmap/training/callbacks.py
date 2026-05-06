import os
import gc
import json
import time
import subprocess
import torch
import wandb
import csv
from transformers import TrainerCallback
from transformers.integrations import WandbCallback

class WandbGpuCpuMemoryCallback(TrainerCallback):
    def __init__(self, log_every_step=True, reset_peak_on_log=True, only_main_process=True, throttle_seconds=0.0):
        self.log_every_step = log_every_step
        self.reset_peak_on_log = reset_peak_on_log
        self.only_main_process = only_main_process
        self.throttle_seconds = throttle_seconds
        self._last_log_t = 0.0
        try:
            import psutil
            self._psutil_ok = True
        except Exception:
            self._psutil_ok = False

    @staticmethod
    def _bytes_to_gb(b): return float(b) / (1024 ** 3)

    def _is_main_proc(self, args, state):
        if not self.only_main_process: return True
        is_zero = getattr(state, "is_world_process_zero", True)
        if is_zero is not None: return bool(is_zero)
        lr = getattr(args, "local_rank", -1)
        return (lr in (-1, 0))

    def _collect_torch_cuda(self):
        stats = {}
        if not torch.cuda.is_available(): return stats
        n = torch.cuda.device_count()
        ta = tr = tpa = tpr = 0.0
        for i in range(n):
            alloc    = self._bytes_to_gb(torch.cuda.memory_allocated(i))
            reserved = self._bytes_to_gb(torch.cuda.memory_reserved(i))
            peak_a   = self._bytes_to_gb(torch.cuda.max_memory_allocated(i))
            peak_r   = self._bytes_to_gb(torch.cuda.max_memory_reserved(i))
            stats[f"gpu/{i}/alloc_GB"] = alloc; stats[f"gpu/{i}/reserved_GB"] = reserved
            stats[f"gpu/{i}/peak_alloc_GB"] = peak_a; stats[f"gpu/{i}/peak_reserved_GB"] = peak_r
            ta += alloc; tr += reserved; tpa += peak_a; tpr += peak_r
        if n > 0:
            stats["gpu/alloc_GB_total"] = ta; stats["gpu/reserved_GB_total"] = tr
            stats["gpu/peak_alloc_GB_total"] = tpa; stats["gpu/peak_reserved_GB_total"] = tpr
        return stats

    def _maybe_reset_peaks(self):
        if torch.cuda.is_available() and self.reset_peak_on_log:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    def _define_step_metrics_once(self):
        if wandb.run and not getattr(self, "_defined_metrics", False):
            wandb.define_metric("trainer/global_step")
            for pattern in ["gpu/*", "cpu/*", "phase"]:
                wandb.define_metric(pattern, step_metric="trainer/global_step")
            wandb.define_metric("trainer/epoch")
            wandb.define_metric("val_epoch/*", step_metric="trainer/epoch")
            wandb.define_metric("test_epoch/*", step_metric="trainer/epoch")
            self._defined_metrics = True

    def _log_now(self, args, state, phase=None):
        if not self._is_main_proc(args, state): return
        t = time.time()
        if self.throttle_seconds > 0 and (t - self._last_log_t) < self.throttle_seconds: return
        self._define_step_metrics_once()

        payload = self._collect_torch_cuda()
        if phase: payload["phase"] = phase
        payload["trainer/global_step"] = int(getattr(state, "global_step", 0))

        if payload and wandb.run: wandb.log(payload)
        self._maybe_reset_peaks()
        self._last_log_t = t

    def on_log(self, args, state, control, **kwargs):
        self._log_now(args, state, phase="train")

class MemoryCleanupCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 50, sync_cuda: bool = True):
        self.every_n_steps = max(1, int(every_n_steps))
        self.sync_cuda = bool(sync_cuda)

    @staticmethod
    def _cleanup():
        try: gc.collect()
        except Exception: pass
        if torch.cuda.is_available():
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception: pass

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step or 0) % self.every_n_steps == 0:
            if self.sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
            self._cleanup()
        return control

class SplitEvalCallback(TrainerCallback):
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, mirror_val_step=False, csv_path="split_eval_log.csv"):
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.mirror_val_step = mirror_val_step
        self.csv_path = csv_path
        self.trainer = None
        self._last_epoch_done = -1

    def attach_trainer(self, trainer):
        self.trainer = trainer

    @staticmethod
    def _panelize(metrics, src_prefix, panel):
        out = {}
        prefix = f"{src_prefix}_"
        for k, v in metrics.items():
            if k.startswith(prefix): out[f"{panel}/{k[len(prefix):]}"] = v
        return out

    def _safe_eval(self, dataset, prefix):
        if dataset is None: return {}
        cb = next((c for c in self.trainer.callback_handler.callbacks if isinstance(c, WandbCallback)), None)
        if cb: self.trainer.callback_handler.callbacks.remove(cb)
        try: metrics = self.trainer.evaluate(eval_dataset=dataset, metric_key_prefix=prefix)
        finally:
            if cb: self.trainer.callback_handler.add_callback(cb)
        return metrics

    def _write_csv(self, logs):
        row = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()}
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None: return control
        cur_epoch = int(float(state.epoch or 0.0))
        if cur_epoch <= self._last_epoch_done: return control

        logs = {"trainer/epoch": float(state.epoch or 0.0), "trainer/global_step": int(getattr(state, "global_step", 0))}

        if self.train_ds is not None: logs.update(self._panelize(self._safe_eval(self.train_ds, "trn_epoch"), "trn_epoch", "train_epoch"))
        if self.val_ds is not None: logs.update(self._panelize(self._safe_eval(self.val_ds, "val_epoch"), "val_epoch", "val_epoch"))
        if self.test_ds is not None: logs.update(self._panelize(self._safe_eval(self.test_ds, "tst_epoch"), "tst_epoch", "test_epoch"))

        if wandb.run: wandb.log(logs)
        try: self._write_csv(logs)
        except Exception: pass

        self._last_epoch_done = cur_epoch
        return control

class SaveManagerCallback(TrainerCallback):
    def __init__(self, processor, class_names, base_model_name, use_lora, keep_last_n_step_aliases=10, keep_last_n_epoch_aliases=10, monitor=None, mode=None):
        self.processor = processor
        self.class_names = list(class_names)
        self.base_model_name = str(base_model_name)
        self.use_lora = bool(use_lora)
        self.keep_last_n_step_aliases = int(keep_last_n_step_aliases)
        self.keep_last_n_epoch_aliases = int(keep_last_n_epoch_aliases)
        self.monitor = monitor
        self.mode = mode
        self._best = None

    @staticmethod
    def _safe_symlink_or_pointer(src_rel: str, dst_path: str):
        try:
            if os.path.lexists(dst_path):
                if os.path.islink(dst_path): os.unlink(dst_path)
                else: os.remove(dst_path)
            os.symlink(src_rel, dst_path); return
        except Exception: pass
        if not dst_path.endswith(".txt"): dst_path += ".txt"
        with open(dst_path, "w") as f: f.write(src_rel + "\n")

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True): return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir): return control

        try: self.processor.save_pretrained(ckpt_dir)
        except Exception: pass
        
        try:
            with open(os.path.join(ckpt_dir, "classes.txt"), "w") as f:
                for name in self.class_names: f.write(f"{name}\n")
        except Exception: pass

        rel = os.path.relpath(ckpt_dir, start=args.output_dir)
        self._safe_symlink_or_pointer(rel, os.path.join(args.output_dir, "last"))
        return control