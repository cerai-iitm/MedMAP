import os
import gc
import json
import time
import subprocess
import torch
import wandb
import csv
from typing import Optional, List
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from ..utils.core import _safe_makedirs, _symlink_dir, export_predictions_csv

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
    
# MedGemma Callbacks
class WandbInitAndPanelsCallback(TrainerCallback):
    def __init__(
        self,
        project: str,
        entity: Optional[str],
        run_name: Optional[str],
        mode: str,
        tags: Optional[List[str]] = None,
        watch_model: bool = False,
        watch_log: str = "gradients",
        watch_log_freq: int = 100,
    ):
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.mode = mode
        self.tags = tags or []
        self.watch_model = watch_model
        self.watch_log = watch_log
        self.watch_log_freq = watch_log_freq

        self._wandb = None
        self._inited = False

    def _maybe_init(self, args, state, **kwargs):
        if self._inited:
            return
        if not state.is_world_process_zero:
            return

        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except Exception:
                # Don't spam this message on every hook
                self._inited = True  # permanently disable this callback
                print("[Warn] wandb not installed; W&B logging disabled.")
                return

        # HF initializes wandb; wait until run exists
        if self._wandb.run is None:
            return

        # ---- define metrics once ----
        try:
            self._wandb.define_metric("trainer/global_step")
            self._wandb.define_metric("trainer/epoch")
            self._wandb.define_metric("train/*", step_metric="trainer/global_step")
            self._wandb.define_metric("val_step/*", step_metric="trainer/global_step")
            self._wandb.define_metric("val_epoch/*", step_metric="trainer/epoch")
            self._wandb.define_metric("test_final/*", step_metric="trainer/epoch")
            self._wandb.define_metric("test_step/*", step_metric="trainer/global_step")
            self._wandb.define_metric("test_epoch/*", step_metric="trainer/epoch")
            self._wandb.define_metric("gpu/*", step_metric="trainer/global_step")
            self._wandb.define_metric("cpu/*", step_metric="trainer/global_step")
        except Exception:
            # Even if define_metric fails, don't repeatedly try
            pass

        # Optional model watch
        model = kwargs.get("model", None)
        if self.watch_model and model is not None:
            try:
                self._wandb.watch(model, log=self.watch_log, log_freq=self.watch_log_freq)
            except Exception:
                pass

        self._inited = True

    def on_train_begin(self, args, state, control, **kwargs):
        self._maybe_init(args, state, **kwargs)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This is a very reliable place because logging starts after run init in most setups
        self._maybe_init(args, state, **kwargs)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._maybe_init(args, state, **kwargs)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Light fallback in case on_log doesn't fire early (rare)
        self._maybe_init(args, state, **kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        return control
    
class WandbEvalLoggingCallback(TrainerCallback):
    """
    Logs evaluation metrics into:
      - val_step/*  for HF step evaluations (eval_* keys)
      - val_epoch/* for epoch-end evaluations (either:
           a) HF epoch evaluations (eval_* keys when eval_strategy="epoch")
           b) forced epoch-end evals we run with metric_key_prefix="eval_epoch" (eval_epoch_* keys)
    """
    def __init__(self, eval_strategy: str):
        self.eval_strategy = eval_strategy
        self._wandb = None

    def _get_wandb(self):
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb
            self._wandb = wandb
            return self._wandb
        except Exception:
            return None

    def on_evaluate(self, args, state, control, metrics: Dict[str, float], **kwargs):
        if not state.is_world_process_zero:
            return control
        wb = self._get_wandb()
        if wb is None or wb.run is None:
            return control

        # Determine whether this evaluation is "epoch" or "step"
        is_forced_epoch_eval = any(k.startswith("eval_epoch_") for k in metrics.keys())

        if is_forced_epoch_eval:
            src_prefix = "eval_epoch_"
            dst_prefix = "val_epoch/"
            # epoch axis
            epoch_val = float(state.epoch) if state.epoch is not None else float("nan")
            payload = {"trainer/epoch": epoch_val}
            for k, v in metrics.items():
                if k.startswith(src_prefix):
                    payload[dst_prefix + k[len(src_prefix):]] = v
            # also put the matching global_step for convenience
            payload["trainer/global_step"] = int(state.global_step)
            wb.log(payload)
            return control

        # Not forced epoch eval; it's the Trainer's standard eval_* keys.
        # If user chose eval_strategy="epoch", treat it as epoch-level for panel separation.
        if self.eval_strategy == "epoch":
            dst_prefix = "val_epoch/"
            epoch_val = float(state.epoch) if state.epoch is not None else float("nan")
            payload = {"trainer/epoch": epoch_val, "trainer/global_step": int(state.global_step)}
            for k, v in metrics.items():
                if k.startswith("eval_"):
                    payload[dst_prefix + k[len("eval_"):]] = v
            wb.log(payload)
        else:
            # eval_strategy == "steps" => step-level eval
            dst_prefix = "val_step/"
            payload = {"trainer/global_step": int(state.global_step)}
            for k, v in metrics.items():
                if k.startswith("eval_"):
                    payload[dst_prefix + k[len("eval_"):]] = v
            # add epoch for context (won't be the x-axis for val_step/*)
            if state.epoch is not None:
                payload["trainer/epoch"] = float(state.epoch)
            wb.log(payload)

        return control
    
class WandbGpuStatsCallback(TrainerCallback):
    """
    Logs GPU memory + utilization (multi-GPU) and some CPU stats to W&B under gpu/* and cpu/*.
    """
    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._wandb = None
        self._psutil = None

    def _get_wandb(self):
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb
            self._wandb = wandb
            return self._wandb
        except Exception:
            return None

    def _get_psutil(self):
        if self._psutil is not None:
            return self._psutil
        try:
            import psutil
            self._psutil = psutil
            return self._psutil
        except Exception:
            return None

    def _nvidia_smi_query(self) -> List[Dict[str, float]]:
        """
        Returns one dict per GPU, best-effort.
        """
        fields = [
            "index",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
            "memory.free",
            "power.draw",
            "power.limit",
            "temperature.gpu",
            "fan.speed",
        ]
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return []

        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue
            d = {}
            for k, v in zip(fields, parts):
                # index is int, others float
                if k == "index":
                    d[k] = int(v)
                else:
                    try:
                        d[k] = float(v)
                    except Exception:
                        d[k] = float("nan")
            gpus.append(d)
        return gpus

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step % self.log_every_n_steps != 0:
            return control

        wb = self._get_wandb()
        if wb is None or wb.run is None:
            return control

        payload: Dict[str, float] = {"trainer/global_step": int(state.global_step)}
        if state.epoch is not None:
            payload["trainer/epoch"] = float(state.epoch)

        # torch-side CUDA memory stats
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                try:
                    payload[f"gpu/{i}/mem_allocated_bytes"] = float(torch.cuda.memory_allocated(i))
                    payload[f"gpu/{i}/mem_reserved_bytes"] = float(torch.cuda.memory_reserved(i))
                    payload[f"gpu/{i}/max_mem_allocated_bytes"] = float(torch.cuda.max_memory_allocated(i))
                    payload[f"gpu/{i}/max_mem_reserved_bytes"] = float(torch.cuda.max_memory_reserved(i))
                except Exception:
                    pass

        # nvidia-smi stats (utilization, power, temps, memory MiB)
        smi = self._nvidia_smi_query()
        for g in smi:
            i = g.get("index", -1)
            if i < 0:
                continue
            payload[f"gpu/{i}/util_gpu_pct"] = g.get("utilization.gpu", float("nan"))
            payload[f"gpu/{i}/util_mem_pct"] = g.get("utilization.memory", float("nan"))
            payload[f"gpu/{i}/mem_total_MiB"] = g.get("memory.total", float("nan"))
            payload[f"gpu/{i}/mem_used_MiB"] = g.get("memory.used", float("nan"))
            payload[f"gpu/{i}/mem_free_MiB"] = g.get("memory.free", float("nan"))
            payload[f"gpu/{i}/power_draw_W"] = g.get("power.draw", float("nan"))
            payload[f"gpu/{i}/power_limit_W"] = g.get("power.limit", float("nan"))
            payload[f"gpu/{i}/temp_C"] = g.get("temperature.gpu", float("nan"))
            payload[f"gpu/{i}/fan_pct"] = g.get("fan.speed", float("nan"))

        # cpu stats
        ps = self._get_psutil()
        if ps is not None:
            try:
                payload["cpu/percent"] = float(ps.cpu_percent(interval=None))
                vm = ps.virtual_memory()
                payload["cpu/ram_used_bytes"] = float(vm.used)
                payload["cpu/ram_total_bytes"] = float(vm.total)
            except Exception:
                pass

        wb.log(payload)
        return control
    
class ForcedEpochEvalCallback(TrainerCallback):
    """
    If eval_strategy == "steps", run an additional evaluation at each epoch end:
      trainer.evaluate(..., metric_key_prefix="eval_epoch")
    """
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset
        self.trainer: Optional[Trainer] = None
        self._in_forced = False

    def attach_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or self.eval_dataset is None:
            return control
        if not state.is_world_process_zero:
            return control

        strat = _strategy_to_str(getattr(args, "eval_strategy", None) or getattr(args, "evaluation_strategy", None))
        if strat != "steps":
            return control

        if self._in_forced:
            return control

        self._in_forced = True
        try:
            self.trainer.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval_epoch")
        finally:
            self._in_forced = False

        return control
    
class MirrorValEvalToTestCallback(TrainerCallback):
    """
    Whenever Trainer finishes a VAL evaluation (step/epoch/forced-epoch),
    immediately run TEST evaluation too.

    We run this on *all processes* (important for DDP) but only rank0 will log to W&B
    via your existing logging callbacks / HF integration.
    """
    def __init__(self, test_dataset, eval_strategy: str):
        self.test_dataset = test_dataset
        self.eval_strategy = eval_strategy
        self.trainer: Optional[Trainer] = None
        self._in_mirror = False

    def attach_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics: Optional[Dict[str, float]] = None, **kwargs):
        if self.trainer is None or self.test_dataset is None:
            return control

        if self._in_mirror:
            return control

        metrics = metrics or {}

        # If this evaluate() call was already for TEST (prefix keys), do nothing.
        # (prevents infinite recursion)
        if any(k.startswith("test_") or k.startswith("test_step_") or k.startswith("test_epoch_") for k in metrics.keys()):
            return control

        # Only trigger when this was a VAL eval (Trainer eval_* keys)
        # - step eval => eval_*
        # - forced epoch eval => eval_epoch_*
        is_val_eval = any(k.startswith("eval_") for k in metrics.keys())
        if not is_val_eval:
            return control

        # Decide whether the VAL eval was "epoch-like" or "step-like"
        is_forced_epoch_eval = any(k.startswith("eval_epoch_") for k in metrics.keys())
        if is_forced_epoch_eval or self.eval_strategy == "epoch":
            test_prefix = "test_epoch"
        else:
            test_prefix = "test_step"

        # Run TEST eval immediately after VAL eval
        self._in_mirror = True
        try:
            self.trainer.evaluate(eval_dataset=self.test_dataset, metric_key_prefix=test_prefix)
        finally:
            self._in_mirror = False

        return control
    
class WandbTestLoggingCallback(TrainerCallback):
    def __init__(self):
        self._wandb = None

    def _get_wandb(self):
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb
            self._wandb = wandb
            return self._wandb
        except Exception:
            return None

    def on_evaluate(self, args, state, control, metrics: Dict[str, float], **kwargs):
        if not state.is_world_process_zero:
            return control
        wb = self._get_wandb()
        if wb is None or wb.run is None:
            return control

        metrics = metrics or {}

        # step-level test
        if any(k.startswith("test_step_") for k in metrics.keys()):
            payload = {"trainer/global_step": int(state.global_step)}
            if state.epoch is not None:
                payload["trainer/epoch"] = float(state.epoch)
            for k, v in metrics.items():
                if k.startswith("test_step_"):
                    payload["test_step/" + k[len("test_step_"):]] = v
            wb.log(payload)
            return control

        # epoch-level test
        if any(k.startswith("test_epoch_") for k in metrics.keys()):
            payload = {"trainer/epoch": float(state.epoch) if state.epoch is not None else float("nan")}
            payload["trainer/global_step"] = int(state.global_step)
            for k, v in metrics.items():
                if k.startswith("test_epoch_"):
                    payload["test_epoch/" + k[len("test_epoch_"):]] = v
            wb.log(payload)

        return control
    
class SaveEverythingCallback(TrainerCallback):
    def __init__(self, processor, class_names: List[str]):
        self.processor = processor
        self.class_names = class_names
        self._pending_epoch_tag: Optional[int] = None

    def on_epoch_end(self, args, state, control, **kwargs):
        # Mandatory save at epoch end (even if save_strategy says otherwise)
        if state.is_world_process_zero:
            # We'll create epoch alias on the next on_save
            if state.epoch is not None:
                self._pending_epoch_tag = int(math.floor(state.epoch + 1e-9))
        control.should_save = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Save only for VAL evaluations (eval_* or eval_epoch_*), not for TEST mirrors.
        if metrics is None:
            control.should_save = True
            return control

        if any(k.startswith("eval_") for k in metrics.keys()):
            control.should_save = True

        return control

    def on_save(self, args, state, control, **kwargs):
        # HF checkpoint directory name (resume expects this pattern)
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _safe_makedirs(ckpt_dir)

        # Save processor/tokenizer for exact resume
        try:
            self.processor.save_pretrained(ckpt_dir)
        except Exception:
            pass

        # Save label metadata
        meta = {"num_labels": len(self.class_names), "class_names": self.class_names}
        try:
            with open(os.path.join(ckpt_dir, "labels_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        # Aliases requested:
        #  - checkpoint-step-XXXXXX -> checkpoint-<global_step>
        #  - checkpoint-epoch-YYY   -> checkpoint-<global_step>  (only when save was at epoch end)
        #  - checkpoint-last        -> latest checkpoint
        step_alias = os.path.join(args.output_dir, f"checkpoint-step-{state.global_step:06d}")
        _symlink_dir(ckpt_dir, step_alias)

        last_alias = os.path.join(args.output_dir, "checkpoint-last")
        _symlink_dir(ckpt_dir, last_alias)

        if self._pending_epoch_tag is not None:
            epoch_alias = os.path.join(args.output_dir, f"checkpoint-epoch-{self._pending_epoch_tag:03d}")
            _symlink_dir(ckpt_dir, epoch_alias)
            self._pending_epoch_tag = None

        # Best alias if this is the best checkpoint
        best_ckpt = state.best_model_checkpoint
        if best_ckpt is not None:
            # best_ckpt is a path; compare normalized
            try:
                if os.path.normpath(best_ckpt) == os.path.normpath(ckpt_dir):
                    best_alias = os.path.join(args.output_dir, "checkpoint-best")
                    _symlink_dir(ckpt_dir, best_alias)
            except Exception:
                pass

        return control
    
class WandbPerLabelRocCurveCallbackInbuilt(TrainerCallback):
    """
    Logs ROC curves on EVERY val evaluation (step evals + epoch-end evals)
    into a SINGLE panel:  {key_prefix}/all

    Also can optionally keep the old per-label and micro/macro panels, but
    default is to avoid clutter.
    """

    def __init__(
        self,
        eval_dataset,
        class_names,
        eval_strategy: str = "steps",
        classes_to_plot=(1,),
        max_labels: int | None = None,
        key_prefix: str = "val/roc",
        macro_grid_size: int = 400,
        # NEW switches
        log_all_in_one_panel: bool = True,
        log_individual_per_label: bool = False,
        log_separate_micro_macro: bool = False,
    ):
        self.eval_dataset = eval_dataset
        self.class_names = class_names
        self.eval_strategy = eval_strategy
        self.classes_to_plot = list(classes_to_plot) if classes_to_plot is not None else None
        self.max_labels = max_labels
        self.key_prefix = key_prefix
        self.macro_grid_size = int(macro_grid_size)

        self.log_all_in_one_panel = bool(log_all_in_one_panel)
        self.log_individual_per_label = bool(log_individual_per_label)
        self.log_separate_micro_macro = bool(log_separate_micro_macro)

        self.trainer = None
        self._wandb = None
        self._in_roc = False

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def _get_wandb(self):
        if self._wandb is not None:
            return self._wandb
        try:
            import wandb
            self._wandb = wandb
            return wandb
        except Exception:
            return None

    @staticmethod
    def _safe_key(name: str) -> str:
        import re
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
        return s if s else "label"

    @staticmethod
    def _diagonal():
        x = np.linspace(0.0, 1.0, 50, dtype=np.float32)
        return x, x

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # rank0 only
        if not state.is_world_process_zero:
            return control

        wb = self._get_wandb()
        if wb is None or wb.run is None:
            return control

        if self._in_roc or self.trainer is None or self.eval_dataset is None:
            return control

        metrics = metrics or {}

        # ---- skip TEST evals ----
        if any(k.startswith("test_") or k.startswith("test_step_") or k.startswith("test_epoch_") for k in metrics.keys()):
            return control

        # ---- only run on VAL evals ----
        if not any(k.startswith("eval_") for k in metrics.keys()):
            return control

        self._in_roc = True
        try:
            # expensive but required for curve points
            pred = self.trainer.predict(self.eval_dataset)
            logits = pred.predictions
            labels = pred.label_ids

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

            payload = {"trainer/global_step": int(state.global_step)}
            if state.epoch is not None:
                payload["trainer/epoch"] = float(state.epoch)

            # -------------------------
            # MICRO (flatten all known 0/1)
            # -------------------------
            fpr_micro = tpr_micro = None
            try:
                m = np.isin(labels, [0.0, 1.0])
                if m.sum() >= 2:
                    y_flat = labels[m].astype(int)
                    p_flat = probs[m].astype(np.float32)
                    if len(np.unique(y_flat)) >= 2:
                        fpr_micro, tpr_micro, _ = roc_curve(y_flat, p_flat, pos_label=1)
            except Exception:
                fpr_micro = tpr_micro = None

            # -------------------------
            # PER-LABEL + collect for MACRO
            # -------------------------
            per_label_rocs = []   # list[(fpr, tpr, name)]
            N, L = labels.shape
            idxs = list(range(L))
            if self.max_labels is not None:
                idxs = idxs[: int(self.max_labels)]

            for j in idxs:
                name = self.class_names[j] if j < len(self.class_names) else f"label_{j}"
                yj = labels[:, j]
                pj = probs[:, j]

                mj = np.isin(yj, [0.0, 1.0])
                if mj.sum() < 2:
                    continue
                yb = yj[mj].astype(int)
                if len(np.unique(yb)) < 2:
                    continue

                p_pos = pj[mj].astype(np.float32)

                # collect arrays for combined + macro
                try:
                    fpr_j, tpr_j, _ = roc_curve(yb, p_pos, pos_label=1)
                    per_label_rocs.append((fpr_j, tpr_j, name))
                except Exception:
                    pass

                # optionally still log individual per-label (cluttery)
                if self.log_individual_per_label:
                    key = self._safe_key(name)
                    y_probas = np.stack([1.0 - p_pos, p_pos], axis=1)
                    roc_chart = wb.plot.roc_curve(
                        y_true=yb.tolist(),
                        y_probas=y_probas.tolist(),
                        labels=["neg", "pos"],
                        classes_to_plot=self.classes_to_plot,
                        title=f"ROC Curve — {name}",
                    )
                    payload[f"{self.key_prefix}/{key}"] = roc_chart

            # -------------------------
            # MACRO (mean of per-label TPR on shared grid)
            # -------------------------
            fpr_grid = tpr_macro = None
            if len(per_label_rocs) > 0:
                fpr_grid = np.linspace(0.0, 1.0, num=max(50, self.macro_grid_size), dtype=np.float32)
                tprs = []
                for fpr_j, tpr_j, _ in per_label_rocs:
                    tprs.append(np.interp(fpr_grid, fpr_j, tpr_j).astype(np.float32))
                tpr_macro = np.mean(np.stack(tprs, axis=0), axis=0)

            # -------------------------
            # FALLBACKS so we ALWAYS log the same keys
            # -------------------------
            if fpr_micro is None or tpr_micro is None:
                fpr_micro, tpr_micro = self._diagonal()
            if fpr_grid is None or tpr_macro is None:
                fpr_grid, tpr_macro = self._diagonal()

            # -------------------------
            # SINGLE PANEL: micro + macro + all per-label lines
            # -------------------------
            if self.log_all_in_one_panel:
                xs, ys, keys = [], [], []

                # micro, macro first
                xs.append(fpr_micro.tolist()); ys.append(tpr_micro.tolist()); keys.append("micro")
                xs.append(fpr_grid.tolist());  ys.append(tpr_macro.tolist()); keys.append("macro")

                # per-label curves
                for fpr_j, tpr_j, name in per_label_rocs:
                    xs.append(fpr_j.tolist())
                    ys.append(tpr_j.tolist())
                    keys.append(name)

                # If no per-label curves exist, still log micro+macro (already diagonal fallbacks)
                all_chart = wb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title="ROC Curves (all labels + micro/macro)",
                    xname="False Positive Rate (FPR)",
                    split_table=True,
                )
                payload[f"{self.key_prefix}/all"] = all_chart

            # -------------------------
            # OPTIONAL: keep old separate panels (micro/macro/micro_vs_macro)
            # -------------------------
            if self.log_separate_micro_macro:
                micro_chart = wb.plot.line_series(
                    xs=[fpr_micro.tolist()],
                    ys=[tpr_micro.tolist()],
                    keys=["micro"],
                    title="ROC Curve (micro)",
                    xname="False Positive Rate (FPR)",
                    split_table=True,
                )
                payload[f"{self.key_prefix}/micro"] = micro_chart

                macro_chart = wb.plot.line_series(
                    xs=[fpr_grid.tolist()],
                    ys=[tpr_macro.tolist()],
                    keys=["macro"],
                    title="ROC Curve (macro avg)",
                    xname="False Positive Rate (FPR)",
                    split_table=True,
                )
                payload[f"{self.key_prefix}/macro"] = macro_chart

                combo_chart = wb.plot.line_series(
                    xs=[fpr_micro.tolist(), fpr_grid.tolist()],
                    ys=[tpr_micro.tolist(), tpr_macro.tolist()],
                    keys=["micro", "macro"],
                    title="ROC Curve (micro vs macro)",
                    xname="False Positive Rate (FPR)",
                    split_table=True,
                )
                payload[f"{self.key_prefix}/micro_vs_macro"] = combo_chart

            wb.log(payload)

        finally:
            self._in_roc = False

        return control
    
class EpochAndBestPredictionDumpCallback(TrainerCallback):
    """
    Dumps per-example y_true + probs (and optional logits)
      1) at the end of every epoch  -> pred_dumps/predictions_{split}_epoch-EEE_step-SSSSSS.csv
      2) after training ends (best model loaded if load_best_model_at_end=True)
                                    -> pred_dumps/predictions_{split}_BEST_step-SSSSSS.csv

    Assumes:
      - you already added `filename=...`, `epoch=...`, `global_step=...` params to export_predictions_csv()
      - you call cb.attach_trainer(trainer)
    """
    def __init__(
        self,
        class_names: List[str],
        val_dataset=None,
        test_dataset=None,
        threshold: float = 0.5,
        dump_val: bool = True,
        dump_test: bool = True,
        every_n_epochs: int = 1,
        save_logits: bool = False,
        include_image_path: bool = False,
        include_text_len: bool = True,
    ):
        self.class_names = class_names
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.threshold = float(threshold)
        self.dump_val = bool(dump_val)
        self.dump_test = bool(dump_test)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_logits = bool(save_logits)
        self.include_image_path = bool(include_image_path)
        self.include_text_len = bool(include_text_len)

        self.trainer: Optional[Trainer] = None
        self._in_dump = False

    def attach_trainer(self, trainer: Trainer):
        self.trainer = trainer

    @staticmethod
    def _best_step_from_ckpt_path(p: Optional[str]) -> Optional[int]:
        if not p:
            return None
        # expects .../checkpoint-12345
        try:
            base = os.path.basename(os.path.normpath(p))
            if base.startswith("checkpoint-"):
                return int(base.split("-")[-1])
        except Exception:
            pass
        return None

    def _dump_split(self, args, state, split_name: str, dataset, filename: str, epoch: Optional[int]):
        if dataset is None or len(dataset) == 0:
            return
        export_dir = os.path.join(args.output_dir, "pred_dumps", split_name)  # <-- split-wise folder
        _safe_makedirs(export_dir)

        export_predictions_csv(
            trainer=self.trainer,
            dataset=dataset,
            split_name=split_name,
            output_dir=export_dir,     # <-- now goes into pred_dumps/val or pred_dumps/test
            class_names=self.class_names,
            threshold=self.threshold,
            save_logits=self.save_logits,
            include_image_path=self.include_image_path,
            include_text_len=self.include_text_len,
            filename=filename,
            epoch=epoch,
            global_step=int(state.global_step),
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        if self.trainer is None or self._in_dump:
            return control

        # Correct epoch index at epoch end: state.epoch is usually 1.0, 2.0, ...
        epoch_idx = None
        if state.epoch is not None:
            try:
                epoch_idx = int(round(float(state.epoch)))
            except Exception:
                epoch_idx = None

        # If we can't determine epoch, skip safely
        if epoch_idx is None or epoch_idx <= 0:
            return control

        if (epoch_idx % self.every_n_epochs) != 0:
            return control

        gs = int(state.global_step)

        self._in_dump = True
        try:
            if self.dump_val and self.val_dataset is not None:
                self._dump_split(
                    args, state,
                    split_name="val",
                    dataset=self.val_dataset,
                    filename=f"predictions_val_epoch-{epoch_idx:03d}_step-{gs:06d}.csv",
                    epoch=epoch_idx,
                )
            if self.dump_test and self.test_dataset is not None:
                self._dump_split(
                    args, state,
                    split_name="test",
                    dataset=self.test_dataset,
                    filename=f"predictions_test_epoch-{epoch_idx:03d}_step-{gs:06d}.csv",
                    epoch=epoch_idx,
                )
        finally:
            self._in_dump = False

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """
        At this point, if TrainingArguments(load_best_model_at_end=True),
        HF Trainer has loaded the best checkpoint into trainer.model.
        So exports here correspond to the BEST model.
        """
        if not state.is_world_process_zero:
            return control
        if self.trainer is None or self._in_dump:
            return control

        gs = int(state.global_step)
        best_ckpt = getattr(state, "best_model_checkpoint", None)
        best_step = self._best_step_from_ckpt_path(best_ckpt)

        # Use best_step in filename if we could parse it
        best_tag = f"{best_step:06d}" if isinstance(best_step, int) else f"{gs:06d}"

        self._in_dump = True
        try:
            if self.dump_val and self.val_dataset is not None:
                self._dump_split(
                    args, state,
                    split_name="val",
                    dataset=self.val_dataset,
                    filename=f"predictions_val_BEST_step-{best_tag}.csv",
                    epoch=None,  # best may not map cleanly to an epoch
                )
            if self.dump_test and self.test_dataset is not None:
                self._dump_split(
                    args, state,
                    split_name="test",
                    dataset=self.test_dataset,
                    filename=f"predictions_test_BEST_step-{best_tag}.csv",
                    epoch=None,
                )

            # Optional: write a small JSON pointer to best checkpoint path/metric
            try:
                export_dir = os.path.join(args.output_dir, "pred_dumps")
                _safe_makedirs(export_dir)
                with open(os.path.join(export_dir, "best_model_info.json"), "w") as f:
                    json.dump(
                        {
                            "best_model_checkpoint": best_ckpt,
                            "best_metric": getattr(state, "best_metric", None),
                            "best_step": best_step,
                            "train_end_global_step": gs,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass
        finally:
            self._in_dump = False

        return control