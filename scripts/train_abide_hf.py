#!/usr/bin/env python3
"""ABIDE I logistic regression — Hugging Face backend.

Trains the linear (logistic-regression) model implemented as an HF
``PreTrainedModel`` (medmap/models/abide_linear.py) with the same trainer
stack as the deep models, which is what enables per-sample clipping and
Feature-DP through the modified dp-transformers fork:

- non-private   (choose views with INCLUDE_FMRI / INCLUDE_SENSITIVE)
- full DP-SGD   (USE_DP=1 FEATURE_DP=0)
- Feature-DP    (USE_DP=1 FEATURE_DP=1; fMRI features are public, the
                 sensitive phenotype features are private)

Configuration is via environment variables (see examples/run_abide_hf_*.sh
and docs/abide.md). For the classic sklearn/NumPy backend that mirrors the
original notebook experiments, use scripts/train_abide_classic.py instead.

Required environment:
    ABIDE_DATA_DIR   ABIDE PCP root (contains the phenotypic CSV; fc_features/
                     by default lives underneath, override with ABIDE_FC_DIR).
"""

from __future__ import annotations

from scripts._bootstrap import setup_project_paths

REPO_ROOT = setup_project_paths()

import gc
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments

from dp_transformers.arguments import PrivacyArguments

from medmap.common.utils import (
    env_bool,
    env_float,
    env_int,
    env_str,
    get_formatted_datetime_string,
    seed_everything,
)

from medmap.datasets.abide.hf_pipeline import build_abide_linear_hf_datasets
from medmap.training.collators import ABIDELinearCollator
from medmap.training.metrics import compute_binary_metrics_fn
from medmap.models.abide_linear import setup_linear_abide_model
from medmap.training.trainers import get_trainer


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=clean)


def maybe_build_callbacks(use_wandb: bool):
    """Lightweight optional callbacks (memory logging / cleanup)."""
    callbacks = []

    try:
        from medmap.training.callbacks import WandbGpuCpuMemoryCallback, MemoryCleanupCallback

        if use_wandb:
            callbacks.append(WandbGpuCpuMemoryCallback())
        callbacks.append(MemoryCleanupCallback(every_n_steps=20))
    except Exception as exc:
        print(f"[WARN] Could not import optional callbacks: {exc}")

    return callbacks


def main():
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )

    # -------------------------
    # Paths / data config
    # -------------------------
    DATA_DIR = env_str("ABIDE_DATA_DIR", "")
    if not DATA_DIR:
        raise ValueError(
            "ABIDE_DATA_DIR must be set to your ABIDE PCP root "
            "(see docs/abide.md for download instructions)."
        )
    FC_DIR = env_str("ABIDE_FC_DIR", os.path.join(DATA_DIR, "fc_features"))
    FEATURE_JSON = env_str("FEATURE_JSON", "")
    PHENOTYPIC_CSV = env_str("PHENOTYPIC_CSV", "")

    # -------------------------
    # Training config
    # -------------------------
    SEED = env_int("SEED", 81)
    EPOCHS = env_int("EPOCHS", 100)
    PER_DEVICE_BATCH = env_int("BATCH", 64)
    GRAD_ACCUM = env_int("GRAD_ACCUM", 1)
    LR = env_float("LR", 1e-3)
    WEIGHT_DECAY = env_float("WEIGHT_DECAY", 1e-4)
    WARMUP_RATIO = env_float("WARMUP_RATIO", 0.0)

    LOGGING_STEPS = env_int("LOGGING_STEPS", 10)
    LOGGING_STRATEGY = env_str("LOGGING_STRATEGY", "steps")
    SAVE_STRATEGY = env_str("SAVE_STRATEGY", "epoch")
    SAVE_STEPS = env_int("SAVE_STEPS", 500)
    EVAL_STRATEGY = env_str("EVAL_STRATEGY", "epoch")
    EVAL_STEPS = env_int("EVAL_STEPS", 500)

    # -------------------------
    # Privacy config
    # -------------------------
    USE_DP = env_bool("USE_DP", False)
    FEATURE_DP = env_bool("FEATURE_DP", False)
    TARGET_EPS = env_float("TARGET_EPS", 8.0)
    CLIP_NORM = env_float("CLIP_NORM", 1.0)
    PUBLIC_CLIP = env_float("PUBLIC_CLIP", 50.0)

    # -------------------------
    # Modality config
    # -------------------------
    INCLUDE_FMRI = env_bool("INCLUDE_FMRI", True)
    INCLUDE_SENSITIVE = env_bool("INCLUDE_SENSITIVE", True)
    INCLUDE_SITE_ID_AS_FEATURE = env_bool("INCLUDE_SITE_ID_AS_FEATURE", False)

    # -------------------------
    # W&B config (WANDB_MODE=disabled turns W&B off entirely)
    # -------------------------
    WANDB_MODE = env_str("WANDB_MODE", "online")
    use_wandb = WANDB_MODE != "disabled"
    os.environ.setdefault("WANDB_PROJECT", "medmap-abide-linear")
    os.environ["WANDB_MODE"] = WANDB_MODE

    if not INCLUDE_FMRI and not INCLUDE_SENSITIVE:
        raise ValueError("At least one of INCLUDE_FMRI or INCLUDE_SENSITIVE must be True.")

    # For DP/Feature-DP, force both views (same convention as MIMIC/SmolVLM).
    if USE_DP:
        INCLUDE_FMRI = True
        INCLUDE_SENSITIVE = True

        # DP microbatching: the effective batch is batch * grad_accum.
        PER_DEVICE_BATCH = PER_DEVICE_BATCH * GRAD_ACCUM

    timestamp = get_formatted_datetime_string()

    if USE_DP:
        if FEATURE_DP:
            print("Feature DP")
            default_out = (
                f"outputs_abide_linear_featdp_"
                f"eps_{TARGET_EPS}_clip_{CLIP_NORM}_lr_{LR}_seed_{SEED}_{timestamp}"
            )
            default_run = (
                f"ABIDE-Linear-FeatureDP-eps-{TARGET_EPS}-"
                f"clip-{CLIP_NORM}-lr-{LR}-seed-{SEED}-{timestamp}"
            )
        else:
            print("Full DP")
            default_out = (
                f"outputs_abide_linear_fulldp_"
                f"eps_{TARGET_EPS}_clip_{CLIP_NORM}_lr_{LR}_seed_{SEED}_{timestamp}"
            )
            default_run = (
                f"ABIDE-Linear-FullDP-eps-{TARGET_EPS}-"
                f"clip-{CLIP_NORM}-lr-{LR}-seed-{SEED}-{timestamp}"
            )
    elif INCLUDE_FMRI and INCLUDE_SENSITIVE:
        print("Both Non Private")
        default_out = f"outputs_abide_linear_fmri_sensitive_{timestamp}"
        default_run = f"ABIDE-Linear-fMRI-Sensitive-{timestamp}"
    elif INCLUDE_FMRI:
        print("fMRI Only")
        default_out = f"outputs_abide_linear_fmri_only_{timestamp}"
        default_run = f"ABIDE-Linear-fMRIOnly-{timestamp}"
    else:
        print("Sensitive Only")
        default_out = f"outputs_abide_linear_sensitive_only_{timestamp}"
        default_run = f"ABIDE-Linear-SensitiveOnly-{timestamp}"

    OUT_DIR = env_str("OUT_DIR", default_out)
    RUN_NAME = env_str("RUN_NAME", default_run)

    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Dataset
    # -------------------------
    bundle = build_abide_linear_hf_datasets(
        data_dir=DATA_DIR,
        fc_dir=FC_DIR,
        feature_json_path=FEATURE_JSON if FEATURE_JSON else None,
        phenotypic_csv=PHENOTYPIC_CSV if PHENOTYPIC_CSV else None,
        seed=SEED,
        validation_size=0.10,
        test_size=0.20,
        include_site_id_as_feature=INCLUDE_SITE_ID_AS_FEATURE,
    )

    train_ds = bundle.train_ds
    val_ds = bundle.val_ds
    test_ds = bundle.test_ds
    metadata = bundle.metadata

    save_json(metadata, Path(OUT_DIR) / "abide_dataset_metadata.json")

    # -------------------------
    # Model
    # -------------------------
    model = setup_linear_abide_model(
        fmri_dim=metadata["fmri_dim"],
        sensitive_dim=metadata["sensitive_dim"],
        num_labels=metadata["num_labels"],
    )
    model = model.to(device)

    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # -------------------------
    # Collator
    # -------------------------
    data_collator = ABIDELinearCollator(
        want_fmri=INCLUDE_FMRI,
        want_sensitive=INCLUDE_SENSITIVE,
    )

    # -------------------------
    # TrainingArguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_torch",
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        logging_first_step=True,
        logging_strategy=LOGGING_STRATEGY,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        include_inputs_for_metrics=False,
        report_to=["wandb"] if use_wandb else [],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        skip_memory_metrics=False,
        save_safetensors=True,
        data_seed=SEED,
        run_name=RUN_NAME,
        load_best_model_at_end=True if SAVE_STRATEGY == EVAL_STRATEGY else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_binary_metrics_fn,
        "callbacks": maybe_build_callbacks(use_wandb),
    }

    if USE_DP:
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=CLIP_NORM / PER_DEVICE_BATCH,
            public_clip=PUBLIC_CLIP / PER_DEVICE_BATCH,
            target_epsilon=TARGET_EPS,
            feature_dp=FEATURE_DP,
        )

    trainer = get_trainer(
        use_dp=USE_DP,
        loss_mode="decomposed" if USE_DP else "standard",
        **trainer_kwargs,
    )

    trainer.model_accepts_loss_kwargs = False

    # -------------------------
    # Train
    # -------------------------
    train_result = trainer.train()

    trainer.save_model(OUT_DIR)

    train_metrics = train_result.metrics
    train_metrics["num_trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    train_metrics["include_fmri"] = bool(INCLUDE_FMRI)
    train_metrics["include_sensitive"] = bool(INCLUDE_SENSITIVE)
    train_metrics["use_dp"] = bool(USE_DP)
    train_metrics["feature_dp"] = bool(FEATURE_DP)

    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # -------------------------
    # Final validation/test eval
    # -------------------------
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    trainer.log_metrics("validation", val_metrics)
    trainer.save_metrics("validation", val_metrics)

    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    save_json(
        {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "metadata": metadata,
        },
        Path(OUT_DIR) / "final_results.json",
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_wandb:
        try:
            import wandb

            if wandb.run:
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
