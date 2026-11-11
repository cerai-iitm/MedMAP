#!/usr/bin/env python3
"""Fine-tune SmolVLM on the Tiny-ImageNet-200 ImageNet-Captions task.

The command accepts normal CLI arguments while retaining environment-variable
fallbacks used by older launchers. Run ``--validate-only`` after downloading
the data to verify the source layout, 200-class caption join, and fixed split
indices without downloading a model or starting training.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scripts._bootstrap import setup_project_paths

setup_project_paths()

import medmap.datasets.imagenet as _imagenet_pkg

SPLIT_DIR = Path(_imagenet_pkg.__file__).resolve().parent / "splits"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off.")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data/model paths. Empty environment fallbacks remain required at parse time.
    parser.add_argument("--captions-json", default=os.environ.get("CAPTIONS_JSON"))
    parser.add_argument("--root-zip-path", default=os.environ.get("ROOT_ZIP_PATH"))
    parser.add_argument(
        "--train-txt",
        default=os.environ.get("TRAIN_TXT", "ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
        help="Path inside the extracted ILSVRC root or competition ZIP.",
    )
    parser.add_argument(
        "--tiny-imagenet-wnids", default=os.environ.get("TINY_IMAGENET_WNIDS")
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolVLM-256M-Base"),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUT_DIR"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))

    # Modalities/privacy.
    parser.add_argument(
        "--include-img", action=argparse.BooleanOptionalAction,
        default=_env_bool("INCLUDE_IMG", True),
    )
    parser.add_argument(
        "--include-text", action=argparse.BooleanOptionalAction,
        default=_env_bool("INCLUDE_TEXT", True),
    )
    parser.add_argument(
        "--use-dp", action=argparse.BooleanOptionalAction,
        default=_env_bool("USE_DP", False),
    )
    parser.add_argument(
        "--feature-dp", action=argparse.BooleanOptionalAction,
        default=_env_bool("FEATURE_DP", False),
        help="With --use-dp: Feature-DP when enabled, full DP-SGD when disabled.",
    )
    parser.add_argument(
        "--target-epsilon", type=float,
        default=float(os.environ.get("TARGET_EPS", "1.0")),
    )
    parser.add_argument(
        "--clip-norm", type=float, default=float(os.environ.get("CLIP_NORM", "1.0"))
    )
    parser.add_argument(
        "--public-clip-norm", type=float,
        default=float(os.environ.get("PUBLIC_CLIP", "50.0")),
    )

    # Optimization/splits.
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "100")))
    parser.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", "10")))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("BATCH", "16")),
        help="Non-private per-device batch; DP microbatch base before accumulation expansion.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int,
        default=int(os.environ.get("GRAD_ACCUM", "64")),
    )
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.environ.get("LR", "5e-5"))
    )
    parser.add_argument(
        "--weight-decay", type=float,
        default=float(os.environ.get("WEIGHT_DECAY", "0.01")),
    )
    parser.add_argument(
        "--warmup-ratio", type=float,
        default=float(os.environ.get("WARMUP_RATIO", "0.03")),
    )
    parser.add_argument(
        "--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "128"))
    )
    parser.add_argument(
        "--use-fixed-split", action=argparse.BooleanOptionalAction,
        default=_env_bool("USE_SPLIT_IDX", True),
        help="Use the packaged reported split indices (default: enabled).",
    )

    # Adapter/memory.
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=_env_bool("GRAD_CHECKPOINT", False),
    )
    parser.add_argument(
        "--use-lora", action=argparse.BooleanOptionalAction,
        default=_env_bool("USE_LORA", True),
    )
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "16")))
    parser.add_argument(
        "--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32"))
    )
    parser.add_argument(
        "--lora-dropout", type=float,
        default=float(os.environ.get("LORA_DROPOUT", "0.05")),
    )

    # Trainer behavior.
    parser.add_argument(
        "--logging-strategy", choices=["steps", "epoch"],
        default=os.environ.get("LOGGING_STRATEGY", "steps"),
    )
    parser.add_argument(
        "--logging-steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "1"))
    )
    parser.add_argument(
        "--save-strategy", choices=["steps", "epoch"],
        default=os.environ.get("SAVE_STRATEGY", "steps"),
    )
    parser.add_argument(
        "--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "700"))
    )
    parser.add_argument(
        "--eval-strategy", choices=["steps", "epoch"],
        default=os.environ.get("EVAL_STRATEGY", "steps"),
    )
    parser.add_argument(
        "--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "700"))
    )
    parser.add_argument(
        "--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume-from-checkpoint", default=None)

    # Tracking is disabled by default for anonymous/offline artifact use.
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "medmap-tiny-imagenet")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate and assemble the dataset, check split indices, then exit before model loading.",
    )
    return parser


def _validate_source_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.captions_json:
        parser.error("--captions-json is required (or set CAPTIONS_JSON).")
    if not args.root_zip_path:
        parser.error("--root-zip-path is required (or set ROOT_ZIP_PATH).")
    if not args.tiny_imagenet_wnids:
        parser.error("--tiny-imagenet-wnids is required (or set TINY_IMAGENET_WNIDS).")

    captions = Path(args.captions_json)
    wnids = Path(args.tiny_imagenet_wnids)
    source = Path(args.root_zip_path)
    if not captions.is_file():
        parser.error(f"ImageNet-Captions JSON does not exist: {captions}")
    if not wnids.is_file():
        parser.error(f"Tiny ImageNet wnids.txt does not exist: {wnids}")
    class_ids = [line.strip() for line in wnids.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(class_ids) != 200 or len(set(class_ids)) != 200:
        parser.error(
            f"Expected 200 unique Tiny ImageNet class IDs in {wnids}; found "
            f"{len(class_ids)} rows and {len(set(class_ids))} unique values."
        )

    if source.is_dir():
        train_list = source / args.train_txt
        if not train_list.is_file():
            parser.error(f"ILSVRC train listing does not exist: {train_list}")
    elif source.is_file():
        if not zipfile.is_zipfile(source):
            parser.error(f"--root-zip-path is not a readable ZIP or directory: {source}")
        with zipfile.ZipFile(source) as archive:
            if args.train_txt not in archive.namelist():
                parser.error(f"{args.train_txt} is missing from ILSVRC ZIP: {source}")
    else:
        parser.error(f"ILSVRC root/ZIP does not exist: {source}")


def _validate_fixed_indices(dataset_size: int) -> dict[str, int]:
    import numpy as np

    arrays = {
        "train": np.load(SPLIT_DIR / "train_idx.npy"),
        "validation": np.load(SPLIT_DIR / "val_idx.npy"),
        "test": np.load(SPLIT_DIR / "test_idx.npy"),
    }
    for name, values in arrays.items():
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Packaged {name} indices must be a one-dimensional integer array.")
        if len(values) and (int(values.min()) < 0 or int(values.max()) >= dataset_size):
            raise ValueError(
                f"Packaged {name} indices are out of bounds for dataset size {dataset_size}."
            )
        if len(values) != len(np.unique(values)):
            raise ValueError(f"Packaged {name} indices contain duplicates.")
    names = list(arrays)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if np.intersect1d(arrays[left], arrays[right]).size:
                raise ValueError(f"Packaged split indices overlap: {left} and {right}.")
    combined = np.concatenate(list(arrays.values()))
    if len(np.unique(combined)) != dataset_size:
        raise ValueError(
            "Packaged split indices do not cover the assembled dataset exactly: "
            f"covered={len(np.unique(combined))}, dataset_size={dataset_size}."
        )
    return {name: len(values) for name, values in arrays.items()}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_source_paths(args, parser)

    if not args.use_dp and not (args.include_img or args.include_text):
        parser.error("At least one of --include-img or --include-text must be enabled.")
    if args.target_epsilon <= 0 or args.clip_norm <= 0 or args.public_clip_norm <= 0:
        parser.error("Privacy epsilon and clip norms must be positive.")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("--batch-size and --gradient-accumulation-steps must be positive.")
    if args.load_best_model_at_end and args.save_strategy != args.eval_strategy:
        parser.error(
            "--save-strategy and --eval-strategy must match when best-model loading is enabled."
        )

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    from medmap.datasets.imagenet.builder import build_imagenet_dataset

    full_ds, num_classes, class_names = build_imagenet_dataset(
        args.captions_json,
        args.root_zip_path,
        args.train_txt,
        use_tiny=True,
        tiny_wnids_path=args.tiny_imagenet_wnids,
        validate_images=args.validate_only,
    )
    if num_classes != 200:
        raise ValueError(
            f"Expected the caption/image join to contain all 200 Tiny ImageNet classes; "
            f"found {num_classes}. Check the downloaded sources."
        )

    fixed_split_sizes = None
    if args.use_fixed_split or args.validate_only:
        fixed_split_sizes = _validate_fixed_indices(len(full_ds))

    if args.validate_only:
        print(
            "Tiny ImageNet data validation passed: "
            f"records={len(full_ds)}, classes={num_classes}, "
            f"fixed_split_sizes={fixed_split_sizes}"
        )
        return

    import numpy as np
    import torch
    from datasets import Image as DatasetsImage
    from dp_transformers.arguments import PrivacyArguments
    from transformers import AutoProcessor, TrainingArguments
    from transformers.models.smolvlm import SmolVLMConfig

    from medmap.common.utils import seed_everything
    from medmap.models.smolvlm import setup_model
    from medmap.training.callbacks import (
        MemoryCleanupCallback,
        SaveManagerCallback,
        SplitEvalCallback,
        WandbGpuCpuMemoryCallback,
    )
    from medmap.training.collators import SmolVLMHeteroNoDummyCollator
    from medmap.training.metrics import compute_metrics_fn
    from medmap.training.trainers import get_trainer

    use_wandb = args.wandb_mode != "disabled"
    wandb = None
    if use_wandb:
        import wandb as wandb_module
        wandb = wandb_module

    include_img = args.include_img
    include_text = args.include_text
    per_device_batch = args.batch_size
    if args.use_dp:
        include_img = True
        include_text = True
        per_device_batch *= args.gradient_accumulation_steps

    setting = "nonprivate"
    if args.use_dp:
        setting = "feature-dp" if args.feature_dp else "full-dp"
    stamp = _timestamp()
    output_dir = args.output_dir or str(Path("outputs") / "tiny-imagenet" / f"{setting}-{stamp}")
    run_name = args.run_name or f"tiny-imagenet-{setting}-seed-{args.seed}-{stamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.truncation_side = "right"
        if "<image>" not in processor.tokenizer.added_tokens_encoder:
            processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    config = SmolVLMConfig.from_pretrained(args.model_name, attn_implementation="eager")
    config.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    full_ds = full_ds.cast_column("image_path", DatasetsImage(decode=True))

    if args.use_fixed_split:
        full_ds = full_ds.add_column("orig_idx", list(range(len(full_ds))))
        train_ds = full_ds.select(np.load(SPLIT_DIR / "train_idx.npy"))
        val_ds = full_ds.select(np.load(SPLIT_DIR / "val_idx.npy"))
        test_ds = full_ds.select(np.load(SPLIT_DIR / "test_idx.npy"))
    else:
        split_1 = full_ds.train_test_split(
            test_size=0.40, seed=args.seed, stratify_by_column="class_idx"
        )
        train_ds = split_1["train"]
        split_2 = split_1["test"].train_test_split(
            test_size=0.50, seed=args.seed, stratify_by_column="class_idx"
        )
        val_ds, test_ds = split_2["train"], split_2["test"]

    model = setup_model(
        args.model_name,
        config,
        num_classes,
        args.use_lora,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(processor, "tokenizer"):
        model.resize_token_embeddings(len(processor.tokenizer))

    data_collator = SmolVLMHeteroNoDummyCollator(
        processor=processor,
        text_key="prompt",
        image_key="image_path",
        max_length=args.max_length,
        want_image=include_img,
        want_text=include_text,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        adam_beta2=0.999,
        adam_epsilon=1e-4,
        logging_first_step=True,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        eval_on_start=False,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        include_inputs_for_metrics=False,
        report_to=["wandb"] if use_wandb else [],
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        skip_memory_metrics=False,
        save_safetensors=True,
        data_seed=args.seed,
        seed=args.seed,
        run_name=run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [
        MemoryCleanupCallback(every_n_steps=20),
        SaveManagerCallback(
            processor=processor,
            class_names=class_names,
            base_model_name=args.model_name,
            use_lora=args.use_lora,
        ),
    ]
    if use_wandb:
        callbacks.insert(0, WandbGpuCpuMemoryCallback())

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "tokenizer": processor.tokenizer if hasattr(processor, "tokenizer") else None,
        "compute_metrics": compute_metrics_fn,
        "callbacks": callbacks,
    }
    if args.use_dp:
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=args.clip_norm / per_device_batch,
            public_clip=args.public_clip_norm / per_device_batch,
            target_epsilon=args.target_epsilon,
            feature_dp=args.feature_dp,
        )

    trainer = get_trainer(
        use_dp=args.use_dp,
        loss_mode="decomposed" if args.use_dp else "standard",
        **trainer_kwargs,
    )
    trainer.model_accepts_loss_kwargs = False

    split_callback = SplitEvalCallback(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)
    trainer.add_callback(split_callback)
    split_callback.attach_trainer(trainer)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    train_metrics = train_result.metrics
    train_metrics["num_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    trainer.save_state()

    if use_wandb and wandb is not None and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
