#!/usr/bin/env python3
"""Fine-tune a Gemma 3 family VLM (MedGemma 4B or Gemma 3 4B) on MIMIC-CXR
multi-label classification.

Backbones (``--backbone``):

- medgemma  google/medgemma-4b-pt  (medically adapted reference model; delta = 1/n)
- gemma     google/gemma-3-4b-pt   (general-purpose base model; delta = n^-1.1)

Both share the Gemma 3 architecture, processor, and classifier wrapper
(medmap/models/gemma.py). The backbone preset fills in --model_name,
--delta_power and --max_length when they are not given explicitly; any of them
can be overridden on the command line.

Supports three privacy modes on the same model/data pipeline:

- non-private   (default; choose modalities with --include_img / --include_txt)
- full DP-SGD   (--use_dp --no-feature_dp)
- Feature-DP    (--use_dp --feature_dp; clips/noises only the private
                 gradient component computed against the sensitive modality)

DP training always uses BOTH modalities (image is treated as public, text as
private) and runs through the modified dp-transformers OpacusDPTrainer
vendored at library/dp-transformers (see MODIFICATIONS.md there).

The dataset is built once and cached on disk (see
medmap/datasets/mimic_cxr/builder.py), so seed / epsilon sweeps skip the
expensive build. See docs/mimic_cxr.md for the full preprocessing pipeline
and example commands.
"""

from scripts._bootstrap import setup_project_paths
REPO_ROOT = setup_project_paths()

import os
import argparse
import json

from medmap.models.mimic_backbones import BACKBONE_PRESETS, DEFAULT_BACKBONE, resolve_backbone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Backbone / path Args
    ap.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, choices=sorted(BACKBONE_PRESETS),
                    help="Which Gemma 3 family backbone to fine-tune. Sets the defaults of "
                         "--model_name, --delta_power and --max_length (see medmap/models/gemma.py).")
    ap.add_argument("--model_name", type=str, default=None,
                    help="Hugging Face model id. Default: the id of --backbone "
                         "(google/medgemma-4b-pt or google/gemma-3-4b-pt).")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--top_level_buckets", type=str, nargs="+", required=True,
                    help="MIMIC shards to train on, e.g. p10 p11 p12.")
    ap.add_argument("--best_img_index_root", type=str, required=True,
                    help="Root of the best-image index written by scripts/select_mimic_images.py "
                         "(<root>/<shard>/<shard>_best_img_final.csv).")
    ap.add_argument("--image_root", type=str, default=None,
                    help="MIMIC-CXR-JPG version directory used to resolve portable relative "
                         "image paths in the best-image index. Required for indexes generated "
                         "by the current selector; legacy absolute-path indexes do not need it.")
    ap.add_argument("--chexpert_csv", type=str, required=True,
                    help="mimic-cxr-2.0.0-chexpert.csv from MIMIC-CXR-JPG.")
    ap.add_argument("--split_csv", type=str, required=True,
                    help="mimic-cxr-2.0.0-split.csv from MIMIC-CXR-JPG (official split).")
    ap.add_argument("--report_root", type=str, required=True,
                    help="MIMIC-CXR report root containing the p10..p19 shard folders with the "
                         "leakage-filtered s<study_id>_mmp.txt files (scripts/extract_mimic_reports.py).")
    ap.add_argument("--dataset_cache_root", type=str, required=True,
                    help="Directory where the built HF dataset splits are cached.")
    ap.add_argument("--dataset_cache_tag", type=str, default="v1")
    ap.add_argument("--rebuild_dataset_cache", action=argparse.BooleanOptionalAction, default=False)

    # HP Args
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=None,
                    help="Tokenizer truncation length (image tokens + text). Default: the backbone "
                         "preset (482 = 256 image soft tokens + delimiters + up to the P99 report "
                         "length of 211 tokens). Use ~226 for text-only runs.")

    # Execution Args
    ap.add_argument("--include_img", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_txt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_dp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--feature_dp", action=argparse.BooleanOptionalAction, default=True,
                    help="With --use_dp: True = Feature-DP, False = full DP-SGD.")
    ap.add_argument("--target_epsilon", type=float, default=4.0)
    ap.add_argument("--delta_power", type=float, default=None,
                    help="Target delta = n_train ** delta_power. Default: the backbone preset "
                         "(-1.1 for gemma, i.e. delta = n^-1.1 as for ABIDE/PAD; -1.0 for medgemma, "
                         "i.e. the dp-transformers default delta = 1/n used in the MedGemma runs).")
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--public_clip_norm", type=float, default=None,
                    help="Per-sample clip norm of the public (image-only) gradient component under "
                         "Feature-DP. Default: the backbone preset (50 for medgemma, 69 for gemma).")

    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=800)
    ap.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction, default=False,
                    help="Evaluate the untrained model once before the first step.")

    ap.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"])
    ap.add_argument("--save_steps", type=int, default=800)
    ap.add_argument("--save_total_limit", type=int, default=None)

    # precision flags
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)

    # gradient checkpointing flag
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    # LoRA
    ap.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.00)

    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--attempt", type=int, required=True)

    # Best checkpoint loading
    ap.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--metric_for_best_model", type=str, default="eval_loss")  # or "macro_auc", "macro_f1", ...
    ap.add_argument("--greater_is_better", action=argparse.BooleanOptionalAction, default=None)

    # W&B (python-only setup; use --wandb_mode disabled to run without W&B)
    ap.add_argument("--wandb_project", type=str, default=None,
                    help="Default: medmap-mimic-<backbone>.")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags", type=str, default="")  # comma-separated
    ap.add_argument("--wandb_watch_model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gpu_log_every_n_steps", type=int, default=1)

    args = ap.parse_args()

    # Resolve backbone-dependent defaults (explicit CLI values always win).
    preset = resolve_backbone(args.backbone)
    if args.model_name is None:
        args.model_name = preset.model_name
    if args.delta_power is None:
        args.delta_power = preset.delta_power
    if args.max_length is None:
        args.max_length = preset.max_length
    if args.public_clip_norm is None:
        args.public_clip_norm = preset.public_clip_norm
    if args.wandb_project is None:
        args.wandb_project = f"medmap-mimic-{preset.key}"
    backbone_tag = preset.key            # used in output directory names
    backbone_label = preset.display_name.replace(" ", "")  # used in W&B run names
    print(f"[Backbone] {preset.key}: {args.model_name} ({preset.display_name}); "
          f"max_length={args.max_length}, delta = n_train ** {args.delta_power}")

    # Import training dependencies only after argument parsing so --help works
    # in a lightweight preprocessing environment.
    from dp_transformers.arguments import PrivacyArguments
    from transformers import AutoProcessor, TrainingArguments
    from transformers.utils import logging as hf_logging

    from medmap.common.utils import (
        _final_cleanup,
        _is_main_process,
        get_formatted_datetime_string,
        seed_everything,
    )
    from medmap.datasets.mimic_cxr.builder import load_or_build_cached_mimic_splits
    from medmap.models.gemma import setup_gemma_model
    from medmap.training.callbacks import (
        EpochAndBestPredictionDumpCallback,
        ForcedEpochEvalCallback,
        MemoryCleanupCallback,
        MirrorValEvalToTestCallback,
        SaveEverythingCallback,
        WandbEvalLoggingCallback,
        WandbGpuStatsCallback,
        WandbInitAndPanelsCallback,
        WandbPerLabelRocCurveCallbackInbuilt,
        WandbTestLoggingCallback,
    )
    from medmap.training.collators import Gemma3HeteroNoDummyCollator
    from medmap.training.metrics import build_compute_metrics
    from medmap.training.trainers import get_trainer

    seed_everything(args.seed)
    use_wandb = args.wandb_mode != "disabled"

    if args.use_dp:
        args.include_img, args.include_txt = True, True  # Force both modalities for DP training
        args.per_device_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if not args.feature_dp:
            print("Using FULL DP")
            output_dir = f"outputs_mimic_cxr_{backbone_tag}_fulldp_eps_{args.target_epsilon}_{args.attempt}"
            wandb_run_name = f"MIMIC-CXR-{backbone_label}-FullDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
        else:
            print("Using FEATURE DP")
            output_dir = f"outputs_mimic_cxr_{backbone_tag}_featdp_eps_{args.target_epsilon}_{args.attempt}"
            wandb_run_name = f"MIMIC-CXR-{backbone_label}-FeatureDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-public-clip-{args.public_clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
    elif args.include_img and args.include_txt:
        print(f"Using both image and text inputs for Non Private {preset.display_name}.")
        output_dir = f"outputs_mimic_cxr_{backbone_tag}_cls_both_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-{backbone_label}-Both-ImgTxtClassification-{get_formatted_datetime_string()}"
    elif args.include_img:
        print(f"Using ONLY image inputs for Non Private {preset.display_name}.")
        output_dir = f"outputs_mimic_cxr_{backbone_tag}_cls_img_only_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-{backbone_label}-ImageOnly-{get_formatted_datetime_string()}"
    elif args.include_txt:
        print(f"Using ONLY text inputs for Non Private {preset.display_name}.")
        output_dir = f"outputs_mimic_cxr_{backbone_tag}_cls_txt_only_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-{backbone_label}-TextOnly-{get_formatted_datetime_string()}"
    else:
        raise ValueError("At least one of --include_img or --include_txt must be set to True.")

    output_dir = os.path.join(args.output_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    hf_logging.set_verbosity_info()

    # W&B env setup in python
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = wandb_run_name
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    # optional: tags (comma-separated)
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags
    # online/offline/disabled
    os.environ["WANDB_MODE"] = args.wandb_mode

    # 2. Data Preparation (built once, cached on disk)
    splits, num_labels, class_names, dataset_cache_dir = load_or_build_cached_mimic_splits(
        top_level_buckets=args.top_level_buckets,
        best_img_index_root=args.best_img_index_root,
        image_root=args.image_root,
        chexpert_csv=args.chexpert_csv,
        split_csv_path=args.split_csv,
        report_root=args.report_root,
        cache_root=args.dataset_cache_root,
        cache_tag=args.dataset_cache_tag,
        rebuild_cache=args.rebuild_dataset_cache,
    )
    train_ds, val_ds, test_ds = splits["train"], splits["validation"], splits["test"]

    print(f"[Split] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[Labels] num_labels={num_labels} labels={class_names}")
    print(f"[Dataset cache] {dataset_cache_dir}")

    # 3. Model & Processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.truncation_side = "right"
        processor.tokenizer.padding_side = "right"

    # 4. Collator & Metrics
    import torch

    collator = Gemma3HeteroNoDummyCollator(
        processor=processor,
        max_length=args.max_length,
        num_labels=num_labels,
        pad_to_multiple_of=8,
        want_image=args.include_img,
        want_text=args.include_txt,
        pixel_values_dtype=torch.bfloat16,  # backbone weights are loaded in bf16
    )

    model = setup_gemma_model(args.model_name, num_labels, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to("cuda")

    # Gradient checkpointing (user-controlled)
    if args.gradient_checkpointing:
        # Trainer will call model.gradient_checkpointing_enable if args.gradient_checkpointing=True,
        # but we also do best-effort enable here for safety.
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    compute_metrics = build_compute_metrics(class_names)

    if args.load_best_model_at_end and args.save_strategy != args.eval_strategy:
        ap.error(
            "--save_strategy and --eval_strategy must match when "
            "--load_best_model_at_end is enabled."
        )
    actual_save_strategy = args.save_strategy
    actual_save_steps = args.save_steps if args.save_strategy == "steps" else None

    greater_is_better = args.greater_is_better
    if greater_is_better is None:
        # heuristic: loss => False, otherwise True
        mib = args.metric_for_best_model
        if mib.endswith("loss") or mib == "loss" or mib == "eval_loss":
            greater_is_better = False
        else:
            greater_is_better = True

    # 5. Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        optim="adamw_torch",
        adam_beta2=0.999,
        adam_epsilon=1e-4,

        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,

        seed=args.seed,
        data_seed=args.seed,

        eval_strategy=args.eval_strategy if val_ds is not None else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        eval_on_start=args.eval_on_start,

        save_strategy=actual_save_strategy,
        save_steps=actual_save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,

        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,

        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=greater_is_better,

        report_to=["wandb"] if use_wandb else [],
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
    )

    # Callbacks. The W&B family is only attached when W&B is enabled.
    callbacks = []
    if use_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        callbacks.append(WandbInitAndPanelsCallback(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=wandb_run_name,
            mode=args.wandb_mode,
            tags=tags,
            watch_model=args.wandb_watch_model,
        ))
        callbacks.append(WandbEvalLoggingCallback(eval_strategy=args.eval_strategy))
        callbacks.append(WandbTestLoggingCallback())
        callbacks.append(WandbGpuStatsCallback(log_every_n_steps=args.gpu_log_every_n_steps))
    callbacks.append(MemoryCleanupCallback(every_n_steps=50, sync_cuda=True))
    cb_force_epoch_eval = ForcedEpochEvalCallback(eval_dataset=val_ds)
    callbacks.append(cb_force_epoch_eval)
    cb_mirror_test = MirrorValEvalToTestCallback(test_dataset=test_ds, eval_strategy=args.eval_strategy)
    callbacks.append(cb_mirror_test)
    callbacks.append(SaveEverythingCallback(processor=processor, class_names=class_names))
    cb_roc_per_label = None
    if use_wandb:
        cb_roc_per_label = WandbPerLabelRocCurveCallbackInbuilt(
            eval_dataset=val_ds,
            class_names=class_names,
            eval_strategy=args.eval_strategy,
            key_prefix="val/roc",
            macro_grid_size=400,
            log_all_in_one_panel=True,
            log_individual_per_label=True,   # avoids 12+ separate panels
            log_separate_micro_macro=True,   # keeps ONLY val/roc/all
        )
        callbacks.append(cb_roc_per_label)
    cb_dump = EpochAndBestPredictionDumpCallback(
        class_names=class_names,
        val_dataset=val_ds,
        test_dataset=test_ds,
        threshold=0.5,
        dump_val=True,
        dump_test=True,      # set False if too expensive
        every_n_epochs=1,
        save_logits=False,
        include_image_path=False,
        include_text_len=True,
    )
    callbacks.append(cb_dump)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    if args.use_dp:
        # delta = n_train ** delta_power (gemma: -1.1, i.e. n^-1.1 as for ABIDE/PAD;
        # medgemma: -1.0, i.e. the dp-transformers default 1/n used in those runs).
        target_delta = float(len(train_ds) ** args.delta_power)
        print(f"[Privacy] target_delta = {len(train_ds)} ** {args.delta_power} = {target_delta:.3e}; "
              f"clip_norm={args.clip_norm}, public_clip_norm={args.public_clip_norm}")
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=args.clip_norm/args.per_device_train_batch_size,
            public_clip=args.public_clip_norm/args.per_device_train_batch_size,
            target_epsilon=args.target_epsilon,
            target_delta=target_delta,
            feature_dp=args.feature_dp
        )

    trainer = get_trainer(
        use_dp=args.use_dp,
        loss_mode="decomposed" if args.use_dp else "standard",
        **trainer_kwargs,
    )
    trainer.model_accepts_loss_kwargs = False

    if cb_roc_per_label is not None:
        cb_roc_per_label.attach_trainer(trainer)
    cb_dump.attach_trainer(trainer)
    cb_force_epoch_eval.attach_trainer(trainer)
    cb_mirror_test.attach_trainer(trainer)

    # Save label metadata and the resolved run configuration at root as well
    with open(os.path.join(output_dir, "labels_metadata.json"), "w") as f:
        json.dump({"num_labels": num_labels, "class_names": class_names}, f, indent=2)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"backbone": preset.key, "backbone_display_name": preset.display_name, **vars(args)},
                  f, indent=2, default=str)
    try:
        processor.save_pretrained(output_dir)
    except Exception:
        pass

    # 6. Execution (resume supported)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model + processor
    trainer.save_model(output_dir)
    metrics = train_result.metrics
    metrics["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    try:
        processor.save_pretrained(output_dir)
    except Exception:
        pass

    # Final test evaluation (logged under test/* with epoch axis)
    if test_ds is not None:
        print("[Eval] Running final test evaluation...")
        metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(metrics)
        if use_wandb and _is_main_process():
            try:
                import wandb
                if wandb.run is not None:
                    payload = {"trainer/epoch": float(trainer.state.epoch or float("nan"))}
                    for k, v in metrics.items():
                        if k.startswith("test_"):
                            payload["test/" + k[len("test_"):]] = v
                    wandb.log(payload)
            except Exception:
                pass

    # Final cleanup
    _final_cleanup(trainer=trainer, model=model, backbone=model.backbone)


if __name__ == "__main__":
    main()
