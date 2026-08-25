#!/usr/bin/env python3
"""Fine-tune MedGemma on MIMIC-CXR multi-label classification.

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

from transformers import AutoProcessor, TrainingArguments
from transformers.utils import logging as hf_logging
from dp_transformers.arguments import PrivacyArguments

from medmap.common.utils import seed_everything, get_formatted_datetime_string, _final_cleanup, _is_main_process
from medmap.datasets.mimic_cxr.builder import load_or_build_cached_mimic_splits
from medmap.training.collators import Gemma3HeteroNoDummyCollator
from medmap.training.metrics import build_compute_metrics
from medmap.models.medgemma import setup_medgemma_model
from medmap.training.trainers import get_trainer
from medmap.training.callbacks import (
    WandbInitAndPanelsCallback, WandbEvalLoggingCallback, WandbGpuStatsCallback,
    ForcedEpochEvalCallback, MirrorValEvalToTestCallback, WandbTestLoggingCallback,
    SaveEverythingCallback, MemoryCleanupCallback, WandbPerLabelRocCurveCallbackInbuilt,
    EpochAndBestPredictionDumpCallback
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Path Args
    ap.add_argument("--model_name", type=str, default="google/medgemma-4b-pt")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--top_level_buckets", type=str, nargs="+", required=True,
                    help="MIMIC shards to train on, e.g. p10 p11 p12.")
    ap.add_argument("--best_img_index_root", type=str, required=True,
                    help="Root of the best-image index written by scripts/select_mimic_images.py "
                         "(<root>/<shard>/<shard>_best_img_final.csv).")
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
    ap.add_argument("--max_length", type=int, default=426)

    # Execution Args
    ap.add_argument("--include_img", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_txt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_dp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--feature_dp", action=argparse.BooleanOptionalAction, default=True,
                    help="With --use_dp: True = Feature-DP, False = full DP-SGD.")
    ap.add_argument("--target_epsilon", type=float, default=4.0)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--public_clip_norm", type=float, default=50.0)

    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=800)

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
    ap.add_argument("--wandb_project", type=str, default="medmap-mimic-medgemma")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags", type=str, default="")  # comma-separated
    ap.add_argument("--wandb_watch_model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gpu_log_every_n_steps", type=int, default=1)

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)
    use_wandb = args.wandb_mode != "disabled"

    if args.use_dp:
        args.include_img, args.include_txt = True, True  # Force both modalities for DP training
        args.per_device_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if not args.feature_dp:
            print("Using FULL DP")
            output_dir = f"outputs_mimic_cxr_medgemma_fulldp_eps_{args.target_epsilon}_{args.attempt}"
            wandb_run_name = f"MIMIC-CXR-Medgemma-FullDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
        else:
            print("Using FEATURE DP")
            output_dir = f"outputs_mimic_cxr_medgemma_featdp_eps_{args.target_epsilon}_{args.attempt}"
            wandb_run_name = f"MIMIC-CXR-Medgemma-FeatureDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-public-clip-{args.public_clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
    elif args.include_img and args.include_txt:
        print("Using both image and text inputs for Non Private Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_both_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-Both-ImgTxtClassification-{get_formatted_datetime_string()}"
    elif args.include_img:
        print("Using ONLY image inputs for Non Private Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_img_only_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-ImageOnly-{get_formatted_datetime_string()}"
    elif args.include_txt:
        print("Using ONLY text inputs for Non Private Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_txt_only_{args.attempt}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-TextOnly-{get_formatted_datetime_string()}"
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
    collator = Gemma3HeteroNoDummyCollator(
        processor=processor,
        max_length=args.max_length,
        num_labels=num_labels,
        pad_to_multiple_of=8,
        want_image=args.include_img,
        want_text=args.include_txt,
    )

    model = setup_medgemma_model(args.model_name, num_labels, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to("cuda")

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

    if args.eval_strategy == "steps":
        actual_save_strategy = "steps"
        actual_save_steps = args.save_steps
    else:
        actual_save_strategy = "epoch"
        actual_save_steps = None

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
        eval_on_start=False,

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
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=args.clip_norm/args.per_device_train_batch_size,
            public_clip=args.public_clip_norm/args.per_device_train_batch_size,
            target_epsilon=args.target_epsilon,
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

    # Save label metadata at root as well
    with open(os.path.join(output_dir, "labels_metadata.json"), "w") as f:
        json.dump({"num_labels": num_labels, "class_names": class_names}, f, indent=2)
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
