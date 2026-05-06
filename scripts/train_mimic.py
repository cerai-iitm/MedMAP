import os
import argparse
import torch
import json
from transformers import AutoProcessor, TrainingArguments
from transformers.utils import logging as hf_logging
from dp_transformers.arguments import PrivacyArguments

from medmap.utils.core import seed_everything, get_formatted_datetime_string, _final_cleanup, _is_main_process
from medmap.data.dataset import build_mimic_chexpert_dataset, split_mimic_dataset_official_by_ids
from medmap.data.collators import Gemma3HeteroNoDummyCollator
from medmap.data.metrics import build_compute_metrics
from medmap.models.medgemma import setup_medgemma_model
from medmap.training.trainers import get_trainer
from medmap.training.callbacks import (
    WandbInitAndPanelsCallback, WandbEvalLoggingCallback, WandbGpuStatsCallback,
    ForcedEpochEvalCallback, MirrorValEvalToTestCallback, WandbTestLoggingCallback,
    SaveEverythingCallback, MemoryCleanupCallback, WandbPerLabelRocCurveCallbackInbuilt,
    EpochAndBestPredictionDumpCallback
)

def main():
    ap = argparse.ArgumentParser()
    # Path Args
    ap.add_argument("--model_name", type=str, default="google/medgemma-4b-pt")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--top_level_buckets", type=str, nargs="+", required=True)
    ap.add_argument("--xrv_root", type=str, required=True)
    ap.add_argument("--chexpert_csv", type=str, required=True)
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--report_root", type=str, required=True)
    
    # HP Args
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1024)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=426) 
    
    # Execution Args
    ap.add_argument("--include_img", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_txt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_dp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--selective_dp", action=argparse.BooleanOptionalAction, default=True)
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
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # gradient checkpointing flag
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    # LoRA
    ap.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.00)

    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--itr", type=int, required=True)

    # Best checkpoint loading
    ap.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--metric_for_best_model", type=str, default="eval_loss")  # or "macro_auc", "macro_f1", ...
    ap.add_argument("--greater_is_better", action=argparse.BooleanOptionalAction, default=None)

    # W&B (python-only setup)
    ap.add_argument("--wandb_project", type=str, default="medgemma-mimic-multilabel")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags", type=str, default="")  # comma-separated
    ap.add_argument("--wandb_watch_model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gpu_log_every_n_steps", type=int, default=1)

    
    args, _ = ap.parse_known_args()
    seed_everything(args.seed)

    if args.use_dp:
        args.include_img, args.include_txt = True, True  # Force both modalities for DP training
        args.per_device_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if not args.selective_dp:
            output_dir = f"outputs_mimic_cxr_medgemma_fulldp_eps_{args.target_epsilon}"
            wandb_run_name = f"MIMIC-CXR-Medgemma-FullDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
        else:
            output_dir = f"outputs_mimic_cxr_medgemma_featdp_eps_{args.target_epsilon}"
            wandb_run_name = f"MIMIC-CXR-Medgemma-FeatureDP-eps-{args.target_epsilon}-clip-{args.clip_norm}-public-clip-{args.public_clip_norm}-lr-{args.learning_rate}-seed-{args.seed}-{get_formatted_datetime_string()}"
    elif args.include_img and args.include_txt:
        print("Using both image and text inputs for Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_both_{args.itr}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-Both-ImgTxtClassification-{get_formatted_datetime_string()}"
    elif args.include_img:
        print("Using ONLY image inputs for Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_img_only_{args.itr}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-ImageOnly-{get_formatted_datetime_string()}"
    elif args.include_txt:
        print("Using ONLY text inputs for Gemma.")
        output_dir = f"outputs_mimic_cxr_medgemma_cls_txt_only_{args.itr}"
        wandb_run_name = f"MIMIC-CXR-Medgemma-TextOnly-{get_formatted_datetime_string()}"
    else:
        raise ValueError("At least one of --include_img or --include_txt must be set to True.")


        
    output_dir = os.path.join(args.output_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    os.environ["WANDB_PROJECT"] = "medmap-mimic-medgemma"
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

    # 2. Data Preparation
    ds, num_labels, class_names = build_mimic_chexpert_dataset(args.top_level_buckets, args.xrv_root, args.chexpert_csv, args.report_root, drop_all_nan_rows=True)
    splits = split_mimic_dataset_official_by_ids(
        ds=ds, 
        split_csv_path=args.split_csv,
        strict=True,
        keep_ids=True
    )
    train_ds, val_ds, test_ds = splits["train"], splits["validation"], splits["test"]

    print(f"[Split] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[Labels] num_labels={num_labels} labels={class_names}")
    print("Train:", train_ds)
    print("Val:", val_ds)
    print("Test:", test_ds)

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
    
    model, backbone = setup_medgemma_model(args.model_name, num_labels, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to("cuda")

    # 8) Gradient checkpointing (user-controlled)
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

        seed=args.seed
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
        greater_is_better=args.greater_is_better,

        report_to="wandb", # Python callbacks handle W&B
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    cb_wandb = WandbInitAndPanelsCallback(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        mode=args.wandb_mode,
        tags=tags,
        watch_model=args.wandb_watch_model,
    )
    cb_eval = WandbEvalLoggingCallback(eval_strategy=args.eval_strategy)
    cb_gpu = WandbGpuStatsCallback(log_every_n_steps=args.gpu_log_every_n_steps)
    cb_force_epoch_eval = ForcedEpochEvalCallback(eval_dataset=val_ds)
    cb_mirror_test = MirrorValEvalToTestCallback(test_dataset=test_ds, eval_strategy=args.eval_strategy)
    cb_testlog = WandbTestLoggingCallback()
    cb_save_all = SaveEverythingCallback(processor=processor, class_names=class_names)
    cb_mem = MemoryCleanupCallback(every_n_steps=50, sync_cuda=True)
    # cb_roc = WandbRocCurvesCallback(
    #     eval_dataset=val_ds,
    #     class_names=class_names,
    #     eval_strategy=args.eval_strategy,
    #     max_labels=None,  # or e.g. 8 if you want fewer curves
    # )
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

    trainer_kwargs = {
        "model": model, 
        "args": training_args, 
        "train_dataset": train_ds, 
        "eval_dataset": val_ds,
        "data_collator": collator, 
        "compute_metrics": compute_metrics,
        "callbacks": [
            cb_wandb,
            cb_eval,
            cb_testlog,
            cb_gpu,
            cb_mem,
            cb_force_epoch_eval,
            cb_mirror_test,
            cb_save_all,
            # cb_roc,qA
            cb_roc_per_label,
            cb_dump,
        ]
    }

    cb_roc_per_label.attach_trainer(trainer)
    cb_dump.attach_trainer(trainer)
    cb_force_epoch_eval.attach_trainer(trainer)
    cb_mirror_test.attach_trainer(trainer)

    if args.use_dp:
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=args.clip_norm/args.per_device_train_batch_size,
            public_clip=args.public_clip_norm/args.per_device_train_batch_size,
            target_epsilon=args.target_epsilon, selective_dp=args.selective_dp
        )

    trainer = get_trainer(use_dp=args.use_dp, **trainer_kwargs)
    trainer.model_accepts_loss_kwargs = False
    
    # Save label metadata at root as well
    with open(os.path.join(output_dir, "labels_metadata.json"), "w") as f:
        json.dump({"num_labels": num_labels, "class_names": class_names}, f, indent=2)
    try:
        processor.save_pretrained(output_dir)
    except Exception:
        pass

    # 6. Execution
    # 12) Train (resume supported)
    train_result =trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 13) Save final model + processor
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

    # 14) Final test evaluation (logged under test/* with epoch axis)
    if test_ds is not None:
        print("[Eval] Running final test evaluation...")
        metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print(metrics)
        if _is_main_process():
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

    # 15) Final cleanup
    _final_cleanup(trainer=trainer, model=model, backbone=backbone)

if __name__ == "__main__":
    main()