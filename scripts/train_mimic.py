import os
import argparse
import torch
import json
from transformers import AutoProcessor, TrainingArguments
from dp_transformers.arguments import PrivacyArguments

from MedMAP.utils.core import seed_everything, env_str, env_bool, get_formatted_datetime_string
from MedMAP.data.dataset import build_mimic_chexpert_dataset, split_mimic_dataset_official_by_ids
from MedMAP.data.collators import Gemma3HeteroNoDummyCollator
from MedMAP.data.metrics import build_compute_metrics
from MedMAP.models.medgemma import setup_medgemma_model
from MedMAP.training.trainers import get_trainer
from MedMAP.training.callbacks import (
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
    ap.add_argument("--top_level_buckets", type=str, nargs="+", required=True)
    ap.add_argument("--xrv_root", type=str, required=True)
    ap.add_argument("--chexpert_csv", type=str, required=True)
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--report_root", type=str, required=True)
    
    # Execution Args
    ap.add_argument("--include_img", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_txt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_dp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--selective_dp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target_epsilon", type=float, default=4.0)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--public_clip_norm", type=float, default=50.0)
    
    # HP Args
    ap.add_argument("--per_device_train_batch_size", type=int, default=1024)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=2e-3)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    
    args, _ = ap.parse_known_args()
    seed_everything(42)

    # 1. Output Dir & W&B Naming
    if args.use_dp:
        out_name = f"DP_MedGemma_eps_{args.target_epsilon}_LR_{args.learning_rate}"
    else:
        out_name = f"Standard_MedGemma_LR_{args.learning_rate}"
        
    output_dir = os.path.join(args.output_root, out_name)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "medmap-mimic-medgemma"

    # 2. Data Preparation
    ds, num_labels, class_names = build_mimic_chexpert_dataset(args.top_level_buckets, args.xrv_root, args.chexpert_csv, args.report_root)
    splits = split_mimic_dataset_official_by_ids(ds, args.split_csv)
    train_ds, val_ds, test_ds = splits["train"], splits["validation"], splits["test"]

    # 3. Model & Processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "right"
    model = setup_medgemma_model(args.model_name, num_labels, use_lora=True, r=16, alpha=32, dropout=0.0).to("cuda")

    # 4. Collator & Metrics
    collator = Gemma3HeteroNoDummyCollator(processor, num_labels=num_labels, want_image=args.include_img, want_text=args.include_txt)
    compute_metrics = build_compute_metrics(class_names)

    # 5. Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none", # Python callbacks handle W&B
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    trainer_kwargs = {
        "model": model, "args": training_args, "train_dataset": train_ds, "eval_dataset": val_ds,
        "data_collator": collator, "compute_metrics": compute_metrics,
        "callbacks": [
            WandbInitAndPanelsCallback(project="medmap-mimic", entity=None, run_name=out_name, mode="online"),
            WandbEvalLoggingCallback(eval_strategy="epoch"), WandbTestLoggingCallback(),
            WandbGpuStatsCallback(), MemoryCleanupCallback(), ForcedEpochEvalCallback(val_ds),
            MirrorValEvalToTestCallback(test_ds, "epoch"), SaveEverythingCallback(processor, class_names),
            WandbPerLabelRocCurveCallbackInbuilt(val_ds, class_names),
            EpochAndBestPredictionDumpCallback(class_names, val_ds, test_ds)
        ]
    }

    if args.use_dp:
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            per_sample_max_grad_norm=args.clip_norm/args.per_device_train_batch_size,
            public_clip=args.public_clip_norm/args.per_device_train_batch_size,
            target_epsilon=args.target_epsilon, selective_dp=args.selective_dp
        )

    trainer = get_trainer(use_dp=args.use_dp, **trainer_kwargs)
    trainer.model_accepts_loss_kwargs = False
    
    # Bind external evaluators
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, 'attach_trainer'): cb.attach_trainer(trainer)

    # 6. Execution
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()