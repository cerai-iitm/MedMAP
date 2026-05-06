import os
import gc
import torch
import wandb
from transformers import AutoProcessor, TrainingArguments
from transformers.models.smolvlm import SmolVLMConfig

from datasets import Dataset, Features, Value, ClassLabel, Image as DatasetsImage

from scripts._bootstrap import setup_project_paths
REPO_ROOT = setup_project_paths()

from medmap.utils.core import env_str, env_int, env_float, env_bool, seed_everything, get_formatted_datetime_string
from medmap.data.dataset import build_dataset
from medmap.data.collators import SmolVLMHeteroNoDummyCollator
from medmap.data.metrics import compute_metrics_fn
from medmap.models.smolvlm import setup_model
from medmap.training.trainers import get_trainer
from medmap.training.callbacks import WandbGpuCpuMemoryCallback, SaveManagerCallback, MemoryCleanupCallback, SplitEvalCallback

from dp_transformers.arguments import PrivacyArguments


def main():
    # Environment Setup
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8")
    
    MODEL_NAME = env_str("MODEL_NAME", "HuggingFaceTB/SmolVLM-256M-Base")
    SEED = env_int("SEED", 100)
    CAPTIONS_JSON = env_str("CAPTIONS_JSON", "/data/common/data/imagenet_captions.json")
    ROOT_ZIP_PATH = env_str("ROOT_ZIP_PATH", "/data/common/data/imagenet_object_localization_challenge")
    TRAIN_TXT = env_str("TRAIN_TXT", "ILSVRC/ImageSets/CLS-LOC/train_cls.txt")
    
    EPOCHS = env_int("EPOCHS", 10) 
    PER_DEVICE_BATCH = env_int("BATCH", 32)
    GRAD_ACCUM = env_int("GRAD_ACCUM", 1)
    LR = env_float("LR", 5e-5) 
    WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.01)
    WARMUP_RATIO = env_float("WARMUP_RATIO", 0.03)
    MAX_LENGTH = env_int("MAX_LENGTH", 128)
    
    LOGGING_STEPS = env_int("LOGGING_STEPS", 1)
    LOGGING_STRATEGY = env_str("LOGGING_STRATEGY", "steps")
    SAVE_STRATEGY = env_str("SAVE_STRATEGY", "steps")
    SAVE_STEPS = env_int("SAVE_STEPS", 700)  # save checkpoint every N optimizer steps
    EVAL_STRATEGY = env_str("EVAL_STRATEGY", "steps")
    EVAL_STEPS = env_int("EVAL_STEPS", 700)  # evaluate on validation every N optimizer steps
    
    USE_DP = env_bool("USE_DP", False)
    SELECTIVE_DP = env_bool("SELECTIVE_DP", False)
    TARGET_EPS = env_float("TARGET_EPS", 1.0)
    CLIP_NORM = env_float("CLIP_NORM", 1.0)

    GRAD_CHECKPOINT = env_bool("GRAD_CHECKPOINT", False)
    USE_LORA = env_bool("USE_LORA", True)
    LORA_R = env_int("LORA_R", 16)
    LORA_ALPHA = env_int("LORA_ALPHA", 32)
    LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)

    INCLUDE_IMG = env_bool("INCLUDE_IMG", True)
    INCLUDE_TEXT = env_bool("INCLUDE_TEXT", True)
    
    if INCLUDE_IMG and INCLUDE_TEXT:
        OUT_DIR = env_str("OUT_DIR", f"outputs_smolvlm_cls_img_txt_{get_formatted_datetime_string()}")
        RUN_NAME = env_str("RUN_NAME", f"SmolVLM-Both-ImageTextClassification-{get_formatted_datetime_string()}")
    elif INCLUDE_IMG:
        OUT_DIR = env_str("OUT_DIR", f"outputs_smolvlm_cls_img_only_{get_formatted_datetime_string()}")
        RUN_NAME = env_str("RUN_NAME", f"SmolVLM-ImageOnly-{get_formatted_datetime_string()}")
    elif INCLUDE_TEXT:
        OUT_DIR = env_str("OUT_DIR", f"outputs_smolvlm_cls_txt_only_{get_formatted_datetime_string()}")
        RUN_NAME = env_str("RUN_NAME", f"SmolVLM-TextOnly-{get_formatted_datetime_string()}")

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Processor & Config
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.truncation_side = "right"
        if "<image>" not in processor.tokenizer.added_tokens_encoder:
            processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    config = SmolVLMConfig.from_pretrained(MODEL_NAME, attn_implementation="eager")
    config.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    # Dataset
    full_ds, num_classes, class_names = build_dataset(CAPTIONS_JSON, ROOT_ZIP_PATH, TRAIN_TXT, use_tiny=True)
    full_ds = full_ds.cast_column("image_path", DatasetsImage(decode=True))
    split_1 = full_ds.train_test_split(test_size=0.40, seed=SEED, stratify_by_column="class_idx")
    train_ds = split_1["train"]
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=SEED, stratify_by_column="class_idx")
    val_ds, test_ds = split_2["train"], split_2["test"]

    if GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Model
    model = setup_model(MODEL_NAME, config, num_classes, USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT).to(device)
    if hasattr(processor, "tokenizer"):
        model.resize_token_embeddings(len(processor.tokenizer))

    # Training Setup
    data_collator = SmolVLMHeteroNoDummyCollator(
        processor=processor, 
        text_key="prompt", 
        image_key="image_path",
        max_length=MAX_LENGTH, 
        want_image=INCLUDE_IMG, 
        want_text=INCLUDE_TEXT
    )

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
        adam_epsilon=1e-4,
        logging_first_step=True,
        logging_strategy=LOGGING_STRATEGY,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        eval_on_start=False,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        include_inputs_for_metrics=False,
        report_to=["wandb"],
        gradient_checkpointing=GRAD_CHECKPOINT,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        skip_memory_metrics=False,
        save_safetensors=True,
        data_seed=SEED,
        run_name=RUN_NAME,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "tokenizer": processor.tokenizer if hasattr(processor, "tokenizer") else None,
        "compute_metrics": compute_metrics_fn,
        "callbacks": [
            WandbGpuCpuMemoryCallback(),
            MemoryCleanupCallback(every_n_steps=20),
            SaveManagerCallback(processor=processor, class_names=class_names, base_model_name=MODEL_NAME, use_lora=USE_LORA)
        ]
    }

    if USE_DP:
        trainer_kwargs["privacy_args"] = PrivacyArguments(
            target_epsilon=TARGET_EPS, 
            per_sample_max_grad_norm=CLIP_NORM/PER_DEVICE_BATCH,
            selective_dp=SELECTIVE_DP
        )

    trainer = get_trainer(use_dp=USE_DP, **trainer_kwargs)
    trainer.model_accepts_loss_kwargs = False # force HF Trainer to divide loss by grad_acc steps

    split_cb = SplitEvalCallback(
        train_ds=train_ds, 
        val_ds=val_ds, 
        test_ds=test_ds
    )
    trainer.add_callback(split_cb)
    split_cb.attach_trainer(trainer)

    # Execution
    train_result = trainer.train()
    trainer.save_model(OUT_DIR)
    metrics= train_result.metrics
    metrics['num_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    if wandb.run: wandb.finish()
    
if __name__ == "__main__":
    main()