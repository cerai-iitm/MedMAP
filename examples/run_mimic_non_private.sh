#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python -m scripts.train_mimic \
  --output_root /storage/karthick/data/MedGemma \
  --top_level_buckets p10 p11 p12 \
  --xrv_root /data/kaushik/MedGemma/mimic/xrv_all_vs_mimic_ch_model_comparison \
  --chexpert_csv /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv \
  --split_csv /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv \
  --report_root /storage/kaushik/datasets/mimic_cxr/physionet.org/files/mimic-cxr/2.1.0/files/ \
  --no-use_dp \
  --include_img \
  --include_txt \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0