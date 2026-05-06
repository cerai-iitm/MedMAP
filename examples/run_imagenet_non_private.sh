#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export USE_DP=0
export INCLUDE_IMG=1
export INCLUDE_TEXT=1
export MODEL_NAME="HuggingFaceTB/SmolVLM-256M-Base"
export BATCH=32
export GRAD_ACCUM=1
export LR=5e-5

python -m scripts.train