#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export USE_DP=1
export SELECTIVE_DP=0  # Set to 1 for Feature DP
export TARGET_EPS=1.0
export CLIP_NORM=1.0
export BATCH=1024
export GRAD_ACCUM=64
export LR=5e-4

python -m scripts.train