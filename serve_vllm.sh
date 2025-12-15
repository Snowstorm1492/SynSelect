#!/bin/bash
INDEX=$1
CKPT_PATH=$2
MODEL_NAME=$3
DEVICE=$INDEX
PORT=$((8000 + $DEVICE * 100))

export CUDA_VISIBLE_DEVICES=$DEVICE
export VLLM_USE_MODELSCOPE=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve $CKPT_PATH \
    --served_model_name $MODEL_NAME_$INDEX \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 12288 \
    --max-num-seqs 128 \
    --block-size 64 \
    --disable-log-requests \
    --trust-remote-code







