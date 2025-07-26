#!/bin/bash

# Launch script for main_ddp.py using torchrun
# Usage: ./run_ddp.sh [num_gpus]

NUM_GPUS=${1:-auto}  # Auto-detect GPUs if not specified

echo "Launching DDP training with $NUM_GPUS GPUs..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    src/main_ddp.py