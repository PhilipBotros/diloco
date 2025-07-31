#!/bin/bash

# Launch script for main_rpc.py using torchRPC
# Usage: ./run_rpc.sh [world_size]

WORLD_SIZE=${1:-3}  # Default to 3 processes (1 PS + 2 workers)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching DiLoCo RPC training with world size $WORLD_SIZE..."

# Launch parameter server in background
WORLD_SIZE=$WORLD_SIZE RANK=0 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT python src/main_rpc.py &
PS_PID=$!

# Wait a moment for PS to initialize
sleep 2

# Launch workers in background
for ((rank=1; rank<$WORLD_SIZE; rank++)); do
    WORLD_SIZE=$WORLD_SIZE RANK=$rank MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT python src/main_rpc.py &
    WORKER_PIDS[$rank]=$!
done

# Wait for all processes to complete
wait $PS_PID
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
done

echo "Training completed."