#!/bin/bash

# Multi-node launch script for main_rpc.py using torchRPC
# Usage: ./run_rpc_multinode.sh [world_size] [node_rank] [procs_per_node] [master_addr]
# 
# Example for your setup:
#   Node 0 (master): ./run_rpc_multinode.sh 5 0 3 10.164.0.2
#   Node 1 (worker): ./run_rpc_multinode.sh 5 1 2 10.164.0.2

WORLD_SIZE=${1:-5}      # Default to 5 processes total (1 PS + 4 workers)
NODE_RANK=${2:-0}       # Current node rank (0 for master node)
PROCS_PER_NODE=${3:-2}  # Number of processes to run on this node
MASTER_ADDR=${4:-localhost}  # Master node address
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching DiLoCo RPC training on node $NODE_RANK..."
echo "Configuration:"
echo "  World size: $WORLD_SIZE"
echo "  Node rank: $NODE_RANK"
echo "  Processes on this node: $PROCS_PER_NODE"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"

# Calculate rank offset for this node based on node rank
if [ $NODE_RANK -eq 0 ]; then
    RANK_OFFSET=0  # Node 0 starts with rank 0 (parameter server)
else
    RANK_OFFSET=2  # Node 1+ starts after node 0's processes
fi

# Array to store process IDs
PIDS=()

# Launch processes for this node
for ((local_rank=0; local_rank<$PROCS_PER_NODE; local_rank++)); do
    GLOBAL_RANK=$((RANK_OFFSET + local_rank))
    
    # Skip if this would exceed world size
    if [ $GLOBAL_RANK -ge $WORLD_SIZE ]; then
        echo "Skipping rank $GLOBAL_RANK as it exceeds world size $WORLD_SIZE"
        break
    fi
    
    if [ $GLOBAL_RANK -eq 0 ]; then
        echo "Starting Parameter Server (rank $GLOBAL_RANK) on node $NODE_RANK..."
    else
        echo "Starting Worker (rank $GLOBAL_RANK, local rank $local_rank) on node $NODE_RANK..."
    fi
    
    WORLD_SIZE=$WORLD_SIZE \
    RANK=$GLOBAL_RANK \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    python src/main_rpc.py &
    
    PIDS[$local_rank]=$!
    
    # Add a small delay between launches to avoid race conditions
    sleep 2
done

echo "All processes launched on node $NODE_RANK. Waiting for completion..."

# Wait for all processes to complete
for pid in "${PIDS[@]}"; do
    if [ -n "$pid" ]; then
        wait $pid
        echo "Process $pid completed"
    fi
done

echo "Training completed on node $NODE_RANK."