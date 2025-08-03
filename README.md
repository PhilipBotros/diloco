# DiLoCo - Distributed Local Communication

A simple PyTorch implementation of [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105). The paper proposes a two-level optimization approach where workers perform local training steps before communicating parameter updates, reducing communication overhead in distributed training.

## Features

- **Distributed Data Parallel (DDP)** training with PyTorch
- **Two-level optimization**: Inner local steps + outer global synchronization
- **Communication-efficient**: Reduces network overhead by batching updates
- **MNIST training example** with configurable hyperparameters
- **Torchrun integration** for easy multi-GPU deployment

## Quick Start

### Prerequisites

- Python 3.12+
- PyTorch 2.7.1+
- CUDA-capable GPUs (for multi-GPU training)

### Installation

```bash
# Install dependencies (using uv)
uv sync
```

### Running Training

#### Single Command Launch
```bash
# Auto-detect available GPUs
./scripts/run_ddp.sh

# Specify number of GPUs
./scripts/run_ddp.sh 4
```

#### Manual Launch
```bash
# Example: 2 GPUs
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 src/main_ddp.py
```

## Architecture

The implementation uses a two-level optimization strategy:

1. **Inner Loop**: Each worker performs local SGD steps on its data shard
2. **Outer Loop**: Workers communicate parameter deltas and apply averaged updates

## Todo
- [ x ] [Async DiLoCo](https://arxiv.org/abs/2401.09135)
- [ ] [SWARM](https://arxiv.org/abs/2301.11913)
- [ ] Support generic remote parameter servers and connect based on run ID
- [ ] Can we filter out bad actors during training?
- [ ] Basic interface to display the current connected nodes for a particular run