import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from src.model import create_model

INNER_STEPS = 10    
BATCH_SIZE = 64
NUM_EPOCHS = 10 

def setup():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    dist.init_process_group(backend="nccl")
    
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def average_gradients(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()


def outer_step(global_model, local_model, outer_optimizer):
    """Compute deltas from local to global, average them, apply outer optimizer to global model."""
    with torch.no_grad():
        # Collect all deltas into a single flat tensor
        deltas = []
        for (g_param, l_param) in zip(global_model.parameters(), local_model.parameters()):
            delta = g_param.data - l_param.data  # local step delta
            deltas.append(delta.view(-1))
        
        # Concatenate all deltas and perform single all_reduce
        flat_deltas = torch.cat(deltas)
        dist.all_reduce(flat_deltas, op=dist.ReduceOp.SUM)
        flat_deltas /= dist.get_world_size()
        
        # Reshape and assign back to gradients
        offset = 0
        for g_param in global_model.parameters():
            numel = g_param.numel()
            g_param.grad = flat_deltas[offset:offset+numel].view(g_param.shape)
            offset += numel

    outer_optimizer.step()
    outer_optimizer.zero_grad()


def train():
    # Setup distributed training
    rank, world_size, local_rank = setup()
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)

    global_model = create_model().cuda(local_rank)
    local_model = create_model().cuda(local_rank)

    inner_optimizer = optim.AdamW(local_model.parameters(), lr=0.001)
    outer_optimizer = optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    global_model.train()
    local_model.train()
    # TODO: Outer steps not the correct name here, we need to calculate num steps wrt epochs and batch size
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(local_rank), y.cuda(local_rank)

            # Step 1: sync local model with current global state
            local_model.load_state_dict(global_model.state_dict())

            # Step 2: inner training loop
            for _ in range(INNER_STEPS):
                pred = local_model(x)
                loss = loss_fn(pred, y)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # Step 3: outer update
            outer_step(global_model, local_model, outer_optimizer)

            if rank == 0 and (i + 1) % 100 == 0:
                print(f"[Epoch {epoch}] Step {i+1}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    train()