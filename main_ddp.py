import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

SYNC_INTERVAL = 5

def setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def average_model(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

def train(rank, world_size):
    torch.cuda.set_device(rank)
    setup()

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).cuda(rank)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)

    model.train()
    for epoch in range(5):
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(rank), y.cuda(rank)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every SYNC_INTERVAL steps, average weights
            if (i + 1) % SYNC_INTERVAL == 0:
                average_model(model)

            if rank == 0 and (i + 1) % 100 == 0:
                print(f"[Epoch {epoch}] Step {i+1}, Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)