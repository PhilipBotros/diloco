import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.rpc import rpc, rpc_sync
import torchvision
import torchvision.transforms as transforms
from threading import Lock

SYNC_INTERVAL = 5

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

class ParameterServer:
    def __init__(self):
        self.lock = Lock()
        self.model = create_model().cpu()
        self.buffer = []
        self.world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude server

    def push(self, state_dict):
        with self.lock:
            self.buffer.append(state_dict)
            if len(self.buffer) == self.world_size:
                self._average_models()
                self.buffer = []

    def pull(self):
        with self.lock:
            return self.model.state_dict()

    def _average_models(self):
        avg_state = {}
        for k in self.buffer[0].keys():
            avg_state[k] = sum([sd[k] for sd in self.buffer]) / self.world_size
        self.model.load_state_dict(avg_state)

def run_worker(rank, ps_rref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                         transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if (i + 1) % SYNC_INTERVAL == 0:
                # Push weights to PS
                rpc_sync(ps_rref, ParameterServer.push, args=(model.cpu().state_dict(),))
                # Pull averaged model
                new_state = rpc_sync(ps_rref, ParameterServer.pull, args=())
                model.load_state_dict(new_state)
                model.to(device)

            if rank == 1 and (i + 1) % 100 == 0:
                print(f"[Epoch {epoch}] Step {i+1}, Loss: {loss.item():.4f}")


def run(rank, world_size):
    if rank == 0:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        ps = ParameterServer()
        print("Parameter server initialized.")
        rpc.shutdown()
    else:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
        ps_rref = rpc.remote("ps", ParameterServer)
        run_worker(rank, ps_rref)
        rpc.shutdown()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 3))
    rank = int(os.environ.get("RANK", 0))
    run(rank, world_size)