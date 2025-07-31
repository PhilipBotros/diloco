import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from threading import Lock
from torch.distributed.rpc import (
    init_rpc,
    shutdown,
    remote
)

INNER_STEPS = 10
BATCH_SIZE = 64
NUM_EPOCHS = 10

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
        self.global_model = create_model().cpu()
        self.outer_optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=0.01, momentum=0.9, nesterov=True
        )
        self.delta_buffer = []
        self.world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude server

    def push_deltas(self, deltas):
        with self.lock:
            self.delta_buffer.append(deltas)
            if len(self.delta_buffer) == self.world_size:
                self._apply_outer_step()
                self.delta_buffer = []

    def pull_global_model(self):
        with self.lock:
            return self.global_model.state_dict()

    def _apply_outer_step(self):
        avg_deltas = {}
        for k in self.delta_buffer[0].keys():
            avg_deltas[k] = sum([deltas[k] for deltas in self.delta_buffer]) / self.world_size
        
        for param_name, param in self.global_model.named_parameters():
            param.grad = avg_deltas[param_name]
        
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

def run_worker(rank, ps_rref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global_model = create_model().to(device)
    local_model = create_model().to(device)
    
    inner_optimizer = optim.AdamW(local_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create a simple random sampler for this worker
    world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude parameter server
    indices = torch.randperm(len(dataset))
    worker_indices = indices[rank-1::world_size]
    sampler = torch.utils.data.SubsetRandomSampler(worker_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    global_model.train()
    local_model.train()
    
    for epoch in range(NUM_EPOCHS):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Step 1: sync local model with current global state
            global_state = ps_rref.rpc_sync().pull_global_model()
            global_model.load_state_dict(global_state)
            local_model.load_state_dict(global_state)

            # Step 2: inner training loop
            for _ in range(INNER_STEPS):
                pred = local_model(x)
                loss = loss_fn(pred, y)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # Step 3: compute deltas and send to parameter server
            deltas = {}
            with torch.no_grad():
                for (name, g_param), (_, l_param) in zip(global_model.named_parameters(), local_model.named_parameters()):
                    deltas[name] = g_param.data - l_param.data  # local step delta

            # Send deltas to parameter server
            # Move tensors to CPU before sending to parameter server
            deltas = {k: v.cpu() for k, v in deltas.items()}
            ps_rref.rpc_sync().push_deltas(deltas)

            if rank == 1 and (i + 1) % 100 == 0:
                print(f"[Epoch {epoch}] Step {i+1}, Loss: {loss.item():.4f}")

def run(rank, world_size):
    if rank == 0:
        init_rpc("ps", rank=rank, world_size=world_size)
        ps = ParameterServer()
        print("Parameter server initialized with DiLoCo algorithm.")
        shutdown()
    else:
        init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
        ps_rref = remote("ps", ParameterServer)
        run_worker(rank, ps_rref)
        shutdown()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 3))
    rank = int(os.environ.get("RANK", 0))
    run(rank, world_size)