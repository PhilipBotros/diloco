import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from threading import Lock, Condition
from torch.distributed.rpc import (
    init_rpc,
    shutdown,
    remote
)
import torch.distributed as dist
import time

BATCH_SIZE = 64
NUM_EPOCHS = 10
INPUT_SIZE = 28 * 28  
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
NUM_INNER_STEPS = 10

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    )

class ParameterServer:
    def __init__(self, grace_period=10):
        self.lock = Lock()
        self.cv = Condition(self.lock)
        self.global_model = create_model().cpu()
        self.outer_optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=0.01, momentum=0.9, nesterov=True
        )
        self.delta_buffer = []
        self.inference_times = []
        self.max_inference_time = 0.0
        self.world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude server
        self.grace_period = grace_period
        self.last_update = time.time()

    def push_inference_time(self, inference_time):
        with self.cv:
            self.inference_times.append(inference_time)
            if len(self.inference_times) == self.world_size:
                self.max_inference_time = max(self.inference_times)
                self.cv.notify_all() 
    
    def pull_max_inference_time(self):
        with self.cv:
            while len(self.inference_times) < self.world_size:
                self.cv.wait()  # Wait until all inference times are pushed
            return self.max_inference_time

    def push_deltas(self, deltas):
        with self.lock:
            self.delta_buffer.append(deltas)
            if len(self.delta_buffer) == self.world_size or time.time() - self.last_update > self.grace_period:
                self._apply_outer_step()
                self.last_update = time.time()

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
        self.delta_buffer = []

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
    

    def calc_inner_steps_process(model, wall_time=10, num_steps_inference=50, return_time=False):
        """ Wall time is the time to process one inner step,
            we calculate the inner steps of this process such that it's bounded by wall_time """
        inference_time = 0
        x = torch.randn(BATCH_SIZE, INPUT_SIZE).to(device)
        for _ in range(num_steps_inference):
            start_time = time.time()
            _ = model(x)
            inference_time += time.time() - start_time  
        avg_inference_time = inference_time / num_steps_inference
        if return_time:
            return avg_inference_time
        return int(wall_time / avg_inference_time)
    
    # Calculate local avg inference time
    local_avg_inference_time = calc_inner_steps_process(model=local_model, wall_time=10, return_time=True)
    ps_rref.rpc_sync().push_inference_time(local_avg_inference_time)
    
    # Broadcast all inference times and pick the max
    max_inference_time = ps_rref.rpc_sync().pull_max_inference_time()
    
    # Recalculate inner steps using the max inference time
    inner_steps_process = int(local_avg_inference_time / max_inference_time) * NUM_INNER_STEPS
    
    for epoch in range(NUM_EPOCHS):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Step 1: sync local model with current global state
            global_state = ps_rref.rpc_sync().pull_global_model()
            global_model.load_state_dict(global_state)
            local_model.load_state_dict(global_state)

            # Step 2: inner training loop
            for _ in range(inner_steps_process):
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