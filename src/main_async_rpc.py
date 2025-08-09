import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.distributed.rpc import shutdown
from src.model import create_model
from src.utils import init_parameter_server, init_worker
from src.config import load_training_params

def run_worker(rank, worker_config):
    training_params = load_training_params()
    ps_rref = worker_config.ps_rref
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global_model = create_model().to(device)
    local_model = create_model().to(device)
    
    inner_optimizer = optim.AdamW(local_model.parameters(), lr=worker_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create a simple random sampler for this worker
    world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude parameter server
    indices = torch.randperm(len(dataset))
    worker_indices = indices[rank-1::world_size]
    sampler = torch.utils.data.SubsetRandomSampler(worker_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=training_params.batch_size, sampler=sampler)

    global_model.train()
    local_model.train()
    

    def calc_inner_steps_process(model, wall_time=10, num_steps_inference=50, return_time=False):
        """ Wall time is the time to process one inner step,
            we calculate the inner steps of this process such that it's bounded by wall_time """
        inference_time = 0
        x = torch.randn(training_params.batch_size, training_params.input_size).to(device)
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
    inner_steps_process = int(local_avg_inference_time / max_inference_time) * training_params.local_updates
    
    for epoch in range(training_params.num_epochs):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Step 1: sync local model with current global state and get updated learning rate
            global_state = ps_rref.rpc_sync().pull_global_model()
            global_model.load_state_dict(global_state)
            local_model.load_state_dict(global_state)
            
            # Update learning rate for this worker's shard
            current_lr = ps_rref.rpc_sync().calculate_learning_rate(worker_config.shard_id)
            for param_group in inner_optimizer.param_groups:
                param_group['lr'] = current_lr

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
            ps_rref.rpc_sync().push_deltas(deltas, rank, worker_config.shard_id)

            if rank == 1 and (i + 1) % 100 == 0:
                print(f"[Epoch {epoch}] Step {i+1}, Loss: {loss.item():.4f}")

def run(rank, world_size):
    if rank == 0:
        init_parameter_server(world_size)
        print("Parameter server initialized with DiLoCo algorithm.")
        shutdown()
    else:
        worker_config = init_worker(rank, world_size)
        run_worker(rank, worker_config)
        shutdown()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 3))
    rank = int(os.environ.get("RANK", 0))
    run(rank, world_size)