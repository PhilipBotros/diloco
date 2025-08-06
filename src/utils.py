from dataclasses import dataclass
from torch.distributed.rpc import init_rpc, remote
from torch.distributed.rpc.api import RRef
from torch.distributed.rpc.api import DistNetworkError
from src.parameter_server import ParameterServerAsync
from src.model import create_model
from src.config import load_training_params
import time
import hashlib
import torch

@dataclass
class WorkerConfig:
    ps_rref: RRef
    shard_id: int
    learning_rate: float

def connect_to_parameter_server(rank, base_interval=2, max_retries=5, max_interval=30):
    for attempt in range(1, max_retries + 1):
        try:
            ps_rref = remote("ps", ParameterServerAsync)
            print(f"[INFO] Worker {rank} successfully connected to PS on attempt {attempt}")
            return ps_rref
        except (RuntimeError, DistNetworkError) as e:
            wait_time = min(base_interval * (2 ** (attempt - 1)), max_interval)
            print(f"[WARN] Worker {rank} failed to connect to PS ({e}), "
                  f"retry {attempt}/{max_retries} in {wait_time}s")
            if attempt == max_retries:
                raise TimeoutError(f"Worker {rank} could not connect to PS after {max_retries} retries")
            time.sleep(wait_time)
    raise TimeoutError(f"Worker {rank} could not connect to PS after {max_retries} retries")

def init_parameter_server(world_size):
    training_params = load_training_params()
    init_rpc("ps", rank=0, world_size=world_size)
    ps = ParameterServerAsync(create_model(), training_params)
    return ps

def init_worker(rank, world_size):
    """ Initialize the worker and return the parameter server reference 
        We want to do a few things here:
        1. Check if the parameter server is initialized
        2. Check if the setup on this worker is what the parameter server expects
        3. Sample a data shard based on the inverse progress
        4. Assign this worker a learning rate schedule
    """
    training_params = load_training_params()
    init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    ps_rref = connect_to_parameter_server(rank)
    if not ps_rref.rpc_sync().is_initialized():
        raise RuntimeError("Parameter server is not initialized")
    if not ps_rref.rpc_sync().is_worker_setup_correct(create_model(), training_params):
        raise RuntimeError("This worker is not setup correctly")
    
    shard_id = rank - 1
    learning_rate = training_params.learning_rate
    config = WorkerConfig(ps_rref, shard_id, learning_rate)
    return config

def compute_model_hash(model, include_buffers=True):
    hasher = hashlib.sha256()
    with torch.no_grad():
        state_dict = model.state_dict()  # includes both params and buffers
        for name, tensor in state_dict.items():
            hasher.update(name.encode('utf-8'))
            hasher.update(tensor.detach().cpu().view(torch.uint8).numpy().tobytes())
    
    return hasher.hexdigest()