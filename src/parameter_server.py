import os
import torch.optim as optim
import time
from threading import Lock, Condition
from src.config import TrainingParams
from src.utils import compute_model_hash
import numpy as np

class ParameterServerAsync:
    def __init__(self, model, training_params: TrainingParams = None):
        if training_params is None:
            from src.config import load_training_params
            training_params = load_training_params()
            
        self.initialized = False
        self.lock_inference = Lock()
        self.cv_inference = Condition(self.lock_inference)
        self.lock_grace_period = Lock()
        self.cv_grace_period = Condition(self.lock_grace_period)
        self.global_model = model.cpu()
        self.outer_optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=training_params.learning_rate, 
            momentum=training_params.tau, 
            nesterov=True
        )
        self.delta_buffer = []
        self.inference_times = []
        self.max_inference_time = 0.0
        self.world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude server
        self.num_shards = self.world_size # TODO: Make this configurable
        self.update_steps_per_shard = np.zeros(self.num_shards, dtype=np.int32)
        self.grace_period = training_params.grace_period
        self.last_update = time.time()
        self.num_sgd_steps = training_params.local_updates
        self.sgd_momentum = training_params.momentum
        self.learning_rate = training_params.learning_rate
        self.step = 0
        self.training_params = training_params
        self.set_initialized()

    def set_initialized(self):
        self.initialized = True

    def is_initialized(self):
        return self.initialized
    
    def is_worker_setup_correct(self, model, params):
        # TODO: We now run this check when connecting to the parameter server 
        # but there are no explicit checks during training
        if compute_model_hash(self.global_model) != compute_model_hash(model):
            raise RuntimeError("Model mismatch")
        
        if self.training_params != params:
            raise RuntimeError("Training parameters mismatch")
        return True

    def push_inference_time(self, inference_time):
        with self.cv_inference:
            self.inference_times.append(inference_time)
            if len(self.inference_times) == self.world_size:
                self.max_inference_time = max(self.inference_times)
                self.cv_inference.notify_all() 
    
    def pull_max_inference_time(self):
        with self.cv_inference:
            while len(self.inference_times) < self.world_size:
                self.cv_inference.wait()  # Wait until all inference times are pushed
            return self.max_inference_time

    def is_grace_period_over(self):
        return time.time() - self.last_update > self.grace_period

    def push_deltas(self, deltas, node_id, shard_id):
        with self.lock_grace_period:
            self.delta_buffer.append(deltas)
            self._apply_outer_step()
            self.update_steps_per_shard[shard_id] += 1
            if self.is_grace_period_over():
                self.last_update = time.time()
                # Notify any waiting threads
                with self.cv_grace_period:   
                    self.cv_grace_period.notify_all()

    def pull_global_model(self):
        with self.cv_grace_period:
            while not self.is_grace_period_over():
                self.cv_grace_period.wait()
            return self.global_model.state_dict()
        
    @property
    def shard_weights(self):
        weights = 1.0 / (self.update_steps_per_shard + 1.0)  
        return weights / weights.sum() 
        
    def assign_shard(self):
        # Sample a shard based on the inverse training progress
        # Simpler than the paper for now as our data shards are the same size
        return np.random.choice(self.num_shards, p=self.shard_weights)

    def _apply_outer_step(self):
        avg_deltas = {}
        for k in self.delta_buffer[0].keys():
            avg_deltas[k] = sum([deltas[k] for deltas in self.delta_buffer]) / self.world_size
        
        for param_name, param in self.global_model.named_parameters():
            param.grad = avg_deltas[param_name]

        # Nesterov momentum step every num_sgd_steps
        if self.step % self.num_sgd_steps == 0:
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            self.delta_buffer = [] 
        # Standard SGD step based on gradient
        else:
            local_pseudograd = self.delta_buffer[-1]
            for param_name, param in self.global_model.named_parameters():
                param.data -= self.learning_rate * local_pseudograd[param_name]

        self.step += 1

class ParameterServerSync:
    def __init__(self, model):
        self.lock = Lock()
        self.global_model = model.cpu()
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