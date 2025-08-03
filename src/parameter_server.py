import os
import torch.optim as optim
import time
from threading import Lock, Condition

class ParameterServerAsync:
    def __init__(self, model,grace_period=10, num_sgd_steps=10, sgd_momentum=0.9, nesterov_momentum=0.9, learning_rate=0.01):
        self.lock_inference = Lock()
        self.cv_inference = Condition(self.lock_inference)
        self.lock_grace_period = Lock()
        self.cv_grace_period = Condition(self.lock_grace_period)
        self.global_model = model.cpu()
        self.outer_optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=0.01, momentum=nesterov_momentum, nesterov=True
        )
        self.delta_buffer = []
        self.inference_times = []
        self.max_inference_time = 0.0
        self.world_size = int(os.environ.get("WORLD_SIZE", 3)) - 1  # exclude server
        self.grace_period = grace_period
        self.last_update = time.time()
        self.num_sgd_steps = num_sgd_steps
        self.sgd_momentum = sgd_momentum
        self.learning_rate = learning_rate
        self.step = 0

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

    def push_deltas(self, deltas):
        with self.lock_grace_period:
            self.delta_buffer.append(deltas)
            self._apply_outer_step()
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
                param -= self.learning_rate * local_pseudograd[param_name]

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