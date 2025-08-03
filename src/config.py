from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingParams:
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    momentum: float = 0.9
    
    # Model architecture
    input_size: int = 28 * 28
    hidden_size: int = 128
    num_classes: int = 10
    
    # DiLoCo specific
    local_updates: int = 10
    grace_period: int = 5
    tau: float = 0.99  # momentum for outer optimizer
    
    # Data
    data_path: Optional[str] = None
    num_workers: int = 4
    
    # Optimization
    optimizer: str = "sgd"
    scheduler: Optional[str] = None
    gradient_clipping: Optional[float] = None

def load_training_params() -> TrainingParams:
    """Load training parameters. Can be extended to load from file or environment."""
    return TrainingParams()