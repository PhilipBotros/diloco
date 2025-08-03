import torch.nn as nn

INPUT_SIZE = 28 * 28  
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    )