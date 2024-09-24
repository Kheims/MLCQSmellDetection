import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass


@dataclass
class LSTMConfig(nn.Module):
    vocab_size: int = 50257 
    embedding_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 2  
    dropout: float = 0.1  
    max_seq_len: int = 1024  
    num_classes: int = 5  


@dataclass
class OptimizerConfig:
    optimizer_type: str = "Adam"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    momentum: float = 0.9 # for SGD


@dataclass
class SchedulerConfig:
    scheduler_type: str = "linear" 
    step_size: int = 100 # for 'step' scheduler
    gamma: float = 0.1 # for 'step' or 'exponential' scheduler
    warmup_steps: int = 0 # for 'linear' scheduler


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    clip_grad: float = 1.0


def get_optimizer(model, config: OptimizerConfig):
    if config.optimizer_type == "Adam":
        return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
    

def get_scheduler(optimizer, config: SchedulerConfig):
    if config.scheduler_type == "none":
        return None
    elif config.scheduler_type == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(1.0 - step / config.warmup_steps, 0.0))
    elif config.scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")