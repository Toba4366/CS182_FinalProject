"""
Training utilities shared across all model architectures.

This module contains common training functions like optimizers, schedulers,
checkpointing, and training steps that can be used with any model.
"""

import math
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay for regularization
        betas: Adam betas for momentum and second moment
        
    Returns:
        AdamW optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=betas)
    return optimizer


def create_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to 10% of initial lr.
    
    Args:
        optimizer: AdamW optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save model checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: PyTorch model
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        step: Training step
        extra: Extra data to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    map_location: Optional[str] = None,
) -> int:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        map_location: Device to load onto
        
    Returns:
        Training step from checkpoint
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("step", 0)


def train_step_generic(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    batch: Tuple[torch.Tensor, torch.Tensor],
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    model_type: str = "rnn",  # "rnn", "transformer"
    hidden: Optional[torch.Tensor] = None,
) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Generic training step that works with different model types.
    
    Args:
        model: PyTorch model
        optimizer: AdamW optimizer
        scheduler: Optional LR scheduler
        batch: (input_ids, targets)
        max_grad_norm: Gradient clipping norm
        device: Device to run on
        model_type: "rnn" or "transformer"
        hidden: Hidden state for RNN models
        
    Returns:
        (loss_value, new_hidden_state)
    """
    model.train()
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    
    if model_type == "transformer":
        # Transformers handle targets internally
        _, loss = model(input_ids, targets)
        new_hidden = None
    elif model_type == "rnn":
        # RNN models return (logits, loss, hidden)
        if hidden is None:
            hidden = model.init_hidden(batch_size=input_ids.size(0), device=device)
        _, loss, new_hidden = model(input_ids, targets, hidden)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    loss.backward()
    
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    return float(loss.item()), new_hidden