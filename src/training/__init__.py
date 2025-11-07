"""
Training module for in-context learning of Moore machines.

This module contains PyTorch implementations for training models on ICL tasks.
"""

from .trainer import ICLTrainer
from .dataset import MooreMachineDataset
from .models import SimpleTransformer, TransformerConfig

__all__ = [
    "ICLTrainer",
    "MooreMachineDataset", 
    "SimpleTransformer",
    "TransformerConfig"
]