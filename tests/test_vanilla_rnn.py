#!/usr/bin/env python3
"""
Quick test for Vanilla RNN model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.state_space.vanilla_rnn import VanillaRNNSequenceModel


def test_vanilla_rnn_basic():
    """Basic functionality test for Vanilla RNN."""
    print("Testing Vanilla RNN basic functionality...")
    
    # Create model
    model = VanillaRNNSequenceModel(
        vocab_size=36,
        d_model=128,
        n_layers=2,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, 36, (batch_size, seq_len))
    targets = torch.randint(0, 36, (batch_size, seq_len))
    
    logits, loss, hidden = model(input_ids, targets)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Verify shapes
    assert logits.shape == (batch_size, seq_len, 36), f"Wrong logits shape: {logits.shape}"
    assert hidden.shape == (2, batch_size, 128), f"Wrong hidden shape: {hidden.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    
    print("✅ Vanilla RNN model test passed!")
    return True


if __name__ == "__main__":
    test_vanilla_rnn_basic()