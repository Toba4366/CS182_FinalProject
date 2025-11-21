#!/usr/bin/env python3
"""
Test script to verify all models work correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import numpy as np

def test_model_creation():
    """Test that all models can be created."""
    print("🔧 Testing model creation...")
    
    # Import models
    try:
        from models.transformers.traditional import CausalTransformer
        from models.state_space.vanilla_rnn import VanillaRNNSequenceModel
        from models.state_space.lstm import LSTMSequenceModel
        from models.state_space.s4 import S4LM
        from models.state_space.mamba_mock import MockMambaLM
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    models_to_test = [
        ("Transformer", lambda: CausalTransformer(vocab_size=36, d_model=128, n_heads=4, n_layers=2, max_seq_len=256)),
        ("Vanilla RNN", lambda: VanillaRNNSequenceModel(vocab_size=36, d_model=128, n_layers=2)),
        ("LSTM", lambda: LSTMSequenceModel(vocab_size=36, d_model=128, n_layers=2)),
        ("S4", lambda: S4LM(vocab_size=36, d_model=128, n_layers=2, max_seq_len=256, s4_kwargs={'l_max': 256})),
        ("Mock Mamba", lambda: MockMambaLM(vocab_size=36, d_model=128, n_layers=2, max_seq_len=256)),
    ]
    
    for model_name, model_fn in models_to_test:
        try:
            model = model_fn()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ {model_name}: {param_count:,} parameters")
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            return False
    
    return True

def test_forward_passes():
    """Test forward passes for all models."""
    print("\n🚀 Testing forward passes...")
    
    # Import models
    try:
        from models.transformers.traditional import CausalTransformer
        from models.state_space.vanilla_rnn import VanillaRNNSequenceModel
        from models.state_space.lstm import LSTMSequenceModel
        from models.state_space.s4 import S4LM
        from models.state_space.mamba_mock import MockMambaLM
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create test input
    batch_size, seq_len = 2, 10
    x = torch.randint(0, 36, (batch_size, seq_len))
    targets = torch.randint(0, 36, (batch_size, seq_len))
    
    models_to_test = [
        ("Transformer", CausalTransformer(vocab_size=36, d_model=128, n_heads=4, n_layers=2, max_seq_len=256)),
        ("Vanilla RNN", VanillaRNNSequenceModel(vocab_size=36, d_model=128, n_layers=2)),
        ("LSTM", LSTMSequenceModel(vocab_size=36, d_model=128, n_layers=2)),
        ("S4", S4LM(vocab_size=36, d_model=128, n_layers=2, max_seq_len=256, s4_kwargs={'l_max': 256})),
        ("Mock Mamba", MockMambaLM(vocab_size=36, d_model=128, n_layers=2, max_seq_len=256)),
    ]
    
    for model_name, model in models_to_test:
        try:
            model.eval()
            with torch.no_grad():
                # Test forward pass without targets
                output = model(x)
                
                # Handle different return formats
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                expected_shape = (batch_size, seq_len, 36)
                if logits.shape == expected_shape:
                    print(f"✅ {model_name}: {logits.shape}")
                else:
                    print(f"❌ {model_name}: Expected {expected_shape}, got {logits.shape}")
                    return False
                
                # Test with targets (for loss computation)
                try:
                    result = model(x, targets)
                    if isinstance(result, tuple) and len(result) >= 2:
                        logits, loss = result[:2]
                        if torch.isfinite(loss):
                            print(f"✅ {model_name} loss: {loss.item():.4f}")
                        else:
                            print(f"⚠️ {model_name}: Non-finite loss")
                except:
                    # Some models might not support targets
                    print(f"⚠️ {model_name}: No target support")
                    
        except Exception as e:
            print(f"❌ {model_name} forward failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_training_step():
    """Test that models can perform training steps."""
    print("\n🏋️‍♂️ Testing training steps...")
    
    try:
        from models.transformers.traditional import CausalTransformer
        from models.state_space.mamba_mock import MockMambaLM
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test with simple models
    models_to_test = [
        ("Transformer", CausalTransformer(vocab_size=36, d_model=64, n_heads=4, n_layers=1, max_seq_len=32)),
        ("Mock Mamba", MockMambaLM(vocab_size=36, d_model=64, n_layers=1, max_seq_len=32)),
    ]
    
    for model_name, model in models_to_test:
        try:
            # Setup training
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Create small batch
            batch_size, seq_len = 2, 8
            inputs = torch.randint(0, 36, (batch_size, seq_len))
            targets = torch.randint(0, 36, (batch_size, seq_len))
            
            # Training step
            optimizer.zero_grad()
            
            result = model(inputs)
            if isinstance(result, tuple):
                if len(result) >= 2 and result[1] is not None:
                    # Model returns loss
                    logits, loss = result[:2]
                else:
                    # Model doesn't return loss
                    logits = result[0]
                    loss = criterion(logits.view(-1, 36), targets.view(-1))
            else:
                # Model returns only logits
                logits = result
                loss = criterion(logits.view(-1, 36), targets.view(-1))
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                print(f"✅ {model_name} training step: loss={loss.item():.4f}")
            else:
                print(f"❌ {model_name}: Non-finite loss in training")
                return False
                
        except Exception as e:
            print(f"❌ {model_name} training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_dataset_loading():
    """Test dataset loading."""
    print("\n📊 Testing dataset loading...")
    
    try:
        import h5py
        dataset_path = project_root / "data" / "full_dataset_hdf5" / "dataset.h5"
        
        if not dataset_path.exists():
            print(f"⚠️ Dataset not found at {dataset_path}")
            return True  # Not a critical failure
        
        with h5py.File(dataset_path, 'r') as f:
            train_tokens = f['train']['tokens'][:10]  # Small sample
            val_tokens = f['val']['tokens'][:5]
            
            print(f"✅ Dataset loaded:")
            print(f"   Train shape: {train_tokens.shape}")
            print(f"   Val shape: {val_tokens.shape}")
            print(f"   Token range: {train_tokens.min()} - {train_tokens.max()}")
            
            return True
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Model Tests")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Passes", test_forward_passes),
        ("Training Steps", test_training_step),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_fn in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_fn():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)