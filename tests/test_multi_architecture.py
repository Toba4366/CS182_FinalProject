#!/usr/bin/env python3
"""
Comprehensive multi-architecture testing suite.

Tests all model architectures (Transformer, LSTM, Vanilla RNN, and future S4/Mamba)
for token compatibility, training functionality, and cross-architecture comparisons.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
import pandas as pd
import h5py
from typing import Dict, List, Tuple, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our models
from models.transformers.traditional import CausalTransformer
from models.state_space.vanilla_rnn import VanillaRNNSequenceModel

# Try to import LSTM (from the new location)
try:
    from models.state_space.lstm import LSTMSequenceModel
    LSTM_AVAILABLE = True
except Exception as e:
    print(f"⚠️  LSTM not available: {e}")
    LSTM_AVAILABLE = False

# Import data loaders
sys.path.append(str(Path(__file__).parent))
from test_data_integrity import load_pkl_data, load_json_data, load_parquet_data, load_hdf5_data


class MultiArchitectureTestSuite:
    """Comprehensive testing for all model architectures."""
    
    def __init__(self):
        self.vocab_size = 36
        self.test_results = {}
        print("🧪 Multi-Architecture Test Suite")
        print("=" * 50)
    
    def test_token_compatibility_all_models(self):
        """Test that all models handle our 36-token vocabulary correctly."""
        print("\n1. 🔤 Token Compatibility Test - All Models")
        
        models_to_test = [
            ("Transformer", self._create_transformer),
            ("Vanilla RNN", self._create_vanilla_rnn),
        ]
        
        if LSTM_AVAILABLE:
            models_to_test.append(("LSTM", self._create_lstm))
        
        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        for model_name, model_creator in models_to_test:
            print(f"   Testing {model_name}...")
            
            try:
                model = model_creator()
                model.eval()
                
                with torch.no_grad():
                    if model_name == "Transformer":
                        logits, _ = model(input_ids)  # Transformer returns (logits, loss)
                        hidden = None
                    else:  # RNN-based models
                        logits, _, hidden = model(input_ids)
                
                # Verify output dimensions
                assert logits.shape == (batch_size, seq_len, self.vocab_size), \
                    f"{model_name} output shape mismatch: {logits.shape}"
                
                # Verify logits are reasonable
                assert torch.isfinite(logits).all(), f"{model_name} produced non-finite logits"
                assert not torch.isnan(logits).any(), f"{model_name} produced NaN logits"
                
                # Test that we can compute probabilities
                probs = F.softmax(logits, dim=-1)
                assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), \
                    f"{model_name} probabilities don't sum to 1"
                
                print(f"   ✅ {model_name}: vocab_size={self.vocab_size}, output_shape={logits.shape}")
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {str(e)}")
                return False
        
        print("   🎉 All models handle 36-token vocabulary correctly!")
        return True
    
    def test_training_step_all_models(self):
        """Test that all models can perform a training step without errors."""
        print("\n2. 🏋️ Training Step Test - All Models")
        
        models_to_test = [
            ("Transformer", self._create_transformer, self._train_step_transformer),
            ("Vanilla RNN", self._create_vanilla_rnn, self._train_step_rnn),
        ]
        
        if LSTM_AVAILABLE:
            models_to_test.append(("LSTM", self._create_lstm, self._train_step_rnn))
        
        batch_size, seq_len = 4, 32  # Smaller for faster testing
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        for model_name, model_creator, train_step_fn in models_to_test:
            print(f"   Testing {model_name} training...")
            
            try:
                model = model_creator()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                
                # Forward + backward pass
                loss = train_step_fn(model, optimizer, input_ids, targets)
                
                # Verify loss is reasonable
                assert torch.isfinite(torch.tensor(loss)), f"{model_name} loss is not finite: {loss}"
                assert loss > 0, f"{model_name} loss should be positive: {loss}"
                
                # Verify gradients exist
                has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                assert has_grads, f"{model_name} gradients not computed"
                
                print(f"   ✅ {model_name}: loss={loss:.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_name} training failed: {str(e)}")
                return False
        
        print("   🎉 All models can train successfully!")
        return True
    
    def test_model_parameter_counts(self):
        """Compare parameter counts across architectures."""
        print("\n3. 🔢 Parameter Count Comparison")
        
        models_to_test = [
            ("Transformer (d=128, 2L)", lambda: self._create_transformer(d_model=128, n_layers=2)),
            ("Vanilla RNN (d=128, 2L)", lambda: self._create_vanilla_rnn(d_model=128, n_layers=2)),
        ]
        
        if LSTM_AVAILABLE:
            models_to_test.append(("LSTM (d=128, 2L)", lambda: self._create_lstm(d_model=128, n_layers=2)))
        
        param_counts = {}
        
        for model_name, model_creator in models_to_test:
            try:
                model = model_creator()
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                param_counts[model_name] = {
                    'total': total_params,
                    'trainable': trainable_params
                }
                
                print(f"   {model_name}:")
                print(f"     Total: {total_params:,} parameters")
                print(f"     Trainable: {trainable_params:,} parameters")
                
            except Exception as e:
                print(f"   ❌ {model_name} parameter count failed: {str(e)}")
        
        # Analysis
        if len(param_counts) >= 2:
            models = list(param_counts.keys())
            print(f"\n   📊 Analysis:")
            for i, model1 in enumerate(models[:-1]):
                model2 = models[i+1]
                ratio = param_counts[model2]['total'] / param_counts[model1]['total']
                print(f"     {model2} has {ratio:.1f}x parameters vs {model1}")
        
        return True
    
    def test_dataset_loading_with_models(self):
        """Test that all models work with all dataset formats."""
        print("\n4. 💾 Dataset Integration Test - All Formats × All Models")
        
        data_formats = [
            ("PKL", Path("data/full_dataset_pkl"), load_pkl_data),
            ("JSON", Path("data/full_dataset_json"), load_json_data),
            ("Parquet", Path("data/full_dataset_parquet"), load_parquet_data),
            ("HDF5", Path("data/full_dataset_hdf5"), load_hdf5_data),
        ]
        
        models_to_test = [
            ("Transformer", self._create_transformer),
            ("Vanilla RNN", self._create_vanilla_rnn),
        ]
        
        if LSTM_AVAILABLE:
            models_to_test.append(("LSTM", self._create_lstm))
        
        # Test a small sample from each format
        for format_name, data_dir, load_fn in data_formats:
            if not data_dir.exists():
                print(f"   ⚠️ Skipping {format_name} - directory not found")
                continue
            
            print(f"   Testing {format_name} format...")
            
            try:
                # Load small sample
                data = load_fn(data_dir, "train")
                if hasattr(data, 'samples'):
                    samples = data.samples[:4]  # First 4 samples
                elif isinstance(data, dict) and 'samples' in data:
                    samples = data['samples'][:4]
                else:
                    samples = data[:4]
                
                # Convert to tensors
                if isinstance(samples[0], dict) and 'tokens' in samples[0]:
                    tokens = torch.tensor([s['tokens'][:64] for s in samples])  # Truncate to 64
                else:
                    tokens = torch.tensor([s[:64] for s in samples])
                
                # Pad if needed
                if tokens.size(1) < 64:
                    pad_width = 64 - tokens.size(1)
                    tokens = F.pad(tokens, (0, pad_width), value=0)
                
                # Test with each model
                for model_name, model_creator in models_to_test:
                    try:
                        model = model_creator()
                        model.eval()
                        
                        with torch.no_grad():
                            if model_name == "Transformer":
                                logits, _ = model(tokens)  # Transformer returns (logits, loss)
                            else:
                                logits, _, _ = model(tokens)
                        
                        assert logits.shape == (4, 64, self.vocab_size)
                        print(f"     ✅ {model_name} + {format_name}")
                        
                    except Exception as e:
                        print(f"     ❌ {model_name} + {format_name}: {str(e)}")
                        return False
            
            except Exception as e:
                print(f"   ❌ {format_name} loading failed: {str(e)}")
                return False
        
        print("   🎉 All models work with all dataset formats!")
        return True
    
    def test_gradient_flow(self):
        """Test gradient flow in all architectures."""
        print("\n5. 🌊 Gradient Flow Test")
        
        models_to_test = [
            ("Transformer", self._create_transformer),
            ("Vanilla RNN", self._create_vanilla_rnn),
        ]
        
        if LSTM_AVAILABLE:
            models_to_test.append(("LSTM", self._create_lstm))
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        for model_name, model_creator in models_to_test:
            print(f"   Testing {model_name}...")
            
            try:
                model = model_creator()
                model.train()
                
                # Forward pass
                if model_name == "Transformer":
                    logits, loss = model(input_ids, targets)  # Transformer handles targets internally
                else:
                    _, loss, _ = model(input_ids, targets)
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                grad_norms = []
                zero_grads = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        if grad_norm == 0:
                            zero_grads += 1
                
                avg_grad = np.mean(grad_norms) if grad_norms else 0
                max_grad = np.max(grad_norms) if grad_norms else 0
                
                print(f"     Avg gradient norm: {avg_grad:.6f}")
                print(f"     Max gradient norm: {max_grad:.6f}")
                print(f"     Zero gradients: {zero_grads}/{len(list(model.parameters()))}")
                
                # Verify reasonable gradients
                assert avg_grad > 0, f"{model_name} has zero average gradients"
                assert max_grad < 100, f"{model_name} has exploding gradients: {max_grad}"
                
                print(f"   ✅ {model_name} gradients look healthy")
                
            except Exception as e:
                print(f"   ❌ {model_name} gradient test failed: {str(e)}")
                return False
        
        print("   🎉 All models have healthy gradient flow!")
        return True
    
    # Helper methods
    def _create_transformer(self, **kwargs):
        default_args = {
            'vocab_size': self.vocab_size,
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'dropout': 0.1
        }
        default_args.update(kwargs)
        return CausalTransformer(**default_args)
    
    def _create_vanilla_rnn(self, **kwargs):
        default_args = {
            'vocab_size': self.vocab_size,
            'd_model': 128,
            'n_layers': 2,
            'dropout': 0.1
        }
        default_args.update(kwargs)
        return VanillaRNNSequenceModel(**default_args)
    
    def _create_lstm(self, **kwargs):
        if not LSTM_AVAILABLE:
            raise ValueError("LSTM not available")
        default_args = {
            'vocab_size': self.vocab_size,
            'd_model': 128,
            'n_layers': 2,
            'dropout': 0.1
        }
        default_args.update(kwargs)
        return LSTMSequenceModel(**default_args)
    
    def _train_step_transformer(self, model, optimizer, input_ids, targets):
        """Training step for transformer."""
        model.train()
        optimizer.zero_grad()
        
        logits, loss = model(input_ids, targets)  # Transformer handles targets internally
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _train_step_rnn(self, model, optimizer, input_ids, targets):
        """Training step for RNN-based models (LSTM, Vanilla RNN)."""
        model.train()
        optimizer.zero_grad()
        
        _, loss, _ = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("Starting comprehensive multi-architecture testing...\n")
        
        tests = [
            ("Token Compatibility", self.test_token_compatibility_all_models),
            ("Training Functionality", self.test_training_step_all_models),
            ("Parameter Counts", self.test_model_parameter_counts),
            ("Dataset Integration", self.test_dataset_loading_with_models),
            ("Gradient Flow", self.test_gradient_flow),
        ]
        
        results = {}
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, test_fn in tests:
            try:
                result = test_fn()
                results[test_name] = result
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results[test_name] = False
        
        # Final summary
        print("\n" + "=" * 50)
        print("🏁 FINAL TEST SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Multi-architecture system ready for experiments!")
            return True
        else:
            print("⚠️ Some tests failed. Review output above.")
            return False


if __name__ == "__main__":
    test_suite = MultiArchitectureTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🚀 Ready for multi-architecture FSM experiments!")
        print("   • Transformer (current baseline)")
        print("   • Vanilla RNN (simple baseline)")
        if LSTM_AVAILABLE:
            print("   • LSTM (memory-enhanced RNN)")
        print("   • S4/Mamba (coming soon)")
    else:
        print("\n🔧 Fix issues above before proceeding.")
    
    sys.exit(0 if success else 1)