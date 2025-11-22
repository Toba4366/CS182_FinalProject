#!/usr/bin/env python3
"""
🧊 Test Suite: Frozen Layer Functionality

This test suite verifies that the frozen layer experiments work correctly,
which is critical for our In-Context Learning (ICL) research.

The core hypothesis: Can transformers learn FSM rules when only the final 
linear layer is trainable? This tests whether ICL happens in the attention 
layers or just in the output projection.

Test Coverage:
1. ✅ Model Creation: Can we create models with frozen parameters?
2. ✅ Parameter Counting: Are the right parameters frozen/trainable?
3. ✅ Training Step: Do gradients only flow to unfrozen parameters?
4. ✅ Gradient Verification: Are frozen parameters really not updating?
5. ✅ Training Integration: Does the full training loop work with frozen layers?

Critical for experiment validity - ensures our ICL tests are meaningful!
"""

import sys
import os
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our team's transformer (not the scaffold)
from models.transformers.traditional import CausalTransformer


class TestFrozenLayers(unittest.TestCase):
    """
    🧪 Comprehensive test suite for frozen layer functionality.
    
    Each test verifies a different aspect of the freezing mechanism:
    • Parameter states (frozen vs trainable)
    • Gradient flow behavior  
    • Training step integration
    • Component-wise freezing (embeddings, layers, head)
    """

    def setUp(self):
        """Set up test fixtures - run before each test method."""
        self.vocab_size = 36
        self.d_model = 64  # Small for fast tests
        self.n_layers = 2
        self.batch_size = 4
        self.seq_len = 16
        self.device = torch.device('cpu')  # Use CPU for consistent tests
        
        # Create test batch
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_normal_model_creation(self):
        """🎯 Test 1: Verify normal (unfrozen) model has all parameters trainable."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=False,
            freeze_embeddings=False
        )
        
        # All parameters should be trainable in normal mode
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertEqual(total_params, trainable_params, 
                        "Normal model should have all parameters trainable")
        
        # Verify get_frozen_info works
        frozen_info = model.get_frozen_info()
        self.assertEqual(frozen_info['frozen_percentage'], 0.0,
                        "Normal model should have 0% frozen parameters")
        
        print(f"✅ Normal model: {total_params:,} trainable parameters")

    def test_frozen_layers_only(self):
        """🧊 Test 2: Verify freeze_layers=True freezes transformer blocks but not embeddings."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=False
        )
        
        # Check embeddings are still trainable
        for param in model.token_embed.parameters():
            self.assertTrue(param.requires_grad, "Token embeddings should be trainable")
        for param in model.pos_embed.parameters():
            self.assertTrue(param.requires_grad, "Position embeddings should be trainable")
        
        # Check transformer blocks are frozen
        for param in model.blocks.parameters():
            self.assertFalse(param.requires_grad, "Transformer blocks should be frozen")
        for param in model.ln_f.parameters():
            self.assertFalse(param.requires_grad, "Final LayerNorm should be frozen")
        
        # Check output head is trainable
        for param in model.head.parameters():
            self.assertTrue(param.requires_grad, "Output head should always be trainable")
        
        frozen_info = model.get_frozen_info()
        self.assertGreater(frozen_info['frozen_percentage'], 0, 
                          "Should have some frozen parameters")
        self.assertLess(frozen_info['frozen_percentage'], 100, 
                       "Should not have all parameters frozen")
        
        print(f"✅ Frozen layers: {frozen_info['frozen_percentage']:.1f}% parameters frozen")

    def test_frozen_all(self):
        """🧊 Test 3: Verify freeze_embeddings=True freezes everything except head."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=True
        )
        
        # Check embeddings are frozen
        for param in model.token_embed.parameters():
            self.assertFalse(param.requires_grad, "Token embeddings should be frozen")
        for param in model.pos_embed.parameters():
            self.assertFalse(param.requires_grad, "Position embeddings should be frozen")
        
        # Check transformer blocks are frozen
        for param in model.blocks.parameters():
            self.assertFalse(param.requires_grad, "Transformer blocks should be frozen")
        for param in model.ln_f.parameters():
            self.assertFalse(param.requires_grad, "Final LayerNorm should be frozen")
        
        # Check ONLY output head is trainable
        for param in model.head.parameters():
            self.assertTrue(param.requires_grad, "Output head should always be trainable")
        
        frozen_info = model.get_frozen_info()
        self.assertGreater(frozen_info['frozen_percentage'], 80, 
                          "Should have >80% parameters frozen (almost everything)")
        
        # Verify only head parameters are trainable
        head_params = sum(p.numel() for p in model.head.parameters())
        trainable_params = frozen_info['trainable_parameters']
        self.assertEqual(head_params, trainable_params,
                        "Only output head should be trainable when everything is frozen")
        
        print(f"✅ Frozen all: {frozen_info['frozen_percentage']:.1f}% parameters frozen")
        print(f"   Only {trainable_params:,} parameters trainable (head only)")

    def test_gradient_flow(self):
        """🔬 Test 4: Verify gradients only flow to unfrozen parameters."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=True
        )
        
        # Forward pass with loss computation
        logits, loss = model(self.input_ids, self.targets)
        
        # Backward pass
        loss.backward()
        
        # Check that only unfrozen parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, 
                                   f"Trainable parameter {name} should have gradients")
                self.assertGreater(torch.abs(param.grad).sum().item(), 0,
                                 f"Trainable parameter {name} should have non-zero gradients")
            else:
                # Frozen parameters should not have gradients computed
                # (PyTorch skips gradient computation for requires_grad=False)
                if param.grad is not None:
                    self.assertEqual(torch.abs(param.grad).sum().item(), 0,
                                   f"Frozen parameter {name} should have zero gradients")
        
        print("✅ Gradient flow: Only unfrozen parameters receive gradients")

    def test_parameter_update_isolation(self):
        """🔒 Test 5: Verify frozen parameters don't change during training steps."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=True
        )
        
        # Store initial parameter values
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Set up optimizer (should only affect trainable parameters)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                                     lr=1e-3)
        
        # Perform training step
        optimizer.zero_grad()
        logits, loss = model(self.input_ids, self.targets)
        loss.backward()
        optimizer.step()
        
        # Verify parameter changes
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Trainable parameters should have changed
                param_changed = not torch.allclose(initial_params[name], param.data, atol=1e-6)
                self.assertTrue(param_changed, 
                               f"Trainable parameter {name} should change during training")
            else:
                # Frozen parameters should be identical
                param_unchanged = torch.allclose(initial_params[name], param.data, atol=1e-8)
                self.assertTrue(param_unchanged, 
                               f"Frozen parameter {name} should not change during training")
        
        print("✅ Parameter isolation: Frozen parameters unchanged, trainable parameters updated")

    def test_training_loop_integration(self):
        """🏃 Test 6: Verify frozen layers work in realistic training scenario."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=True
        )
        
        # Create optimizer for trainable parameters only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        
        # Simulate several training steps
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            
            # Create random batch
            batch_inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            batch_targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            
            # Forward pass
            logits, loss = model(batch_inputs, batch_targets)
            self.assertIsNotNone(loss, "Loss should be computed")
            self.assertTrue(torch.isfinite(loss), "Loss should be finite")
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            losses.append(loss.item())
        
        # Verify training can proceed (loss should change)
        self.assertGreater(len(set(losses)), 1, 
                          "Loss should change across training steps")
        
        frozen_info = model.get_frozen_info()
        print(f"✅ Training integration: {len(losses)} steps completed")
        print(f"   Model: {frozen_info['frozen_percentage']:.1f}% frozen, "
              f"{frozen_info['trainable_parameters']:,} trainable parameters")

    def test_component_breakdown(self):
        """📊 Test 7: Verify detailed parameter breakdown is accurate."""
        model = CausalTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            freeze_layers=True,
            freeze_embeddings=True
        )
        
        frozen_info = model.get_frozen_info()
        
        # Verify breakdown sums match totals
        component_total = sum(frozen_info['component_breakdown'].values())
        self.assertEqual(component_total, frozen_info['total_parameters'],
                        "Component breakdown should sum to total parameters")
        
        trainable_total = sum(frozen_info['trainable_breakdown'].values())
        self.assertEqual(trainable_total, frozen_info['trainable_parameters'],
                        "Trainable breakdown should sum to trainable parameters")
        
        # In fully frozen mode, only head should be trainable
        self.assertEqual(frozen_info['trainable_breakdown']['token_embeddings'], 0)
        self.assertEqual(frozen_info['trainable_breakdown']['positional_embeddings'], 0)
        self.assertEqual(frozen_info['trainable_breakdown']['transformer_blocks'], 0)
        self.assertEqual(frozen_info['trainable_breakdown']['final_layernorm'], 0)
        self.assertGreater(frozen_info['trainable_breakdown']['output_head'], 0)
        
        print("✅ Component breakdown: Detailed parameter accounting is accurate")

def run_frozen_layer_tests():
    """🚀 Run all frozen layer tests and report results."""
    print("🧊 FROZEN LAYER FUNCTIONALITY TESTS")
    print("=" * 60)
    print("Testing In-Context Learning experiment infrastructure...")
    print("This verifies that frozen layer experiments work correctly.\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFrozenLayers)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL FROZEN LAYER TESTS PASSED!")
        print("✅ Frozen layer experiments are ready for ICL research")
        print("✅ Parameter freezing works correctly")
        print("✅ Training integration is functional")
        print("\n💡 You can now run frozen layer experiments with confidence!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Fix the issues before running frozen layer experiments")
        print(f"   Failed: {len(result.failures)} tests")
        print(f"   Errors: {len(result.errors)} tests")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_frozen_layer_tests()
    sys.exit(0 if success else 1)