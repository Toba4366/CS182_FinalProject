#!/usr/bin/env python3
"""
Quick start script to test the FSM generation and model setup.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.fsm.moore_machine import MooreMachineGenerator
from src.training.dataset import MooreMachineDataset
from src.training.models import SimpleTransformer, TransformerConfig
from src.utils.visualization import plot_fsm_diagram


def test_fsm_generation():
    """Test FSM generation."""
    print("Testing FSM generation...")
    
    generator = MooreMachineGenerator(seed=42)
    fsm = generator.generate()
    
    print(f"Generated FSM:")
    print(f"  States: {fsm.states}")
    print(f"  Actions: {fsm.actions}")
    print(f"  Transitions: {len(fsm.transitions)}")
    print(f"  Outputs: {fsm.outputs}")
    print(f"  Valid: {fsm.is_valid()}")
    
    # Test sequence generation
    actions, states, outputs = fsm.run_sequence([0, 1, 2, 1, 0])
    print(f"  Test sequence:")
    print(f"    Actions: {[0, 1, 2, 1, 0]}")
    print(f"    States:  {states}")
    print(f"    Outputs: {outputs}")
    
    return fsm


def test_dataset_creation():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    
    dataset = MooreMachineDataset(
        num_machines=10,
        examples_per_machine=3,
        sequence_length=5,
        test_sequence_length=7,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Sample vocab: {dataset.vocab[:10]}")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample target shape: {sample['target_ids'].shape}")
    
    return dataset


def test_model_creation():
    """Test model creation and frozen parameter functionality."""
    print("\nTesting model creation...")
    
    # Test normal model
    config = TransformerConfig(
        vocab_size=50,
        max_seq_len=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1
    )
    
    model = SimpleTransformer(config)
    print(f"Normal model parameters: {model.count_parameters():,}")
    
    # Test frozen model
    frozen_config = TransformerConfig(
        vocab_size=50,
        max_seq_len=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        freeze_layers=True,
        freeze_embeddings=True
    )
    
    frozen_model = SimpleTransformer(frozen_config)
    frozen_info = frozen_model.get_frozen_info()
    
    print(f"Frozen model parameter breakdown:")
    print(f"  Total: {frozen_info['total_parameters']:,}")
    print(f"  Trainable: {frozen_info['trainable_parameters']:,}")
    print(f"  Frozen: {frozen_info['frozen_parameters']:,}")
    print(f"  Frozen %: {frozen_info['frozen_percentage']:.1f}%")
    
    # Test forward pass for both models
    import torch
    input_ids = torch.randint(0, 50, (2, 20))
    
    logits_normal = model(input_ids)
    logits_frozen = frozen_model(input_ids)
    
    print(f"Normal model output shape: {logits_normal.shape}")
    print(f"Frozen model output shape: {logits_frozen.shape}")
    
    return model, frozen_model


def main():
    """Run quick tests."""
    print("=== Moore Machine ICL Quick Test ===")
    
    try:
        # Test FSM
        fsm = test_fsm_generation()
        
        # Test dataset
        dataset = test_dataset_creation()
        
        # Test model
        model, frozen_model = test_model_creation()
        
        print("\n✅ All tests passed!")
        print("\nAvailable configurations:")
        print("1. configs/base_config.yaml - 2-layer transformer")
        print("2. configs/3layer_config.yaml - 3-layer transformer")
        print("3. configs/frozen_layers_config.yaml - frozen layer experiment")
        print("\nYou can now:")
        print("1. Run training: python experiments/run_experiment.py --config configs/base_config.yaml")
        print("2. Test frozen layers: python experiments/run_experiment.py --config configs/frozen_layers_config.yaml")
        print("3. Run automated training: ./scripts/train.sh")
        print("4. Explore notebooks in: notebooks/")
        
        # Save a sample FSM diagram
        try:
            import matplotlib.pyplot as plt
            fig = plot_fsm_diagram(fsm, title="Sample Generated Moore Machine")
            fig.savefig("sample_fsm.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("4. Sample FSM diagram saved as: sample_fsm.png")
        except Exception as e:
            print(f"Could not save FSM diagram: {e}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())