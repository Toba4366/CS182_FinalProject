#!/usr/bin/env python3
"""
Simple training test to generate some data for visualization.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fsm.moore_machine import MooreMachineGenerator
from src.training.dataset import MooreMachineDataset
from src.training.models import SimpleTransformer, TransformerConfig
from src.training.trainer import ICLTrainer

def run_quick_training():
    """Run a very quick training session to generate some data."""
    print("🚀 Starting quick training run...")
    
    # Create small dataset for speed
    print("📊 Creating dataset...")
    train_dataset = MooreMachineDataset(
        num_machines=10,  # Very small for speed
        examples_per_machine=3,
        sequence_length=5,
        test_sequence_length=8,
        vocab_size=50,
        seed=42
    )
    
    val_dataset = MooreMachineDataset(
        num_machines=5,
        examples_per_machine=3,
        sequence_length=5,
        test_sequence_length=8,
        vocab_size=50,
        seed=123
    )
    
    print(f"✅ Dataset created: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create small model
    print("🧠 Creating model...")
    config = TransformerConfig(
        vocab_size=50,
        max_seq_len=64,
        d_model=64,  # Very small
        num_heads=2,
        num_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    model = SimpleTransformer(config)
    print(f"✅ Model created: {model.count_parameters():,} parameters")
    
    # Create trainer
    print("🏃 Creating trainer...")
    trainer = ICLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1e-3,  # Higher for quick convergence
        weight_decay=0.01,
        warmup_steps=10,
        max_steps=50,  # Very few steps
        batch_size=4,
        eval_steps=20,
        save_steps=50,
        device="cpu",  # Use CPU for simplicity
        save_dir="results/quick_test",
        use_wandb=False
    )
    
    print("✅ Trainer created")
    
    # Run training
    print("🔥 Starting training...")
    history = trainer.train()
    
    print("✅ Training completed!")
    print(f"📈 Final training loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"📊 Final validation loss: {history['val_losses'][-1]:.4f}")
    
    # Save results
    results_dir = Path("results/quick_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"💾 Results saved to {results_dir}")
    
    return history

if __name__ == '__main__':
    try:
        history = run_quick_training()
        print("\n🎉 Quick training successful! You now have data for visualization.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)