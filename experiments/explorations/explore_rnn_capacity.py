"""
Exploration Experiment: Test RNN Capacity and Depth
========================================================

This script explores whether vanilla RNNs can achieve better ICL performance
through architectural changes WITHOUT adding gating mechanisms:

1. **Capacity Test**: Increase hidden dimension (256 → 512 → 1024)
2. **Depth Test**: Increase number of layers (2 → 5 → 16)

Research Questions:
- Does capacity matter more than gating mechanisms for ICL?
- Can deep RNNs overcome the limitations seen with 2-layer RNNs?
- What's the relationship between model capacity and ICL performance?
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.moore_dataset import MooreICLDataset, ICLDatasetConfig, load_or_create_icl_samples
from src.models.moore_vanilla_rnn import create_moore_vanilla_rnn
from src.training.vanilla_rnn_trainer import MooreVanillaRNNTrainer, TrainingConfig, ICLDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="RNN Capacity/Depth Exploration")
    
    # Model architecture
    parser.add_argument("--d-model", type=int, default=256, 
                       help="Hidden dimension (256/512/1024)")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of RNN layers (2/5/16)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=800)
    
    # Data
    parser.add_argument("--dataset-path", type=str, 
                       default="data/icl_dataset.pt")
    parser.add_argument("--experiment-name", type=str, required=True,
                       help="Name for this experiment (e.g., 'rnn_d512' or 'rnn_l16')")
    
    # Output
    parser.add_argument("--save-dir", type=str, default="experiments/explorations/results")
    
    return parser.parse_args()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset samples
    print(f"📦 Loading dataset from {args.dataset_path}")
    all_samples = torch.load(args.dataset_path)
    
    # Create dataset config (matching the saved dataset)
    dataset_cfg = ICLDatasetConfig(
        num_samples=len(all_samples),
        max_seq_len=args.max_seq_len,
        cache_path=Path(args.dataset_path),
    )
    
    # Split indices (standard 60/20/20 split)
    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))
    
    # Create dataset objects
    train_dataset = MooreICLDataset(
        train_indices, all_samples, dataset_cfg.traj_sampler_config, dataset_cfg.max_seq_len
    )
    val_dataset = MooreICLDataset(
        val_indices, all_samples, dataset_cfg.traj_sampler_config, dataset_cfg.max_seq_len
    )
    test_dataset = MooreICLDataset(
        test_indices, all_samples, dataset_cfg.traj_sampler_config, dataset_cfg.max_seq_len
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create data loaders
    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Get vocab info from the dataset object
    vocab_size = train_dataset.vocab_size
    num_states = train_dataset.num_states
    
    print(f"\n🏗️  Building Vanilla RNN Model")
    print(f"  Configuration:")
    print(f"    Hidden Dimension: {args.d_model}")
    print(f"    Number of Layers: {args.num_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    Activation: {args.activation}")
    print(f"    Vocabulary Size: {vocab_size}")
    print(f"    Num States: {num_states}")
    
    # Create model
    model = create_moore_vanilla_rnn(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
    )
    
    param_count = count_parameters(model)
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Parameter Ratio vs Baseline (256d, 2L): {param_count / 200_000:.2f}x")
    
    # Training config
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    
    # Create trainer
    trainer = MooreVanillaRNNTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=training_config,
    )
    
    # Train
    print(f"\n🚀 Starting Training ({args.epochs} epochs)")
    print("=" * 60)
    
    history = trainer.train()
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Calculate validation and test accuracy
    val_acc = trainer._calculate_accuracy(trainer.val_loader)
    test_acc = trainer._calculate_accuracy(test_loader)
    val_loss = trainer.evaluate()  # Returns validation loss
    
    # Calculate test loss
    trainer.model.eval()
    test_loss = 0.0
    test_steps = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = trainer._move_to_device(batch)
            _, loss = trainer.model(
                batch["input_ids"],
                targets=batch["target_ids"],
                unknown_mask=batch["loss_mask"],
            )
            if loss is not None:
                test_loss += loss.item()
                test_steps += 1
    test_loss = test_loss / max(1, test_steps)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model checkpoint
    checkpoint_path = save_dir / f"{args.experiment_name}_{timestamp}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': history,
        'config': {
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'activation': args.activation,
            'vocab_size': vocab_size,
            'num_states': num_states,
        },
        'final_results': {
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'val_loss': val_loss,
            'test_loss': test_loss,
        },
        'parameter_count': param_count,
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved to: {checkpoint_path}")
    
    # Save metrics to JSON
    metrics = {
        'experiment': args.experiment_name,
        'timestamp': timestamp,
        'model_config': {
            'architecture': 'Vanilla RNN',
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'activation': args.activation,
            'vocab_size': vocab_size,
            'num_states': num_states,
            'parameter_count': param_count,
        },
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.epochs,
            'max_seq_len': args.max_seq_len,
        },
        'training_history': history,
        'final_results': {
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'val_loss': val_loss,
            'test_loss': test_loss,
        }
    }
    
    metrics_path = save_dir / f"{args.experiment_name}_{timestamp}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETED: {args.experiment_name}")
    print(f"📊 Test Accuracy: {test_acc:.4f}")
    print(f"📦 Parameters: {param_count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
