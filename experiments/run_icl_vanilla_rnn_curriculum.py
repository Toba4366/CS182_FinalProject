"""
Entry point for training the Moore Vanilla RNN with curriculum learning support.
This script supports training on simple vs complex FSMs for curriculum learning experiments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.models.moore_vanilla_rnn import MooreVanillaRNN, VanillaRNNConfig
from src.training.vanilla_rnn_trainer import (
    ICLDataCollator,
    MooreVanillaRNNTrainer,
    TrainingConfig,
    evaluate_vanilla_rnn_model,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vanilla RNN for Moore ICL with Curriculum Learning")
    
    # Standard training arguments
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose training output")
    
    # Curriculum learning arguments
    parser.add_argument("--dataset-type", type=str, choices=["simple", "complex"], required=True,
                        help="Type of FSM dataset: simple (3 states, 3 actions) or complex (5 states, 4-5 actions)")
    parser.add_argument("--checkpoint-name", type=str, required=True,
                        help="Name for saving checkpoint (e.g., 'curriculum_stage1', 'curriculum_stage2')")
    parser.add_argument("--load-from", type=str, default=None,
                        help="Load model from previous checkpoint (for stage 2 of curriculum)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    return parser.parse_args()


def get_dataset_config(dataset_type: str, num_samples: int, max_seq_len: int) -> ICLDatasetConfig:
    """Create dataset configuration based on type."""
    
    if dataset_type == "simple":
        # Simple FSM: 3 states, 3 actions
        sampler_config = TrajectorySamplerConfig(
            num_states=3,
            min_actions_per_state=3,
            max_actions_per_state=3,
            seed=42
        )
        cache_path = Path("data/icl_dataset_stage_3states_3actions.pt")
        
    elif dataset_type == "complex":
        # Complex FSM: 5 states, 4-5 actions  
        sampler_config = TrajectorySamplerConfig(
            num_states=5,
            min_actions_per_state=4,
            max_actions_per_state=5,
            seed=42
        )
        cache_path = Path("data/icl_dataset_stage_5states_5actions.pt")
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return ICLDatasetConfig(
        num_samples=num_samples,
        traj_sampler_config=sampler_config,
        max_seq_len=max_seq_len,
        cache_path=cache_path,
    )


def main():
    args = parse_args()
    
    print(f"🎓 CURRICULUM LEARNING: Vanilla RNN - {args.dataset_type.upper()} FSM")
    print(f"📁 Checkpoint: {args.checkpoint_name}")
    if args.load_from:
        print(f"🔄 Loading from: {args.load_from}")
    print("=" * 60)

    # Get appropriate dataset configuration
    dataset_cfg = get_dataset_config(args.dataset_type, args.num_samples, args.max_seq_len)
    
    print(f"📊 Dataset: {args.dataset_type} FSM")
    print(f"📊 Sampler config: {dataset_cfg.traj_sampler_config.num_states} states, "
          f"{dataset_cfg.traj_sampler_config.min_actions_per_state}-{dataset_cfg.traj_sampler_config.max_actions_per_state} actions")

    # Load dataset
    all_samples = load_or_create_icl_samples(dataset_cfg)
    print(f"✅ Loaded {len(all_samples)} samples")

    # Create train/val splits
    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, min(10000, len(all_samples))))

    train_dataset = MooreICLDataset(
        train_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    val_dataset = MooreICLDataset(
        val_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    test_dataset = MooreICLDataset(
        test_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )

    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model configuration
    model_cfg = VanillaRNNConfig(
        vocab_size=train_dataset.vocab_size,
        num_states=train_dataset.num_states,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
    )
    
    # Create or load model
    model = MooreVanillaRNN(model_cfg)
    
    # Load from previous checkpoint if specified (for curriculum stage 2)
    if args.load_from:
        checkpoint_path = Path(args.load_from)
        if checkpoint_path.exists():
            print(f"🔄 Loading model from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # For curriculum learning: only load compatible layers (skip head if size mismatch)
            model_dict = model.state_dict()
            compatible_dict = {}
            
            for name, param in state_dict.items():
                if name in model_dict and param.shape == model_dict[name].shape:
                    compatible_dict[name] = param
                    print(f"✅ Loaded: {name} {param.shape}")
                else:
                    print(f"⚠️  Skipped: {name} {param.shape} (incompatible with {model_dict.get(name, 'missing').shape if name in model_dict else 'missing'})")
            
            model.load_state_dict(compatible_dict, strict=False)
            print(f"📦 Loaded {len(compatible_dict)}/{len(state_dict)} layers from Stage 1")
        else:
            print(f"⚠️  Checkpoint {checkpoint_path} not found, starting fresh")

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup training
    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        verbose=not args.no_verbose,
    )
    
    trainer = MooreVanillaRNNTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )
    
    # Train the model
    print(f"🚀 Starting training for {args.epochs} epochs...")
    history = trainer.train()

    # Evaluate
    print("\n📊 Final Evaluation:")
    val_acc = evaluate_vanilla_rnn_model(trainer.model, trainer.val_loader, trainer.device)
    print(f"Validation Accuracy: {val_acc.item():.4f}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_acc = evaluate_vanilla_rnn_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {test_acc.item():.4f}")

    # Save checkpoint
    save_path = Path(args.save_dir) / f"vanilla_rnn_{args.checkpoint_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")
    
    # Save training logs
    log_dir = Path("checkpoints/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full checkpoint with training history
    checkpoint_path = log_dir / f"vanilla_rnn_{args.checkpoint_name}_{timestamp}.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "train_accs": history["train_accs"],
        "val_accs": history["val_accs"],
        "final_val_acc": val_acc.item(),
        "final_test_acc": test_acc.item(),
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "num_states": model_cfg.num_states,
            "d_model": model_cfg.d_model,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "activation": model_cfg.activation,
        },
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
        },
        "curriculum_info": {
            "dataset_type": args.dataset_type,
            "checkpoint_name": args.checkpoint_name,
            "loaded_from": args.load_from,
        }
    }, checkpoint_path)
    print(f"✅ Full checkpoint saved to: {checkpoint_path}")
    
    # Save training metrics to JSON
    metrics = {
        "experiment": f"vanilla_rnn_curriculum_{args.dataset_type}",
        "checkpoint_name": args.checkpoint_name,
        "timestamp": timestamp,
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "num_states": model_cfg.num_states,
            "d_model": model_cfg.d_model,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "activation": model_cfg.activation,
        },
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
        },
        "curriculum_info": {
            "dataset_type": args.dataset_type,
            "loaded_from": args.load_from,
        },
        "training_history": {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "train_accs": history["train_accs"],
            "val_accs": history["val_accs"],
        },
        "final_results": {
            "val_accuracy": val_acc.item(),
            "test_accuracy": test_acc.item(),
        }
    }
    
    metrics_path = log_dir / f"vanilla_rnn_{args.checkpoint_name}_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETED: Vanilla RNN - {args.dataset_type.upper()} FSM")
    print(f"📊 Final Validation Accuracy: {val_acc.item():.4f}")
    print(f"📊 Final Test Accuracy: {test_acc.item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()