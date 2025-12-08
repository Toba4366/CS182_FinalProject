"""
Entry point for training the Moore LSTM in an ICL setting.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch  # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.models.moore_lstm import MooreLSTM, LSTMConfig
from src.training.lstm_trainer import (
    ICLDataCollator,
    MooreLSTMTrainer,
    TrainingConfig,
    evaluate_lstm_model,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM for Moore ICL")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose training output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create trajectory sampler config with explicit min/max actions
    traj_sampler_config = TrajectorySamplerConfig(
        num_states=5,
        min_actions_per_state=5,
        max_actions_per_state=5,
        seed=42,
    )

    # Use same cache path as ablation script to ensure same dataset
    cache_path = Path("data/icl_dataset_stage_5s_5a_demo90_query100.pt")

    dataset_cfg = ICLDatasetConfig(
        num_samples=max(args.num_samples, 10_000),
        max_seq_len=args.max_seq_len,
        traj_sampler_config=traj_sampler_config,
        cache_path=cache_path,
        demo_length=90,
        query_length=100,
    )

    if dataset_cfg.num_samples < 10_000:
        raise ValueError("Dataset must contain at least 10,000 samples.")

    all_samples = load_or_create_icl_samples(dataset_cfg)

    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))

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

    model_cfg = LSTMConfig(
        vocab_size=train_dataset.vocab_size,
        num_states=train_dataset.num_states,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = MooreLSTM(model_cfg)

    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        verbose=not args.no_verbose,  # verbose=True by default, unless --no-verbose
    )
    trainer = MooreLSTMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )
    
    # Train and get history
    history = trainer.train()

    print("\nEvaluating on validation set...")
    val_acc = evaluate_lstm_model(trainer.model, trainer.val_loader, trainer.device)
    print(f"Validation Accuracy: {val_acc.item():.4f}")

    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_acc = evaluate_lstm_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {test_acc.item():.4f}")
    
    # Save checkpoint and training logs
    log_dir = Path("checkpoints/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save checkpoint
    checkpoint_path = log_dir / f"lstm_direct_{timestamp}.pt"
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
            "bidirectional": model_cfg.bidirectional,
        },
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
        }
    }, checkpoint_path)
    print(f"\n✅ Checkpoint saved to: {checkpoint_path}")
    
    # Save training metrics to JSON
    metrics = {
        "experiment": "lstm_direct",
        "timestamp": timestamp,
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "num_states": model_cfg.num_states,
            "d_model": model_cfg.d_model,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "bidirectional": model_cfg.bidirectional,
        },
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
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
    
    metrics_path = log_dir / f"lstm_direct_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()