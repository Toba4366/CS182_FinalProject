"""
Entry point for training models on 8 states, 8 actions FSMs (hardest configuration).

More complex FSM with both more states and actions.
Required demo length: ~80 demos
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig
from src.models.moore_lstm import MooreLSTM, LSTMConfig
from src.models.moore_deep_vanilla_rnn import MooreDeepVanillaRNN, DeepVanillaRNNConfig
from src.training.lstm_trainer import (
    ICLDataCollator,
    MooreLSTMTrainer,
    TrainingConfig,
    evaluate_lstm_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on 8 states, 8 actions FSM")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "deep_rnn"])
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1600)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--demo-length", type=int, default=80)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # FSM configuration: 8 states, 8 actions
    traj_config = TrajectorySamplerConfig(
        num_states=8,
        min_actions_per_state=3,
        max_actions_per_state=8,
        seed=42,
    )
    
    dataset_cfg = ICLDatasetConfig(
        num_samples=max(args.num_samples, 10_000),
        traj_sampler_config=traj_config,
        max_seq_len=args.max_seq_len,
        demo_length=args.demo_length,
        cache_path=Path("data/icl_dataset_8s8a.pt"),
    )

    if dataset_cfg.num_samples < 10_000:
        raise ValueError("Dataset must contain at least 10,000 samples.")

    all_samples = load_or_create_icl_samples(dataset_cfg)

    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))

    train_dataset = MooreICLDataset(
        train_indices, all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    val_dataset = MooreICLDataset(
        val_indices, all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    test_dataset = MooreICLDataset(
        test_indices, all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )

    # Create model
    if args.model == "lstm":
        num_layers = args.num_layers or 2
        model_cfg = LSTMConfig(
            vocab_size=train_dataset.vocab_size,
            num_states=train_dataset.num_states,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            num_layers=num_layers,
            dropout=args.dropout,
            bidirectional=False,
        )
        model = MooreLSTM(model_cfg)
    else:  # deep_rnn
        num_layers = args.num_layers or 8
        model_cfg = DeepVanillaRNNConfig(
            vocab_size=train_dataset.vocab_size,
            num_states=train_dataset.num_states,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            num_layers=num_layers,
            dropout=args.dropout,
            activation="tanh",
        )
        model = MooreDeepVanillaRNN(model_cfg)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n8s8a {args.model}: {model_cfg.num_layers} layers, {model_cfg.d_model} d_model")
    print(f"Parameters: {param_count:,}")

    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        verbose=not args.no_verbose,
    )
    trainer = MooreLSTMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )
    
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
    
    # Save
    log_dir = Path("checkpoints/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_path = log_dir / f"8s8a_{args.model}_{timestamp}.pt"
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
        "model_config": model_cfg.__dict__,
        "fsm_config": {
            "num_states": 8,
            "num_actions": 8,
        },
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
        },
        "parameters": param_count,
    }, checkpoint_path)
    print(f"\n✅ Checkpoint saved to: {checkpoint_path}")
    
    metrics = {
        "experiment": f"8s8a_{args.model}",
        "timestamp": timestamp,
        "fsm_config": {"num_states": 8, "num_actions": 8},
        "model_config": model_cfg.__dict__,
        "training_config": {
            "batch_size": trainer_cfg.batch_size,
            "learning_rate": trainer_cfg.learning_rate,
            "num_epochs": trainer_cfg.num_epochs,
        },
        "parameters": param_count,
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
    
    metrics_path = log_dir / f"8s8a_{args.model}_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
