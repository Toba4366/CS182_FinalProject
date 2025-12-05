"""
Entry point for training the Moore S4 model (official implementation) in an ICL setting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch  # For saving checkpoint
from torch.utils.data import DataLoader  # type: ignore

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.models.moore_s4_official import MooreS4, S4Config
from src.training.icl_trainer import (
    ICLDataCollator,
    MooreICLTrainer,
    TrainingConfig,
    evaluate_model,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig
from src.fsm import MAX_STATES, MAX_ACTIONS


@dataclass
class SimpleS4Stage:
    """
    Configuration for a single (non-curriculum) training stage.
    You can tweak these from main() or via CLI later.
    """
    num_states: int = 5
    min_actions_per_state: int = 4
    max_actions_per_state: int = 5
    num_samples: int = 10_000
    epochs: int = 5
    demo_length: Optional[int] = None
    query_length: Optional[int] = None
    cache_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Moore S4 (official) ICL model with no curriculum learning."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # Fixed vocabulary:
    # states + actions + eos + query + pad
    vocab_size = MAX_STATES + MAX_ACTIONS + 3
    pad_token = MAX_STATES + MAX_ACTIONS + 2   # last token is pad

    # Define a single stage
    stage = SimpleS4Stage(
        num_states=args.num_states,
        min_actions_per_state=args.min_actions,
        max_actions_per_state=args.max_actions,
        num_samples=args.num_samples,
        epochs=args.epochs,
        cache_path=Path(
            f"data/icl_dataset_s4_official_{args.num_states}states_{args.max_actions}actions.pt"
        ),
    )

    # Build sampler and dataset configurations
    traj_sampler_config = TrajectorySamplerConfig(
        num_states=stage.num_states,
        min_actions_per_state=stage.min_actions_per_state,
        max_actions_per_state=stage.max_actions_per_state,
        seed=42,
    )

    dataset_cfg = ICLDatasetConfig(
        num_samples=stage.num_samples,
        traj_sampler_config=traj_sampler_config,
        max_seq_len=args.max_seq_len,
        cache_path=stage.cache_path,
        demo_length=stage.demo_length,
        query_length=stage.query_length,
    )

    # Load FSM samples
    all_samples = load_or_create_icl_samples(dataset_cfg)

    # 60 / 20 / 20 split
    train_size = int(0.6 * len(all_samples))
    val_size = int(0.2 * len(all_samples))
    test_size = len(all_samples) - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(all_samples)))

    train_dataset = MooreICLDataset(
        sample_indices=train_indices,
        all_samples=all_samples,
        traj_sampler_config=traj_sampler_config,
        max_seq_len=args.max_seq_len,
    )
    val_dataset = MooreICLDataset(
        sample_indices=val_indices,
        all_samples=all_samples,
        traj_sampler_config=traj_sampler_config,
        max_seq_len=args.max_seq_len,
    )
    test_dataset = MooreICLDataset(
        sample_indices=test_indices,
        all_samples=all_samples,
        traj_sampler_config=traj_sampler_config,
        max_seq_len=args.max_seq_len,
    )

    # Build S4_official model
    model_cfg = S4Config(
        vocab_size=vocab_size,
        num_states=MAX_STATES,        # always 8-state head, like transformer
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        # d_state, dropout, d_ff use defaults in S4Config
    )
    model = MooreS4(model_cfg)

    collator = ICLDataCollator(pad_token_id=pad_token)
    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=stage.epochs,
        device=args.device,
    )

    trainer = MooreICLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )

    # Train and collect history
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    for epoch in range(1, trainer_cfg.num_epochs + 1):
        train_loss = trainer._run_epoch(epoch)      
        val_loss = trainer.evaluate()

        # Accuracy on train + val
        train_acc = evaluate_model(trainer.model, trainer.train_loader, trainer.device)
        val_acc = evaluate_model(trainer.model, trainer.val_loader, trainer.device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(float(train_acc))
        history["val_accs"].append(float(val_acc))

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Final train and test evaluation
    print("\nEvaluating on training set...")
    final_train_acc = evaluate_model(trainer.model, trainer.train_loader, trainer.device)
    print(f"Train Accuracy: {final_train_acc.item():.4f}")

    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_acc = evaluate_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {test_acc.item():.4f}")

    # Save checkpoint and metrics for later plot generation
    log_dir = Path("checkpoints/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Checkpoint
    checkpoint_path = log_dir / f"s4_direct_{timestamp}.pt"
    torch.save(
        {
            "epoch": trainer_cfg.num_epochs,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "train_accs": history["train_accs"],
            "val_accs": history["val_accs"],
            "final_train_acc": final_train_acc.item(),
            "final_val_acc": history["val_accs"][-1],
            "final_test_acc": test_acc.item(),
            "model_config": {
                "vocab_size": model_cfg.vocab_size,
                "num_states": model_cfg.num_states,
                "d_model": model_cfg.d_model,
                "num_layers": model_cfg.num_layers,
                "max_seq_len": model_cfg.max_seq_len,
            },
            "training_config": {
                "batch_size": trainer_cfg.batch_size,
                "learning_rate": trainer_cfg.learning_rate,
                "num_epochs": trainer_cfg.num_epochs,
            },
        },
        checkpoint_path,
    )
    print(f"\n S4 checkpoint saved to: {checkpoint_path}")

    # Metrics (JSON)
    metrics = {
        "experiment": "s4_direct",
        "timestamp": timestamp,
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "num_states": model_cfg.num_states,
            "d_model": model_cfg.d_model,
            "num_layers": model_cfg.num_layers,
            "max_seq_len": model_cfg.max_seq_len,
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
            "train_accuracy": final_train_acc.item(),
            "val_accuracy": history["val_accs"][-1],
            "test_accuracy": test_acc.item(),
        },
    }
    metrics_path = log_dir / f"s4_direct_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"S4 metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()