"""
Entry point for training S4_official with curriculum learning support.
This script supports training on simple vs complex FSMs for curriculum learning experiments.

"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train S4 (official) for Moore ICL with Curriculum Learning"
    )

    # Standard training arguments (aligned with LSTM/RNN scripts)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")

    # Curriculum learning arguments
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["simple", "complex"],
        required=True,
        help="simple (3 states, 3 actions) or complex (5 states, 4-5 actions)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        required=True,
        help="Name for saving checkpoint (e.g., 'curriculum_stage1', 'curriculum_stage2')",
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="Load model from previous checkpoint (for stage 2)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save lightweight model checkpoints",
    )

    return parser.parse_args()


def get_dataset_config(dataset_type: str, num_samples: int, max_seq_len: int) -> ICLDatasetConfig:
    """
    Create dataset configuration based on type.
    Mirrors LSTM/RNN cache naming so all models can share the same cached datasets.
    """
    if dataset_type == "simple":
        sampler_config = TrajectorySamplerConfig(
            num_states=3,
            min_actions_per_state=3,
            max_actions_per_state=3,
            seed=42,
        )
        cache_path = Path("data/icl_dataset_stage_3states_3actions.pt")

    elif dataset_type == "complex":
        sampler_config = TrajectorySamplerConfig(
            num_states=5,
            min_actions_per_state=4,
            max_actions_per_state=5,
            seed=42,
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

    print(f"CURRICULUM LEARNING: S4 (official) - {args.dataset_type.upper()} FSM")
    print(f"Checkpoint name: {args.checkpoint_name}")
    if args.load_from:
        print(f"Loading from: {args.load_from}")
    print("=" * 60)

    # Fixed vocabulary (same logic as your S4 direct script)
    vocab_size = MAX_STATES + MAX_ACTIONS + 3
    pad_token = MAX_STATES + MAX_ACTIONS + 2

    # Dataset config
    dataset_cfg = get_dataset_config(args.dataset_type, args.num_samples, args.max_seq_len)

    print(
        f"Sampler config: {dataset_cfg.traj_sampler_config.num_states} states, "
        f"{dataset_cfg.traj_sampler_config.min_actions_per_state}-"
        f"{dataset_cfg.traj_sampler_config.max_actions_per_state} actions"
    )

    # Load dataset samples
    all_samples = load_or_create_icl_samples(dataset_cfg)
    print(f"Loaded {len(all_samples)} samples")

    # Fixed split indice
    # Safeguard bounds if num_samples differs
    n = len(all_samples)
    train_end = min(6000, n)
    val_end = min(8000, n)
    test_end = min(10000, n)

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, test_end))

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

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model config (official S4)
    model_cfg = S4Config(
        vocab_size=vocab_size,
        num_states=MAX_STATES,  # keep consistent head convention
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
    )
    model = MooreS4(model_cfg)

    # Load from previous checkpoint (stage 2)
    if args.load_from:
        checkpoint_path = Path(args.load_from)
        if checkpoint_path.exists():
            print(f"🔄 Loading model from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            model_dict = model.state_dict()
            compatible_dict = {}

            for name, param in state_dict.items():
                if name in model_dict and param.shape == model_dict[name].shape:
                    compatible_dict[name] = param
                    if not args.no_verbose:
                        print(f"Loaded: {name} {tuple(param.shape)}")
                else:
                    if not args.no_verbose:
                        expected = (
                            tuple(model_dict[name].shape) if name in model_dict else "missing"
                        )
                        print(f"Skipped: {name} {tuple(param.shape)} (incompatible with {expected})")

            model.load_state_dict(compatible_dict, strict=False)
            print(f"Loaded {len(compatible_dict)}/{len(state_dict)} layers from previous stage")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting fresh")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training setup
    collator = ICLDataCollator(pad_token_id=pad_token)

    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
    )

    trainer = MooreICLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )

    # --------------------------
    # Custom loop (Option 2)
    # --------------------------
    print(f"Starting training for {args.epochs} epochs...")

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    for epoch in range(1, trainer_cfg.num_epochs + 1):
        train_loss = trainer._run_epoch(epoch)
        val_loss = trainer.evaluate()

        train_acc = evaluate_model(trainer.model, trainer.train_loader, trainer.device)
        val_acc = evaluate_model(trainer.model, trainer.val_loader, trainer.device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(float(train_acc))
        history["val_accs"].append(float(val_acc))

        if not args.no_verbose:
            print(
                f"[Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

    # Final evaluation (match LSTM/RNN style)
    print("\n Final Evaluation:")
    final_val_acc = evaluate_model(trainer.model, trainer.val_loader, trainer.device)
    print(f"Validation Accuracy: {final_val_acc.item():.4f}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    final_test_acc = evaluate_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {final_test_acc.item():.4f}")


    # Save lightweight checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lightweight_path = save_dir / f"s4_{args.checkpoint_name}.pt"
    torch.save(trainer.model.state_dict(), lightweight_path)
    print(f"Model saved to {lightweight_path}")

    # Save full checkpoint and metrics
    log_dir = Path("checkpoints/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_ckpt_path = log_dir / f"s4_{args.checkpoint_name}_{timestamp}.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "train_accs": history["train_accs"],
            "val_accs": history["val_accs"],
            "final_val_acc": final_val_acc.item(),
            "final_test_acc": final_test_acc.item(),
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
            "curriculum_info": {
                "dataset_type": args.dataset_type,
                "checkpoint_name": args.checkpoint_name,
                "loaded_from": args.load_from,
            },
        },
        full_ckpt_path,
    )
    print(f"Full checkpoint saved to: {full_ckpt_path}")

    metrics = {
        "experiment": f"s4_curriculum_{args.dataset_type}",
        "checkpoint_name": args.checkpoint_name,
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
            "val_accuracy": final_val_acc.item(),
            "test_accuracy": final_test_acc.item(),
        },
    }

    metrics_path = log_dir / f"s4_{args.checkpoint_name}_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"COMPLETED: S4 (official) - {args.dataset_type.upper()} FSM")
    print(f"Final Validation Accuracy: {final_val_acc.item():.4f}")
    print(f"Final Test Accuracy: {final_test_acc.item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()