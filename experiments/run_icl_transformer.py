"""
Entry point for training the Moore transformer in an ICL setting with curriculum learning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader  # type: ignore

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.models.moore_transformer import MooreTransformer, TransformerConfig
from src.training.icl_trainer import (
    ICLDataCollator,
    MooreICLTrainer,
    TrainingConfig,
    evaluate_model,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig
from src.fsm import MAX_STATES, MAX_ACTIONS


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum learning stage."""

    num_states: int
    min_actions_per_state: int
    max_actions_per_state: int
    num_samples: int
    epochs: int
    demo_length: Optional[int] = None  # If None, computed as base_len * log2(base_len)
    query_length: Optional[int] = None  # If None, computed as base_len * log2(base_len)
    cache_path: Optional[Path] = None  # Will be set automatically if None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer for Moore ICL with curriculum learning")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage 1 (3 states, 3 actions)")
    return parser.parse_args()


def train_stage(
    stage: CurriculumStage,
    model: MooreTransformer,
    args: argparse.Namespace,
    vocab_size: int,
    pad_token: int,
) -> MooreTransformer:
    """Train on a single curriculum stage."""
    print(f"\n{'='*60}")
    print(f"Curriculum Stage: {stage.num_states} states, "
          f"{stage.min_actions_per_state}-{stage.max_actions_per_state} actions")
    print(f"{'='*60}\n")

    # Create sampler config for this stage
    sampler_config = TrajectorySamplerConfig(
        num_states=stage.num_states,
        min_actions_per_state=stage.min_actions_per_state,
        max_actions_per_state=stage.max_actions_per_state,
        seed=42,
    )

    # Create dataset config
    cache_path = stage.cache_path
    if cache_path is None:
        cache_path = Path(f"data/icl_dataset_stage_{stage.num_states}states_{stage.max_actions_per_state}actions.pt")

    dataset_cfg = ICLDatasetConfig(
        num_samples=stage.num_samples,
        sampler_config=sampler_config,
        max_seq_len=args.max_seq_len,
        cache_path=cache_path,
        demo_length=stage.demo_length,
        query_length=stage.query_length,
    )

    # Load or create samples
    all_samples = load_or_create_icl_samples(dataset_cfg)

    # Split dataset: 60% train, 20% val, 20% test
    train_size = int(0.6 * len(all_samples))
    val_size = int(0.2 * len(all_samples))
    test_size = len(all_samples) - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(all_samples)))

    train_dataset = MooreICLDataset(
        train_indices,
        all_samples,
        sampler_config,
        args.max_seq_len,
    )
    val_dataset = MooreICLDataset(
        val_indices,
        all_samples,
        sampler_config,
        args.max_seq_len,
    )
    test_dataset = MooreICLDataset(
        test_indices,
        all_samples,
        sampler_config,
        args.max_seq_len,
    )

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
    trainer.train()

    # Evaluate
    print("\nEvaluating on training set...")
    train_acc = evaluate_model(trainer.model, trainer.train_loader, trainer.device)
    print(f"Train Accuracy: {train_acc.item():.4f}")

    print("Evaluating on validation set...")
    val_acc = evaluate_model(trainer.model, trainer.val_loader, trainer.device)
    print(f"Validation Accuracy: {val_acc.item():.4f}")

    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_acc = evaluate_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {test_acc.item():.4f}\n")

    return trainer.model


def main():
    args = parse_args()

    # Fixed vocabulary sizes
    vocab_size = MAX_STATES + MAX_ACTIONS + 3  # states + actions + eos + query + pad
    pad_token = MAX_STATES + MAX_ACTIONS + 2

    # Create model with fixed 8-state output
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        num_states=MAX_STATES,  # Always 8
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
    )
    model = MooreTransformer(model_cfg)

    # Curriculum learning stages
    stages = []

    # Stage 1: 3 states, 3 actions
    if not args.skip_stage1:
        stages.append(CurriculumStage(
            num_states=3,
            min_actions_per_state=3,
            max_actions_per_state=3,
            num_samples=10_000,
            epochs=10,
            # demo_length and query_length default to None, will use base_len * log2(base_len)
        ))

    # Stage 2: 5 states, 6-8 actions
    stages.append(CurriculumStage(
        num_states=5,
        min_actions_per_state=4,
        max_actions_per_state=5,
        num_samples=10_000,
        epochs=3,
        # demo_length and query_length default to None, will use base_len * log2(base_len)
    ))

    # Train on each stage sequentially
    for stage in stages:
        model = train_stage(stage, model, args, vocab_size, pad_token)


if __name__ == "__main__":
    main()
