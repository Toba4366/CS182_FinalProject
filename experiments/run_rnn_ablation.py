"""
Vanilla RNN ablation study with hyperparameter search and early stopping.

Tests different AdamW hyperparameter settings, uses early stopping,
and runs multiple times for error bars. Trains directly on 5s5a dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import copy

import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

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
from src.fsm import MAX_STATES, MAX_ACTIONS
from src.fsm.baseline_evaluator import evaluate_deterministic_baseline


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum learning stage."""

    num_states: int
    min_actions_per_state: int
    max_actions_per_state: int
    num_samples: int
    max_epochs: int
    early_stop_threshold: Optional[float] = None  # Early stop if val_loss < threshold
    epochs_after_threshold: int = 0  # Train N more epochs after reaching threshold
    demo_length: Optional[int] = None
    query_length: Optional[int] = None
    cache_path: Optional[Path] = None
    learning_rate: Optional[float] = None  # If None, uses the base learning rate from trainer_cfg


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float


class EarlyStoppingRNNTrainer(MooreVanillaRNNTrainer):
    """Vanilla RNN Trainer with early stopping and detailed metrics tracking."""

    def __init__(self, *args, checkpoint_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics: List[EpochMetrics] = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_since_threshold = 0
        self.threshold_reached = False
        self.checkpoint_path = checkpoint_path

    def train_with_early_stopping(
        self,
        max_epochs: int,
        early_stop_threshold: Optional[float] = None,
        epochs_after_threshold: int = 0,
        track_train_metrics: bool = True,
    ) -> List[EpochMetrics]:
        """
        Train with early stopping based on validation loss.
        
        Args:
            max_epochs: Maximum number of epochs
            early_stop_threshold: If val_loss < threshold, trigger early stopping logic
            epochs_after_threshold: Train N more epochs after reaching threshold
            track_train_metrics: If False, skip calculating train accuracy (faster for HP search)
        
        Returns:
            List of epoch metrics
        """
        for epoch in range(1, max_epochs + 1):
            train_loss = self._run_epoch(epoch)
            val_loss = self.evaluate()
            
            if track_train_metrics:
                train_acc = self._calculate_accuracy(self.train_loader)
            else:
                train_acc = 0.0  # Skip expensive train accuracy calculation
            
            val_acc = self._calculate_accuracy(self.val_loader)

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
            )
            self.epoch_metrics.append(metrics)

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                # Save checkpoint if path provided
                if self.checkpoint_path is not None:
                    torch.save(self.model.state_dict(), self.checkpoint_path)

            # Early stopping logic
            if early_stop_threshold is not None:
                if val_loss < early_stop_threshold:
                    if not self.threshold_reached:
                        self.threshold_reached = True
                        self.epochs_since_threshold = 0
                        print(f"  ✓ Validation loss below threshold ({early_stop_threshold:.4f})")
                    
                    self.epochs_since_threshold += 1
                    if self.epochs_since_threshold > epochs_after_threshold:
                        print(f"  ✓ Early stopping: trained {epochs_after_threshold} epochs after threshold")
                        break
                elif self.threshold_reached:
                    # Continue counting even if loss goes back up
                    self.epochs_since_threshold += 1

            if track_train_metrics:
                print(
                    f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            # Ensure model is on the correct device (important for MPS)
            self.model = self.model.to(self.device)
            print(f"  ✓ Loaded best model (val_loss={self.best_val_loss:.4f})")

        return self.epoch_metrics


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_baseline_accuracies(
    all_samples: List[Dict[str, object]],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
) -> Dict[str, float]:
    """Calculate baseline accuracies for train, val, and test sets."""
    print("\nCalculating baseline accuracies...")
    train_baseline = evaluate_deterministic_baseline(all_samples, train_indices, num_demos=3)
    val_baseline = evaluate_deterministic_baseline(all_samples, val_indices, num_demos=3)
    test_baseline = evaluate_deterministic_baseline(all_samples, test_indices, num_demos=3)
    
    baseline_accs = {
        "train_baseline_accuracy": train_baseline["accuracy"],
        "val_baseline_accuracy": val_baseline["accuracy"],
        "test_baseline_accuracy": test_baseline["accuracy"],
    }
    
    print(f"  Train baseline accuracy: {train_baseline['accuracy']:.4f}")
    print(f"  Validation baseline accuracy: {val_baseline['accuracy']:.4f}")
    print(f"  Test baseline accuracy: {test_baseline['accuracy']:.4f}\n")
    
    return baseline_accs


def train_stage(
    stage: CurriculumStage,
    model: MooreVanillaRNN,
    trainer_cfg: TrainingConfig,
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    checkpoint_path: Optional[Path] = None,
    calculate_baseline: bool = False,
    track_train_metrics: bool = True,
) -> Tuple[MooreVanillaRNN, List[EpochMetrics], Optional[Dict[str, float]]]:
    """Train on a single stage with early stopping."""
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage.num_states} states, "
          f"{stage.min_actions_per_state}-{stage.max_actions_per_state} actions")
    print(f"Demo length: {stage.demo_length}, Query length: {stage.query_length}")
    print(f"Max epochs: {stage.max_epochs}, Early stop threshold: {stage.early_stop_threshold}")
    print(f"{'='*60}\n")

    # Create sampler config
    sampler_config = TrajectorySamplerConfig(
        num_states=stage.num_states,
        min_actions_per_state=stage.min_actions_per_state,
        max_actions_per_state=stage.max_actions_per_state,
        seed=42,
    )

    # Create dataset config
    cache_path = stage.cache_path
    if cache_path is None:
        cache_path = Path(
            f"data/icl_dataset_stage_{stage.num_states}s_{stage.max_actions_per_state}a"
            f"_demo{stage.demo_length}_query{stage.query_length}.pt"
        )

    dataset_cfg = ICLDatasetConfig(
        num_samples=stage.num_samples,
        traj_sampler_config=sampler_config,
        max_seq_len=max_seq_len,
        cache_path=cache_path,
        demo_length=stage.demo_length,
        query_length=stage.query_length,
    )

    # Load or create samples
    all_samples = load_or_create_icl_samples(dataset_cfg)

    # Split dataset: Fixed split (train: 0-6000, val: 6000-8000, test: 8000-10000)
    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))

    train_dataset = MooreICLDataset(
        train_indices,
        all_samples,
        sampler_config,
        max_seq_len,
    )
    val_dataset = MooreICLDataset(
        val_indices,
        all_samples,
        sampler_config,
        max_seq_len,
    )

    # Calculate baseline accuracies if requested (only once per dataset)
    baseline_accuracies = None
    if calculate_baseline:
        baseline_accuracies = calculate_baseline_accuracies(
            all_samples, train_indices, val_indices, test_indices
        )

    collator = ICLDataCollator(pad_token_id=pad_token)
    
    # Use stage-specific learning rate if provided, otherwise use base learning rate
    stage_lr = stage.learning_rate if stage.learning_rate is not None else trainer_cfg.learning_rate
    if stage.learning_rate is not None:
        print(f"Using stage-specific learning rate: {stage_lr}")
    
    # Create stage-specific trainer config
    stage_trainer_cfg = TrainingConfig(
        batch_size=trainer_cfg.batch_size,
        learning_rate=stage_lr,
        weight_decay=trainer_cfg.weight_decay,
        num_epochs=1,  # Will be controlled by early stopping
        device=trainer_cfg.device,
        verbose=False,
    )
    
    # Ensure model is on the correct device before creating trainer
    if trainer_cfg.device:
        device = torch.device(trainer_cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    
    trainer = EarlyStoppingRNNTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=stage_trainer_cfg,
        checkpoint_path=checkpoint_path,
    )

    # Train with early stopping
    epoch_metrics = trainer.train_with_early_stopping(
        max_epochs=stage.max_epochs,
        early_stop_threshold=stage.early_stop_threshold,
        epochs_after_threshold=stage.epochs_after_threshold,
        track_train_metrics=track_train_metrics,
    )

    return trainer.model, epoch_metrics, baseline_accuracies


def test_query_lengths(
    model: MooreVanillaRNN,
    stage_config: CurriculumStage,
    batch_size: int,
    device: Optional[str],
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    query_lengths: List[int],
) -> Dict[int, Dict[str, float]]:
    """Test model on different query lengths."""
    results = {}
    
    for query_len in query_lengths:
        print(f"\n{'='*60}")
        print(f"Testing query length: {query_len}")
        print(f"{'='*60}\n")

        sampler_config = TrajectorySamplerConfig(
            num_states=stage_config.num_states,
            min_actions_per_state=stage_config.min_actions_per_state,
            max_actions_per_state=stage_config.max_actions_per_state,
            seed=42,
        )

        cache_path = Path(
            f"data/icl_dataset_test_{stage_config.num_states}s_{stage_config.max_actions_per_state}a"
            f"_demo{stage_config.demo_length}_query{query_len}.pt"
        )

        dataset_cfg = ICLDatasetConfig(
            num_samples=2000,
            traj_sampler_config=sampler_config,
            max_seq_len=max_seq_len,
            cache_path=cache_path,
            demo_length=stage_config.demo_length,
            query_length=query_len,
        )

        all_samples = load_or_create_icl_samples(dataset_cfg)
        test_indices = list(range(len(all_samples)))

        test_dataset = MooreICLDataset(
            test_indices,
            all_samples,
            sampler_config,
            max_seq_len,
        )

        collator = ICLDataCollator(pad_token_id=pad_token)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # Determine test device
        if device:
            test_device = torch.device(device)
        elif torch.cuda.is_available():
            test_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            test_device = torch.device("mps")
        else:
            test_device = torch.device("cpu")
        
        # Ensure model is on the correct device (important for MPS)
        model = model.to(test_device)
        
        test_acc = evaluate_vanilla_rnn_model(model, test_loader, test_device)

        results[query_len] = {
            "test_accuracy": test_acc.item(),
        }

        print(f"Query length {query_len}: Test Acc={test_acc.item():.4f}")
    
    return results


def run_hyperparameter_search(
    stages: List[CurriculumStage],
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    d_model: int,
    num_layers: int,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    baseline_accuracies: Optional[List[Dict[str, float]]] = None,
) -> Dict:
    """Run hyperparameter search for AdamW optimizer."""
    print(f"\n{'='*70}")
    print(f"Hyperparameter Search")
    print(f"{'='*70}\n")

    # Use MAX_STATES for model output head (FSM can use any state IDs from 0 to MAX_STATES-1)
    # even if individual stages have fewer states
    max_num_states = MAX_STATES

    # Hyperparameter grid
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    weight_decays = [0.0, 1e-5, 1e-4]
    batch_sizes = [16, 24, 32]

    best_val_acc = -1.0
    best_config = None
    best_final_metrics = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            # First try without weight decay
            wd_zero_succeeded = False
            
            for wd in weight_decays:
                # Skip weight decay > 0 if wd=0.0 failed to learn
                if wd > 0.0 and not wd_zero_succeeded:
                    print(f"\nSkipping: lr={lr}, weight_decay={wd}, batch_size={batch_size}")
                    print(f"  (weight_decay=0.0 failed to learn for this lr/batch_size)")
                    continue
                
                print(f"\nTrying: lr={lr}, weight_decay={wd}, batch_size={batch_size}")

                # Create model
                model_cfg = VanillaRNNConfig(
                    vocab_size=vocab_size,
                    num_states=max_num_states,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    num_layers=num_layers,
                    dropout=0.1,
                    activation="tanh",
                )
                model = MooreVanillaRNN(model_cfg)

                base_trainer_cfg = TrainingConfig(
                    batch_size=batch_size,
                    learning_rate=lr,
                    weight_decay=wd,
                    num_epochs=1,
                    device=device,
                    verbose=False,
                )

                # Train through curriculum
                # IMPORTANT: Use the same model object throughout all stages for curriculum learning
                all_stage_metrics = []
                any_stage_failed = False
                for stage_idx, stage in enumerate(stages):
                    checkpoint_path = None
                    if checkpoint_dir is not None:
                        checkpoint_path = checkpoint_dir / f"hp_search_lr{lr}_wd{wd}_bs{batch_size}_stage{stage_idx}.pt"
                    # Pass the same model object to continue training (curriculum learning)
                    model, stage_metrics, _ = train_stage(
                        stage, model, base_trainer_cfg, vocab_size, pad_token, max_seq_len, 
                        checkpoint_path, calculate_baseline=False, track_train_metrics=False
                    )
                    all_stage_metrics.append(stage_metrics)
                    
                    # Check if this stage failed to learn (val_loss >= threshold)
                    if stage.early_stop_threshold is not None:
                        best_stage_val_loss = min(m.val_loss for m in stage_metrics)
                        if best_stage_val_loss >= stage.early_stop_threshold:
                            any_stage_failed = True
                            print(f"  ⚠ Stage {stage_idx + 1} failed to reach threshold ({stage.early_stop_threshold:.4f})")
                            print(f"     Best val_loss: {best_stage_val_loss:.4f}")
                            print(f"     Skipping remaining stages for this hyperparameter configuration")
                            # Add empty metrics for skipped stages
                            for remaining_stage in stages[stage_idx + 1:]:
                                all_stage_metrics.append([])
                            break

                # Track if wd=0.0 succeeded (for skipping wd > 0.0)
                # A configuration succeeds only if all stages complete successfully
                if wd == 0.0:
                    if not any_stage_failed:
                        wd_zero_succeeded = True
                        print(f"  ✓ weight_decay=0.0 succeeded (all stages completed), will try weight_decay > 0.0")
                    else:
                        wd_zero_succeeded = False
                        print(f"  ✗ weight_decay=0.0 failed (stage failed), skipping weight_decay > 0.0 for this lr/batch_size")

                # Use final validation accuracy from last completed stage
                # If stage 1 failed, use stage 1's final metrics
                final_val_acc = 0.0
                if all_stage_metrics:
                    # Find the last non-empty stage metrics
                    for stage_metrics in reversed(all_stage_metrics):
                        if stage_metrics:
                            final_val_acc = stage_metrics[-1].val_acc
                            break

                if final_val_acc > best_val_acc:
                    best_val_acc = final_val_acc
                    best_config = {
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "batch_size": batch_size,
                    }
                    best_final_metrics = all_stage_metrics

                print(f"Final Val Acc: {final_val_acc:.4f}")

    print(f"\n{'='*70}")
    print(f"Best config: {best_config}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*70}\n")

    return {
        "best_config": best_config,
        "best_val_acc": best_val_acc,
        "best_metrics": [[asdict(m) for m in stage] for stage in best_final_metrics] if best_final_metrics else None,
        "baseline_accuracies": baseline_accuracies,  # Use pre-calculated baselines
    }


def run_experiment(
    stages: List[CurriculumStage],
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    d_model: int,
    num_layers: int,
    best_hp_config: Dict,
    query_lengths: List[int],
    output_dir: Path,
    run_id: int,
    device: Optional[str] = None,
) -> Dict:
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*70}")
    print(f"Run {run_id + 1}/5")
    print(f"{'='*70}\n")

    # Use MAX_STATES for model output head (FSM can use any state IDs from 0 to MAX_STATES-1)
    # even if individual stages have fewer states
    max_num_states = MAX_STATES

    # Create model
    model_cfg = VanillaRNNConfig(
        vocab_size=vocab_size,
        num_states=max_num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=0.1,
        activation="tanh",
    )
    model = MooreVanillaRNN(model_cfg)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    base_trainer_cfg = TrainingConfig(
        batch_size=best_hp_config["batch_size"],
        learning_rate=best_hp_config["learning_rate"],
        weight_decay=best_hp_config["weight_decay"],
        num_epochs=1,
        device=device,
        verbose=False,
    )

    # Determine device
    if device:
        train_device = torch.device(device)
    elif torch.cuda.is_available():
        train_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        train_device = torch.device("mps")
    else:
        train_device = torch.device("cpu")
    
    # Ensure model is on the correct device
    model = model.to(train_device)
    
    # Train through curriculum
    # IMPORTANT: Use the same model object throughout all stages for curriculum learning
    all_stage_metrics = []
    run_checkpoint_dir = output_dir / f"run_{run_id + 1}_checkpoints"
    run_checkpoint_dir.mkdir(exist_ok=True)
    
    for stage_idx, stage in enumerate(stages):
        checkpoint_path = run_checkpoint_dir / f"stage_{stage_idx}_best.pt"
        # Pass the same model object to continue training (curriculum learning)
        model, stage_metrics, _ = train_stage(
            stage, model, base_trainer_cfg, vocab_size, pad_token, max_seq_len, checkpoint_path, calculate_baseline=False
        )
        # Ensure model stays on device after training
        model = model.to(train_device)
        all_stage_metrics.append(stage_metrics)
        
        # Check if first stage failed to learn (val_loss >= 0.05)
        if stage_idx == 0 and stage.early_stop_threshold is not None:
            best_stage_val_loss = min(m.val_loss for m in stage_metrics)
            if best_stage_val_loss >= stage.early_stop_threshold:
                print(f"\n{'='*70}")
                print(f"⚠ Stage 1 failed to reach threshold ({stage.early_stop_threshold:.4f})")
                print(f"   Best val_loss: {best_stage_val_loss:.4f}")
                print(f"   Skipping Stage 2 - model unable to learn Stage 1")
                print(f"{'='*70}\n")
                # Add empty metrics for skipped stages
                for remaining_stage in stages[stage_idx + 1:]:
                    all_stage_metrics.append([])
                break

    # Get last stage for testing
    last_stage = stages[-1] if stages else None
    if last_stage is None:
        raise ValueError("No stages defined")

    # Load test dataset
    sampler_config = TrajectorySamplerConfig(
        num_states=last_stage.num_states,
        min_actions_per_state=last_stage.min_actions_per_state,
        max_actions_per_state=last_stage.max_actions_per_state,
        seed=42,
    )

    cache_path = last_stage.cache_path
    if cache_path is None:
        cache_path = Path(
            f"data/icl_dataset_stage_{last_stage.num_states}s_{last_stage.max_actions_per_state}a"
            f"_demo{last_stage.demo_length}_query{last_stage.query_length}.pt"
        )

    dataset_cfg = ICLDatasetConfig(
        num_samples=10_000,
        traj_sampler_config=sampler_config,
        max_seq_len=max_seq_len,
        cache_path=cache_path,
        demo_length=last_stage.demo_length,
        query_length=last_stage.query_length,
    )

    all_samples = load_or_create_icl_samples(dataset_cfg)
    test_indices = list(range(8000, 10_000))

    test_dataset = MooreICLDataset(
        test_indices,
        all_samples,
        sampler_config,
        max_seq_len,
    )

    collator = ICLDataCollator(pad_token_id=pad_token)
    test_loader = DataLoader(
        test_dataset,
        batch_size=base_trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Determine test device
    if device:
        test_device = torch.device(device)
    elif torch.cuda.is_available():
        test_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        test_device = torch.device("mps")
    else:
        test_device = torch.device("cpu")
    
    # Ensure model is on the correct device before testing
    model = model.to(test_device)
    
    # Teacher forcing evaluation
    initial_test_acc = evaluate_vanilla_rnn_model(model, test_loader, test_device)
    print(f"\nInitial Test Accuracy: {initial_test_acc.item():.4f}")

    # Test on different query lengths
    query_results = test_query_lengths(
        model, last_stage, base_trainer_cfg.batch_size, device, vocab_size, pad_token, max_seq_len, query_lengths
    )

    return {
        "run_id": run_id,
        "num_parameters": num_params,
        "hyperparameters": best_hp_config,
        "stage_metrics": [[asdict(m) for m in stage] for stage in all_stage_metrics],
        "initial_test_accuracy": initial_test_acc.item(),
        "query_length_results": query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Vanilla RNN ablation study")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--skip-hp-search", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for error bars")
    parser.add_argument("--output-dir", type=str, default="results/rnn_ablation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Set random seeds for reproducibility (model initialization)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Fixed vocabulary
    vocab_size = MAX_STATES + MAX_ACTIONS + 3
    pad_token = MAX_STATES + MAX_ACTIONS + 2

    # Fixed architecture: 2 layers, hidden dimension 128
    d_model = 128
    num_layers = 2

    # Single stage: Train directly on 5s5a dataset (no curriculum learning)
    stages = [
        CurriculumStage(
            num_states=3,
            min_actions_per_state=3,
            max_actions_per_state=3,
            num_samples=10_000,
            max_epochs=40,
            early_stop_threshold=0.05,
            epochs_after_threshold=2,
            demo_length=90,
            query_length=100,
        ),
    ]

    query_lengths = [200, 300, 400, 500, 600]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Calculate baseline accuracies once for each curriculum stage dataset
    print(f"\n{'='*70}")
    print(f"Calculating Baseline Accuracies for Curriculum Stages")
    print(f"{'='*70}\n")
    
    stage_baseline_accuracies = []
    for stage_idx, stage in enumerate(stages):
        print(f"Stage {stage_idx + 1}: {stage.num_states} states, {stage.max_actions_per_state} actions")
        
        sampler_config = TrajectorySamplerConfig(
            num_states=stage.num_states,
            min_actions_per_state=stage.min_actions_per_state,
            max_actions_per_state=stage.max_actions_per_state,
            seed=42,
        )

        cache_path = stage.cache_path
        if cache_path is None:
            cache_path = Path(
                f"data/icl_dataset_stage_{stage.num_states}s_{stage.max_actions_per_state}a"
                f"_demo{stage.demo_length}_query{stage.query_length}.pt"
            )

        dataset_cfg = ICLDatasetConfig(
            num_samples=stage.num_samples,
            traj_sampler_config=sampler_config,
            max_seq_len=args.max_seq_len,
            cache_path=cache_path,
            demo_length=stage.demo_length,
            query_length=stage.query_length,
        )

        all_samples = load_or_create_icl_samples(dataset_cfg)
        train_indices = list(range(0, 6000))
        val_indices = list(range(6000, 8000))
        test_indices = list(range(8000, 10_000))

        baseline_accs = calculate_baseline_accuracies(
            all_samples, train_indices, val_indices, test_indices
        )
        stage_baseline_accuracies.append(baseline_accs)

    # Calculate baseline for query length tests
    last_stage = stages[-1]
    query_length_baselines = {}
    for query_len in query_lengths:
        sampler_config = TrajectorySamplerConfig(
            num_states=last_stage.num_states,
            min_actions_per_state=last_stage.min_actions_per_state,
            max_actions_per_state=last_stage.max_actions_per_state,
            seed=42,
        )

        cache_path = Path(
            f"data/icl_dataset_test_{last_stage.num_states}s_{last_stage.max_actions_per_state}a"
            f"_demo{last_stage.demo_length}_query{query_len}.pt"
        )

        dataset_cfg = ICLDatasetConfig(
            num_samples=2000,
            traj_sampler_config=sampler_config,
            max_seq_len=args.max_seq_len,
            cache_path=cache_path,
            demo_length=last_stage.demo_length,
            query_length=query_len,
        )

        all_samples = load_or_create_icl_samples(dataset_cfg)
        test_indices = list(range(len(all_samples)))

        test_baseline = evaluate_deterministic_baseline(all_samples, test_indices, num_demos=3)
        query_length_baselines[query_len] = test_baseline["accuracy"]
        print(f"Query length {query_len} baseline accuracy: {test_baseline['accuracy']:.4f}")

    # Hyperparameter search
    if not args.skip_hp_search:
        hp_checkpoint_dir = run_dir / "hp_search_checkpoints"
        hp_checkpoint_dir.mkdir(exist_ok=True)
        
        hp_results = run_hyperparameter_search(
            stages,
            vocab_size,
            pad_token,
            args.max_seq_len,
            d_model,
            num_layers,
            device=args.device,
            checkpoint_dir=hp_checkpoint_dir,
            baseline_accuracies=stage_baseline_accuracies,
        )
        best_hp_config = hp_results["best_config"]
        print(f"\nBest hyperparameters: {best_hp_config}")
    else:
        # Use default config if skipping HP search
        best_hp_config = {
            "learning_rate": 5e-3,
            "weight_decay": 0.0,
            "batch_size": 24,
        }
        print(f"\nSkipping hyperparameter search, using: {best_hp_config}")

    # Run multiple experiments for error bars
    all_runs = []
    for run_id in range(args.num_runs):
        run_result = run_experiment(
            stages,
            vocab_size,
            pad_token,
            args.max_seq_len,
            d_model,
            num_layers,
            best_hp_config,
            query_lengths,
            run_dir,
            run_id,
            device=args.device,
        )
        all_runs.append(run_result)
        
        # Save individual run results
        run_json_path = run_dir / f"run_{run_id + 1}.json"
        with open(run_json_path, "w") as f:
            json.dump(run_result, f, indent=2)
        print(f"✓ Saved run {run_id + 1} results to {run_json_path}")

    # Create summary
    summary = {
        "architecture": {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_parameters": all_runs[0]["num_parameters"] if all_runs else 0,
        },
        "hyperparameters": best_hp_config,
        "curriculum_stages": [asdict(s) for s in stages],
        "baseline_accuracies": {
            "curriculum_stages": [
                {
                    f"stage_{i}": accs
                    for i, accs in enumerate(stage_baseline_accuracies)
                }
            ],
            "query_lengths": query_length_baselines,
        },
        "runs": all_runs,
    }

    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to {summary_path}")

    # Print final statistics
    print(f"\n{'='*70}")
    print(f"Final Results Summary")
    print(f"{'='*70}")
    print(f"Architecture: d_model={d_model}, num_layers={num_layers}")
    print(f"Parameters: {all_runs[0]['num_parameters']:,}")
    print(f"Best hyperparameters: {best_hp_config}")
    
    if all_runs:
        initial_test_accs = [r["initial_test_accuracy"] for r in all_runs]
        print(f"\nInitial Test Accuracy: {sum(initial_test_accs)/len(initial_test_accs):.4f} ± {torch.std(torch.tensor(initial_test_accs)).item():.4f}")
        
        print(f"\nQuery Length Extrapolation:")
        for query_len in query_lengths:
            accs = [r["query_length_results"][query_len]["test_accuracy"] for r in all_runs]
            mean_acc = sum(accs) / len(accs)
            std_acc = torch.std(torch.tensor(accs)).item()
            baseline_acc = query_length_baselines.get(query_len, 0.0)
            print(f"  Query {query_len}: {mean_acc:.4f} ± {std_acc:.4f} (baseline: {baseline_acc:.4f})")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

