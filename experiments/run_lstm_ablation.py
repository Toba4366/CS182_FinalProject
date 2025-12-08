"""
LSTM ablation study with hyperparameter search and early stopping.

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
from src.models.moore_lstm import MooreLSTM, LSTMConfig
from src.training.lstm_trainer import (
    ICLDataCollator,
    MooreLSTMTrainer,
    TrainingConfig,
    evaluate_lstm_model,
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


class EarlyStoppingLSTMTrainer(MooreLSTMTrainer):
    """LSTM Trainer with early stopping and detailed metrics tracking."""

    def __init__(self, *args, checkpoint_path: Optional[Path] = None, use_lr_schedule: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics: List[EpochMetrics] = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_since_threshold = 0
        self.threshold_reached = False
        self.checkpoint_path = checkpoint_path
        self.use_lr_schedule = use_lr_schedule
        # For LR schedule: start at 1e-2, end at 1e-3
        if self.use_lr_schedule:
            self.initial_lr = 1e-2
            self.final_lr = 1e-3
            # Update optimizer to use initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr
        else:
            self.initial_lr = self.config.learning_rate
            self.final_lr = self.config.learning_rate
        self.current_epoch = 0

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
            self.current_epoch = epoch
            
            # Update learning rate schedule if enabled
            if self.use_lr_schedule:
                # Exponential decay from initial_lr to final_lr over max_epochs
                # Formula: lr = initial_lr * (final_lr / initial_lr) ^ (epoch / max_epochs)
                progress = epoch / max_epochs
                decay_rate = self.final_lr / self.initial_lr
                current_lr = self.initial_lr * (decay_rate ** progress)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            
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
                lr_info = f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}" if self.use_lr_schedule else ""
                print(
                    f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}{lr_info}"
                )
            else:
                lr_info = f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}" if self.use_lr_schedule else ""
                print(
                    f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{lr_info}"
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
    model: MooreLSTM,
    trainer_cfg: TrainingConfig,
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    checkpoint_path: Optional[Path] = None,
    calculate_baseline: bool = False,
    track_train_metrics: bool = True,
) -> Tuple[MooreLSTM, List[EpochMetrics], Optional[Dict[str, float]]]:
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
        use_absorbing_state=True,  # Use FSMs with 1 absorbing state
    )

    # Create dataset config
    cache_path = stage.cache_path
    if cache_path is None:
        cache_path = Path(
            f"data/icl_dataset_stage_{stage.num_states}s_{stage.max_actions_per_state}a"
            f"_demo{stage.demo_length}_query{stage.query_length}_absorbing.pt"
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

    # Split dataset: Fixed split to match run_icl_lstm.py (train: 0-6000, val: 6000-8000, test: 8000-10000)
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
    
    # Enable LR schedule if stage has learning_rate >= 1e-2 (schedule from 1e-2 to 1e-3)
    use_lr_schedule = (stage.learning_rate is not None and stage.learning_rate >= 1e-2) or \
                      (stage.learning_rate is None and stage_lr >= 1e-2)
    
    trainer = EarlyStoppingLSTMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=stage_trainer_cfg,
        checkpoint_path=checkpoint_path,
        use_lr_schedule=use_lr_schedule,
    )

    # Train with early stopping
    epoch_metrics = trainer.train_with_early_stopping(
        max_epochs=stage.max_epochs,
        early_stop_threshold=stage.early_stop_threshold,
        epochs_after_threshold=stage.epochs_after_threshold,
        track_train_metrics=track_train_metrics,
    )

    return trainer.model, epoch_metrics, baseline_accuracies


def calculate_query_length_baselines(
    stage_config: CurriculumStage,
    vocab_size: int,
    pad_token: int,
    max_seq_len: int,
    query_lengths: List[int],
) -> Dict[int, Dict[str, float]]:
    """Calculate baseline accuracies for different query lengths."""
    baseline_results = {}
    
    for query_len in query_lengths:
        print(f"\n{'='*60}")
        print(f"Calculating baseline for query length: {query_len}")
        print(f"{'='*60}\n")

        sampler_config = TrajectorySamplerConfig(
            num_states=stage_config.num_states,
            min_actions_per_state=stage_config.min_actions_per_state,
            max_actions_per_state=stage_config.max_actions_per_state,
            seed=42,
            use_absorbing_state=True,  # Use FSMs with 1 absorbing state
        )

        cache_path = Path(
            f"data/icl_dataset_test_{stage_config.num_states}s_{stage_config.max_actions_per_state}a"
            f"_demo{stage_config.demo_length}_query{query_len}_absorbing.pt"
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

        baseline_result = evaluate_deterministic_baseline(all_samples, test_indices, num_demos=3)
        baseline_results[query_len] = {
            "baseline_accuracy": baseline_result["accuracy"],
        }

        print(f"Query length {query_len}: Baseline Acc={baseline_result['accuracy']:.4f}")

    return baseline_results


def test_query_lengths(
    model: MooreLSTM,
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
        print(f"Testing on query length: {query_len}")
        print(f"{'='*60}\n")

        # Create test dataset with specified query length
        sampler_config = TrajectorySamplerConfig(
            num_states=stage_config.num_states,
            min_actions_per_state=stage_config.min_actions_per_state,
            max_actions_per_state=stage_config.max_actions_per_state,
            seed=42,
            use_absorbing_state=True,  # Use FSMs with 1 absorbing state
        )

        cache_path = Path(
            f"data/icl_dataset_test_{stage_config.num_states}s_{stage_config.max_actions_per_state}a"
            f"_demo{stage_config.demo_length}_query{query_len}_absorbing.pt"
        )

        dataset_cfg = ICLDatasetConfig(
            num_samples=2000,  # Smaller for testing
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
        
        test_acc = evaluate_lstm_model(model, test_loader, test_device)

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

    # Get maximum number of states across all stages
    max_num_states = max(stage.num_states for stage in stages)

    # Hyperparameter grid
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    weight_decays = [0.0, 1e-5, 1e-4]
    batch_sizes = [16, 32]

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
                model_cfg = LSTMConfig(
                    vocab_size=vocab_size,
                    num_states=max_num_states,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    num_layers=num_layers,
                    dropout=0.1,
                    bidirectional=False,
                )
                model = MooreLSTM(model_cfg)

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

    # Get maximum number of states across all stages
    max_num_states = max(stage.num_states for stage in stages)

    # Create model
    model_cfg = LSTMConfig(
        vocab_size=vocab_size,
        num_states=max_num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=0.1,
        bidirectional=False,
    )
    model = MooreLSTM(model_cfg)

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
        # Ensure model stays on device after training (model.to() returns same object if already on device)
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

    # Test on initial test set (from last completed stage)
    # Find the last stage that was actually trained
    last_completed_stage_idx = 0
    for idx, stage_metrics in enumerate(all_stage_metrics):
        if stage_metrics:
            last_completed_stage_idx = idx
    
    last_stage = stages[last_completed_stage_idx]
    sampler_config = TrajectorySamplerConfig(
        num_states=last_stage.num_states,
        min_actions_per_state=last_stage.min_actions_per_state,
        max_actions_per_state=last_stage.max_actions_per_state,
        seed=42,
        use_absorbing_state=True,  # Use FSMs with 1 absorbing state
    )

    cache_path = Path(
        f"data/icl_dataset_stage_{last_stage.num_states}s_{last_stage.max_actions_per_state}a"
        f"_demo{last_stage.demo_length}_query{last_stage.query_length}_absorbing.pt"
    )

    dataset_cfg = ICLDatasetConfig(
        num_samples=last_stage.num_samples,
        traj_sampler_config=sampler_config,
        max_seq_len=max_seq_len,
        cache_path=cache_path,
        demo_length=last_stage.demo_length,
        query_length=last_stage.query_length,
    )

    all_samples = load_or_create_icl_samples(dataset_cfg)
    # Use fixed split to match run_icl_lstm.py
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
    initial_test_acc = evaluate_lstm_model(model, test_loader, test_device)
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
    parser = argparse.ArgumentParser(description="LSTM ablation study")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--skip-hp-search", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for error bars")
    parser.add_argument("--output-dir", type=str, default="results/lstm_ablation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Set random seeds for reproducibility (model initialization)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Fixed vocabulary
    vocab_size = MAX_STATES + MAX_ACTIONS + 3
    pad_token = MAX_STATES + MAX_ACTIONS + 2

    # Fixed architecture: 2 layers, hidden dimension 64
    d_model = 128
    num_layers = 2

    # Curriculum stages (matching transformer ablation)
    stages = [
        # CurriculumStage(
        #     num_states=3,
        #     min_actions_per_state=3,
        #     max_actions_per_state=3,
        #     num_samples=10_000,
        #     max_epochs=20,
        #     early_stop_threshold=0.1,
        #     epochs_after_threshold=1,  # Train 1 more epoch after threshold
        #     demo_length=30,
        #     query_length=30,
        #     learning_rate=1e-3,
        # ),
        CurriculumStage(
            num_states=5,
            min_actions_per_state=5,
            max_actions_per_state=5,
            num_samples=10_000,
            max_epochs=10,
            early_stop_threshold=0.05,
            epochs_after_threshold=4,  # Train 2 more epochs after threshold
            demo_length=90,
            query_length=100,
            learning_rate=1e-2,
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
        # Use fixed split to match run_icl_lstm.py
        train_indices = list(range(0, 6000))
        val_indices = list(range(6000, 8000))
        test_indices = list(range(8000, 10_000))

        baseline_accs = calculate_baseline_accuracies(
            all_samples, train_indices, val_indices, test_indices
        )
        stage_baseline_accuracies.append(baseline_accs)

    # Calculate baseline accuracies for query length test datasets
    print(f"\n{'='*70}")
    print(f"Calculating Baseline Accuracies for Query Length Tests")
    print(f"{'='*70}\n")
    
    last_stage = stages[-1]
    query_length_baselines = calculate_query_length_baselines(
        last_stage, vocab_size, pad_token, args.max_seq_len, query_lengths
    )

    # Hyperparameter search
    if not args.skip_hp_search:
        hp_checkpoint_dir = run_dir / "hp_search_checkpoints"
        hp_checkpoint_dir.mkdir(exist_ok=True)
        hp_results = run_hyperparameter_search(
            stages, vocab_size, pad_token, args.max_seq_len, d_model, num_layers, args.device, hp_checkpoint_dir, stage_baseline_accuracies
        )
        best_hp = hp_results["best_config"]
        
        # Save HP search results
        with open(run_dir / "hyperparameter_search.json", "w") as f:
            json.dump(hp_results, f, indent=2)
    else:
        # Use default config
        best_hp = {"learning_rate": 1e-2, "weight_decay": 0.0, "batch_size": 24}
        print(f"Using default hyperparameters: {best_hp}")

    # Run multiple times for error bars
    run_results = []
    best_run_id = 0
    best_final_val_acc = -1.0
    
    for run_id in range(args.num_runs):
        run_result = run_experiment(
            stages,
            vocab_size,
            pad_token,
            args.max_seq_len,
            d_model,
            num_layers,
            best_hp,
            query_lengths,
            run_dir,
            run_id,
            args.device,
        )
        
        # Merge baseline accuracies into query results
        for query_len in query_lengths:
            if query_len in run_result["query_length_results"] and query_len in query_length_baselines:
                run_result["query_length_results"][query_len]["baseline_accuracy"] = query_length_baselines[query_len]["baseline_accuracy"]
        
        run_results.append(run_result)

        # Save individual run
        with open(run_dir / f"run_{run_id + 1}.json", "w") as f:
            json.dump(run_result, f, indent=2)

        # Track best run (by final validation accuracy from last completed stage)
        # Find the last non-empty stage metrics
        final_val_acc = 0.0
        for stage_metrics in reversed(run_result["stage_metrics"]):
            if stage_metrics:
                final_val_acc = stage_metrics[-1]["val_acc"]
                break
        
        if final_val_acc > best_final_val_acc:
            best_final_val_acc = final_val_acc
            best_run_id = run_id

    # Load and save best model
    print(f"\n{'='*70}")
    print(f"Best run: {best_run_id + 1} (val_acc={best_final_val_acc:.4f})")
    print(f"{'='*70}\n")

    # Get maximum number of states across all stages
    max_num_states = max(stage.num_states for stage in stages)

    # Recreate and load best model from checkpoint
    model_cfg = LSTMConfig(
        vocab_size=vocab_size,
        num_states=max_num_states,
        max_seq_len=args.max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=0.1,
        bidirectional=False,
    )
    best_model = MooreLSTM(model_cfg)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load the best model from the last completed stage of the best run
    # Find which stage was actually completed
    best_run_result = run_results[best_run_id]
    last_completed_stage_idx = 0
    for idx, stage_metrics in enumerate(best_run_result["stage_metrics"]):
        if stage_metrics:
            last_completed_stage_idx = idx
    
    best_run_checkpoint_dir = run_dir / f"run_{best_run_id + 1}_checkpoints"
    best_stage_checkpoint = best_run_checkpoint_dir / f"stage_{last_completed_stage_idx}_best.pt"
    
    if best_stage_checkpoint.exists():
        # Load state dict and ensure it's on the correct device
        state_dict = torch.load(best_stage_checkpoint, map_location=device)
        best_model.load_state_dict(state_dict)
        best_model = best_model.to(device)
        print(f"✓ Loaded best model from checkpoint: {best_stage_checkpoint}")
    else:
        print(f"⚠ Warning: Checkpoint not found, saving config only")
    
    # Save best model
    best_model_path = run_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "model_config": asdict(model_cfg),
        "hyperparameters": best_hp,
        "num_parameters": run_results[best_run_id]["num_parameters"],
        "best_run_id": best_run_id,
        "best_val_acc": best_final_val_acc,
    }, best_model_path)
    print(f"✓ Best model saved to: {best_model_path}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "architecture": {
            "d_model": d_model,
            "num_layers": num_layers,
        },
        "num_parameters": run_results[0]["num_parameters"],  # Same for all runs
        "config": {
            "max_seq_len": args.max_seq_len,
            "num_runs": args.num_runs,
            "query_lengths": query_lengths,
        },
        "curriculum_stages": [asdict(s) for s in stages],
        "best_hyperparameters": best_hp,
        "best_run_id": best_run_id,
        "best_val_acc": best_final_val_acc,
        "baseline_accuracies": {
            "curriculum_stages": {
                f"stage_{i}": accs for i, accs in enumerate(stage_baseline_accuracies)
            } if stage_baseline_accuracies else {},
            "query_lengths": query_length_baselines,
        },
        "runs": run_results,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Results saved to: {run_dir}")
    print(f"Model parameters: {run_results[0]['num_parameters']:,}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

