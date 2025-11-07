"""
Training loop implementation with AdamW optimizer for ICL experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Optional, Tuple, Callable
import wandb
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

from .models import SimpleTransformer, TransformerConfig
from .dataset import MooreMachineDataset, collate_fn


logger = logging.getLogger(__name__)


class ICLTrainer:
    """
    Trainer class for in-context learning of Moore machines.
    """
    
    def __init__(self,
                 model: SimpleTransformer,
                 train_dataset: MooreMachineDataset,
                 val_dataset: Optional[MooreMachineDataset] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 10000,
                 batch_size: int = 16,
                 eval_steps: int = 500,
                 save_steps: int = 1000,
                 gradient_clip_norm: float = 1.0,
                 device: Optional[str] = None,
                 save_dir: str = "./checkpoints",
                 use_wandb: bool = False,
                 wandb_project: str = "moore-icl",
                 seed: int = 42):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            learning_rate: Peak learning rate for AdamW
            weight_decay: Weight decay coefficient
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            batch_size: Training batch size
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoint saves
            gradient_clip_norm: Gradient clipping norm
            device: Device to train on
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            seed: Random seed
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.max_steps = max_steps
        self.use_wandb = use_wandb
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Setup save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer - only train parameters that require grad
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,  # Only optimize trainable parameters
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # Standard values for AdamW
            eps=1e-8
        )
        
        # Setup learning rate scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues with PyTorch
            pin_memory=torch.cuda.is_available()
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
        else:
            self.val_loader = None
        
        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "model_config": model.config.__dict__,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "seed": seed
                }
            )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dictionary containing input_ids, target_ids, attention_mask
            
        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = target_ids.view(-1)
        
        # Compute cross-entropy loss (ignore_index=-100 for masked tokens)
        loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one training step."""
        self.model.train()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.compute_loss(batch)
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = np.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': step,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved at step {step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {self.max_steps} steps")
        logger.info(f"Model has {self.model.count_parameters():,} parameters")
        
        # Log parameter freezing info if model supports it
        if hasattr(self.model, 'get_frozen_info'):
            frozen_info = self.model.get_frozen_info()
            logger.info(f"Parameter breakdown:")
            logger.info(f"  Total parameters: {frozen_info['total_parameters']:,}")
            logger.info(f"  Trainable parameters: {frozen_info['trainable_parameters']:,}")
            logger.info(f"  Frozen parameters: {frozen_info['frozen_parameters']:,}")
            logger.info(f"  Frozen percentage: {frozen_info['frozen_percentage']:.1f}%")
        
        logger.info(f"Training on device: {self.device}")
        
        self.model.train()
        
        # Create infinite iterator over training data
        train_iter = iter(self.train_loader)
        
        progress_bar = tqdm(range(self.max_steps), desc="Training")
        
        for step in progress_bar:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator when epoch ends
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            train_loss = self.train_step(batch)
            self.train_losses.append(train_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{train_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation
            if (step + 1) % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                
                if eval_metrics:
                    val_loss = eval_metrics['val_loss']
                    self.val_losses.append(val_loss)
                    
                    # Check if best model
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    logger.info(f"Step {step + 1}: train_loss={train_loss:.4f}, "
                              f"val_loss={val_loss:.4f}, val_ppl={eval_metrics['val_perplexity']:.2f}")
                    
                    # W&B logging
                    if self.use_wandb:
                        wandb.log({
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'val_perplexity': eval_metrics['val_perplexity'],
                            'learning_rate': self.scheduler.get_last_lr()[0],
                            'step': step + 1
                        })
                    
                    # Save checkpoint
                    if (step + 1) % self.save_steps == 0:
                        self.save_checkpoint(step + 1, is_best)
                
                self.model.train()  # Return to training mode
            
            # Regular W&B logging
            elif self.use_wandb and step % 100 == 0:
                wandb.log({
                    'train_loss': train_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'step': step + 1
                })
            
            self.global_step = step + 1
        
        # Final evaluation and save
        final_metrics = self.evaluate()
        if final_metrics:
            is_best = final_metrics['val_loss'] < self.best_val_loss
            self.save_checkpoint(self.max_steps, is_best)
        
        logger.info("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }