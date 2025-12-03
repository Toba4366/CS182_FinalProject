"""
Training utilities for the Moore machine ICL GRU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader, Dataset  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("ICL trainer requires PyTorch. Install via `pip install torch`.") from exc

from src.models.moore_gru import MooreGRU


class ICLDataCollator:
    """Pads variable-length sequences in a batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        target_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for idx, item in enumerate(batch):
            seq_len = item["input_ids"].size(0)
            input_ids[idx, :seq_len] = item["input_ids"]
            target_ids[idx, :seq_len] = item["target_ids"]
            loss_mask[idx, :seq_len] = item["loss_mask"]
            attention_mask[idx, :seq_len] = True

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_every: int = 1
    save_best: bool = True


class MooreGRUTrainer:
    """Trainer for Moore Machine ICL tasks using GRU."""

    def __init__(
        self,
        model: MooreGRU,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or TrainingConfig()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.best_val_loss = float("inf")
        
        # Track training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def train(self) -> Dict:
        """Train the model and return training history."""
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Calculate training accuracy
            train_acc = self._calculate_accuracy(self.train_loader)
            self.train_accs.append(train_acc)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            if self.val_loader and (epoch + 1) % self.config.eval_every == 0:
                val_loss = self._eval_epoch(self.val_loader)
                val_acc = self._calculate_accuracy(self.val_loader)
                
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if self.config.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print("  New best validation loss!")

        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

            self.optimizer.zero_grad()

            logits, loss = self.model(
                input_ids=input_ids, targets=target_ids, unknown_mask=loss_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _eval_epoch(self, loader: DataLoader) -> float:
        """Evaluate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                logits, loss = self.model(
                    input_ids=input_ids, targets=target_ids, unknown_mask=loss_mask
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _calculate_accuracy(self, loader: DataLoader) -> float:
        """Calculate token-level accuracy on the masked positions."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                logits, _ = self.model(
                    input_ids=input_ids, targets=target_ids, unknown_mask=loss_mask
                )

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Only consider masked positions
                masked_preds = predictions[loss_mask]
                masked_targets = target_ids[loss_mask]

                correct += (masked_preds == masked_targets).sum().item()
                total += loss_mask.sum().item()

        return correct / total if total > 0 else 0.0

    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """Evaluate the model and return (loss, accuracy)."""
        loader = loader or self.val_loader
        if loader is None:
            raise ValueError("No evaluation loader provided")

        loss = self._eval_epoch(loader)
        accuracy = self._calculate_accuracy(loader)
        return loss, accuracy

    def get_model_state(self) -> dict:
        """Return model state dict."""
        return self.model.state_dict()

    def load_model_state(self, state_dict: dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)
