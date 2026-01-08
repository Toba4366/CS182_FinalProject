"""
Training utilities for the Moore machine ICL Vanilla RNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader, Dataset  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("ICL trainer requires PyTorch. Install via `pip install torch`.") from exc

from src.models.moore_vanilla_rnn import MooreVanillaRNN


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
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 3
    device: Optional[str] = None
    verbose: bool = True  # Print detailed training info


class MooreVanillaRNNTrainer:
    """Thin training loop wrapper for the Vanilla RNN."""

    def __init__(
        self,
        model: MooreVanillaRNN,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collator: ICLDataCollator,
        config: TrainingConfig,
    ):
        self.config = config
        self.device = (
            torch.device(config.device)
            if config.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.collator = collator

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def train(self) -> Dict[str, List[float]]:
        """Train the model and return training history."""
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._run_epoch(epoch)
            val_loss = self.evaluate()
            
            # Calculate accuracies
            train_acc = self._calculate_accuracy(self.train_loader)
            val_acc = self._calculate_accuracy(self.val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(
                f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for step, batch in enumerate(self.train_loader, 1):
            batch = self._move_to_device(batch)
            _, loss = self.model(
                batch["input_ids"],
                targets=batch["target_ids"],
                unknown_mask=batch["loss_mask"],
            )

            if loss is None:
                continue

            # Verbose logging for first batch of each epoch
            if self.config.verbose and step == 1:
                print(f"\n🔍 [Epoch {epoch}] Detailed Batch Info:")
                print(f"   Input IDs shape: {batch['input_ids'].shape}")
                print(f"   Input IDs (first sample): {batch['input_ids'][0]}")
                print(f"   Target IDs (first sample): {batch['target_ids'][0]}")
                print(f"   Loss mask (first sample): {batch['loss_mask'][0]}")
                print(f"   Loss: {loss.item():.4f}")
                print(f"   Unknown positions: {batch['loss_mask'][0].sum().item()} out of {batch['loss_mask'][0].shape[0]}")
                print("   " + "="*60)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if step % 50 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

        return total_loss / max(1, total_steps)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            _, loss = self.model(
                batch["input_ids"],
                targets=batch["target_ids"],
                unknown_mask=batch["loss_mask"],
            )
            if loss is None:
                continue
            total_loss += loss.item()
            total_steps += 1

        return total_loss / max(1, total_steps)

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}
    
    @torch.no_grad()
    def _calculate_accuracy(self, dataloader: DataLoader) -> float:
        """Calculate accuracy on a dataset."""
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        
        for batch in dataloader:
            batch = self._move_to_device(batch)
            logits, _ = self.model(batch["input_ids"])
            predictions = logits.argmax(dim=-1)
            
            mask = batch["loss_mask"]
            targets = batch["target_ids"]
            
            total_correct += ((predictions == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
        
        return total_correct / max(1, total_tokens)


@torch.no_grad()
def evaluate_vanilla_rnn_model(
    model: MooreVanillaRNN,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    total_correct = torch.tensor(0, device=device)
    total_tokens = torch.tensor(0, device=device)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch["input_ids"])
        predictions = logits.argmax(dim=-1)

        mask = batch["loss_mask"]
        targets = batch["target_ids"]

        total_correct += ((predictions == targets) & mask).sum()
        total_tokens += mask.sum()

    return (total_correct.float() / total_tokens.float()).cpu()