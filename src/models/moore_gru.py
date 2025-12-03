"""
GRU for predicting Moore machine states.

The model consumes sequences of the form:
    S, A, S, A, ..., <eos>, S, A, ..., <eos>, S, A, <unk>, A, <unk>, ...
and is trained to predict the unknown states given the preceding context.

GRU provides gating mechanisms (reset and update gates) as a middle ground
between vanilla RNN and LSTM complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


@dataclass
class GRUConfig:
    vocab_size: int
    num_states: int
    max_seq_len: int = 512
    d_model: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class MooreGRU(nn.Module):
    """A GRU specialised for Moore machine state prediction."""

    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Output head for state prediction
        output_dim = config.d_model * 2 if config.bidirectional else config.d_model
        self.head = nn.Linear(output_dim, config.num_states, bias=False)
        
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        unknown_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) tensor of token ids
            targets: (batch, seq_len) tensor of ground-truth tokens
            unknown_mask: boolean mask of shape (batch, seq_len) where True
                          indicates positions corresponding to unknown states.
        Returns:
            Tuple of (logits, optional loss)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)
        
        # Pass through GRU
        gru_output, _ = self.gru(x)  # (B, T, d_model) or (B, T, 2*d_model) if bidirectional
        gru_output = self.dropout(gru_output)
        
        # Apply output head
        logits = self.head(gru_output)  # (B, T, num_states)
        
        # Compute loss if targets provided
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            mask = (
                unknown_mask
                if unknown_mask is not None
                else torch.ones_like(targets, dtype=torch.bool)
            )

            # Cross entropy only on the masked (unknown) positions.
            flat_mask = mask.view(-1)
            flat_logits = logits.view(-1, self.config.num_states)
            flat_targets = targets.view(-1)
            
            if flat_mask.any():
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = torch.tensor(0.0, device=input_ids.device)

        return logits, loss

    def _init_weights(self, module: nn.Module):
        """Initialize weights following best practices for RNNs."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_moore_gru(
    vocab_size: int,
    num_states: int,
    max_seq_len: int = 512,
    d_model: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    bidirectional: bool = False,
) -> MooreGRU:
    """Factory function for creating a Moore GRU."""
    config = GRUConfig(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return MooreGRU(config)
