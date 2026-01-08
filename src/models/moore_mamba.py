"""
Mamba-style selective state space model for predicting Moore machine states.

This is a simplified Mamba-inspired architecture. Each block uses:
  - a depthwise convolution over the sequence (SSM-like dynamics),
  - an input-dependent gate that modulates the SSM output,
  - a position-wise MLP,
with pre-layer normalization and residual connections.

The model consumes sequences of the form:
    S, A, S, A, ..., <eos>, S, A, ..., <eos>, S, A, <unk>, A, <unk>, ...
and is trained to predict the unknown states given the preceding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


@dataclass
class MambaConfig:
    vocab_size: int
    num_states: int
    max_seq_len: int = 512
    d_model: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    kernel_size: int = 16
    layer_norm_eps: float = 1e-5
    d_ff: int = 512


class MambaLayer(nn.Module):
    """
    Simplified Mamba layer: depthwise convolution + input-dependent gating.

    Input / output shapes:
        x: (B, T, d_model)
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Depthwise conv as SSM backbone
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )

        # Input-dependent gate
        self.gate_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, C = x.shape

        # SSM-style conv over sequence
        x_perm = x.transpose(1, 2)   # (B, C, T)
        y = self.conv(x_perm)        # (B, C, T + k - 1)
        y = y[:, :, :T]              # trim to original length
        y = y.transpose(1, 2)        # (B, T, C)

        # Input-dependent gating
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, C)
        out = gate * y + (1.0 - gate) * x       # selective mixing

        out = self.dropout(out)
        return out


class MambaBlock(nn.Module):
    """
    One Mamba-style block: pre-LN -> MambaLayer -> residual,
    then pre-LN -> MLP -> residual.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mamba = MambaLayer(
            d_model=config.d_model,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.mamba(h)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class MooreMamba(nn.Module):
    """Mamba-style sequence model specialised for Moore machine state prediction."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Stacked Mamba blocks
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output head for state prediction
        self.head = nn.Linear(config.d_model, config.num_states, bias=False)

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
            targets:   (batch, seq_len) tensor of ground-truth state ids
            unknown_mask: boolean mask (batch, seq_len) where True indicates
                          positions corresponding to unknown states to predict.
        Returns:
            Tuple of (logits, optional loss)
        """
        B, T = input_ids.shape
        assert (
            T <= self.config.max_seq_len
        ), f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, num_states)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            mask = (
                unknown_mask
                if unknown_mask is not None
                else torch.ones_like(targets, dtype=torch.bool)
            )

            flat_mask = mask.view(-1)
            flat_logits = logits.view(-1, self.config.num_states)
            flat_targets = targets.view(-1)

            if flat_mask.any():
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = torch.tensor(0.0, device=input_ids.device)

        return logits, loss

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def create_moore_mamba(
    vocab_size: int,
    num_states: int,
    max_seq_len: int = 512,
    d_model: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    kernel_size: int = 16,
    d_ff: int = 512,
) -> MooreMamba:
    """Factory function for creating a Moore Mamba-style model."""
    config = MambaConfig(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        d_ff=d_ff,
    )
    return MooreMamba(config)
