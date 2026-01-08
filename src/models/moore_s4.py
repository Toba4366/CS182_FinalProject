"""
S4-style state space model for predicting Moore machine states.

This is a simplified S4-inspired architecture: each block consists of
a depthwise 1D convolution (standing in for the SSM kernel) followed
by a position-wise MLP, with pre-layer normalization and residual
connections.

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
class S4Config:
    vocab_size: int
    num_states: int
    max_seq_len: int = 512
    d_model: int = 256
    d_state: int = 64          # not used explicitly, but kept for completeness
    num_layers: int = 2
    dropout: float = 0.1
    kernel_size: int = 16      # length of the SSM convolution kernel
    layer_norm_eps: float = 1e-5
    d_ff: int = 512            # MLP hidden dim


class S4Layer(nn.Module):
    """
    Simplified S4-style layer implemented as a depthwise 1D convolution
    over the sequence dimension.

    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Depthwise convolution: separate SSM per channel.
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,  # so we can slice back to original length
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, d_model, T)
        x_perm = x.transpose(1, 2)
        y = self.conv(x_perm)  # (B, d_model, T + kernel_size - 1)
        # Trim back to original length
        y = y[:, :, : x_perm.size(2)]
        # Back to (B, T, d_model)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        return y


class S4Block(nn.Module):
    """
    One S4-style block: pre-LN -> S4Layer -> residual, then
    pre-LN -> MLP -> residual.
    """

    def __init__(self, config: S4Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.s4 = S4Layer(
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
        h = self.s4(h)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class MooreS4(nn.Module):
    """S4-style sequence model specialised for Moore machine state prediction."""

    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Stacked S4 blocks
        self.blocks = nn.ModuleList([S4Block(config) for _ in range(config.num_layers)])
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
            targets: (batch, seq_len) tensor of ground-truth state ids
            unknown_mask: boolean mask (batch, seq_len) where True indicates
                          positions corresponding to unknown states to predict.
        Returns:
            Tuple of (logits, optional loss)
        """
        B, T = input_ids.shape
        assert (
            T <= self.config.max_seq_len
        ), f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Embed tokens
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)

        # S4 blocks
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


def create_moore_s4(
    vocab_size: int,
    num_states: int,
    max_seq_len: int = 512,
    d_model: int = 256,
    d_state: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    kernel_size: int = 16,
    d_ff: int = 512,
) -> MooreS4:
    """Factory function for creating a Moore S4-style model."""
    config = S4Config(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        dropout=dropout,
        kernel_size=kernel_size,
        d_ff=d_ff,
    )
    return MooreS4(config)
