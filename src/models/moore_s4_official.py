"""
Modified S4 model using the official S4 implementation

This model:
- Internally uses the official S4Block (FFTConv + S4 kernel) with HIPPO-style
  initialization, etc., as implemented in the `state-spaces/s4` repository.

Input / output interface matches the other Moore models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

# Import the official S4 block
from src.s4_official.models.s4.s4 import S4Block as OfficialS4Block


@dataclass
class S4Config:
    vocab_size: int         
    num_states: int         
    max_seq_len: int = 512  
    d_model: int = 256       
    d_state: int = 64       
    num_layers: int = 2  # Number of S4 blocks 
    dropout: float = 0.1     
    layer_norm_eps: float = 1e-5
    d_ff: int = 512       


class MooreS4Block(nn.Module):
    """
    One S4 block with pre-layer normalization and a position-wise MLP.

    Structure:
        x -> LN -> OfficialS4Block -> + residual ->
        -> LN -> MLP -> + residual
    """

    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Official S4 block
        self.s4 = OfficialS4Block(
            d_model=config.d_model,
            l_max=config.max_seq_len,
            dropout=config.dropout,
            transposed=False,
            tie_dropout=True, # tie_dropout=True is common, but not required
        )

        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)

        # S4 sub-layer
        h = self.ln1(x)
        # Official S4Block returns (y, state): we only need y.
        y, _ = self.s4(h)  # y: (B, T, d_model)
        x = x + y

        # Feedforward sub-layer
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class MooreS4(nn.Module):
    """S4-based model for Moore machine state prediction."""

    def __init__(self, config: S4Config):
        super().__init__()
        self.config = config

        # Token embeddin
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Stack of S4 blocks
        self.blocks = nn.ModuleList(
            [MooreS4Block(config) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output head for state prediction
        self.head = nn.Linear(config.d_model, config.num_states, bias=False)

        # Initialize only simple layers (leave S4 kernel init as implemented in official code)
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        unknown_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) tensor of token ids from MooreICLDataset
            targets: (batch, seq_len) tensor of ground-truth state ids
            unknown_mask: (batch, seq_len) bool tensor; True where loss is applied

        Returns:
            logits: (batch, seq_len, num_states)
            loss:   scalar tensor or None
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

        # Final norm + linear head
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, num_states)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # By convention in this repo, trainer passes `loss_mask` as 3rd positional arg,
            # which we call `unknown_mask` here.
            if unknown_mask is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                mask = unknown_mask

            flat_mask = mask.view(-1)
            flat_logits = logits.view(-1, self.config.num_states)
            flat_targets = targets.view(-1)

            if flat_mask.any():
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = torch.tensor(0.0, device=input_ids.device)

        return logits, loss

    def _init_weights(self, module: nn.Module):
        # Do NOT touch the internal S4 kernel parameters; just handle
        # simple linear / embedding layers on top.
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

def create_moore_s4(
    vocab_size: int,
    num_states: int,
    max_seq_len: int = 512,
    d_model: int = 256,
    d_state: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    d_ff: int = 512,
) -> MooreS4:
    """Factory function for creating a Moore S4 model with the official S4Block."""
    config = S4Config(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        dropout=dropout,
        d_ff=d_ff,
    )
    return MooreS4(config)