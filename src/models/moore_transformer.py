"""
Simple causal transformer for predicting Moore machine states.

The model consumes sequences of the form:
    S, A, S, A, ..., <eos>, S, A, ..., <eos>, S, A, <unk>, A, <unk>, ...
and is trained to predict the unknown states given the preceding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


@dataclass
class TransformerConfig:
    vocab_size: int
    num_states: int
    max_seq_len: int = 512
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0  # Base frequency for RoPE


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as described in RoFormer."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency matrix
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim) - used for device/dtype
            seq_len: Optional sequence length (uses x.shape[2] if not provided)
        
        Returns:
            Tuple of (cos, sin) tensors for applying rotations
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Create position indices on the same device as input
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        
        # Concatenate cos and sin for each position
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)
    
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-head attention implemented via scaled_dot_product_attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, rope: RotaryPositionEmbedding):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=(self.dropout_p if self.training else 0.0), is_causal=True
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Standard Transformer decoder block."""

    def __init__(self, config: TransformerConfig, rope: RotaryPositionEmbedding):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            rope=rope,
        )
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.ln1(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MooreTransformer(nn.Module):
    """A minimal decoder-only transformer specialised for state prediction."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Create RoPE for all attention layers
        self.rope = RotaryPositionEmbedding(
            dim=config.d_model // config.num_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=self.rope) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.num_states)

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
            targets:   (batch, seq_len) tensor of ground-truth tokens
            unknown_mask: boolean mask of shape (batch, seq_len) where True
                          indicates positions corresponding to unknown states.
        Returns:
            Tuple of (logits, optional loss)
        """
        bsz, seq_len = input_ids.size()
        assert (
            seq_len <= self.config.max_seq_len
        ), f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"

        # Token embeddings only - RoPE is applied in attention layers
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

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
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

