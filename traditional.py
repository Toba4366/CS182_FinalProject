import math
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import scipy


# Optional: if you implement RoPE in models/transformers/utils/rope.py
try:
    from .utils.rope import RotaryEmbedding  # type: ignore
except ImportError:
    RotaryEmbedding = None


class MultiHeadSelfAttention(nn.Module):
    """
    Decoder-only multi-head self-attention with causal masking.
    Optionally supports RoPE (rotary positional embeddings) if provided.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: int = 10000,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        if use_rope and RotaryEmbedding is not None:
            self.rope = RotaryEmbedding(self.d_head, base=rope_base)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to q, k, v
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE if available
        if self.rope is not None:
            q, k = self.rope(q, k)  # expects (B, n_heads, T, d_head)

        # Use PyTorch 2 scaled_dot_product_attention (flash-attn backend capable)
        # is_causal=True handles the causal masking for us.
        # If you want to also support padding masks, pass attn_mask and set is_causal accordingly.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # typically None here; causal handled by is_causal
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )  # (B, n_heads, T, d_head)

        # Rearrange back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.o_proj(attn_output)
        return self.out_dropout(attn_output)


class FeedForward(nn.Module):
    """
    Standard Transformer MLP block.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    One decoder-only transformer block:
    - Multi-head self-attention (causal)
    - MLP
    - Pre-LN style
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.mlp = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        h = self.ln1(x)
        h = self.attn(h)
        x = x + h

        # MLP with residual
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class CausalTransformer(nn.Module):
    """
    Decoder-only transformer with:
    - token embedding
    - (optional) learned positional embedding
    - N transformer blocks
    - final layer norm
    - linear head for prediction

    This is the model you can plug directly into your FSM experiments,
    then add a linear layer readout to states / actions / next-state labels.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        tie_weights: bool = True,
        out_dim: Optional[int] = None,  # None -> same as vocab_size
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        # If you're using RoPE, you often don't need absolute pos embeddings,
        # but we keep them here and you can disable them by zeroing or removing.
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rope=use_rope,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Linear head (your "linear layer at the end")
        if out_dim is None:
            out_dim = vocab_size
        self.head = nn.Linear(d_model, out_dim, bias=False)

        if tie_weights and out_dim == vocab_size:
            # classic LM head tying
            self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len) for LM-type training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            logits: (batch, seq_len, out_dim)
            loss: scalar (if targets provided), else None
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # Token + positional embeddings
        tok_emb = self.token_embed(input_ids)  # (B, T, d_model)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embed(pos_ids)  # (1, T, d_model)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Transformer blocks (causal masking is internal to attention via is_causal=True)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, out_dim)

        loss = None
        if targets is not None:
            # Cross-entropy over last dimension
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # optional ignore index
            )

        return logits, loss


# ------------------------------------------------------------------------------
# Optimizer + Scheduler + Training Utilities
# ------------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Standard AdamW with decoupled weight decay. You can do parameter grouping here
    if you want to exclude LayerNorm and biases from weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )
    return optimizer


def create_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to 10% of initial lr.
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay from 1.0 -> 0.1
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    map_location: Optional[str] = None,
) -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("step", 0)


# ------------------------------------------------------------------------------
# Example training step with gradient clipping
# ------------------------------------------------------------------------------

def train_step(
    model: CausalTransformer,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    batch: Tuple[torch.Tensor, torch.Tensor],
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> float:
    """
    One training step:
      - forward
      - compute loss
      - backward
      - gradient clipping
      - optimizer + scheduler step

    batch: (input_ids, targets)
    """
    model.train()
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, targets)
    loss.backward()

    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return float(loss.item())
