import math
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class LSTMSequenceModel(nn.Module):
    """
    LSTM-based sequence model with:

    - token embedding
    - multi-layer LSTM
    - optional bidirectionality
    - final linear head (for prediction)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        out_dim: Optional[int] = None,  # None -> vocab_size
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        hidden_size = d_model  # you can decouple if you want

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,  # inputs are (B, T, C)
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Output dimension (e.g., vocab_size, or number of FSM states)
        if out_dim is None:
            out_dim = vocab_size

        self.out_dim = out_dim

        # Linear head on top of LSTM outputs
        lstm_output_dim = hidden_size * self.num_directions
        self.head = nn.Linear(lstm_output_dim, out_dim, bias=False)

        if tie_weights and out_dim == vocab_size and not bidirectional and hidden_size == d_model:
            # Only safe to tie weights when shapes match
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
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len) for LM-style training
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            logits: (batch, seq_len, out_dim)
            loss: scalar (if targets provided), else None
            new_hidden: (h_n, c_n) LSTM hidden state tuple
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.dropout(x)

        # LSTM forward
        # x_out: (B, T, hidden_size * num_directions)
        x_out, new_hidden = self.lstm(x, hidden)

        x_out = self.dropout(x_out)

        # Linear head at every time step
        logits = self.head(x_out)  # (B, T, out_dim)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # optional
            )

        return logits, loss, new_hidden

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        """
        Create zero-initialized hidden state for the LSTM.
        """
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(
            self.n_layers * self.num_directions,
            batch_size,
            self.d_model,
            device=device,
        )
        c0 = torch.zeros(
            self.n_layers * self.num_directions,
            batch_size,
            self.d_model,
            device=device,
        )
        return (h0, c0)


# ------------------------------------------------------------------------------
# Optimizer + Scheduler + Training Utilities (mirroring transformer file)
# ------------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Standard AdamW with decoupled weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name or "ln" in name:
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


def train_step(
    model: LSTMSequenceModel,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    batch: Tuple[torch.Tensor, torch.Tensor],
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
    """
    One training step:
      - forward
      - compute loss
      - backward
      - gradient clipping
      - optimizer + scheduler step

    batch: (input_ids, targets)
    Returns:
      loss_value, new_hidden
    """
    model.train()
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    if hidden is None:
        hidden = model.init_hidden(batch_size=input_ids.size(0), device=device)

    optimizer.zero_grad(set_to_none=True)
    _, loss, new_hidden = model(input_ids, targets, hidden)
    loss.backward()

    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return float(loss.item()), new_hidden
