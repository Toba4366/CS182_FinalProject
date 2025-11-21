import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# -------------------------------------------------------
#   MAMBA BACKBONE
# -------------------------------------------------------
class MambaBackbone(nn.Module):
    """
    Pure Mamba feature extractor.
    Input:  [B, T, d_model]
    Output: [B, T, d_model]
    """

    def __init__(self, d_model: int = 128, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            residual = x
            x = blk(x)
            x = x + residual
        return self.norm(x)


# -------------------------------------------------------
#   FULL LANGUAGE MODEL (TRAINABLE)
# -------------------------------------------------------
class MambaLM(nn.Module):
    """
    Full LM: embeddings → Mamba → LM head
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))[None, :, :]
        x = tok + pos

        # mamba backbone
        x = self.backbone(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss


# -------------------------------------------------------
#   LINEAR PROBE MODEL
# -------------------------------------------------------
class MambaLinearProbe(nn.Module):
    """
    Frozen Mamba backbone + trainable linear head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = MambaBackbone(d_model=d_model, n_layers=n_layers)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))[None, :, :]
        x = tok + pos

        # frozen mamba backbone
        with torch.no_grad():
            h = self.backbone(x)

        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
