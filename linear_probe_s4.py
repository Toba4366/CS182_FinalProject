import torch
import torch.nn as nn
from s4_backbone import S4Backbone


class S4LinearProbe(nn.Module):
    """
    Frozen S4 backbone + trainable linear head.
    """

    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 128,
        seq_len: int = 256,
        out_dim: int = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = vocab_size

        # Backbone
        self.backbone = S4Backbone(
            vocab_size=vocab_size,
            d_model=d_model,
            seq_len=seq_len,
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Linear probe head
        self.head = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, input_ids):
        with torch.no_grad():
            h = self.backbone(input_ids)   # (B, T, d_model)

        logits = self.head(h)              # (B, T, out_dim)
        return logits
