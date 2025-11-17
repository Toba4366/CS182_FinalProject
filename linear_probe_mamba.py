import torch
import torch.nn as nn
from mamba_backbone import MambaBackbone


class MambaLinearProbe(nn.Module):
    """
    Frozen Mamba backbone + trainable linear classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        out_dim: int = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = vocab_size

        self.backbone = MambaBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
        )

        # Freeze backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Linear probe
        self.head = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, input_ids):
        with torch.no_grad():
            h = self.backbone(input_ids)

        return self.head(h)
