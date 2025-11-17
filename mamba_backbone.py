import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBackbone(nn.Module):
    """
    Official Mamba backbone 
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        use_embedding: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.use_embedding = use_embedding
        if use_embedding:
            self.token_embed = nn.Embedding(vocab_size, d_model)

        # Stacked Mamba blocks
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        """
        input_ids: (B, T)
        returns: (B, T, d_model)
        """

        if self.use_embedding:
            x = self.token_embed(input_ids)
        else:
            x = input_ids

        # Apply stacked Mamba blocks with residuals
        for blk in self.blocks:
            residual = x
            x = blk(x)
            x = x + residual

        return self.norm(x)
