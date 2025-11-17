import torch
import torch.nn as nn
from state_spaces.s4 import S4Layer   # Official S4 implementation


class S4Backbone(nn.Module):
    """
    Official S4 backbone 
    Produces hidden representations for linear probing.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        use_embedding: bool = True,
        s4_kwargs=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        if s4_kwargs is None:
            s4_kwargs = {}

        # Embedding layer
        self.use_embedding = use_embedding
        if use_embedding:
            self.token_embed = nn.Embedding(vocab_size, d_model)

        # S4 stack
        self.layers = nn.ModuleList([
            S4Layer(
                d_model=d_model,
                dropout=dropout,
                **s4_kwargs,
            )
            for _ in range(n_layers)
        ])

        # LayerNorm after stack
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        """
        input_ids: (B, T)
        returns: (B, T, d_model)
        """

        # Embed tokens
        if self.use_embedding:
            x = self.token_embed(input_ids)  # (B, T, d_model)
        else:
            x = input_ids

        # Apply each S4 layer with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x)           # (B, T, d_model)
            x = x + residual       # residual connection

        return self.norm(x)
