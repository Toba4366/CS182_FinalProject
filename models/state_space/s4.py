import torch
import torch.nn as nn
import torch.nn.functional as F

# Official S4 import
from extern_s4.models.s4.s4 import S4Block


# -------------------------------------------------------
#   S4 BLOCK WRAPPER
# -------------------------------------------------------
class S4Wrapper(nn.Module):
    def __init__(self, d_model, dropout=0.0, l_max=256):
        super().__init__()
        self.block = S4Block(
            d_model=d_model,
            dropout=dropout,
            l_max=l_max,    # required
        )

    def forward(self, x):
        y, _ = self.block(x)
        return y



# -------------------------------------------------------
#   S4 BACKBONE
# -------------------------------------------------------
class S4Backbone(nn.Module):
    def __init__(self, d_model=128, n_layers=4, dropout=0.0, s4_kwargs=None):
        super().__init__()
        if s4_kwargs is None:
            raise ValueError("Must pass s4_kwargs with l_max")

        l_max = s4_kwargs["l_max"]

        self.layers = nn.ModuleList([
        S4Wrapper(
            d_model=d_model,
            dropout=dropout,
            l_max=s4_kwargs["l_max"],   # explicitly
        )

            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


# -------------------------------------------------------
#   FULL LANGUAGE MODEL 
# -------------------------------------------------------
class S4LM(nn.Module):
    """
    Full language model using S4 backbone.
    Matches transformer interface.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        max_seq_len=512,
        dropout=0.0,
        s4_kwargs=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = S4Backbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            s4_kwargs={"l_max": max_seq_len},
        )

        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))[None, :, :]
        x = tok + pos

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
#   LINEAR PROBE
# -------------------------------------------------------
class S4LinearProbe(nn.Module):
    """
    Frozen S4 backbone + trainable linear probe head.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        max_seq_len=512,
        dropout=0.0,
        s4_kwargs=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = S4Backbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            s4_kwargs=s4_kwargs,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(T, device=input_ids.device))[None, :, :]
        x = tok + pos

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
