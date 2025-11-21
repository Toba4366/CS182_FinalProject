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
        # Force d_model == l_max for this S4 implementation
        self.l_max = min(d_model, l_max)
        self.d_model = self.l_max  # Must be equal
        
        self.block = S4Block(
            d_model=self.d_model,
            dropout=dropout,
            l_max=self.l_max,
        )
        
        # Projection layers if we need to change dimensions
        self.input_proj = nn.Linear(d_model, self.d_model) if d_model != self.d_model else nn.Identity()
        self.output_proj = nn.Linear(self.d_model, d_model) if d_model != self.d_model else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        
        # Project input to S4 dimension
        x = self.input_proj(x)
        
        # Handle sequence length
        if L > self.l_max:
            # Process in chunks
            outputs = []
            for i in range(0, L, self.l_max):
                chunk = x[:, i:i+self.l_max, :]
                if chunk.size(1) < self.l_max:
                    # Pad last chunk
                    padding = torch.zeros(B, self.l_max - chunk.size(1), chunk.size(-1), 
                                        device=x.device, dtype=x.dtype)
                    chunk = torch.cat([chunk, padding], dim=1)
                
                chunk_out, _ = self.block(chunk)
                # Remove padding
                if i + self.l_max > L:
                    chunk_out = chunk_out[:, :L-i, :]
                outputs.append(chunk_out)
            y = torch.cat(outputs, dim=1)
        elif L < self.l_max:
            # Pad sequence
            padding = torch.zeros(B, self.l_max - L, x.size(-1), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            y, _ = self.block(x_padded)
            y = y[:, :L, :]  # Remove padding
        else:
            y, _ = self.block(x)
        
        # Project back to original dimension
        y = self.output_proj(y)
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
        
        # S4 internal dimension - use a reasonable size that works
        if s4_kwargs and 'l_max' in s4_kwargs:
            s4_dim = min(64, s4_kwargs['l_max'])  # Cap at 64 for stability
        else:
            s4_dim = min(64, d_model)  # Cap at 64 for stability

        # token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.backbone = S4Backbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            s4_kwargs={"l_max": s4_dim},
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
