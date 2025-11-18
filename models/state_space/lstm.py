"""
LSTM implementation for FSM sequence modeling.

This implements an LSTM with:
- Token embedding
- Multi-layer LSTM with optional bidirectionality
- Optional dropout
- Linear output head
- Same interface as other models for easy comparison
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSequenceModel(nn.Module):
    """
    LSTM-based sequence model with:
    - Token embedding
    - Multi-layer LSTM
    - Optional bidirectionality
    - Final linear head for prediction
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
        hidden_size = d_model  # Can decouple if needed

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

        # Output dimension
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
        """Initialize weights following best practices for LSTMs."""
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
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            input_ids: Token indices (batch, seq_len)
            targets: Target tokens for loss computation (batch, seq_len)
            hidden: Initial (h_0, c_0) hidden state tuple
            
        Returns:
            logits: (batch, seq_len, out_dim)
            loss: Scalar loss if targets provided, else None
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
                ignore_index=-100,
            )

        return logits, loss, new_hidden

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        """Create zero-initialized hidden state for the LSTM."""
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