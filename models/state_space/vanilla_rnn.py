"""
Vanilla RNN implementation for FSM sequence modeling.

This implements a basic RNN (Elman network) with:
- Token embedding
- Multi-layer RNN with tanh activation
- Optional dropout
- Linear output head
- Same interface as transformer and LSTM models for easy comparison
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNSequenceModel(nn.Module):
    """
    Vanilla RNN-based sequence model with:
    
    - Token embedding
    - Multi-layer RNN with tanh activation
    - Optional dropout between layers
    - Final linear head for prediction
    
    This is the simplest recurrent architecture - a baseline for comparing
    against LSTM, Transformer, and state-space models.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        out_dim: Optional[int] = None,  # None -> vocab_size
        tie_weights: bool = False,
        activation: str = "tanh",  # tanh or relu
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.activation = activation
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # RNN layers
        self.rnn_layers = nn.ModuleList()
        for i in range(n_layers):
            input_size = d_model if i == 0 else d_model
            self.rnn_layers.append(nn.Linear(input_size + d_model, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        if out_dim is None:
            out_dim = vocab_size
        self.out_dim = out_dim
        
        # Linear head
        self.head = nn.Linear(d_model, out_dim, bias=False)
        
        # Weight tying (only if dimensions match)
        if tie_weights and out_dim == vocab_size and d_model == d_model:
            self.head.weight = self.token_embed.weight
            
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following best practices for RNNs."""
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
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len)
        hidden: Optional[torch.Tensor] = None,  # (n_layers, batch, d_model)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through vanilla RNN.
        
        Args:
            input_ids: Token indices (batch, seq_len)
            targets: Target tokens for loss computation (batch, seq_len)
            hidden: Initial hidden state (n_layers, batch, d_model)
            
        Returns:
            logits: (batch, seq_len, out_dim)
            loss: Scalar loss if targets provided, else None
            new_hidden: Final hidden state (n_layers, batch, d_model)
        """
        B, T = input_ids.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size=B, device=input_ids.device)
        
        # Embed tokens
        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.dropout(x)
        
        # Process sequence step by step
        outputs = []
        current_hidden = hidden
        
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_model)
            
            # Process through RNN layers
            new_hidden = []
            layer_input = x_t
            
            for layer_idx, rnn_layer in enumerate(self.rnn_layers):
                h_prev = current_hidden[layer_idx]  # (B, d_model)
                
                # Concatenate input and previous hidden state
                rnn_input = torch.cat([layer_input, h_prev], dim=-1)  # (B, 2*d_model)
                
                # Apply linear transformation and activation
                h_new = rnn_layer(rnn_input)  # (B, d_model)
                
                if self.activation == "tanh":
                    h_new = torch.tanh(h_new)
                elif self.activation == "relu":
                    h_new = F.relu(h_new)
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
                
                # Apply dropout between layers (not on last layer output)
                if layer_idx < self.n_layers - 1:
                    h_new = self.dropout(h_new)
                
                new_hidden.append(h_new)
                layer_input = h_new
            
            current_hidden = torch.stack(new_hidden)  # (n_layers, B, d_model)
            outputs.append(layer_input)  # Use final layer output
        
        # Stack outputs for all time steps
        rnn_output = torch.stack(outputs, dim=1)  # (B, T, d_model)
        rnn_output = self.dropout(rnn_output)
        
        # Apply output head
        logits = self.head(rnn_output)  # (B, T, out_dim)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss, current_hidden
    
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        """Create zero-initialized hidden state."""
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros(
            self.n_layers,
            batch_size, 
            self.d_model,
            device=device
        )