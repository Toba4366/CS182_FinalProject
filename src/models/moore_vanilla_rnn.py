"""
Vanilla RNN for predicting Moore machine states.

The model consumes sequences of the form:
    S, A, S, A, ..., <eos>, S, A, ..., <eos>, S, A, <unk>, A, <unk>, ...
and is trained to predict the unknown states given the preceding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


@dataclass
class VanillaRNNConfig:
    vocab_size: int
    num_states: int
    max_seq_len: int = 512
    d_model: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "tanh"  # tanh or relu


class MooreVanillaRNN(nn.Module):
    """A vanilla RNN specialised for Moore machine state prediction."""

    def __init__(self, config: VanillaRNNConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # RNN layers
        self.rnn_layers = nn.ModuleList()
        for i in range(config.num_layers):
            input_size = config.d_model if i == 0 else config.d_model
            self.rnn_layers.append(nn.Linear(input_size + config.d_model, config.d_model))
        
        # Output head for state prediction
        self.head = nn.Linear(config.d_model, config.num_states, bias=False)
        
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        unknown_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) tensor of token ids
            targets: (batch, seq_len) tensor of ground-truth tokens
            unknown_mask: boolean mask of shape (batch, seq_len) where True
                          indicates positions corresponding to unknown states.
        Returns:
            Tuple of (logits, optional loss)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)
        
        # Initialize hidden state
        hidden = self._init_hidden(batch_size=B, device=input_ids.device)
        
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
                
                if self.config.activation == "tanh":
                    h_new = torch.tanh(h_new)
                elif self.config.activation == "relu":
                    h_new = F.relu(h_new)
                else:
                    raise ValueError(f"Unknown activation: {self.config.activation}")
                
                # Apply dropout between layers (not on last layer output)
                if layer_idx < self.config.num_layers - 1:
                    h_new = self.dropout(h_new)
                
                new_hidden.append(h_new)
                layer_input = h_new
            
            current_hidden = torch.stack(new_hidden)  # (n_layers, B, d_model)
            outputs.append(layer_input)  # Use final layer output
        
        # Stack outputs for all time steps
        rnn_output = torch.stack(outputs, dim=1)  # (B, T, d_model)
        rnn_output = self.dropout(rnn_output)
        
        # Apply output head
        logits = self.head(rnn_output)  # (B, T, num_states)
        
        # Compute loss if targets provided
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            mask = (
                unknown_mask
                if unknown_mask is not None
                else torch.ones_like(targets, dtype=torch.bool)
            )

            # Cross entropy only on the masked (unknown) positions.
            flat_mask = mask.view(-1)
            flat_logits = logits.view(-1, self.config.num_states)
            flat_targets = targets.view(-1)
            
            if flat_mask.any():
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = torch.tensor(0.0, device=input_ids.device)

        return logits, loss

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Create zero-initialized hidden state."""
        return torch.zeros(
            self.config.num_layers,
            batch_size, 
            self.config.d_model,
            device=device
        )

    def _init_weights(self, module: nn.Module):
        """Initialize weights following best practices for RNNs."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_moore_vanilla_rnn(
    vocab_size: int,
    num_states: int,
    max_seq_len: int = 512,
    d_model: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    activation: str = "tanh",
) -> MooreVanillaRNN:
    """Factory function for creating a Moore Vanilla RNN."""
    config = VanillaRNNConfig(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
    )
    return MooreVanillaRNN(config)