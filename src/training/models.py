"""
Simple transformer model for in-context learning of Moore machines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 64
    max_seq_len: int = 256
    d_model: int = 128
    num_heads: int = 4  # Reduced for smaller models
    num_layers: int = 2  # Optimized for 2-3 layer experiments
    d_ff: int = 256  # Reduced for smaller models
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    freeze_layers: bool = False  # Whether to freeze all layers except final output
    freeze_embeddings: bool = False  # Whether to freeze embeddings too


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
        assert config.d_model % config.num_heads == 0
        
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and reshape to (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (for decoder-only architecture)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                           (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    """
    Simple decoder-only transformer for in-context learning.
    Supports freezing layers to test if only the final linear layer can solve ICL.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply freezing if requested
        if config.freeze_layers or config.freeze_embeddings:
            self._freeze_parameters()
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def _freeze_parameters(self):
        """
        Freeze parameters based on config settings.
        This tests whether only the final linear layer can solve ICL.
        """
        # Freeze embeddings if requested
        if self.config.freeze_embeddings:
            for param in self.token_embedding.parameters():
                param.requires_grad = False
            for param in self.position_encoding.parameters():
                param.requires_grad = False
        
        # Freeze all transformer layers if requested
        if self.config.freeze_layers:
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.ln_f.parameters():
                param.requires_grad = False
            # Note: lm_head remains trainable for the experiment
    
    def get_frozen_info(self) -> dict:
        """Get information about which parameters are frozen."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'frozen_percentage': (frozen_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_hidden_states: Whether to return hidden states
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            hidden_states: Hidden states (if requested)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        x = self.dropout(self.position_encoding(token_emb))
        
        # Transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: int = 50,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 pad_token_id: int = 0,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Initial token IDs
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        batch_size, input_len = input_ids.size()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_len):
                # Get logits for last token
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)