"""
Mock Mamba implementation for CPU-only environments.

This provides a simplified Mamba-like architecture using standard PyTorch components
when the full mamba_ssm package is not available (e.g., on CPU-only machines).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MockMambaBlock(nn.Module):
    """
    Simplified Mamba-like block using standard PyTorch operations.
    
    This captures some of the key ideas of Mamba (selective state spaces)
    but using CPU-friendly operations instead of the optimized CUDA kernels.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection (like Mamba's in_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution (like Mamba's conv1d)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)  # dt and B
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # State space matrices
        A_log = torch.log(torch.rand(self.d_inner, d_state))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mock Mamba block.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Convolution (temporal mixing)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B, L, d_inner)
        x_conv = self.act(x_conv)
        
        # SSM parameters
        x_ssm = self.x_proj(x_conv)  # (B, L, 2*d_state)
        dt, B_ssm = x_ssm.chunk(2, dim=-1)  # (B, L, d_state) each
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Simplified SSM computation (CPU-friendly)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Simplified selective scan (not optimized like real Mamba)
        y = self._simple_ssm_scan(x_conv, dt, A, B_ssm)
        
        # Skip connection
        y = y + x_conv * self.D[None, None, :]
        
        # Gate with z
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def _simple_ssm_scan(self, x, dt, A, B):
        """
        Simplified SSM scan operation.
        This is much slower than the optimized CUDA version but works on CPU.
        """
        B_batch, L, D_inner = x.shape
        D_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(B_batch, D_inner, D_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for i in range(L):
            # Get inputs for this timestep
            x_t = x[:, i, :]  # (B, D_inner)
            dt_t = dt[:, i, :]  # (B, D_inner)
            B_t = B[:, i, :]  # (B, D_state)
            
            # Discretize A matrix
            dA = torch.exp(dt_t[:, :, None] * A[None, :, :])  # (B, D_inner, D_state)
            dB = dt_t[:, :, None] * B_t[:, None, :]  # (B, D_inner, D_state)
            
            # Update state
            h = h * dA + dB * x_t[:, :, None]  # (B, D_inner, D_state)
            
            # Compute output (simplified - just sum over state)
            y_t = h.sum(dim=2)  # (B, D_inner)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (B, L, D_inner)


class MockMambaLM(nn.Module):
    """
    Mock Mamba Language Model using simplified Mamba blocks.
    
    This provides the same interface as the real MambaLM but uses
    CPU-friendly operations instead of optimized CUDA kernels.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        max_seq_len: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MockMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through mock Mamba LM.
        
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) for loss computation
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Token embeddings
        x = self.token_embed(input_ids)  # (B, T, d_model)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        # Final layer norm
        x = self.norm(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


# Alias for compatibility
MambaLM = MockMambaLM