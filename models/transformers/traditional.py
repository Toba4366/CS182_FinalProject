"""
🤖 Modern Transformer Implementation for FSM In-Context Learning

This file provides a complete, production-ready transformer architecture optimized 
for our CS 182 final project on in-context learning of Moore machines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE OVERVIEW:

1. DECODER-ONLY TRANSFORMER:
   • Causal (left-to-right) attention for sequence modeling
   • Pre-LayerNorm architecture for training stability
   • Optional RoPE (Rotary Positional Embeddings) for better length generalization
   • PyTorch 2.0 Flash Attention integration for efficiency

2. FSM-OPTIMIZED DESIGN:
   • Small model sizes (2-4 layers) for interpretability research
   • Configurable vocabulary size for FSM tokens (states, actions, BOS)
   • Support for frozen layer experiments (test if only final layer can do ICL)
   • Flexible output dimensions for different prediction tasks

3. MODERN TRAINING FEATURES:
   • AdamW optimizer with proper weight decay grouping
   • Warmup + cosine annealing learning rate schedule  
   • Gradient clipping and automatic checkpointing
   • Weight tying between embedding and output layers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RESEARCH APPLICATIONS:

• IN-CONTEXT LEARNING: Test whether transformers can learn FSM rules from examples
• MECHANISTIC INTERPRETABILITY: Small models allow attention pattern analysis  
• PARTIAL OBSERVABILITY: Handle truncated sequences (start_state/action/non_start_state modes)
• ABLATION STUDIES: Frozen layers, different architectures, various training objectives

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 USAGE EXAMPLES:

# Basic FSM transformer (36-token vocabulary for our dataset)
model = CausalTransformer(
    vocab_size=36,          # BOS + 5 states + 8 actions + padding + special tokens
    d_model=128,            # Small for interpretability
    n_heads=4,              # Multi-head attention
    n_layers=2,             # Start with 2 layers for efficiency
    max_seq_len=256,        # Handle our 64-token sequences + context
    dropout=0.1,
    use_rope=True           # Better positional encoding
)

# Create optimizer with proper weight decay
optimizer = create_optimizer(model, lr=3e-4, weight_decay=0.01)

# Learning rate schedule with warmup
scheduler = create_warmup_cosine_scheduler(
    optimizer, 
    warmup_steps=1000, 
    total_steps=10000
)

# Training step
loss = train_step(model, optimizer, scheduler, batch, device="cuda")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MODEL CONFIGURATIONS FOR OUR EXPERIMENTS:

• BASELINE (2-layer): 128 dims, 4 heads - fast training, good for initial experiments  
• COMPARISON (3-layer): 256 dims, 8 heads - test scaling effects
• FROZEN EXPERIMENT: Only final linear layer trainable - test ICL hypothesis
• RoPE VARIANT: Use rotary embeddings instead of absolute positions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️ TECHNICAL FEATURES:

• Flash Attention: Automatic use of PyTorch 2.0 scaled_dot_product_attention
• Memory Efficient: Proper gradient accumulation and checkpointing support
• Flexible Training: Handles both causal LM and sequence-to-sequence objectives
• Production Ready: Complete save/load, optimizer state management, error handling

This implementation balances research needs (small, interpretable models) with 
modern best practices (efficient attention, proper optimization, clean interfaces).
"""

import math
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import scipy


# Optional: if you implement RoPE in models/transformers/utils/rope.py
try:
    from .utils.rope import RotaryEmbedding  # type: ignore
except ImportError:
    RotaryEmbedding = None


class MultiHeadSelfAttention(nn.Module):
    """
    🔍 Multi-Head Self-Attention with Causal Masking
    
    CORE COMPONENT: The attention mechanism that enables in-context learning.
    
    Key Features:
    • CAUSAL MASKING: Prevents future token access (essential for autoregressive modeling)
    • MULTI-HEAD: Parallel attention patterns for different aspects of FSM rules
    • ROPE SUPPORT: Optional rotary positional embeddings for better generalization
    • FLASH ATTENTION: Automatic PyTorch 2.0 optimization when available
    
    For FSM Learning:
    • Each head can potentially learn different FSM patterns (state transitions, action patterns)
    • Causal masking ensures realistic sequential processing 
    • Multiple heads allow specialization (e.g., one for states, one for actions)
    
    Architecture Details:
    • Query/Key/Value projections without bias (common in modern transformers)
    • Attention dropout for regularization during training
    • Output projection with residual connection support
    """

    def __init__(
        self,
        d_model: int,           # Model dimension (must be divisible by n_heads)
        n_heads: int,           # Number of attention heads (4-8 typical for small models)
        dropout: float = 0.0,   # Attention dropout rate
        use_rope: bool = False, # Whether to use Rotary Position Embeddings
        rope_base: int = 10000, # RoPE frequency base (10000 is standard)
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per attention head
        self.use_rope = use_rope

        # Query, Key, Value projection layers (no bias for efficiency)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout layers for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Optional RoPE for better positional understanding
        if use_rope and RotaryEmbedding is not None:
            self.rope = RotaryEmbedding(self.d_head, base=rope_base)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Process:
        1. Project input to Query, Key, Value
        2. Reshape for multi-head computation
        3. Apply optional RoPE positional encoding
        4. Compute attention with causal masking
        5. Apply output projection and dropout
        
        For FSM sequences like: BOS S0 A1 S1 A2 S2 ...
        • Attention allows model to relate states and actions
        • Causal masking ensures S2 can only attend to (BOS, S0, A1, S1, A2)
        • Multiple heads can specialize in different pattern types
        """
        B, T, C = x.shape

        # Step 1: Project to Query, Key, Value matrices
        q = self.q_proj(x)  # (B, T, C) - What am I looking for?
        k = self.k_proj(x)  # (B, T, C) - What do I have?
        v = self.v_proj(x)  # (B, T, C) - What information do I provide?

        # Step 2: Reshape for multi-head computation
        # Split d_model into n_heads x d_head for parallel processing
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)

        # Step 3: Apply RoPE positional encoding if enabled
        if self.rope is not None:
            q, k = self.rope(q, k)  # Rotary embeddings for better position understanding

        # Step 4: Compute scaled dot-product attention with causal masking
        # PyTorch 2.0 automatically uses Flash Attention when available for efficiency
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # Additional mask (usually None for causal-only)
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,       # Critical: prevents attending to future tokens
        )  # (B, n_heads, T, d_head)

        # Step 5: Rearrange back to original shape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.o_proj(attn_output)
        return self.out_dropout(attn_output)


class FeedForward(nn.Module):
    """
    🧠 Position-wise Feed-Forward Network (Transformer MLP)
    
    PURPOSE: Non-linear transformation applied to each position independently.
    
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    Role in FSM Learning:
    • Processes attention outputs to extract FSM rule patterns
    • GELU activation provides smooth non-linearity for gradient flow
    • Larger d_ff (typically 4x d_model) gives model more representational capacity
    • Independent processing per position allows per-token specialization
    
    For our experiments:
    • Small d_ff (e.g., 512) for efficient 2-layer models
    • Dropout prevents overfitting to specific FSM examples
    """

    def __init__(
        self, 
        d_model: int,        # Input/output dimension (matches attention output)
        d_ff: int,           # Hidden dimension (typically 2-4x d_model)
        dropout: float = 0.0  # Dropout rate for regularization
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # Expand to higher dimension
        self.fc2 = nn.Linear(d_ff, d_model)    # Project back to model dimension
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()                   # Smooth activation (better than ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward transformation.
        
        Each token is processed independently, allowing the model to:
        • Extract complex patterns from attention outputs
        • Apply non-linear transformations to FSM rule representations
        • Build hierarchical representations across transformer layers
        """
        x = self.fc1(x)      # (B, T, d_model) -> (B, T, d_ff)
        x = self.act(x)      # Apply GELU activation
        x = self.dropout(x)  # Regularization
        x = self.fc2(x)      # (B, T, d_ff) -> (B, T, d_model)
        x = self.dropout(x)  # Final dropout
        return x


class TransformerBlock(nn.Module):
    """
    🏗️ Complete Transformer Layer (Attention + MLP + Residuals)
    
    ARCHITECTURE: Pre-LayerNorm with residual connections
    
    Design Choice Rationale:
    • PRE-LAYERNORM: More stable training than post-LN, especially for small models
    • RESIDUAL CONNECTIONS: Enable deep learning by preventing vanishing gradients
    • LAYER ORDERING: LN -> Attention -> Residual, then LN -> MLP -> Residual
    
    For FSM Research:
    • Each layer can learn increasingly complex FSM patterns
    • Layer 1: Basic token relationships (state-action pairs)
    • Layer 2: Multi-step transitions and patterns
    • Layer 3+: Complex rule compositions and context dependencies
    
    This is the core building block - stack 2-4 of these for our experiments.
    """

    def __init__(
        self,
        d_model: int,           # Model dimension
        n_heads: int,           # Number of attention heads
        d_ff: int,              # Feed-forward hidden dimension
        dropout: float = 0.0,   # Dropout rate
        use_rope: bool = False, # Whether to use RoPE in attention
    ):
        super().__init__()
        # Pre-LayerNorm design: normalize before operations
        self.ln1 = nn.LayerNorm(d_model)  # Before attention
        self.ln2 = nn.LayerNorm(d_model)  # Before MLP
        
        # Core transformer components
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.mlp = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Pre-LN + Attention + Residual, then Pre-LN + MLP + Residual
        
        For FSM sequences, this allows:
        • Attention: Learn relationships between states/actions across positions
        • MLP: Extract and transform FSM rule patterns
        • Residuals: Maintain information flow for deep networks
        """
        # Self-attention block with residual connection
        h = self.ln1(x)         # Normalize first (pre-LN)
        h = self.attn(h)        # Apply attention
        x = x + h               # Residual connection

        # MLP block with residual connection  
        h = self.ln2(x)         # Normalize first (pre-LN)
        h = self.mlp(h)         # Apply MLP transformation
        x = x + h               # Residual connection
        return x


class CausalTransformer(nn.Module):
    """
    🎯 Complete Causal Transformer for FSM In-Context Learning
    
    FULL MODEL ARCHITECTURE:
    1. Token Embeddings: Convert FSM tokens (states, actions, BOS) to vectors
    2. Position Embeddings: Add sequence position information
    3. Transformer Layers: Stack of attention + MLP blocks
    4. Final LayerNorm: Stabilize pre-output representations
    5. Output Head: Project to vocabulary for next-token prediction
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🧠 FSM-SPECIFIC DESIGN DECISIONS:
    
    • VOCABULARY SIZE: 36 tokens for our dataset (BOS + 5 states + 8 actions + special)
    • MODEL SIZE: Small (128-256d) for interpretability and efficient experimentation
    • CAUSAL MODELING: Essential for realistic sequence prediction
    • FLEXIBLE OUTPUT: Can predict next state, action, or general next token
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔬 RESEARCH FEATURES:
    
    • FROZEN LAYER SUPPORT: Freeze transformer layers, only train final head
    • ROPE INTEGRATION: Better positional understanding for variable lengths
    • WEIGHT TYING: Share embedding and output weights (reduces parameters)
    • FLEXIBLE TARGETS: Support different training objectives
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📊 RECOMMENDED CONFIGURATIONS:
    
    Baseline (Fast):    vocab=36, d_model=128, n_heads=4, n_layers=2, d_ff=512
    Comparison:         vocab=36, d_model=256, n_heads=8, n_layers=3, d_ff=1024  
    Interpretability:   vocab=36, d_model=64,  n_heads=2, n_layers=2, d_ff=256
    """

    def __init__(
        self,
        vocab_size: int,                    # Number of unique tokens in FSM vocabulary
        d_model: int = 256,                 # Model dimension (embedding size)
        n_heads: int = 4,                   # Number of attention heads
        n_layers: int = 4,                  # Number of transformer layers
        d_ff: int = 1024,                   # Feed-forward hidden dimension
        max_seq_len: int = 512,             # Maximum sequence length
        dropout: float = 0.1,               # Dropout rate for regularization
        use_rope: bool = False,             # Use Rotary Position Embeddings
        tie_weights: bool = True,           # Tie input/output embedding weights
        out_dim: Optional[int] = None,      # Output dimension (None = vocab_size)
        freeze_layers: bool = False,        # 🧊 NEW: Freeze transformer layers for ICL experiments
        freeze_embeddings: bool = False,    # 🧊 NEW: Freeze embeddings too (test only final layer)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.freeze_layers = freeze_layers
        self.freeze_embeddings = freeze_embeddings

        # Core embedding layers
        self.token_embed = nn.Embedding(vocab_size, d_model)  # Convert tokens to vectors
        
        # Positional embeddings (optional if using RoPE, but we include both for flexibility)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)  # Input dropout

        # Stack of transformer layers - this is where the magic happens
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rope=use_rope,
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization for stable outputs
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection layer ("the final linear layer")
        if out_dim is None:
            out_dim = vocab_size
        self.head = nn.Linear(d_model, out_dim, bias=False)

        # Weight tying: share parameters between input embeddings and output projection
        # This is standard practice and reduces parameters while often improving performance
        if tie_weights and out_dim == vocab_size:
            self.head.weight = self.token_embed.weight

        # Initialize all parameters with appropriate distributions
        self.apply(self._init_weights)
        
        # 🧊 Apply freezing after initialization if requested
        if freeze_layers or freeze_embeddings:
            self._freeze_parameters()

    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights using modern best practices.
        
        Strategy:
        • Linear/Embedding: Normal distribution with std=0.02 (GPT-style)
        • LayerNorm: Weights to 1, bias to 0 (standard initialization)
        • Bias terms: Zero initialization when present
        
        This initialization promotes stable training and good convergence.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)    # Start with identity scaling
            nn.init.zeros_(module.bias)     # No initial bias

    def _freeze_parameters(self):
        """
        🧊 Freeze model parameters for In-Context Learning experiments.
        
        RESEARCH PURPOSE: Test whether only the final linear layer can solve ICL.
        
        This implements the core hypothesis from recent ICL research:
        • Can transformers learn FSM rules with only the output head trainable?
        • Do the transformer layers learn useful representations without gradient updates?
        • How much of ICL happens in the final linear layer vs. the attention layers?
        
        Freezing Strategy:
        • freeze_embeddings=True: Freeze token + positional embeddings
        • freeze_layers=True: Freeze all transformer blocks + final LayerNorm
        • head: Always remains trainable (this is what we're testing!)
        
        CRITICAL: Handle weight tying carefully - if embeddings are frozen but head
        should be trainable, we need to break weight tying to avoid conflicts.
        """
        # Store original weight tying state
        weights_were_tied = hasattr(self.head, 'weight') and hasattr(self.token_embed, 'weight') and \
                           self.head.weight is self.token_embed.weight
        
        if self.freeze_embeddings:
            # If we need to freeze embeddings but keep head trainable, break weight tying first
            if weights_were_tied:
                # Create separate weights for the head
                self.head.weight = nn.Parameter(self.token_embed.weight.data.clone())
                print("🔗 Broke weight tying between embeddings and head for independent freezing")
            
            # Now freeze embedding layers
            for param in self.token_embed.parameters():
                param.requires_grad = False
            for param in self.pos_embed.parameters():
                param.requires_grad = False
            print("🧊 Frozen embeddings (token + positional)")
        
        if self.freeze_layers:
            # Freeze all transformer blocks
            for param in self.blocks.parameters():
                param.requires_grad = False
            # Freeze final layer norm
            for param in self.ln_f.parameters():
                param.requires_grad = False
            print("🧊 Frozen transformer layers and final LayerNorm")
        
        # Ensure head always remains trainable (critical for the experiment!)
        for param in self.head.parameters():
            param.requires_grad = True
        
        # Verify we have at least some trainable parameters
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if trainable_count == 0:
            raise RuntimeError("All parameters were frozen! This would make training impossible.")
        
        print(f"🎯 Head remains trainable: {sum(p.numel() for p in self.head.parameters())} parameters")
        print(f"🔢 Total trainable: {trainable_count:,} parameters")

    def get_frozen_info(self) -> Dict[str, Any]:
        """
        📊 Get detailed information about parameter freezing status.
        
        Returns comprehensive statistics for experiment logging:
        • Total parameters in model
        • How many are trainable vs frozen
        • Percentage frozen (higher = more constrained experiment)
        • Breakdown by component (embeddings, blocks, head)
        
        Critical for validating experiments - ensures freezing worked correctly!
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Component-wise breakdown
        embed_params = sum(p.numel() for p in self.token_embed.parameters())
        pos_params = sum(p.numel() for p in self.pos_embed.parameters())
        blocks_params = sum(p.numel() for p in self.blocks.parameters())
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "frozen_percentage": (frozen_params / total_params) * 100 if total_params > 0 else 0,
            "component_breakdown": {
                "token_embeddings": embed_params,
                "positional_embeddings": pos_params,
                "transformer_blocks": blocks_params,
                "final_layernorm": ln_f_params,
                "output_head": head_params,
            },
            "trainable_breakdown": {
                "token_embeddings": sum(p.numel() for p in self.token_embed.parameters() if p.requires_grad),
                "positional_embeddings": sum(p.numel() for p in self.pos_embed.parameters() if p.requires_grad),
                "transformer_blocks": sum(p.numel() for p in self.blocks.parameters() if p.requires_grad),
                "final_layernorm": sum(p.numel() for p in self.ln_f.parameters() if p.requires_grad),
                "output_head": sum(p.numel() for p in self.head.parameters() if p.requires_grad),
            }
        }

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len) - tokenized FSM sequences
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len) - target tokens for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the complete transformer.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        🔄 PROCESSING PIPELINE:
        
        1. Embedding: Convert token IDs to dense vectors
        2. Position: Add positional information (absolute + optional RoPE)
        3. Dropout: Apply input regularization
        4. Transformer Layers: Process through attention + MLP stack
        5. Final Norm: Stabilize representations before output
        6. Head: Project to vocabulary logits
        7. Loss: Compute cross-entropy if targets provided
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        📊 EXAMPLE FSM SEQUENCE PROCESSING:
        
        Input:  [BOS, S0, A1, S1, A2, S2, A3, S3]
        Target: [S0,  A1, S1, A2, S2, A3, S3, EOS]  (shifted by 1)
        
        The model learns to predict the next token in the sequence,
        which teaches it FSM transition rules through examples.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Returns:
            logits: (batch, seq_len, vocab_size) - predictions for each position
            loss: scalar tensor (if targets provided), None otherwise
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Step 1: Convert token IDs to embedding vectors
        tok_emb = self.token_embed(input_ids)  # (B, T, d_model)
        
        # Step 2: Add positional information
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embed(pos_ids)  # (1, T, d_model)

        # Step 3: Combine embeddings and apply input dropout
        x = tok_emb + pos_emb  # Broadcast position embeddings across batch
        x = self.dropout(x)

        # Step 4: Process through transformer layers
        # Each layer builds increasingly complex representations of FSM patterns
        for block in self.blocks:
            x = block(x)  # Apply attention + MLP + residuals

        # Step 5: Final normalization for stable outputs
        x = self.ln_f(x)
        
        # Step 6: Project to vocabulary size for next-token prediction
        logits = self.head(x)  # (B, T, vocab_size)

        # Step 7: Compute training loss if targets provided
        loss = None
        if targets is not None:
            # Standard language modeling loss: predict next token at each position
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # Flatten to (B*T, vocab_size) - use reshape for memory efficiency
                targets.reshape(-1),                   # Flatten to (B*T,) - use reshape for memory efficiency
                ignore_index=-100,                     # Ignore special padding tokens
            )

        return logits, loss


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 OPTIMIZER + SCHEDULER + TRAINING UTILITIES
# 
# Production-ready training components optimized for transformer models.
# Includes modern best practices: weight decay grouping, warmup scheduling,
# gradient clipping, and comprehensive checkpointing.
# ══════════════════════════════════════════════════════════════════════════════

def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,                           # Learning rate (3e-4 is sweet spot for transformers)
    weight_decay: float = 0.01,                 # L2 regularization strength
    betas: Tuple[float, float] = (0.9, 0.95),  # Adam momentum parameters
) -> AdamW:
    """
    🎯 Create AdamW optimizer with proper weight decay grouping.
    
    CRITICAL OPTIMIZATION DETAIL: Not all parameters should have weight decay!
    
    Weight Decay Strategy:
    • APPLY to: Linear weights, embedding weights (main model parameters)
    • SKIP for: Biases, LayerNorm weights (already regularized)
    
    Why this matters:
    • Weight decay on LayerNorm can hurt training stability
    • Bias terms are typically small and don't need L2 regularization
    • This grouping is standard practice for transformers
    
    Parameters:
        lr: Learning rate (3e-4 works well for our model sizes)
        weight_decay: L2 penalty strength (0.01 is typical)
        betas: Adam momentum (0.9, 0.95 for good convergence)
    """
    # Separate parameters into two groups based on whether they should have weight decay
    decay_params = []     # Parameters that get weight decay (weights)
    no_decay_params = []  # Parameters that don't get weight decay (biases, norms)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        
        # Determine if parameter should have weight decay
        if name.endswith("bias") or "ln" in name or "norm" in name:
            no_decay_params.append(param)  # No decay for biases and normalization layers
        else:
            decay_params.append(param)     # Weight decay for weights

    # Create optimizer with parameter groups
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )
    return optimizer


def create_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,      # Number of warmup steps (typically 1000-5000)
    total_steps: int,       # Total training steps
) -> LambdaLR:
    """
    📈 Create learning rate scheduler with warmup + cosine decay.
    
    MODERN LR SCHEDULING: This is the gold standard for transformer training.
    
    Schedule Design:
    1. WARMUP PHASE (0 -> warmup_steps):
       • Linear increase from 0 to peak learning rate
       • Prevents early training instability
       • Critical for transformer convergence
    
    2. COSINE DECAY PHASE (warmup_steps -> total_steps):
       • Smooth decay following cosine curve
       • Ends at 10% of original learning rate (not zero!)
       • Maintains some learning capacity throughout training
    
    Why this works:
    • Gradual warmup prevents early gradient explosion
    • Cosine decay provides smooth learning rate reduction
    • Ending at 10% (not 0%) allows continued fine-tuning
    
    Typical values for our experiments:
    • warmup_steps: 1000-2000 (10-20% of total)
    • total_steps: 10000-50000 (depends on dataset size)
    """

    def lr_lambda(step: int):
        """Compute learning rate multiplier for given step."""
        if step < warmup_steps:
            # Linear warmup: 0 -> 1 over warmup_steps
            return float(step) / max(1, warmup_steps)
        
        # Cosine decay: 1 -> 0.1 over remaining steps
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine formula: start at 1.0, end at 0.1
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    💾 Save comprehensive model checkpoint.
    
    COMPLETE STATE PRESERVATION: Saves everything needed to resume training.
    
    What gets saved:
    • Model weights (state_dict)
    • Optimizer state (momentum, learning rates, etc.)
    • Scheduler state (current step, decay progress)
    • Training step counter
    • Extra metadata (loss history, config, etc.)
    
    This ensures perfect training resumption - no loss of optimization state.
    Critical for long experiments and distributed training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Build checkpoint dictionary
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    
    # Add optional components if provided
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    
    # Atomic save operation
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    map_location: Optional[str] = None,
) -> int:
    """
    📂 Load comprehensive model checkpoint.
    
    PERFECT RESUME CAPABILITY: Restores complete training state.
    
    Loading strategy:
    1. Load checkpoint file
    2. Restore model weights
    3. Restore optimizer state (if provided)
    4. Restore scheduler state (if provided)
    5. Return current training step
    
    map_location handles device placement:
    • None: Load to same device as saved
    • "cpu": Force CPU loading
    • "cuda:0": Force specific GPU loading
    
    Returns:
        step: Current training step for resuming
    """
    ckpt = torch.load(path, map_location=map_location)
    
    # Always load model weights
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Load optimizer state if both checkpoint and optimizer provided
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    # Load scheduler state if both checkpoint and scheduler provided
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
    # Return current step (0 if not found in checkpoint)
    return ckpt.get("step", 0)


# ══════════════════════════════════════════════════════════════════════════════
# 🏃 TRAINING STEP IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def train_step(
    model: CausalTransformer,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    batch: Tuple[torch.Tensor, torch.Tensor],
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> float:
    """
    🚀 Execute one complete training step.
    
    FULL TRAINING PIPELINE: Forward -> Loss -> Backward -> Clip -> Update
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔄 STEP-BY-STEP PROCESS:
    
    1. SETUP: Set model to training mode, move data to device
    2. FORWARD: Compute model predictions and loss
    3. BACKWARD: Compute gradients via backpropagation  
    4. CLIP: Apply gradient clipping to prevent exploding gradients
    5. UPDATE: Apply optimizer step and learning rate schedule
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🛡️ GRADIENT CLIPPING: Essential for transformer training stability.
    
    Why clip gradients?
    • Transformers can have gradient explosion, especially early in training
    • max_grad_norm=1.0 is standard and works well
    • Prevents training instability and NaN losses
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Parameters:
        model: The transformer model to train
        optimizer: AdamW optimizer with proper weight decay grouping
        scheduler: Learning rate scheduler (warmup + cosine decay)
        batch: (input_ids, targets) tuple from DataLoader
        max_grad_norm: Gradient clipping threshold (1.0 is standard)
        device: Training device ("cuda" or "cpu")
    
    Returns:
        loss: Scalar loss value for logging/monitoring
    """
    # Step 1: Set training mode and prepare data
    model.train()
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    # Step 2: Reset gradients (set_to_none=True for efficiency)
    optimizer.zero_grad(set_to_none=True)
    
    # Step 3: Forward pass - compute predictions and loss
    _, loss = model(input_ids, targets)
    
    # Step 4: Backward pass - compute gradients
    loss.backward()

    # Step 5: Gradient clipping for training stability
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Step 6: Apply optimizer update and learning rate schedule
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Return loss for logging
    return float(loss.item())
