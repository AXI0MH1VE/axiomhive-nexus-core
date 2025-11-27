"""Hybrid State Space Model - Mamba-2 + Attention.

Combines state space models for efficiency with attention for recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class Mamba2Layer(nn.Module):
    """Mamba-2 state space layer for efficient sequence modeling."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand_factor: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # State projections
        self.x_proj = nn.Linear(self.d_inner, d_state)
        self.dt_proj = nn.Linear(self.d_inner, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Activation
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba-2 layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)
        x_inner, res = x_and_res.chunk(2, dim=-1)
        x_inner = self.activation(x_inner)
        
        # Compute state space parameters
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.dt_proj(x_inner))
        
        # State space computation (simplified)
        # Full Mamba-2 uses selective scan algorithm
        state = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x_inner[:, t, :]
            delta_t = delta[:, t, :]
            
            # Discretize state space
            A_discrete = torch.exp(A * delta_t.mean(dim=-1, keepdim=True))
            
            # State update
            state = A_discrete * state + self.x_proj(x_t)
            
            # Output
            y_t = state.sum(dim=-1, keepdim=True).expand(-1, self.d_inner) * x_t
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        
        # Skip connection and output projection
        y = y * self.activation(res)
        y = self.out_proj(y)
        y = y + x * self.D
        
        return y


class MultiHeadAttention(nn.Module):
    """Multi-head attention for high-resolution recall."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch, seq_len, d)
        output = self.out_proj(attn_output)
        
        return output


class HybridStateSpaceModel(nn.Module):
    """Hybrid model combining Mamba-2 SSM with Attention.
    
    Uses:
    - Mamba-2 for efficient state transition
    - Attention for high-resolution recall
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        n_heads: int = 8,
        n_layers: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mamba': Mamba2Layer(d_model, d_state),
                'attention': MultiHeadAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': self._build_ffn(d_model)
            }))

    def _build_ffn(self, d_model: int) -> nn.Module:
        """Build feed-forward network."""
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through hybrid model.
        
        Args:
            x: Input tensor [batch, seq_len, d_model] or [batch, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model] or [batch, d_model]
        """
        # Handle 2D input
        squeeze_output = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        # Process through layers
        for layer in self.layers:
            # Mamba-2 state space processing
            mamba_out = layer['mamba'](x)
            x = layer['norm1'](x + mamba_out)
            
            # Attention for recall
            attn_out = layer['attention'](x, mask)
            x = layer['norm2'](x + attn_out)
            
            # Feed-forward
            ffn_out = layer['ffn'](x)
            x = x + ffn_out
        
        # Squeeze if needed
        if squeeze_output:
            x = x.squeeze(1)
        
        return x

    def get_state_representation(self) -> torch.Tensor:
        """Extract current state representation.
        
        Returns:
            State tensor
        """
        # Aggregate state from first Mamba layer
        first_mamba = self.layers[0]['mamba']
        return first_mamba.A_log.detach()