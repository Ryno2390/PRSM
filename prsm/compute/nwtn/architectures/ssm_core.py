"""
PRSM SSM Core Architecture
==========================

Lightweight State Space Model (SSM) implementation for edge-efficient reasoning.
Provides linear-time complexity and constant memory footprint for long sequences,
enabling advanced AI reasoning on resource-constrained edge nodes.

Inspired by Mamba and S4 architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

class SSMConfig:
    """Configuration for SSM Model"""
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        inner_layernorm: bool = False,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank
            
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.inner_layernorm = inner_layernorm

class SSMBlock(nn.Module):
    """
    A single SSM block implementing the selection mechanism and state evolution.
    This is the "Brain" of the linear-scaling reasoning node.
    """
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_inner
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=config.bias)
        
        # Conv layer for local temporal dependency
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )
        
        # S6 mechanism parameters (Selective State Space)
        self.x_proj = nn.Linear(self.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, self.d_inner, bias=True)
        
        # State matrices initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=config.bias)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Forward pass with optional state for recurrent inference.
        x: (batch, seq_len, d_model)
        """
        (b, l, d) = x.shape
        
        # 1. Input projection and split
        x_and_res = self.in_proj(x)  # (b, l, 2*d_inner)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)
        
        # 2. Conv1d branch
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # 3. Selective SSM (S6)
        y, next_state = self.selective_scan(x, state)
        
        # 4. Multiplicative gate with residual
        y = y * F.silu(res)
        
        # 5. Output projection
        output = self.out_proj(y)
        
        return output, next_state

    def selective_scan(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Implements the discretization and scanning of the state space.
        This is what replaces the "Attention" mechanism.
        """
        b, l, d = x.shape
        n = self.config.d_state
        
        # Project x to get delta, B, and C (selection mechanism)
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, n, n], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (b, l, d_inner)
        
        # Discretize A and B
        A = -torch.exp(self.A_log)  # (d_inner, n)
        
        # Simple Euler discretization for edge efficiency
        # In a real Mamba implementation, this uses parallel associative scan
        # For our edge nodes, we implement both sequential (for inference) and 
        # vectorized (for training/batch processing) modes.
        
        if l > 1:
            # Vectorized mode
            return self._vectorized_scan(x, dt, A, B, C, self.D)
        else:
            # Sequential mode (Real-time edge inference)
            return self._step_scan(x, dt, A, B, C, self.D, state)

    def _vectorized_scan(self, x, dt, A, B, C, D):
        """Vectorized scan for batch processing"""
        b, l, d = x.shape
        n = A.shape[-1]
        
        # Compute discretized matrices
        # dA = exp(dt * A)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        # dB = dt * B
        dB = torch.einsum('bld,bln->bldn', dt, B)
        
        # Scan
        states = torch.zeros(b, d, n, device=x.device)
        outputs = []
        
        for i in range(l):
            states = dA[:, i] * states + dB[:, i] * x[:, i, :, None]
            y = torch.einsum('bdn,bn->bd', states, C[:, i])
            outputs.append(y)
            
        y_out = torch.stack(outputs, dim=1) + x * D
        return y_out, states

    def _step_scan(self, x, dt, A, B, C, D, state):
        """Single-step scan for recurrent inference"""
        if state is None:
            state = torch.zeros(x.shape[0], x.shape[2], A.shape[-1], device=x.device)
            
        dA = torch.exp(dt * A)
        dB = torch.einsum('bd,bn->bdn', dt.squeeze(1), B.squeeze(1))
        
        state = dA * state + dB * x.transpose(1, 2)
        y = torch.einsum('bdn,bn->bd', state, C.squeeze(1))
        
        y_out = (y + x.squeeze(1) * D).unsqueeze(1)
        return y_out, state

class PRSM_SSM_Model(nn.Module):
    """
    Complete PRSM SSM Model for Scientific Modeling.
    Wraps multiple SSM blocks into a deep reasoning stack.
    """
    def __init__(self, config: SSMConfig, num_layers: int = 6, vocab_size: int = 50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            SSMBlock(config) for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, states: Optional[list] = None, return_logits: bool = True):
        if input_ids.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            x = self.embedding(input_ids)
        else:
            # Already embedded
            x = input_ids
        
        new_states = []
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, new_layer_state = layer(x, layer_state)
            new_states.append(new_layer_state)
            
        x = self.norm_f(x)
        
        if return_logits:
            logits = self.lm_head(x)
            return logits, new_states
        else:
            return x, new_states

def get_ssm_reasoner(d_model=512, layers=6):
    """Factory function for creating an SSM reasoner"""
    config = SSMConfig(d_model=d_model)
    return PRSM_SSM_Model(config, num_layers=layers)
