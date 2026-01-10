"""
PRSM FSMN Core Architecture
===========================

Implements Feedforward Sequential Memory Networks (FSMN).
Provides non-recurrent sequence modeling with memory blocks,
ideal for low-latency speech and signal processing on edge nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple

class FSMNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lorder=None, rorder=None, lstride=1, rstride=1):
        super(FSMNBlock, self).__init__()
        self.dim = input_dim
        if lorder is None: return
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        # Use 1D convolution to simulate the memory filter
        self.conv_left = nn.Conv1d(
            self.dim, self.dim, lorder, 
            dilation=lstride, groups=self.dim, bias=False, padding=(lorder-1)*lstride
        )

        if self.rorder > 0:
            self.conv_right = nn.Conv1d(
                self.dim, self.dim, rorder, 
                dilation=rstride, groups=self.dim, bias=False, padding=(rorder-1)*rstride
            )
        else:
            self.conv_right = None

    def forward(self, input: torch.Tensor, cache: Optional[torch.Tensor] = None):
        # input: (B, T, D) -> (B, D, T)
        x = input.transpose(1, 2)
        batch, dim, seq_len = x.shape

        y_left = self.conv_left(x)[:, :, :seq_len]
        out = x + y_left

        if self.conv_right is not None:
            y_right = self.conv_right(x)[:, :, :seq_len]
            out += y_right

        return out.transpose(1, 2), cache

class PRSM_FSMN_Model(nn.Module):
    """
    Feedforward Sequential Memory Network for high-efficiency signal reasoning.
    """
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.fsmn_layers = nn.ModuleList([
            FSMNBlock(hidden_dim, hidden_dim, lorder=10, rorder=2) 
            for _ in range(layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None):
        x = F.relu(self.in_proj(x))
        for layer in self.fsmn_layers:
            x, _ = layer(x)
        return self.out_proj(x), cache

def get_fsmn_reasoner(input_dim=512, hidden_dim=512):
    return PRSM_FSMN_Model(input_dim, hidden_dim)
