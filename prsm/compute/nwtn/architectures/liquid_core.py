"""
PRSM Liquid Neural Network Core
===============================

Implements Liquid Neural Networks (LNNs) based on ODE-based time-continuous neurons.
Provides extreme efficiency for time-series and sensor data on the edge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

class LiquidCell(nn.Module):
    """
    A single neuron that follows an Ordinary Differential Equation (ODE).
    Its state 'liquifies' and adapts over time based on input and intrinsic dynamics.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # ODE Parameters
        self.w_input = nn.Parameter(torch.randn(input_size, hidden_size) / np.sqrt(hidden_size))
        self.w_recurrent = nn.Parameter(torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size))
        self.gleak = nn.Parameter(torch.randn(hidden_size)) # Leakage conductance
        self.vleak = nn.Parameter(torch.randn(hidden_size)) # Leakage potential
        self.tau = nn.Parameter(torch.ones(hidden_size))    # Time constant

    def forward(self, x: torch.Tensor, state: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        One time-step update using an Euler solver for the ODE.
        """
        # Calculate synaptic input
        synapse = torch.matmul(x, self.w_input) + torch.matmul(state, self.w_recurrent)
        synapse = torch.tanh(synapse)
        
        # dV/dt = -[G_leak + Synapse] * V + [G_leak * V_leak + Synapse * E_syn]
        # (Simplified implementation for Edge nodes)
        derivative = -(torch.exp(self.gleak) + synapse) * state + (torch.exp(self.gleak) * self.vleak + synapse)
        
        new_state = state + dt * (derivative / torch.exp(self.tau))
        return new_state

class PRSM_Liquid_Model(nn.Module):
    """
    Continuous-time Liquid Network for Extreme Edge IoT nodes.
    Ideal for real-time sensor processing in the scientific network.
    """
    def __init__(self, input_size: int = 64, hidden_size: int = 128, output_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiquidCell(input_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            
        # Process sequence step-by-step (time-continuous)
        outputs = []
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)
            outputs.append(self.output_head(h))
            
        return torch.stack(outputs, dim=1), h

def get_liquid_reasoner(input_size=64, hidden_size=128):
    """Factory for Liquid Networks"""
    return PRSM_Liquid_Model(input_size, hidden_size)
