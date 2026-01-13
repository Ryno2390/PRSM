"""
PRSM Cross-Core Gating Network (CCGN)
=====================================

A specialized 'Mixture of Experts' gating layer that dynamically 
distributes weights between SSM, SANM, and FSMN cores.
Optimizes for both inference speed (SSM) and logical precision (SANM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple

class CrossCoreGater(nn.Module):
    """
    Dynamic Gating Network for NWTN cores.
    Uses a lightweight linear head to predict which core is best for the current token/context.
    """
    def __init__(self, d_model: int, num_cores: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_cores = num_cores
        
        # Gating Head: predicts weights for each core
        self.gate = nn.Linear(d_model, num_cores)
        
        # Softmax for normalized probability distribution
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a probability distribution across cores.
        x: (batch, seq_len, d_model)
        """
        # Global pooling across sequence to get context-aware weights
        # (Or token-level gating if seq_len > 1)
        if x.dim() == 3:
            context = torch.mean(x, dim=1) 
        else:
            context = x
            
        gate_logits = self.gate(context)
        return self.softmax(gate_logits)

class HybridMultiCoreModel(nn.Module):
    """
    Orchestrates multiple NWTN cores with dynamic weight distribution.
    """
    def __init__(self, cores: Dict[str, nn.Module], d_model: int):
        super().__init__()
        self.cores = nn.ModuleDict(cores)
        self.gater = CrossCoreGater(d_model, len(cores))
        self.core_names = list(cores.keys())

    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Any]:
        # 1. Calculate Core Weights
        core_weights = self.gater(x) # (batch, num_cores)
        
        # 2. Parallel Execution (Simulated)
        # In a real MoE, we would only execute the top-k cores.
        # For our hybrid, we blend the outputs based on gating weights.
        final_output = torch.zeros_like(x)
        
        for i, name in enumerate(self.core_names):
            if name == "ssm":
                core_output, _ = self.cores[name](x, return_logits=False)
            else:
                core_output, _ = self.cores[name](x)
            
            # Weight the output
            weight = core_weights[:, i].view(-1, 1, 1)
            final_output += core_output * weight
            
        return final_output, core_weights
