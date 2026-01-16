import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from prometheus_client import Gauge, Histogram

# Metric Definitions
PRSM_GATING_ENTROPY = Gauge('prsm_gating_entropy', 'Shannon entropy of Top-K routing weights')
PRSM_CORE_UTILIZATION = Gauge('prsm_core_utilization_ratio', 'Percentage of tokens routed to each core', ['core_type'])
PRSM_REASONING_DEPTH = Histogram('prsm_reasoning_depth', 'Number of logic jumps in WisdomPackage')

class TopKRouter(nn.Module):
    """
    Top-K Token-Level Routing mechanism for NWTN Gating.
    Replaces sequence-level batch-mean statistics with precise token-level routing.
    Includes Neural Observability metrics.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        super(TopKRouter, self).__init__()
        self.num_experts = num_experts
        self.k = k
        # Learnable router linear layer that outputs logits for each token
        self.router = nn.Linear(input_dim, num_experts)
        
        # Expert mapping for metrics (assuming 3 experts based on requirements)
        self.expert_names = {0: 'SSM', 1: 'SANM', 2: 'FSMN'}

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            routing_weights: Tensor of shape (batch_size, seq_len, k)
            selected_experts: Tensor of shape (batch_size, seq_len, k)
        """
        # Calculate logits: (batch_size, seq_len, num_experts)
        logits = self.router(x)
        
        # Neural Observability: Calculate Entropy
        # We use the full logits softmax for entropy calculation to see the router's uncertainty
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        PRSM_GATING_ENTROPY.set(entropy.item())
        
        # Select top-k experts per token
        # topk_logits: (batch_size, seq_len, k)
        # selected_experts: (batch_size, seq_len, k)
        topk_logits, selected_experts = torch.topk(logits, self.k, dim=-1)
        
        # Neural Observability: Core Utilization
        # Flatten selected experts to count occurrences
        flat_experts = selected_experts.flatten()
        total_tokens = flat_experts.numel()
        
        if total_tokens > 0:
            for idx, name in self.expert_names.items():
                count = (flat_experts == idx).sum().item()
                ratio = count / total_tokens
                PRSM_CORE_UTILIZATION.labels(core_type=name).set(ratio)
        
        # Normalize weights via softmax across the selected experts to maintain gradient flow
        routing_weights = F.softmax(topk_logits, dim=-1)
        
        return routing_weights, selected_experts

    def record_reasoning_depth(self, logic_jumps: int):
        """Records the depth of reasoning for a processed package."""
        PRSM_REASONING_DEPTH.observe(logic_jumps)
