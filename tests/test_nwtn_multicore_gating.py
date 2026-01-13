
import torch
import pytest
from prsm.compute.nwtn.architectures.gating_network import CrossCoreGater, HybridMultiCoreModel
from prsm.compute.nwtn.architectures.ssm_core import PRSM_SSM_Model, SSMConfig
from prsm.compute.nwtn.architectures.sanm_core import PRSM_SANM_Model
from prsm.compute.nwtn.architectures.fsmn_core import PRSM_FSMN_Model

def test_cross_core_gater_distribution():
    """Verify that the gater produces a valid probability distribution across cores"""
    d_model = 512
    num_cores = 3
    gater = CrossCoreGater(d_model, num_cores)
    
    # Dummy input (batch=2, seq=10, d_model=512)
    x = torch.randn(2, 10, d_model)
    weights = gater(x)
    
    assert weights.shape == (2, num_cores)
    # Weights should sum to 1.0 (Softmax)
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(2))

def test_hybrid_multicore_execution():
    """Verify that the hybrid model executes all cores and blends outputs"""
    d_model = 512
    
    # Use smaller configs for test speed
    config = SSMConfig(d_model=d_model, d_state=8)
    cores = {
        "ssm": PRSM_SSM_Model(config, num_layers=1),
        "sanm": PRSM_SANM_Model(d_model=d_model, nhead=4, layers=1),
        "fsmn": PRSM_FSMN_Model(input_dim=d_model, hidden_dim=d_model, layers=1)
    }
    
    model = HybridMultiCoreModel(cores, d_model)
    x = torch.randn(1, 5, d_model)
    
    # We need to monkey-patch forward for SSM because it expects input_ids
    # For this test, we'll just check that it runs without error if the gater is working.
    # Note: Real SSMBlock.forward expects (batch, seq, dim), which we provide.
    
    output, weights = model(x)
    
    assert output.shape == x.shape
    assert weights.shape == (1, 3)
    assert not torch.isnan(output).any()
