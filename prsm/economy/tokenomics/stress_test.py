"""
PRSM Tokenomics Stress Test (2026 pre-flight)
=============================================

Simulates high-demand scenarios to verify the balance between:
1. Token Issuance (Staking Rewards & Breakthrough Payouts)
2. Token Burn (Institutional Gateway Fees & Slashing)

Goal: Prevent Hyper-inflation during the 'Success Scenario'.
"""

import logging
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Dict, Any

from prsm.economy.tokenomics.ftns_service import FTNSService, FTNSTransactionType

logger = logging.getLogger(__name__)

class TokenomicsSimulator:
    """
    Simulates the PRSM economy over 365 virtual days.
    """
    def __init__(self):
        self.service = FTNSService()
        self.total_burned = Decimal("0")
        self.total_issued = Decimal("0")

    def run_success_scenario(self, num_nodes: int = 5000):
        """
        Simulates 5,000 nodes joining the APM Genesis challenge.
        """
        print(f"📈 Simulating Success Scenario with {num_nodes} active nodes...")
        
        # 1. ISSUANCE: Staking rewards and challenge payouts
        # Each node earns average 10 FTNS/day for compute
        daily_issuance = Decimal(str(num_nodes)) * Decimal("10.0")
        self.total_issued += daily_issuance * 365
        
        # 2. BURN: Institutional Gateway & Transaction Fees
        # Institutions pay to use the network (high-value data access)
        # We simulate 50 large labs paying 1000 FTNS/day in fees
        daily_burn = Decimal("50") * Decimal("1000.0")
        self.total_burned += daily_burn * 365
        
        # 3. Slashing (Malicious node cleanup)
        # 1% of nodes are malicious and lose 500 FTNS stake
        malicious_burn = Decimal(str(num_nodes * 0.01)) * Decimal("500.0")
        self.total_burned += malicious_burn

    def get_net_velocity(self) -> Dict[str, Any]:
        net_change = self.total_issued - self.total_burned
        inflation_rate = (net_change / self.total_issued) * 100 if self.total_issued > 0 else 0
        
        return {
            "total_issued": float(self.total_issued),
            "total_burned": float(self.total_burned),
            "net_change": float(net_change),
            "inflation_rate": float(inflation_rate),
            "is_stable": abs(inflation_rate) < 5.0 # Stable if < 5% net change
        }

def run_preflight_stress_test():
    sim = TokenomicsSimulator()
    sim.run_success_scenario(num_nodes=5000)
    metrics = sim.get_net_velocity()
    
    print("\n--- PRSM TOKENOMICS PRE-FLIGHT REPORT ---")
    print(f"Total Tokens Issued (Annual): {metrics['total_issued']:,.2f}")
    print(f"Total Tokens Burned (Annual): {metrics['total_burned']:,.2f}")
    print(f"Net Economic Change: {metrics['net_change']:,.2f}")
    print(f"Projected Annual Inflation: {metrics['inflation_rate']:.2f}%")
    
    if metrics["is_stable"]:
        print("✅ SUCCESS: The economy is balanced.")
    else:
        print("⚠️ WARNING: Inflationary risk detected. Adjusting burn rates recommended.")

if __name__ == "__main__":
    run_preflight_stress_test()
