"""
Revenue Split Engine
====================

Distributes FTNS payments according to the hybrid pricing model:
- Data owner: 80% of data access fees
- Compute providers: 15% of total (split by PCU contribution)
- PRSM Treasury: 5% network fee

When no proprietary data is involved:
- Compute providers: 95% of total
- PRSM Treasury: 5% network fee
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default revenue splits
DATA_OWNER_SHARE = Decimal("0.80")
COMPUTE_PROVIDER_SHARE = Decimal("0.15")
TREASURY_SHARE = Decimal("0.05")

# When no data owner
COMPUTE_ONLY_SHARE = Decimal("0.95")
TREASURY_ONLY_SHARE = Decimal("0.05")


@dataclass
class RevenueSplit:
    """Computed revenue distribution for a single job."""
    total_payment: Decimal
    data_owner_amount: Decimal = Decimal("0")
    data_owner_id: str = ""
    compute_amounts: Dict[str, Decimal] = field(default_factory=dict)  # provider_id -> amount
    treasury_amount: Decimal = Decimal("0")
    has_data_owner: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_payment": str(self.total_payment),
            "data_owner_amount": str(self.data_owner_amount),
            "data_owner_id": self.data_owner_id,
            "compute_amounts": {k: str(v) for k, v in self.compute_amounts.items()},
            "treasury_amount": str(self.treasury_amount),
            "has_data_owner": self.has_data_owner,
        }


class RevenueSplitEngine:
    """Calculates revenue distribution for completed jobs."""

    def __init__(
        self,
        data_owner_share: Decimal = DATA_OWNER_SHARE,
        compute_share: Decimal = COMPUTE_PROVIDER_SHARE,
        treasury_share: Decimal = TREASURY_SHARE,
    ):
        self.data_owner_share = data_owner_share
        self.compute_share = compute_share
        self.treasury_share = treasury_share

    def calculate_split(
        self,
        total_payment: Decimal,
        data_owner_id: str = "",
        compute_providers: Optional[Dict[str, float]] = None,
    ) -> RevenueSplit:
        """Calculate the revenue split for a job.

        Args:
            total_payment: Total FTNS paid for the job.
            data_owner_id: ID of data owner (empty if no proprietary data).
            compute_providers: Dict of {provider_id: pcu_contributed}.
                Used to split compute share proportionally.

        Returns:
            RevenueSplit with amounts per party.
        """
        providers = compute_providers or {}
        has_data = bool(data_owner_id)

        if has_data:
            data_amount = total_payment * self.data_owner_share
            compute_pool = total_payment * self.compute_share
            treasury = total_payment * self.treasury_share
        else:
            data_amount = Decimal("0")
            compute_pool = total_payment * COMPUTE_ONLY_SHARE
            treasury = total_payment * TREASURY_ONLY_SHARE

        # Split compute pool proportionally by PCU contribution
        compute_amounts = {}
        total_pcu = sum(providers.values())
        if total_pcu > 0 and providers:
            for pid, pcu in providers.items():
                share = Decimal(str(pcu / total_pcu))
                compute_amounts[pid] = compute_pool * share
        elif providers:
            # Equal split if no PCU data
            per_provider = compute_pool / Decimal(str(len(providers)))
            for pid in providers:
                compute_amounts[pid] = per_provider

        return RevenueSplit(
            total_payment=total_payment,
            data_owner_amount=data_amount,
            data_owner_id=data_owner_id,
            compute_amounts=compute_amounts,
            treasury_amount=treasury,
            has_data_owner=has_data,
        )
