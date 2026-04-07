"""
Data Listing Manager
====================

Manages dataset listings with pricing, access control, and staking requirements.
Data owners publish datasets; researchers pay to access them.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataListing:
    """A dataset published by a data owner with pricing."""
    listing_id: str = field(default_factory=lambda: f"dl-{uuid.uuid4().hex[:12]}")
    dataset_id: str = ""
    owner_id: str = ""
    title: str = ""
    description: str = ""
    shard_count: int = 0
    total_size_bytes: int = 0

    # Pricing (owner-controlled)
    base_access_fee: Decimal = Decimal("0")
    per_shard_fee: Decimal = Decimal("0")
    bulk_discount: float = 0.0
    success_royalty_rate: float = 0.0  # Extra % if query produces valuable results

    # Access control
    requires_stake: Decimal = Decimal("0")
    max_queries_per_day: int = 0  # 0 = unlimited
    allowed_operations: List[str] = field(default_factory=list)
    exclusive_until: float = 0.0  # Timestamp for exclusive access window

    # Metadata
    created_at: float = field(default_factory=time.time)
    active: bool = True
    total_queries: int = 0
    total_revenue: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "dataset_id": self.dataset_id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "shard_count": self.shard_count,
            "base_access_fee": str(self.base_access_fee),
            "per_shard_fee": str(self.per_shard_fee),
            "bulk_discount": self.bulk_discount,
            "success_royalty_rate": self.success_royalty_rate,
            "requires_stake": str(self.requires_stake),
            "max_queries_per_day": self.max_queries_per_day,
            "active": self.active,
            "total_queries": self.total_queries,
            "total_revenue": str(self.total_revenue),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataListing":
        return cls(
            listing_id=d.get("listing_id", f"dl-{uuid.uuid4().hex[:12]}"),
            dataset_id=d.get("dataset_id", ""),
            owner_id=d.get("owner_id", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            shard_count=d.get("shard_count", 0),
            total_size_bytes=d.get("total_size_bytes", 0),
            base_access_fee=Decimal(str(d.get("base_access_fee", "0"))),
            per_shard_fee=Decimal(str(d.get("per_shard_fee", "0"))),
            bulk_discount=d.get("bulk_discount", 0.0),
            success_royalty_rate=d.get("success_royalty_rate", 0.0),
            requires_stake=Decimal(str(d.get("requires_stake", "0"))),
            max_queries_per_day=d.get("max_queries_per_day", 0),
            active=d.get("active", True),
        )


class DataListingManager:
    """Manages published dataset listings."""

    def __init__(self):
        self._listings: Dict[str, DataListing] = {}

    def publish(self, listing: DataListing) -> str:
        """Publish a dataset listing. Returns the listing_id."""
        self._listings[listing.listing_id] = listing
        logger.info(f"Dataset listed: {listing.title} ({listing.listing_id})")
        return listing.listing_id

    def get_listing(self, listing_id: str) -> Optional[DataListing]:
        return self._listings.get(listing_id)

    def find_by_dataset(self, dataset_id: str) -> Optional[DataListing]:
        for listing in self._listings.values():
            if listing.dataset_id == dataset_id and listing.active:
                return listing
        return None

    def search(
        self,
        query: str = "",
        max_price: Optional[Decimal] = None,
        limit: int = 20,
    ) -> List[DataListing]:
        """Search active listings by title/description keyword and max price."""
        results = []
        query_lower = query.lower()
        for listing in self._listings.values():
            if not listing.active:
                continue
            if query_lower and query_lower not in listing.title.lower() and query_lower not in listing.description.lower():
                continue
            if max_price is not None and listing.base_access_fee > max_price:
                continue
            results.append(listing)
        return results[:limit]

    def check_access(
        self,
        listing_id: str,
        accessor_stake: Decimal = Decimal("0"),
    ) -> tuple:
        """Check if accessor meets staking requirements.

        Returns (allowed: bool, reason: str).
        """
        listing = self._listings.get(listing_id)
        if listing is None:
            return False, "Listing not found"
        if not listing.active:
            return False, "Listing is inactive"
        if listing.requires_stake > 0 and accessor_stake < listing.requires_stake:
            return False, f"Insufficient stake: {accessor_stake} < {listing.requires_stake} FTNS required"
        if listing.exclusive_until > 0 and time.time() < listing.exclusive_until:
            return False, "Dataset is in exclusive access window"
        return True, ""

    def record_query(self, listing_id: str, revenue: Decimal) -> None:
        """Record a completed query against a listing."""
        listing = self._listings.get(listing_id)
        if listing:
            listing.total_queries += 1
            listing.total_revenue += revenue

    def list_all(self, active_only: bool = True) -> List[DataListing]:
        if active_only:
            return [l for l in self._listings.values() if l.active]
        return list(self._listings.values())

    def deactivate(self, listing_id: str) -> bool:
        listing = self._listings.get(listing_id)
        if listing:
            listing.active = False
            return True
        return False
