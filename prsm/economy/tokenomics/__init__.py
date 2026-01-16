"""
PRSM Tokenomics Module
Enhanced FTNS token economy for context allocation and rewards

This module provides multiple FTNS implementations:
- ftns_service: Legacy simulation-based service (for testing/development)
- database_ftns_service: Database-backed service with basic persistence
- atomic_ftns_service: Production service with atomic operations (RECOMMENDED)
- production_ledger: Full-featured production ledger

The default export uses the atomic service for production readiness.

Security Note:
The atomic_ftns_service implements double-spend prevention via:
1. SELECT FOR UPDATE row-level locking
2. Optimistic concurrency control via version columns
3. Idempotency keys to prevent duplicate operations
"""

# Import all services for flexibility
from .ftns_service import FTNSService, get_ftns_service
from .database_ftns_service import DatabaseFTNSService, database_ftns_service
from .atomic_ftns_service import (
    AtomicFTNSService,
    get_atomic_ftns_service,
    TransactionResult,
    BalanceInfo,
    AtomicOperationError,
    InsufficientBalanceError,
    ConcurrentModificationError,
    IdempotencyViolationError
)

# Export atomic service as default for production use (double-spend safe)
# Use: from prsm.economy.tokenomics import atomic_ftns_service
# Then: service = await get_atomic_ftns_service()

# Legacy alias for backwards compatibility
ftns_service = database_ftns_service

# Export all database models
from .models import (
    FTNSWallet, FTNSTransaction, FTNSProvenanceRecord,
    FTNSDividendDistribution, FTNSDividendPayment, FTNSRoyaltyPayment,
    FTNSMarketplaceListing, FTNSMarketplaceTransaction,
    FTNSGovernanceVote, FTNSAuditLog,
    TransactionType, TransactionStatus, WalletType,
    DividendStatus, RoyaltyStatus
)

__all__ = [
    # Atomic Service (RECOMMENDED for production)
    "AtomicFTNSService",
    "get_atomic_ftns_service",
    "TransactionResult",
    "BalanceInfo",
    "AtomicOperationError",
    "InsufficientBalanceError",
    "ConcurrentModificationError",
    "IdempotencyViolationError",

    # Legacy Services
    "FTNSService", "DatabaseFTNSService",
    "ftns_service", "database_ftns_service",
    "get_ftns_service",

    # Database Models
    "FTNSWallet", "FTNSTransaction", "FTNSProvenanceRecord",
    "FTNSDividendDistribution", "FTNSDividendPayment", "FTNSRoyaltyPayment",
    "FTNSMarketplaceListing", "FTNSMarketplaceTransaction",
    "FTNSGovernanceVote", "FTNSAuditLog",

    # Enums
    "TransactionType", "TransactionStatus", "WalletType",
    "DividendStatus", "RoyaltyStatus"
]