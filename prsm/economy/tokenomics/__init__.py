"""
PRSM Tokenomics Module
Enhanced FTNS token economy for context allocation and rewards

This module provides both simulation-based and database-backed FTNS implementations:
- ftns_service: Legacy simulation-based service (for testing/development)
- database_ftns_service: Production database-backed service
- models: Database models for FTNS token economy

The default export uses the database-backed service for production readiness.
"""

# Import both services for flexibility
from .ftns_service import FTNSService, get_ftns_service
from .database_ftns_service import DatabaseFTNSService, database_ftns_service

# Export database service as default for production use
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
    # Services
    "FTNSService", "DatabaseFTNSService",
    "ftns_service", "simulation_ftns_service", "database_ftns_service",
    
    # Database Models
    "FTNSWallet", "FTNSTransaction", "FTNSProvenanceRecord",
    "FTNSDividendDistribution", "FTNSDividendPayment", "FTNSRoyaltyPayment",
    "FTNSMarketplaceListing", "FTNSMarketplaceTransaction",
    "FTNSGovernanceVote", "FTNSAuditLog",
    
    # Enums
    "TransactionType", "TransactionStatus", "WalletType",
    "DividendStatus", "RoyaltyStatus"
]