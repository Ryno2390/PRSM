"""
Integration Core Components
==========================

Core infrastructure for PRSM's integration layer, providing:
- Base connector abstraction for platform integrations
- Central integration manager for coordination
- FTNS-integrated provenance engine for creator rewards
"""

from .base_connector import BaseConnector, ConnectorStatus
from .integration_manager import IntegrationManager
from .provenance_engine import ProvenanceEngine

__all__ = [
    "BaseConnector",
    "ConnectorStatus", 
    "IntegrationManager",
    "ProvenanceEngine"
]