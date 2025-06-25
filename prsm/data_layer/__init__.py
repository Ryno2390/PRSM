"""
PRSM Data Layer Module
Enhanced IPFS integration for model storage and provenance tracking
"""

from .enhanced_ipfs import PRSMIPFSClient, get_ipfs_client

__all__ = ["PRSMIPFSClient", "get_ipfs_client"]