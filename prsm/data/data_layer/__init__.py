"""
PRSM Data Layer Module
Enhanced IPFS integration for model storage and provenance tracking

⚠️ DEPRECATION NOTICE ⚠️
This module is deprecated. Please use prsm.core.ipfs_client instead.
The imports below are provided for backward compatibility only.
This module will be removed in a future version of PRSM.

Migration Guide:
  Old: from prsm.data.data_layer import PRSMIPFSClient
  New: from prsm.core.ipfs_client import IPFSClient, PRSMIPFSOperations

  Old: from prsm.data.data_layer import get_ipfs_client
  New: from prsm.core.ipfs_client import create_ipfs_client

The PRSMIPFSClient class provided model-specific storage features.
These are now available through PRSMIPFSOperations in prsm.core.ipfs_client.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "prsm.data.data_layer is deprecated. Use prsm.core.ipfs_client instead. "
    "See the module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import the enhanced IPFS client for backward compatibility
from .enhanced_ipfs import PRSMIPFSClient, get_ipfs_client

# Also export from the core module for easy migration
from prsm.core.ipfs_client import (
    # Exceptions
    IPFSError,
    IPFSTimeoutError,
    IPFSConnectionError,
    
    # Configuration and Data Classes
    IPFSConfig,
    IPFSContent,
    IPFSStats,
    
    # Main client and operations
    IPFSClient,
    PRSMIPFSOperations,
    
    # Utility Functions
    create_ipfs_client,
    add_text_to_ipfs,
    get_text_from_ipfs,
    add_json_to_ipfs,
    get_json_from_ipfs,
    list_pinned_content,
    get_content_info,
)

__all__ = [
    # Legacy exports (deprecated)
    'PRSMIPFSClient',  # Deprecated - use PRSMIPFSOperations from core
    'get_ipfs_client',  # Deprecated - use create_ipfs_client from core
    
    # New exports from core (for migration)
    'IPFSError',
    'IPFSTimeoutError',
    'IPFSConnectionError',
    'IPFSConfig',
    'IPFSContent',
    'IPFSStats',
    'IPFSClient',
    'PRSMIPFSOperations',
    'create_ipfs_client',
    'add_text_to_ipfs',
    'get_text_from_ipfs',
    'add_json_to_ipfs',
    'get_json_from_ipfs',
    'list_pinned_content',
    'get_content_info',
]