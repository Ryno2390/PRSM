# PRSM IPFS Integration Module
#
# ⚠️ DEPRECATION NOTICE ⚠️
# This module is deprecated. Please use prsm.core.ipfs_client instead.
# The imports below are provided for backward compatibility only.
# This module will be removed in a future version of PRSM.
#
# Migration Guide:
#   Old: from prsm.data.ipfs import IPFSClient, IPFSConfig
#   New: from prsm.core.ipfs_client import IPFSClient, IPFSConfig
#
#   Canonical accessor: from prsm.core.ipfs_client import get_ipfs_client
#
#   Old: from prsm.data.ipfs import IPFSContent, IPFSStats
#   New: from prsm.core.ipfs_client import IPFSContent, IPFSStats
#
#   Old: from prsm.data.ipfs import IPFSError, IPFSTimeoutError
#   New: from prsm.core.ipfs_client import IPFSError, IPFSTimeoutError, IPFSConnectionError

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "prsm.data.ipfs is deprecated. Use prsm.core.ipfs_client instead. "
    "See the module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the consolidated core module
from prsm.core.ipfs_client import (
    # Exceptions
    IPFSError,
    IPFSTimeoutError,
    IPFSConnectionError,
    
    # Configuration and Data Classes
    IPFSConfig,
    IPFSContent,
    IPFSStats,
    
    # Utility Functions
    get_ipfs_client,
    create_ipfs_client,
    add_text_to_ipfs,
    get_text_from_ipfs,
    add_json_to_ipfs,
    get_json_from_ipfs,
    list_pinned_content,
    get_content_info,
)

# Keep the local IPFSClient for now (it has some different method signatures)
# This will be fully migrated in a future phase
from .ipfs_client import IPFSClient

# Content addressing and verification remain in this module
from .content_addressing import (
    ContentAddressingSystem, 
    AddressedContent,
    ContentCategory,
    ContentStatus,
    ContentProvenance,
    ContentLicense,
    create_addressing_system,
    create_basic_provenance,
    create_open_license
)
from .content_verification import (
    ContentVerificationSystem,
    VerificationResult,
    VerificationStatus,
    ProvenanceChain,
    ProvenanceEvent,
    ProvenanceEventType,
    create_verification_system
)

__all__ = [
    # IPFS Client (deprecated - use prsm.core.ipfs_client)
    'IPFSClient',  # Deprecated - local implementation
    
    # Exceptions (from core)
    'IPFSError',
    'IPFSTimeoutError', 
    'IPFSConnectionError',
    
    # Configuration and Data Classes (from core)
    'IPFSConfig', 
    'IPFSContent',
    'IPFSStats',
    
    # Utility Functions (from core)
    'get_ipfs_client',
    'create_ipfs_client',
    'add_text_to_ipfs',
    'get_text_from_ipfs',
    'add_json_to_ipfs',
    'get_json_from_ipfs',
    'list_pinned_content',
    'get_content_info',
    
    # Content Addressing (local - not deprecated)
    'ContentAddressingSystem',
    'AddressedContent',
    'ContentCategory',
    'ContentStatus', 
    'ContentProvenance',
    'ContentLicense',
    'create_addressing_system',
    'create_basic_provenance',
    'create_open_license',
    
    # Content Verification (local - not deprecated)
    'ContentVerificationSystem',
    'VerificationResult',
    'VerificationStatus',
    'ProvenanceChain',
    'ProvenanceEvent',
    'ProvenanceEventType',
    'create_verification_system'
]
