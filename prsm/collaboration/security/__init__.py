"""
PRSM Post-Quantum Security Layer

This module provides comprehensive post-quantum cryptographic security for
PRSM's P2P collaboration platform, implementing the "Coca Cola Recipe"
security model with enterprise-grade key management and access control.

Key Components:
- Post-Quantum Key Management: Distributed key generation and sharing
- Access Control: Fine-grained permissions with multi-signature authorization
- File Reconstruction: Secure reassembly of distributed shards
- Integrity Validation: Tamper detection and cryptographic verification
- Crypto Sharding: Post-quantum file encryption and distribution

The security layer ensures that sensitive collaborative data remains secure
even against quantum computing attacks while maintaining usability and
performance for legitimate users.
"""

from .crypto_sharding import CryptoSharding
from .post_quantum_crypto_sharding import PostQuantumCryptoSharding

from .key_management import (
    DistributedKeyManager,
    CryptographicKey,
    KeyShare,
    PostQuantumCrypto,
    ShamirSecretSharing,
    KeyType,
    KeyStatus,
    PostQuantumAlgorithm
)

from .access_control import (
    PostQuantumAccessController,
    AccessRule,
    MultiSigRequest,
    AccessAuditLog,
    AccessControlMatrix,
    Permission,
    AccessLevel,
    ResourceType
)

from .reconstruction_engine import (
    PostQuantumReconstructionEngine,
    ReconstructionTask,
    ReconstructionShard,
    ShardCollector,
    ParallelReconstructor,
    ReconstructionStatus,
    ShardVerificationStatus
)

from .integrity_validator import (
    IntegrityValidator,
    IntegrityProof,
    ValidationResult,
    MerkleTree,
    PostQuantumSigner,
    IntegrityStatus,
    ValidationLevel
)

# Version information
__version__ = "1.0.0"
__author__ = "PRSM Development Team"

# Export main classes for easy import
__all__ = [
    # Legacy crypto sharding (compatibility)
    'CryptoSharding',
    'PostQuantumCryptoSharding',
    
    # Key Management
    'DistributedKeyManager',
    'CryptographicKey',
    'KeyShare',
    'PostQuantumCrypto',
    'ShamirSecretSharing',
    'KeyType',
    'KeyStatus',
    'PostQuantumAlgorithm',
    
    # Access Control
    'PostQuantumAccessController',
    'AccessRule',
    'MultiSigRequest',
    'AccessAuditLog',
    'AccessControlMatrix',
    'Permission',
    'AccessLevel',
    'ResourceType',
    
    # File Reconstruction
    'PostQuantumReconstructionEngine',
    'ReconstructionTask',
    'ReconstructionShard',
    'ShardCollector',
    'ParallelReconstructor',
    'ReconstructionStatus',
    'ShardVerificationStatus',
    
    # Integrity Validation
    'IntegrityValidator',
    'IntegrityProof',
    'ValidationResult',
    'MerkleTree',
    'PostQuantumSigner',
    'IntegrityStatus',
    'ValidationLevel'
]

# Configuration constants
DEFAULT_SECURITY_CONFIG = {
    'key_management': {
        'key_lifetime': 86400 * 365,  # 1 year
        'min_threshold': 3,
        'max_shares': 7,
        'rotation_interval': 86400 * 30  # 30 days
    },
    'access_control': {
        'multisig_threshold': 2,
        'audit_retention': 86400 * 90,  # 90 days
        'max_rules_per_resource': 100
    },
    'reconstruction': {
        'max_concurrent': 5,
        'timeout': 300,  # 5 minutes
        'verification_required': True
    },
    'integrity_validation': {
        'default_level': 'standard',
        'batch_size': 100,
        'cache_ttl': 300,  # 5 minutes
        'merkle_chunk_size': 4096
    },
    'crypto_sharding': {
        'default_redundancy': 3,
        'max_shard_size': 10 * 1024 * 1024,  # 10MB
        'encryption_algorithm': 'AES-256-GCM'
    }
}

def get_default_security_config():
    """Get default security configuration"""
    return DEFAULT_SECURITY_CONFIG.copy()

def create_security_layer(node_id: str, config=None):
    """
    Factory function to create a complete security layer instance
    
    This creates and configures all security components with proper
    integration between them.
    
    Args:
        node_id: Unique identifier for this node
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing initialized security components
    """
    import asyncio
    
    # Use provided config or defaults
    full_config = DEFAULT_SECURITY_CONFIG.copy()
    if config:
        for section, values in config.items():
            if section in full_config:
                full_config[section].update(values)
            else:
                full_config[section] = values
    
    # Initialize components
    components = {}
    
    # Key Manager (foundation for all other components)
    components['key_manager'] = DistributedKeyManager(
        node_id, full_config['key_management']
    )
    
    # Access Controller
    components['access_controller'] = PostQuantumAccessController(
        node_id, components['key_manager'], full_config['access_control']
    )
    
    # Integrity Validator
    components['integrity_validator'] = IntegrityValidator(
        components['key_manager'], full_config['integrity_validation']
    )
    
    # Reconstruction Engine (requires access controller and key manager)
    components['reconstruction_engine'] = PostQuantumReconstructionEngine(
        components['key_manager'],
        components['access_controller'],
        None,  # P2P network will be set later
        full_config['reconstruction']
    )
    
    # Crypto Sharding (legacy and post-quantum)
    components['crypto_sharding'] = CryptoSharding()
    components['pq_crypto_sharding'] = PostQuantumCryptoSharding()
    
    return components

async def initialize_security_layer(components):
    """Initialize all security layer components"""
    initialization_tasks = []
    
    # Initialize access controller (requires async setup)
    if 'access_controller' in components:
        initialization_tasks.append(
            components['access_controller'].initialize()
        )
    
    # Add other async initialization tasks here
    
    if initialization_tasks:
        await asyncio.gather(*initialization_tasks, return_exceptions=True)

def integrate_with_p2p(security_components, p2p_components):
    """
    Integrate security layer with P2P network layer
    
    This connects the security components with the P2P network
    for distributed operations.
    """
    # Set P2P network references
    if 'key_manager' in security_components and 'node_discovery' in p2p_components:
        security_components['key_manager'].set_p2p_network(
            p2p_components['node_discovery']
        )
    
    if 'access_controller' in security_components and 'node_discovery' in p2p_components:
        security_components['access_controller'].set_p2p_network(
            p2p_components['node_discovery']
        )
    
    if 'reconstruction_engine' in security_components:
        # Set P2P network for reconstruction
        security_components['reconstruction_engine'].p2p_network = p2p_components.get('node_discovery')
        
        # Set reputation system for shard collection
        if 'reputation_system' in p2p_components:
            security_components['reconstruction_engine'].set_reputation_system(
                p2p_components['reputation_system']
            )

# Security level constants
class SecurityLevel:
    """Security level constants for different use cases"""
    BASIC = "basic"           # Basic encryption, minimal validation
    STANDARD = "standard"     # Standard PQ crypto, access control
    HIGH = "high"            # Full security with multi-sig
    MAXIMUM = "maximum"      # Paranoid security with all features

# Recommended configurations for different security levels
SECURITY_LEVEL_CONFIGS = {
    SecurityLevel.BASIC: {
        'key_management': {'min_threshold': 2, 'max_shares': 3},
        'access_control': {'multisig_threshold': 1},
        'integrity_validation': {'default_level': 'basic'}
    },
    SecurityLevel.STANDARD: {
        'key_management': {'min_threshold': 3, 'max_shares': 5},
        'access_control': {'multisig_threshold': 2},
        'integrity_validation': {'default_level': 'standard'}
    },
    SecurityLevel.HIGH: {
        'key_management': {'min_threshold': 4, 'max_shares': 7},
        'access_control': {'multisig_threshold': 3},
        'integrity_validation': {'default_level': 'comprehensive'}
    },
    SecurityLevel.MAXIMUM: {
        'key_management': {'min_threshold': 5, 'max_shares': 9},
        'access_control': {'multisig_threshold': 4},
        'integrity_validation': {'default_level': 'forensic'}
    }
}

def get_security_config_for_level(level: str) -> dict:
    """Get security configuration for a specific security level"""
    base_config = get_default_security_config()
    level_config = SECURITY_LEVEL_CONFIGS.get(level, {})
    
    # Merge level-specific config with defaults
    for section, values in level_config.items():
        if section in base_config:
            base_config[section].update(values)
    
    return base_config

# Module documentation
DESCRIPTION = """
PRSM Post-Quantum Security Layer

This comprehensive security module provides enterprise-grade cryptographic
protection for PRSM's distributed collaboration platform. It implements
a multi-layered security architecture designed to withstand both classical
and quantum computing attacks.

Core Security Features:

1. **Post-Quantum Cryptography**
   - Kyber-1024 for key encapsulation
   - ML-DSA (Dilithium) for digital signatures
   - Future-proof against quantum attacks
   - Seamless fallback to classical algorithms

2. **Distributed Key Management**
   - Shamir's Secret Sharing for key distribution
   - No single point of failure for keys
   - Automatic key rotation and lifecycle management
   - Hardware security module (HSM) integration ready

3. **Fine-Grained Access Control**
   - Role-based access control (RBAC)
   - Attribute-based access control (ABAC)
   - Multi-signature authorization for sensitive operations
   - Comprehensive audit logging with tamper-proof records

4. **Secure File Reconstruction**
   - Parallel shard collection from P2P network
   - Cryptographic verification of each shard
   - Fault-tolerant reconstruction with missing shards
   - Performance optimization for large files

5. **Integrity Validation**
   - Multiple validation levels (basic to forensic)
   - Merkle tree verification for efficient batch checking
   - Tamper detection and forensic analysis
   - Post-quantum digital signatures for authenticity

The "Coca Cola Recipe" Security Model ensures that sensitive collaborative
data is cryptographically sharded and distributed such that:
- No single node can access complete files without authorization
- Quantum-resistant encryption protects against future attacks
- Distributed key management eliminates single points of failure
- Comprehensive audit trails ensure accountability
- Multi-signature authorization prevents unauthorized access

This security layer integrates seamlessly with PRSM's P2P network layer
to provide a complete solution for secure distributed collaboration.
"""

# Example usage documentation
EXAMPLE_USAGE = """
Example Usage:

    import asyncio
    from prsm.collaboration.security import create_security_layer, initialize_security_layer

    async def main():
        # Create security layer for a node
        security_components = create_security_layer("node_123", {
            'key_management': {'min_threshold': 3},
            'access_control': {'multisig_threshold': 2}
        })
        
        # Initialize async components
        await initialize_security_layer(security_components)
        
        # Get components
        key_manager = security_components['key_manager']
        access_controller = security_components['access_controller']
        validator = security_components['integrity_validator']
        
        # Generate encryption keys
        encryption_key = await key_manager.generate_keypair(
            KeyType.ENCRYPTION, PostQuantumAlgorithm.KYBER_1024
        )
        
        # Grant access permissions
        rule_id = await access_controller.grant_access(
            subject_id="user456",
            resource_id="sensitive_document",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        # Create integrity proof
        test_data = b"Sensitive collaborative document content"
        proof_id = await validator.create_integrity_proof(
            test_data, "doc_123", encryption_key
        )
        
        # Validate data integrity
        result = await validator.validate_data_integrity(
            test_data, proof_id, ValidationLevel.COMPREHENSIVE
        )
        
        print(f"Data integrity: {result.is_valid}")

    asyncio.run(main())
"""