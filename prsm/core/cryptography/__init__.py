"""
PRSM Production-Grade Cryptography Module
========================================

Enterprise cryptography infrastructure providing advanced encryption,
zero-knowledge proofs, secure key management, and privacy-preserving
protocols for production deployment.
"""

from .key_management import KeyManager, get_key_manager
from .encryption import EncryptionService, get_encryption_service
from .zk_proofs import ZKProofSystem, get_zk_proof_system
from .post_quantum import (
    PostQuantumCrypto, PostQuantumKeyPair, PostQuantumSignature,
    SecurityLevel, SignatureType, get_post_quantum_crypto,
    generate_pq_keypair, sign_with_pq, verify_pq_signature
)
from .crypto_models import (
    KeyType, EncryptionAlgorithm, HashAlgorithm, CurveType,
    CryptoKey, EncryptionResult, SignatureResult, ProofResult,
    ZKProof, PrivacyPolicy, SecureMessage, KeyUsage, PrivacyLevel,
    KeyGenerationRequest, EncryptionRequest, DecryptionRequest, ZKProofRequest
)
from .dag_signatures import (
    DAGSignatureManager, KeyPair, SignatureError, InvalidSignatureError,
    MissingSignatureError, MissingPublicKeyError,
    create_signing_key_pair, sign_hash, verify_hash_signature
)

__all__ = [
    "KeyManager",
    "get_key_manager",
    "EncryptionService", 
    "get_encryption_service",
    "ZKProofSystem",
    "get_zk_proof_system",
    "PostQuantumCrypto",
    "PostQuantumKeyPair", 
    "PostQuantumSignature",
    "SecurityLevel",
    "SignatureType",
    "get_post_quantum_crypto",
    "generate_pq_keypair",
    "sign_with_pq",
    "verify_pq_signature",
    "KeyType",
    "EncryptionAlgorithm",
    "HashAlgorithm", 
    "CurveType",
    "CryptoKey",
    "EncryptionResult",
    "SignatureResult",
    "ProofResult",
    "ZKProof",
    "PrivacyPolicy",
    "SecureMessage",
    "KeyUsage",
    "PrivacyLevel",
    "KeyGenerationRequest",
    "EncryptionRequest",
    "DecryptionRequest",
    "ZKProofRequest",
    # DAG Transaction Signatures
    "DAGSignatureManager",
    "KeyPair",
    "SignatureError",
    "InvalidSignatureError",
    "MissingSignatureError",
    "MissingPublicKeyError",
    "create_signing_key_pair",
    "sign_hash",
    "verify_hash_signature",
]