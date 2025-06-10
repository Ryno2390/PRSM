"""
PRSM Cryptography Models
=======================

Comprehensive data models for cryptographic operations, key management,
privacy protocols, and secure communication infrastructure.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..core.database import Base


class KeyType(str, Enum):
    """Cryptographic key types"""
    RSA = "rsa"
    ECDSA = "ecdsa"
    ECDH = "ecdh"
    ED25519 = "ed25519"
    AES = "aes"
    CHACHA20 = "chacha20"
    SECP256K1 = "secp256k1"
    BLS12_381 = "bls12_381"


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"
    XCHACHA20_POLY1305 = "xchacha20_poly1305"
    FERNET = "fernet"


class HashAlgorithm(str, Enum):
    """Cryptographic hash algorithms"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    KECCAK256 = "keccak256"


class CurveType(str, Enum):
    """Elliptic curve types"""
    SECP256K1 = "secp256k1"
    SECP256R1 = "secp256r1"
    ED25519 = "ed25519"
    BLS12_381 = "bls12_381"
    CURVE25519 = "curve25519"


class KeyUsage(str, Enum):
    """Key usage types"""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    KEY_AGREEMENT = "key_agreement"
    AUTHENTICATION = "authentication"
    PRIVACY = "privacy"
    PROOF_GENERATION = "proof_generation"
    TOKEN_OPERATIONS = "token_operations"


class PrivacyLevel(str, Enum):
    """Privacy protection levels"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    ZERO_KNOWLEDGE = "zero_knowledge"


# === Database Models ===

class CryptoKeyStore(Base):
    """Secure cryptographic key storage"""
    __tablename__ = "crypto_keys"
    
    # Primary identifiers
    key_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_name = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    
    # Key metadata
    key_type = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    key_usage = Column(String(100), nullable=False)
    curve_type = Column(String(50), nullable=True)
    key_size = Column(Integer, nullable=True)
    
    # Security properties
    is_hardware_backed = Column(Boolean, default=False)
    is_exportable = Column(Boolean, default=False)
    requires_authentication = Column(Boolean, default=True)
    
    # Encrypted key material (never store raw keys)
    encrypted_private_key = Column(LargeBinary, nullable=True)
    encrypted_public_key = Column(LargeBinary, nullable=True)
    key_material_hash = Column(String(128), nullable=False)
    
    # Key derivation information
    derivation_path = Column(String(255), nullable=True)
    parent_key_id = Column(UUID(as_uuid=True), nullable=True)
    derivation_index = Column(Integer, nullable=True)
    
    # Lifecycle management
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    rotation_schedule = Column(String(100), nullable=True)
    
    # Status and compliance
    is_active = Column(Boolean, default=True)
    is_compromised = Column(Boolean, default=False)
    compliance_level = Column(String(50), nullable=True)
    
    # Additional metadata
    metadata = Column(JSONB, default=dict)
    tags = Column(JSONB, default=list)
    
    # Relationships
    signatures = relationship("CryptoSignature", back_populates="key")
    encrypted_data = relationship("EncryptedData", back_populates="key")


class CryptoSignature(Base):
    """Cryptographic signature records"""
    __tablename__ = "crypto_signatures"
    
    # Primary identifiers
    signature_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(UUID(as_uuid=True), ForeignKey("crypto_keys.key_id"), nullable=False)
    
    # Signature details
    algorithm = Column(String(100), nullable=False)
    message_hash = Column(String(128), nullable=False)
    signature_value = Column(LargeBinary, nullable=False)
    
    # Context information
    signer_id = Column(String(255), nullable=False)
    purpose = Column(String(255), nullable=False)
    context_data = Column(JSONB, default=dict)
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verification_timestamp = Column(DateTime(timezone=True), nullable=True)
    verification_result = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    key = relationship("CryptoKeyStore", back_populates="signatures")


class EncryptedData(Base):
    """Encrypted data storage"""
    __tablename__ = "encrypted_data"
    
    # Primary identifiers
    data_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(UUID(as_uuid=True), ForeignKey("crypto_keys.key_id"), nullable=False)
    
    # Encryption details
    algorithm = Column(String(100), nullable=False)
    encrypted_content = Column(LargeBinary, nullable=False)
    initialization_vector = Column(LargeBinary, nullable=True)
    authentication_tag = Column(LargeBinary, nullable=True)
    
    # Metadata
    content_type = Column(String(100), nullable=False)
    content_hash = Column(String(128), nullable=False)
    encryption_context = Column(JSONB, default=dict)
    
    # Access control
    owner_id = Column(String(255), nullable=False)
    access_policy = Column(JSONB, default=dict)
    privacy_level = Column(String(50), default=PrivacyLevel.CONFIDENTIAL.value)
    
    # Lifecycle
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    accessed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    key = relationship("CryptoKeyStore", back_populates="encrypted_data")


class ZKProofRecord(Base):
    """Zero-knowledge proof records"""
    __tablename__ = "zk_proofs"
    
    # Primary identifiers
    proof_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    circuit_id = Column(String(255), nullable=False, index=True)
    
    # Proof details
    proof_system = Column(String(100), nullable=False)
    proof_data = Column(LargeBinary, nullable=False)
    public_inputs = Column(JSONB, default=list)
    verification_key = Column(LargeBinary, nullable=True)
    
    # Context
    prover_id = Column(String(255), nullable=False)
    statement = Column(Text, nullable=False)
    purpose = Column(String(255), nullable=False)
    
    # Verification status
    is_verified = Column(Boolean, default=False)
    verification_timestamp = Column(DateTime(timezone=True), nullable=True)
    verifier_id = Column(String(255), nullable=True)
    
    # Performance metrics
    generation_time_ms = Column(Integer, nullable=True)
    verification_time_ms = Column(Integer, nullable=True)
    proof_size_bytes = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional data
    metadata = Column(JSONB, default=dict)


class PrivacyPolicyRecord(Base):
    """Privacy policy records"""
    __tablename__ = "privacy_policies"
    
    # Primary identifiers
    policy_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_name = Column(String(255), nullable=False, unique=True)
    
    # Policy details
    privacy_level = Column(String(50), nullable=False)
    policy_rules = Column(JSONB, nullable=False)
    encryption_requirements = Column(JSONB, default=dict)
    access_controls = Column(JSONB, default=dict)
    
    # Compliance
    regulatory_frameworks = Column(JSONB, default=list)
    compliance_requirements = Column(JSONB, default=dict)
    data_retention_days = Column(Integer, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    version = Column(String(50), nullable=False, default="1.0")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    effective_from = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional data
    metadata = Column(JSONB, default=dict)


# === Pydantic Models ===

class CryptoKey(BaseModel):
    """Cryptographic key model"""
    key_id: str
    key_name: str
    key_type: KeyType
    algorithm: str
    key_usage: KeyUsage
    curve_type: Optional[CurveType] = None
    key_size: Optional[int] = None
    is_hardware_backed: bool = False
    is_exportable: bool = False
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EncryptionResult(BaseModel):
    """Encryption operation result"""
    success: bool
    encrypted_data: Optional[str] = None
    algorithm: Optional[EncryptionAlgorithm] = None
    initialization_vector: Optional[str] = None
    authentication_tag: Optional[str] = None
    key_id: Optional[str] = None
    content_hash: Optional[str] = None
    encryption_context: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    operation_time_ms: Optional[int] = None


class DecryptionResult(BaseModel):
    """Decryption operation result"""
    success: bool
    decrypted_data: Optional[str] = None
    content_verified: bool = False
    key_id: Optional[str] = None
    decryption_context: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    operation_time_ms: Optional[int] = None


class SignatureResult(BaseModel):
    """Digital signature result"""
    success: bool
    signature: Optional[str] = None
    algorithm: Optional[str] = None
    key_id: Optional[str] = None
    message_hash: Optional[str] = None
    signature_context: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    operation_time_ms: Optional[int] = None


class VerificationResult(BaseModel):
    """Signature verification result"""
    success: bool
    is_valid: bool = False
    signer_key_id: Optional[str] = None
    verification_context: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    operation_time_ms: Optional[int] = None


class ProofResult(BaseModel):
    """Zero-knowledge proof result"""
    success: bool
    proof_data: Optional[str] = None
    public_inputs: List[Any] = Field(default_factory=list)
    circuit_id: Optional[str] = None
    proof_system: Optional[str] = None
    generation_time_ms: Optional[int] = None
    proof_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class ZKProof(BaseModel):
    """Zero-knowledge proof model"""
    proof_id: str
    circuit_id: str
    proof_system: str
    proof_data: str
    public_inputs: List[Any]
    statement: str
    prover_id: str
    is_verified: bool = False
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PrivacyPolicy(BaseModel):
    """Privacy policy model"""
    policy_id: str
    policy_name: str
    privacy_level: PrivacyLevel
    policy_rules: Dict[str, Any]
    encryption_requirements: Dict[str, Any] = Field(default_factory=dict)
    access_controls: Dict[str, Any] = Field(default_factory=dict)
    regulatory_frameworks: List[str] = Field(default_factory=list)
    compliance_requirements: Dict[str, Any] = Field(default_factory=dict)
    data_retention_days: Optional[int] = None
    is_active: bool = True
    version: str = "1.0"
    created_at: datetime
    effective_from: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class SecureMessage(BaseModel):
    """Secure communication message"""
    message_id: str
    sender_id: str
    recipient_id: str
    encrypted_content: str
    encryption_algorithm: EncryptionAlgorithm
    signature: Optional[str] = None
    message_type: str
    privacy_level: PrivacyLevel
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KeyGenerationRequest(BaseModel):
    """Key generation request"""
    key_name: str
    key_type: KeyType
    algorithm: str
    key_usage: KeyUsage
    curve_type: Optional[CurveType] = None
    key_size: Optional[int] = None
    user_id: Optional[str] = None
    is_hardware_backed: bool = False
    is_exportable: bool = False
    expires_in_days: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class EncryptionRequest(BaseModel):
    """Encryption request"""
    data: str
    key_id: str
    algorithm: EncryptionAlgorithm
    content_type: str = "text/plain"
    privacy_level: PrivacyLevel = PrivacyLevel.CONFIDENTIAL
    encryption_context: Dict[str, Any] = Field(default_factory=dict)
    expires_in_days: Optional[int] = None


class DecryptionRequest(BaseModel):
    """Decryption request"""
    encrypted_data_id: str
    decryption_context: Dict[str, Any] = Field(default_factory=dict)


class SigningRequest(BaseModel):
    """Digital signing request"""
    data: str
    key_id: str
    algorithm: Optional[str] = None
    purpose: str
    context_data: Dict[str, Any] = Field(default_factory=dict)


class VerificationRequest(BaseModel):
    """Signature verification request"""
    signature_id: str
    original_data: str
    verification_context: Dict[str, Any] = Field(default_factory=dict)


class ZKProofRequest(BaseModel):
    """Zero-knowledge proof generation request"""
    circuit_id: str
    private_inputs: Dict[str, Any]
    public_inputs: List[Any] = Field(default_factory=list)
    statement: str
    purpose: str
    proof_system: str = "groth16"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CryptoSystemStatus(BaseModel):
    """Cryptographic system status"""
    system_health: str
    total_keys: int
    active_keys: int
    hardware_backed_keys: int
    total_encryptions: int
    total_signatures: int
    total_proofs: int
    key_rotation_due: int
    security_alerts: int
    compliance_status: str
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))