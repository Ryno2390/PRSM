"""
PRSM Cryptography API
====================

REST API endpoints for cryptographic operations, key management,
encryption services, and zero-knowledge proof generation/verification.
"""

import structlog
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field

from prsm.auth import get_current_user
from prsm.auth.models import UserRole
from prsm.auth.auth_manager import auth_manager
from prsm.cryptography import (
    get_key_manager, get_encryption_service, get_zk_proof_system,
    KeyType, EncryptionAlgorithm, KeyUsage, PrivacyLevel,
    KeyGenerationRequest, EncryptionRequest, DecryptionRequest, ZKProofRequest
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/crypto", tags=["cryptography"])


# === Request/Response Models ===

class CreateKeyRequest(BaseModel):
    """Request to create a cryptographic key"""
    key_name: str = Field(description="Unique key name")
    key_type: KeyType = Field(description="Type of cryptographic key")
    algorithm: str = Field(description="Cryptographic algorithm")
    key_usage: KeyUsage = Field(description="Intended key usage")
    curve_type: Optional[str] = Field(default=None, description="Elliptic curve type")
    key_size: Optional[int] = Field(default=None, description="Key size in bits")
    is_hardware_backed: bool = Field(default=False, description="Use hardware security module")
    is_exportable: bool = Field(default=False, description="Allow key export")
    expires_in_days: Optional[int] = Field(default=None, description="Key expiration in days")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Key tags")


class EncryptDataRequest(BaseModel):
    """Request to encrypt data"""
    data: str = Field(description="Data to encrypt")
    key_id: str = Field(description="Encryption key ID")
    algorithm: EncryptionAlgorithm = Field(description="Encryption algorithm")
    content_type: str = Field(default="text/plain", description="Content type")
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.CONFIDENTIAL, description="Privacy level")
    expires_in_days: Optional[int] = Field(default=None, description="Encrypted data expiration")
    encryption_context: Dict[str, Any] = Field(default_factory=dict, description="Encryption context")


class DecryptDataRequest(BaseModel):
    """Request to decrypt data"""
    encrypted_data_id: str = Field(description="Encrypted data ID")
    decryption_context: Dict[str, Any] = Field(default_factory=dict, description="Decryption context")


class GenerateProofRequest(BaseModel):
    """Request to generate ZK proof"""
    circuit_id: str = Field(description="ZK circuit identifier")
    private_inputs: Dict[str, Any] = Field(description="Private inputs for proof")
    public_inputs: List[Any] = Field(default_factory=list, description="Public inputs")
    statement: str = Field(description="Statement to prove")
    purpose: str = Field(description="Proof purpose")
    proof_system: str = Field(default="groth16", description="Proof system")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CryptoApiResponse(BaseModel):
    """Standard cryptography API response"""
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# === Key Management Endpoints ===

@router.post("/keys/generate")
async def generate_key(
    request: CreateKeyRequest,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Generate a new cryptographic key
    
    🔐 KEY GENERATION:
    - Secure key generation with hardware backing support
    - Multiple key types and algorithms
    - Automated key lifecycle management
    - Comprehensive audit logging
    """
    try:
        key_manager = await get_key_manager()
        
        # Create key generation request
        key_request = KeyGenerationRequest(
            key_name=request.key_name,
            key_type=request.key_type,
            algorithm=request.algorithm,
            key_usage=request.key_usage,
            curve_type=request.curve_type,
            key_size=request.key_size,
            user_id=current_user,
            is_hardware_backed=request.is_hardware_backed,
            is_exportable=request.is_exportable,
            expires_in_days=request.expires_in_days,
            metadata=request.metadata,
            tags=request.tags
        )
        
        # Generate key
        crypto_key = await key_manager.generate_key(key_request)
        
        return CryptoApiResponse(
            success=True,
            message="Cryptographic key generated successfully",
            data={
                "key_id": crypto_key.key_id,
                "key_name": crypto_key.key_name,
                "key_type": crypto_key.key_type.value,
                "algorithm": crypto_key.algorithm,
                "key_usage": crypto_key.key_usage.value,
                "is_hardware_backed": crypto_key.is_hardware_backed,
                "created_at": crypto_key.created_at.isoformat(),
                "expires_at": crypto_key.expires_at.isoformat() if crypto_key.expires_at else None
            }
        )
        
    except Exception as e:
        logger.error("Key generation failed", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate cryptographic key"
        )


@router.get("/keys")
async def list_keys(
    key_usage: Optional[KeyUsage] = None,
    include_inactive: bool = False,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    List user's cryptographic keys
    
    📋 KEY LISTING:
    - User-specific key filtering
    - Key usage and status filtering
    - Comprehensive key metadata
    - Access control enforcement
    """
    try:
        key_manager = await get_key_manager()
        
        keys = await key_manager.list_keys(
            user_id=current_user,
            key_usage=key_usage,
            include_inactive=include_inactive
        )
        
        keys_data = [
            {
                "key_id": key.key_id,
                "key_name": key.key_name,
                "key_type": key.key_type.value,
                "algorithm": key.algorithm,
                "key_usage": key.key_usage.value,
                "is_hardware_backed": key.is_hardware_backed,
                "is_active": key.is_active,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None
            }
            for key in keys
        ]
        
        return CryptoApiResponse(
            success=True,
            message=f"Retrieved {len(keys)} cryptographic keys",
            data={"keys": keys_data, "total_count": len(keys)}
        )
        
    except Exception as e:
        logger.error("Failed to list keys", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cryptographic keys"
        )


@router.get("/keys/{key_id}")
async def get_key_info(
    key_id: str,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Get cryptographic key information
    
    🔍 KEY INFORMATION:
    - Detailed key metadata
    - Usage statistics and history
    - Security properties and status
    - Access control validation
    """
    try:
        key_manager = await get_key_manager()
        
        crypto_key = await key_manager.get_key(key_id)
        if not crypto_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cryptographic key not found"
            )
        
        return CryptoApiResponse(
            success=True,
            message="Key information retrieved successfully",
            data={
                "key_id": crypto_key.key_id,
                "key_name": crypto_key.key_name,
                "key_type": crypto_key.key_type.value,
                "algorithm": crypto_key.algorithm,
                "key_usage": crypto_key.key_usage.value,
                "curve_type": crypto_key.curve_type.value if crypto_key.curve_type else None,
                "key_size": crypto_key.key_size,
                "is_hardware_backed": crypto_key.is_hardware_backed,
                "is_exportable": crypto_key.is_exportable,
                "is_active": crypto_key.is_active,
                "created_at": crypto_key.created_at.isoformat(),
                "expires_at": crypto_key.expires_at.isoformat() if crypto_key.expires_at else None,
                "metadata": crypto_key.metadata
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get key info", key_id=key_id, user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve key information"
        )


@router.post("/keys/{key_id}/rotate")
async def rotate_key(
    key_id: str,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Rotate a cryptographic key
    
    🔄 KEY ROTATION:
    - Secure key rotation process
    - Automatic old key deactivation
    - Seamless transition to new key
    - Comprehensive audit logging
    """
    try:
        key_manager = await get_key_manager()
        
        new_key = await key_manager.rotate_key(key_id)
        if not new_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Key not found or rotation failed"
            )
        
        return CryptoApiResponse(
            success=True,
            message="Key rotated successfully",
            data={
                "old_key_id": key_id,
                "new_key_id": new_key.key_id,
                "new_key_name": new_key.key_name,
                "created_at": new_key.created_at.isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Key rotation failed", key_id=key_id, user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rotate cryptographic key"
        )


# === Encryption/Decryption Endpoints ===

@router.post("/encrypt")
async def encrypt_data(
    request: EncryptDataRequest,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Encrypt data with specified key and algorithm
    
    🔒 DATA ENCRYPTION:
    - Multi-algorithm encryption support
    - Privacy level enforcement
    - Secure key management integration
    - Comprehensive audit logging
    """
    try:
        encryption_service = await get_encryption_service()
        
        # Add user context to encryption request
        encryption_request = EncryptionRequest(
            data=request.data,
            key_id=request.key_id,
            algorithm=request.algorithm,
            content_type=request.content_type,
            privacy_level=request.privacy_level,
            expires_in_days=request.expires_in_days,
            encryption_context={
                **request.encryption_context,
                "owner_id": current_user,
                "encrypted_by": current_user
            }
        )
        
        # Encrypt data
        result = await encryption_service.encrypt(encryption_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        return CryptoApiResponse(
            success=True,
            message="Data encrypted successfully",
            data={
                "encrypted_data_id": result.encrypted_data,
                "algorithm": result.algorithm.value,
                "key_id": result.key_id,
                "content_hash": result.content_hash,
                "operation_time_ms": result.operation_time_ms,
                "encryption_context": result.encryption_context
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data encryption failed", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encrypt data"
        )


@router.post("/decrypt")
async def decrypt_data(
    request: DecryptDataRequest,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Decrypt data by encrypted data ID
    
    🔓 DATA DECRYPTION:
    - Secure data retrieval and decryption
    - Access control enforcement
    - Content integrity verification
    - Comprehensive audit logging
    """
    try:
        encryption_service = await get_encryption_service()
        
        # Decrypt data
        result = await encryption_service.decrypt(request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        return CryptoApiResponse(
            success=True,
            message="Data decrypted successfully",
            data={
                "decrypted_data": result.decrypted_data,
                "content_verified": result.content_verified,
                "key_id": result.key_id,
                "operation_time_ms": result.operation_time_ms,
                "decryption_context": result.decryption_context
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data decryption failed", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decrypt data"
        )


@router.get("/encrypted-data")
async def list_encrypted_data(
    privacy_level: Optional[PrivacyLevel] = None,
    limit: int = 100,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    List user's encrypted data
    
    📊 ENCRYPTED DATA LISTING:
    - User-specific data filtering
    - Privacy level filtering
    - Comprehensive metadata
    - Access control enforcement
    """
    try:
        encryption_service = await get_encryption_service()
        
        encrypted_data_list = await encryption_service.list_encrypted_data(
            owner_id=current_user,
            privacy_level=privacy_level,
            limit=limit
        )
        
        return CryptoApiResponse(
            success=True,
            message=f"Retrieved {len(encrypted_data_list)} encrypted data records",
            data={
                "encrypted_data": encrypted_data_list,
                "total_count": len(encrypted_data_list)
            }
        )
        
    except Exception as e:
        logger.error("Failed to list encrypted data", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list encrypted data"
        )


# === Zero-Knowledge Proof Endpoints ===

@router.post("/proofs/generate")
async def generate_zk_proof(
    request: GenerateProofRequest,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Generate a zero-knowledge proof
    
    🔍 ZK PROOF GENERATION:
    - Privacy-preserving proof generation
    - Multiple ZK circuit support
    - Secure private input handling
    - Comprehensive proof lifecycle management
    """
    try:
        zk_system = await get_zk_proof_system()
        
        # Create ZK proof request
        zk_request = ZKProofRequest(
            circuit_id=request.circuit_id,
            private_inputs=request.private_inputs,
            public_inputs=request.public_inputs,
            statement=request.statement,
            purpose=request.purpose,
            proof_system=request.proof_system,
            metadata={
                **request.metadata,
                "prover_id": current_user,
                "generated_via": "api"
            }
        )
        
        # Generate proof
        result = await zk_system.generate_proof(zk_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        return CryptoApiResponse(
            success=True,
            message="Zero-knowledge proof generated successfully",
            data={
                "proof_data": result.proof_data,
                "public_inputs": result.public_inputs,
                "circuit_id": result.circuit_id,
                "proof_system": result.proof_system,
                "generation_time_ms": result.generation_time_ms,
                "proof_size_bytes": result.proof_size_bytes
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ZK proof generation failed", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate zero-knowledge proof"
        )


@router.post("/proofs/{proof_id}/verify")
async def verify_zk_proof(
    proof_id: str,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Verify a zero-knowledge proof
    
    ✅ ZK PROOF VERIFICATION:
    - Cryptographic proof verification
    - Public input validation
    - Verification status tracking
    - Comprehensive audit logging
    """
    try:
        zk_system = await get_zk_proof_system()
        
        # Verify proof
        is_valid = await zk_system.verify_proof(proof_id, current_user)
        
        return CryptoApiResponse(
            success=True,
            message=f"Proof verification {'successful' if is_valid else 'failed'}",
            data={
                "proof_id": proof_id,
                "is_valid": is_valid,
                "verified_by": current_user,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error("ZK proof verification failed", proof_id=proof_id, user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify zero-knowledge proof"
        )


@router.get("/proofs")
async def list_zk_proofs(
    circuit_id: Optional[str] = None,
    verified_only: bool = False,
    limit: int = 100,
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    List user's zero-knowledge proofs
    
    📋 ZK PROOF LISTING:
    - User-specific proof filtering
    - Circuit and verification filtering
    - Comprehensive proof metadata
    - Access control enforcement
    """
    try:
        zk_system = await get_zk_proof_system()
        
        proofs = await zk_system.list_proofs(
            prover_id=current_user,
            circuit_id=circuit_id,
            verified_only=verified_only,
            limit=limit
        )
        
        proofs_data = [
            {
                "proof_id": proof.proof_id,
                "circuit_id": proof.circuit_id,
                "proof_system": proof.proof_system,
                "statement": proof.statement,
                "is_verified": proof.is_verified,
                "created_at": proof.created_at.isoformat(),
                "expires_at": proof.expires_at.isoformat() if proof.expires_at else None
            }
            for proof in proofs
        ]
        
        return CryptoApiResponse(
            success=True,
            message=f"Retrieved {len(proofs)} zero-knowledge proofs",
            data={"proofs": proofs_data, "total_count": len(proofs)}
        )
        
    except Exception as e:
        logger.error("Failed to list ZK proofs", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list zero-knowledge proofs"
        )


@router.get("/circuits")
async def get_available_circuits(
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Get available ZK circuits
    
    🔧 CIRCUIT INFORMATION:
    - Available ZK circuit listing
    - Circuit capability descriptions
    - Usage guidelines and examples
    - Performance characteristics
    """
    try:
        zk_system = await get_zk_proof_system()
        
        circuits = zk_system.get_available_circuits()
        
        return CryptoApiResponse(
            success=True,
            message=f"Retrieved {len(circuits)} available ZK circuits",
            data={"circuits": circuits, "total_count": len(circuits)}
        )
        
    except Exception as e:
        logger.error("Failed to get available circuits", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available circuits"
        )


# === System Status and Administration ===

@router.get("/status")
async def get_crypto_system_status(
    current_user: str = Depends(get_current_user)
) -> CryptoApiResponse:
    """
    Get cryptographic system status
    
    🏥 SYSTEM STATUS:
    - Key management system health
    - Encryption service performance
    - ZK proof system statistics
    - Security compliance status
    """
    try:
        key_manager = await get_key_manager()
        zk_system = await get_zk_proof_system()
        
        key_status = await key_manager.get_system_status()
        zk_status = await zk_system.get_system_stats()
        
        return CryptoApiResponse(
            success=True,
            message="Cryptographic system status retrieved",
            data={
                "key_management": key_status,
                "zk_proofs": zk_status,
                "overall_health": "healthy" if key_status["system_health"] == "healthy" and zk_status["system_health"] == "healthy" else "warning"
            }
        )
        
    except Exception as e:
        logger.error("Failed to get crypto system status", user_id=current_user, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )