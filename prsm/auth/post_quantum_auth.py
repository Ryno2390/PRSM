"""
Post-Quantum Authentication Module
Integrates CRYSTALS-Dilithium / ML-DSA signatures with PRSM identity system

This module extends the existing authentication system with quantum-resistant
digital signatures for enhanced security against future quantum attacks.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from uuid import UUID, uuid4

# Import our post-quantum crypto directly
import importlib.util
from pathlib import Path

# Load post-quantum module
pq_module_path = Path(__file__).parent.parent / "cryptography" / "post_quantum.py"
spec = importlib.util.spec_from_file_location("post_quantum", pq_module_path)
pq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq_module)

# Import post-quantum classes
PostQuantumCrypto = pq_module.PostQuantumCrypto
PostQuantumKeyPair = pq_module.PostQuantumKeyPair
PostQuantumSignature = pq_module.PostQuantumSignature
SecurityLevel = pq_module.SecurityLevel
SignatureType = pq_module.SignatureType

# from .models import User, UserRole, Permission  # Optional for integration


class AuthSignatureType(str, Enum):
    """Authentication signature types"""
    TRADITIONAL = "traditional"     # RSA/ECDSA for legacy compatibility
    POST_QUANTUM = "post_quantum"   # ML-DSA for quantum resistance
    HYBRID = "hybrid"              # Both signatures for transition period


class PostQuantumIdentity(BaseModel):
    """Post-quantum identity with cryptographic keys"""
    user_id: UUID
    security_level: SecurityLevel
    pq_keypair: PostQuantumKeyPair
    signature_type: AuthSignatureType = AuthSignatureType.POST_QUANTUM
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "user_id": str(self.user_id),
            "security_level": self.security_level.value,
            "pq_keypair": self.pq_keypair.to_dict(),
            "signature_type": self.signature_type.value,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> 'PostQuantumIdentity':
        """Create from database dictionary"""
        return cls(
            user_id=UUID(data["user_id"]),
            security_level=SecurityLevel(data["security_level"]),
            pq_keypair=PostQuantumKeyPair.from_dict(data["pq_keypair"]),
            signature_type=AuthSignatureType(data["signature_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        )


@dataclass
class AuthenticationChallenge:
    """Authentication challenge for post-quantum signing"""
    challenge_id: str
    user_id: UUID
    challenge_data: str
    expires_at: datetime
    security_level: SecurityLevel
    signature_type: AuthSignatureType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            "challenge_id": self.challenge_id,
            "user_id": str(self.user_id),
            "challenge_data": self.challenge_data,
            "expires_at": self.expires_at.isoformat(),
            "security_level": self.security_level.value,
            "signature_type": self.signature_type.value
        }
    
    def is_expired(self) -> bool:
        """Check if challenge has expired"""
        return datetime.now(timezone.utc) > self.expires_at


class PostQuantumAuthManager:
    """
    Post-Quantum Authentication Manager
    
    Manages quantum-resistant authentication using ML-DSA signatures
    """
    
    def __init__(self, default_security_level: SecurityLevel = SecurityLevel.LEVEL_1):
        """
        Initialize post-quantum auth manager
        
        Args:
            default_security_level: Default security level for new identities
        """
        self.default_security_level = default_security_level
        self.pq_crypto = PostQuantumCrypto(default_security_level)
        
        # In-memory storage for demo (replace with database in production)
        self.identities: Dict[UUID, PostQuantumIdentity] = {}
        self.challenges: Dict[str, AuthenticationChallenge] = {}
    
    def create_post_quantum_identity(self, 
                                   user_id: UUID,
                                   security_level: Optional[SecurityLevel] = None,
                                   signature_type: AuthSignatureType = AuthSignatureType.POST_QUANTUM) -> PostQuantumIdentity:
        """
        Create a new post-quantum identity for a user
        
        Args:
            user_id: User ID to create identity for
            security_level: Post-quantum security level
            signature_type: Type of signatures to use
            
        Returns:
            PostQuantumIdentity object
        """
        if security_level is None:
            security_level = self.default_security_level
        
        # Generate post-quantum key pair
        pq_keypair = self.pq_crypto.generate_keypair(security_level)
        
        # Create identity
        identity = PostQuantumIdentity(
            user_id=user_id,
            security_level=security_level,
            pq_keypair=pq_keypair,
            signature_type=signature_type
        )
        
        # Store identity
        self.identities[user_id] = identity
        
        return identity
    
    def get_identity(self, user_id: UUID) -> Optional[PostQuantumIdentity]:
        """Get post-quantum identity for a user"""
        return self.identities.get(user_id)
    
    def create_auth_challenge(self, 
                            user_id: UUID,
                            challenge_lifetime_minutes: int = 5) -> Optional[AuthenticationChallenge]:
        """
        Create an authentication challenge for post-quantum signing
        
        Args:
            user_id: User ID to create challenge for
            challenge_lifetime_minutes: Challenge validity period
            
        Returns:
            AuthenticationChallenge object or None if user has no PQ identity
        """
        identity = self.get_identity(user_id)
        if not identity:
            return None
        
        # Generate unique challenge
        challenge_id = hashlib.sha256(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Create challenge data (timestamp + nonce)
        challenge_data = f"PRSM_AUTH_{datetime.now(timezone.utc).isoformat()}_{challenge_id}"
        
        # Set expiration
        expires_at = datetime.now(timezone.utc).replace(microsecond=0)
        expires_at = expires_at.replace(minute=expires_at.minute + challenge_lifetime_minutes)
        
        challenge = AuthenticationChallenge(
            challenge_id=challenge_id,
            user_id=user_id,
            challenge_data=challenge_data,
            expires_at=expires_at,
            security_level=identity.security_level,
            signature_type=identity.signature_type
        )
        
        # Store challenge
        self.challenges[challenge_id] = challenge
        
        return challenge
    
    def verify_auth_signature(self, 
                            challenge_id: str,
                            signature_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Verify post-quantum authentication signature
        
        Args:
            challenge_id: Challenge ID
            signature_data: Signature data from client
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Get challenge
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            return False, "Challenge not found"
        
        # Check if challenge expired
        if challenge.is_expired():
            del self.challenges[challenge_id]
            return False, "Challenge expired"
        
        # Get user identity
        identity = self.get_identity(challenge.user_id)
        if not identity:
            return False, "User identity not found"
        
        try:
            # Reconstruct signature object
            signature = PostQuantumSignature.from_dict(signature_data)
            
            # Verify signature
            is_valid = self.pq_crypto.verify_signature(
                challenge.challenge_data,
                signature,
                identity.pq_keypair.public_key
            )
            
            if is_valid:
                # Update last used time
                identity.last_used = datetime.now(timezone.utc)
                
                # Clean up challenge
                del self.challenges[challenge_id]
                
                return True, "Authentication successful"
            else:
                return False, "Invalid signature"
                
        except Exception as e:
            return False, f"Signature verification error: {str(e)}"
    
    def sign_authentication_challenge(self, 
                                    user_id: UUID,
                                    challenge_data: str) -> Optional[PostQuantumSignature]:
        """
        Sign an authentication challenge (used client-side or for testing)
        
        Args:
            user_id: User ID
            challenge_data: Challenge data to sign
            
        Returns:
            PostQuantumSignature or None if user has no identity
        """
        identity = self.get_identity(user_id)
        if not identity:
            return None
        
        # Sign the challenge
        signature = self.pq_crypto.sign_message(challenge_data, identity.pq_keypair)
        
        return signature
    
    def get_user_public_key(self, user_id: UUID) -> Optional[bytes]:
        """Get user's post-quantum public key"""
        identity = self.get_identity(user_id)
        return identity.pq_keypair.public_key if identity else None
    
    def upgrade_security_level(self, 
                             user_id: UUID,
                             new_security_level: SecurityLevel) -> bool:
        """
        Upgrade user's post-quantum security level
        
        Args:
            user_id: User ID
            new_security_level: New security level
            
        Returns:
            True if upgrade successful, False otherwise
        """
        identity = self.get_identity(user_id)
        if not identity:
            return False
        
        # Create new keypair with higher security level
        new_keypair = self.pq_crypto.generate_keypair(new_security_level)
        
        # Update identity
        identity.pq_keypair = new_keypair
        identity.security_level = new_security_level
        identity.created_at = datetime.now(timezone.utc)
        identity.last_used = None
        
        return True
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication system statistics"""
        security_level_counts = {}
        signature_type_counts = {}
        
        for identity in self.identities.values():
            # Count security levels
            level = identity.security_level.value
            security_level_counts[level] = security_level_counts.get(level, 0) + 1
            
            # Count signature types
            sig_type = identity.signature_type.value
            signature_type_counts[sig_type] = signature_type_counts.get(sig_type, 0) + 1
        
        active_challenges = len([c for c in self.challenges.values() if not c.is_expired()])
        
        return {
            "total_identities": len(self.identities),
            "security_level_distribution": security_level_counts,
            "signature_type_distribution": signature_type_counts,
            "active_challenges": active_challenges,
            "total_challenges": len(self.challenges),
            "post_quantum_enabled": True,
            "default_security_level": self.default_security_level.value
        }


# Global instance for easy access
_pq_auth_manager: Optional[PostQuantumAuthManager] = None


def get_post_quantum_auth_manager(security_level: SecurityLevel = SecurityLevel.LEVEL_1) -> PostQuantumAuthManager:
    """
    Get or create global post-quantum auth manager
    
    Args:
        security_level: Default security level
        
    Returns:
        PostQuantumAuthManager instance
    """
    global _pq_auth_manager
    if _pq_auth_manager is None:
        _pq_auth_manager = PostQuantumAuthManager(security_level)
    return _pq_auth_manager


def reset_post_quantum_auth_manager():
    """Reset the global post-quantum auth manager"""
    global _pq_auth_manager
    _pq_auth_manager = None


# Example authentication flow
async def example_auth_flow():
    """Example post-quantum authentication flow"""
    print("🔐 PRSM Post-Quantum Authentication Example")
    print("=" * 50)
    
    # Initialize auth manager
    auth_manager = get_post_quantum_auth_manager(SecurityLevel.LEVEL_1)
    
    # Create user and post-quantum identity
    user_id = uuid4()
    print(f"1. Creating post-quantum identity for user {str(user_id)[:8]}...")
    
    identity = auth_manager.create_post_quantum_identity(
        user_id=user_id,
        security_level=SecurityLevel.LEVEL_1
    )
    print(f"   ✅ Created identity with key ID: {identity.pq_keypair.key_id}")
    
    # Create authentication challenge
    print("\n2. Creating authentication challenge...")
    challenge = auth_manager.create_auth_challenge(user_id)
    if challenge:
        print(f"   ✅ Challenge created: {challenge.challenge_id}")
        print(f"   Challenge data: {challenge.challenge_data[:50]}...")
        print(f"   Expires at: {challenge.expires_at}")
    
    # Sign challenge (simulating client-side signing)
    print("\n3. Signing challenge...")
    signature = auth_manager.sign_authentication_challenge(user_id, challenge.challenge_data)
    if signature:
        print(f"   ✅ Challenge signed")
        print(f"   Signature size: {len(signature.signature)} bytes")
        print(f"   Signature type: {signature.signature_type.value}")
    
    # Verify signature
    print("\n4. Verifying authentication...")
    is_valid, message = auth_manager.verify_auth_signature(
        challenge.challenge_id,
        signature.to_dict()
    )
    print(f"   Authentication result: {message}")
    print(f"   Valid: {is_valid}")
    
    # Get stats
    print("\n5. Authentication system statistics:")
    stats = auth_manager.get_authentication_stats()
    print(f"   Total identities: {stats['total_identities']}")
    print(f"   Security levels: {stats['security_level_distribution']}")
    print(f"   Post-quantum enabled: {stats['post_quantum_enabled']}")
    
    print("\n✅ Post-quantum authentication flow complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_auth_flow())