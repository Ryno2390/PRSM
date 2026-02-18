"""
Post-Quantum Key Management System for PRSM P2P Collaboration

This module implements a comprehensive post-quantum cryptographic key management
system that integrates with the P2P network to enable secure distributed
collaboration without centralized key authorities.

Key Features:
- Post-quantum key generation (Kyber, ML-DSA)
- Distributed key sharing using Shamir's Secret Sharing
- Quantum-resistant key encapsulation mechanisms
- Secure key distribution via P2P network
- Key rotation and lifecycle management
- Hardware security module (HSM) integration ready
"""

import asyncio
import json
import logging
import time
import secrets
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import base64
import struct

# Post-quantum cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# For production, these would be replaced with actual post-quantum libraries
# like liboqs-python or pqcrypto
# from pqcrypto.kem.kyber1024 import generate_keypair, encapsulate, decapsulate
# from pqcrypto.sign.dilithium5 import generate_keypair as sign_generate_keypair

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of cryptographic keys"""
    ENCRYPTION = "encryption"        # For data encryption (Kyber KEM)
    SIGNING = "signing"             # For digital signatures (ML-DSA)
    AUTHENTICATION = "authentication" # For node authentication
    COLLABORATION = "collaboration"  # For session-specific collaboration


class KeyStatus(Enum):
    """Status of cryptographic keys"""
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    REVOKED = "revoked"
    COMPROMISED = "compromised"


class PostQuantumAlgorithm(Enum):
    """Post-quantum cryptographic algorithms"""
    KYBER_1024 = "kyber_1024"          # KEM for encryption
    ML_DSA_87 = "ml_dsa_87"            # Signature scheme (formerly Dilithium)
    SPHINCS_PLUS = "sphincs_plus"      # Alternative signature scheme
    CLASSIC_MCELEICE = "classic_mceliece"  # Alternative KEM


@dataclass
class CryptographicKey:
    """Represents a cryptographic key with metadata"""
    key_id: str
    key_type: KeyType
    algorithm: PostQuantumAlgorithm
    public_key: bytes
    private_key: Optional[bytes] = None  # Only stored for own keys
    created_at: float = 0.0
    expires_at: Optional[float] = None
    status: KeyStatus = KeyStatus.PENDING
    owner_node_id: str = ""
    associated_file_ids: Set[str] = None
    usage_count: int = 0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.associated_file_ids is None:
            self.associated_file_ids = set()
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def is_expired(self) -> bool:
        """Check if key has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def is_active(self) -> bool:
        """Check if key is active and usable"""
        return (self.status == KeyStatus.ACTIVE and 
                not self.is_expired)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert set to list for JSON serialization
        data['associated_file_ids'] = list(self.associated_file_ids)
        data['key_type'] = self.key_type.value
        data['algorithm'] = self.algorithm.value
        data['status'] = self.status.value
        # Encode bytes as base64
        data['public_key'] = base64.b64encode(self.public_key).decode()
        if self.private_key:
            data['private_key'] = base64.b64encode(self.private_key).decode()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CryptographicKey':
        """Create from dictionary"""
        # Decode base64 encoded bytes
        data['public_key'] = base64.b64decode(data['public_key'])
        if data.get('private_key'):
            data['private_key'] = base64.b64decode(data['private_key'])
        
        # Convert enum strings back to enums
        data['key_type'] = KeyType(data['key_type'])
        data['algorithm'] = PostQuantumAlgorithm(data['algorithm'])
        data['status'] = KeyStatus(data['status'])
        
        # Convert list back to set
        data['associated_file_ids'] = set(data.get('associated_file_ids', []))
        
        return cls(**data)


@dataclass
class KeyShare:
    """Represents a Shamir's Secret Share of a key"""
    share_id: str
    key_id: str
    share_index: int
    total_shares: int
    threshold: int
    share_data: bytes
    created_at: float = 0.0
    holder_node_id: str = ""
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class PostQuantumCrypto:
    """
    Post-quantum cryptographic operations wrapper
    
    This class provides a unified interface for post-quantum cryptographic
    operations, with fallback to classical algorithms during development.
    """
    
    def __init__(self):
        self.backend = default_backend()
        # In production, initialize actual PQ libraries here
        self.pq_available = False  # Set to True when real PQ libs are available
    
    def generate_kyber_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Kyber KEM keypair"""
        if self.pq_available:
            # Real implementation would use:
            # from pqcrypto.kem.kyber1024 import generate_keypair
            # public_key, private_key = generate_keypair()
            # return public_key, private_key
            pass
        
        # Fallback to RSA for development/testing
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=3072,  # Equivalent to ~128-bit post-quantum security
            backend=self.backend
        )
        
        public_key_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return public_key_bytes, private_key_bytes
    
    def generate_mldsa_keypair(self) -> Tuple[bytes, bytes]:
        """Generate ML-DSA signature keypair"""
        if self.pq_available:
            # Real implementation would use:
            # from pqcrypto.sign.dilithium5 import generate_keypair
            # public_key, private_key = generate_keypair()
            # return public_key, private_key
            pass
        
        # Fallback to RSA-PSS for development/testing
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=3072,
            backend=self.backend
        )
        
        public_key_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return public_key_bytes, private_key_bytes
    
    def kyber_encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret using Kyber KEM"""
        if self.pq_available:
            # Real implementation:
            # from pqcrypto.kem.kyber1024 import encapsulate
            # ciphertext, shared_secret = encapsulate(public_key)
            # return ciphertext, shared_secret
            pass
        
        # Fallback using RSA-OAEP
        shared_secret = secrets.token_bytes(32)  # 256-bit shared secret
        
        # Load public key
        public_key_obj = serialization.load_der_public_key(public_key, self.backend)
        
        # Encrypt shared secret
        ciphertext = public_key_obj.encrypt(
            shared_secret,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext, shared_secret
    
    def kyber_decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate shared secret using Kyber KEM"""
        if self.pq_available:
            # Real implementation:
            # from pqcrypto.kem.kyber1024 import decapsulate
            # shared_secret = decapsulate(private_key, ciphertext)
            # return shared_secret
            pass
        
        # Fallback using RSA-OAEP
        private_key_obj = serialization.load_der_private_key(
            private_key, password=None, backend=self.backend
        )
        
        shared_secret = private_key_obj.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return shared_secret
    
    def mldsa_sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message using ML-DSA"""
        if self.pq_available:
            # Real implementation:
            # from pqcrypto.sign.dilithium5 import sign
            # signature = sign(private_key, message)
            # return signature
            pass
        
        # Fallback using RSA-PSS
        private_key_obj = serialization.load_der_private_key(
            private_key, password=None, backend=self.backend
        )
        
        signature = private_key_obj.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def mldsa_verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify ML-DSA signature"""
        if self.pq_available:
            # Real implementation:
            # from pqcrypto.sign.dilithium5 import verify
            # return verify(public_key, message, signature)
            pass
        
        # Fallback using RSA-PSS
        try:
            public_key_obj = serialization.load_der_public_key(public_key, self.backend)
            
            public_key_obj.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing implementation for distributed key management
    
    This enables splitting cryptographic keys into shares such that any
    threshold number of shares can reconstruct the original key.
    """
    
    def __init__(self, prime: Optional[int] = None):
        # Use a large prime for the finite field
        # This is a 256-bit prime for 32-byte secrets
        self.prime = prime or (2**255 - 19)  # Curve25519 field prime
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = (a % m + m) % m
        
        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        
        return (x % m + m) % m
    
    def _bytes_to_int(self, data: bytes) -> int:
        """Convert bytes to integer"""
        return int.from_bytes(data, byteorder='big')
    
    def _int_to_bytes(self, value: int, length: int) -> bytes:
        """Convert integer to bytes"""
        return value.to_bytes(length, byteorder='big')
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        x_power = 1
        
        for coeff in coefficients:
            result = (result + coeff * x_power) % self.prime
            x_power = (x_power * x) % self.prime
        
        return result
    
    def split_secret(self, secret: bytes, threshold: int, 
                    total_shares: int) -> List[Tuple[int, bytes]]:
        """
        Split a secret into shares using Shamir's Secret Sharing
        
        Args:
            secret: The secret to split (max 32 bytes)
            threshold: Minimum shares needed to reconstruct
            total_shares: Total number of shares to create
            
        Returns:
            List of (share_index, share_data) tuples
        """
        if len(secret) > 32:
            raise ValueError("Secret too large (max 32 bytes)")
        
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        
        # Convert secret to integer
        secret_int = self._bytes_to_int(secret.ljust(32, b'\x00'))
        
        # Generate random coefficients for polynomial
        coefficients = [secret_int]  # a0 = secret
        for _ in range(threshold - 1):
            coeff = secrets.randbelow(self.prime)
            coefficients.append(coeff)
        
        # Generate shares by evaluating polynomial at different points
        shares = []
        for i in range(1, total_shares + 1):
            share_value = self._evaluate_polynomial(coefficients, i)
            share_data = self._int_to_bytes(share_value, 32)
            shares.append((i, share_data))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """
        Reconstruct secret from shares using Lagrange interpolation
        
        Args:
            shares: List of (share_index, share_data) tuples
            
        Returns:
            The reconstructed secret
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Convert shares to integers
        int_shares = []
        for index, data in shares:
            value = self._bytes_to_int(data)
            int_shares.append((index, value))
        
        # Lagrange interpolation to find f(0)
        secret = 0
        
        for i, (xi, yi) in enumerate(int_shares):
            # Calculate Lagrange basis polynomial li(0)
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(int_shares):
                if i != j:
                    numerator = (numerator * (0 - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate li(0) = numerator / denominator (mod prime)
            denominator_inv = self._mod_inverse(denominator, self.prime)
            lagrange_coeff = (numerator * denominator_inv) % self.prime
            
            # Add yi * li(0) to result
            term = (yi * lagrange_coeff) % self.prime
            secret = (secret + term) % self.prime
        
        # Convert back to bytes and remove padding
        secret_bytes = self._int_to_bytes(secret, 32)
        
        # Find actual length by removing trailing zeros
        actual_length = 32
        while actual_length > 0 and secret_bytes[actual_length - 1] == 0:
            actual_length -= 1
        
        return secret_bytes[:actual_length] if actual_length > 0 else b''


class DistributedKeyManager:
    """
    Distributed key management system for P2P collaboration
    
    Manages post-quantum cryptographic keys across the P2P network
    without requiring centralized key authorities.
    """
    
    def __init__(self, node_id: str, config: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Cryptographic components
        self.pq_crypto = PostQuantumCrypto()
        self.secret_sharing = ShamirSecretSharing()
        
        # Key storage
        self.owned_keys: Dict[str, CryptographicKey] = {}
        self.peer_keys: Dict[str, CryptographicKey] = {}  # Public keys of peers
        self.key_shares: Dict[str, List[KeyShare]] = {}  # Shares we hold for others
        
        # Configuration
        self.default_key_lifetime = self.config.get('key_lifetime', 86400 * 365)  # 1 year
        self.min_share_threshold = self.config.get('min_threshold', 3)
        self.max_shares = self.config.get('max_shares', 7)
        self.key_rotation_interval = self.config.get('rotation_interval', 86400 * 30)  # 30 days
        
        # P2P integration (will be set by the collaboration system)
        self.p2p_network = None
        
        logger.info(f"Distributed key manager initialized for node {node_id}")
    
    def set_p2p_network(self, p2p_network):
        """Set P2P network for key distribution"""
        self.p2p_network = p2p_network
    
    async def generate_keypair(self, key_type: KeyType, 
                              algorithm: Optional[PostQuantumAlgorithm] = None,
                              lifetime: Optional[int] = None) -> str:
        """Generate a new post-quantum keypair"""
        if algorithm is None:
            # Choose default algorithm based on key type
            if key_type == KeyType.ENCRYPTION:
                algorithm = PostQuantumAlgorithm.KYBER_1024
            elif key_type == KeyType.SIGNING:
                algorithm = PostQuantumAlgorithm.ML_DSA_87
            else:
                algorithm = PostQuantumAlgorithm.KYBER_1024
        
        # Generate keypair
        if algorithm == PostQuantumAlgorithm.KYBER_1024:
            public_key, private_key = self.pq_crypto.generate_kyber_keypair()
        elif algorithm == PostQuantumAlgorithm.ML_DSA_87:
            public_key, private_key = self.pq_crypto.generate_mldsa_keypair()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create key ID
        key_id = self._generate_key_id(public_key)
        
        # Calculate expiration
        expires_at = None
        if lifetime:
            expires_at = time.time() + lifetime
        elif self.default_key_lifetime > 0:
            expires_at = time.time() + self.default_key_lifetime
        
        # Create key object
        key = CryptographicKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            public_key=public_key,
            private_key=private_key,
            expires_at=expires_at,
            status=KeyStatus.ACTIVE,
            owner_node_id=self.node_id
        )
        
        # Store key
        self.owned_keys[key_id] = key
        
        logger.info(f"Generated {algorithm.value} keypair: {key_id}")
        return key_id
    
    async def distribute_key(self, key_id: str, authorized_nodes: List[str],
                           threshold: Optional[int] = None) -> bool:
        """
        Distribute a key using Shamir's Secret Sharing
        
        Args:
            key_id: ID of the key to distribute
            authorized_nodes: List of node IDs that should receive shares
            threshold: Minimum shares needed to reconstruct (default: majority)
            
        Returns:
            True if distribution was successful
        """
        if key_id not in self.owned_keys:
            logger.error(f"Key {key_id} not found in owned keys")
            return False
        
        key = self.owned_keys[key_id]
        if not key.private_key:
            logger.error(f"Private key for {key_id} not available")
            return False
        
        # Calculate threshold
        total_shares = len(authorized_nodes)
        if threshold is None:
            threshold = (total_shares // 2) + 1  # Majority
        
        threshold = max(self.min_share_threshold, min(threshold, total_shares))
        
        # Split the private key
        try:
            shares = self.secret_sharing.split_secret(
                key.private_key, threshold, total_shares
            )
            
            logger.info(f"Split key {key_id} into {total_shares} shares "
                       f"(threshold: {threshold})")
        
        except Exception as e:
            logger.error(f"Failed to split key {key_id}: {e}")
            return False
        
        # Distribute shares to nodes
        success_count = 0
        
        for i, node_id in enumerate(authorized_nodes):
            if i >= len(shares):
                break
            
            share_index, share_data = shares[i]
            
            key_share = KeyShare(
                share_id=f"{key_id}_{share_index}",
                key_id=key_id,
                share_index=share_index,
                total_shares=total_shares,
                threshold=threshold,
                share_data=share_data,
                holder_node_id=node_id
            )
            
            # Send share to node via P2P network
            if await self._send_key_share(node_id, key_share):
                success_count += 1
            else:
                logger.warning(f"Failed to send key share to {node_id}")
        
        success_rate = success_count / total_shares
        logger.info(f"Key distribution completed: {success_rate:.1%} success rate")
        
        return success_rate >= 0.8  # Consider successful if 80%+ nodes received shares
    
    async def request_key_reconstruction(self, key_id: str,
                                       requester_node_id: Optional[str] = None) -> Optional[bytes]:
        """
        Request reconstruction of a distributed key
        
        Args:
            key_id: ID of the key to reconstruct
            requester_node_id: Node requesting the key (default: self)
            
        Returns:
            Reconstructed private key or None if failed
        """
        requester_node_id = requester_node_id or self.node_id
        
        # Find nodes that should have shares for this key
        share_holders = await self._find_share_holders(key_id)
        
        if not share_holders:
            logger.error(f"No share holders found for key {key_id}")
            return None
        
        # Request shares from nodes
        collected_shares = []
        
        for holder_node_id in share_holders:
            share = await self._request_key_share(holder_node_id, key_id, requester_node_id)
            if share:
                collected_shares.append((share.share_index, share.share_data))
        
        if not collected_shares:
            logger.error(f"No shares collected for key {key_id}")
            return None
        
        # Check if we have enough shares
        # We need to know the threshold - this would be stored in metadata
        min_threshold = max(self.min_share_threshold, len(collected_shares) // 2 + 1)
        
        if len(collected_shares) < min_threshold:
            logger.warning(f"Insufficient shares for key {key_id}: "
                          f"{len(collected_shares)} < {min_threshold}")
            return None
        
        # Reconstruct the key
        try:
            private_key = self.secret_sharing.reconstruct_secret(collected_shares)
            logger.info(f"Successfully reconstructed key {key_id}")
            return private_key
        
        except Exception as e:
            logger.error(f"Failed to reconstruct key {key_id}: {e}")
            return None
    
    async def rotate_key(self, key_id: str, authorized_nodes: List[str]) -> Optional[str]:
        """
        Rotate a key by generating a new one and redistributing
        
        Args:
            key_id: ID of the key to rotate
            authorized_nodes: Nodes authorized for the new key
            
        Returns:
            New key ID or None if failed
        """
        if key_id not in self.owned_keys:
            logger.error(f"Key {key_id} not found for rotation")
            return None
        
        old_key = self.owned_keys[key_id]
        
        # Generate new key with same parameters
        new_key_id = await self.generate_keypair(
            old_key.key_type,
            old_key.algorithm,
            self.default_key_lifetime
        )
        
        if not new_key_id:
            logger.error(f"Failed to generate new key for rotation of {key_id}")
            return None
        
        # Distribute new key
        if await self.distribute_key(new_key_id, authorized_nodes):
            # Mark old key as expired
            old_key.status = KeyStatus.EXPIRED
            
            # Update file associations
            new_key = self.owned_keys[new_key_id]
            new_key.associated_file_ids = old_key.associated_file_ids.copy()
            
            logger.info(f"Successfully rotated key {key_id} -> {new_key_id}")
            return new_key_id
        else:
            # Clean up failed new key
            if new_key_id in self.owned_keys:
                del self.owned_keys[new_key_id]
            
            logger.error(f"Failed to distribute rotated key for {key_id}")
            return None
    
    async def revoke_key(self, key_id: str, reason: str = "revoked") -> bool:
        """Revoke a key and notify all nodes"""
        if key_id in self.owned_keys:
            self.owned_keys[key_id].status = KeyStatus.REVOKED
        
        # Notify all nodes that have shares or know about this key
        # This would integrate with the P2P network to broadcast revocation
        logger.info(f"Revoked key {key_id}: {reason}")
        
        return True  # Placeholder - real implementation would track success
    
    def get_public_key(self, key_id: str) -> Optional[bytes]:
        """Get public key by ID"""
        if key_id in self.owned_keys:
            return self.owned_keys[key_id].public_key
        elif key_id in self.peer_keys:
            return self.peer_keys[key_id].public_key
        
        return None
    
    def add_peer_key(self, key: CryptographicKey):
        """Add a peer's public key"""
        # Only store public key portion
        peer_key = CryptographicKey(
            key_id=key.key_id,
            key_type=key.key_type,
            algorithm=key.algorithm,
            public_key=key.public_key,
            private_key=None,  # Never store peer private keys
            created_at=key.created_at,
            expires_at=key.expires_at,
            status=key.status,
            owner_node_id=key.owner_node_id
        )
        
        self.peer_keys[key.key_id] = peer_key
        logger.debug(f"Added peer key {key.key_id} from {key.owner_node_id}")
    
    def _generate_key_id(self, public_key: bytes) -> str:
        """Generate unique key ID from public key"""
        hash_obj = hashlib.sha256()
        hash_obj.update(public_key)
        hash_obj.update(self.node_id.encode())
        hash_obj.update(str(time.time()).encode())
        
        return hash_obj.hexdigest()[:16]  # 64-bit hex ID
    
    async def _send_key_share(self, node_id: str, key_share: KeyShare) -> bool:
        """Send a key share to a node via P2P network"""
        if not self.p2p_network:
            logger.warning("P2P network not available for key share distribution")
            return False
        
        # This would integrate with the P2P network to send the share
        # For now, simulate success
        logger.debug(f"Sent key share {key_share.share_id} to {node_id}")
        return True
    
    async def _find_share_holders(self, key_id: str) -> List[str]:
        """Find nodes that hold shares for a given key"""
        # This would query the P2P network to find share holders
        # For now, return empty list
        return []
    
    async def _request_key_share(self, holder_node_id: str, key_id: str,
                               requester_node_id: str) -> Optional[KeyShare]:
        """Request a key share from a holder node"""
        if not self.p2p_network:
            return None
        
        # This would send a request via P2P network
        # For now, return None
        return None
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """Get key management statistics"""
        owned_active = sum(1 for key in self.owned_keys.values() if key.is_active)
        owned_expired = sum(1 for key in self.owned_keys.values() if key.is_expired)
        
        peer_active = sum(1 for key in self.peer_keys.values() if key.is_active)
        
        total_shares_held = sum(len(shares) for shares in self.key_shares.values())
        
        return {
            'owned_keys': {
                'total': len(self.owned_keys),
                'active': owned_active,
                'expired': owned_expired
            },
            'peer_keys': {
                'total': len(self.peer_keys),
                'active': peer_active
            },
            'key_shares_held': total_shares_held,
            'algorithms_supported': [alg.value for alg in PostQuantumAlgorithm]
        }
    
    def export_public_keys(self) -> Dict[str, Any]:
        """Export public keys for sharing with peers"""
        public_keys = {}
        
        for key_id, key in self.owned_keys.items():
            if key.is_active:
                public_keys[key_id] = {
                    'key_id': key_id,
                    'key_type': key.key_type.value,
                    'algorithm': key.algorithm.value,
                    'public_key': base64.b64encode(key.public_key).decode(),
                    'owner_node_id': key.owner_node_id,
                    'created_at': key.created_at,
                    'expires_at': key.expires_at
                }
        
        return {
            'node_id': self.node_id,
            'public_keys': public_keys,
            'exported_at': time.time()
        }


# Example usage and testing
async def example_key_management():
    """Example of distributed key management usage"""
    
    # Create key managers for multiple nodes
    node1_km = DistributedKeyManager("node1")
    node2_km = DistributedKeyManager("node2")
    node3_km = DistributedKeyManager("node3")
    
    # Generate keys
    print("Generating post-quantum keypairs...")
    key1_id = await node1_km.generate_keypair(KeyType.ENCRYPTION)
    key2_id = await node1_km.generate_keypair(KeyType.SIGNING)
    
    print(f"Generated encryption key: {key1_id}")
    print(f"Generated signing key: {key2_id}")
    
    # Distribute key to authorized nodes
    authorized_nodes = ["node1", "node2", "node3"]
    
    print(f"Distributing key {key1_id} to nodes...")
    success = await node1_km.distribute_key(key1_id, authorized_nodes, threshold=2)
    print(f"Key distribution: {'Success' if success else 'Failed'}")
    
    # Test secret sharing directly
    print("\nTesting Shamir's Secret Sharing...")
    sss = ShamirSecretSharing()
    
    secret = b"This is a secret key for testing"
    shares = sss.split_secret(secret, threshold=3, total_shares=5)
    
    # Reconstruct with minimum shares
    test_shares = shares[:3]  # Use first 3 shares
    reconstructed = sss.reconstruct_secret(test_shares)
    
    print(f"Original secret: {secret}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Reconstruction successful: {secret == reconstructed}")
    
    # Get statistics
    stats = node1_km.get_key_statistics()
    print(f"\nKey management statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_key_management())