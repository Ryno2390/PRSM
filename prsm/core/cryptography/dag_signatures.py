"""
DAG Transaction Signature Utilities
====================================

Ed25519-based cryptographic signature system for DAG ledger transactions.
Provides secure transaction signing and verification to prevent impersonation
and fraud in the PRSM network.

Signature Scheme:
- Algorithm: Ed25519 (Edwards-curve Digital Signature Algorithm)
- Key Size: 256-bit private key, 256-bit public key
- Signature Size: 64 bytes
- Hash: Transaction data is hashed with SHA-256 before signing

Security Properties:
- Non-repudiation: Signatures prove transaction origin
- Integrity: Any modification invalidates the signature
- Authentication: Only the private key holder can sign
- Efficiency: Ed25519 provides fast signing and verification
"""

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


class SignatureError(Exception):
    """Base exception for signature-related errors."""
    pass


class InvalidSignatureError(SignatureError):
    """Raised when signature verification fails."""
    pass


class MissingSignatureError(SignatureError):
    """Raised when a required signature is missing."""
    pass


class MissingPublicKeyError(SignatureError):
    """Raised when public key is not available for verification."""
    pass


@dataclass
class KeyPair:
    """Ed25519 key pair for transaction signing."""
    private_key: ed25519.Ed25519PrivateKey
    public_key: ed25519.Ed25519PublicKey
    
    @classmethod
    def generate(cls) -> "KeyPair":
        """Generate a new Ed25519 key pair."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(private_key=private_key, public_key=public_key)
    
    def get_private_key_bytes(self) -> bytes:
        """Get private key as raw bytes (32 bytes)."""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key as raw bytes (32 bytes)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    
    def get_private_key_hex(self) -> str:
        """Get private key as hex string."""
        return self.get_private_key_bytes().hex()
    
    def get_public_key_hex(self) -> str:
        """Get public key as hex string."""
        return self.get_public_key_bytes().hex()
    
    def get_private_key_base64(self) -> str:
        """Get private key as base64 string."""
        return base64.b64encode(self.get_private_key_bytes()).decode('utf-8')
    
    def get_public_key_base64(self) -> str:
        """Get public key as base64 string."""
        return base64.b64encode(self.get_public_key_bytes()).decode('utf-8')


class DAGSignatureManager:
    """
    Manages Ed25519 signatures for DAG transactions.
    
    This class provides methods for:
    - Generating Ed25519 key pairs
    - Signing transaction hashes
    - Verifying transaction signatures
    - Serializing/deserializing keys and signatures
    
    Usage:
        # Generate a new key pair
        key_pair = DAGSignatureManager.generate_key_pair()
        
        # Sign a transaction
        signature = DAGSignatureManager.sign_transaction(tx_hash, key_pair.private_key)
        
        # Verify a signature
        is_valid = DAGSignatureManager.verify_signature(tx_hash, signature, public_key)
    """
    
    # Signature size in bytes (Ed25519 produces 64-byte signatures)
    SIGNATURE_SIZE = 64
    
    # Public key size in bytes
    PUBLIC_KEY_SIZE = 32
    
    # Private key size in bytes
    PRIVATE_KEY_SIZE = 32
    
    @staticmethod
    def generate_key_pair() -> KeyPair:
        """
        Generate a new Ed25519 key pair for transaction signing.
        
        Returns:
            KeyPair: A new Ed25519 key pair
        """
        return KeyPair.generate()
    
    @staticmethod
    def load_private_key(key_bytes: bytes) -> ed25519.Ed25519PrivateKey:
        """
        Load an Ed25519 private key from raw bytes.
        
        Args:
            key_bytes: Raw private key bytes (32 bytes)
            
        Returns:
            Ed25519PrivateKey: The loaded private key
            
        Raises:
            ValueError: If key bytes are invalid
        """
        if len(key_bytes) != DAGSignatureManager.PRIVATE_KEY_SIZE:
            raise ValueError(
                f"Invalid private key size: expected {DAGSignatureManager.PRIVATE_KEY_SIZE} bytes, "
                f"got {len(key_bytes)} bytes"
            )
        return ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
    
    @staticmethod
    def load_public_key(key_bytes: bytes) -> ed25519.Ed25519PublicKey:
        """
        Load an Ed25519 public key from raw bytes.
        
        Args:
            key_bytes: Raw public key bytes (32 bytes)
            
        Returns:
            Ed25519PublicKey: The loaded public key
            
        Raises:
            ValueError: If key bytes are invalid
        """
        if len(key_bytes) != DAGSignatureManager.PUBLIC_KEY_SIZE:
            raise ValueError(
                f"Invalid public key size: expected {DAGSignatureManager.PUBLIC_KEY_SIZE} bytes, "
                f"got {len(key_bytes)} bytes"
            )
        return ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)
    
    @staticmethod
    def load_private_key_from_hex(hex_string: str) -> ed25519.Ed25519PrivateKey:
        """
        Load an Ed25519 private key from hex string.
        
        Args:
            hex_string: Hex-encoded private key string
            
        Returns:
            Ed25519PrivateKey: The loaded private key
        """
        try:
            key_bytes = bytes.fromhex(hex_string)
            return DAGSignatureManager.load_private_key(key_bytes)
        except ValueError as e:
            raise ValueError(f"Invalid hex private key: {e}")
    
    @staticmethod
    def load_public_key_from_hex(hex_string: str) -> ed25519.Ed25519PublicKey:
        """
        Load an Ed25519 public key from hex string.
        
        Args:
            hex_string: Hex-encoded public key string
            
        Returns:
            Ed25519PublicKey: The loaded public key
        """
        try:
            key_bytes = bytes.fromhex(hex_string)
            return DAGSignatureManager.load_public_key(key_bytes)
        except ValueError as e:
            raise ValueError(f"Invalid hex public key: {e}")
    
    @staticmethod
    def load_private_key_from_base64(b64_string: str) -> ed25519.Ed25519PrivateKey:
        """
        Load an Ed25519 private key from base64 string.
        
        Args:
            b64_string: Base64-encoded private key string
            
        Returns:
            Ed25519PrivateKey: The loaded private key
        """
        try:
            key_bytes = base64.b64decode(b64_string)
            return DAGSignatureManager.load_private_key(key_bytes)
        except Exception as e:
            raise ValueError(f"Invalid base64 private key: {e}")
    
    @staticmethod
    def load_public_key_from_base64(b64_string: str) -> ed25519.Ed25519PublicKey:
        """
        Load an Ed25519 public key from base64 string.
        
        Args:
            b64_string: Base64-encoded public key string
            
        Returns:
            Ed25519PublicKey: The loaded public key
        """
        try:
            key_bytes = base64.b64decode(b64_string)
            return DAGSignatureManager.load_public_key(key_bytes)
        except Exception as e:
            raise ValueError(f"Invalid base64 public key: {e}")
    
    @staticmethod
    def sign_transaction_hash(
        tx_hash: str,
        private_key: ed25519.Ed25519PrivateKey
    ) -> str:
        """
        Sign a transaction hash with an Ed25519 private key.
        
        The signature is produced by signing the raw bytes of the transaction
        hash (interpreted as a hex string).
        
        Args:
            tx_hash: The transaction hash (hex string) to sign
            private_key: The Ed25519 private key to sign with
            
        Returns:
            str: Base64-encoded signature (64 bytes when decoded)
            
        Raises:
            ValueError: If tx_hash is invalid
        """
        if not tx_hash:
            raise ValueError("Transaction hash cannot be empty")
        
        # Convert hex hash to bytes for signing
        try:
            tx_hash_bytes = bytes.fromhex(tx_hash)
        except ValueError:
            # If not valid hex, hash the string directly
            tx_hash_bytes = hashlib.sha256(tx_hash.encode()).digest()
        
        # Sign the hash
        signature = private_key.sign(tx_hash_bytes)
        
        # Return base64-encoded signature
        return base64.b64encode(signature).decode('utf-8')
    
    @staticmethod
    def verify_signature(
        tx_hash: str,
        signature_b64: str,
        public_key: ed25519.Ed25519PublicKey
    ) -> bool:
        """
        Verify a transaction signature.
        
        Args:
            tx_hash: The transaction hash (hex string) that was signed
            signature_b64: Base64-encoded signature
            public_key: The Ed25519 public key to verify with
            
        Returns:
            bool: True if signature is valid, False otherwise
            
        Raises:
            InvalidSignatureError: If signature verification fails
            ValueError: If inputs are invalid
        """
        if not tx_hash:
            raise ValueError("Transaction hash cannot be empty")
        if not signature_b64:
            raise ValueError("Signature cannot be empty")
        
        try:
            # Decode the signature
            signature = base64.b64decode(signature_b64)
            
            # Convert hex hash to bytes
            try:
                tx_hash_bytes = bytes.fromhex(tx_hash)
            except ValueError:
                # If not valid hex, hash the string directly
                tx_hash_bytes = hashlib.sha256(tx_hash.encode()).digest()
            
            # Verify the signature
            public_key.verify(signature, tx_hash_bytes)
            return True
            
        except InvalidSignature:
            raise InvalidSignatureError("Signature verification failed - signature is invalid")
        except Exception as e:
            raise InvalidSignatureError(f"Signature verification failed: {e}")
    
    @staticmethod
    def sign_transaction_data(
        tx_data: dict,
        private_key: ed25519.Ed25519PrivateKey
    ) -> Tuple[str, str]:
        """
        Sign transaction data and return both hash and signature.
        
        This method creates a canonical hash of the transaction data,
        signs it, and returns both the hash and signature.
        
        Args:
            tx_data: Dictionary of transaction data to sign
            private_key: The Ed25519 private key to sign with
            
        Returns:
            Tuple[str, str]: (transaction_hash, signature) both as strings
        """
        # Create canonical JSON representation
        canonical_json = json.dumps(tx_data, sort_keys=True, separators=(',', ':'))
        
        # Hash the transaction data
        tx_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        
        # Sign the hash
        signature = DAGSignatureManager.sign_transaction_hash(tx_hash, private_key)
        
        return tx_hash, signature
    
    @staticmethod
    def verify_transaction_data(
        tx_data: dict,
        signature_b64: str,
        public_key: ed25519.Ed25519PublicKey
    ) -> Tuple[bool, str]:
        """
        Verify transaction data against a signature.
        
        Args:
            tx_data: Dictionary of transaction data to verify
            signature_b64: Base64-encoded signature
            public_key: The Ed25519 public key to verify with
            
        Returns:
            Tuple[bool, str]: (is_valid, transaction_hash)
        """
        # Create canonical JSON representation (must match signing)
        canonical_json = json.dumps(tx_data, sort_keys=True, separators=(',', ':'))
        
        # Hash the transaction data
        tx_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        
        try:
            is_valid = DAGSignatureManager.verify_signature(tx_hash, signature_b64, public_key)
            return is_valid, tx_hash
        except InvalidSignatureError:
            return False, tx_hash


def create_signing_key_pair() -> KeyPair:
    """
    Convenience function to create a new Ed25519 key pair.
    
    Returns:
        KeyPair: A new Ed25519 key pair for signing transactions
    """
    return DAGSignatureManager.generate_key_pair()


def sign_hash(tx_hash: str, private_key: ed25519.Ed25519PrivateKey) -> str:
    """
    Convenience function to sign a transaction hash.
    
    Args:
        tx_hash: Transaction hash to sign
        private_key: Private key to sign with
        
    Returns:
        str: Base64-encoded signature
    """
    return DAGSignatureManager.sign_transaction_hash(tx_hash, private_key)


def verify_hash_signature(
    tx_hash: str,
    signature: str,
    public_key: ed25519.Ed25519PublicKey
) -> bool:
    """
    Convenience function to verify a transaction signature.
    
    Args:
        tx_hash: Transaction hash that was signed
        signature: Base64-encoded signature
        public_key: Public key to verify with
        
    Returns:
        bool: True if signature is valid
    """
    try:
        return DAGSignatureManager.verify_signature(tx_hash, signature, public_key)
    except InvalidSignatureError:
        return False
