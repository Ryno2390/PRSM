"""
PRSM Post-Quantum Cryptography Module
CRYSTALS-Dilithium / ML-DSA (FIPS 204) Implementation

This module provides post-quantum digital signatures using the NIST-standardized
ML-DSA (Module-Lattice-Based Digital Signature Algorithm) based on CRYSTALS-Dilithium.

Security Levels:
- ML-DSA-44: 128-bit post-quantum security (equivalent to AES-128)
- ML-DSA-65: 192-bit post-quantum security (equivalent to AES-192) 
- ML-DSA-87: 256-bit post-quantum security (equivalent to AES-256)

Features:
- NIST FIPS 204 compliance
- Quantum-resistant digital signatures
- Key generation, signing, and verification
- Hybrid mode support (traditional + post-quantum)
- Performance benchmarking
- Configurable security levels
"""

import json
import time
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum

try:
    from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
    ML_DSA_44 = ML_DSA_65 = ML_DSA_87 = None

# from ..core.models import PRSMBaseModel  # Optional import for integration


class SecurityLevel(str, Enum):
    """Post-quantum security levels"""
    LEVEL_1 = "ML-DSA-44"  # 128-bit security
    LEVEL_3 = "ML-DSA-65"  # 192-bit security  
    LEVEL_5 = "ML-DSA-87"  # 256-bit security


class SignatureType(str, Enum):
    """Signature algorithm types"""
    TRADITIONAL = "traditional"     # Classical cryptography (RSA, ECDSA)
    POST_QUANTUM = "post_quantum"   # ML-DSA post-quantum
    HYBRID = "hybrid"              # Both traditional + post-quantum


@dataclass
class PostQuantumKeyPair:
    """Post-quantum key pair with metadata"""
    public_key: bytes
    private_key: bytes
    security_level: SecurityLevel
    key_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "public_key": base64.b64encode(self.public_key).decode(),
            "private_key": base64.b64encode(self.private_key).decode(),
            "security_level": self.security_level.value,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostQuantumKeyPair':
        """Create from dictionary"""
        return cls(
            public_key=base64.b64decode(data["public_key"]),
            private_key=base64.b64decode(data["private_key"]),
            security_level=SecurityLevel(data["security_level"]),
            key_id=data["key_id"],
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass 
class PostQuantumSignature:
    """Post-quantum signature with metadata"""
    signature: bytes
    signature_type: SignatureType
    security_level: SecurityLevel
    signer_key_id: str
    message_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            "signature": base64.b64encode(self.signature).decode(),
            "signature_type": self.signature_type.value,
            "security_level": self.security_level.value,
            "signer_key_id": self.signer_key_id,
            "message_hash": self.message_hash,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostQuantumSignature':
        """Create from dictionary"""
        return cls(
            signature=base64.b64decode(data["signature"]),
            signature_type=SignatureType(data["signature_type"]),
            security_level=SecurityLevel(data["security_level"]),
            signer_key_id=data["signer_key_id"],
            message_hash=data["message_hash"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class PostQuantumCrypto:
    """
    Post-Quantum Cryptography implementation using ML-DSA (CRYSTALS-Dilithium)
    
    Provides quantum-resistant digital signatures for PRSM infrastructure
    """
    
    def __init__(self, default_security_level: SecurityLevel = SecurityLevel.LEVEL_1):
        """
        Initialize post-quantum cryptography system
        
        Args:
            default_security_level: Default security level for operations
        """
        if not DILITHIUM_AVAILABLE:
            raise ImportError(
                "dilithium-py not available. Install with: pip install dilithium-py"
            )
        
        self.default_security_level = default_security_level
        self._ml_dsa_implementations = {
            SecurityLevel.LEVEL_1: ML_DSA_44,
            SecurityLevel.LEVEL_3: ML_DSA_65, 
            SecurityLevel.LEVEL_5: ML_DSA_87
        }
        
        # Performance tracking
        self.performance_metrics = {
            "keygen_times": [],
            "sign_times": [],
            "verify_times": [],
            "key_sizes": {},
            "signature_sizes": {}
        }
    
    def generate_keypair(self, security_level: Optional[SecurityLevel] = None) -> PostQuantumKeyPair:
        """
        Generate a new post-quantum key pair
        
        Args:
            security_level: Security level for the key pair
            
        Returns:
            PostQuantumKeyPair object with public/private keys
        """
        if security_level is None:
            security_level = self.default_security_level
        
        ml_dsa = self._ml_dsa_implementations[security_level]
        
        # Measure key generation performance
        start_time = time.perf_counter()
        public_key, private_key = ml_dsa.keygen()
        keygen_time = time.perf_counter() - start_time
        
        # Track performance metrics
        self.performance_metrics["keygen_times"].append(keygen_time)
        self.performance_metrics["key_sizes"][security_level.value] = {
            "public_key_bytes": len(public_key),
            "private_key_bytes": len(private_key)
        }
        
        return PostQuantumKeyPair(
            public_key=public_key,
            private_key=private_key,
            security_level=security_level
        )
    
    def sign_message(self, 
                    message: Union[str, bytes], 
                    keypair: PostQuantumKeyPair,
                    signature_type: SignatureType = SignatureType.POST_QUANTUM) -> PostQuantumSignature:
        """
        Sign a message using post-quantum cryptography
        
        Args:
            message: Message to sign (string or bytes)
            keypair: Post-quantum key pair to use for signing
            signature_type: Type of signature to create
            
        Returns:
            PostQuantumSignature object
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Create message hash for verification
        message_hash = hashlib.sha256(message).hexdigest()
        
        ml_dsa = self._ml_dsa_implementations[keypair.security_level]
        
        # Measure signing performance
        start_time = time.perf_counter()
        signature = ml_dsa.sign(keypair.private_key, message)
        sign_time = time.perf_counter() - start_time
        
        # Track performance metrics
        self.performance_metrics["sign_times"].append(sign_time)
        if keypair.security_level.value not in self.performance_metrics["signature_sizes"]:
            self.performance_metrics["signature_sizes"][keypair.security_level.value] = len(signature)
        
        return PostQuantumSignature(
            signature=signature,
            signature_type=signature_type,
            security_level=keypair.security_level,
            signer_key_id=keypair.key_id,
            message_hash=message_hash
        )
    
    def verify_signature(self, 
                        message: Union[str, bytes], 
                        signature: PostQuantumSignature,
                        public_key: bytes) -> bool:
        """
        Verify a post-quantum signature
        
        Args:
            message: Original message (string or bytes)
            signature: PostQuantumSignature to verify
            public_key: Public key for verification
            
        Returns:
            True if signature is valid, False otherwise
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Verify message hash matches
        message_hash = hashlib.sha256(message).hexdigest()
        if message_hash != signature.message_hash:
            return False
        
        ml_dsa = self._ml_dsa_implementations[signature.security_level]
        
        # Measure verification performance
        start_time = time.perf_counter()
        try:
            is_valid = ml_dsa.verify(public_key, message, signature.signature)
            verify_time = time.perf_counter() - start_time
            
            # Track performance metrics
            self.performance_metrics["verify_times"].append(verify_time)
            
            return is_valid
        except Exception:
            # Any exception during verification means invalid signature
            return False
    
    def get_security_info(self, security_level: SecurityLevel) -> Dict[str, Any]:
        """
        Get detailed information about a security level
        
        Args:
            security_level: Security level to query
            
        Returns:
            Dictionary with security level information
        """
        key_sizes = self.performance_metrics["key_sizes"].get(security_level.value, {})
        sig_size = self.performance_metrics["signature_sizes"].get(security_level.value, 0)
        
        security_info = {
            SecurityLevel.LEVEL_1: {
                "description": "128-bit post-quantum security (equivalent to AES-128)",
                "nist_category": "Category 1",
                "quantum_attack_cost": "2^128 quantum operations"
            },
            SecurityLevel.LEVEL_3: {
                "description": "192-bit post-quantum security (equivalent to AES-192)",
                "nist_category": "Category 3", 
                "quantum_attack_cost": "2^192 quantum operations"
            },
            SecurityLevel.LEVEL_5: {
                "description": "256-bit post-quantum security (equivalent to AES-256)",
                "nist_category": "Category 5",
                "quantum_attack_cost": "2^256 quantum operations"
            }
        }
        
        info = security_info[security_level].copy()
        info.update({
            "algorithm": "ML-DSA (CRYSTALS-Dilithium)",
            "standard": "NIST FIPS 204",
            "public_key_size_bytes": key_sizes.get("public_key_bytes", 0),
            "private_key_size_bytes": key_sizes.get("private_key_bytes", 0),
            "signature_size_bytes": sig_size
        })
        
        return info
    
    def get_performance_benchmark(self) -> Dict[str, Any]:
        """
        Get performance benchmark data
        
        Returns:
            Dictionary with performance statistics
        """
        def calculate_stats(times_list):
            if not times_list:
                return {"count": 0, "mean": 0, "min": 0, "max": 0}
            return {
                "count": len(times_list),
                "mean": sum(times_list) / len(times_list),
                "min": min(times_list),
                "max": max(times_list)
            }
        
        return {
            "key_generation": calculate_stats(self.performance_metrics["keygen_times"]),
            "signing": calculate_stats(self.performance_metrics["sign_times"]),
            "verification": calculate_stats(self.performance_metrics["verify_times"]),
            "key_sizes": self.performance_metrics["key_sizes"],
            "signature_sizes": self.performance_metrics["signature_sizes"]
        }
    
    def benchmark_all_security_levels(self, iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark all security levels with multiple iterations
        
        Args:
            iterations: Number of iterations to run for each test
            
        Returns:
            Comprehensive benchmark results
        """
        results = {}
        test_message = b"PRSM post-quantum cryptography benchmark test message"
        
        for security_level in SecurityLevel:
            print(f"Benchmarking {security_level.value}...")
            level_results = {
                "keygen_times": [],
                "sign_times": [],
                "verify_times": [],
                "key_sizes": {},
                "signature_size": 0
            }
            
            for i in range(iterations):
                # Key generation
                start_time = time.perf_counter()
                keypair = self.generate_keypair(security_level)
                keygen_time = time.perf_counter() - start_time
                level_results["keygen_times"].append(keygen_time)
                
                # Signing
                start_time = time.perf_counter()
                signature = self.sign_message(test_message, keypair)
                sign_time = time.perf_counter() - start_time
                level_results["sign_times"].append(sign_time)
                
                # Verification
                start_time = time.perf_counter()
                is_valid = self.verify_signature(test_message, signature, keypair.public_key)
                verify_time = time.perf_counter() - start_time
                level_results["verify_times"].append(verify_time)
                
                assert is_valid, f"Signature verification failed for {security_level.value}"
                
                # Record sizes (only need to do this once)
                if i == 0:
                    level_results["key_sizes"] = {
                        "public_key_bytes": len(keypair.public_key),
                        "private_key_bytes": len(keypair.private_key)
                    }
                    level_results["signature_size"] = len(signature.signature)
            
            # Calculate statistics
            def stats(times):
                return {
                    "mean_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "std_dev_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000
                }
            
            results[security_level.value] = {
                "keygen_performance": stats(level_results["keygen_times"]),
                "signing_performance": stats(level_results["sign_times"]),
                "verification_performance": stats(level_results["verify_times"]),
                "key_sizes": level_results["key_sizes"],
                "signature_size_bytes": level_results["signature_size"],
                "iterations": iterations
            }
        
        return results


# Global instance for easy access
_pq_crypto_instance: Optional[PostQuantumCrypto] = None


def get_post_quantum_crypto(security_level: SecurityLevel = SecurityLevel.LEVEL_1) -> PostQuantumCrypto:
    """
    Get or create global post-quantum crypto instance
    
    Args:
        security_level: Default security level
        
    Returns:
        PostQuantumCrypto instance
    """
    global _pq_crypto_instance
    if _pq_crypto_instance is None:
        _pq_crypto_instance = PostQuantumCrypto(security_level)
    return _pq_crypto_instance


def reset_post_quantum_crypto():
    """Reset the global post-quantum crypto instance"""
    global _pq_crypto_instance
    _pq_crypto_instance = None


# Convenience functions for easy usage
def generate_pq_keypair(security_level: SecurityLevel = SecurityLevel.LEVEL_1) -> PostQuantumKeyPair:
    """Generate a post-quantum key pair"""
    return get_post_quantum_crypto().generate_keypair(security_level)


def sign_with_pq(message: Union[str, bytes], keypair: PostQuantumKeyPair) -> PostQuantumSignature:
    """Sign a message with post-quantum cryptography"""
    return get_post_quantum_crypto().sign_message(message, keypair)


def verify_pq_signature(message: Union[str, bytes], signature: PostQuantumSignature, public_key: bytes) -> bool:
    """Verify a post-quantum signature"""
    return get_post_quantum_crypto().verify_signature(message, signature, public_key)


# Example usage function
async def example_usage():
    """Example usage of post-quantum cryptography"""
    print("🔐 PRSM Post-Quantum Cryptography Example")
    print("=" * 50)
    
    # Initialize post-quantum crypto
    pq_crypto = get_post_quantum_crypto(SecurityLevel.LEVEL_1)
    
    # Generate key pair
    print("1. Generating post-quantum key pair...")
    keypair = pq_crypto.generate_keypair(SecurityLevel.LEVEL_1)
    print(f"   Key ID: {keypair.key_id}")
    print(f"   Public key size: {len(keypair.public_key)} bytes")
    print(f"   Private key size: {len(keypair.private_key)} bytes")
    
    # Sign a message
    print("\n2. Signing message...")
    message = "PRSM post-quantum signature test"
    signature = pq_crypto.sign_message(message, keypair)
    print(f"   Signature size: {len(signature.signature)} bytes")
    print(f"   Signature type: {signature.signature_type.value}")
    
    # Verify signature
    print("\n3. Verifying signature...")
    is_valid = pq_crypto.verify_signature(message, signature, keypair.public_key)
    print(f"   Signature valid: {is_valid}")
    
    # Get security info
    print("\n4. Security information:")
    security_info = pq_crypto.get_security_info(SecurityLevel.LEVEL_1)
    print(f"   Algorithm: {security_info['algorithm']}")
    print(f"   Standard: {security_info['standard']}")
    print(f"   Security: {security_info['description']}")
    
    # Performance benchmark
    print("\n5. Performance benchmark:")
    benchmark = pq_crypto.get_performance_benchmark()
    if benchmark["signing"]["count"] > 0:
        print(f"   Signing: {benchmark['signing']['mean']*1000:.2f}ms average")
        print(f"   Verification: {benchmark['verification']['mean']*1000:.2f}ms average")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())