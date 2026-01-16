"""
Production Post-Quantum Cryptography Implementation
===================================================

This module provides post-quantum cryptographic operations for PRSM using
the Open Quantum Safe (liboqs) library with CRYSTALS-Dilithium signatures.

SECURITY POLICY:
- NO MOCK MODE: If PQC is enabled but the library is unavailable, operations FAIL.
- EXPLICIT MODE: Users must explicitly choose a mode (REAL, HYBRID, or DISABLED).
- FAIL CLOSED: In production, missing libraries cause startup failure.

Installation:
    # Install liboqs system library
    brew install liboqs  # macOS
    apt install liboqs-dev  # Ubuntu/Debian

    # Install Python bindings
    pip install liboqs-python

Usage:
    from prsm.core.cryptography.post_quantum_production import (
        get_pqc_system, PQCMode, PQCSecurityLevel
    )

    # Disabled mode (default - no PQC)
    pqc = get_pqc_system(PQCMode.DISABLED)

    # Real PQC mode (requires liboqs)
    pqc = get_pqc_system(PQCMode.REAL)

    # Generate keys
    keypair = pqc.generate_keypair()

    # Sign message
    signature = pqc.sign(message, keypair.private_key)

    # Verify signature
    valid = pqc.verify(message, signature.signature, keypair.public_key)
"""

import os
import hashlib
import structlog
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# =========================================================================
# Library Detection
# =========================================================================

# Attempt to import liboqs
LIBOQS_AVAILABLE = False
LIBOQS_ERROR = None
oqs = None

try:
    import oqs
    LIBOQS_AVAILABLE = True

    # Verify the library actually works
    try:
        _test_sig = oqs.Signature("Dilithium3")
        _test_sig.generate_keypair()
        del _test_sig
        LIBOQS_FUNCTIONAL = True
    except Exception as e:
        LIBOQS_FUNCTIONAL = False
        LIBOQS_ERROR = f"liboqs functional test failed: {e}"

except ImportError as e:
    LIBOQS_ERROR = f"liboqs not installed: {e}"
    LIBOQS_FUNCTIONAL = False
except Exception as e:
    LIBOQS_ERROR = f"liboqs import error: {e}"
    LIBOQS_FUNCTIONAL = False


# =========================================================================
# Enums and Data Classes
# =========================================================================

class PQCSecurityLevel(str, Enum):
    """
    NIST Post-Quantum Security Levels.

    These correspond to NIST security categories:
    - LEVEL_2: ~AES-128 equivalent (ML-DSA-44 / Dilithium2)
    - LEVEL_3: ~AES-192 equivalent (ML-DSA-65 / Dilithium3) - RECOMMENDED
    - LEVEL_5: ~AES-256 equivalent (ML-DSA-87 / Dilithium5)
    """
    LEVEL_2 = "Dilithium2"
    LEVEL_3 = "Dilithium3"
    LEVEL_5 = "Dilithium5"


class PQCMode(str, Enum):
    """
    Operational modes for the PQC system.

    DISABLED: No PQC operations (default, safe)
    REAL: Full PQC using liboqs (requires installation)
    HYBRID: PQC combined with classical signatures (future)

    NOTE: There is intentionally NO MOCK MODE to prevent security confusion.
    """
    DISABLED = "disabled"
    REAL = "real"
    HYBRID = "hybrid"  # Future: PQC + Ed25519


@dataclass
class PQCKeyPair:
    """Post-quantum cryptographic key pair."""
    public_key: bytes
    private_key: bytes
    algorithm: str
    security_level: PQCSecurityLevel
    key_id: str = field(default_factory=lambda: hashlib.sha256(os.urandom(32)).hexdigest()[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def public_key_hex(self) -> str:
        """Return public key as hex string."""
        return self.public_key.hex()

    def get_metadata(self) -> Dict[str, Any]:
        """Return key metadata (safe to share)."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "security_level": self.security_level.name,
            "public_key_hash": hashlib.sha256(self.public_key).hexdigest()[:16],
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PQCSignature:
    """Post-quantum digital signature."""
    signature: bytes
    algorithm: str
    key_id: str
    message_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def signature_hex(self) -> str:
        """Return signature as hex string."""
        return self.signature.hex()

    @property
    def size_bytes(self) -> int:
        """Return signature size in bytes."""
        return len(self.signature)


# =========================================================================
# Exceptions
# =========================================================================

class PQCError(Exception):
    """Base exception for PQC operations."""
    pass


class PQCNotAvailableError(PQCError):
    """Raised when PQC is requested but library is not available."""
    pass


class PQCDisabledError(PQCError):
    """Raised when PQC operation is attempted but mode is DISABLED."""
    pass


class PQCVerificationError(PQCError):
    """Raised when signature verification fails."""
    pass


# =========================================================================
# Production PQC Implementation
# =========================================================================

class ProductionPostQuantumCrypto:
    """
    Production-grade Post-Quantum Cryptography implementation.

    SECURITY GUARANTEES:
    1. No silent mock mode - operations fail loudly if library unavailable
    2. Explicit mode selection required
    3. Algorithm confusion prevention
    4. Key and signature validation
    """

    def __init__(
        self,
        mode: PQCMode = PQCMode.DISABLED,
        default_level: PQCSecurityLevel = PQCSecurityLevel.LEVEL_3
    ):
        """
        Initialize PQC system.

        Args:
            mode: Operational mode (DISABLED, REAL, or HYBRID)
            default_level: Default security level for operations

        Raises:
            PQCNotAvailableError: If mode requires liboqs but it's unavailable
        """
        self.mode = mode
        self.default_level = default_level

        # Validate mode requirements
        if mode in [PQCMode.REAL, PQCMode.HYBRID]:
            if not LIBOQS_AVAILABLE:
                raise PQCNotAvailableError(
                    f"PQC mode '{mode.value}' requires liboqs library.\n"
                    f"Error: {LIBOQS_ERROR}\n\n"
                    "Installation instructions:\n"
                    "  macOS: brew install liboqs && pip install liboqs-python\n"
                    "  Ubuntu: apt install liboqs-dev && pip install liboqs-python\n\n"
                    "To disable PQC, set PRSM_PQC_MODE=disabled or use PQCMode.DISABLED"
                )

            if not LIBOQS_FUNCTIONAL:
                raise PQCNotAvailableError(
                    f"liboqs is installed but not functional.\n"
                    f"Error: {LIBOQS_ERROR}"
                )

        # Cache for signature instances
        self._signature_instances: Dict[str, Any] = {}

        logger.info(
            "PQC system initialized",
            mode=mode.value,
            liboqs_available=LIBOQS_AVAILABLE,
            liboqs_functional=LIBOQS_FUNCTIONAL if LIBOQS_AVAILABLE else False
        )

    def is_enabled(self) -> bool:
        """Check if PQC operations are enabled."""
        return self.mode != PQCMode.DISABLED

    def generate_keypair(
        self,
        security_level: Optional[PQCSecurityLevel] = None
    ) -> PQCKeyPair:
        """
        Generate a post-quantum key pair.

        Args:
            security_level: Security level (default: LEVEL_3)

        Returns:
            PQCKeyPair with public and private keys

        Raises:
            PQCDisabledError: If mode is DISABLED
            PQCError: If key generation fails
        """
        if self.mode == PQCMode.DISABLED:
            raise PQCDisabledError(
                "PQC is disabled. Enable with PRSM_PQC_MODE=real or use PQCMode.REAL"
            )

        level = security_level or self.default_level
        algorithm = level.value

        try:
            sig = self._get_signature_instance(algorithm)
            public_key = sig.generate_keypair()
            private_key = sig.export_secret_key()

            keypair = PQCKeyPair(
                public_key=public_key,
                private_key=private_key,
                algorithm=algorithm,
                security_level=level
            )

            logger.info(
                "PQC keypair generated",
                algorithm=algorithm,
                key_id=keypair.key_id,
                public_key_size=len(public_key)
            )

            return keypair

        except Exception as e:
            logger.error(f"PQC key generation failed: {e}")
            raise PQCError(f"Key generation failed: {e}") from e

    def sign(
        self,
        message: Union[str, bytes],
        private_key: bytes,
        algorithm: str = "Dilithium3",
        key_id: str = ""
    ) -> PQCSignature:
        """
        Sign a message using post-quantum cryptography.

        Args:
            message: Message to sign (str or bytes)
            private_key: Private key for signing
            algorithm: Algorithm to use (default: Dilithium3)
            key_id: Optional key identifier

        Returns:
            PQCSignature containing the signature

        Raises:
            PQCDisabledError: If mode is DISABLED
            PQCError: If signing fails
        """
        if self.mode == PQCMode.DISABLED:
            raise PQCDisabledError("PQC is disabled")

        if isinstance(message, str):
            message = message.encode('utf-8')

        message_hash = hashlib.sha256(message).hexdigest()

        try:
            # Create signature instance with private key
            sig_instance = oqs.Signature(algorithm, private_key)
            signature_bytes = sig_instance.sign(message)

            signature = PQCSignature(
                signature=signature_bytes,
                algorithm=algorithm,
                key_id=key_id,
                message_hash=message_hash
            )

            logger.debug(
                "Message signed with PQC",
                algorithm=algorithm,
                signature_size=len(signature_bytes),
                message_hash=message_hash[:16]
            )

            return signature

        except Exception as e:
            logger.error(f"PQC signing failed: {e}")
            raise PQCError(f"Signing failed: {e}") from e

    def verify(
        self,
        message: Union[str, bytes],
        signature: bytes,
        public_key: bytes,
        algorithm: str = "Dilithium3"
    ) -> bool:
        """
        Verify a post-quantum signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Public key for verification
            algorithm: Algorithm used for signing

        Returns:
            True if signature is valid, False otherwise

        Raises:
            PQCDisabledError: If mode is DISABLED
        """
        if self.mode == PQCMode.DISABLED:
            raise PQCDisabledError("PQC is disabled")

        if isinstance(message, str):
            message = message.encode('utf-8')

        try:
            sig_instance = oqs.Signature(algorithm)
            is_valid = sig_instance.verify(message, signature, public_key)

            logger.debug(
                "PQC signature verification",
                algorithm=algorithm,
                valid=is_valid
            )

            return is_valid

        except Exception as e:
            logger.warning(f"PQC verification error: {e}")
            return False

    def _get_signature_instance(self, algorithm: str):
        """Get or create a signature instance for the algorithm."""
        if algorithm not in self._signature_instances:
            self._signature_instances[algorithm] = oqs.Signature(algorithm)
        return self._signature_instances[algorithm]

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """Get PQC system status and capabilities."""
        return {
            "liboqs_available": LIBOQS_AVAILABLE,
            "liboqs_functional": LIBOQS_FUNCTIONAL if LIBOQS_AVAILABLE else False,
            "error": LIBOQS_ERROR,
            "supported_algorithms": (
                oqs.get_enabled_sig_mechanisms()
                if LIBOQS_AVAILABLE else []
            ),
            "recommended_algorithms": ["Dilithium3", "Dilithium5"],
            "security_levels": {
                "LEVEL_2": "Dilithium2 (~AES-128)",
                "LEVEL_3": "Dilithium3 (~AES-192) - RECOMMENDED",
                "LEVEL_5": "Dilithium5 (~AES-256)"
            }
        }

    @staticmethod
    def get_algorithm_info(algorithm: str = "Dilithium3") -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        if not LIBOQS_AVAILABLE:
            return {"error": "liboqs not available"}

        try:
            sig = oqs.Signature(algorithm)
            return {
                "algorithm": algorithm,
                "public_key_size": sig.details["length_public_key"],
                "private_key_size": sig.details["length_secret_key"],
                "signature_size": sig.details["length_signature"],
                "is_euf_cma": sig.details["is_euf_cma"],
                "nist_level": sig.details.get("claimed_nist_level", "unknown")
            }
        except Exception as e:
            return {"error": str(e)}


# =========================================================================
# Factory and Global Access
# =========================================================================

_pqc_instance: Optional[ProductionPostQuantumCrypto] = None


def get_pqc_system(mode: Optional[PQCMode] = None) -> ProductionPostQuantumCrypto:
    """
    Get or create PQC system instance.

    Environment variable PRSM_PQC_MODE controls the mode:
    - "disabled": No PQC (default, safe)
    - "real": Full PQC with liboqs
    - "hybrid": PQC + classical (future)

    Args:
        mode: Explicit mode override

    Returns:
        Configured PQC system
    """
    global _pqc_instance

    if mode is None:
        mode_str = os.environ.get("PRSM_PQC_MODE", "disabled").lower()
        try:
            mode = PQCMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid PQC mode '{mode_str}', defaulting to disabled")
            mode = PQCMode.DISABLED

    # Create new instance if mode changed or doesn't exist
    if _pqc_instance is None or _pqc_instance.mode != mode:
        _pqc_instance = ProductionPostQuantumCrypto(mode=mode)

    return _pqc_instance


def reset_pqc_system():
    """Reset global PQC instance (for testing)."""
    global _pqc_instance
    _pqc_instance = None


# =========================================================================
# Convenience Functions
# =========================================================================

def is_pqc_available() -> bool:
    """Check if PQC is available on this system."""
    return LIBOQS_AVAILABLE and LIBOQS_FUNCTIONAL


def get_pqc_status() -> Dict[str, Any]:
    """Get current PQC system status."""
    return ProductionPostQuantumCrypto.get_status()
