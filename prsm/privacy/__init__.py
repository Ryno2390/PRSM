"""
PRSM Privacy Infrastructure Layer
================================

The 12th subsystem providing comprehensive privacy, anonymity, and censorship 
resistance for PRSM participants. Designed to protect researchers, especially 
those in authoritarian regimes or working on sensitive topics.

Core Components:
- Anonymous networking (Tor/I2P integration)
- Zero-knowledge model contributions
- Encrypted communication protocols
- Anonymous identity management
- Private FTNS transactions with mixing
- Censorship-resistant governance
- Private research queries

Key Features:
- Default anonymity with opt-in transparency
- Cryptographic guarantees (no trusted third parties)
- Jurisdiction diversity for legal protection
- Performance-optimized privacy protocols
- Tiered privacy levels (basic → enhanced → maximum)
- Anonymous FTNS transactions with ring signatures
- Stealth addresses and mixing protocols
- Sybil-resistant anonymous identities
"""

from .anonymous_networking import anonymous_network_manager
from .zk_proofs import zk_proof_system
from .encrypted_comms import encrypted_communication_layer
from .anonymous_identity import anonymous_identity_manager
from .private_ftns import private_ftns_system

__version__ = "1.0.0-beta"

__all__ = [
    "anonymous_network_manager",
    "zk_proof_system", 
    "encrypted_communication_layer",
    "anonymous_identity_manager",
    "private_ftns_system"
]