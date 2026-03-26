"""
Ledger Signer
=============

Ed25519-based hash-chain signing for the Project Ledger.

Each ledger entry is signed with an Ed25519 keypair and includes the
SHA-256 hash of the previous entry, creating a tamper-evident chain:

    chain_hash_N = SHA-256(content_hash_N + ":" + chain_hash_{N-1})
    signature_N  = Ed25519_sign(private_key, chain_hash_N)

Tamper-evidence properties
--------------------------
- **Content integrity**: modifying entry N changes content_hash_N, which
  changes chain_hash_N, breaking signature_N.
- **Order integrity**: swapping entries changes the chain_hash sequence,
  breaking all subsequent signatures (because previous_hash is signed).
- **Authorship**: the Ed25519 signature proves the private key holder wrote
  the entry; the public key is stored with each entry for verification.

Builds on PRSM's existing Ed25519 infrastructure in
``prsm.core.cryptography.dag_signatures``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64  # Sentinel for the first entry in a chain


# ======================================================================
# Data models
# ======================================================================

@dataclass
class EntrySignature:
    """Cryptographic proof for a single ledger entry."""
    chain_hash: str
    """SHA-256 of (content_hash + ':' + previous_hash). This is what is signed."""

    signature_b64: str
    """Base64-encoded Ed25519 signature over chain_hash."""

    public_key_b64: str
    """Base64-encoded raw public key bytes (32 bytes) for verification."""


@dataclass
class VerificationResult:
    """Result of verifying a chain of signed entries."""
    valid: bool
    entry_count: int
    first_bad_index: Optional[int] = None
    reason: Optional[str] = None

    def __str__(self) -> str:
        if self.valid:
            return f"Chain OK ({self.entry_count} entries verified)"
        return (
            f"Chain INVALID at entry #{self.first_bad_index}: {self.reason}"
        )


# ======================================================================
# LedgerSigner
# ======================================================================

class LedgerSigner:
    """
    Signs and verifies ledger entries using Ed25519.

    A new keypair is generated for each session by default.
    For persistent identity across sessions, provide a *keyfile_path*
    where the private key is serialised (PEM format) between runs.

    Parameters
    ----------
    keyfile_path : Path, optional
        If provided, the private key is loaded from this file (if it exists)
        or written to it after generation.  Allows the same key to sign
        entries across multiple sessions of the same long-running project.
    """

    def __init__(self, keyfile_path: Optional[Path] = None) -> None:
        self._keyfile = keyfile_path
        self._keypair = None
        self._public_key_b64: str = ""

    # ------------------------------------------------------------------
    # Key lifecycle
    # ------------------------------------------------------------------

    def load_or_generate(self) -> None:
        """Load the keypair from disk or generate a fresh one."""
        from prsm.core.cryptography.dag_signatures import KeyPair
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if self._keyfile and self._keyfile.exists():
            try:
                pem = self._keyfile.read_bytes()
                private_key = serialization.load_pem_private_key(pem, password=None)
                public_key = private_key.public_key()
                self._keypair = KeyPair(
                    private_key=private_key, public_key=public_key
                )
                logger.debug("LedgerSigner: loaded keypair from %s", self._keyfile)
            except Exception as exc:
                logger.warning(
                    "Failed to load keypair from %s (%s); generating new one",
                    self._keyfile, exc,
                )
                self._keypair = KeyPair.generate()
        else:
            self._keypair = KeyPair.generate()
            if self._keyfile:
                self._persist_key()

        # Cache base64 public key
        raw_pub = self._keypair.get_public_key_bytes()
        self._public_key_b64 = base64.b64encode(raw_pub).decode()

    def _persist_key(self) -> None:
        """Write the private key to the configured keyfile (PEM format)."""
        from cryptography.hazmat.primitives import serialization

        if not self._keyfile or not self._keypair:
            return
        try:
            self._keyfile.parent.mkdir(parents=True, exist_ok=True)
            pem = self._keypair.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            self._keyfile.write_bytes(pem)
            logger.debug("LedgerSigner: persisted keypair to %s", self._keyfile)
        except Exception as exc:
            logger.warning("Could not persist keypair: %s", exc)

    @property
    def public_key_b64(self) -> str:
        """Base64-encoded raw public key bytes."""
        if not self._public_key_b64:
            self.load_or_generate()
        return self._public_key_b64

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign_entry(
        self, content_hash: str, previous_hash: str
    ) -> EntrySignature:
        """
        Compute and sign the chain hash for a new ledger entry.

        Parameters
        ----------
        content_hash : str
            SHA-256 hex digest of the entry content.
        previous_hash : str
            ``chain_hash`` of the immediately preceding entry, or
            ``GENESIS_HASH`` for the first entry.

        Returns
        -------
        EntrySignature
        """
        if self._keypair is None:
            self.load_or_generate()

        chain_hash = _compute_chain_hash(content_hash, previous_hash)
        from prsm.core.cryptography.dag_signatures import sign_hash
        sig_b64 = sign_hash(chain_hash, self._keypair.private_key)

        return EntrySignature(
            chain_hash=chain_hash,
            signature_b64=sig_b64,
            public_key_b64=self._public_key_b64,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_entry(
        content: str,
        content_hash: str,
        previous_hash: str,
        entry_sig: EntrySignature,
    ) -> bool:
        """
        Verify a single entry's content and signature.

        Returns True if:
        - SHA-256(content) == content_hash
        - chain_hash recomputes correctly from content_hash + previous_hash
        - Ed25519 signature is valid over chain_hash
        """
        # 1. Recompute content hash
        computed_content = _sha256(content.encode("utf-8"))
        if computed_content != content_hash:
            return False

        # 2. Recompute chain hash
        computed_chain = _compute_chain_hash(content_hash, previous_hash)
        if computed_chain != entry_sig.chain_hash:
            return False

        # 3. Verify signature
        from prsm.core.cryptography.dag_signatures import verify_hash_signature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives import serialization

        try:
            raw_pub = base64.b64decode(entry_sig.public_key_b64)
            pub_key = Ed25519PublicKey.from_public_bytes(raw_pub)
            return verify_hash_signature(computed_chain, entry_sig.signature_b64, pub_key)
        except Exception:
            return False

    @staticmethod
    def verify_chain(entries: list) -> VerificationResult:
        """
        Verify the full chain of ``LedgerEntry`` objects.

        Parameters
        ----------
        entries : list[LedgerEntry]
            Ordered list of entries from the ledger (index 0 = oldest).

        Returns
        -------
        VerificationResult
        """
        if not entries:
            return VerificationResult(valid=True, entry_count=0)

        previous_chain_hash = GENESIS_HASH

        for i, entry in enumerate(entries):
            sig = EntrySignature(
                chain_hash=entry.chain_hash,
                signature_b64=entry.signature_b64,
                public_key_b64=entry.public_key_b64,
            )
            ok = LedgerSigner.verify_entry(
                content=entry.content,
                content_hash=entry.content_hash,
                previous_hash=previous_chain_hash,
                entry_sig=sig,
            )
            if not ok:
                return VerificationResult(
                    valid=False,
                    entry_count=i,
                    first_bad_index=i,
                    reason=(
                        f"Entry #{i} failed verification — "
                        "content, chain hash, or signature is invalid"
                    ),
                )
            # Advance chain
            previous_chain_hash = entry.chain_hash

        return VerificationResult(valid=True, entry_count=len(entries))


# ======================================================================
# Helpers
# ======================================================================

def hash_content(content: str) -> str:
    """Return the SHA-256 hex digest of *content* (UTF-8 encoded)."""
    return _sha256(content.encode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compute_chain_hash(content_hash: str, previous_hash: str) -> str:
    """
    chain_hash = SHA-256(content_hash + ':' + previous_hash)

    Signing the chain_hash (not just the content_hash) means the chain
    linkage itself is authenticated — an attacker cannot reorder entries
    while keeping per-entry signatures valid.
    """
    combined = f"{content_hash}:{previous_hash}".encode("utf-8")
    return _sha256(combined)
