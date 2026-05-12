"""Sprint 304 — recipient-encrypted upload primitive.

Vision §7 Enterprise Confidentiality Mode foundation.

Hybrid encryption scheme:
  1. Random 32-byte symmetric key + random 12-byte nonce
  2. ChaCha20-Poly1305 AEAD encrypts plaintext under that key
  3. Per recipient, generate an ephemeral X25519 keypair,
     perform ECDH with the recipient's static pubkey,
     HKDF-derive a sealing key, and seal the symmetric key
     under ChaCha20-Poly1305 with the recipient's
     identifier as additional-authenticated-data (AAD).

OR-decrypt semantics: any one designated recipient can
decrypt independently. No one outside the recipient set
can — FTNS balance is irrelevant to the cryptography.

The AAD-binding of the recipient identifier into the sealed
key prevents manifest entry-swap attacks: an attacker
who relabels Alice's sealed key as Bob's cannot trick Bob's
private key into decrypting Alice's sealed key, because the
AAD ("Alice") no longer matches the relabeled identifier
("Bob") at unseal time.

Decryption is fully client-side. The PRSM node serving the
retrieve never sees plaintext.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import (
    ChaCha20Poly1305,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import (
    hashes,
    serialization,
)

# Sprint 307 — threshold mode (Shamir share splitting)
from prsm.enterprise.shamir import (
    Share, reconstruct_secret, split_secret,
)


MANIFEST_VERSION = "v1"

_HKDF_INFO = b"prsm-recipient-encryption-v1"
_AAD_PREFIX = b"prsm-recipient-id-v1:"
_CONTENT_NONCE_LEN = 12
_SEAL_NONCE_LEN = 12
_SYMMETRIC_KEY_LEN = 32


# ── b64 helpers ──────────────────────────────────────


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(s: str) -> bytes:
    if not isinstance(s, str):
        raise ValueError(
            f"expected base64 string, got {type(s).__name__}"
        )
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}")


# ── Dataclasses ──────────────────────────────────────


@dataclass
class EnterpriseRecipient:
    identifier: str
    x25519_pubkey_b64: str


@dataclass
class RecipientEntry:
    identifier: str
    ephemeral_pubkey_b64: str
    nonce_b64: str
    sealed_symmetric_key_b64: str
    # Sprint 307 — threshold mode. When non-None, the
    # `sealed_symmetric_key_b64` payload is actually a
    # sealed Shamir SHARE (y-values), and `share_index`
    # is the Shamir x-coordinate (1..n).
    share_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "identifier": self.identifier,
            "ephemeral_pubkey_b64": self.ephemeral_pubkey_b64,
            "nonce_b64": self.nonce_b64,
            "sealed_symmetric_key_b64": (
                self.sealed_symmetric_key_b64
            ),
        }
        if self.share_index is not None:
            d["share_index"] = int(self.share_index)
        return d

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "RecipientEntry":
        si = d.get("share_index")
        return cls(
            identifier=d["identifier"],
            ephemeral_pubkey_b64=d["ephemeral_pubkey_b64"],
            nonce_b64=d["nonce_b64"],
            sealed_symmetric_key_b64=d[
                "sealed_symmetric_key_b64"
            ],
            share_index=(
                int(si) if si is not None else None
            ),
        )


@dataclass
class ThresholdParams:
    """Sprint 307 — t-of-n threshold parameters.
    Stored on the manifest when threshold mode is active."""

    t: int
    n: int

    def __post_init__(self):
        if self.t < 1:
            raise ValueError(
                f"threshold t must be >= 1, got {self.t}"
            )
        if self.n < self.t:
            raise ValueError(
                f"threshold n must be >= t; got n={self.n}, "
                f"t={self.t}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"t": int(self.t), "n": int(self.n)}

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "ThresholdParams":
        return cls(t=int(d["t"]), n=int(d["n"]))


@dataclass
class ShareContribution:
    """Sprint 307 — a recipient's unsealed Shamir share,
    ready to contribute to the t-of-n reconstruction. The
    share_y_values are the polynomial evaluations for each
    secret byte at the recipient's x-coordinate."""

    share_index: int
    share_y_values: bytes


@dataclass
class RecipientManifest:
    version: str = MANIFEST_VERSION
    entries: List[RecipientEntry] = field(
        default_factory=list,
    )
    # Sprint 307 — threshold mode. None = OR-decrypt
    # (sprint 304 default); non-None = t-of-n Shamir.
    threshold: Optional[ThresholdParams] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "version": self.version,
            "entries": [e.to_dict() for e in self.entries],
        }
        if self.threshold is not None:
            d["threshold"] = self.threshold.to_dict()
        return d

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "RecipientManifest":
        version = d.get("version", "")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"unknown manifest version {version!r}; "
                f"expected {MANIFEST_VERSION!r}"
            )
        th_raw = d.get("threshold")
        threshold = (
            ThresholdParams.from_dict(th_raw)
            if th_raw is not None else None
        )
        return cls(
            version=version,
            entries=[
                RecipientEntry.from_dict(e)
                for e in (d.get("entries") or [])
            ],
            threshold=threshold,
        )


@dataclass
class EncryptedPayload:
    ciphertext_b64: str
    manifest: RecipientManifest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ciphertext_b64": self.ciphertext_b64,
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "EncryptedPayload":
        return cls(
            ciphertext_b64=d["ciphertext_b64"],
            manifest=RecipientManifest.from_dict(
                d["manifest"],
            ),
        )


# ── Keypair generation ───────────────────────────────


def generate_recipient_keypair() -> tuple[str, str]:
    """Generate a fresh X25519 recipient keypair. Returns
    (private_key_b64, public_key_b64). The public key is
    what the enterprise distributes / posts to PRSM; the
    private key never leaves the recipient's environment."""
    priv = X25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64e(priv_raw), _b64e(pub_raw)


# ── Internals ────────────────────────────────────────


def _load_pubkey(pub_b64: str) -> X25519PublicKey:
    raw = _b64d(pub_b64)
    if len(raw) != 32:
        raise ValueError(
            f"x25519 pubkey must be 32 bytes, got {len(raw)}"
        )
    return X25519PublicKey.from_public_bytes(raw)


def _load_privkey(priv_b64: str) -> X25519PrivateKey:
    raw = _b64d(priv_b64)
    if len(raw) != 32:
        raise ValueError(
            f"x25519 privkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return X25519PrivateKey.from_private_bytes(raw)


def _derive_sealing_key(shared_secret: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_SYMMETRIC_KEY_LEN,
        salt=None,
        info=_HKDF_INFO,
    ).derive(shared_secret)


def _aad_for_identifier(identifier: str) -> bytes:
    return _AAD_PREFIX + identifier.encode("utf-8")


# ── Encrypt ──────────────────────────────────────────


def encrypt_for_recipients(
    plaintext: bytes,
    recipients: List[EnterpriseRecipient],
) -> EncryptedPayload:
    if not recipients:
        raise ValueError(
            "must specify at least one recipient"
        )
    seen_ids: set[str] = set()
    for r in recipients:
        if r.identifier in seen_ids:
            raise ValueError(
                f"duplicate recipient identifier "
                f"{r.identifier!r}"
            )
        seen_ids.add(r.identifier)
        # Validate pubkey eagerly so misconfigured recipient
        # lists fail BEFORE we burn an encryption.
        _load_pubkey(r.x25519_pubkey_b64)

    sym_key = os.urandom(_SYMMETRIC_KEY_LEN)
    content_nonce = os.urandom(_CONTENT_NONCE_LEN)
    content_ct = ChaCha20Poly1305(sym_key).encrypt(
        content_nonce, plaintext, None,
    )
    ciphertext_b64 = _b64e(content_nonce + content_ct)

    entries: List[RecipientEntry] = []
    for r in recipients:
        recipient_pub = _load_pubkey(r.x25519_pubkey_b64)
        ephemeral_priv = X25519PrivateKey.generate()
        ephemeral_pub = ephemeral_priv.public_key()
        shared = ephemeral_priv.exchange(recipient_pub)
        sealing_key = _derive_sealing_key(shared)
        seal_nonce = os.urandom(_SEAL_NONCE_LEN)
        sealed = ChaCha20Poly1305(sealing_key).encrypt(
            seal_nonce,
            sym_key,
            _aad_for_identifier(r.identifier),
        )
        eph_pub_raw = ephemeral_pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        entries.append(RecipientEntry(
            identifier=r.identifier,
            ephemeral_pubkey_b64=_b64e(eph_pub_raw),
            nonce_b64=_b64e(seal_nonce),
            sealed_symmetric_key_b64=_b64e(sealed),
        ))

    return EncryptedPayload(
        ciphertext_b64=ciphertext_b64,
        manifest=RecipientManifest(entries=entries),
    )


# ── Decrypt ──────────────────────────────────────────


def decrypt_for_recipient(
    payload: EncryptedPayload,
    recipient_privkey_b64: str,
) -> bytes:
    """OR-decrypt: try every manifest entry against the
    provided private key. First entry that unseals wins.

    Refuses on threshold-mode payloads — the caller must
    use the two-phase API
    (unseal_share_for_recipient + combine_shares_and_decrypt)
    to reconstruct the symmetric key from t shares."""
    if payload.manifest.threshold is not None:
        raise ValueError(
            "this payload is threshold-encrypted; use "
            "unseal_share_for_recipient + "
            "combine_shares_and_decrypt instead of "
            "decrypt_for_recipient"
        )
    recipient_priv = _load_privkey(recipient_privkey_b64)

    sym_key: bytes | None = None
    for entry in payload.manifest.entries:
        try:
            eph_pub = _load_pubkey(
                entry.ephemeral_pubkey_b64,
            )
        except ValueError:
            continue
        try:
            shared = recipient_priv.exchange(eph_pub)
            sealing_key = _derive_sealing_key(shared)
            seal_nonce = _b64d(entry.nonce_b64)
            sealed = _b64d(entry.sealed_symmetric_key_b64)
            sym_key = ChaCha20Poly1305(sealing_key).decrypt(
                seal_nonce,
                sealed,
                _aad_for_identifier(entry.identifier),
            )
            break
        except Exception:
            sym_key = None
            continue

    if sym_key is None:
        raise ValueError(
            "no entry in manifest decryptable with the "
            "provided private key"
        )

    full = _b64d(payload.ciphertext_b64)
    if len(full) < _CONTENT_NONCE_LEN:
        raise ValueError("ciphertext too short")
    content_nonce = full[:_CONTENT_NONCE_LEN]
    content_ct = full[_CONTENT_NONCE_LEN:]
    return ChaCha20Poly1305(sym_key).decrypt(
        content_nonce, content_ct, None,
    )


# ── Sprint 307 — threshold (t-of-n) encryption mode ──


def encrypt_for_threshold(
    plaintext: bytes,
    recipients: List[EnterpriseRecipient],
    *,
    threshold: int,
) -> EncryptedPayload:
    """Encrypt for a t-of-n recipient set. Any t of the n
    recipients must cooperate to decrypt; t-1 reveal
    nothing about the plaintext.

    Composes onto the OR-decrypt EncryptedPayload shape:
    same hybrid encryption (ChaCha20-Poly1305 over the
    plaintext under a random symmetric key), but the
    symmetric key is split into n Shamir shares (each 32
    bytes) and each share is sealed for one recipient with
    that recipient's identifier in the AEAD AAD.

    Decryption is two-phase:
      1. Each recipient unseals their share locally
         via unseal_share_for_recipient
      2. Any t share contributions are pooled via
         combine_shares_and_decrypt
    """
    if not recipients:
        raise ValueError(
            "must specify at least one recipient"
        )
    n = len(recipients)
    if threshold < 1 or threshold > n:
        raise ValueError(
            f"threshold must be in [1, n={n}]; got "
            f"{threshold}"
        )

    # Validate recipient pubkeys + identifier uniqueness
    # eagerly (sprint 304 contract).
    seen_ids: set[str] = set()
    for r in recipients:
        if r.identifier in seen_ids:
            raise ValueError(
                f"duplicate recipient identifier "
                f"{r.identifier!r}"
            )
        seen_ids.add(r.identifier)
        _load_pubkey(r.x25519_pubkey_b64)

    # 1. Generate symmetric key + encrypt content
    sym_key = os.urandom(_SYMMETRIC_KEY_LEN)
    content_nonce = os.urandom(_CONTENT_NONCE_LEN)
    content_ct = ChaCha20Poly1305(sym_key).encrypt(
        content_nonce, plaintext, None,
    )
    ciphertext_b64 = _b64e(content_nonce + content_ct)

    # 2. Split symmetric key into n Shamir shares
    shares = split_secret(sym_key, t=threshold, n=n)

    # 3. Seal each share for the corresponding recipient
    entries: List[RecipientEntry] = []
    for r, share in zip(recipients, shares):
        recipient_pub = _load_pubkey(r.x25519_pubkey_b64)
        ephemeral_priv = X25519PrivateKey.generate()
        ephemeral_pub = ephemeral_priv.public_key()
        shared = ephemeral_priv.exchange(recipient_pub)
        sealing_key = _derive_sealing_key(shared)
        seal_nonce = os.urandom(_SEAL_NONCE_LEN)
        sealed = ChaCha20Poly1305(sealing_key).encrypt(
            seal_nonce,
            share.y_values,
            _aad_for_identifier(r.identifier),
        )
        eph_pub_raw = ephemeral_pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        entries.append(RecipientEntry(
            identifier=r.identifier,
            ephemeral_pubkey_b64=_b64e(eph_pub_raw),
            nonce_b64=_b64e(seal_nonce),
            sealed_symmetric_key_b64=_b64e(sealed),
            share_index=share.index,
        ))

    return EncryptedPayload(
        ciphertext_b64=ciphertext_b64,
        manifest=RecipientManifest(
            entries=entries,
            threshold=ThresholdParams(t=threshold, n=n),
        ),
    )


def unseal_share_for_recipient(
    payload: EncryptedPayload,
    recipient_privkey_b64: str,
) -> ShareContribution:
    """A recipient unseals THEIR share of the symmetric
    key. Pure client-side: no plaintext is exposed yet —
    only the recipient's share. Returns a
    ShareContribution that the recipient can later pool
    with t-1 others to reconstruct the symmetric key."""
    if payload.manifest.threshold is None:
        raise ValueError(
            "this payload is OR-decrypt mode; use "
            "decrypt_for_recipient instead"
        )
    recipient_priv = _load_privkey(recipient_privkey_b64)

    for entry in payload.manifest.entries:
        if entry.share_index is None:
            continue
        try:
            eph_pub = _load_pubkey(
                entry.ephemeral_pubkey_b64,
            )
        except ValueError:
            continue
        try:
            shared = recipient_priv.exchange(eph_pub)
            sealing_key = _derive_sealing_key(shared)
            seal_nonce = _b64d(entry.nonce_b64)
            sealed = _b64d(entry.sealed_symmetric_key_b64)
            share_y = ChaCha20Poly1305(sealing_key).decrypt(
                seal_nonce,
                sealed,
                _aad_for_identifier(entry.identifier),
            )
            return ShareContribution(
                share_index=int(entry.share_index),
                share_y_values=share_y,
            )
        except Exception:
            continue

    raise ValueError(
        "no entry in manifest decryptable with the "
        "provided private key"
    )


def combine_shares_and_decrypt(
    payload: EncryptedPayload,
    contributions: List[ShareContribution],
) -> bytes:
    """Pool >= t share contributions to reconstruct the
    symmetric key and decrypt the ciphertext. Each
    contribution is one recipient's unsealed share —
    typically obtained via unseal_share_for_recipient.

    Tampering at any layer surfaces here:
      - tampered ciphertext → ChaCha20-Poly1305 InvalidTag
      - tampered sealed share → AEAD failure during the
        recipient's unseal_share call (never reaches here)
      - tampered share y-values → reconstructed symmetric
        key is wrong → ChaCha20-Poly1305 InvalidTag
    """
    if payload.manifest.threshold is None:
        raise ValueError(
            "this payload is OR-decrypt mode; use "
            "decrypt_for_recipient"
        )
    threshold = payload.manifest.threshold
    if len(contributions) < threshold.t:
        raise ValueError(
            f"combine requires at least t={threshold.t} "
            f"share contributions; got "
            f"{len(contributions)}"
        )
    indices = [c.share_index for c in contributions]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "duplicate share indices not allowed in "
            "share contributions"
        )

    # Reconstruct symmetric key from the first t shares
    used = contributions[:threshold.t]
    shares = [
        Share(
            index=c.share_index,
            y_values=c.share_y_values,
        )
        for c in used
    ]
    sym_key = reconstruct_secret(shares, t=threshold.t)

    # Decrypt content
    full = _b64d(payload.ciphertext_b64)
    if len(full) < _CONTENT_NONCE_LEN:
        raise ValueError("ciphertext too short")
    content_nonce = full[:_CONTENT_NONCE_LEN]
    content_ct = full[_CONTENT_NONCE_LEN:]
    return ChaCha20Poly1305(sym_key).decrypt(
        content_nonce, content_ct, None,
    )
