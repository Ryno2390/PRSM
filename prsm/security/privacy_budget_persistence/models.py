"""
Privacy budget persistence — entry data models.

Phase 3.x.4 Task 1.

Defines the wire format for the chained-per-entry privacy-budget journal:

- ``PrivacyBudgetEntryType`` — enum distinguishing a spend event from a
  signed reset event.
- ``PrivacyBudgetEntry`` — one row per spend/reset, sequence-numbered
  and chained via ``prev_entry_hash`` so any historical tamper invalidates
  every subsequent signature.

Mirrors the canonical-bytes idiom from
``prsm.compute.model_registry.models.ModelManifest`` so the project has
exactly one signing-payload pattern across all Ed25519-signed artifacts.
Verifiers reading either format can apply the same mental model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


# Schema version for the canonical signing payload. Bump when the payload
# format changes; verifiers checking signatures across protocol versions
# will refuse to validate entries whose declared schema_version they
# don't understand.
ENTRY_SCHEMA_VERSION = 1

# Domain separator that prevents an Ed25519 signature over one PRSM
# artifact from being replayed against another. Distinct from
# ``MANIFEST_SIGNING_DOMAIN`` (Phase 3.x.2) and the inference-receipt
# signing prefix (Phase 3.x.1).
ENTRY_SIGNING_DOMAIN = b"prsm-privacy-budget-entry:v1"

# Genesis prev-hash sentinel: the first entry in any journal has no
# predecessor. Use 32 zero bytes (matches sha256 output length so chain
# verification logic is uniform — every entry's prev_entry_hash is
# always 32 bytes long).
GENESIS_PREV_HASH = b"\x00" * 32


class PrivacyBudgetEntryType(str, Enum):
    """Distinguishes the two journal event types.

    SPEND — caller called ``record_spend(epsilon, operation, model_id)``.
    RESET — caller called ``reset()``. Stored in the journal as a signed
    event rather than wiping in-memory state silently, so auditors can
    reconstruct exactly when and at what cumulative ε the reset happened.
    """

    SPEND = "spend"
    RESET = "reset"


@dataclass(frozen=True)
class PrivacyBudgetEntry:
    """One signed event in a privacy-budget journal.

    Per the Phase 3.x.4 trust model: each entry is signed under the
    node's Ed25519 identity AND chains to its predecessor via
    ``prev_entry_hash``. Tampering with entry N changes its signing
    payload bytes; the next entry's ``prev_entry_hash`` then no longer
    matches, so every subsequent entry's signature also fails to
    verify. Auditors detect tampering by walking the chain end-to-end.

    The journal's wire format. Filesystem (Task 4), in-memory (Task 3),
    and any future DHT/on-chain transport (Phase 3.x.3) all treat
    ``json.dumps(entry.to_dict(), sort_keys=True)`` as the canonical
    serialization unit.
    """

    sequence_number: int                 # 0-indexed, monotonic, gap-free
    entry_type: PrivacyBudgetEntryType
    node_id: str                         # signer's NodeIdentity.node_id
    epsilon: float                       # 0.0 for RESET; positive for SPEND
    operation: str                       # e.g. "inference", "forge_query"; "" for RESET
    model_id: str                        # "" for RESET
    timestamp: float                     # unix timestamp (float seconds)
    prev_entry_hash: bytes               # 32-byte sha256 of prev signing payload (or GENESIS)
    schema_version: int = ENTRY_SCHEMA_VERSION
    signature: bytes = b""               # Excluded from signing payload

    def __post_init__(self) -> None:
        # Frozen dataclass — coerce types after init.
        if not isinstance(self.entry_type, PrivacyBudgetEntryType):
            object.__setattr__(
                self, "entry_type", PrivacyBudgetEntryType(self.entry_type)
            )
        if not isinstance(self.sequence_number, int):
            object.__setattr__(self, "sequence_number", int(self.sequence_number))
        if not isinstance(self.epsilon, float):
            object.__setattr__(self, "epsilon", float(self.epsilon))
        if not isinstance(self.timestamp, float):
            object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.schema_version, int):
            object.__setattr__(self, "schema_version", int(self.schema_version))

        # Validate prev_entry_hash length: every chain link must be the
        # same width as a sha256 digest, so chain-verification logic is
        # uniform without a special case for genesis.
        if not isinstance(self.prev_entry_hash, bytes):
            raise TypeError(
                f"prev_entry_hash must be bytes, got {type(self.prev_entry_hash).__name__}"
            )
        if len(self.prev_entry_hash) != 32:
            raise ValueError(
                f"prev_entry_hash must be 32 bytes (sha256 width), "
                f"got {len(self.prev_entry_hash)} bytes"
            )

        if not isinstance(self.signature, bytes):
            raise TypeError(
                f"signature must be bytes, got {type(self.signature).__name__}"
            )

        # Sequence numbers are externally-assigned; gap-free invariant
        # is enforced by the store, not the dataclass. But we can reject
        # negatives here as a sanity check — they're always wrong.
        if self.sequence_number < 0:
            raise ValueError(
                f"sequence_number must be >= 0, got {self.sequence_number}"
            )

    def signing_payload(self) -> bytes:
        """Canonical bytes used for ``signature`` generation/verification.

        Excludes ``signature`` itself (would be circular). Includes
        ``schema_version`` (downgrade-attack defense) and
        ``prev_entry_hash`` (chain tamper-detection).

        Field order is fixed and pinned by ``ENTRY_SCHEMA_VERSION``;
        do not reorder without bumping the schema version + adding a
        migration test.

        Numeric formats:
          - epsilon as ``:.10f``: pins precision so two clients
            reconstructing the payload byte-for-byte agree.
          - timestamp as ``:.6f``: same rationale, microsecond precision.
        """
        parts = [
            ENTRY_SIGNING_DOMAIN.decode("ascii"),
            str(self.schema_version),
            str(self.sequence_number),
            self.entry_type.value,
            self.node_id,
            f"{self.epsilon:.10f}",
            self.operation,
            self.model_id,
            f"{self.timestamp:.6f}",
            self.prev_entry_hash.hex(),
        ]
        return "|".join(parts).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_number": self.sequence_number,
            "entry_type": self.entry_type.value,
            "node_id": self.node_id,
            "epsilon": self.epsilon,
            "operation": self.operation,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "prev_entry_hash": self.prev_entry_hash.hex(),
            "schema_version": self.schema_version,
            "signature": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyBudgetEntry":
        d = dict(data)
        if "prev_entry_hash" in d and isinstance(d["prev_entry_hash"], str):
            d["prev_entry_hash"] = bytes.fromhex(d["prev_entry_hash"])
        if "signature" in d and isinstance(d["signature"], str):
            d["signature"] = bytes.fromhex(d["signature"])
        if "entry_type" in d and not isinstance(d["entry_type"], PrivacyBudgetEntryType):
            d["entry_type"] = PrivacyBudgetEntryType(d["entry_type"])
        # Drop unknown keys gracefully so additive schema changes don't
        # break old loaders.
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})
