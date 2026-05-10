"""A6 randomness beacon for the QueryOrchestrator's `beacon_provider`
callable contract.

Per threat-model `docs/2026-05-07-aggregator-selector-threat-model.md`
§A6 + §"Open governance questions" item 4: production wiring uses a
daily Foundation-multisig beacon + every-100th-query on-chain anchor.
This module supplies the daily-beacon half via deterministic
derivation from the Foundation Safe address + UTC day index.

The beacon binds aggregator selection to:
  - Foundation governance entity (network running under a different
    Safe gets a different beacon series — A6 forensic)
  - UTC day (rotates daily; the orchestrator's selection becomes
    replayable within the same day for audit but unreplayable across
    days)

v1 limitations (separately followed-on):
  - Daily rotation is deterministic — adversaries can predict
    tomorrow's beacon. Mitigated by the every-100th-query on-chain
    anchor (Foundation Safe member's signed beacon update). That
    anchor lands in a separate ratification + tooling sprint.
  - No multi-source mixing (e.g., block.prevrandao). v1 trusts the
    Foundation Safe key as the entropy authority; cross-source
    mixing is a follow-on once the on-chain anchor lands.

Per `docs/2026-05-08-query-orchestrator-wiring-readiness.md` Blocker 4
"beacon_provider" — supporting piece for B7.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable

# Approximate UTC day length in seconds. Leap seconds intentionally
# ignored — the beacon doesn't need to align with civil calendar
# perfectly, only rotate "roughly daily". Pinned for test
# reproducibility.
_DAY_SECONDS = 86_400


@dataclass
class FoundationBeaconProvider:
    """Callable adapter returning a 32-byte daily beacon.

    Produces a deterministic-per-day beacon by hashing
    `f"{foundation_safe_address}|{day_index}"`. The day index is
    `unix_timestamp // 86_400` — rolls over at UTC midnight (close
    enough; threat model doesn't require civil-calendar alignment).

    Attributes
    ----------
    foundation_safe_address:
        The Foundation Safe's deployment address (e.g., Base mainnet
        `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`). Bound into the
        beacon material so a network operating under a different Safe
        gets a different beacon series — forensic anchor for A6.
    time_source:
        Callable returning current unix timestamp. Defaults to
        `time.time()`. Tests inject a deterministic source.
    """

    foundation_safe_address: str
    time_source: Callable[[], float] = field(
        default_factory=lambda: time.time,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.foundation_safe_address, str):
            raise TypeError(
                f"foundation_safe_address must be str, got "
                f"{type(self.foundation_safe_address).__name__}"
            )
        if not self.foundation_safe_address:
            raise ValueError("foundation_safe_address must be non-empty")

    def __call__(self) -> bytes:
        """Return the 32-byte beacon for the current UTC day."""
        day_index = int(self.time_source()) // _DAY_SECONDS
        material = f"{self.foundation_safe_address}|{day_index}".encode("utf-8")
        return hashlib.sha256(material).digest()
