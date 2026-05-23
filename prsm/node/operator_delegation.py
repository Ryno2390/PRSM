"""Sprint 788 — verify operator_address claim before trusting it.

Pre-788 the daemon trusted whatever `operator_address` a peer
announced in its hardware_profile. This was a real authorization
gap — any peer could claim to be 0xRich's node and ride 0xRich's
stake-gated treatment.

Sprint 788 closes the gap. A peer's hardware_profile now carries
an `operator_delegation` blob: an EIP-191 signed claim from the
operator's ETH key that they authorize a specific node_id under
their address. The signing scheme reuses sprint 786's
`build_binding_message` so the same canonical message format
covers both onboarding (wallet_binding) and ongoing peer-attest
(this module).

Verifier contract:
    verify_operator_delegation_blob(
        node_id, operator_address, delegation,
    ) -> bool

  Returns True iff:
  - delegation is a well-shaped dict with the 4 required keys
  - delegation.node_id_hex == node_id (peer can't claim someone
    else's delegation as their own)
  - delegation.wallet_address recovers from the EIP-191 signature
  - recovered wallet matches the claimed operator_address

  Returns False on any malformed input, missing field, recovery
  failure, or mismatch. Never raises.

Pool-provider integration: `_hardware_profile_to_parallax_gpu`
treats operator_address as effectively unset (stake=0) when the
delegation is missing or invalid. Pre-788 behavior (trust the
bare claim) is removed but no peer is BROKEN — they just lose
stake-tier privilege until they ship a valid delegation.

Multi-device note: this is the same primitive that sprint 786's
multi-wallet-binding work uses on the onboarding side. One
operator's ETH key can sign delegations for ALL their devices;
each device's hardware_profile carries its own (operator_address,
node_id) attestation, all valid simultaneously.
"""
from __future__ import annotations

import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


def verify_operator_delegation_blob(
    *,
    node_id: str,
    operator_address: str,
    delegation: Optional[Any],
) -> bool:
    """Verify that `delegation` proves the holder of
    `operator_address`'s ETH key authorized this `node_id`.

    Returns False on any failure: malformed input, missing field,
    signature-recovery failure, recovered address mismatch,
    node_id mismatch. Never raises."""
    if not isinstance(delegation, dict):
        return False

    required = ("wallet_address", "node_id_hex", "issued_at_iso", "signature")
    for key in required:
        if key not in delegation:
            return False
        if not isinstance(delegation[key], str):
            return False

    # Peer can't present a delegation for some OTHER node_id and
    # claim it as their own.
    if delegation["node_id_hex"] != node_id:
        return False

    # The blob's claimed wallet must match the operator_address
    # the peer is announcing. Otherwise an attacker could swap
    # claimed addresses freely.
    try:
        from eth_utils import to_checksum_address
        claimed_wallet = to_checksum_address(delegation["wallet_address"])
        claimed_op = to_checksum_address(operator_address)
    except Exception:
        return False
    if claimed_wallet != claimed_op:
        return False

    # Recover the EIP-191 signer and check it matches the claimed
    # wallet. This is the load-bearing crypto check.
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        from prsm.interface.onboarding.wallet_binding import (
            build_binding_message,
        )
        msg = build_binding_message(
            claimed_wallet,
            delegation["node_id_hex"],
            delegation["issued_at_iso"],
        )
        encoded = encode_defunct(text=msg)
        recovered = Account.recover_message(
            encoded, signature=delegation["signature"],
        )
        return to_checksum_address(recovered) == claimed_wallet
    except Exception as exc:  # noqa: BLE001
        # Any recovery error (bad signature shape, encoding issue)
        # → reject. Don't leak the exception to the caller.
        logger.debug(
            "operator_delegation recover failed for node %s: %s",
            node_id[:8], exc,
        )
        return False
