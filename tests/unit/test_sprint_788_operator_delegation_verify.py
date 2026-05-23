"""Sprint 788 — verify operator_address claim before trusting it.

The pre-788 daemon trusts whatever `operator_address` a peer
announces in its hardware_profile. This means peer A can claim
to be operator_address=0xRich's node and get whatever stake-
gated treatment 0xRich has earned — the network has no proof
that 0xRich actually authorized A.

Sprint 788 closes the gap. A peer's hardware_profile now carries
an `operator_delegation` blob: an EIP-191 signed claim from the
operator's ETH key that they authorize node_id=X under their
address. Verifier reconstructs the canonical binding message
(reusing sprint 786's `build_binding_message` so the signing
scheme is uniform across the codebase) + recovers the signer +
checks it matches the claimed operator_address.

Without a valid delegation, the consumer treats operator_address
as effectively unset (stake_amount=0). Sprint 788 ships the
verifier + wires it into the pool provider's stake-lookup gate.

Pin tests for the pure verifier:
- valid delegation → True
- absent delegation (None) → False
- claimed operator_address doesn't match the signer → False
- claimed node_id doesn't match the delegation's node_id → False
- tampered signature → False (recovery fails OR mismatched recovery)
- malformed blob (missing keys / wrong types) → False (no raise)

Pin tests for pool-provider integration:
- hardware_profile with valid delegation → stake_reader called
  with operator_address
- hardware_profile with invalid delegation → stake_reader NOT
  called with operator_address (treated as unstaked)
- hardware_profile without delegation → stake_reader NOT called
  (treated as unstaked; the bare claim is no longer enough)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock


def _make_delegation_blob(eth_account, node_id_hex: str):
    """Construct a valid operator_delegation blob for a given
    ETH account + Ed25519 node_id."""
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(
        eth_account.address, node_id_hex, issued_at_iso,
    )
    encoded = encode_defunct(text=msg)
    signed = eth_account.sign_message(encoded)
    return {
        "wallet_address": eth_account.address,
        "node_id_hex": node_id_hex,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }


# ---- Pure verifier ---------------------------------------------


def test_verify_accepts_honest_delegation():
    from eth_account import Account
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    acct = Account.create()
    node_id = "a" * 32
    blob = _make_delegation_blob(acct, node_id)
    assert verify_operator_delegation_blob(
        node_id=node_id,
        operator_address=acct.address,
        delegation=blob,
    ) is True


def test_verify_rejects_none_delegation():
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    assert verify_operator_delegation_blob(
        node_id="a" * 32,
        operator_address="0x" + "1" * 40,
        delegation=None,
    ) is False


def test_verify_rejects_wrong_signer():
    """Operator A signs; peer claims operator_address = B."""
    from eth_account import Account
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    acct_a = Account.create()
    acct_b = Account.create()
    node_id = "a" * 32
    # Blob signed by A (legitimate)
    blob = _make_delegation_blob(acct_a, node_id)
    # Peer claims B is the operator → verifier should reject
    assert verify_operator_delegation_blob(
        node_id=node_id,
        operator_address=acct_b.address,
        delegation=blob,
    ) is False


def test_verify_rejects_mismatched_node_id():
    """Delegation says node X; peer claims to be node Y."""
    from eth_account import Account
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    acct = Account.create()
    blob_for_x = _make_delegation_blob(acct, "x" * 32)
    # Verify with a DIFFERENT node_id → reject
    assert verify_operator_delegation_blob(
        node_id="y" * 32,  # peer's actual node_id
        operator_address=acct.address,
        delegation=blob_for_x,
    ) is False


def test_verify_rejects_tampered_signature():
    """Flip one hex char in the signature → recover fails or
    yields a different signer."""
    from eth_account import Account
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    acct = Account.create()
    node_id = "a" * 32
    blob = _make_delegation_blob(acct, node_id)
    # Flip one char in the signature
    sig = blob["signature"]
    flipped_char = "1" if sig[10] != "1" else "2"
    blob["signature"] = sig[:10] + flipped_char + sig[11:]
    assert verify_operator_delegation_blob(
        node_id=node_id,
        operator_address=acct.address,
        delegation=blob,
    ) is False


def test_verify_rejects_malformed_blob():
    """Garbage shape (missing keys / wrong types) → False, no raise."""
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    cases = [
        {},
        {"wallet_address": "0x" + "a" * 40},  # missing other keys
        {"signature": "not-hex"},
        "not-a-dict",
        42,
    ]
    for bad in cases:
        assert verify_operator_delegation_blob(
            node_id="a" * 32,
            operator_address="0x" + "1" * 40,
            delegation=bad,
        ) is False


# ---- Pool-provider integration ---------------------------------


def _mock_stake_reader():
    r = MagicMock()
    r.stake_amount_for = MagicMock(return_value=1000)
    return r


def _make_hw(node_id="a" * 32, operator_address=None, delegation=None):
    hw = {
        "node_id": node_id,
        "gpu_api": "",
        "gpu_name": "",
        "tflops_fp16": 1.0,
        "gpu_vram_gb": 4.0,
        "ram_total_gb": 8.0,
    }
    if operator_address:
        hw["operator_address"] = operator_address
    if delegation:
        hw["operator_delegation"] = delegation
    return hw


def test_pool_provider_accepts_valid_delegation():
    """Peer with valid delegation → stake_reader called with
    the operator_address."""
    from eth_account import Account
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    acct = Account.create()
    node_id = "a" * 32
    blob = _make_delegation_blob(acct, node_id)
    hw = _make_hw(
        node_id=node_id,
        operator_address=acct.address,
        delegation=blob,
    )
    reader = _mock_stake_reader()
    gpu = _hw_dict_to_parallax_gpu(
        node_id, hw, region="default",
        stake_reader=reader,
    )
    assert gpu is not None
    # The reader was consulted with the legitimate operator_address.
    reader.stake_amount_for.assert_called_with(acct.address)
    assert gpu.stake_amount == 1000


def test_pool_provider_rejects_missing_delegation():
    """Peer claims operator_address WITHOUT a delegation →
    operator_address NOT trusted; reader not called with it;
    stake_amount=0."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = _make_hw(
        node_id="a" * 32,
        operator_address="0x" + "f" * 40,  # claimed but no proof
        delegation=None,
    )
    reader = _mock_stake_reader()
    gpu = _hw_dict_to_parallax_gpu(
        "a" * 32, hw, region="default",
        stake_reader=reader,
    )
    assert gpu is not None
    assert gpu.stake_amount == 0  # treated as unstaked
    # Reader was not asked about the unverified operator_address
    reader.stake_amount_for.assert_not_called()


def test_pool_provider_rejects_invalid_delegation():
    """Delegation present but invalid (wrong signer) →
    operator_address NOT trusted; stake_amount=0."""
    from eth_account import Account
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    acct_a = Account.create()
    acct_b = Account.create()
    node_id = "a" * 32
    # Delegation signed by A; peer claims to be B
    blob_a = _make_delegation_blob(acct_a, node_id)
    hw = _make_hw(
        node_id=node_id,
        operator_address=acct_b.address,
        delegation=blob_a,
    )
    reader = _mock_stake_reader()
    gpu = _hw_dict_to_parallax_gpu(
        node_id, hw, region="default",
        stake_reader=reader,
    )
    assert gpu is not None
    assert gpu.stake_amount == 0
    reader.stake_amount_for.assert_not_called()
