"""Sprint 512 — inbound FTNS detection.

Sprints 498-511 fully covered OUTBOUND TX (broadcast by the daemon).
INBOUND FTNS (sent to the operator wallet by external parties) are
invisible — the operator only discovers them via manual BaseScan
check or by polling balanceOf.

Sprint 512 ships the read-side: scan ERC-20 Transfer event logs
where `to == operator_address`. Pure helper + SQLite persistence +
endpoint. Background polling is sprint-513.

Boundary: tests inject a mocked w3 returning canned log structures.
Real chain-scan + endpoint integration is exercised by the live
verify against the operator's actual 2 FTNS receipt from
Foundation Safe earlier in this conversation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _mk_log(block_number, tx_hash, from_addr, to_addr, amount):
    """Build a Web3-style Transfer event log dict."""
    from eth_utils import to_checksum_address
    return {
        "blockNumber": block_number,
        "transactionHash": MagicMock(
            hex=lambda: tx_hash,
        ),
        "args": MagicMock(
            **{"from": to_checksum_address(from_addr)},
        ),
        "to": to_checksum_address(to_addr),
        "amount": amount,
    }


def test_helper_returns_empty_when_no_logs():
    """Empty log set → empty list."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers

    contract = MagicMock()
    contract.events.Transfer.get_logs.return_value = []
    result = scan_inbound_transfers(
        contract,
        recipient="0x" + "a" * 40,
        from_block=0, to_block=100,
    )
    assert result == []


def test_helper_returns_inbound_entries():
    """Mocked log set → list of dicts with canonical schema."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers

    contract = MagicMock()
    log1 = MagicMock()
    log1.blockNumber = 46159546
    log1.transactionHash = b"\xde\xad\xbe\xef" + b"\x00" * 28
    log1.args.__getitem__ = lambda self, k: {
        "from": "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
        "to": "0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57",
        "value": 2 * 10**18,
    }[k]
    contract.events.Transfer.get_logs.return_value = [log1]

    result = scan_inbound_transfers(
        contract,
        recipient="0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57",
        from_block=46100000, to_block=46200000,
    )
    assert len(result) == 1
    r = result[0]
    assert r["block_number"] == 46159546
    assert r["from_address"] == (
        "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
    )
    assert r["to_address"] == (
        "0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57"
    )
    assert r["amount_ftns"] == 2.0


def test_helper_uses_to_filter():
    """The contract.events.Transfer filter must be created
    with `argument_filters={'to': recipient}` so the RPC
    returns only relevant logs — no client-side filtering
    of the whole chain's Transfer history."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers
    from eth_utils import to_checksum_address

    contract = MagicMock()
    contract.events.Transfer.get_logs.return_value = []
    scan_inbound_transfers(
        contract,
        recipient="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        from_block=100, to_block=200,
    )
    contract.events.Transfer.get_logs.assert_called_once()
    kwargs = (
        contract.events.Transfer.get_logs.call_args.kwargs
    )
    assert kwargs["from_block"] == 100
    assert kwargs["to_block"] == 200
    af = kwargs["argument_filters"]
    # to address must be checksummed
    assert af["to"] == to_checksum_address(
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )


# Endpoint test moved to
# test_sprint_512_inbound_endpoint.py to dodge asyncio.AUTO
# mode interaction with sync TestClient (same pattern as
# sprint 501/510).
