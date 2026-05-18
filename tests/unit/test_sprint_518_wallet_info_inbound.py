"""Sprint 518 — `prsm wallet info` shows recent inbound count.

Existing `prsm wallet info` (sprints 446, 508) shows FTNS balance,
ETH balance, claimable royalties — all RPC-direct. Sprint 518 adds
recent inbound FTNS count by reusing sprint-512's scan_inbound
helper. RPC-direct so it works without a running daemon.

This makes `prsm wallet info` the canonical single-command
operator dashboard: balance + gas + claimable + recent inbound,
all readable from any RPC.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner


def test_helper_reads_inbound_count():
    """New _wallet_read_inbound_count helper must return
    (count, total_ftns) for a given block window."""
    from prsm.cli import _wallet_read_inbound_count

    # Mock Web3 + contract paths
    with _MockWeb3(
        block_number=1000,
        transfer_logs=[
            ({"from": "0x" + "f" * 40, "to": "0x" + "a" * 40,
              "value": int(2.5e18)}, 900),
            ({"from": "0x" + "e" * 40, "to": "0x" + "a" * 40,
              "value": int(1.5e18)}, 950),
        ],
    ):
        count, total = _wallet_read_inbound_count(
            rpc_url="http://fake",
            ftns_token="0x" + "b" * 40,
            address="0x" + "a" * 40,
            lookback_blocks=200,
        )
    assert count == 2
    assert total == 4.0


def test_wallet_info_shows_inbound_line(monkeypatch):
    """`prsm wallet info` output must include an inbound
    line with count + total_ftns."""
    from prsm.cli import main as cli

    fake_cfg = MagicMock()
    fake_cfg.name = "mainnet"
    fake_cfg.chain_id = 8453
    fake_cfg.explorer_url = "https://basescan.org"
    fake_cfg.ftns_token = "0x" + "b" * 40
    fake_cfg.royalty_distributor = None
    fake_cfg.notes = []

    monkeypatch.setattr(
        "prsm.cli._wallet_load_signer",
        lambda _n: {
            "network": fake_cfg,
            "address": "0x4acdE458766C704B2511583572303e77109cFFE8",
            "rpc_url": "http://fake",
        },
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_balance_wei",
        lambda *a, **kw: int(2e18),
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_eth_balance_wei",
        lambda *a, **kw: int(1e15),
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_inbound_count",
        lambda *a, **kw: (9, 2.000008),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "info", "--network", "mainnet"],
    )
    assert result.exit_code == 0, result.output
    assert "Inbound" in result.output or "inbound" in result.output
    assert "9" in result.output
    assert "2.000008" in result.output


def test_wallet_info_inbound_failure_is_yellow_skip(
    monkeypatch,
):
    """Inbound RPC failure must not fail the whole
    command — just print a yellow warning so the other
    sections still display."""
    from prsm.cli import main as cli

    fake_cfg = MagicMock()
    fake_cfg.name = "mainnet"
    fake_cfg.chain_id = 8453
    fake_cfg.explorer_url = "https://basescan.org"
    fake_cfg.ftns_token = "0x" + "b" * 40
    fake_cfg.royalty_distributor = None
    fake_cfg.notes = []

    monkeypatch.setattr(
        "prsm.cli._wallet_load_signer",
        lambda _n: {
            "network": fake_cfg, "address": "0xABC",
            "rpc_url": "http://fake",
        },
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_balance_wei",
        lambda *a, **kw: 0,
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_eth_balance_wei",
        lambda *a, **kw: int(1e15),
    )

    def fail(*a, **kw):
        raise RuntimeError("rpc broken")

    monkeypatch.setattr(
        "prsm.cli._wallet_read_inbound_count", fail,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "info", "--network", "mainnet"],
    )
    # Other sections still printed; warning surfaced
    assert result.exit_code == 0
    assert "FTNS balance" in result.output
    assert "ETH balance" in result.output
    assert (
        "Inbound" in result.output
        or "inbound" in result.output
    )
    assert (
        "rpc broken" in result.output
        or "read failed" in result.output
    )


class _MockWeb3:
    """Context manager that monkeypatches web3.Web3 to a
    fake implementation returning configured logs +
    block_number."""

    def __init__(self, block_number, transfer_logs):
        self.block_number = block_number
        self.transfer_logs = transfer_logs

    def __enter__(self):
        import web3
        self._orig = web3.Web3

        outer = self
        class _FakeEth:
            block_number = outer.block_number

            @staticmethod
            def contract(address=None, abi=None):
                c = MagicMock()
                logs = []
                for argdict, blk in outer.transfer_logs:
                    log = MagicMock()
                    log.blockNumber = blk
                    log.transactionHash = b"\x00" * 32
                    log.args.__getitem__ = (
                        lambda self, k, _d=argdict: _d[k]
                    )
                    logs.append(log)
                c.events.Transfer.get_logs.return_value = logs
                return c

        class _FakeW3:
            def __init__(self, *a, **kw):
                self.eth = _FakeEth()
            @staticmethod
            def HTTPProvider(*a, **kw):
                return None
            @staticmethod
            def to_checksum_address(a):
                return a

        web3.Web3 = _FakeW3
        return self

    def __exit__(self, *a):
        import web3
        web3.Web3 = self._orig
