"""Sprint 508 — `prsm wallet info` shows ETH balance + gas status.

The existing sprint-446 `prsm wallet info` shows FTNS balance +
claimable royalties via direct RPC reads. It does NOT show ETH
balance, which means operators have to leave the CLI to check
their gas runway on BaseScan.

Sprint 508 adds an ETH balance line + a color-coded gas warning
to `prsm wallet info`. Same thresholds as sprint-502
(/wallet/gas-status: <0.0001 critical, <0.0005 low, ≥0.0005 ok).

Boundary: tests patch Web3 calls via the existing
_wallet_read_balance_wei pattern + a new
_wallet_read_eth_balance_wei helper. The CLI live-path was
already verified on Base mainnet (operator wallet 0x4acdE458…).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_helper_reads_eth_balance():
    """_wallet_read_eth_balance_wei must call eth.get_balance
    on a Web3 instance constructed with the supplied RPC URL."""
    from prsm.cli import _wallet_read_eth_balance_wei
    fake_w3 = MagicMock()
    fake_w3.eth.get_balance.return_value = 12345
    fake_w3.to_checksum_address.side_effect = lambda a: a

    with patch("prsm.cli.Web3" if False else "web3.Web3") as W:
        W.HTTPProvider = MagicMock()
        W.return_value = fake_w3
        W.to_checksum_address = lambda a: a
        result = _wallet_read_eth_balance_wei(
            "http://rpc", "0xAAAA",
        )
    assert result == 12345


def test_wallet_info_shows_eth_balance(monkeypatch):
    """`prsm wallet info` must include an ETH-balance line in
    the output. Patches both FTNS + ETH balance reads + the
    network/signer loaders so we don't hit a real RPC."""
    from prsm.cli import main as cli

    fake_cfg = MagicMock()
    fake_cfg.name = "Base Mainnet"
    fake_cfg.chain_id = 8453
    fake_cfg.explorer_url = "https://basescan.org"
    fake_cfg.ftns_token = (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )
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
        lambda *a, **kw: int(2 * 10**18),  # 2 FTNS
    )
    monkeypatch.setattr(
        "prsm.cli._wallet_read_eth_balance_wei",
        lambda *a, **kw: int(10**15),  # 0.001 ETH = ok
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "info", "--network", "mainnet"],
    )
    assert result.exit_code == 0, result.output
    assert "ETH balance" in result.output
    assert "0.001" in result.output  # 0.001 ETH


def test_wallet_info_shows_low_gas_warning(monkeypatch):
    """When ETH balance is below 'low' threshold, output must
    include a warning so operators don't miss it."""
    from prsm.cli import main as cli

    fake_cfg = MagicMock()
    fake_cfg.name = "Base"
    fake_cfg.chain_id = 8453
    fake_cfg.explorer_url = "https://basescan.org"
    fake_cfg.ftns_token = (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )
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
    # 0.0003 ETH → low
    monkeypatch.setattr(
        "prsm.cli._wallet_read_eth_balance_wei",
        lambda *a, **kw: int(3 * 10**14),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "info", "--network", "mainnet"],
    )
    assert result.exit_code == 0
    assert (
        "low" in result.output.lower()
        or "warning" in result.output.lower()
        or "⚠" in result.output
    )


def test_wallet_info_shows_critical_gas_warning(monkeypatch):
    """Critical balance must show prominent red warning."""
    from prsm.cli import main as cli

    fake_cfg = MagicMock()
    fake_cfg.name = "Base"
    fake_cfg.chain_id = 8453
    fake_cfg.explorer_url = "https://basescan.org"
    fake_cfg.ftns_token = (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )
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
    # 0.00005 ETH → critical
    monkeypatch.setattr(
        "prsm.cli._wallet_read_eth_balance_wei",
        lambda *a, **kw: int(5 * 10**13),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["wallet", "info", "--network", "mainnet"],
    )
    assert result.exit_code == 0
    assert (
        "critical" in result.output.lower()
        or "top up" in result.output.lower()
    )
