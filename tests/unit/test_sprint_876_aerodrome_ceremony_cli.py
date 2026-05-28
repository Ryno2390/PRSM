"""Sprint 876 — `prsm node aerodrome-ceremony` CLI pin tests."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from prsm.cli import node_aerodrome_ceremony


_SAFE = "0xCCAc7b21695De068979b1ca47B0cfBD328654220"


# ── Validation ───────────────────────────────────────────────

def test_rejects_invalid_safe_address():
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", "notanaddr",
            "--seed-usdc", "1", "--seed-ftns", "1",
        ],
    )
    assert result.exit_code == 2
    assert "foundation_safe" in result.output


def test_rejects_zero_seed_amount():
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "0", "--seed-ftns", "1",
        ],
    )
    assert result.exit_code == 2


def test_exits_1_when_no_outputs_specified():
    """User must explicitly request artifacts — running with no
    --output flags should exit non-zero with an example command,
    so we don't silently no-op."""
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
        ],
    )
    assert result.exit_code == 1
    assert (
        "No --output-json" in result.output
        or "nothing written" in result.output
    )


# ── JSON output ──────────────────────────────────────────────

def test_json_output_writes_safe_tx_builder_format(tmp_path):
    out = tmp_path / "batch.json"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "50000", "--seed-ftns", "50000",
            "--output-json", str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    batch = json.loads(out.read_text())
    assert batch["version"] == "1.0"
    assert batch["chainId"] == "84532"  # sepolia
    assert len(batch["transactions"]) == 3


def test_mainnet_batch_chain_id_8453(tmp_path):
    out = tmp_path / "batch.json"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "mainnet",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
            "--output-json", str(out),
        ],
    )
    assert result.exit_code == 0
    batch = json.loads(out.read_text())
    assert batch["chainId"] == "8453"


def test_mainnet_run_shows_real_money_warning():
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "mainnet",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "50000", "--seed-ftns", "50000",
        ],
    )
    # Even though it exits 1 (no outputs), the warning should
    # appear in the output before that exit
    assert "real money" in result.output.lower() or "MAINNET" in result.output


# ── Runbook output ───────────────────────────────────────────

def test_runbook_output_writes_markdown(tmp_path):
    out = tmp_path / "runbook.md"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
            "--output-runbook", str(out),
        ],
    )
    assert result.exit_code == 0
    body = out.read_text()
    assert body.startswith("# PRSM Aerodrome USDC↔FTNS")
    assert _SAFE in body


def test_both_outputs_can_be_written_together(tmp_path):
    out_j = tmp_path / "b.json"
    out_r = tmp_path / "r.md"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
            "-j", str(out_j), "-r", str(out_r),
        ],
    )
    assert result.exit_code == 0
    assert out_j.exists()
    assert out_r.exists()


# ── Slippage + opening price math ────────────────────────────

def test_summary_shows_opening_price(tmp_path):
    """Opening price = USDC / FTNS — the load-bearing economic
    number co-signers see before signing."""
    out = tmp_path / "b.json"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            # $10/FTNS opening
            "--seed-usdc", "50000", "--seed-ftns", "5000",
            "-j", str(out),
        ],
    )
    assert result.exit_code == 0
    assert "$10.000000 per FTNS" in result.output


def test_custom_slippage_passed_through(tmp_path):
    out = tmp_path / "b.json"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
            "--slippage-bps", "500",
            "-j", str(out),
        ],
    )
    assert result.exit_code == 0
    batch = json.loads(out.read_text())
    assert "slippage_bps=500" in batch["meta"]["description"]


# ── Next-steps guidance ──────────────────────────────────────

def test_next_steps_shown_after_successful_output(tmp_path):
    out = tmp_path / "b.json"
    runner = CliRunner()
    result = runner.invoke(
        node_aerodrome_ceremony,
        [
            "--network", "sepolia",
            "--foundation-safe", _SAFE,
            "--seed-usdc", "1", "--seed-ftns", "1",
            "-j", str(out),
        ],
    )
    assert result.exit_code == 0
    assert "Next steps" in result.output
    assert "wallet.safe.global" in result.output
    assert "AERODROME_USDC_FTNS_POOL_ADDRESS" in result.output
