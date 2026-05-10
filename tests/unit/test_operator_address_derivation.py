"""Operator-address derivation from FTNS_WALLET_PRIVATE_KEY.

Operator running a node typically has FTNS_WALLET_PRIVATE_KEY
set already (for FTNS ledger / royalty claim / staking). The
on-chain address derivable from that key IS their operator
address — requiring a second `PRSM_OPERATOR_ADDRESS` env var is
redundant and error-prone (typo = wrong address surfaces in
heartbeat status).

Derivation rule:
  if PRSM_OPERATOR_ADDRESS set: use it (explicit override wins)
  elif FTNS_WALLET_PRIVATE_KEY set: derive via eth_account
  else: None (heartbeat stream of earnings-summary degrades)
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.node.operator_address import resolve_operator_address


# Test private key (well-known test key, never used on mainnet).
TEST_PK = (
    "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
)
# Address derived from the above private key.
TEST_ADDR = "0x2c7536E3605D9C16a7a3D7b1898e529396a65c23"


class TestExplicitOverride:
    def test_explicit_address_wins(self):
        with patch.dict(os.environ, {
            "PRSM_OPERATOR_ADDRESS": "0xEXPLICIT",
            "FTNS_WALLET_PRIVATE_KEY": TEST_PK,
        }, clear=False):
            assert resolve_operator_address() == "0xEXPLICIT"

    def test_explicit_blank_falls_through_to_derive(self):
        with patch.dict(os.environ, {
            "PRSM_OPERATOR_ADDRESS": "  ",  # whitespace = unset
            "FTNS_WALLET_PRIVATE_KEY": TEST_PK,
        }, clear=False):
            result = resolve_operator_address()
            assert result is not None
            assert result.lower() == TEST_ADDR.lower()


class TestDerivation:
    def test_derives_from_private_key_when_not_set(self):
        env = {
            "FTNS_WALLET_PRIVATE_KEY": TEST_PK,
        }
        env.pop("PRSM_OPERATOR_ADDRESS", None)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("PRSM_OPERATOR_ADDRESS", None)
            result = resolve_operator_address()
        assert result is not None
        assert result.lower() == TEST_ADDR.lower()

    def test_returns_none_when_neither_set(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_operator_address()
            assert result is None

    def test_handles_pk_without_0x_prefix(self):
        with patch.dict(os.environ, {
            "FTNS_WALLET_PRIVATE_KEY": TEST_PK[2:],  # strip 0x
        }, clear=True):
            result = resolve_operator_address()
            assert result is not None
            assert result.lower() == TEST_ADDR.lower()

    def test_invalid_pk_returns_none(self):
        with patch.dict(os.environ, {
            "FTNS_WALLET_PRIVATE_KEY": "not_a_valid_key",
        }, clear=True):
            result = resolve_operator_address()
            # Fail-soft: bad PK shouldn't crash node startup
            assert result is None
