"""Sprint 857 — Onramp conversion funnel pin tests.

Defends the polling-based conversion tracker that uses on-chain
truth (sp862 balance reader) instead of webhook delivery. Closes
the analytics gap from the sp853/sp855 onramp flows.
"""
from __future__ import annotations

import time

import pytest

from prsm.economy.web3.onramp_funnel import (
    OnrampFunnel,
    OnrampIntent,
    STATUS_INTENT_RECORDED,
    STATUS_PENDING_SETTLEMENT,
    STATUS_CONFIRMED,
    STATUS_EXPIRED,
)


class _FakeBalance:
    def __init__(self, usdc=0.0):
        self.usdc = usdc
        self.usdc_units = int(usdc * 10**6)


class _FakeReader:
    def __init__(self, balances=None, raises_for=None):
        self._b = balances or {}
        self._raises = raises_for or set()

    def get_balances(self, address):
        if address in self._raises:
            raise RuntimeError(f"rpc fail {address}")
        return self._b.get(address, _FakeBalance(0))


# ── Recording intents ────────────────────────────────────────

def test_record_intent_assigns_id_and_status(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    rec = f.record_intent(
        user_id="alice", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk_abc",
    )
    assert rec.intent_id.startswith("onramp_")
    assert rec.status == STATUS_INTENT_RECORDED
    assert rec.user_id == "alice"
    assert rec.expected_usd == 5.0


def test_record_intent_persists_to_disk(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    rec = f.record_intent(
        user_id="alice", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )
    assert (tmp_path / f"{rec.intent_id}.json").exists()


def test_record_intent_cross_restart_roundtrip(tmp_path):
    f1 = OnrampFunnel(persist_dir=tmp_path)
    rec = f1.record_intent(
        user_id="alice", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )
    # Fresh funnel reload from same dir
    f2 = OnrampFunnel(persist_dir=tmp_path)
    reloaded = f2.get_intent(rec.intent_id)
    assert reloaded is not None
    assert reloaded.user_id == "alice"
    assert reloaded.expected_usd == 5.0


# ── Sweep transitions ────────────────────────────────────────

def test_sweep_first_call_moves_to_pending(tmp_path):
    """First sweep with no USDC yet promotes
    INTENT_RECORDED → PENDING_SETTLEMENT."""
    f = OnrampFunnel(persist_dir=tmp_path)
    rec = f.record_intent(
        user_id="alice", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )
    reader = _FakeReader()
    summary = f.sweep(balance_reader=reader)
    assert summary["checked"] == 1
    assert summary["confirmed_new"] == 0
    assert f.get_intent(rec.intent_id).status == (
        STATUS_PENDING_SETTLEMENT
    )


def test_sweep_confirms_when_usdc_above_threshold(tmp_path):
    """5.0 USD expected; 4.92 USDC arrived (after Coinbase 1.5%
    fee) → CONFIRMED (above 95% threshold)."""
    f = OnrampFunnel(persist_dir=tmp_path)
    addr = "0x" + "11" * 20
    rec = f.record_intent(
        user_id="alice", destination_address=addr,
        expected_usd=5.0, session_token="tk",
    )
    reader = _FakeReader({addr: _FakeBalance(usdc=4.92)})
    summary = f.sweep(balance_reader=reader)
    assert summary["confirmed_new"] == 1
    confirmed = f.get_intent(rec.intent_id)
    assert confirmed.status == STATUS_CONFIRMED
    assert confirmed.usdc_received == 4.92
    assert confirmed.confirmed_at > 0


def test_sweep_does_not_confirm_below_threshold(tmp_path):
    """4.0 USDC on 5.0 expected = 80% < 95% threshold → stays
    PENDING_SETTLEMENT (could be partial-fund glitch, await
    full settlement)."""
    f = OnrampFunnel(persist_dir=tmp_path)
    addr = "0x" + "11" * 20
    rec = f.record_intent(
        user_id="alice", destination_address=addr,
        expected_usd=5.0, session_token="tk",
    )
    reader = _FakeReader({addr: _FakeBalance(usdc=4.0)})
    summary = f.sweep(balance_reader=reader)
    assert summary["confirmed_new"] == 0
    assert f.get_intent(rec.intent_id).status == (
        STATUS_PENDING_SETTLEMENT
    )


def test_sweep_expires_after_24h(tmp_path):
    """Intent created 25h ago with no USDC → EXPIRED."""
    f = OnrampFunnel(persist_dir=tmp_path)
    addr = "0x" + "11" * 20
    rec = f.record_intent(
        user_id="alice", destination_address=addr,
        expected_usd=5.0, session_token="tk",
    )
    # Backdate the intent 25h
    rec.created_at = time.time() - 25 * 3600
    f._persist(rec)
    reader = _FakeReader({addr: _FakeBalance(usdc=0)})
    summary = f.sweep(balance_reader=reader)
    assert summary["expired_new"] == 1
    expired = f.get_intent(rec.intent_id)
    assert expired.status == STATUS_EXPIRED


def test_sweep_idempotent_on_confirmed(tmp_path):
    """A CONFIRMED intent should not be re-checked. The sweep
    saves RPC quota by skipping terminal states."""
    f = OnrampFunnel(persist_dir=tmp_path)
    addr = "0x" + "11" * 20
    rec = f.record_intent(
        user_id="alice", destination_address=addr,
        expected_usd=5.0, session_token="tk",
    )
    reader = _FakeReader({addr: _FakeBalance(usdc=4.92)})
    f.sweep(balance_reader=reader)  # → CONFIRMED
    second = f.sweep(balance_reader=reader)
    assert second["checked"] == 0  # skipped
    assert second["confirmed_new"] == 0


def test_sweep_fail_soft_on_bad_rpc(tmp_path):
    """One flaky RPC read doesn't kill the sweep — other intents
    continue. Failed intent stays in its current status."""
    f = OnrampFunnel(persist_dir=tmp_path)
    bad_addr = "0x" + "ee" * 20
    good_addr = "0x" + "11" * 20
    bad = f.record_intent(
        user_id="alice", destination_address=bad_addr,
        expected_usd=5.0, session_token="tk",
    )
    good = f.record_intent(
        user_id="bob", destination_address=good_addr,
        expected_usd=5.0, session_token="tk",
    )
    reader = _FakeReader(
        {good_addr: _FakeBalance(usdc=4.92)},
        raises_for={bad_addr},
    )
    summary = f.sweep(balance_reader=reader)
    assert summary["confirmed_new"] == 1
    assert f.get_intent(good.intent_id).status == STATUS_CONFIRMED
    # Bad intent unchanged
    assert f.get_intent(bad.intent_id).status == (
        STATUS_INTENT_RECORDED
    )


# ── Summary ──────────────────────────────────────────────────

def test_summary_counts_by_status(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    r1 = f.record_intent(
        user_id="a", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="t1",
    )
    r2 = f.record_intent(
        user_id="b", destination_address="0x" + "22" * 20,
        expected_usd=10.0, session_token="t2",
    )
    # Manually transition r1 → CONFIRMED
    r1.status = STATUS_CONFIRMED
    r1.usdc_received = 4.92
    f._persist(r1)
    s = f.summary()
    assert s["total_intents"] == 2
    assert s["status_counts"][STATUS_CONFIRMED] == 1
    assert s["status_counts"][STATUS_INTENT_RECORDED] == 1
    assert s["total_expected_usd"] == 15.0
    assert s["total_confirmed_usdc"] == 4.92
    assert s["conversion_rate"] == 0.5


def test_summary_empty_funnel(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    s = f.summary()
    assert s["total_intents"] == 0
    assert s["conversion_rate"] == 0.0


# ── List filtering ───────────────────────────────────────────

def test_list_intents_filter_by_status(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    r1 = f.record_intent(
        user_id="a", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="t1",
    )
    r2 = f.record_intent(
        user_id="b", destination_address="0x" + "22" * 20,
        expected_usd=5.0, session_token="t2",
    )
    r1.status = STATUS_CONFIRMED
    f._persist(r1)
    confirmed = f.list_intents(status=STATUS_CONFIRMED)
    assert len(confirmed) == 1
    assert confirmed[0].intent_id == r1.intent_id


def test_list_intents_newest_first(tmp_path):
    f = OnrampFunnel(persist_dir=tmp_path)
    r1 = f.record_intent(
        user_id="a", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="t1",
    )
    time.sleep(0.01)
    r2 = f.record_intent(
        user_id="b", destination_address="0x" + "22" * 20,
        expected_usd=5.0, session_token="t2",
    )
    intents = f.list_intents()
    assert intents[0].intent_id == r2.intent_id
    assert intents[1].intent_id == r1.intent_id


# ── :memory: opt-out ─────────────────────────────────────────

def test_memory_sentinel_opts_out_of_persistence(monkeypatch):
    monkeypatch.setenv("PRSM_ONRAMP_FUNNEL_DIR", ":memory:")
    f = OnrampFunnel()
    assert f._persist_dir is None
    rec = f.record_intent(
        user_id="alice", destination_address="0x" + "11" * 20,
        expected_usd=5.0, session_token="tk",
    )
    # Same instance still works
    assert f.get_intent(rec.intent_id) is not None
    # No disk file (path is None, can't check)


def test_explicit_persist_dir_takes_precedence(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("PRSM_ONRAMP_FUNNEL_DIR", ":memory:")
    f = OnrampFunnel(persist_dir=tmp_path)
    assert f._persist_dir == tmp_path
    f.record_intent(
        user_id="x", destination_address="0x" + "11" * 20,
        expected_usd=1.0, session_token="t",
    )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
