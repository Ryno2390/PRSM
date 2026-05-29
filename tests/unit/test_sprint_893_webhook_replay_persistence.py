"""Sprint 893 — webhook replay ring survives daemon restart.

sp887 replay finding. The sp284 WebhookReplayRing is purely in-memory:
on daemon restart `_order` + `_set` reset to empty. node.py even
rationalizes this ("restart = fresh replay window, matching the
timestamp tolerance window") — but that rationale only holds for
PERSONA, which carries a signing timestamp `t=<unix>` and so has the
300s freshness window as a SECOND layer.

ONFIDO signatures carry NO timestamp. For Onfido the dedup ring is the
ONLY replay defense. So a captured, validly-signed Onfido webhook can
be replayed indefinitely across the next daemon restart (which happens
routinely — deploys, crashes, OOM-kills) to re-trigger the
KYC→VERIFIED→auto-provision + raised-tier-limit side effects. Even for
Persona, a restart timed inside the 300s window re-opens replay.

Sp893 makes the ring optionally persistent (persist_dir). Each seen
token is stored with its first-seen timestamp so a restart reloads the
recently-seen set. Retention is bounded by BOTH a FIFO count cap
(existing) AND a time window (new — addresses the sp887 unbounded-disk
DoS concern: expired tokens are dropped on reload). Pure in-memory
behavior (persist_dir=None) is unchanged — every sp284 test still holds.
"""
from __future__ import annotations

import json

from prsm.economy.web3.webhook_replay_defense import WebhookReplayRing


def _onfido_token(n: int) -> str:
    # Onfido replay token = the hex sha2 signature (no timestamp).
    return f"{n:064x}"


# ── The fix: a recorded token survives a restart (Onfido) ────

def test_replay_token_survives_restart(tmp_path):
    """The core gap. Record a token, then construct a FRESH ring on
    the same persist dir (simulating a daemon restart). The token
    must still be seen — a replay after restart is rejected."""
    ring = WebhookReplayRing(persist_dir=tmp_path)
    tok = _onfido_token(1)
    assert ring.record(tok) is True  # first occurrence

    # Daemon restarts: brand-new ring object, same dir.
    reloaded = WebhookReplayRing(persist_dir=tmp_path)
    assert reloaded.seen(tok) is True
    # A replay of the captured webhook after restart is rejected.
    assert reloaded.record(tok) is False


def test_record_after_reload_still_dedups_new_tokens(tmp_path):
    """Post-restart the ring still works for fresh tokens: first
    occurrence accepted, immediate replay rejected."""
    WebhookReplayRing(persist_dir=tmp_path).record(_onfido_token(1))
    reloaded = WebhookReplayRing(persist_dir=tmp_path)
    fresh = _onfido_token(99)
    assert reloaded.record(fresh) is True
    assert reloaded.record(fresh) is False


# ── In-memory behavior unchanged (regression on sp284) ───────

def test_in_memory_ring_has_no_persistence(tmp_path):
    """persist_dir=None (default) → nothing written, behaves exactly
    as the sp284 in-memory ring."""
    ring = WebhookReplayRing()  # default: no persist
    tok = _onfido_token(1)
    assert ring.record(tok) is True
    assert ring.record(tok) is False
    # A fresh in-memory ring knows nothing.
    assert WebhookReplayRing().seen(tok) is False


# ── Disk bounded by FIFO count (DoS defense, existing cap) ───

def test_persisted_ring_respects_fifo_count_bound(tmp_path):
    """Reload retains at most max_entries (oldest evicted from disk
    too) — disk cannot grow past the FIFO cap."""
    ring = WebhookReplayRing(max_entries=3, persist_dir=tmp_path)
    for i in range(5):
        ring.record(_onfido_token(i))  # 0,1,2,3,4 → keep 2,3,4
    reloaded = WebhookReplayRing(max_entries=3, persist_dir=tmp_path)
    assert reloaded.count() == 3
    assert reloaded.seen(_onfido_token(0)) is False  # evicted
    assert reloaded.seen(_onfido_token(1)) is False  # evicted
    assert reloaded.seen(_onfido_token(4)) is True   # kept


# ── Disk bounded by time (DoS defense, new) ──────────────────

def test_expired_tokens_dropped_on_reload(tmp_path):
    """Tokens older than the retention window are dropped on reload,
    bounding disk by TIME (addresses the sp887 unbounded-disk DoS)."""
    ring = WebhookReplayRing(
        persist_dir=tmp_path, retention_sec=100,
    )
    old = _onfido_token(1)
    recent = _onfido_token(2)
    ring.record(old, now=1_000.0)       # t=1000
    ring.record(recent, now=1_090.0)    # t=1090

    # Reload at t=1150: old (age 150 > 100) dropped, recent kept.
    reloaded = WebhookReplayRing(
        persist_dir=tmp_path, retention_sec=100, now=1_150.0,
    )
    assert reloaded.seen(old) is False
    assert reloaded.seen(recent) is True


# ── Fail-soft on a corrupt persist file (OnrampFunnel pattern) ─

def test_corrupt_persist_file_fails_soft(tmp_path):
    """A garbage ring file must not crash construction — start empty
    (matches sp857 OnrampFunnel fail-soft on bad records)."""
    (tmp_path / "replay-ring.json").write_text("{not valid json")
    ring = WebhookReplayRing(persist_dir=tmp_path)  # no raise
    assert ring.count() == 0
    # Still functional after recovering from corruption.
    assert ring.record(_onfido_token(1)) is True


def test_persist_file_is_valid_json_list(tmp_path):
    """The on-disk format is a JSON list of [token, ts] pairs —
    auditable + the load path's contract."""
    ring = WebhookReplayRing(persist_dir=tmp_path)
    ring.record(_onfido_token(7), now=1234.5)
    data = json.loads((tmp_path / "replay-ring.json").read_text())
    assert isinstance(data, list)
    assert [_onfido_token(7), 1234.5] in [list(x) for x in data]
