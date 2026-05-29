"""Sprint 897 — OnrampFunnel persistence is bounded.

sp887 DoS finding ("unbounded funnel intents on disk"). sp857's
OnrampFunnel writes one JSON file per intent and NEVER prunes:
``_load_from_disk`` globs EVERY file on each restart, so a production
node serving onboarding for months accumulates unbounded files —
monotonic disk growth AND a startup that degrades linearly with
lifetime intent count (the glob + parse of every record).

sp897 bounds it the way sp893 bounded the replay ring:
  - TERMINAL intents (CONFIRMED / EXPIRED) older than retention_sec are
    pruned (in-memory + their file deleted). Active intents are never
    age-pruned (the 24h sweep expiry << retention, so anything still
    active is genuinely recent).
  - A hard count cap drops the oldest-by-created_at when exceeded
    (flood protection).
  - Pruning runs on load, after each sweep, and on record_intent
    overflow — so disk stays bounded even without restarts.

Long-term conversion analytics live in the compliance ring + sp872 CSV
export (the durable regulatory record); the funnel is an operational
RECENT-conversion surface, so bounding it to recent history is correct.
"""
from __future__ import annotations

from prsm.economy.web3.onramp_funnel import (
    OnrampFunnel,
    OnrampIntent,
    STATUS_CONFIRMED,
    STATUS_EXPIRED,
    STATUS_INTENT_RECORDED,
)


def _write_intent(
    funnel, intent_id, *, status, created_at, confirmed_at=0.0,
    expired_at=0.0,
):
    """Inject a record directly + persist, to simulate aged history."""
    rec = OnrampIntent(
        intent_id=intent_id,
        user_id="u",
        destination_address="0x" + "11" * 20,
        expected_usd=100.0,
        session_token="tok",
        created_at=created_at,
        status=status,
        confirmed_at=confirmed_at,
        expired_at=expired_at,
    )
    funnel._records[intent_id] = rec
    funnel._persist(rec)
    return rec


# ── Age-prune terminal records on reload ─────────────────────

def test_old_terminal_intents_pruned_on_load(tmp_path):
    """A CONFIRMED intent older than retention is dropped on reload —
    in memory AND its file deleted."""
    seed = OnrampFunnel(persist_dir=tmp_path)
    _write_intent(
        seed, "old_confirmed", status=STATUS_CONFIRMED,
        created_at=1_000.0, confirmed_at=1_000.0,
    )
    assert (tmp_path / "old_confirmed.json").exists()

    # Reload "now" = 1_000 + 40 days, retention 30 days → pruned.
    reloaded = OnrampFunnel(
        persist_dir=tmp_path,
        retention_sec=30 * 86_400,
        now=1_000.0 + 40 * 86_400,
    )
    assert reloaded.get_intent("old_confirmed") is None
    assert not (tmp_path / "old_confirmed.json").exists()


def test_recent_terminal_intents_preserved(tmp_path):
    seed = OnrampFunnel(persist_dir=tmp_path)
    _write_intent(
        seed, "recent_confirmed", status=STATUS_CONFIRMED,
        created_at=1_000.0, confirmed_at=1_000.0,
    )
    reloaded = OnrampFunnel(
        persist_dir=tmp_path,
        retention_sec=30 * 86_400,
        now=1_000.0 + 5 * 86_400,  # 5 days < 30 → kept
    )
    assert reloaded.get_intent("recent_confirmed") is not None


def test_active_intents_never_age_pruned(tmp_path):
    """An INTENT_RECORDED that's somehow old must NOT be age-pruned —
    only terminal records age out (active ones are bounded by count)."""
    seed = OnrampFunnel(persist_dir=tmp_path)
    _write_intent(
        seed, "stuck_active", status=STATUS_INTENT_RECORDED,
        created_at=1_000.0,
    )
    reloaded = OnrampFunnel(
        persist_dir=tmp_path,
        retention_sec=30 * 86_400,
        now=1_000.0 + 999 * 86_400,  # ancient, but still active
    )
    assert reloaded.get_intent("stuck_active") is not None


# ── Hard count cap (flood protection) ────────────────────────

def test_count_cap_drops_oldest_on_load(tmp_path):
    seed = OnrampFunnel(persist_dir=tmp_path)
    for i in range(6):
        _write_intent(
            seed, f"i{i}", status=STATUS_EXPIRED,
            created_at=float(i), expired_at=float(i),
        )
    # Reload with cap=3, generous retention → keep 3 newest by created.
    reloaded = OnrampFunnel(
        persist_dir=tmp_path,
        retention_sec=10 ** 12,
        max_intents=3,
        now=10.0,
    )
    ids = {r.intent_id for r in reloaded.list_intents()}
    assert ids == {"i3", "i4", "i5"}
    assert not (tmp_path / "i0.json").exists()
    assert (tmp_path / "i5.json").exists()


# ── Pruning runs after a sweep (bounded without restarts) ────

class _NoBalance:
    class _B:
        usdc = 0.0
        usdc_units = 0
    def get_balances(self, addr):
        return self._B()


def test_sweep_prunes_old_terminal_intents(tmp_path):
    """A long-running node (no restart) stays bounded: a sweep prunes
    terminal records older than retention."""
    funnel = OnrampFunnel(
        persist_dir=tmp_path, retention_sec=30 * 86_400,
    )
    _write_intent(
        funnel, "old_expired", status=STATUS_EXPIRED,
        created_at=1_000.0, expired_at=1_000.0,
    )
    # Sweep with "now" 40 days later → old terminal pruned.
    funnel.sweep(
        balance_reader=_NoBalance(), now=1_000.0 + 40 * 86_400,
    )
    assert funnel.get_intent("old_expired") is None
    assert not (tmp_path / "old_expired.json").exists()


# ── Regression: normal record/sweep behavior intact ──────────

def test_record_and_sweep_still_work(tmp_path):
    funnel = OnrampFunnel(persist_dir=tmp_path)
    rec = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "22" * 20,
        expected_usd=50.0,
        session_token="tok",
    )
    assert funnel.get_intent(rec.intent_id) is not None
    summary = funnel.sweep(balance_reader=_NoBalance())
    assert summary["checked"] == 1
    # First sweep with no balance → PENDING_SETTLEMENT, still present.
    assert funnel.get_intent(rec.intent_id) is not None
