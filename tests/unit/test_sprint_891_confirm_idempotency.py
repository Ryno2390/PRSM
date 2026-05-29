"""Sprint 891 — funnel CONFIRMED transition is persisted before the
callback fires (at-most-once compliance record + webhook).

sp887 finding #19/#22. The sweep set rec.status=CONFIRMED in memory,
fired on_confirmed (sp885 compliance record + sp874 webhook), and
ONLY persisted the CONFIRMED state at the end of the loop iteration
— AFTER the callback. So a concurrent reader / post-crash reload
that swept the same persist dir during the callback still saw the
intent as PENDING on disk → re-confirmed it → fired the callback a
SECOND time. Impact: double-counted settled volume against the sp884
tier limit (corrupting the enforcement sp884/885 established) +
duplicate downstream completion webhook.

Sp891 persists the CONFIRMED transition BEFORE firing the callback,
so any concurrent/reloaded sweeper observes CONFIRMED and skips it.
The callback fires at most once per intent.
"""
from __future__ import annotations

from prsm.economy.web3.onramp_funnel import (
    OnrampFunnel,
    STATUS_CONFIRMED,
)


class _FakeBalance:
    def __init__(self, usdc):
        self.usdc = usdc
        self.usdc_units = int(usdc * 1e6)


class _ConfirmingReader:
    def get_balances(self, addr):
        return _FakeBalance(98.5)  # >= expected*0.95 → CONFIRM


def _intent(funnel):
    return funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=100.0,
        session_token="tok",
    )


# ── Persist-before-callback ordering ─────────────────────────

def test_intent_durably_confirmed_before_callback(tmp_path):
    """A reader/reload opened DURING the on_confirmed callback must
    see the intent already CONFIRMED on disk — proving the
    transition was persisted before the callback ran."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    intent = _intent(funnel)
    seen = []

    def on_confirmed(rec):
        # Simulate a concurrent process / post-crash reload reading
        # the same persist dir mid-callback.
        reloaded = OnrampFunnel(persist_dir=tmp_path)
        r2 = reloaded.get_intent(rec.intent_id)
        seen.append(r2.status if r2 else None)

    funnel.sweep(
        balance_reader=_ConfirmingReader(), on_confirmed=on_confirmed,
    )
    assert seen == [STATUS_CONFIRMED]


def test_concurrent_reload_sweep_does_not_double_fire(tmp_path):
    """The real double-fire scenario: while process A's
    on_confirmed callback runs, process B (a reload of the same
    dir) sweeps the same intent. B must observe CONFIRMED and NOT
    re-fire its own callback."""
    funnel_a = OnrampFunnel(persist_dir=tmp_path)
    intent = _intent(funnel_a)
    b_fired = []

    def on_confirmed_a(rec):
        funnel_b = OnrampFunnel(persist_dir=tmp_path)
        funnel_b.sweep(
            balance_reader=_ConfirmingReader(),
            on_confirmed=lambda r: b_fired.append(r.intent_id),
        )

    summary_a = funnel_a.sweep(
        balance_reader=_ConfirmingReader(),
        on_confirmed=on_confirmed_a,
    )
    assert summary_a["confirmed_new"] == 1
    # Process B saw the already-persisted CONFIRMED → did NOT
    # re-confirm or re-fire.
    assert b_fired == []


def test_compliance_not_double_counted_on_concurrent_reload(
    tmp_path,
):
    """End-to-end integrity: with the sp885 compliance ring wired,
    a concurrent reload-sweep during the callback must NOT
    double-count the user's settled volume against the tier limit."""
    from prsm.economy.web3.fiat_compliance_ring import (
        FiatComplianceRing,
    )
    from prsm.economy.web3.onramp_to_swap_orchestrator import (
        make_on_confirmed_callback,
    )

    class _Aero:
        def is_configured(self): return False
        def quote_swap(self, *a, **k): return None

    ring = FiatComplianceRing()
    funnel_a = OnrampFunnel(persist_dir=tmp_path)
    intent = _intent(funnel_a)

    # Process A's real callback (records onramp_execute) +, nested,
    # a process-B reload sweep with its OWN ring-recording callback.
    def on_confirmed_a(rec):
        # record A's settled volume
        make_on_confirmed_callback(
            funnel=funnel_a, aerodrome_client=_Aero(),
            ftns_address="0x" + "ab" * 20, compliance_ring=ring,
        )(rec)
        # B reloads + sweeps the same dir mid-callback
        funnel_b = OnrampFunnel(persist_dir=tmp_path)
        funnel_b.sweep(
            balance_reader=_ConfirmingReader(),
            on_confirmed=make_on_confirmed_callback(
                funnel=funnel_b, aerodrome_client=_Aero(),
                ftns_address="0x" + "ab" * 20, compliance_ring=ring,
            ),
        )

    funnel_a.sweep(
        balance_reader=_ConfirmingReader(),
        on_confirmed=on_confirmed_a,
    )
    # Settled volume counted EXACTLY once ($98.5), not $197.0.
    assert ring.total_usd_for_user("alice") == 98.5


def test_normal_single_sweep_still_confirms_and_fires(tmp_path):
    """Regression: the happy path still confirms + fires once."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    intent = _intent(funnel)
    fired = []
    summary = funnel.sweep(
        balance_reader=_ConfirmingReader(),
        on_confirmed=lambda r: fired.append(r.intent_id),
    )
    assert summary["confirmed_new"] == 1
    assert fired == [intent.intent_id]
    assert funnel.get_intent(intent.intent_id).status == (
        STATUS_CONFIRMED
    )


def test_second_sweep_after_confirm_does_not_refire(tmp_path):
    """In-process re-sweep skips the already-CONFIRMED intent."""
    funnel = OnrampFunnel(persist_dir=tmp_path)
    intent = _intent(funnel)
    fired = []
    cb = lambda r: fired.append(r.intent_id)
    funnel.sweep(balance_reader=_ConfirmingReader(), on_confirmed=cb)
    funnel.sweep(balance_reader=_ConfirmingReader(), on_confirmed=cb)
    assert fired == [intent.intent_id]  # exactly once
