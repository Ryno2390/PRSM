"""Sprint 830 — invariant: every router_registry router is
either mounted in node/api.py:create_api_app OR explicitly
documented as deferred/inert.

The F28 lesson (sprint 829): `prsm.interface.api.wallet_api`'s
router was only mounted in `app_factory.py`'s router_registry
path, but `prsm node start` builds its app via
`prsm.node.api.create_api_app` which never included it. Result:
the entire 9-sprint multi-device arc (sprints 786-794) was
INERT in production despite 84+ passing unit tests.

Sprint 829 fixed wallet_api specifically. Sprint 830 generalizes
the fix into a CI invariant so this class of bug can't recur
silently:

  - Enumerate every `from prsm.interface.api.X import router`
    in `router_registry.py`.
  - For each, resolve its prefix.
  - Verify either:
      (a) the prefix appears inline in `prsm/node/api.py` (route
          defined directly) OR `prsm/node/api.py` calls
          `app.include_router(<that_router>)`, OR
      (b) the router is in the documented allow-list of
          deferred/inert arcs (with a clear reason).

Adding a new router to router_registry without either mounting
it in create_api_app OR allow-listing it fails the test.

The allow-list is a knowingly-deferred set — each entry should
be paired with a sprint task to either mount it or remove the
router. It is NOT a place to suppress findings without action.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent


# Documented deferred/inert arcs. Each entry must explain WHY
# the router is not mounted. New entries require a follow-up
# sprint ticket — this list is NOT a "shut up the test" knob.
#
# Sprint 830 baseline (2026-05-24): 10 router-only arcs found
# unmounted in create_api_app during the audit. They are allow-
# listed here pending individual operator-impact assessment +
# selective mount sprints (831+). Removing one from this list
# without mounting it should fail the test.
DEFERRED_INERT_ROUTERS = {
    "credential_api": (
        "Sprint 830 — deferred. /api/v1/credentials/* (register/"
        "validate/rotate/status) ships in router but no CLI/SDK "
        "consumer surfaced operator-side need yet."
    ),
    "security_status_api": (
        "Sprint 830 — deferred. /api/v1/security/* overlaps with "
        "inline /audit + /admin/security surfaces; needs scope "
        "review before mount."
    ),
    "security_logging_api": (
        "Sprint 830 — deferred. /api/v1/security/logging/* — see "
        "security_status_api note."
    ),
    "payment_api": (
        "Sprint 830 — deferred. /api/v1/payments/* — Phase-5 "
        "fiat ramp uses /wallet/onramp/quote inline; payment_api "
        "is older surface; needs scope review."
    ),
    "cryptography_api": (
        "Sprint 830 — deferred. /api/v1/crypto/* — operator-side "
        "crypto ops use compute/inference verify path; this "
        "router is unscoped."
    ),
    "governance_api": (
        "Sprint 830 — deferred. /api/v1/governance/* — Foundation "
        "Safe governance is multi-sig-only per Vision §14; this "
        "router predates that decision."
    ),
    "budget_api": (
        "Sprint 830 — deferred. /api/v1/budget/* — budgets ship "
        "via /compute/inference body's budget_ftns field; this "
        "router is unscoped."
    ),
    "ftns_api": (
        "Sprint 830 — deferred. /api/v1/ftns/* (balance/transfer/"
        "transactions) — inline /balance + /balance/onchain "
        "cover the operator-facing path; this router is legacy."
    ),
    "mainnet_deployment_api": (
        "Sprint 830 — deferred. /api/v1/mainnet/* — mainnet "
        "deploys are hardware-wallet ceremonies (Vision §14); "
        "no daemon-mediated path."
    ),
    "content_api": (
        "Sprint 830 — deferred. /api/v1/content/* — inline "
        "/content/* paths (publish/retrieve/search) cover the "
        "live operator surface; this router is parallel."
    ),
    "analytics_api": (
        "Sprint 830 — deferred. /api/v1/analytics/* — operator "
        "analytics ship via /metrics + /admin/* dashboards; this "
        "router is parallel."
    ),
    "teams_api": (
        "Sprint 830 — deferred. teams router (no prefix) — Teams "
        "feature is design-phase; no production CLI/SDK consumer "
        "exercises this surface yet."
    ),
    "monitoring_api": (
        "Sprint 830 — deferred. monitoring router (no prefix) — "
        "operator monitoring ships via /admin/* + /metrics; this "
        "router is parallel."
    ),
    "compliance_api": (
        "Sprint 830 — deferred. compliance router (no prefix) — "
        "AUSTRAC/FinCEN/IRS-ready compliance ring fires "
        "automatically inline (sprint 451); this router is "
        "redundant."
    ),
    "contributor_api": (
        "Sprint 830 — deferred. contributor router (no prefix) — "
        "contributor flows are design-phase; no production "
        "consumer."
    ),
    "session_api": (
        "Sprint 830 — deferred. session router (no prefix) — "
        "session management uses inline /agents + /compute "
        "paths; this router is legacy."
    ),
    "task_api": (
        "Sprint 830 — deferred. task router (no prefix) — task "
        "queue uses inline /compute/jobs + /compute/job/{id} "
        "paths; this router is parallel."
    ),
    "ui_api": (
        "Sprint 830 — deferred. ui router (no prefix) — UI "
        "surfaces ship via /onboarding/* + dashboard mount; "
        "this router is legacy."
    ),
    "bittorrent_router": (
        "Sprint 830 — deferred. /api/v1/torrents/* — bittorrent "
        "content distribution is design-phase; no production "
        "operator consumer yet."
    ),
}


def _registry_router_imports():
    """Returns dict[module_name, router_alias] for every
    `from prsm.interface.api.<X> import router as <alias>` in
    router_registry.py."""
    src = (
        REPO_ROOT / "prsm" / "interface" / "api"
        / "router_registry.py"
    ).read_text()
    pattern = re.compile(
        r"from prsm\.interface\.api(?:\.routers)?\.(\w+)"
        r"\s+import\s+router\s+as\s+(\w+)"
    )
    out = {}
    for m in pattern.finditer(src):
        module, alias = m.group(1), m.group(2)
        out[module] = alias
    return out


def _router_prefix(module: str):
    """Returns the prefix= argument from APIRouter() in the
    given prsm.interface.api.<module>.py file. Returns None if
    no prefix declared (those routers register paths directly)."""
    # Two possible paths.
    candidates = [
        REPO_ROOT / "prsm" / "interface" / "api" / f"{module}.py",
        REPO_ROOT / "prsm" / "interface" / "api" / "routers"
        / f"{module}.py",
    ]
    for p in candidates:
        if p.exists():
            src = p.read_text()
            m = re.search(r'prefix\s*=\s*["\']([^"\']+)["\']', src)
            return m.group(1) if m else None
    return None


def _node_api_src():
    return (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()


def test_every_registry_router_is_mounted_or_deferred():
    """The F28-prevention invariant."""
    registry = _registry_router_imports()
    node_api = _node_api_src()

    findings = []
    for module, _alias in registry.items():
        if module in DEFERRED_INERT_ROUTERS:
            continue
        # Mount check #1: include_router(<wallet_router>) style
        # for the SAME module name. Loose match because the
        # alias used in node/api.py may differ from the
        # router_registry alias.
        explicit_import = (
            f"from prsm.interface.api.{module} import" in node_api
            or f"from prsm.interface.api.routers.{module} import"
            in node_api
        )
        prefix = _router_prefix(module)
        prefix_present_inline = (
            prefix is not None
            and f'"{prefix}' in node_api
        )
        if not (explicit_import or prefix_present_inline):
            findings.append(
                f"{module} (prefix={prefix!r}) — NOT mounted in "
                f"create_api_app AND not in "
                f"DEFERRED_INERT_ROUTERS allow-list. Either "
                f"mount it next to onboarding/wallet, or add "
                f"it to the allow-list with a reason."
            )

    assert not findings, (
        "Sprint 830 — F28-class regression: "
        + "\n  ".join([""] + findings)
    )


def test_deferred_list_entries_each_have_reason():
    """The allow-list is NOT a 'shut up the test' knob. Each
    entry must have a non-empty reason string explaining why
    the router is unmounted."""
    for module, reason in DEFERRED_INERT_ROUTERS.items():
        assert reason, (
            f"Deferred router {module} has empty reason. "
            f"Document why it's inert or remove from the list."
        )
        assert len(reason.strip()) > 30, (
            f"Deferred router {module} reason too short to be "
            f"meaningful: {reason!r}"
        )


def test_wallet_api_is_mounted_not_deferred():
    """Regression guard for sprint 829: wallet_api MUST be in
    the mounted set, NOT in the deferred allow-list."""
    assert "wallet_api" not in DEFERRED_INERT_ROUTERS, (
        "Sprint 829 fixed wallet_api by mounting it. Don't "
        "regress by moving it back to deferred."
    )
    # Mount is verified by sprint 829's own tests; this is just
    # an explicit guard against the allow-list move.


def test_onboarding_router_is_mounted_not_deferred():
    """Regression guard for sprint 547's F6 fix."""
    assert "onboarding_router" not in DEFERRED_INERT_ROUTERS
