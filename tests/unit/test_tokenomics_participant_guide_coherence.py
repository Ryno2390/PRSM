"""CI assertion for the R-2026-05-08-2 Tokenomics ↔ PARTICIPANT_GUIDE
coherence rule.

This test encodes the council-ratified regression-discipline rule
from `docs/governance/PRSM-CR-2026-05-08.md` §3 RESOLVED 6 as a
narrow anti-regression assertion. The rule:

  > When `PRSM_Tokenomics.md` is updated, ALSO run a freshness
  > check on `docs/PARTICIPANT_GUIDE.md` to catch contradicting
  > wording. The participant-facing doc surfaces tokenomic claims
  > for end-users; investors compare narratives across both
  > surfaces.

Audit-prep §7.27 originally flagged CI enforcement of R-2 as
"non-trivial without false-positives" because full text-coherence
checking is hard. This test ships the **tier-1 narrow anti-
regression version** that sidesteps that complexity:

  - Negative assertions: pin specific prior-contradicting phrases
    that MUST NOT reappear (the exact phrases yesterday's
    PARTICIPANT_GUIDE refresh removed).
  - Positive assertions: pin specific load-bearing canonical
    phrases that MUST be present (matching Tokenomics §3.5 /
    §3.7 / §4.10 / §5.5 framing).

This narrow approach catches the SPECIFIC regression class that
yesterday's refresh closed, without attempting full text-coherence
which would produce false-positives.

When PRSM_Tokenomics.md (Obsidian vault, NOT in repo) is updated
in a way that legitimately changes the canonical phrases below,
the supersession protocol is:
  (a) update PARTICIPANT_GUIDE.md to match the new Tokenomics
      framing in the SAME change-set as the Tokenomics edit,
  (b) update THIS test file's pinned phrases to reflect the new
      canonical wording in the same change-set,
  (c) commit message references the Tokenomics-edit + this test
      update + the rationale for the canonical-phrase change.
Without that explicit coordinated update, this test catches the
drift.

Note: PRSM_Tokenomics.md lives in the founder's iCloud Obsidian
vault (per memory entry: portfolio docs aimed at capital raise
belong on a separate private Prismatica surface; engineering
docs DO belong in repo). This test cannot reach Tokenomics
directly. It enforces the IN-REPO half of the coherence rule —
that the participant-facing doc continues to render the canonical
tokenomic claims correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PARTICIPANT_GUIDE_PATH = (
    Path(__file__).parent.parent.parent / "docs" / "PARTICIPANT_GUIDE.md"
)


def _load_doc() -> str:
    """Load PARTICIPANT_GUIDE.md as a single lowercased string for
    case-insensitive matching."""
    assert PARTICIPANT_GUIDE_PATH.exists(), (
        f"PARTICIPANT_GUIDE.md not found at {PARTICIPANT_GUIDE_PATH} "
        f"— R-2026-05-08-2 enforcement requires the doc to exist for "
        f"coherence checking."
    )
    return PARTICIPANT_GUIDE_PATH.read_text().lower()


# ──────────────────────────────────────────────────────────────────────
# Negative assertions: prior-contradicting phrases MUST NOT appear
# ──────────────────────────────────────────────────────────────────────


# These are the exact phrases yesterday's PARTICIPANT_GUIDE refresh
# REMOVED because they contradicted the updated Tokenomics §3.5
# bootstrap-vs-ongoing-operations distinction.
FORBIDDEN_PHRASES = [
    # Direct contradiction of Tokenomics §3.5 (post-2026-05-08 update):
    "foundation does not seed amm pools",
    # Variant phrasings that would constitute the same contradiction:
    "the foundation does not seed",
    # Outdated testnet framing (would contradict mainnet reality):
    "currently in sepolia testnet bake-in",
    "mainnet launch imminent",
    "mainnet-imminent",
    # Outdated phase status framing:
    "phase 4 (q4 2026)",
    "phase 3 mcp server target q3 2026",
    "when phase 3's mcp server ships",
]


class TestNegativeAssertions:
    """Prior-contradicting phrases MUST NOT reappear in
    PARTICIPANT_GUIDE.md."""

    @pytest.mark.parametrize("forbidden", FORBIDDEN_PHRASES)
    def test_forbidden_phrase_absent(self, forbidden: str):
        doc = _load_doc()
        assert forbidden not in doc, (
            f"R-2026-05-08-2 violation: PARTICIPANT_GUIDE.md contains "
            f"prior-contradiction phrase {forbidden!r}. This phrase "
            f"was REMOVED in yesterday's PARTICIPANT_GUIDE refresh "
            f"(commit 9c26f07d) because it directly contradicts the "
            f"updated PRSM_Tokenomics.md §3.5 bootstrap-vs-ongoing-"
            f"market-making distinction. Reintroducing it without "
            f"superseding R-2026-05-08-2 = regression-discipline "
            f"breach per PRSM-CR-2026-05-08 §3 RESOLVED 6.\n\n"
            f"Supersession protocol: update Tokenomics + this test's "
            f"FORBIDDEN_PHRASES list + PARTICIPANT_GUIDE.md in the "
            f"SAME change-set; commit message references all three."
        )


# ──────────────────────────────────────────────────────────────────────
# Positive assertions: canonical Tokenomics framings MUST be present
# ──────────────────────────────────────────────────────────────────────


# Each tuple: (Tokenomics-section, list of phrases that MUST appear
# at least once in PARTICIPANT_GUIDE — case-insensitive substring).
# At least ONE phrase from each list must be present (some are
# alternative wordings to allow legitimate doc evolution).
CANONICAL_FRAMINGS = [
    (
        "§3.5 bootstrap-vs-ongoing distinction",
        [
            "one-time bootstrap",
            "discrete bootstrap event",
            "distinct from ongoing market-making",
        ],
    ),
    (
        "§3.7 Aerodrome USDC-FTNS pool architecture",
        [
            "aerodrome usdc-ftns",
            "aerodrome pool",
        ],
    ),
    (
        "§3.5 + §3.7 Helium/io.net DePIN precedent",
        [
            "helium / io.net",
            "helium",
            "io.net",
        ],
    ),
    (
        "§4.10 USD-denominated services pricing",
        [
            "usd-denominated",
            "priced in usd",
            "denominated in usd",
        ],
    ),
    (
        "§5.5 Coinbase as regulated gateway",
        [
            "coinbase performs kyc",
            "coinbase, not the foundation, performs kyc",
            "regulated gateway",
        ],
    ),
    (
        "§5.5 no-PII-transit guarantee",
        [
            "never transmits banking pii",
            "no banking pii",
            "does not store routing numbers",
        ],
    ),
    (
        "Mainnet status (post-2026-05-04 / 2026-05-07)",
        [
            "live on base mainnet",
            "mainnet since 2026-05-04",
        ],
    ),
]


class TestPositiveAssertions:
    """Canonical Tokenomics framings MUST be present (at least one
    alternative wording from each group)."""

    @pytest.mark.parametrize("framing,phrases", CANONICAL_FRAMINGS)
    def test_canonical_framing_present(
        self, framing: str, phrases: list,
    ):
        doc = _load_doc()
        matches = [p for p in phrases if p in doc]
        assert matches, (
            f"R-2026-05-08-2 violation: PARTICIPANT_GUIDE.md missing "
            f"canonical framing for '{framing}'. None of the expected "
            f"phrases found: {phrases!r}.\n\n"
            f"This framing is load-bearing for the participant-facing "
            f"render of PRSM_Tokenomics.md. Removing it without "
            f"updating Tokenomics in coordination = silent contradiction "
            f"between investor-facing surfaces (Tokenomics + "
            f"PARTICIPANT_GUIDE).\n\n"
            f"Supersession protocol: if Tokenomics intentionally "
            f"changed this framing, update this test's CANONICAL_FRAMINGS "
            f"list + PARTICIPANT_GUIDE.md + Tokenomics in the same "
            f"change-set; commit message references all three."
        )


# ──────────────────────────────────────────────────────────────────────
# Cross-reference assertions: PARTICIPANT_GUIDE must reference the
# Tokenomics doc explicitly so readers can trace claims back
# ──────────────────────────────────────────────────────────────────────


# At least N references to PRSM_Tokenomics.md; if the doc stops
# citing back to Tokenomics, claims become orphan. Markdown allows
# multiple shapes (`PRSM_Tokenomics.md` §3 / [PRSM_Tokenomics](...)
# / etc.) so we count filename references rather than specific
# section-citation shapes.


class TestCrossReference:
    """PARTICIPANT_GUIDE.md must explicitly cite PRSM_Tokenomics.md
    so readers can trace claims back. Without these references,
    claims become orphan (and silent contradictions become harder
    to catch in future audits)."""

    def test_at_least_three_tokenomics_references(self):
        # Count substring occurrences of the filename (lowercased).
        # Today's doc has 6; we require ≥ 3 to allow some
        # consolidation / restructuring without breaking the test,
        # but not so few that the cross-reference framing can be
        # silently stripped.
        doc = _load_doc()
        count = doc.count("prsm_tokenomics.md")
        assert count >= 3, (
            f"R-2026-05-08-2: PARTICIPANT_GUIDE.md must reference "
            f"PRSM_Tokenomics.md at least 3 times so readers can "
            f"trace claims back to the canonical tokenomics doc. "
            f"Currently {count} reference(s) found. Removing cross-"
            f"references makes future coherence audits harder + "
            f"orphans the participant-facing claims from their "
            f"Tokenomics source."
        )


# ──────────────────────────────────────────────────────────────────────
# Composer-only invariant cross-check (R-2026-05-08-1 surface area)
# ──────────────────────────────────────────────────────────────────────


# R-2 is the freshness rule; R-1 is the composer-only invariant.
# But because the participant doc is where end-users see the
# composer-only framing, this test also asserts R-1's user-facing
# framing renders correctly in the participant doc. If R-1 framing
# disappears from PARTICIPANT_GUIDE, end users wouldn't know the
# tool is composer-only — which would be a different but
# similarly-shaped regression class.


class TestComposerOnlyFraming:
    """The PARTICIPANT_GUIDE 'Cashing Out to Bank' section must
    render the R-2026-05-08-1 composer-only framing for end users.
    This is the user-visible signal that today's offramp tool does
    NOT initiate any on-chain or fiat-side action."""

    def test_pending_commission_rendered_for_end_users(self):
        doc = _load_doc()
        assert "pending_commission" in doc.lower() or "pending commission" in doc.lower(), (
            "PARTICIPANT_GUIDE.md must render 'PENDING_COMMISSION' "
            "framing in the Cashing Out to Bank section. End users "
            "reading the doc must see the composer-only signal that "
            "matches what the MCP tool returns. Removing this would "
            "create a UX-vs-tool-output drift (different surfaces "
            "telling different stories about the same primitive)."
        )

    def test_does_not_initiate_phrase_present(self):
        # Defensive: the explicit "does NOT initiate" framing is the
        # load-bearing reassurance. Removing it would make the
        # composer-only constraint less obvious to end users.
        doc = _load_doc()
        does_not_phrases = [
            "does not initiate",
            "do not initiate",
            "does not trigger",
            "do not trigger",
        ]
        present = any(p in doc for p in does_not_phrases)
        assert present, (
            f"PARTICIPANT_GUIDE.md must explicitly state that today's "
            f"`coinbase_offramp_initiate` v1 'does NOT initiate' (or "
            f"equivalent) any on-chain/fiat action. None of "
            f"{does_not_phrases!r} found. The composer-only constraint "
            f"is the most material UX claim of the cash-out section; "
            f"removing this framing weakens R-2026-05-08-1's user-"
            f"visible signal."
        )
