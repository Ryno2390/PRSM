"""Sprint 494 — cross-feature integration fixes (F34/F35).

Coverage matrix priority #5 (XF column): each feature
tested in isolation. Sprint 494's chain test walked:
  upload (§4) → retrieve (§4) → creator-reputation
  auto-record (§14) → reputation surface (§14)

Pre-sprint-494 the chain broke at TWO points:

F34 — CreatorReputationTracker + CreatorStakeClient
  were both init'd inside `_build_query_orchestrator_or_none`
  (same code block as the sprint-488/F26 fingerprint
  registry). On default daemons (no PRSM_QUERY_ORCHESTRATOR_
  ENABLED), the function early-returns and these stay
  None → /marketplace/creator-reputation/{id} returns 503
  on every default daemon. Sprint 488 moved only the
  fingerprint registry out; F34 moves the other two
  siblings.

F35 — content_index single-node lookup gap. The retrieve
  handler called `content_index.lookup(cid)` to get
  `creator_eth_address` for the auto-record-creator-access
  hook. content_index is populated via
  GOSSIP_CONTENT_ADVERTISE which delivers to local
  subscribers ONLY when `sent == 0`; on real multi-peer
  setups the local subscriber is skipped and the local
  node's content_index never sees its own uploads.
  Pre-fix: on single-node + bootstrap-only daemons,
  every local retrieve silently skipped the auto-record
  because `creator_eth_address` came back None.
  Fix: fall back to `uploader.uploaded_content[cid].
  creator_eth_address` (definitely populated at upload
  time).

Live-verified post-sprint-494: upload with
creator_eth_address → retrieve → reputation
`total_accesses=1, distinct_purchasers=1, known=true`.
6 retrieves → `total_accesses=6, repeat_purchaser_count=1`.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_f34_creator_reputation_tracker_wired_unconditionally():
    """Source pin: Node.__init__ must construct
    CreatorReputationTracker OUTSIDE
    _build_query_orchestrator_or_none. Pre-fix it lived
    inside the QO-gated block and was None on every
    non-QO daemon."""
    node_src = (
        REPO_ROOT / "prsm" / "node" / "node.py"
    ).read_text()
    # The marker comment from sprint 494 must remain.
    normalized = " ".join(
        line.lstrip("#").strip() for line in node_src.splitlines()
    )
    assert (
        "Sprint 494 (F34 fix) — CreatorReputationTracker"
    ) in normalized, (
        "F34 marker missing — CreatorReputationTracker may "
        "have been moved back inside the QO-gated block"
    )
    # The first instantiation must be BEFORE
    # `def _build_query_orchestrator_or_none`.
    first_tracker_idx = node_src.find(
        "self._creator_reputation_tracker = ("
    )
    qo_def_idx = node_src.find(
        "def _build_query_orchestrator_or_none"
    )
    assert 0 < first_tracker_idx < qo_def_idx, (
        "CreatorReputationTracker init must appear before "
        "_build_query_orchestrator_or_none method"
    )


def test_f34_creator_stake_client_wired_unconditionally():
    """Source pin: same sibling-fix for CreatorStakeClient."""
    node_src = (
        REPO_ROOT / "prsm" / "node" / "node.py"
    ).read_text()
    first_stake_idx = node_src.find(
        "self._creator_stake_client = ("
    )
    qo_def_idx = node_src.find(
        "def _build_query_orchestrator_or_none"
    )
    assert 0 < first_stake_idx < qo_def_idx, (
        "CreatorStakeClient init must appear before "
        "_build_query_orchestrator_or_none method"
    )


def test_f35_retrieve_falls_back_to_uploaded_content():
    """Source pin: the /content/retrieve handler must
    fall back to `uploader.uploaded_content[cid].
    creator_eth_address` when content_index returns None
    or has no creator_eth_address. Without this, every
    single-node retrieve silently skipped the §14
    creator-access auto-record."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    # The sprint 494 F35 marker must remain.
    assert "Sprint 494 (F35 fix)" in api_src
    # The fallback must reference uploaded_content + use cid.
    idx = api_src.find("Sprint 494 (F35 fix)")
    region = api_src[idx:idx + 2500]
    assert "uploaded_content" in region
    assert "creator_eth_address" in region


def test_f35_fallback_runs_only_when_index_lacks_creator():
    """The fallback must be CONDITIONAL on creator_eth_address
    being None — otherwise it would clobber a valid
    content_index record from a remote peer's gossip
    advertisement."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    idx = api_src.find("Sprint 494 (F35 fix)")
    region = api_src[idx:idx + 2500]
    # Must be guarded by an `if creator_eth_address is None`
    assert "if creator_eth_address is None:" in region, (
        "F35 fallback must only fire when content_index "
        "lookup didn't provide creator_eth_address — "
        "otherwise multi-node gossip records get clobbered "
        "by local-only fallback"
    )


def test_f35_fallback_preserves_existing_content_hash_filename():
    """Defensive: the fallback shouldn't overwrite
    content_hash / filename if they were ALREADY populated
    by the content_index lookup. Use `if not <field>:`
    guards."""
    api_src = (
        REPO_ROOT / "prsm" / "node" / "api.py"
    ).read_text()
    idx = api_src.find("Sprint 494 (F35 fix)")
    region = api_src[idx:idx + 2500]
    assert "if not content_hash:" in region
    assert "if not filename:" in region
