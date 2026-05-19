"""Sprint 558 — production wiring for ParallaxScheduledExecutor.

First sprint in the arc taking /compute/inference from "mock only"
(sprint 438) to "real ParallaxScheduledExecutor". Sprint 546 wired
sprint-419's ActivationDPAware decorator into the chain-RPC
factory, but no production caller invokes that factory — /compute/
inference still falls through to MockInferenceExecutor or 503.

Sprint 558 closes the loop at the OPT-IN PATH:

  PRSM_INFERENCE_EXECUTOR=parallax  → node.py calls
      build_parallax_executor_or_none(node)

The builder reads three component-kind env vars:

  PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON file, top-level dict
      mapping model_id → ModelInfo kwargs.
  PRSM_PARALLAX_TRUST_STACK_KIND   — currently only "mock"; sprint
      560 adds "production" backed by real anchor + stake-lookup
      adapters.
  PRSM_PARALLAX_GPU_POOL_KIND      — currently only "static-empty"
      (no peers); sprint 561 adds "dht" backed by bootstrap-derived
      peer list.

Every missing/invalid piece logs a structured warning naming the
env var + returns None. The daemon stays bootable; /compute/
inference returns the existing 503 from sprint 438 with the
operator-readable detail. No new failure modes for existing
operators — this is a purely additive opt-in surface.

Future sprints fill in the "production" / "dht" kinds + the
chain_executor wiring (sprint 546's factory). Sprint 558 just
locks the contract so the component knobs are operationally
discoverable.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


_KNOWN_TRUST_STACK_KINDS = ("mock", "production")
_KNOWN_GPU_POOL_KINDS = ("static-empty",)

# Sprint 559 — catalog schema versioning. v1 shape:
#   {
#     "schema_version": "v1",
#     "models": { "<model_id>": { ...ModelInfo kwargs... }, ... }
#   }
_CATALOG_SCHEMA_VERSIONS = ("v1",)

# Required ModelInfo fields per entry. ModelInfo's __init__ silently
# accepts ANY kwargs (no positional validation), so operator typos
# like "model_namee" used to construct a degenerate ModelInfo that
# failed at scheduling time with cryptic errors. Sprint 559 catches
# the typo at boot.
_REQUIRED_MODEL_INFO_FIELDS = (
    "model_name",
    "num_layers",
    "hidden_dim",
    "num_attention_heads",
    "num_kv_heads",
    "vocab_size",
    "head_size",
    "intermediate_dim",
)


def _build_mock_trust_stack():
    """Permissive trust stack — passes every GPU through unchanged.
    Clearly labeled in env-var ("mock"); production operators MUST
    pass PRSM_PARALLAX_TRUST_STACK_KIND=production (sprint 560)."""
    from prsm.compute.parallax_scheduling.trust_adapter import (
        AnchorVerifyAdapter,
        ConsensusMismatchHook,
        StakeWeightedTrustAdapter,
        TierGateAdapter,
        TrustStack,
    )
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )

    class _AcceptAllAnchor:
        """Test/dev anchor — claims every node_id is registered."""
        def lookup(self, node_id: str):
            return f"{node_id}.local"

    class _ZeroStakeLookup:
        def get_stake(self, node_id: str) -> int:
            return 0

    # Sprint 562: mock kind uses the same _LoggingChallengeSubmitter
    # as production — fixes the structurally-broken _RecordingSubmitter
    # (had async def submit() but the hook invokes the submitter as
    # a sync Callable). Sample-rate=0.0 in both kinds today so the
    # hook never actually fires; the fix is defensive.
    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=_AcceptAllAnchor()),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=InMemoryProfileSource(snapshots={}),
            stake_lookup=_ZeroStakeLookup(),
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_build_consensus_submitter(),
            sample_rate=0.0,
        ),
    )


class _LoggingChallengeSubmitter:
    """Sprint 562 — interim consensus_hook submitter.

    The trust-stack's ``ConsensusMismatchHook`` calls this when a
    sampled redundant execution produces a different output than the
    primary chain. The legacy ``_NoOpSubmitter`` silently dropped
    these — operators had zero visibility into compute-side fraud
    attempts. This submitter emits a structured WARNING log naming
    the request_id, both chain stage lists, and both output hashes
    so the audit trail is complete even before the real
    ``ConsensusChallengeSubmitter`` (Phase 7.1x on-chain dispatch)
    is wired.

    The full on-chain submitter requires a translation layer from
    ``ChallengeRecord`` to the ``challengeReceipt(batchId, leaf,
    merkleProof, reason, auxData)`` ABI shape — its own multi-piece
    concern. Logging closes the silent-drop bug TODAY without
    blocking on that translation layer.

    Honors the ``ChallengeSubmitter`` Callable contract: synchronous,
    returns None, must not raise. Malformed records (defensive
    against future schema changes) still log + return None instead
    of raising.
    """

    def __call__(self, record) -> None:
        try:
            request_id = getattr(record, "request_id", "<missing>")
            primary = getattr(
                record, "primary_chain_stages", "<missing>",
            )
            secondary = getattr(
                record, "secondary_chain_stages", "<missing>",
            )
            primary_hash = getattr(
                record, "primary_output_hash", "<missing>",
            )
            secondary_hash = getattr(
                record, "secondary_output_hash", "<missing>",
            )
            logger.warning(
                "Consensus mismatch detected (sprint 562 logging "
                "submitter): request_id=%s primary_chain=%s "
                "secondary_chain=%s primary_output_hash=%s "
                "secondary_output_hash=%s. On-chain challenge "
                "dispatch deferred — Phase 7.1x translation layer "
                "not yet wired.",
                request_id, primary, secondary,
                primary_hash, secondary_hash,
            )
        except Exception as exc:  # noqa: BLE001
            # Defensive — submitter must not raise per the
            # ChallengeSubmitter contract. Log + swallow.
            logger.warning(
                "Consensus mismatch submitter raised on log "
                "formatting (non-fatal): %s", exc,
            )


class AnchorMediatedStakeLookup:
    """Sprint 561 — production stake_lookup. Maps trust-stack node_id
    (32-char hex publisher ID) → ETH address via the anchor's
    ``lookup`` method, then queries on-chain stake via the
    StakeManagerClient.

    Conservative fail-soft semantics:
      - anchor.lookup returns None  → 0 stake (unknown node)
      - stake_of raises (RPC error) → 0 stake (logged at DEBUG to
                                      avoid log spam on transient
                                      network errors)

    Returning 0 under uncertainty matches the
    ``StakeWeightedTrustAdapter`` design: zero stake = "effectively
    excluded from routing" — the conservative posture under any
    resolution failure.
    """

    def __init__(self, *, anchor, stake_client):
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "AnchorMediatedStakeLookup requires an anchor with "
                ".lookup(node_id) → Optional[str]"
            )
        if stake_client is None or not hasattr(
            stake_client, "stake_of",
        ):
            raise RuntimeError(
                "AnchorMediatedStakeLookup requires a stake_client "
                "with .stake_of(provider) → StakeRecord"
            )
        self._anchor = anchor
        self._stake = stake_client

    def get_stake(self, node_id: str) -> int:
        try:
            eth = self._anchor.lookup(node_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AnchorMediatedStakeLookup: anchor.lookup raised "
                "for node_id=%s: %s", node_id, exc,
            )
            return 0
        if not eth:
            return 0
        try:
            record = self._stake.stake_of(eth)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "AnchorMediatedStakeLookup: stake_of raised for "
                "node_id=%s eth=%s: %s", node_id, eth, exc,
            )
            return 0
        return int(getattr(record, "amount_wei", 0))


def _build_production_trust_stack_or_none():
    """Sprint 560 + 561 — `production` trust-stack kind.

    Wires REAL components from operator-supplied env vars:

      - anchor          : PublisherKeyAnchorClient @
                          PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS
                          (required — no fallback).
      - stake_lookup    : (sprint 561) AnchorMediatedStakeLookup
                          via StakeManagerClient @
                          PRSM_STAKE_BOND_ADDRESS. Falls back to
                          ZeroStakeLookup PLACEHOLDER + warning
                          when the env var is unset OR construction
                          fails (operator on partial-production).

    Still PLACEHOLDER (pending sprint 562):
      - profile_source  → empty InMemoryProfileSource.
      - consensus_hook  → no-op submitter.

    Returns None when the anchor address isn't set — there's no
    sensible fallback for the anchor under `production` kind.

    Caller logs the per-component REAL/PLACEHOLDER enumeration at
    INFO level so operators see what's actually verified.
    """
    from prsm.compute.parallax_scheduling.trust_adapter import (
        AnchorVerifyAdapter,
        ConsensusMismatchHook,
        StakeWeightedTrustAdapter,
        TierGateAdapter,
        TrustStack,
    )
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )

    # Sprint 580 — anchor construction extracted to module helper
    # so chain_executor Phase 2 can share. None handling stays here
    # since "production trust-stack requires anchor" is a trust-stack
    # invariant, not an anchor-helper concern.
    anchor_addr = os.environ.get(
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "",
    ).strip()
    if not anchor_addr:
        logger.warning(
            "Sprint 560 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_TRUST_STACK_KIND=production requires "
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS (the Phase-3.x.3 "
            "publisher-key anchor contract address). Set it to "
            "the deployed anchor address on your chain, or use "
            "PRSM_PARALLAX_TRUST_STACK_KIND=mock for dev."
        )
        return None
    anchor = _build_anchor_or_none()
    if anchor is None:
        # _build_anchor_or_none already logged the underlying
        # construction failure; the trust-stack caller surfaces
        # the kind-specific guidance.
        return None

    # Sprint 561 — production stake_lookup. PRSM_STAKE_BOND_ADDRESS
    # set → build real StakeManagerClient + AnchorMediatedStakeLookup.
    # Unset OR construction fails → fall back to ZeroStakeLookup
    # placeholder with a structured warning (operator on partial-
    # production keeps a working daemon).
    class _ZeroStakeLookup:
        def get_stake(self, node_id: str) -> int:
            return 0

    # Sprint 580 — rpc_url resolved here (was earlier in pre-580
    # inline construction; sprint-561 StakeManagerClient still needs it).
    rpc_url = os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    stake_addr = os.environ.get(
        "PRSM_STAKE_BOND_ADDRESS", "",
    ).strip()
    if stake_addr:
        try:
            from prsm.economy.web3.stake_manager import (
                StakeManagerClient,
            )
            stake_client = StakeManagerClient(
                contract_address=stake_addr,
                rpc_url=rpc_url,
            )
            stake_lookup = AnchorMediatedStakeLookup(
                anchor=anchor, stake_client=stake_client,
            )
            stake_lookup_status = "REAL"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sprint 561 ParallaxScheduledExecutor wiring: "
                "StakeManagerClient construction failed against %s "
                "— %s. Falling back to ZeroStakeLookup placeholder.",
                stake_addr, exc,
            )
            stake_lookup = _ZeroStakeLookup()
            stake_lookup_status = (
                "PLACEHOLDER (StakeManagerClient construction failed)"
            )
    else:
        logger.warning(
            "Sprint 561 ParallaxScheduledExecutor wiring: "
            "PRSM_STAKE_BOND_ADDRESS not set; stake_lookup falls "
            "back to ZeroStakeLookup placeholder. Set it to the "
            "deployed StakeBond contract address (Base mainnet "
            "default: see networks.py:stake_bond) to wire real "
            "on-chain stake verification."
        )
        stake_lookup = _ZeroStakeLookup()
        stake_lookup_status = "PLACEHOLDER (zero stake)"

    # Sprint 562 — consensus_hook submitter upgraded from
    # structurally-broken _NoOpSubmitter (had .submit() method but
    # hook invokes the submitter as a Callable) to a
    # _LoggingChallengeSubmitter that emits WARNING per mismatch.
    # On-chain Phase 7.1x ConsensusChallengeSubmitter wiring needs
    # a translation layer (ChallengeRecord → on-chain ABI shape) —
    # deferred until that translation lands.
    logger.info(
        "Sprint 560+561+562 ParallaxScheduledExecutor wiring: "
        "trust_stack=production. anchor=REAL "
        "(PublisherKeyAnchorClient @ %s); stake_lookup=%s; "
        "profile_source=PLACEHOLDER (empty InMemoryProfileSource "
        "— ProfileDHT requires multi-host send_message + peers, "
        "out of scope for single-node); consensus_hook=LOGGING "
        "(WARNING per ChallengeRecord — on-chain dispatch via "
        "Phase 7.1x ConsensusChallengeSubmitter deferred pending "
        "translation layer). Operators must not treat partial-"
        "production as fully production-verified until all 4 "
        "components are REAL.",
        anchor_addr, stake_lookup_status,
    )
    # Sprint 576 — inner ProfileSource selection is env-driven via
    # PRSM_PARALLAX_PROFILE_SOURCE_KIND (default in_memory preserves
    # behavior; "dht" is the Phase 2 hook).
    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=anchor),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=_build_inner_profile_source(),
            stake_lookup=stake_lookup,
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_build_consensus_submitter(),
            sample_rate=0.0,
        ),
    )


def _build_anchor_or_none():
    """Sprint 580 — env-driven PublisherKeyAnchorClient construction.

    Single source of truth for ``PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS``
    interpretation. Used by:
      - ``_build_production_trust_stack_or_none`` (existing caller)
      - ``_build_chain_executor`` (sprint-581+ Phase 2 — needs the
        anchor for ``make_rpc_chain_executor``)

    Semantics:
      - env unset / empty → return None (silent; the trust-stack
        caller is the one that decides whether None is OK or
        production-blocking)
      - construction succeeds → return PublisherKeyAnchorClient
      - construction fails → log WARNING + return None

    RPC URL falls back to Base mainnet (PRSM_BASE_RPC_URL override
    matches the pre-580 inline behavior).
    """
    anchor_addr = os.environ.get(
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "",
    ).strip()
    if not anchor_addr:
        return None
    rpc_url = os.environ.get(
        "PRSM_BASE_RPC_URL", "https://mainnet.base.org",
    )
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        return PublisherKeyAnchorClient(
            contract_address=anchor_addr,
            rpc_url=rpc_url,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 580 _build_anchor_or_none: "
            "PublisherKeyAnchorClient construction failed against "
            "%s — %s. Anchor unavailable; returning None.",
            anchor_addr, exc,
        )
        return None


def _build_chain_executor(node: Any) -> Any:
    """Sprint 578 — env-driven chain_executor construction.

    Read PRSM_PARALLAX_CHAIN_EXECUTOR_KIND:
      - unset / "stub" → _StubChainExecutor (current default;
        raises on execute_chain so callers must hit a non-empty
        pool first — sprint 558 invariant).
      - "rpc" → Phase 2 wires real via make_rpc_chain_executor
        with node.identity/transport/anchor. Phase 1 logs WARNING
        + falls back to stub so daemon stays up.
      - anything else → WARNING + stub fallback.

    Mirrors sprints-576/577 plumbing pattern. Phase 2 (real RPC
    chain executor) becomes a non-churn additive change here.
    """
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND", "",
    ).strip().lower()
    kind = kind_raw or "stub"
    if kind == "stub":
        return _StubChainExecutor()
    if kind == "rpc":
        logger.warning(
            "Sprint 578 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc set but Phase 2 "
            "(real make_rpc_chain_executor wiring with node "
            "settler_identity + transport.send_message + anchor) "
            "has not landed yet — falling back to _StubChainExecutor. "
            "Track sprint 579+ for the actual RPC chain wiring."
        )
        return _StubChainExecutor()
    logger.warning(
        "Sprint 578 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=%r is unknown; falling "
        "back to _StubChainExecutor. Valid values: stub, rpc "
        "(Phase 2 pending).",
        kind_raw,
    )
    return _StubChainExecutor()


def _build_consensus_submitter():
    """Sprint 577 — env-driven consensus_hook submitter construction.

    Read PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND:
      - unset / "logging" → _LoggingChallengeSubmitter (current
        default; WARNING per ChallengeRecord; no on-chain dispatch).
      - "onchain" → Phase 2 returns real OnChainChallengeSubmitter
        that maps ChallengeRecord → Phase 7.1x
        ConsensusChallengeSubmitter ABI + dispatches. Phase 1 logs
        WARNING + falls back to logging so daemon stays up.
      - anything else → WARNING + logging fallback.

    Mirrors sprint-576's profile_source helper pattern. Phase 2
    (real on-chain) becomes a non-churn additive change here.
    """
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND", "",
    ).strip().lower()
    kind = kind_raw or "logging"
    if kind == "logging":
        return _LoggingChallengeSubmitter()
    if kind == "onchain":
        logger.warning(
            "Sprint 577 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND=onchain set but "
            "Phase 2 (real OnChainChallengeSubmitter w/ "
            "ChallengeRecord → Phase 7.1x ABI translation layer) "
            "has not landed yet — falling back to "
            "_LoggingChallengeSubmitter. Track sprint 578+ for the "
            "actual on-chain dispatch wiring."
        )
        return _LoggingChallengeSubmitter()
    logger.warning(
        "Sprint 577 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND=%r is unknown; "
        "falling back to _LoggingChallengeSubmitter. Valid values: "
        "logging, onchain (Phase 2 pending).",
        kind_raw,
    )
    return _LoggingChallengeSubmitter()


def _build_inner_profile_source():
    """Sprint 576 — env-driven inner ProfileSource construction.

    Read PRSM_PARALLAX_PROFILE_SOURCE_KIND:
      - unset / "in_memory" → InMemoryProfileSource(snapshots={})
        (current default behavior; existing operators unaffected)
      - "dht"               → Phase 2 will return ProfileDHT(...);
        Phase 1 logs a structured warning + falls back to
        in_memory so the daemon stays up.
      - anything else       → warn + in_memory fallback.

    The trust-stack constructor calls this helper instead of
    hardcoding the InMemoryProfileSource — Phase 2 (real DHT
    wiring) becomes a non-churn additive change here.
    """
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    kind_raw = os.environ.get(
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND", "",
    ).strip().lower()
    kind = kind_raw or "in_memory"
    if kind == "in_memory":
        return InMemoryProfileSource(snapshots={})
    if kind == "dht":
        logger.warning(
            "Sprint 576 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_PROFILE_SOURCE_KIND=dht set but Phase 2 "
            "(real ProfileDHT integration) has not landed yet — "
            "falling back to InMemoryProfileSource(snapshots={}). "
            "Track sprint 577+ for the actual DHT-backed source."
        )
        return InMemoryProfileSource(snapshots={})
    logger.warning(
        "Sprint 576 ParallaxScheduledExecutor wiring: "
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND=%r is unknown; falling "
        "back to InMemoryProfileSource. Valid values: in_memory, "
        "dht (Phase 2 pending).",
        kind_raw,
    )
    return InMemoryProfileSource(snapshots={})


def _build_static_empty_pool_provider():
    """Pool provider that returns an empty list. /compute/inference
    requests will fail Phase-1 allocation with a clear error; this
    is the explicit "no peers wired" path. Sprint 561 replaces with
    a DHT-backed provider."""
    def _provider():
        return []
    return _provider


class _StubChainExecutor:
    """Sprint 558 placeholder. The real chain executor comes from
    sprint 546's make_rpc_chain_executor factory in a future sprint.
    Today, no /compute/inference request gets this far because the
    static-empty pool fails Phase-1 first."""

    def execute_chain(self, *, request, chain, **kwargs):
        raise RuntimeError(
            "Sprint 558 stub chain executor — not callable until "
            "sprint 562+ wires make_rpc_chain_executor."
        )


def build_parallax_executor_or_none(node: Any) -> Optional[Any]:
    """Construct a ParallaxScheduledExecutor if every required
    operator-supplied component is present + valid.

    Returns ``None`` and logs a structured warning naming the
    missing piece otherwise. Daemon caller treats None the same
    as "no inference executor wired" (the existing default).

    Component contract (env vars):
      PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON path
      PRSM_PARALLAX_TRUST_STACK_KIND   — "mock"
      PRSM_PARALLAX_GPU_POOL_KIND      — "static-empty"

    Each kind is a tagged-union opt-in. Sprint 558 ships the dev/
    mock kinds only; sprints 560/561 add production kinds.
    """
    catalog_path = os.environ.get(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", "",
    ).strip()
    trust_kind = os.environ.get(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "",
    ).strip()
    pool_kind = os.environ.get(
        "PRSM_PARALLAX_GPU_POOL_KIND", "",
    ).strip()

    # No-opt-in case: silent return. The operator simply didn't set
    # any of the parallax env vars.
    if not (catalog_path or trust_kind or pool_kind):
        return None

    # ── model catalog ────────────────────────────────────
    if not catalog_path:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE not set. Provide a "
            "JSON file mapping model_id → ModelInfo kwargs."
        )
        return None
    cat_p = Path(catalog_path)
    if not cat_p.exists():
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE=%s does not exist "
            "on disk.", catalog_path,
        )
        return None
    try:
        raw = json.loads(cat_p.read_text())
    except json.JSONDecodeError as exc:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s failed JSON parse: %s",
            catalog_path, exc,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s top-level must be a dict; got %s",
            catalog_path, type(raw).__name__,
        )
        return None

    # Sprint 559 — schema version + required-field validation. The
    # canonical v1 shape is {schema_version: "v1", models: {...}}.
    # Sprint 558's transitional shape (top-level dict of models)
    # is rejected with a migration hint.
    schema_version = raw.get("schema_version")
    if schema_version is None:
        if "models" in raw:
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s missing required top-level "
                "`schema_version` field. Add `\"schema_version\": "
                "\"v1\"` to the top level.",
                catalog_path,
            )
        else:
            # Legacy sprint-558 shape (top-level dict of models).
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s uses the deprecated sprint-558 "
                "top-level dict format. Migrate to the v1 schema: "
                "{\"schema_version\": \"v1\", \"models\": {...}}.",
                catalog_path,
            )
        return None
    if schema_version not in _CATALOG_SCHEMA_VERSIONS:
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s has schema_version=%r — this daemon only "
            "supports %s. Either upgrade the daemon or downgrade "
            "the catalog format.",
            catalog_path, schema_version, _CATALOG_SCHEMA_VERSIONS,
        )
        return None

    models_section = raw.get("models")
    if not isinstance(models_section, dict):
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s missing/invalid `models` section "
            "(must be a dict mapping model_id → ModelInfo kwargs; "
            "got %s).",
            catalog_path, type(models_section).__name__,
        )
        return None

    # Sprint 559 — per-entry required-field validation. ModelInfo's
    # __init__(**kwargs) silently accepts any input; operator typos
    # in field names would otherwise fail at scheduling time with
    # cryptic AttributeErrors.
    for model_id, kw in models_section.items():
        if not isinstance(kw, dict):
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s entry %r must be a dict of "
                "ModelInfo kwargs; got %s.",
                catalog_path, model_id, type(kw).__name__,
            )
            return None
        missing = [
            f for f in _REQUIRED_MODEL_INFO_FIELDS if f not in kw
        ]
        if missing:
            logger.warning(
                "Sprint 559 ParallaxScheduledExecutor wiring: "
                "model catalog %s entry %r is missing required "
                "ModelInfo field(s): %s",
                catalog_path, model_id, missing,
            )
            return None

    try:
        from prsm.compute.parallax_scheduling.model_info import (
            ModelInfo,
        )
        catalog = {
            mid: ModelInfo(**kw)
            for mid, kw in models_section.items()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s contains an entry that failed ModelInfo "
            "construction: %s", catalog_path, exc,
        )
        return None

    if not catalog:
        logger.warning(
            "Sprint 559 ParallaxScheduledExecutor wiring: model "
            "catalog %s has an empty `models` section — daemon "
            "will advertise no models. Add at least one entry to "
            "serve inference.",
            catalog_path,
        )

    # ── trust stack ──────────────────────────────────────
    if trust_kind not in _KNOWN_TRUST_STACK_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_TRUST_STACK_KIND=%r unrecognized. "
            "Known: %s", trust_kind, _KNOWN_TRUST_STACK_KINDS,
        )
        return None
    if trust_kind == "mock":
        trust_stack = _build_mock_trust_stack()
    elif trust_kind == "production":
        trust_stack = _build_production_trust_stack_or_none()
        if trust_stack is None:
            return None
    else:  # defensive — already gated above
        return None

    # ── gpu pool ─────────────────────────────────────────
    if pool_kind not in _KNOWN_GPU_POOL_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_GPU_POOL_KIND=%r unrecognized. "
            "Known: %s", pool_kind, _KNOWN_GPU_POOL_KINDS,
        )
        return None
    if pool_kind == "static-empty":
        pool_provider = _build_static_empty_pool_provider()
    else:  # defensive
        return None

    # ── identity ─────────────────────────────────────────
    identity = getattr(node, "identity", None)
    if identity is None:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: node has "
            "no .identity — daemon must initialize identity before "
            "this builder runs."
        )
        return None

    # ── construct ────────────────────────────────────────
    try:
        from prsm.compute.inference.parallax_executor import (
            ParallaxScheduledExecutor,
        )
        return ParallaxScheduledExecutor(
            gpu_pool_provider=pool_provider,
            trust_stack=trust_stack,
            model_catalog=catalog,
            chain_executor=_build_chain_executor(node),
            node_identity=identity,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "ParallaxScheduledExecutor(...) raised at construction: "
            "%s", exc,
        )
        return None
