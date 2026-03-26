"""
prsm.compute.nwtn.openclaw — OpenClaw Integration Layer
=========================================================

Sub-phase 10.5: bridges NWTN's Agent Team pipeline to OpenClaw's
runtime infrastructure.

Components
----------
OpenClawGateway
    WebSocket client for the OpenClaw Gateway (``ws://127.0.0.1:18789``).
    Receives user messages, sends replies, delivers heartbeat events.
    Dependency-injection-friendly: inject a ``MockGatewayConnection``
    in tests to avoid a live WebSocket server.

SkillsBridge
    Translates between PRSM ``ModelCapability`` vocabulary and OpenClaw
    Skills Registry vocabulary.  ``team_member_to_skill()`` converts a
    ``TeamMember`` to an OpenClaw skill spec ready for registration.

HeartbeatHook
    Triggers the Nightly Synthesis pipeline.  Listens for Gateway
    heartbeat events; falls back to a self-timer when the Gateway is
    absent.  Calls ``NarrativeSynthesizer → ProjectLedger.append()``
    automatically.

NWTNOpenClawAdapter
    Top-level coordinator.  Receives a user goal, runs Interview →
    MetaPlanner → TeamAssembler → BranchManager → WhiteboardMonitor →
    HeartbeatHook in sequence.  Handles Gateway event routing and
    whiteboard lifecycle.

Standalone mode
---------------
All components work without an OpenClaw installation.  Pass ``gateway=None``
and supply a ``QuestionCallback`` directly to ``NWTNOpenClawAdapter.start_session``.

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.openclaw import (
        OpenClawGateway, NWTNOpenClawAdapter,
    )
    from prsm.compute.nwtn.whiteboard import WhiteboardStore
    from prsm.compute.nwtn.synthesis import (
        ProjectLedger, LedgerSigner, NarrativeSynthesizer,
    )
    from prsm.compute.nwtn.bsc import BSCPromoter, BSCDeploymentConfig

    store = WhiteboardStore(".prsm/whiteboard.db")
    await store.open()
    await store.create_session("my-session")

    promoter = BSCPromoter.from_config(BSCDeploymentConfig.auto())
    await promoter.warmup()

    ledger = ProjectLedger(Path(".prsm/ledger"), "My Project")
    ledger.load()
    signer = LedgerSigner(Path(".prsm/ledger.key"))

    adapter = NWTNOpenClawAdapter(
        whiteboard_store=store,
        promoter=promoter,
        ledger=ledger,
        signer=signer,
        synthesizer=NarrativeSynthesizer(),
    )

    # Standalone (no OpenClaw required)
    state = await adapter.start_session(
        goal="Build a distributed AI coordination layer",
        ask=lambda q: ainput(q),
    )
"""

from .adapter import NWTNOpenClawAdapter, SessionState
from .gateway import (
    OpenClawGateway,
    OpenClawMessage,
    AbstractGatewayConnection,
    DEFAULT_GATEWAY_URL,
)
from .heartbeat_hook import HeartbeatHook
from .skills_bridge import OpenClawSkillSpec, SkillsBridge

__all__ = [
    # Gateway
    "OpenClawGateway",
    "OpenClawMessage",
    "AbstractGatewayConnection",
    "DEFAULT_GATEWAY_URL",
    # Skills
    "SkillsBridge",
    "OpenClawSkillSpec",
    # Heartbeat
    "HeartbeatHook",
    # Adapter
    "NWTNOpenClawAdapter",
    "SessionState",
]
