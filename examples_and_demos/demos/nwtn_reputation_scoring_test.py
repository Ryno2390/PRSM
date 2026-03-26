#!/usr/bin/env python3
"""
NWTN Rigorous Long-Horizon Test
================================
Task: Reputation-Weighted Model Discovery for PRSM

Unlike the toy demo (pre-scripted outputs), this test uses REAL LLM agents
that don't know what other agents will produce.  Cross-agent discoveries
drive genuine pivots mid-stream.  Output is evaluated by running it.

Problem being solved
--------------------
prsm/compute/federation/model_registry.py currently selects models by
``availability × performance_score`` — a naive static score set at
registration.  This is gameable and ignores actual runtime behaviour.

This test designs and implements a reputation scoring system that weights
models by:
  - usage_frequency  (how often the network actually calls this model)
  - user_rating      (explicit 1-5 star feedback from callers)
  - task_success_rate (fraction of calls that completed without error)
  - node_uptime      (the hosting node's recent availability)

Agent team (4 real LLMs, 3 rounds each)
----------------------------------------
  architect       anthropic/claude-3-haiku        schema + API design
  data-scientist  mistralai/mistral-small-3.1-24b  scoring formula + math
  backend-coder   deepseek/deepseek-chat-v3-0324   Python implementation
  security        qwen/qwen2.5-coder-7b-instruct   threat modelling

Multi-round structure
---------------------
Round 1 — each agent receives their domain brief and the existing codebase
Round 2 — each agent reads the shared whiteboard (other agents' R1 findings)
          and responds: pivots, cross-domain refinements, implementation
Round 3 — integration: coder finalizes code, scientist validates formula,
          architect writes docs, security signs off

What the test produces
----------------------
  examples_and_demos/demos/test_outputs/reputation_scoring.py   — the module
  examples_and_demos/demos/test_outputs/test_reputation_scoring.py — tests
  ~/.prsm/sessions/<session-id>/ledger/project_ledger.md         — narrative

Usage
-----
  python examples_and_demos/demos/nwtn_reputation_scoring_test.py

  # Override specific models:
  python ... --coder-model "meta-llama/llama-3.1-70b-instruct"

  # Dry-run with heuristic BSC (no API cost, instant):
  python ... --fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── repo root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.disable(logging.WARNING)

# ── Colours ───────────────────────────────────────────────────────────────
def _c(code): return f"\033[{code}m" if sys.stdout.isatty() else ""
RESET = _c("0"); BOLD = _c("1"); DIM = _c("2")
GREEN = _c("32"); YELLOW = _c("33"); CYAN = _c("36")
RED = _c("31"); BLUE = _c("34"); MAGENTA = _c("35"); WHITE = _c("37")

def hdr(t):  print(f"\n{BOLD}{CYAN}{'═'*68}{RESET}\n{BOLD}{CYAN}  {t}{RESET}\n{BOLD}{CYAN}{'═'*68}{RESET}")
def step(n, total, t): print(f"\n{BOLD}[{n}/{total}]{RESET} {t}")
def ok(t, i=2):   print(f"{' '*i}{GREEN}✅{RESET} {t}")
def info(t, i=4): print(f"{' '*i}{DIM}↳{RESET}  {t}")
def warn(t, i=4): print(f"{' '*i}{YELLOW}⚠️ {RESET} {t}")
def agent_hdr(name, role, model, round_n):
    print(f"\n  {BOLD}{MAGENTA}[Round {round_n}] {name}{RESET}  {DIM}({role} via {model}){RESET}")
def promoted_line(score, text):
    short = text[:70] + "…" if len(text) > 70 else text
    print(f"    {GREEN}[{score:.2f} ▲ PROMOTE]{RESET}  {short}")
def discarded_line(score, text, reason):
    short = text[:60] + "…" if len(text) > 60 else text
    print(f"    {DIM}[{score:.2f}  discard ]  {short}  ({reason}){RESET}")


# ══════════════════════════════════════════════════════════════════════════
# Agent definitions
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    id: str          # e.g. "agent/architect"
    role: str        # human-readable role
    model: str       # OpenRouter model ID
    system_prompt: str
    round_prompts: List[str] = field(default_factory=list)


def _build_agents(
    orchestrator_model: str,
    coder_model: str,
    scientist_model: str,
    security_model: str,
) -> List[Agent]:

    # ── Existing codebase snippet (ground-truth context for all agents) ───
    registry_snippet = """
# Excerpt from prsm/compute/federation/model_registry.py

@dataclass
class ModelDetails:
    model_id: str
    name: str
    provider: ModelProvider          # OPENAI | ANTHROPIC | HUGGINGFACE | LOCAL | PRSM
    capabilities: List[ModelCapability]
    context_length: int = 4096
    max_tokens: int = 2048
    cost_per_token: float = 0.0
    availability: float = 1.0        # 0.0–1.0  (currently set once at registration)
    performance_score: float = 0.8   # 0.0–1.0  (currently set once at registration)
    specialization_domains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(...)

class ModelRegistry:
    models: Dict[str, ModelDetails] = {}

    def discover_models(self, capability, domain=None) -> List[ModelDetails]:
        # Currently: returns all capable models sorted by performance_score
        # PROBLEM: performance_score is static — set at registration, never updated.
        # A model can register with performance_score=0.95, degrade over time,
        # and the registry will keep recommending it forever.
        ...

    def select_best_model(self, capability, domain=None) -> Optional[ModelDetails]:
        candidates = self.discover_models(capability, domain)
        return max(candidates, key=lambda m: m.availability * m.performance_score, default=None)
"""

    task_brief = (
        "TASK: Design and implement a ReputationScoringSystem for PRSM's model registry.\n"
        "The system should replace the static performance_score with a dynamic score that\n"
        "reflects actual runtime behaviour across four dimensions:\n"
        "  usage_count     — how many times this model has been called on the network\n"
        "  avg_rating      — average user-submitted rating (1.0–5.0 stars)\n"
        "  success_rate    — fraction of calls that returned without error (0.0–1.0)\n"
        "  uptime_pct      — the hosting node's availability over the last 30 days\n\n"
        "The final reputation score must be a single float in [0.0, 1.0] that can be\n"
        "used as a drop-in replacement for performance_score in ModelDetails.\n"
    )

    return [
        Agent(
            id="agent/architect",
            role="System Architect",
            model=orchestrator_model,
            system_prompt=(
                "You are a senior distributed systems architect working on PRSM, "
                "an open-source peer-to-peer AI protocol.  You design clean, extensible "
                "Python APIs.  You think carefully about data models, storage formats, "
                "and backward compatibility.  Be specific and concrete — write actual "
                "Pydantic/dataclass definitions, not just prose descriptions."
            ),
            round_prompts=[
                # Round 1
                (
                    task_brief + registry_snippet +
                    "\nROUND 1 — SCHEMA DESIGN\n"
                    "Design the Python data model for storing reputation data.\n"
                    "Produce: (a) a ReputationRecord dataclass/Pydantic model with all fields,\n"
                    "(b) a ReputationStore class interface with update() and get() methods,\n"
                    "(c) your reasoning about score decay — should old data matter less over time?\n"
                    "Write actual Python code for (a) and (b).  Be specific about types.\n"
                    "Flag any design decisions that the data scientist or security reviewer\n"
                    "should weigh in on."
                ),
                # Round 2 — receives whiteboard context at runtime
                (
                    "ROUND 2 — SCHEMA REVISION\n"
                    "You have now seen the data scientist's scoring formula and the security\n"
                    "reviewer's threat analysis (summarised in the shared whiteboard above).\n"
                    "Revise your schema to:\n"
                    "  1. Support whatever minimum sample size the scientist recommended\n"
                    "  2. Add any fields needed for the security mitigations\n"
                    "  3. Ensure the decay function is implementable given the storage model\n"
                    "Produce the FINAL ReputationRecord definition (complete Python code).\n"
                    "Explicitly note any changes from Round 1 and why you made them."
                ),
                # Round 3
                (
                    "ROUND 3 — API DOCUMENTATION\n"
                    "Write the integration guide for how reputation_scoring.py plugs into\n"
                    "the existing ModelRegistry.  Specifically:\n"
                    "  1. What changes (if any) are needed to ModelDetails or ModelRegistry?\n"
                    "  2. Show a code example of calling update_reputation() after a model call\n"
                    "  3. Show a code example of select_best_model() using the new score\n"
                    "  4. Document the migration path for existing deployments\n"
                    "Write this as Python docstrings / inline comments, not separate prose."
                ),
            ],
        ),

        Agent(
            id="agent/data-scientist",
            role="Data Scientist",
            model=scientist_model,
            system_prompt=(
                "You are a data scientist and applied mathematician working on PRSM. "
                "You design statistically sound scoring systems that are robust to noise "
                "and manipulation.  You think carefully about sample sizes, confidence "
                "intervals, and distribution assumptions.  Show your working — write "
                "actual formulas using LaTeX-style notation (e.g. score = 0.4*u + 0.3*r)."
            ),
            round_prompts=[
                # Round 1
                (
                    task_brief + registry_snippet +
                    "\nROUND 1 — SCORING FORMULA\n"
                    "Propose the reputation scoring formula combining the four dimensions.\n"
                    "Produce:\n"
                    "  (a) The weighted formula with explicit weights that sum to 1.0\n"
                    "  (b) Normalisation approach for each dimension (usage_count is unbounded!)\n"
                    "  (c) A decay function — how should scores shrink if a model goes quiet?\n"
                    "  (d) Five concrete worked examples:\n"
                    "      - new model (0 calls so far)\n"
                    "      - established well-rated model\n"
                    "      - high-usage but poorly-rated model\n"
                    "      - recently degraded model (success_rate dropped last week)\n"
                    "      - recovering model (improving after a bad patch)\n"
                    "Show all arithmetic.  Flag statistical concerns the security reviewer\n"
                    "or architect should consider."
                ),
                # Round 2
                (
                    "ROUND 2 — FORMULA HARDENING\n"
                    "You have seen the security reviewer's threat vectors (in the whiteboard above).\n"
                    "Specifically address:\n"
                    "  1. Minimum sample size: at what N does a rating become statistically valid?\n"
                    "     Propose a confidence-weighted formula: score blends toward prior (0.5)\n"
                    "     until N reaches your threshold.\n"
                    "  2. Outlier detection: how do you catch a sudden spike in ratings?\n"
                    "  3. Does your decay function inadvertently punish models that are simply\n"
                    "     specialised (used only for rare tasks)?\n"
                    "Revise the formula and re-run the five worked examples with the hardened version."
                ),
                # Round 3
                (
                    "ROUND 3 — VALIDATION\n"
                    "The coder has produced an implementation (see whiteboard).\n"
                    "Validate it against your formula:\n"
                    "  1. Does the implementation correctly reproduce your Round 2 formula?\n"
                    "     If not, flag the discrepancy exactly.\n"
                    "  2. Provide a Python snippet (usable as a unit test) that checks all\n"
                    "     five worked examples produce scores within ±0.02 of your expected values.\n"
                    "  3. What is the worst-case scenario score under your formula?\n"
                    "     (i.e. what inputs produce score → 0 and score → 1?)"
                ),
            ],
        ),

        Agent(
            id="agent/backend-coder",
            role="Backend Engineer",
            model=coder_model,
            system_prompt=(
                "You are a senior Python engineer working on PRSM.  You write clean, "
                "well-typed, production-ready Python 3.11+ code.  You use dataclasses "
                "and Pydantic where appropriate.  You write thorough docstrings.  "
                "You never leave TODOs — if you can't implement something completely, "
                "you implement what you can and explicitly flag the gap."
            ),
            round_prompts=[
                # Round 1
                (
                    task_brief + registry_snippet +
                    "\nROUND 1 — MODULE SKELETON\n"
                    "Write the skeleton for prsm/compute/federation/reputation_scoring.py.\n"
                    "Include:\n"
                    "  - Module docstring explaining purpose + integration points\n"
                    "  - All imports\n"
                    "  - ReputationRecord class (use placeholder fields for now)\n"
                    "  - ReputationScoringSystem class with method stubs:\n"
                    "      update_reputation(model_id, rating, success, response_time_ms)\n"
                    "      get_reputation_score(model_id) -> float\n"
                    "      get_all_scores() -> Dict[str, float]\n"
                    "  - Integration hook: patch_model_registry(registry) that monkeypatches\n"
                    "    select_best_model() to use reputation scores\n"
                    "Do NOT implement the actual scoring formula yet — wait for Round 2\n"
                    "once the scientist and architect have finalised their designs."
                ),
                # Round 2
                (
                    "ROUND 2 — CORE IMPLEMENTATION\n"
                    "The whiteboard above contains the architect's final schema (Round 2)\n"
                    "and the data scientist's hardened formula (Round 2).  Implement both.\n\n"
                    "Write the COMPLETE implementation of:\n"
                    "  - ReputationRecord (use the architect's exact field definitions)\n"
                    "  - The scoring formula (use the scientist's exact weights + decay)\n"
                    "  - update_reputation() — records one event and recalculates score\n"
                    "  - get_reputation_score() — returns current score, handles cold-start\n"
                    "Include all error handling.  Flag any ambiguities in the formula\n"
                    "spec that you had to resolve — the scientist needs to know."
                ),
                # Round 3
                (
                    "ROUND 3 — FINALISATION + TESTS\n"
                    "Produce the COMPLETE final file that goes into the PRSM repository:\n\n"
                    "File 1: prsm/compute/federation/reputation_scoring.py\n"
                    "  - All classes fully implemented\n"
                    "  - The patch_model_registry() function\n"
                    "  - A module-level get_reputation_system() singleton factory\n"
                    "  - No stubs, no TODOs\n\n"
                    "File 2: tests/test_reputation_scoring.py\n"
                    "  - At least 6 pytest test functions\n"
                    "  - Test cold-start (brand-new model, no history)\n"
                    "  - Test the scientist's five worked examples\n"
                    "  - Test that score stays in [0.0, 1.0] under adversarial inputs\n"
                    "  - Test the registry integration (patch_model_registry)\n\n"
                    "Wrap each file in a clearly labelled code block:\n"
                    "```python\n# FILE: prsm/compute/federation/reputation_scoring.py\n...\n```"
                ),
            ],
        ),

        Agent(
            id="agent/security-reviewer",
            role="Security Reviewer",
            model=security_model,
            system_prompt=(
                "You are a security engineer specialising in distributed systems and "
                "adversarial ML.  You think like an attacker first.  For every design "
                "you see, you ask: how could a malicious node exploit this?  You are "
                "specific — you give concrete attack scenarios with realistic assumptions, "
                "not vague hand-waving.  You also propose concrete mitigations."
            ),
            round_prompts=[
                # Round 1
                (
                    task_brief + registry_snippet +
                    "\nROUND 1 — THREAT MODELLING\n"
                    "Identify the top attack vectors against a reputation scoring system\n"
                    "on a permissionless peer-to-peer network like PRSM.\n\n"
                    "For each threat, provide:\n"
                    "  - Attack name and realistic scenario (who, how, what they gain)\n"
                    "  - Severity (Critical / High / Medium / Low)\n"
                    "  - Concrete mitigation that can be built INTO the scoring formula\n"
                    "    or data model (not just 'use HTTPS' or 'add authentication')\n\n"
                    "Consider at minimum:\n"
                    "  1. Sybil inflation (operator runs N nodes, all rate their own model highly)\n"
                    "  2. Wash trading (model calls itself to inflate usage_count)\n"
                    "  3. Rating bombs (coordinate to tank a competitor's avg_rating)\n"
                    "  4. Success-rate spoofing (return HTTP 200 but garbage output)\n"
                    "  5. Cold-start exploitation (new models get 'benefit of the doubt')\n"
                    "Flag which threats the data scientist's formula needs to address."
                ),
                # Round 2
                (
                    "ROUND 2 — IMPLEMENTATION REVIEW\n"
                    "The coder has produced a module skeleton (see whiteboard).\n"
                    "The data scientist has proposed a formula (see whiteboard).\n"
                    "Review both for security gaps:\n"
                    "  1. Does the implementation validate inputs? (negative ratings? NaN?)\n"
                    "  2. Are there integer overflow or float precision issues?\n"
                    "  3. Does the schema make Sybil detection possible, or does it\n"
                    "     destroy the evidence needed to detect it?\n"
                    "  4. Is the cold-start handling (scientist's confidence weighting)\n"
                    "     sufficient to prevent cold-start exploitation?\n"
                    "Provide a ranked list of remaining concerns, most critical first."
                ),
                # Round 3
                (
                    "ROUND 3 — FINAL SIGN-OFF\n"
                    "Review the final implementation (see whiteboard for coder's Round 2 output).\n"
                    "Produce:\n"
                    "  1. Security Assessment Summary (3-5 sentences)\n"
                    "  2. Resolved threats (addressed in the implementation)\n"
                    "  3. Open risks (not yet addressed — with recommended next steps)\n"
                    "  4. Overall security confidence level: 1 (do not ship) to 5 (ship it)\n"
                    "  5. One concrete code change the coder must make before production.\n"
                    "     Write the actual Python diff (before/after)."
                ),
            ],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════
# BSC pipeline (fast / real)
# ══════════════════════════════════════════════════════════════════════════

def _build_promoter(fast: bool, bsc_model: str):
    import math as _math
    from prsm.compute.nwtn.bsc.predictor import SurpriseScore
    from prsm.compute.nwtn.bsc.semantic_dedup import DedupResult
    from prsm.compute.nwtn.bsc.kl_filter import AdaptiveKLFilter
    from prsm.compute.nwtn.bsc.promoter import BSCPromoter

    if fast:
        _SIGNAL = ["CRITICAL", "FINDING", "ATTACK", "VULNERABILITY", "CONCERN",
                   "DECISION", "FORMULA", "REVISED", "SCHEMA", "MISMATCH",
                   "SEVERITY", "THRESHOLD", "OVERFLOW", "CONFIDENCE", "SIGN-OFF"]

        class _HeuristicPredictor:
            baseline_perplexity = 50.0
            async def score_surprise(self, context, chunk):
                hit = any(w in chunk.upper() for w in _SIGNAL)
                raw = 110.0 if hit else 20.0
                z = (raw - 45.0) / 22.5
                score = 1.0 / (1.0 + _math.exp(-z))
                return SurpriseScore(score=round(score, 3), raw_perplexity=raw,
                                     token_count=len(chunk.split()),
                                     context_tokens=50, adaptive_baseline=45.0)
            async def warmup(self): pass

        class _NullDedup:
            index_size = 0
            async def check(self, _): return DedupResult(is_redundant=False, max_similarity=0.0,
                                                          most_similar_index=None, reason="fast")
            async def add_to_index(self, _): return 0
            def clear(self): pass

        return BSCPromoter(predictor=_HeuristicPredictor(),
                           kl_filter=AdaptiveKLFilter(epsilon=0.50),
                           deduplicator=_NullDedup())
    else:
        from prsm.compute.nwtn.bsc import BSCDeploymentConfig, BSCPromoter, DeploymentMode
        cfg = BSCDeploymentConfig(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            model_name=bsc_model,
            epsilon=0.50,
            similarity_threshold=0.82,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        promoter = BSCPromoter.from_config(cfg)
        return promoter


# ══════════════════════════════════════════════════════════════════════════
# OpenRouter agent calls
# ══════════════════════════════════════════════════════════════════════════

async def call_agent(
    agent: Agent,
    prompt: str,
    api_key: str,
    max_tokens: int = 1500,
    temperature: float = 0.4,
) -> str:
    """Call the agent's model via OpenRouter and return the text response."""
    from prsm.compute.nwtn.backends.openrouter_backend import OpenRouterBackend
    b = OpenRouterBackend(api_key=api_key, default_model=agent.model)
    await b.initialize()
    try:
        result = await asyncio.wait_for(
            b.generate(
                prompt=prompt,
                system_prompt=agent.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=60.0,
        )
        return result.content
    finally:
        await b._session.close()


# ══════════════════════════════════════════════════════════════════════════
# Main test orchestration
# ══════════════════════════════════════════════════════════════════════════

async def run_test(
    session_id: str,
    output_dir: Path,
    api_key: str,
    orchestrator_model: str,
    coder_model: str,
    scientist_model: str,
    security_model: str,
    bsc_model: str,
    fast: bool,
    n_rounds: int,
) -> dict:
    from prsm.compute.nwtn.whiteboard import WhiteboardStore
    from prsm.compute.nwtn.whiteboard.query import WhiteboardQuery
    from prsm.compute.nwtn.synthesis import NarrativeSynthesizer, ProjectLedger, LedgerSigner
    from prsm.compute.nwtn.bsc import (
        PromotionDecision, ChunkMetadata, FilterDecision, KLFilterResult,
    )
    from prsm.compute.nwtn.team import MetaPlanner
    from prsm.compute.nwtn.team.interview import ProjectBrief

    total_steps = n_rounds + 3   # rounds + setup + synthesis + verify

    # ── Step 0: setup ─────────────────────────────────────────────────────
    step(0, total_steps, "Setting up session, BSC pipeline, and whiteboard…")

    store = WhiteboardStore(output_dir / "whiteboard.db")
    await store.open()
    await store.create_session(session_id)

    promoter = _build_promoter(fast, bsc_model)
    if not fast:
        info(f"Loading BSC predictor: {bsc_model} (may take 15-30s)")
        await promoter.warmup()
        ok("BSC predictor ready")
    else:
        ok("Fast mode: keyword heuristic BSC active")

    agents = _build_agents(orchestrator_model, coder_model, scientist_model, security_model)
    ok(f"Agent team assembled: {len(agents)} agents")
    for a in agents:
        info(f"[{a.id.split('/')[1]:<20}] {a.role:<20} {a.model}")

    # ── Step 1: Generate MetaPlan via orchestrator ─────────────────────────
    step(1, total_steps, f"Orchestrator ({orchestrator_model}) generating MetaPlan…")

    class _TextWrap:
        def __init__(self, content): self.text = content

    class _OrchestratorBackend:
        async def generate(self, prompt, max_tokens=1200, temperature=0.3, **kw):
            result = await call_agent(
                Agent(id="orchestrator", role="Orchestrator", model=orchestrator_model,
                      system_prompt="You are NWTN, a project planning AI. Generate structured MetaPlans."),
                prompt=prompt, api_key=api_key, max_tokens=max_tokens, temperature=temperature,
            )
            return _TextWrap(result)

    brief = ProjectBrief(
        session_id=session_id,
        goal="Implement ReputationScoringSystem for PRSM model discovery",
        technology_stack=["Python 3.11", "Pydantic", "asyncio", "SQLite"],
        constraints=[
            "Must not break existing ModelRegistry API",
            "Score must be a float in [0.0, 1.0]",
            "Must be robust against Sybil attacks",
        ],
        success_criteria=[
            "All pytest tests pass",
            "patch_model_registry() is a one-line integration",
            "Security confidence level ≥ 4/5",
        ],
        timeline="1 session",
        llm_assisted=not fast,
    )

    backend_for_planner = None if fast else _OrchestratorBackend()
    planner = MetaPlanner(backend_registry=backend_for_planner)
    meta_plan = await planner.generate(brief)
    ok(f"MetaPlan: {meta_plan.title}")
    for i, m in enumerate(meta_plan.milestones, 1):
        info(f"Milestone {i}: {m.title}")

    # Seed whiteboard with MetaPlan (always-promoted)
    def _force_write(chunk, source):
        return PromotionDecision(
            promoted=True, chunk=chunk,
            metadata=ChunkMetadata(source_agent=source, session_id=session_id,
                                   timestamp=datetime.now(timezone.utc)),
            surprise_score=1.0, raw_perplexity=0.0, similarity_score=0.0,
            kl_result=KLFilterResult(decision=FilterDecision.PROMOTE, score=1.0,
                                     epsilon=0.0, reason="forced"),
            dedup_result=None, reason="forced write",
        )

    await store.write(_force_write(meta_plan.to_whiteboard_entry(), "nwtn/orchestrator"))

    # ── Steps 2..N+1: Agent rounds ─────────────────────────────────────────
    query = WhiteboardQuery(store)
    all_agent_outputs: Dict[str, List[str]] = {a.id: [] for a in agents}
    round_stats = []

    for round_n in range(n_rounds):
        step(round_n + 2, total_steps,
             f"Round {round_n + 1}/{n_rounds} — all agents work in parallel")

        # Get current whiteboard context (what agents share)
        wb_ctx = await query.compressed_state(session_id, max_chars=3000)

        # Build each agent's prompt for this round
        tasks = []
        for agent in agents:
            if round_n >= len(agent.round_prompts):
                continue
            base_prompt = agent.round_prompts[round_n]
            if round_n > 0:
                full_prompt = (
                    f"SHARED WHITEBOARD (discoveries from all agents so far):\n"
                    f"{'─'*60}\n{wb_ctx}\n{'─'*60}\n\n"
                    f"{base_prompt}"
                )
            else:
                full_prompt = base_prompt

            tasks.append((agent, full_prompt))

        # Call all agents in parallel
        async def _call_one(agent, prompt):
            if not api_key or fast:
                return _mock_response(agent, round_n)
            return await call_agent(agent, prompt, api_key, max_tokens=1500)

        responses = await asyncio.gather(
            *[_call_one(a, p) for a, p in tasks],
            return_exceptions=True,
        )

        # Feed responses through BSC
        n_promoted = n_discarded = 0
        for (agent, _), response in zip(tasks, responses):
            if isinstance(response, Exception):
                warn(f"{agent.id}: call failed — {response}")
                continue

            all_agent_outputs[agent.id].append(response)
            agent_hdr(agent.id, agent.role, agent.model, round_n + 1)

            # Split response into chunks (sentences / paragraphs)
            chunks = _split_into_chunks(response)
            for chunk in chunks:
                if len(chunk.split()) < 8:
                    continue
                ctx_for_bsc = wb_ctx[-1500:] if len(wb_ctx) > 1500 else wb_ctx
                decision = await promoter.process_chunk(
                    chunk=chunk,
                    context=ctx_for_bsc,
                    source_agent=agent.id,
                    session_id=session_id,
                )
                if decision.promoted:
                    promoted_line(decision.surprise_score, chunk)
                    await store.write(decision)
                    n_promoted += 1
                else:
                    reason = "low surprise" if decision.surprise_score < 0.50 else "duplicate"
                    discarded_line(decision.surprise_score, chunk, reason)
                    n_discarded += 1

        bsc_stats = promoter.stats
        round_stats.append({
            "round": round_n + 1,
            "promoted": n_promoted,
            "discarded": n_discarded,
            "whiteboard_entries": await store.entry_count(session_id),
        })
        ok(f"Round {round_n+1} complete: {n_promoted} promoted, {n_discarded} discarded | "
           f"whiteboard: {await store.entry_count(session_id)} entries total")

    # ── Synthesis ──────────────────────────────────────────────────────────
    step(n_rounds + 2, total_steps, "Nightly Synthesis — narrative generation…")

    snapshot = await query.snapshot(session_id)
    synth_backend = None
    if api_key and not fast:
        class _SynthBackend:
            async def generate(self, prompt, max_tokens=1000, temperature=0.4, **kw):
                result = await call_agent(
                    Agent(id="synthesizer", role="Synthesizer", model=orchestrator_model,
                          system_prompt="You are NWTN's synthesis agent. Write clear, structured summaries."),
                    prompt=prompt, api_key=api_key, max_tokens=max_tokens, temperature=temperature,
                )
                return _TextWrap(result)
        synth_backend = _SynthBackend()

    synthesizer = NarrativeSynthesizer(backend_registry=synth_backend)
    synthesis = await synthesizer.synthesise(snapshot, meta_plan=meta_plan)

    ledger_dir = output_dir / "ledger"
    ledger = ProjectLedger(ledger_dir=ledger_dir, project_title="PRSM Reputation Scoring")
    ledger.load()
    signer = LedgerSigner(keyfile_path=output_dir / "ledger.key")
    signer.load_or_generate()
    entry = ledger.append(synthesis, signer)
    ok(f"Ledger entry #{entry.entry_index} appended (chain hash: {entry.chain_hash[:20]}…)")

    # ── Verification ───────────────────────────────────────────────────────
    step(n_rounds + 3, total_steps, "Verifying tamper-evident chain…")
    v = ledger.verify()
    if v.valid:
        ok(f"Chain VALID — {v.entry_count} entries")
    else:
        warn(f"Chain INVALID at entry #{v.first_bad_index}: {v.reason}")

    await store.close()

    # ── Extract code artifacts ─────────────────────────────────────────────
    coder_outputs = all_agent_outputs.get("agent/backend-coder", [])
    code_files = _extract_code_files(coder_outputs)

    return {
        "session_id": session_id,
        "meta_plan_title": meta_plan.title,
        "agents": [{"id": a.id, "model": a.model} for a in agents],
        "round_stats": round_stats,
        "whiteboard_entries": snapshot.entry_count,
        "chain_valid": v.valid,
        "ledger_entry": entry.entry_index,
        "chain_hash": entry.chain_hash,
        "synthesis_llm": not fast and bool(api_key),
        "code_files": code_files,
        "all_outputs": all_agent_outputs,
    }


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _split_into_chunks(text: str, min_words: int = 15) -> List[str]:
    """
    Split agent output into BSC-evaluable chunks.

    Strategy: split on double-newlines (paragraphs) first, then fall back
    to sentence splitting for very long paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para.split()) >= min_words:
            chunks.append(para)
    return chunks if chunks else [text]


def _extract_code_files(outputs: List[str]) -> Dict[str, str]:
    """
    Extract labelled code blocks from the coder agent's output.
    Looks for: ```python\n# FILE: path/to/file.py\n...\n```
    """
    files: Dict[str, str] = {}
    for output in outputs:
        pattern = r"```python\n# FILE: ([^\n]+)\n(.*?)```"
        for match in re.finditer(pattern, output, re.DOTALL):
            filepath = match.group(1).strip()
            code = match.group(2)
            if filepath in files:
                files[filepath] += "\n\n# --- (continued from later round) ---\n\n" + code
            else:
                files[filepath] = code
    return files


def _mock_response(agent: Agent, round_n: int) -> str:
    """Fast-mode mock: returns a plausible response with some signal words."""
    mocks = {
        "agent/architect": [
            "DECISION: Using Pydantic v2 BaseModel for ReputationRecord. Fields: model_id (str), "
            "usage_count (int, default 0), avg_rating (float, default None), "
            "success_rate (float, default None), uptime_pct (float, default 100.0), "
            "last_updated (datetime), observation_count (int, default 0). "
            "The observation_count is critical for the confidence-weighted formula — "
            "we need to know how many data points back each rating.",

            "SCHEMA REVISION: Adding node_id field to ReputationRecord to enable "
            "Sybil detection. A model's scores from its own hosting node are flagged "
            "separately (self_ratings vs peer_ratings). This was missing in Round 1. "
            "The security reviewer correctly identified this gap.",

            "INTEGRATION: patch_model_registry(registry) replaces select_best_model "
            "in-place. One-line call in existing code. Migration: run backfill_scores() "
            "once on startup to compute initial scores from existing metadata.",
        ],
        "agent/data-scientist": [
            "FORMULA: reputation_score = 0.25*U + 0.35*R + 0.30*S + 0.10*A where "
            "U = min(usage_count/1000, 1.0), R = (avg_rating-1)/4, "
            "S = success_rate, A = uptime_pct/100. "
            "FINDING: usage_count needs logarithmic normalisation not linear — "
            "a model with 10000 calls is not 10x better than one with 1000 calls. "
            "Revised: U = log(1 + usage_count) / log(1001).",

            "CONFIDENCE WEIGHTING: For obs_count < 20, blend toward prior (0.5): "
            "score = score * (obs_count/20) + 0.5 * (1 - obs_count/20). "
            "This prevents cold-start exploitation. CONCERN: decay function "
            "must not penalise specialised models. Using half-life of 30 days "
            "only on stale models (last_used > 60 days ago).",

            "VALIDATION: Five test cases pass within tolerance. Worst case "
            "FINDING: a model with 0 observations and no rating collapses to "
            "exactly 0.5 (prior), not 0.0. This is correct behaviour.",
        ],
        "agent/backend-coder": [
            "Module skeleton written. ReputationRecord and ReputationScoringSystem "
            "stubs in place. FINDING: existing ModelRegistry.select_best_model() "
            "is synchronous but our update pipeline is async. Need to make "
            "patch_model_registry() handle the async/sync boundary carefully.",

            "IMPLEMENTATION COMPLETE: Core scoring logic matches scientist's formula. "
            "CRITICAL BUG FOUND: avg_rating field was None for new models, causing "
            "ZeroDivisionError in formula. Fixed with: R = (avg_rating - 1) / 4 if "
            "avg_rating is not None else 0.5. Added input validation for all fields.",

            "```python\n# FILE: prsm/compute/federation/reputation_scoring.py\n"
            "from __future__ import annotations\nimport math\nfrom dataclasses import dataclass, field\n"
            "from datetime import datetime, timezone\nfrom typing import Dict, Optional\n\n"
            "@dataclass\nclass ReputationRecord:\n    model_id: str\n    usage_count: int = 0\n"
            "    avg_rating: Optional[float] = None\n    success_rate: Optional[float] = None\n"
            "    uptime_pct: float = 100.0\n    observation_count: int = 0\n\n"
            "class ReputationScoringSystem:\n    def __init__(self):\n        self._records: Dict[str, ReputationRecord] = {}\n\n"
            "    def update_reputation(self, model_id, rating=None, success=None, uptime=None):\n"
            "        r = self._records.setdefault(model_id, ReputationRecord(model_id=model_id))\n"
            "        if rating is not None:\n            r.avg_rating = rating\n"
            "        if success is not None:\n            r.success_rate = success\n"
            "        r.observation_count += 1\n\n"
            "    def get_reputation_score(self, model_id) -> float:\n"
            "        r = self._records.get(model_id)\n"
            "        if r is None: return 0.5\n"
            "        U = math.log(1 + r.usage_count) / math.log(1001)\n"
            "        R = (r.avg_rating - 1) / 4 if r.avg_rating else 0.5\n"
            "        S = r.success_rate if r.success_rate is not None else 0.5\n"
            "        A = r.uptime_pct / 100.0\n"
            "        raw = 0.25*U + 0.35*R + 0.30*S + 0.10*A\n"
            "        if r.observation_count < 20:\n"
            "            raw = raw * (r.observation_count/20) + 0.5 * (1 - r.observation_count/20)\n"
            "        return max(0.0, min(1.0, raw))\n"
            "```\n\n"
            "```python\n# FILE: tests/test_reputation_scoring.py\n"
            "import pytest\nfrom prsm.compute.federation.reputation_scoring import ReputationScoringSystem\n\n"
            "def test_cold_start_returns_prior():\n    s = ReputationScoringSystem()\n"
            "    assert s.get_reputation_score('new-model') == 0.5\n\n"
            "def test_score_in_unit_interval():\n    s = ReputationScoringSystem()\n"
            "    s.update_reputation('m', rating=-999, success=-1, uptime=999)\n"
            "    score = s.get_reputation_score('m')\n"
            "    assert 0.0 <= score <= 1.0\n"
            "```",
        ],
        "agent/security-reviewer": [
            "CRITICAL THREAT: Sybil inflation. A node operator runs 50 nodes all "
            "submitting 5-star ratings for their own model. Mitigation: separate "
            "self_ratings (from hosting node) and peer_ratings (from other nodes). "
            "Only peer_ratings count toward avg_rating. HIGH SEVERITY.",

            "FINDING: The coder's implementation does not validate that rating is in "
            "[1.0, 5.0]. A malicious caller can submit rating=100.0 and inflate the "
            "score beyond [0,1]. Must add: if not 1.0 <= rating <= 5.0: raise ValueError. "
            "CRITICAL: this should be in the Pydantic model validator, not just the update method.",

            "SECURITY SIGN-OFF: Confidence level 3/5. Remaining risk: no "
            "rate-limiting on update_reputation() calls. A single node could call "
            "it 10,000 times to inflate observation_count past the confidence threshold "
            "(20 observations). Required fix: add per-model, per-node rate limiting "
            "before production deployment.",
        ],
    }
    agent_key = agent.id
    responses = mocks.get(agent_key, ["Progress on task proceeding normally. No blockers identified."])
    return responses[min(round_n, len(responses) - 1)]


def _print_results(result: dict, output_dir: Path, elapsed: float) -> None:
    hdr("Test Results")

    print(f"\n  {BOLD}Session:{RESET}       {result['session_id']}")
    print(f"  {BOLD}MetaPlan:{RESET}      {result['meta_plan_title']}")
    print(f"  {BOLD}Elapsed:{RESET}       {elapsed:.1f}s")
    print(f"  {BOLD}Chain valid:{RESET}   {'✅' if result['chain_valid'] else '❌'}")

    print(f"\n  {BOLD}BSC Statistics by Round:{RESET}")
    for r in result["round_stats"]:
        print(f"    Round {r['round']}: {GREEN}{r['promoted']} promoted{RESET}, "
              f"{DIM}{r['discarded']} discarded{RESET} | "
              f"whiteboard: {r['whiteboard_entries']} entries")

    print(f"\n  {BOLD}Final Whiteboard:{RESET}  {result['whiteboard_entries']} total entries")
    print(f"  {BOLD}Ledger Entry:{RESET}     #{result['ledger_entry']}  "
          f"(hash: {result['chain_hash'][:24]}…)")

    # Code output
    code_files = result.get("code_files", {})
    if code_files:
        print(f"\n  {BOLD}Code Artifacts Extracted:{RESET}  ({len(code_files)} files)")
        for filepath, code in code_files.items():
            # Write to output dir
            target = output_dir / Path(filepath).name
            target.write_text(code)
            print(f"    ✅ {filepath}")
            print(f"       → {target}  ({len(code.splitlines())} lines)")
    else:
        print(f"\n  {DIM}No code blocks extracted from coder output.{RESET}")
        print(f"  {DIM}(Check {output_dir}/raw_outputs.json for full agent text){RESET}")

    # Save all raw outputs for inspection
    raw_out = output_dir / "raw_outputs.json"
    raw_out.write_text(json.dumps(result["all_outputs"], indent=2, default=str))
    print(f"\n  {BOLD}All agent outputs:{RESET}  {raw_out}")

    print(f"\n  {BOLD}Ledger narrative:{RESET}  {output_dir / 'ledger' / 'project_ledger.md'}")
    print(f"  {BOLD}Whiteboard DB:{RESET}     {output_dir / 'whiteboard.db'}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Load .env
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    p = argparse.ArgumentParser(description="NWTN Rigorous Long-Horizon Test")
    p.add_argument("--orchestrator-model", default="anthropic/claude-3-haiku",
                   help="Orchestrator (MetaPlan + synthesis). Default: claude-3-haiku")
    p.add_argument("--coder-model",        default="deepseek/deepseek-chat-v3-0324",
                   help="Backend coder agent. Default: deepseek-chat-v3-0324")
    p.add_argument("--scientist-model",    default="mistralai/mistral-small-3.1-24b-instruct",
                   help="Data scientist agent. Default: mistral-small-3.1-24b")
    p.add_argument("--security-model",     default="qwen/qwen2.5-coder-7b-instruct",
                   help="Security reviewer agent. Default: qwen2.5-coder-7b")
    p.add_argument("--bsc-model",          default="Qwen/Qwen2.5-3B-Instruct",
                   help="Local BSC predictor (HuggingFace, runs on MPS). Default: Qwen2.5-3B")
    p.add_argument("--rounds",             type=int, default=3,
                   help="Number of work rounds per agent (default: 3)")
    p.add_argument("--fast",               action="store_true",
                   help="Heuristic BSC + mock agent responses. No API calls. Instant.")
    p.add_argument("--session-id",         default=None)
    p.add_argument("--output-dir",         default=None)
    p.add_argument("--openrouter-key",     default=None,
                   help="OpenRouter API key (overrides OPENROUTER_API_KEY env)")
    args = p.parse_args()

    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.fast:
        print(f"{RED}No OPENROUTER_API_KEY found. Use --fast for mock mode or set the key.{RESET}")
        sys.exit(1)

    import uuid as _uuid
    session_id = args.session_id or f"rep-scoring-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path.home() / ".prsm" / "sessions" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr("NWTN Rigorous Long-Horizon Test")
    print(f"\n  {BOLD}Task:{RESET}          Reputation-Weighted Model Discovery for PRSM")
    print(f"  {BOLD}Session:{RESET}       {session_id}")
    print(f"  {BOLD}Mode:{RESET}          {'heuristic (--fast)' if args.fast else 'live LLM via OpenRouter'}")
    print(f"  {BOLD}Rounds:{RESET}        {args.rounds} per agent ({args.rounds * 4} total LLM calls)")
    if not args.fast:
        print(f"\n  {BOLD}Model routing:{RESET}")
        print(f"    {CYAN}Orchestrator{RESET}  {args.orchestrator_model}")
        print(f"    Architect     (same as orchestrator)")
        print(f"    Data Sci      {args.scientist_model}")
        print(f"    Coder         {args.coder_model}")
        print(f"    Security      {args.security_model}")
        print(f"    BSC Predictor {args.bsc_model} {DIM}(local MPS, free){RESET}")
    print(f"\n  {BOLD}Output:{RESET}        {out_dir}")
    print(f"\n  {BOLD}What this tests:{RESET}")
    print(f"    - Real LLMs producing real Python code, not scripted outputs")
    print(f"    - BSC filtering genuine discoveries from routine progress updates")
    print(f"    - Cross-agent pivots: security findings change the coder's impl")
    print(f"    - Output is evaluated by running the produced pytest suite")

    t0 = time.time()
    try:
        result = asyncio.run(run_test(
            session_id=session_id,
            output_dir=out_dir,
            api_key=api_key,
            orchestrator_model=args.orchestrator_model,
            coder_model=args.coder_model,
            scientist_model=args.scientist_model,
            security_model=args.security_model,
            bsc_model=args.bsc_model,
            fast=args.fast,
            n_rounds=args.rounds,
        ))
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted.{RESET}")
        sys.exit(0)
    except Exception as exc:
        import traceback
        print(f"\n{RED}Test failed: {exc}{RESET}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    _print_results(result, out_dir, elapsed)

    # If code was extracted, run the tests
    test_file = out_dir / "test_reputation_scoring.py"
    module_file = out_dir / "reputation_scoring.py"
    if test_file.exists() and module_file.exists():
        hdr("Running Generated Tests")
        import subprocess
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_ROOT) + ":" + str(out_dir)
        r = _subprocess_module.run(  # type: ignore
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
        if r.returncode == 0:
            ok("All generated tests passed — the Agent Team produced working code!")
        else:
            warn("Some generated tests failed — review the output above")

    hdr("Test complete")
    sys.exit(0)


import subprocess as _subprocess_module
if __name__ == "__main__":
    main()
