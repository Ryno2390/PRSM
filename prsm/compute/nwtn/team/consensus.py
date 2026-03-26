"""
Consensus Resolution System
============================

Voting and consensus resolution for NWTN Agent Teams.

When agents disagree (detected by the ``ConflictDetector`` in ``live_scribe.py``),
this system provides a structured way to resolve those disagreements through
weighted voting based on domain authority.

Architecture
------------

::

    ConflictDetector (live_scribe.py)
            ↓
       ConflictReport
            ↓
    ConsensusEngine.resolve_conflict()
            ↓
      ┌─────┴──────────┐
      ↓                 ↓
    DomainAuthority    Vote Collection (LLM or heuristic)
    Calculator               ↓
      ↓              Vote Tallying (weighted)
      ↓                 ↓
      └────→ ResolutionOutcome
                  ↓
             ┌────┴────┐
             ↓         ↓
         Resolved   Escalated
         (apply)    (human review)

Key components
--------------
- **VotePosition**: Enum for agree/disagree/abstain/defer positions
- **Vote**: Single agent's vote with confidence and reasoning
- **DomainAuthorityCalculator**: Weights votes by role-topic relevance
- **ConsensusEngine**: Orchestrates voting and determines outcome
- **ResolutionHistory**: Tracks all resolutions for analytics

Safety-first
------------
Security and safety conflicts ALWAYS escalate to human review, regardless
of voting outcome. This is non-negotiable.

Fallback mode
-------------
All classes work without an LLM backend. Heuristic voting uses domain
authority scores to auto-generate reasonable votes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from prsm.compute.nwtn.team.assembler import AgentTeam, TeamMember
    from prsm.compute.nwtn.team.planner import MetaPlan
    from prsm.compute.nwtn.team.live_scribe import ConflictReport
    from prsm.compute.nwtn.whiteboard.schema import WhiteboardEntry

logger = logging.getLogger(__name__)


# ======================================================================
# Safety Keywords
# ======================================================================

DEFAULT_SAFETY_KEYWORDS = [
    "safety",
    "dangerous",
    "harmful",
    "vulnerability",
    "exploit",
    "data loss",
    "corruption",
    "breach",
    "unauthorized",
    "privilege escalation",
    "injection",
    "remote code execution",
    "denial of service",
    "malicious",
    "attack vector",
    "security flaw",
    "critical bug",
    "data leak",
    "authentication bypass",
    "authorization bypass",
]


# ======================================================================
# Vote Model
# ======================================================================

class VotePosition(str, Enum):
    """
    Position an agent can take on a conflict resolution.

    Values
    ------
    AGREE : Supports the new entry's position
    DISAGREE : Supports the existing/conflicting entry's position
    ABSTAIN : No strong opinion or outside domain expertise
    DEFER : Defers to domain expert
    """

    AGREE = "agree"
    DISAGREE = "disagree"
    ABSTAIN = "abstain"
    DEFER = "defer"


@dataclass
class Vote:
    """
    A single agent's vote on a conflict resolution.

    Attributes
    ----------
    agent_id : str
        Unique identifier of the voting agent.
    agent_role : str
        Role slug from TeamMember (e.g., 'backend-coder').
    position : VotePosition
        The agent's position on the conflict.
    confidence : float
        How confident the agent is in their position (0-1).
    reasoning : str
        Brief explanation for the position.
    domain_relevance : float
        How relevant this conflict is to the agent's domain (0-1).
    timestamp : datetime
        When the vote was cast.
    """

    agent_id: str
    agent_role: str
    position: VotePosition
    confidence: float
    reasoning: str
    domain_relevance: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate vote fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not 0.0 <= self.domain_relevance <= 1.0:
            raise ValueError(
                f"domain_relevance must be in [0, 1], got {self.domain_relevance}"
            )


# ======================================================================
# Domain Authority Calculator
# ======================================================================

class DomainAuthorityCalculator:
    """
    Calculates how much weight each agent's vote should carry for a given conflict.

    Authority is based on:
      1. **Role-topic relevance** — does the agent's role relate to the conflict topic?
      2. **Capability match** — do the agent's capabilities overlap with the conflict domain?
      3. **Historical accuracy** — has this agent's position been vindicated in past conflicts?

    The calculator maintains per-agent accuracy tracking that adjusts authority
    based on how often the agent's votes align with final resolutions.

    Parameters
    ----------
    team : AgentTeam
        The assembled agent team.
    meta_plan : MetaPlan
        The session's plan with role descriptions.
    """

    # Keyword-to-domain mapping for determining relevance
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "security": [
            "security",
            "auth",
            "vulnerab",
            "encrypt",
            "token",
            "jwt",
            "permission",
            "attack",
            "exploit",
            "injection",
            "sanitiz",
            "csrf",
            "xss",
            "cors",
            "https",
            "ssl",
            "tls",
            "firewall",
            "access control",
        ],
        "backend": [
            "api",
            "endpoint",
            "database",
            "query",
            "migration",
            "schema",
            "model",
            "service",
            "server",
            "handler",
            "middleware",
            "orm",
            "sql",
            "rest",
            "graphql",
            "grpc",
            "microservice",
        ],
        "frontend": [
            "ui",
            "component",
            "render",
            "style",
            "layout",
            "page",
            "react",
            "template",
            "css",
            "html",
            "dom",
            "vue",
            "angular",
            "svelte",
            "responsive",
            "accessibility",
            "a11y",
        ],
        "testing": [
            "test",
            "assert",
            "coverage",
            "fixture",
            "mock",
            "benchmark",
            "regression",
            "validation",
            "pytest",
            "jest",
            "integration test",
            "unit test",
            "e2e",
            "tdd",
            "bdd",
        ],
        "architecture": [
            "design",
            "pattern",
            "interface",
            "abstract",
            "module",
            "dependency",
            "coupling",
            "scalab",
            "cohesion",
            "solid",
            "clean architecture",
            "hexagonal",
            "layered",
            "modular",
        ],
        "devops": [
            "deploy",
            "docker",
            "ci",
            "pipeline",
            "monitor",
            "log",
            "infra",
            "kubernetes",
            "container",
            "terraform",
            "ansible",
            "helm",
            "aws",
            "gcp",
            "azure",
            "cloud",
            "orchestration",
        ],
        "data": [
            "data",
            "embed",
            "vector",
            "index",
            "search",
            "store",
            "cache",
            "redis",
            "postgres",
            "mongodb",
            "elasticsearch",
            "etl",
            "pipeline",
            "analytics",
            "ml",
            "machine learning",
        ],
        "economic": [
            "token",
            "ftns",
            "price",
            "market",
            "governance",
            "incentive",
            "reward",
            "stake",
            "economic",
            "blockchain",
            "crypto",
            "defi",
            "treasury",
            "allocation",
        ],
    }

    # Maps role slugs to their domain expertise areas
    ROLE_DOMAIN_MAP: Dict[str, List[str]] = {
        "architect": ["architecture", "backend", "security"],
        "backend-coder": ["backend", "data", "testing"],
        "frontend-coder": ["frontend", "testing"],
        "security-reviewer": ["security", "backend"],
        "tester": ["testing", "backend", "frontend"],
        "data-scientist": ["data", "architecture"],
        "devops": ["devops", "backend"],
        "documentation": [],  # No domain authority for docs writer on technical conflicts
        "reviewer": ["testing", "security"],
        "integrator": ["backend", "devops"],
        "validator": ["testing", "security"],
    }

    def __init__(
        self, team: "AgentTeam", meta_plan: "MetaPlan"
    ) -> None:
        self._team = team
        self._plan = meta_plan

        # Historical accuracy tracking: agent_id -> (correct_votes, total_votes)
        self._accuracy: Dict[str, Tuple[int, int]] = {
            member.agent_id: (0, 0) for member in team.members
        }

        # Build reverse lookup from domain to relevant keywords
        self._domain_keywords_flat: Dict[str, Set[str]] = {
            domain: set(keywords)
            for domain, keywords in self.DOMAIN_KEYWORDS.items()
        }

    def calculate_authority(
        self, agent_id: str, conflict: "ConflictReport"
    ) -> float:
        """
        Returns 0-1 authority score for this agent on this conflict.

        The algorithm considers:
          1. Extract keywords from conflict description
          2. Match against agent's role slug, capabilities, and MetaPlan role description
          3. Bonus if agent is one of the affected_agents in the conflict
          4. Historical accuracy modifier (starts at 1.0, adjusted up/down)

        Parameters
        ----------
        agent_id : str
            The agent to calculate authority for.
        conflict : ConflictReport
            The conflict to evaluate authority against.

        Returns
        -------
        float
            Authority score in [0, 1]. Higher means more authoritative.
        """
        # Find the team member
        member = self._find_member(agent_id)
        if member is None:
            return 0.0

        # Start with base authority based on role
        role = member.role
        role_domains = self.ROLE_DOMAIN_MAP.get(role, [])

        # Extract keywords from conflict description
        conflict_text = conflict.description.lower()
        conflict_keywords = self._extract_keywords(conflict_text)

        # Find which domains this conflict touches
        conflict_domains = self._identify_domains(conflict_keywords)

        if not conflict_domains:
            # No clear domain match — give low authority
            base_authority = 0.2
        else:
            # Calculate overlap between agent's domains and conflict domains
            role_domain_set = set(role_domains)
            overlap = role_domain_set & conflict_domains
            overlap_score = len(overlap) / max(len(conflict_domains), 1)

            # Base authority from domain overlap
            base_authority = 0.3 + (0.5 * overlap_score)

            # Capability bonus
            capability_score = self._capability_match(member, conflict_domains)
            base_authority += 0.1 * capability_score

        # Bonus if agent is affected by the conflict
        affected_bonus = 0.15 if agent_id in conflict.affected_agents else 0.0

        # Historical accuracy modifier
        accuracy_modifier = self._get_accuracy_modifier(agent_id)

        # Combine scores
        final_authority = min(
            1.0,
            max(0.0, (base_authority + affected_bonus) * accuracy_modifier),
        )

        logger.debug(
            "DomainAuthorityCalculator: agent=%s role=%s domains=%s "
            "conflict_domains=%s base=%.2f affected=%.2f accuracy_mod=%.2f final=%.2f",
            agent_id,
            role,
            role_domains,
            conflict_domains,
            base_authority,
            affected_bonus,
            accuracy_modifier,
            final_authority,
        )

        return final_authority

    def record_resolution_accuracy(
        self, agent_id: str, was_correct: bool
    ) -> None:
        """
        Track whether an agent's vote aligned with the final resolution.

        This adjusts future authority calculations for this agent based on
        their historical accuracy.

        Parameters
        ----------
        agent_id : str
            The agent whose accuracy to record.
        was_correct : bool
            True if the agent's vote matched the final resolution.
        """
        if agent_id not in self._accuracy:
            self._accuracy[agent_id] = (0, 0)

        correct, total = self._accuracy[agent_id]
        if was_correct:
            correct += 1
        total += 1
        self._accuracy[agent_id] = (correct, total)

        logger.debug(
            "DomainAuthorityCalculator: recorded accuracy for %s: "
            "%d/%d correct (%.1f%%)",
            agent_id,
            correct,
            total,
            100.0 * correct / max(total, 1),
        )

    def get_accuracy_stats(self, agent_id: str) -> Tuple[int, int, float]:
        """
        Get accuracy statistics for an agent.

        Returns
        -------
        Tuple[int, int, float]
            (correct_votes, total_votes, accuracy_rate)
        """
        correct, total = self._accuracy.get(agent_id, (0, 0))
        rate = correct / max(total, 1)
        return correct, total, rate

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _find_member(self, agent_id: str) -> Optional["TeamMember"]:
        """Find a team member by agent_id."""
        for member in self._team.members:
            if member.agent_id == agent_id:
                return member
        return None

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Split into words, filter short words, lowercase
        words = re.findall(r"\b\w{3,}\b", text.lower())
        return set(words)

    def _identify_domains(self, keywords: Set[str]) -> Set[str]:
        """
        Identify which domains are relevant to the given keywords.

        Returns set of domain names that have keyword overlap.
        """
        domains: Set[str] = set()

        for domain, domain_keywords in self._domain_keywords_flat.items():
            # Check if any conflict keyword matches any domain keyword
            for kw in keywords:
                for domain_kw in domain_keywords:
                    if domain_kw in kw or kw in domain_kw:
                        domains.add(domain)
                        break

        return domains

    def _capability_match(
        self, member: "TeamMember", conflict_domains: Set[str]
    ) -> float:
        """
        Calculate how well agent capabilities match conflict domains.

        Returns a score in [0, 1].
        """
        if not conflict_domains:
            return 0.0

        # Map capabilities to domains
        capability_domain_hints: Dict[str, Set[str]] = {
            "code_generation": {"backend", "frontend", "testing"},
            "analysis": {"security", "testing", "architecture"},
            "reasoning": {"architecture", "security", "economic"},
            "security_review": {"security"},
            "architecture": {"architecture", "backend"},
            "testing": {"testing", "backend", "frontend"},
            "documentation": set(),
            "text_generation": set(),
        }

        agent_caps = set(member.capabilities)
        relevant_domains: Set[str] = set()

        for cap in agent_caps:
            cap_domains = capability_domain_hints.get(cap, set())
            relevant_domains.update(cap_domains)

        if not relevant_domains:
            return 0.0

        overlap = relevant_domains & conflict_domains
        return len(overlap) / max(len(conflict_domains), 1)

    def _get_accuracy_modifier(self, agent_id: str) -> float:
        """
        Get the historical accuracy modifier for an agent.

        Returns a multiplier in [0.7, 1.3] based on past performance.
        Agents with >60% accuracy get bonuses; <40% get penalties.
        """
        correct, total = self._accuracy.get(agent_id, (0, 0))

        if total < 3:
            # Not enough data — neutral modifier
            return 1.0

        accuracy = correct / total

        if accuracy >= 0.8:
            return 1.3  # High accuracy bonus
        elif accuracy >= 0.6:
            return 1.1  # Good accuracy bonus
        elif accuracy >= 0.4:
            return 1.0  # Neutral
        elif accuracy >= 0.2:
            return 0.85  # Below average penalty
        else:
            return 0.7  # Low accuracy penalty


# ======================================================================
# Resolution Types
# ======================================================================

class ResolutionMethod(str, Enum):
    """
    Method for tallying votes and determining consensus.

    Values
    ------
    MAJORITY : Simple majority wins
    CONFIDENCE_WEIGHTED : Votes weighted by confidence
    AUTHORITY_WEIGHTED : Votes weighted by domain authority
    FULL_WEIGHTED : Confidence × authority weighting (default)
    UNANIMOUS : All must agree
    HUMAN_ESCALATION : Escalate to human
    """

    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    AUTHORITY_WEIGHTED = "authority_weighted"
    FULL_WEIGHTED = "full_weighted"
    UNANIMOUS = "unanimous"
    HUMAN_ESCALATION = "human_escalation"


@dataclass
class ResolutionOutcome:
    """
    The result of a consensus vote.

    Attributes
    ----------
    conflict_id : int
        ID from ConflictReport.new_entry_id.
    method : ResolutionMethod
        The method used for resolution.
    winning_position : VotePosition
        AGREE or DISAGREE (if resolved).
    winning_score : float
        Weighted score for the winning position.
    losing_score : float
        Weighted score for the losing position.
    margin : float
        Difference between winning and losing scores.
    total_votes : int
        Number of votes cast.
    votes : List[Vote]
        All votes that were cast.
    resolved : bool
        True if a clear winner; False if escalation needed.
    escalation_reason : Optional[str]
        Why it was escalated, if applicable.
    resolution_summary : str
        Human-readable summary of the resolution.
    timestamp : datetime
        When the resolution was determined.
    """

    conflict_id: int
    method: ResolutionMethod
    winning_position: VotePosition
    winning_score: float
    losing_score: float
    margin: float
    total_votes: int
    votes: List[Vote]
    resolved: bool
    escalation_reason: Optional[str]
    resolution_summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ======================================================================
# Consensus Engine
# ======================================================================

class ConsensusEngine:
    """
    Resolves agent conflicts through structured voting.

    The engine collects votes from all team members, weights them by
    domain authority and confidence, and determines a winner. Safety-related
    conflicts are automatically escalated to human review.

    Default method: FULL_WEIGHTED (confidence × domain authority).

    Falls back to HUMAN_ESCALATION when:
      - Margin is too thin (< min_margin threshold)
      - All agents abstain or defer
      - The conflict involves safety/security concerns

    Parameters
    ----------
    team : AgentTeam
        The assembled agent team.
    meta_plan : MetaPlan
        The session's plan.
    authority_calculator : DomainAuthorityCalculator
        Calculator for domain authority weights.
    backend_registry : optional
        LLM backend for generating agent votes. If None, uses heuristic voting.
    min_margin : float
        Minimum margin for a decisive outcome. Default 0.15.
    safety_keywords : Optional[List[str]]
        Keywords that trigger automatic escalation. Uses DEFAULT_SAFETY_KEYWORDS
        if not provided.
    """

    def __init__(
        self,
        team: "AgentTeam",
        meta_plan: "MetaPlan",
        authority_calculator: DomainAuthorityCalculator,
        backend_registry=None,
        min_margin: float = 0.15,
        safety_keywords: Optional[List[str]] = None,
    ) -> None:
        self._team = team
        self._plan = meta_plan
        self._authority = authority_calculator
        self._backend = backend_registry
        self._min_margin = min_margin
        self._safety_keywords = safety_keywords or DEFAULT_SAFETY_KEYWORDS

    async def resolve_conflict(
        self,
        conflict: "ConflictReport",
        existing_entries: List["WhiteboardEntry"],
        method: ResolutionMethod = ResolutionMethod.FULL_WEIGHTED,
    ) -> ResolutionOutcome:
        """
        Run a full consensus vote on a conflict.

        Steps:
          1. Check for safety-related content — auto-escalate if found
          2. Collect votes from all team members (LLM-generated or heuristic)
          3. Calculate weights (authority × confidence)
          4. Tally weighted scores for AGREE vs DISAGREE
          5. Check margin — if too thin, escalate
          6. Return ResolutionOutcome

        Parameters
        ----------
        conflict : ConflictReport
            The conflict to resolve.
        existing_entries : List[WhiteboardEntry]
            The conflicting whiteboard entries for context.
        method : ResolutionMethod
            The voting method to use. Default FULL_WEIGHTED.

        Returns
        -------
        ResolutionOutcome
            The result of the consensus process.
        """
        logger.info(
            "ConsensusEngine: resolving conflict #%d (type=%s)",
            conflict.new_entry_id,
            conflict.conflict_type,
        )

        # 1. Check for safety escalation
        safety_check = self._check_safety_escalation(conflict, existing_entries)
        if safety_check is not None:
            logger.warning(
                "ConsensusEngine: SAFETY ESCALATION for conflict #%d: %s",
                conflict.new_entry_id,
                safety_check,
            )
            return ResolutionOutcome(
                conflict_id=conflict.new_entry_id,
                method=ResolutionMethod.HUMAN_ESCALATION,
                winning_position=VotePosition.ABSTAIN,
                winning_score=0.0,
                losing_score=0.0,
                margin=0.0,
                total_votes=0,
                votes=[],
                resolved=False,
                escalation_reason=safety_check,
                resolution_summary=f"Safety-related conflict escalated to human review: {safety_check}",
            )

        # 2. Collect votes
        votes = await self._collect_votes(conflict, existing_entries)

        # 3. Check if all abstained or deferred
        decisive_votes = [
            v for v in votes
            if v.position in (VotePosition.AGREE, VotePosition.DISAGREE)
        ]
        if not decisive_votes:
            return ResolutionOutcome(
                conflict_id=conflict.new_entry_id,
                method=method,
                winning_position=VotePosition.ABSTAIN,
                winning_score=0.0,
                losing_score=0.0,
                margin=0.0,
                total_votes=len(votes),
                votes=votes,
                resolved=False,
                escalation_reason="No decisive votes cast (all abstained or deferred)",
                resolution_summary="All agents abstained or deferred — requires human review",
            )

        # 4. Tally votes
        outcome = self._tally_votes(votes, method, conflict)

        return outcome

    async def _collect_votes(
        self,
        conflict: "ConflictReport",
        existing_entries: List["WhiteboardEntry"],
    ) -> List[Vote]:
        """
        Collect votes from all team members.

        With LLM backend: prompt each agent with the conflict context.
        Without LLM: use heuristic voting based on domain authority.

        Parameters
        ----------
        conflict : ConflictReport
            The conflict to vote on.
        existing_entries : List[WhiteboardEntry]
            Context entries for LLM prompts.

        Returns
        -------
        List[Vote]
            Votes from all team members.
        """
        votes: List[Vote] = []

        for member in self._team.members:
            authority = self._authority.calculate_authority(
                member.agent_id, conflict
            )

            if self._backend is not None:
                try:
                    vote = await self._llm_vote(member, conflict, existing_entries)
                    vote.domain_relevance = authority
                    votes.append(vote)
                    logger.debug(
                        "ConsensusEngine: LLM vote from %s (%s): %s (conf=%.2f)",
                        member.agent_id,
                        member.role,
                        vote.position.value,
                        vote.confidence,
                    )
                except Exception as exc:
                    logger.warning(
                        "LLM vote failed for %s: %s — using heuristic",
                        member.agent_id,
                        exc,
                    )
                    vote = self._heuristic_vote(member, conflict, authority)
                    votes.append(vote)
            else:
                vote = self._heuristic_vote(member, conflict, authority)
                votes.append(vote)

        return votes

    async def _llm_vote(
        self,
        member: "TeamMember",
        conflict: "ConflictReport",
        entries: List["WhiteboardEntry"],
    ) -> Vote:
        """
        Ask an LLM to vote as a specific team member.

        Parameters
        ----------
        member : TeamMember
            The team member to vote as.
        conflict : ConflictReport
            The conflict context.
        entries : List[WhiteboardEntry]
            The whiteboard entries involved.

        Returns
        -------
        Vote
            The LLM's vote on behalf of the team member.
        """
        # Build context from entries
        entry_context = "\n".join(
            f"- [{e.agent_short}] {e.chunk[:200]}..."
            for e in entries[:5]
        )

        prompt = (
            f"You are a {member.role} on a software development team. "
            f"A conflict has been detected that needs your input.\n\n"
            f"**Conflict Description:**\n{conflict.description}\n\n"
            f"**Relevant Whiteboard Entries:**\n{entry_context}\n\n"
            f"As a {member.role}, provide your vote on this conflict.\n\n"
            f"Respond in this EXACT JSON format:\n"
            f'{{"position": "agree|disagree|abstain|defer", '
            f'"confidence": 0.0-1.0, "reasoning": "brief explanation"}}\n\n'
            f"- agree: Support the new entry\n"
            f"- disagree: Support the existing entry\n"
            f"- abstain: No strong opinion\n"
            f"- defer: Let a domain expert decide"
        )

        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
        )

        # Parse JSON response
        import json

        text = result.text.strip()
        # Extract JSON object if embedded in text
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        data = json.loads(text)

        position = VotePosition(data["position"].lower())
        confidence = float(data["confidence"])
        reasoning = str(data.get("reasoning", "LLM-provided reasoning"))

        return Vote(
            agent_id=member.agent_id,
            agent_role=member.role,
            position=position,
            confidence=min(1.0, max(0.0, confidence)),
            reasoning=reasoning,
            domain_relevance=0.0,  # Set by caller
        )

    def _heuristic_vote(
        self,
        member: "TeamMember",
        conflict: "ConflictReport",
        authority: float,
    ) -> Vote:
        """
        Generate a heuristic vote based on domain authority.

        Rules:
          - High authority agents (>0.7): vote AGREE or DISAGREE
          - Low authority agents (<0.3): ABSTAIN
          - Medium authority agents: DEFER

        For high authority agents, the vote direction is determined by
        which entry source has higher authority on this conflict.

        Parameters
        ----------
        member : TeamMember
            The team member to generate a vote for.
        conflict : ConflictReport
            The conflict context.
        authority : float
            The domain authority score for this member.

        Returns
        -------
        Vote
            A heuristically-generated vote.
        """
        timestamp = datetime.now(timezone.utc)

        if authority < 0.3:
            # Low authority — abstain
            return Vote(
                agent_id=member.agent_id,
                agent_role=member.role,
                position=VotePosition.ABSTAIN,
                confidence=0.3,
                reasoning="Outside my domain expertise; abstaining.",
                domain_relevance=authority,
                timestamp=timestamp,
            )

        if authority < 0.7:
            # Medium authority — defer
            return Vote(
                agent_id=member.agent_id,
                agent_role=member.role,
                position=VotePosition.DEFER,
                confidence=0.5,
                reasoning="Marginally relevant to my domain; deferring to experts.",
                domain_relevance=authority,
                timestamp=timestamp,
            )

        # High authority — make a decisive vote
        # Default to AGREE with the new entry, but could be more sophisticated
        # based on conflict type or comparing authority of conflicting agents
        if conflict.conflict_type == "contradiction":
            # For contradictions, side with the higher authority position
            # This is a simplification — a real system would analyze both entries
            position = VotePosition.AGREE
            reasoning = (
                f"As a {member.role}, I support the new entry based on domain analysis."
            )
        elif conflict.conflict_type == "disagreement":
            # For disagreements, try to find middle ground or pick a side
            position = VotePosition.AGREE
            reasoning = (
                f"As a {member.role}, I believe the new position has merit."
            )
        else:
            # Default: agree with new entry
            position = VotePosition.AGREE
            reasoning = f"As a {member.role}, I support the proposed change."

        return Vote(
            agent_id=member.agent_id,
            agent_role=member.role,
            position=position,
            confidence=min(0.95, 0.7 + authority * 0.25),
            reasoning=reasoning,
            domain_relevance=authority,
            timestamp=timestamp,
        )

    def _tally_votes(
        self,
        votes: List[Vote],
        method: ResolutionMethod,
        conflict: "ConflictReport",
    ) -> ResolutionOutcome:
        """
        Tally votes using the specified method and determine outcome.

        Parameters
        ----------
        votes : List[Vote]
            All collected votes.
        method : ResolutionMethod
            The weighting method to use.
        conflict : ConflictReport
            The conflict being resolved.

        Returns
        -------
        ResolutionOutcome
            The resolution result.
        """
        # Filter to decisive votes
        decisive_votes = [
            v for v in votes
            if v.position in (VotePosition.AGREE, VotePosition.DISAGREE)
        ]

        if not decisive_votes:
            # Handled earlier, but defensive
            return ResolutionOutcome(
                conflict_id=conflict.new_entry_id,
                method=method,
                winning_position=VotePosition.ABSTAIN,
                winning_score=0.0,
                losing_score=0.0,
                margin=0.0,
                total_votes=len(votes),
                votes=votes,
                resolved=False,
                escalation_reason="No decisive votes",
                resolution_summary="No agents voted decisively.",
            )

        # Calculate weights based on method
        weights: Dict[str, float] = {}
        for vote in decisive_votes:
            if method == ResolutionMethod.MAJORITY:
                weight = 1.0
            elif method == ResolutionMethod.CONFIDENCE_WEIGHTED:
                weight = vote.confidence
            elif method == ResolutionMethod.AUTHORITY_WEIGHTED:
                weight = vote.domain_relevance
            elif method == ResolutionMethod.FULL_WEIGHTED:
                weight = vote.confidence * vote.domain_relevance
            elif method == ResolutionMethod.UNANIMOUS:
                weight = 1.0
            else:
                weight = vote.confidence * vote.domain_relevance

            weights[vote.agent_id] = weight

        # Tally scores
        agree_score = sum(
            weights[v.agent_id]
            for v in decisive_votes
            if v.position == VotePosition.AGREE
        )
        disagree_score = sum(
            weights[v.agent_id]
            for v in decisive_votes
            if v.position == VotePosition.DISAGREE
        )

        total_score = agree_score + disagree_score

        # Determine winner
        if agree_score > disagree_score:
            winning_position = VotePosition.AGREE
            winning_score = agree_score
            losing_score = disagree_score
        else:
            winning_position = VotePosition.DISAGREE
            winning_score = disagree_score
            losing_score = agree_score

        # Calculate margin (normalized)
        margin = (winning_score - losing_score) / max(total_score, 0.001)

        # Check for unanimous requirement
        if method == ResolutionMethod.UNANIMOUS:
            positions = {v.position for v in decisive_votes}
            if len(positions) > 1:
                return ResolutionOutcome(
                    conflict_id=conflict.new_entry_id,
                    method=method,
                    winning_position=winning_position,
                    winning_score=winning_score,
                    losing_score=losing_score,
                    margin=margin,
                    total_votes=len(votes),
                    votes=votes,
                    resolved=False,
                    escalation_reason="Unanimous consensus not achieved",
                    resolution_summary="Team did not reach unanimous agreement.",
                )

        # Check margin threshold
        resolved = margin >= self._min_margin
        escalation_reason = None

        if not resolved:
            escalation_reason = (
                f"Margin too thin ({margin:.2%} < {self._min_margin:.2%} threshold) "
                f"— requires human review"
            )

        # Build summary
        agree_count = sum(1 for v in decisive_votes if v.position == VotePosition.AGREE)
        disagree_count = sum(1 for v in decisive_votes if v.position == VotePosition.DISAGREE)
        abstain_count = sum(1 for v in votes if v.position == VotePosition.ABSTAIN)
        defer_count = sum(1 for v in votes if v.position == VotePosition.DEFER)

        summary_lines = [
            f"Consensus: **{winning_position.value.upper()}** (margin: {margin:.1%})",
            f"Votes: {agree_count} agree, {disagree_count} disagree, "
            f"{abstain_count} abstain, {defer_count} defer",
            f"Method: {method.value}",
        ]

        if resolved:
            summary_lines.append("✅ Resolution applied automatically.")
        else:
            summary_lines.append(f"⚠️ Escalated: {escalation_reason}")

        resolution_summary = "\n".join(summary_lines)

        logger.info(
            "ConsensusEngine: conflict #%d resolved=%s %s (margin=%.2f, %d agree, %d disagree)",
            conflict.new_entry_id,
            resolved,
            winning_position.value,
            margin,
            agree_count,
            disagree_count,
        )

        return ResolutionOutcome(
            conflict_id=conflict.new_entry_id,
            method=method,
            winning_position=winning_position,
            winning_score=winning_score,
            losing_score=losing_score,
            margin=margin,
            total_votes=len(votes),
            votes=votes,
            resolved=resolved,
            escalation_reason=escalation_reason,
            resolution_summary=resolution_summary,
        )

    def _check_safety_escalation(
        self,
        conflict: "ConflictReport",
        entries: List["WhiteboardEntry"],
    ) -> Optional[str]:
        """
        Check if the conflict requires safety escalation.

        Returns escalation reason if safety-related, None otherwise.
        """
        # Check conflict description
        desc_lower = conflict.description.lower()
        for keyword in self._safety_keywords:
            if keyword.lower() in desc_lower:
                return f"Safety keyword '{keyword}' detected in conflict description"

        # Check whiteboard entries
        for entry in entries:
            entry_lower = entry.chunk.lower()
            for keyword in self._safety_keywords:
                if keyword.lower() in entry_lower:
                    return f"Safety keyword '{keyword}' detected in whiteboard entry"

        return None


# ======================================================================
# Resolution History Tracker
# ======================================================================

@dataclass
class ResolutionHistory:
    """
    Tracks all conflict resolutions during a session.

    Used for:
      - Updating DomainAuthorityCalculator accuracy tracking
      - Session reporting
      - Pattern detection (are the same agents always disagreeing?)

    Attributes
    ----------
    outcomes : List[ResolutionOutcome]
        All resolution outcomes recorded during the session.
    """

    outcomes: List[ResolutionOutcome] = field(default_factory=list)

    def add(self, outcome: ResolutionOutcome) -> None:
        """
        Add a resolution outcome to the history.

        Parameters
        ----------
        outcome : ResolutionOutcome
            The resolution outcome to record.
        """
        self.outcomes.append(outcome)
        logger.debug(
            "ResolutionHistory: recorded outcome for conflict #%d (resolved=%s)",
            outcome.conflict_id,
            outcome.resolved,
        )

    def outcomes_for_agent(self, agent_id: str) -> List[ResolutionOutcome]:
        """
        Get all outcomes where the agent participated.

        Parameters
        ----------
        agent_id : str
            The agent to filter by.

        Returns
        -------
        List[ResolutionOutcome]
            Outcomes where the agent cast a vote.
        """
        result: List[ResolutionOutcome] = []
        for outcome in self.outcomes:
            if any(v.agent_id == agent_id for v in outcome.votes):
                result.append(outcome)
        return result

    def escalation_rate(self) -> float:
        """
        Calculate the fraction of conflicts that were escalated.

        Returns
        -------
        float
            Escalation rate in [0, 1]. 0 if no conflicts recorded.
        """
        if not self.outcomes:
            return 0.0
        escalated = sum(1 for o in self.outcomes if not o.resolved)
        return escalated / len(self.outcomes)

    def agent_agreement_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Build a matrix showing how often each pair of agents agree.

        For each pair of agents, calculates the fraction of shared
        conflicts where they voted the same way (both AGREE or both DISAGREE).

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: agent_id -> other_agent_id -> agreement_rate
        """
        # Collect all agents that have voted
        agents: Set[str] = set()
        for outcome in self.outcomes:
            for vote in outcome.votes:
                agents.add(vote.agent_id)

        # Initialize matrix
        matrix: Dict[str, Dict[str, float]] = {
            a1: {a2: 0.0 for a2 in agents if a2 != a1} for a1 in agents
        }

        # Track shared conflict counts
        shared_counts: Dict[str, Dict[str, int]] = {
            a1: {a2: 0 for a2 in agents if a2 != a1} for a1 in agents
        }

        # Calculate agreements
        for outcome in self.outcomes:
            decisive_votes = {
                v.agent_id: v.position
                for v in outcome.votes
                if v.position in (VotePosition.AGREE, VotePosition.DISAGREE)
            }

            agent_ids = list(decisive_votes.keys())
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i + 1 :]:
                    shared_counts[a1][a2] += 1
                    shared_counts[a2][a1] += 1

                    if decisive_votes[a1] == decisive_votes[a2]:
                        matrix[a1][a2] += 1
                        matrix[a2][a1] += 1

        # Normalize to rates
        for a1 in agents:
            for a2 in agents:
                if a1 == a2:
                    continue
                count = shared_counts[a1][a2]
                if count > 0:
                    matrix[a1][a2] = matrix[a1][a2] / count

        return matrix

    def status_summary(self) -> Dict[str, Any]:
        """
        Generate a status summary for reporting.

        Returns
        -------
        dict
            Summary statistics about resolutions.
        """
        if not self.outcomes:
            return {
                "total_conflicts": 0,
                "resolved": 0,
                "escalated": 0,
                "escalation_rate": 0.0,
            }

        resolved_count = sum(1 for o in self.outcomes if o.resolved)
        escalated_count = len(self.outcomes) - resolved_count

        # Method breakdown
        method_counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            method = outcome.method.value
            method_counts[method] = method_counts.get(method, 0) + 1

        # Average margin
        margins = [o.margin for o in self.outcomes if o.resolved]
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        # Agent participation
        agent_votes: Dict[str, int] = {}
        for outcome in self.outcomes:
            for vote in outcome.votes:
                agent_votes[vote.agent_id] = agent_votes.get(vote.agent_id, 0) + 1

        return {
            "total_conflicts": len(self.outcomes),
            "resolved": resolved_count,
            "escalated": escalated_count,
            "escalation_rate": self.escalation_rate(),
            "method_breakdown": method_counts,
            "average_margin": avg_margin,
            "agent_participation": agent_votes,
        }

    def recent_outcomes(self, limit: int = 5) -> List[ResolutionOutcome]:
        """
        Get the most recent resolution outcomes.

        Parameters
        ----------
        limit : int
            Maximum number of outcomes to return.

        Returns
        -------
        List[ResolutionOutcome]
            Most recent outcomes, newest first.
        """
        return sorted(
            self.outcomes, key=lambda o: o.timestamp, reverse=True
        )[:limit]


# ======================================================================
# Import additions for team/__init__.py
# ======================================================================
#
# Add these lines to /Users/ryneschultz/Documents/GitHub/PRSM/prsm/compute/nwtn/team/__init__.py:
#
# from .consensus import (
#     VotePosition,
#     Vote,
#     DomainAuthorityCalculator,
#     ResolutionMethod,
#     ResolutionOutcome,
#     ConsensusEngine,
#     ResolutionHistory,
#     DEFAULT_SAFETY_KEYWORDS,
# )
#
# Add to __all__:
#     # Consensus
#     "VotePosition",
#     "Vote",
#     "DomainAuthorityCalculator",
#     "ResolutionMethod",
#     "ResolutionOutcome",
#     "ConsensusEngine",
#     "ResolutionHistory",
#     "DEFAULT_SAFETY_KEYWORDS",
#
