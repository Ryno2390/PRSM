"""
Team Assembler
==============

Maps the ``AgentRole`` requirements from a ``MetaPlan`` to available agents
on the PRSM network (via ``ModelRegistry`` / ``AgentRegistry``), or to
locally-configured fallback agents.

The assembler produces an ``AgentTeam`` — the complete roster of team members,
each with a role, a resolved model/agent identifier, and an assigned git branch
name.  The ``BranchManager`` (``branch_manager.py``) creates the actual git
branches from this roster.

Selection algorithm
-------------------
For each required role (ordered by priority):
  1. Query the ``ModelRegistry`` for models whose ``specialization_domains``
     or ``capabilities`` intersect with the role's ``capabilities_required``.
  2. Score candidates by: capability_match × performance_score × availability.
  3. Assign the top-scoring model.
  4. If the registry has no candidates, fall back to the ``DEFAULT_AGENT_MAP``.

Fallback agents
---------------
The ``DEFAULT_AGENT_MAP`` provides sensible defaults keyed by common role
slugs.  These reference the PRSM-supported backends (Anthropic, OpenAI,
local Ollama) so the team always has a usable assignment even before the
PRSM network is fully populated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .planner import AgentRole, MetaPlan

logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================

@dataclass
class TeamMember:
    """A single member of the assembled Agent Team."""
    role: str
    """Role slug (e.g. 'backend-coder', 'security-reviewer')."""

    agent_id: str
    """Unique identifier for the resolved agent/model."""

    agent_name: str
    """Human-readable name."""

    model_id: str
    """Model identifier passed to the backend (e.g. 'claude-opus-4-6')."""

    branch_name: str
    """Git branch assigned to this team member (set by BranchManager)."""

    capabilities: List[str] = field(default_factory=list)

    score: float = 1.0
    """Selection score (capability_match × performance × availability)."""

    source: str = "registry"
    """'registry' | 'fallback' — how this member was resolved."""

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTeam:
    """The assembled team for a working session."""
    session_id: str
    meta_plan: MetaPlan
    members: List[TeamMember] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def member_by_role(self, role: str) -> Optional[TeamMember]:
        for m in self.members:
            if m.role == role:
                return m
        return None

    def branch_names(self) -> List[str]:
        return [m.branch_name for m in self.members]

    def summary(self) -> str:
        lines = [f"Agent Team — Session {self.session_id} ({len(self.members)} members)"]
        for m in self.members:
            lines.append(
                f"  [{m.role}] {m.agent_name} ({m.model_id}) "
                f"→ branch: {m.branch_name} [source: {m.source}]"
            )
        return "\n".join(lines)


# ======================================================================
# Default fallback agents
# ======================================================================

# Maps common role slugs → (agent_name, model_id, capabilities)
_DEFAULT_AGENT_MAP: Dict[str, Tuple[str, str, List[str]]] = {
    "architect": (
        "NWTN Architect",
        "anthropic/claude-opus-4-6",
        ["reasoning", "analysis"],
    ),
    "backend-coder": (
        "NWTN Coder",
        "anthropic/claude-sonnet-4-6",
        ["code_generation", "reasoning"],
    ),
    "frontend-coder": (
        "NWTN Frontend",
        "anthropic/claude-sonnet-4-6",
        ["code_generation"],
    ),
    "security-reviewer": (
        "NWTN Security",
        "anthropic/claude-opus-4-6",
        ["analysis", "reasoning"],
    ),
    "tester": (
        "NWTN Tester",
        "anthropic/claude-sonnet-4-6",
        ["code_generation", "analysis"],
    ),
    "data-scientist": (
        "NWTN Data Scientist",
        "anthropic/claude-sonnet-4-6",
        ["analysis", "reasoning"],
    ),
    "devops": (
        "NWTN DevOps",
        "anthropic/claude-sonnet-4-6",
        ["code_generation", "analysis"],
    ),
    "documentation": (
        "NWTN Docs Writer",
        "anthropic/claude-haiku-4-5",
        ["text_generation"],
    ),
}

_GENERIC_FALLBACK = (
    "NWTN Generalist",
    "anthropic/claude-sonnet-4-6",
    ["reasoning", "analysis", "code_generation"],
)


# ======================================================================
# TeamAssembler
# ======================================================================

class TeamAssembler:
    """
    Resolves ``AgentRole`` requirements to concrete ``TeamMember`` assignments.

    Parameters
    ----------
    model_registry : optional
        A ``ModelRegistry`` instance for network-based agent discovery.
        If ``None``, the fallback map is used for all assignments.
    date_suffix : str
        Date string appended to branch names (``YYYYMMDD``).
        Defaults to today's date.
    """

    def __init__(
        self,
        model_registry=None,
        date_suffix: Optional[str] = None,
    ) -> None:
        self._registry = model_registry
        self._date_suffix = date_suffix or datetime.now(timezone.utc).strftime("%Y%m%d")

    async def assemble(self, meta_plan: MetaPlan) -> AgentTeam:
        """
        Assemble a team from the MetaPlan's required roles.

        Returns
        -------
        AgentTeam
            Each member has a role, resolved model, and assigned branch name.
            Branch names follow the convention ``agent/<role>-<YYYYMMDD>``.
        """
        members: List[TeamMember] = []

        for role_def in meta_plan.roles_by_priority():
            member = await self._resolve_role(role_def)
            members.append(member)

        team = AgentTeam(
            session_id=meta_plan.session_id,
            meta_plan=meta_plan,
            members=members,
        )
        logger.info(
            "TeamAssembler: assembled %d-member team for session %s",
            len(members), meta_plan.session_id,
        )
        return team

    async def _resolve_role(self, role_def: AgentRole) -> TeamMember:
        """Resolve a single AgentRole to a TeamMember."""
        branch_name = f"agent/{role_def.role}-{self._date_suffix}"

        # Try registry first
        if self._registry is not None:
            candidate = await self._query_registry(role_def)
            if candidate:
                agent_id, agent_name, model_id, capabilities, score = candidate
                return TeamMember(
                    role=role_def.role,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    model_id=model_id,
                    branch_name=branch_name,
                    capabilities=capabilities,
                    score=score,
                    source="registry",
                )

        # Fallback: match by role slug
        return self._fallback_member(role_def, branch_name)

    async def _query_registry(
        self, role_def: AgentRole
    ) -> Optional[Tuple[str, str, str, List[str], float]]:
        """
        Query the ModelRegistry for the best match for *role_def*.

        Returns (agent_id, agent_name, model_id, capabilities, score) or None.
        """
        try:
            required_caps = set(role_def.capabilities_required)
            candidates = []

            for model_id, details in self._registry.models.items():
                model_caps = {c.value for c in details.capabilities}
                overlap = required_caps & model_caps
                if not overlap and required_caps:
                    continue

                # Score: capability overlap fraction × performance × availability
                cap_score = len(overlap) / max(len(required_caps), 1)
                score = cap_score * details.performance_score * details.availability

                # Bonus if specialisation domains match role slug keywords
                role_words = set(role_def.role.replace("-", " ").split())
                for domain in details.specialization_domains:
                    if any(w in domain.lower() for w in role_words):
                        score *= 1.2
                        break

                candidates.append((
                    model_id,
                    details.name,
                    model_id,
                    [c.value for c in details.capabilities],
                    score,
                ))

            if not candidates:
                return None

            candidates.sort(key=lambda c: c[4], reverse=True)
            best = candidates[0]
            logger.debug(
                "Registry selected %s for role %s (score=%.3f)",
                best[1], role_def.role, best[4],
            )
            return best

        except Exception as exc:
            logger.warning("Registry query failed: %s", exc)
            return None

    def _fallback_member(self, role_def: AgentRole, branch_name: str) -> TeamMember:
        """Return a fallback TeamMember using the default agent map."""
        # Try exact match, then partial match on slug words
        slug = role_def.role
        default = _DEFAULT_AGENT_MAP.get(slug)

        if default is None:
            # Partial match: find any key that shares a word with the role slug
            role_words = set(slug.split("-"))
            for key, val in _DEFAULT_AGENT_MAP.items():
                key_words = set(key.split("-"))
                if role_words & key_words:
                    default = val
                    break

        if default is None:
            default = _GENERIC_FALLBACK

        agent_name, model_id, capabilities = default
        import uuid as _uuid
        return TeamMember(
            role=slug,
            agent_id=str(_uuid.uuid4()),
            agent_name=agent_name,
            model_id=model_id,
            branch_name=branch_name,
            capabilities=capabilities,
            score=0.8,
            source="fallback",
        )
