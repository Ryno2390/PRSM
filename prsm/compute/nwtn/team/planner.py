"""
Meta-Plan Generator
===================

Transforms a ``ProjectBrief`` (from the interview) into a structured
``MetaPlan`` — the "north star" document that anchors the Agent Team's
work throughout the session.

The MetaPlan is:
  - Written to the Active Whiteboard at session start (every agent reads it)
  - Used by the ``CheckpointReviewer`` to evaluate whether branch diffs are
    coherent with the original intent
  - Appended to the Project Ledger as the session's opening entry

Structure
---------
A MetaPlan contains:
  - **Objective** — one clear statement of what success looks like
  - **Milestones** — 3–7 concrete checkpoints with merge criteria
  - **Required roles** — specialist agent roles needed + their capabilities
  - **Success criteria** — measurable outcomes
  - **Constraints** — hard boundaries the team must not cross

Fallback
--------
If no LLM backend is available, a ``TemplatePlanner`` constructs a minimal
but well-formed ``MetaPlan`` directly from the ``ProjectBrief`` fields.
This ensures the pipeline always has a MetaPlan to work with.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .interview import ProjectBrief, _extract_json_object, _as_list

logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================

@dataclass
class Milestone:
    """A discrete checkpoint in the project plan."""
    title: str
    description: str
    merge_criteria: List[str] = field(default_factory=list)
    """Conditions that must be met before an agent's branch is merged at this point."""
    estimated_effort: str = "medium"
    """'small' | 'medium' | 'large'"""


@dataclass
class AgentRole:
    """A specialist role required by the MetaPlan."""
    role: str
    """Short identifier, e.g. 'security-auditor', 'backend-coder', 'architect'."""
    description: str
    capabilities_required: List[str] = field(default_factory=list)
    """Maps to ``ModelCapability`` enum values in the model registry."""
    priority: int = 1
    """Lower = higher priority.  Used by the assembler to order assignments."""


@dataclass
class MetaPlan:
    """
    The structured project plan produced by the MetaPlanner.

    Stored on the Active Whiteboard and checked against all branch diffs
    during checkpoint review.
    """
    session_id: str
    title: str
    objective: str
    milestones: List[Milestone] = field(default_factory=list)
    required_roles: List[AgentRole] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_assisted: bool = True
    source_brief: Optional[ProjectBrief] = None

    def to_whiteboard_entry(self) -> str:
        """
        Render the MetaPlan as a whiteboard-ready string.

        This is the first entry written to the Active Whiteboard at session
        start — every agent reads it before beginning work.
        """
        lines = [
            f"META-PLAN: {self.title}",
            f"Objective: {self.objective}",
            "",
            "Milestones:",
        ]
        for i, m in enumerate(self.milestones, 1):
            lines.append(f"  {i}. {m.title} — {m.description}")
            for c in m.merge_criteria:
                lines.append(f"       ✓ {c}")
        lines += ["", "Required roles:"]
        for r in self.required_roles:
            lines.append(f"  - {r.role}: {r.description}")
        if self.success_criteria:
            lines += ["", "Success criteria:"]
            for s in self.success_criteria:
                lines.append(f"  - {s}")
        if self.constraints:
            lines += ["", "Constraints:"]
            for c in self.constraints:
                lines.append(f"  - {c}")
        return "\n".join(lines)

    def roles_by_priority(self) -> List[AgentRole]:
        """Return roles sorted by priority (ascending)."""
        return sorted(self.required_roles, key=lambda r: r.priority)

    def milestone_count(self) -> int:
        return len(self.milestones)


# ======================================================================
# MetaPlanner
# ======================================================================

class MetaPlanner:
    """
    Generates a ``MetaPlan`` from a ``ProjectBrief``.

    Parameters
    ----------
    backend_registry : optional
        A ``BackendRegistry`` for LLM-powered plan generation.
        If ``None`` the template planner is used.
    """

    def __init__(self, backend_registry=None) -> None:
        self._backend = backend_registry

    async def generate(self, brief: ProjectBrief) -> MetaPlan:
        """
        Generate a ``MetaPlan`` for the given ``ProjectBrief``.

        Attempts LLM generation first; falls back to the template planner.
        """
        if self._backend is not None:
            try:
                return await self._llm_generate(brief)
            except Exception as exc:
                logger.warning(
                    "LLM meta-plan generation failed (%s); using template planner", exc
                )
        return _template_plan(brief)

    async def _llm_generate(self, brief: ProjectBrief) -> MetaPlan:
        prompt = (
            "You are NWTN, a project planning AI. Generate a structured MetaPlan "
            "for the following project brief.\n\n"
            f"{brief.to_prompt_block()}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  title (str): short project title\n"
            "  objective (str): one-sentence success statement\n"
            "  milestones (list): each has title, description, merge_criteria (list), "
            "estimated_effort ('small'|'medium'|'large')\n"
            "  required_roles (list): each has role (slug), description, "
            "capabilities_required (list from: code_generation, analysis, reasoning, "
            "security_review, architecture, testing, documentation), priority (int 1-5)\n"
            "  success_criteria (list of strings)\n"
            "  constraints (list of strings)\n\n"
            "Generate 3–6 milestones and 2–5 required roles. Be specific."
        )
        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.3,
        )
        data = _extract_json_object(result.text.strip())
        if not data:
            raise ValueError("LLM returned no parseable MetaPlan JSON")

        milestones = [
            Milestone(
                title=m.get("title", "Milestone"),
                description=m.get("description", ""),
                merge_criteria=_as_list(m.get("merge_criteria")),
                estimated_effort=m.get("estimated_effort", "medium"),
            )
            for m in data.get("milestones", [])
        ]
        roles = [
            AgentRole(
                role=r.get("role", "generalist"),
                description=r.get("description", ""),
                capabilities_required=_as_list(r.get("capabilities_required")),
                priority=int(r.get("priority", 1)),
            )
            for r in data.get("required_roles", [])
        ]

        return MetaPlan(
            session_id=brief.session_id,
            title=data.get("title", brief.goal[:60]),
            objective=data.get("objective", brief.goal),
            milestones=milestones,
            required_roles=roles,
            success_criteria=_as_list(data.get("success_criteria")) or brief.success_criteria,
            constraints=_as_list(data.get("constraints")) or brief.constraints,
            llm_assisted=True,
            source_brief=brief,
        )


# ======================================================================
# Template planner (no LLM)
# ======================================================================

def _template_plan(brief: ProjectBrief) -> MetaPlan:
    """
    Build a minimal but well-formed MetaPlan directly from the ProjectBrief.
    Used when no LLM backend is available.
    """
    # Derive milestones from success criteria, or use a generic three-stage plan
    if brief.success_criteria:
        milestones = [
            Milestone(
                title=f"Checkpoint {i+1}",
                description=criterion,
                merge_criteria=[f"Criterion met: {criterion}"],
                estimated_effort="medium",
            )
            for i, criterion in enumerate(brief.success_criteria[:5])
        ]
    else:
        milestones = [
            Milestone(
                title="Foundation",
                description="Core architecture and scaffolding in place",
                merge_criteria=["Core modules importable", "Basic tests passing"],
                estimated_effort="medium",
            ),
            Milestone(
                title="Implementation",
                description="Primary features implemented",
                merge_criteria=["Feature tests passing", "No regressions"],
                estimated_effort="large",
            ),
            Milestone(
                title="Hardening",
                description="Security review, edge cases, documentation",
                merge_criteria=["Security review complete", "Docs updated"],
                estimated_effort="small",
            ),
        ]

    # Derive roles from preferred_roles in brief, or use sensible defaults
    if brief.preferred_roles:
        roles = [
            AgentRole(
                role=_slugify(r),
                description=r,
                capabilities_required=["reasoning"],
                priority=i + 1,
            )
            for i, r in enumerate(brief.preferred_roles[:5])
        ]
    else:
        roles = _DEFAULT_ROLES

    tech = (
        f" using {', '.join(brief.technology_stack)}"
        if brief.technology_stack
        else ""
    )

    return MetaPlan(
        session_id=brief.session_id,
        title=brief.goal[:60].rstrip("."),
        objective=f"Successfully complete: {brief.goal}{tech}",
        milestones=milestones,
        required_roles=roles,
        success_criteria=brief.success_criteria
        or ["Project goal achieved", "All tests pass"],
        constraints=brief.constraints,
        llm_assisted=False,
        source_brief=brief,
    )


_DEFAULT_ROLES: List[AgentRole] = [
    AgentRole(
        role="architect",
        description="High-level design and technical decision-making",
        capabilities_required=["reasoning", "analysis"],
        priority=1,
    ),
    AgentRole(
        role="backend-coder",
        description="Core implementation",
        capabilities_required=["code_generation"],
        priority=2,
    ),
    AgentRole(
        role="security-reviewer",
        description="Security audit and vulnerability assessment",
        capabilities_required=["analysis", "reasoning"],
        priority=3,
    ),
    AgentRole(
        role="tester",
        description="Test coverage and quality assurance",
        capabilities_required=["code_generation", "analysis"],
        priority=4,
    ),
]


def _slugify(text: str) -> str:
    """Convert text to a slug safe for use as a branch-name component."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
