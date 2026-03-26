"""
Skills Bridge
=============

Maps between PRSM's model capability vocabulary and OpenClaw's Skills
Registry vocabulary.

OpenClaw agents use "Skills" — named sets of permissions and instructions
that give an agent access to specific tools (file manager, code executor,
web search, etc.).  PRSM describes the same agents using ``ModelCapability``
enum values (code_generation, analysis, reasoning, …).

The bridge translates in both directions so that:
  - The ``TeamAssembler`` can register PRSM specialist agents as OpenClaw
    skills, making them available to the team.
  - The ``TeamAssembler`` can search the OpenClaw Skills Registry using
    PRSM capability names as query terms.

OpenClaw skill specification (the format OpenClaw expects)
----------------------------------------------------------
{
    "skill_id":    "prsm-coder-v1",
    "name":        "PRSM Backend Coder",
    "description": "Python/FastAPI backend implementation specialist",
    "model":       "anthropic/claude-sonnet-4-6",
    "tools":       ["code_executor", "file_manager", "web_search"],
    "metadata":    {"prsm_role": "backend-coder", "source": "prsm_registry"}
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================

@dataclass
class OpenClawSkillSpec:
    """Minimal skill specification understood by the OpenClaw Skills Registry."""
    skill_id: str
    name: str
    description: str
    model: str
    tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id":    self.skill_id,
            "name":        self.name,
            "description": self.description,
            "model":       self.model,
            "tools":       self.tools,
            "metadata":    self.metadata,
        }


# ======================================================================
# Capability ↔ Tools mapping
# ======================================================================

# PRSM ModelCapability value → OpenClaw tool names
_CAPABILITY_TO_TOOLS: Dict[str, List[str]] = {
    "code_generation":  ["code_executor", "file_manager", "web_search"],
    "analysis":         ["web_search", "file_reader", "python_executor"],
    "reasoning":        ["web_search", "file_reader"],
    "text_generation":  ["file_manager"],
    "classification":   ["python_executor"],
    "extraction":       ["file_reader", "python_executor"],
    "summarization":    ["file_reader"],
    "translation":      [],
    "security_review":  ["file_reader", "python_executor"],
    "architecture":     ["file_reader", "web_search"],
    "testing":          ["code_executor", "file_manager"],
    "documentation":    ["file_manager", "file_reader"],
}

# OpenClaw tool name → PRSM capability values
_TOOL_TO_CAPABILITIES: Dict[str, List[str]] = {
    "code_executor":   ["code_generation", "testing"],
    "file_manager":    ["code_generation", "text_generation", "documentation"],
    "file_reader":     ["analysis", "reasoning", "extraction", "summarization"],
    "web_search":      ["analysis", "reasoning", "code_generation"],
    "python_executor": ["analysis", "classification", "extraction"],
    "database_query":  ["analysis", "extraction"],
    "api_client":      ["code_generation", "analysis"],
}

# PRSM role slug → preferred OpenClaw tool set
_ROLE_TO_TOOLS: Dict[str, List[str]] = {
    "architect":         ["web_search", "file_reader"],
    "backend-coder":     ["code_executor", "file_manager", "web_search"],
    "frontend-coder":    ["code_executor", "file_manager", "web_search"],
    "security-reviewer": ["file_reader", "python_executor"],
    "tester":            ["code_executor", "file_manager"],
    "data-scientist":    ["python_executor", "file_reader", "web_search"],
    "devops":            ["code_executor", "file_manager", "api_client"],
    "documentation":     ["file_manager", "file_reader"],
}


# ======================================================================
# SkillsBridge
# ======================================================================

class SkillsBridge:
    """
    Translates between PRSM model registry entries and OpenClaw skill specs.

    Parameters
    ----------
    extra_mappings : dict, optional
        Additional ``{capability: [tool, …]}`` mappings to merge with the
        built-in defaults.
    """

    def __init__(self, extra_mappings: Optional[Dict[str, List[str]]] = None) -> None:
        self._cap_to_tools = dict(_CAPABILITY_TO_TOOLS)
        if extra_mappings:
            self._cap_to_tools.update(extra_mappings)

    # ------------------------------------------------------------------
    # PRSM → OpenClaw
    # ------------------------------------------------------------------

    def team_member_to_skill(self, member) -> OpenClawSkillSpec:
        """
        Convert a ``TeamMember`` (from the assembler) to an OpenClaw skill spec.

        Parameters
        ----------
        member : TeamMember

        Returns
        -------
        OpenClawSkillSpec
        """
        tools = self._tools_for_role(member.role)
        return OpenClawSkillSpec(
            skill_id=f"prsm-{member.role}-{member.agent_id[:8]}",
            name=member.agent_name,
            description=f"PRSM specialist: {member.role}",
            model=member.model_id,
            tools=tools,
            metadata={
                "prsm_role":       member.role,
                "prsm_agent_id":   member.agent_id,
                "prsm_branch":     member.branch_name,
                "source":          "prsm_registry",
            },
        )

    def capabilities_to_tools(self, capabilities: List[str]) -> List[str]:
        """Return the union of OpenClaw tools for a list of PRSM capabilities."""
        tools: set = set()
        for cap in capabilities:
            tools.update(self._cap_to_tools.get(cap, []))
        return sorted(tools)

    def role_to_skill_spec(
        self,
        role: str,
        model_id: str,
        agent_name: str = "",
        agent_id: str = "",
    ) -> OpenClawSkillSpec:
        """Build a skill spec directly from a role slug and model ID."""
        tools = self._tools_for_role(role)
        return OpenClawSkillSpec(
            skill_id=f"prsm-{role}",
            name=agent_name or f"PRSM {role.replace('-', ' ').title()}",
            description=f"PRSM specialist agent for role: {role}",
            model=model_id,
            tools=tools,
            metadata={"prsm_role": role, "prsm_agent_id": agent_id or ""},
        )

    # ------------------------------------------------------------------
    # OpenClaw → PRSM
    # ------------------------------------------------------------------

    def tools_to_capabilities(self, tools: List[str]) -> List[str]:
        """Return PRSM capability values implied by a list of OpenClaw tool names."""
        caps: set = set()
        for tool in tools:
            caps.update(_TOOL_TO_CAPABILITIES.get(tool, []))
        return sorted(caps)

    def skill_to_capabilities(self, skill: OpenClawSkillSpec) -> List[str]:
        """Infer PRSM capabilities from an OpenClaw skill spec."""
        # Prefer explicit prsm metadata if present
        prsm_role = skill.metadata.get("prsm_role", "")
        if prsm_role:
            from prsm.compute.nwtn.team.assembler import _DEFAULT_AGENT_MAP
            default = _DEFAULT_AGENT_MAP.get(prsm_role)
            if default:
                return default[2]  # capabilities list

        return self.tools_to_capabilities(skill.tools)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tools_for_role(self, role: str) -> List[str]:
        """Return the preferred tool set for a given role slug."""
        if role in _ROLE_TO_TOOLS:
            return _ROLE_TO_TOOLS[role]
        # Unknown role: infer from capabilities if available
        return ["file_reader", "web_search"]
