"""PRSM Skills — AI-native skill package system.

Skill packages define MCP-compatible tools, system prompts, and workflows
that AI agents can use to interact with the PRSM network.

Each skill is a directory containing:
  - SKILL.yaml   — Manifest with tool definitions
  - README.md    — Human-readable documentation
  - prompts/     — System prompts for AI agents
"""

from .schema import SkillManifest, SkillTool, SkillParameter
from .loader import load_skill, load_skills_from_directory
from .registry import SkillRegistry

__all__ = [
    "SkillManifest",
    "SkillTool",
    "SkillParameter",
    "load_skill",
    "load_skills_from_directory",
    "SkillRegistry",
]
