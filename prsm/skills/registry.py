"""PRSM Skill Registry — Central registry for discovering and managing skills.

Provides list, search, get, and register operations for skill packages.
Automatically loads built-in skills on initialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .schema import SkillManifest
from .loader import load_builtin_skills, load_skill, load_skills_from_directory

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for PRSM skill packages.

    Manages both built-in and user-installed skills. Provides
    discovery, search, and retrieval operations.

    Usage:
        registry = SkillRegistry()          # Auto-loads builtins
        registry.list_skills()              # All registered skills
        registry.get("prsm-datasets")       # Get by name
        registry.search("dataset")          # Fuzzy search
    """

    def __init__(self, load_builtins: bool = True):
        """Initialize the registry.

        Args:
            load_builtins: Whether to auto-load built-in skill packages.
        """
        self._skills: Dict[str, SkillManifest] = {}

        if load_builtins:
            self._load_builtins()

    def _load_builtins(self) -> None:
        """Load all built-in skill packages."""
        try:
            builtins = load_builtin_skills()
            for skill in builtins:
                self._skills[skill.name] = skill
            logger.info(f"Loaded {len(builtins)} built-in skill(s)")
        except Exception as e:
            logger.warning(f"Failed to load built-in skills: {e}")

    def register(self, skill: SkillManifest) -> None:
        """Register a skill package in the registry.

        Args:
            skill: A validated SkillManifest to register.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        if skill.name in self._skills:
            logger.warning(f"Overwriting existing skill: {skill.name}")
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name} v{skill.version}")

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry.

        Returns True if the skill was found and removed.
        """
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get(self, name: str) -> Optional[SkillManifest]:
        """Get a skill by exact name.

        Args:
            name: The skill package name (e.g., "prsm-datasets").

        Returns:
            The SkillManifest or None if not found.
        """
        return self._skills.get(name)

    def list_skills(self) -> List[SkillManifest]:
        """List all registered skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def search(self, query: str) -> List[SkillManifest]:
        """Search skills by name, description, or capabilities.

        Args:
            query: Search string (case-insensitive substring match).

        Returns:
            List of matching SkillManifest objects.
        """
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            # Search across name, description, and capabilities
            searchable = " ".join([
                skill.name,
                skill.description,
                " ".join(skill.capabilities),
            ]).lower()

            if query_lower in searchable:
                results.append(skill)

        return sorted(results, key=lambda s: s.name)

    def get_all_tools(self) -> List[dict]:
        """Get all tools from all registered skills as MCP-compatible dicts.

        Returns a flat list of tool definitions suitable for MCP server registration.
        """
        tools = []
        for skill in self._skills.values():
            for tool in skill.tools:
                tool_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "skill": skill.name,
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
                for param in tool.parameters:
                    prop = {"type": param.type, "description": param.description}
                    if param.items:
                        prop["items"] = param.items
                    tool_def["inputSchema"]["properties"][param.name] = prop
                    if param.required and not param.optional:
                        tool_def["inputSchema"]["required"].append(param.name)
                tools.append(tool_def)
        return tools

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    @property
    def tool_count(self) -> int:
        return sum(s.tool_count for s in self._skills.values())

    def load_from_directory(self, directory: Path) -> int:
        """Load additional skills from a directory.

        Returns the number of skills loaded.
        """
        skills = load_skills_from_directory(directory)
        for skill in skills:
            self.register(skill)
        return len(skills)

    def load_from_path(self, skill_dir: Path) -> Optional[SkillManifest]:
        """Load and register a single skill from a directory path.

        Returns the loaded SkillManifest or None.
        """
        skill = load_skill(skill_dir)
        if skill:
            self.register(skill)
        return skill

    def __repr__(self) -> str:
        return f"SkillRegistry({self.skill_count} skills, {self.tool_count} tools)"
