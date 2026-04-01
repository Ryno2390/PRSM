"""PRSM Skill Package Schema — Models for SKILL.yaml format.

Defines the structure of skill manifests, tools, and parameters
in an MCP-compatible format that AI agents can consume.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field


@dataclass
class SkillParameter:
    """A single parameter for a skill tool."""
    name: str = ""
    type: str = "string"
    description: str = ""
    required: bool = True
    optional: bool = False
    default: typing.Any = None
    items: typing.Optional[typing.Dict[str, str]] = None


@dataclass
class SkillTool:
    """An MCP-compatible tool definition within a skill package."""
    name: str = ""
    description: str = ""
    parameters: typing.List[SkillParameter] = field(default_factory=list)


@dataclass
class SkillManifest:
    """Complete skill package manifest parsed from SKILL.yaml."""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = "PRSM Core Team"
    capabilities: typing.List[str] = field(default_factory=list)
    requires: typing.List[str] = field(default_factory=list)
    tools: typing.List[SkillTool] = field(default_factory=list)

    # Runtime metadata (not from YAML)
    path: typing.Optional[str] = None
    prompts: typing.Dict[str, str] = field(default_factory=dict)

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def capability_summary(self) -> str:
        return ", ".join(self.capabilities) if self.capabilities else "none"


def parse_tool_parameters(params_dict: typing.Dict[str, typing.Any]) -> typing.List[SkillParameter]:
    """Parse a YAML parameters dict into a list of SkillParameter objects.

    SKILL.yaml format:
        parameters:
          query: {type: string, description: "Search query"}
          domain: {type: string, optional: true}

    Returns list of SkillParameter with .name populated.
    """
    parameters = []
    if not params_dict:
        return parameters

    for param_name, param_spec in params_dict.items():
        if isinstance(param_spec, dict):
            is_optional = param_spec.get("optional", False)
            is_required = param_spec.get("required", not is_optional)
            param = SkillParameter(
                name=param_name,
                type=param_spec.get("type", "string"),
                description=param_spec.get("description", ""),
                required=is_required,
                optional=is_optional,
                default=param_spec.get("default"),
                items=param_spec.get("items"),
            )
        else:
            # Simple type shorthand: query: string
            param = SkillParameter(name=param_name, type=str(param_spec))
        parameters.append(param)

    return parameters


def parse_manifest(data: typing.Dict[str, typing.Any], skill_path: typing.Optional[str] = None) -> SkillManifest:
    """Parse a raw YAML dict into a validated SkillManifest.

    Handles the nested parameter format conversion.
    """
    tools = []
    for tool_data in data.get("tools", []):
        raw_params = tool_data.get("parameters", {})
        parsed_params = parse_tool_parameters(raw_params)
        tool = SkillTool(
            name=tool_data["name"],
            description=tool_data.get("description", ""),
            parameters=parsed_params,
        )
        tools.append(tool)

    manifest = SkillManifest(
        name=data["name"],
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        author=data.get("author", "PRSM Core Team"),
        capabilities=data.get("capabilities", []),
        requires=data.get("requires", []),
        tools=tools,
        path=skill_path,
    )

    return manifest
