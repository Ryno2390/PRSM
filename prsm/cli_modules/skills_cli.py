"""PRSM Skills CLI — List, inspect, search, and export skill packages.

Commands:
    prsm skills list           List installed skill packages
    prsm skills info PACKAGE   Show detailed skill info
    prsm skills search QUERY   Search skills by name/description/capability
    prsm skills export PACKAGE Export skill as MCP tool manifest (JSON)
"""

from __future__ import annotations

import json
import sys

import click
from rich.table import Table

from .ui import console, section, summary_panel, info, error
from .theme import THEME, ICONS


def _get_registry():
    """Lazily load the SkillRegistry to avoid heavy imports at CLI startup."""
    from prsm.skills.registry import SkillRegistry
    return SkillRegistry()


@click.group()
def skills():
    """Skill package management — list, inspect, search, export."""
    pass


@skills.command("list")
def skills_list():
    """List installed skill packages."""
    registry = _get_registry()
    all_skills = registry.list_skills()

    if not all_skills:
        section("PRSM Skills", icon="prism")
        info("No skill packages installed.")
        return

    section("PRSM Skills", icon="prism")
    console.print()

    total_tools = 0
    for skill in all_skills:
        tool_names = [t.name for t in skill.tools]
        tool_count = len(tool_names)
        total_tools += tool_count
        tools_str = ", ".join(tool_names) if tool_names else "none"

        console.print(
            f"  {ICONS['bullet']} {skill.name} v{skill.version}",
            style=THEME.primary,
        )
        console.print(f"    {skill.description}", style=THEME.muted)
        console.print(
            f"    {tool_count} tools: {tools_str}",
            style=THEME.dim,
        )
        console.print()

    console.print(
        f"  {len(all_skills)} packages {ICONS['line']} {total_tools} tools",
        style=THEME.muted,
    )
    console.print()


@skills.command("info")
@click.argument("package")
def skills_info(package: str):
    """Show detailed info for a skill package."""
    registry = _get_registry()
    skill = registry.get(package)

    if not skill:
        error(f"Skill package not found: {package}")
        # Suggest similar names
        all_names = [s.name for s in registry.list_skills()]
        if all_names:
            info(f"Available packages: {', '.join(all_names)}")
        raise SystemExit(1)

    # Summary panel with package metadata
    summary_panel(f"{skill.name} v{skill.version}", {
        "Description": skill.description,
        "Author": skill.author,
        "Capabilities": skill.capability_summary,
        "Dependencies": ", ".join(skill.requires) if skill.requires else "none",
        "Path": skill.path or "built-in",
    })

    # Tools table
    if skill.tools:
        section("Tools", icon="bullet")
        console.print()

        table = Table(show_header=True, header_style=THEME.heading, box=None, padding=(0, 2))
        table.add_column("Tool", style=THEME.primary)
        table.add_column("Description", style=THEME.muted)
        table.add_column("Parameters", style=THEME.dim)

        for tool in skill.tools:
            param_names = [p.name for p in tool.parameters]
            params_str = ", ".join(param_names) if param_names else "none"
            table.add_row(tool.name, tool.description, params_str)

        console.print(table)
        console.print()

    # Prompts if available
    if skill.prompts:
        section("Prompts", icon="bullet")
        console.print()
        for prompt_name, prompt_text_val in skill.prompts.items():
            console.print(f"    {prompt_name}", style=THEME.primary)
            # Show truncated prompt text
            display = prompt_text_val[:80] + "..." if len(prompt_text_val) > 80 else prompt_text_val
            console.print(f"      {display}", style=THEME.dim)
        console.print()


@skills.command("search")
@click.argument("query")
def skills_search(query: str):
    """Search available skills by name, description, or capability."""
    registry = _get_registry()
    results = registry.search(query)

    section(f"Search: {query}", icon="prism")
    console.print()

    if not results:
        info(f"No skills matching '{query}'")
        return

    for skill in results:
        tool_names = [t.name for t in skill.tools]
        tools_str = ", ".join(tool_names) if tool_names else "none"

        console.print(
            f"  {ICONS['bullet']} {skill.name} v{skill.version}",
            style=THEME.primary,
        )
        console.print(f"    {skill.description}", style=THEME.muted)
        console.print(
            f"    {len(tool_names)} tools: {tools_str}",
            style=THEME.dim,
        )
        console.print()

    console.print(
        f"  {len(results)} result{'s' if len(results) != 1 else ''}",
        style=THEME.muted,
    )
    console.print()


@skills.command("export")
@click.argument("package")
def skills_export(package: str):
    """Export skill as MCP tool manifest (JSON to stdout)."""
    registry = _get_registry()
    skill = registry.get(package)

    if not skill:
        # Write error as JSON to stderr, keep stdout clean
        sys.stderr.write(f"Error: Skill package not found: {package}\n")
        raise SystemExit(1)

    # Build MCP-compatible manifest
    manifest = {
        "name": skill.name,
        "version": skill.version,
        "description": skill.description,
        "author": skill.author,
        "capabilities": skill.capabilities,
        "tools": [],
    }

    for tool in skill.tools:
        tool_def = {
            "name": tool.name,
            "description": tool.description,
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
        manifest["tools"].append(tool_def)

    # Output raw JSON to stdout (no Rich formatting)
    sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
    sys.stdout.flush()
