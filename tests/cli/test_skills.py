"""Tests for prsm.skills — schema, loader, registry."""
import pytest
import yaml
from pathlib import Path

from prsm.skills.schema import (
    SkillManifest, SkillTool, SkillParameter,
    parse_tool_parameters, parse_manifest,
)
from prsm.skills.loader import load_skill, load_skills_from_directory, load_builtin_skills
from prsm.skills.registry import SkillRegistry


# ── Schema Tests ─────────────────────────────────────────────────────


class TestSkillParameter:
    def test_defaults(self):
        p = SkillParameter()
        assert p.name == ""
        assert p.type == "string"
        assert p.required is True
        assert p.optional is False

    def test_custom(self):
        p = SkillParameter(name="query", type="string", description="search", optional=True, required=False)
        assert p.name == "query"
        assert p.optional is True


class TestSkillTool:
    def test_defaults(self):
        t = SkillTool()
        assert t.name == ""
        assert t.parameters == []

    def test_with_params(self):
        t = SkillTool(name="search", parameters=[SkillParameter(name="q")])
        assert len(t.parameters) == 1
        assert t.parameters[0].name == "q"


class TestSkillManifest:
    def test_defaults(self):
        m = SkillManifest()
        assert m.name == ""
        assert m.version == "1.0.0"
        assert m.tool_count == 0
        assert m.capability_summary == "none"

    def test_tool_count(self):
        m = SkillManifest(tools=[SkillTool(name="a"), SkillTool(name="b")])
        assert m.tool_count == 2

    def test_capability_summary(self):
        m = SkillManifest(capabilities=["search", "curate"])
        assert m.capability_summary == "search, curate"


class TestParseToolParameters:
    def test_empty(self):
        assert parse_tool_parameters({}) == []
        assert parse_tool_parameters(None) == []

    def test_dict_format(self):
        params = parse_tool_parameters({
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "optional": True},
        })
        assert len(params) == 2
        assert params[0].name == "query"
        assert params[0].required is True
        assert params[1].name == "limit"
        assert params[1].optional is True
        assert params[1].required is False

    def test_shorthand_format(self):
        params = parse_tool_parameters({"name": "string"})
        assert len(params) == 1
        assert params[0].name == "name"
        assert params[0].type == "string"


class TestParseManifest:
    def test_minimal(self):
        data = {"name": "test-skill", "tools": []}
        m = parse_manifest(data)
        assert m.name == "test-skill"
        assert m.tool_count == 0

    def test_full(self):
        data = {
            "name": "my-skill",
            "version": "2.0.0",
            "description": "A test skill",
            "author": "Tester",
            "capabilities": ["search"],
            "requires": ["dep >= 1.0"],
            "tools": [
                {
                    "name": "my_tool",
                    "description": "Does stuff",
                    "parameters": {
                        "query": {"type": "string", "description": "Input"},
                    },
                }
            ],
        }
        m = parse_manifest(data, skill_path="/tmp/test")
        assert m.name == "my-skill"
        assert m.version == "2.0.0"
        assert m.author == "Tester"
        assert m.tool_count == 1
        assert m.tools[0].parameters[0].name == "query"
        assert m.path == "/tmp/test"


# ── Loader Tests ─────────────────────────────────────────────────────


class TestLoadBuiltinSkills:
    def test_loads_three_builtins(self):
        skills = load_builtin_skills()
        assert len(skills) == 3

    def test_builtin_names(self):
        skills = load_builtin_skills()
        names = sorted(s.name for s in skills)
        assert names == ["prsm-compute", "prsm-datasets", "prsm-network"]

    def test_builtin_tools_count(self):
        skills = load_builtin_skills()
        total = sum(s.tool_count for s in skills)
        assert total == 12  # 4 + 4 + 4


class TestLoadSkill:
    def test_load_from_valid_dir(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        manifest = {
            "name": "test-skill",
            "version": "1.0.0",
            "description": "Test",
            "tools": [
                {"name": "tool1", "description": "A tool", "parameters": {}},
            ],
        }
        (skill_dir / "SKILL.yaml").write_text(yaml.dump(manifest))

        result = load_skill(skill_dir)
        assert result is not None
        assert result.name == "test-skill"
        assert result.tool_count == 1

    def test_load_from_dir_without_manifest(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert load_skill(empty_dir) is None

    def test_load_with_prompts(self, tmp_path):
        skill_dir = tmp_path / "prompted-skill"
        skill_dir.mkdir()
        prompts_dir = skill_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "system.md").write_text("You are a helper.")

        manifest = {"name": "prompted", "tools": []}
        (skill_dir / "SKILL.yaml").write_text(yaml.dump(manifest))

        result = load_skill(skill_dir)
        assert result is not None
        assert "system" in result.prompts
        assert "helper" in result.prompts["system"]


class TestLoadSkillsFromDirectory:
    def test_load_multiple(self, tmp_path):
        for name in ("alpha", "beta"):
            d = tmp_path / name
            d.mkdir()
            manifest = {"name": f"skill-{name}", "tools": []}
            (d / "SKILL.yaml").write_text(yaml.dump(manifest))

        # Add a non-skill directory
        (tmp_path / "not-a-skill").mkdir()

        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 2

    def test_nonexistent_directory(self, tmp_path):
        skills = load_skills_from_directory(tmp_path / "nope")
        assert skills == []


# ── Registry Tests ───────────────────────────────────────────────────


class TestSkillRegistry:
    def test_loads_builtins(self, registry):
        assert registry.skill_count == 3
        assert registry.tool_count == 12

    def test_list_skills(self, registry):
        skills = registry.list_skills()
        assert len(skills) == 3
        # Should be sorted by name
        names = [s.name for s in skills]
        assert names == sorted(names)

    def test_get_by_name(self, registry):
        ds = registry.get("prsm-datasets")
        assert ds is not None
        assert ds.name == "prsm-datasets"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_search_by_name(self, registry):
        results = registry.search("datasets")
        assert len(results) == 1
        assert results[0].name == "prsm-datasets"

    def test_search_by_capability(self, registry):
        results = registry.search("network")
        names = [s.name for s in results]
        assert "prsm-network" in names

    def test_search_no_results(self, registry):
        assert registry.search("zzzznothing") == []

    def test_get_all_tools(self, registry):
        tools = registry.get_all_tools()
        assert len(tools) == 12
        # Each tool should have required fields
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_register_custom(self, registry):
        custom = SkillManifest(name="custom-skill", tools=[
            SkillTool(name="custom_tool"),
        ])
        registry.register(custom)
        assert registry.skill_count == 4
        assert registry.get("custom-skill") is not None

    def test_unregister(self, registry):
        assert registry.unregister("prsm-datasets") is True
        assert registry.skill_count == 2
        assert registry.unregister("nonexistent") is False

    def test_load_from_directory(self, registry, tmp_path):
        d = tmp_path / "extra-skill"
        d.mkdir()
        manifest = {"name": "extra", "tools": [{"name": "etool", "description": "x", "parameters": {}}]}
        (d / "SKILL.yaml").write_text(yaml.dump(manifest))

        count = registry.load_from_directory(tmp_path)
        assert count == 1
        assert registry.get("extra") is not None

    def test_repr(self, registry):
        r = repr(registry)
        assert "3 skills" in r
        assert "12 tools" in r

    def test_empty_registry(self):
        reg = SkillRegistry(load_builtins=False)
        assert reg.skill_count == 0
        assert reg.tool_count == 0
        assert reg.list_skills() == []
