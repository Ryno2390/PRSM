"""PRSM Skill Loader — Load and validate skill packages from directories.

Discovers SKILL.yaml manifests, parses them, loads associated prompts,
and returns fully-populated SkillManifest objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .schema import SkillManifest, parse_manifest

logger = logging.getLogger(__name__)

# Path to built-in skill packages shipped with PRSM
BUILTINS_DIR = Path(__file__).parent / "builtins"


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required for skill loading. Install it: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_prompts(skill_dir: Path) -> Dict[str, str]:
    """Load all prompt files from a skill's prompts/ directory.

    Returns a dict mapping prompt name (without extension) to content.
    """
    prompts = {}
    prompts_dir = skill_dir / "prompts"
    if prompts_dir.is_dir():
        for prompt_file in sorted(prompts_dir.glob("*.md")):
            prompt_name = prompt_file.stem
            try:
                prompts[prompt_name] = prompt_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to load prompt {prompt_file}: {e}")
    return prompts


def load_skill(skill_dir: Path) -> Optional[SkillManifest]:
    """Load a single skill package from a directory.

    The directory must contain a SKILL.yaml manifest file.
    Returns None if the directory is not a valid skill package.
    """
    skill_dir = Path(skill_dir)
    manifest_path = skill_dir / "SKILL.yaml"

    if not manifest_path.is_file():
        logger.debug(f"No SKILL.yaml found in {skill_dir}")
        return None

    try:
        raw_data = _load_yaml(manifest_path)
    except Exception as e:
        logger.error(f"Failed to parse {manifest_path}: {e}")
        return None

    if not raw_data or "name" not in raw_data:
        logger.error(f"Invalid SKILL.yaml in {skill_dir}: missing 'name' field")
        return None

    try:
        manifest = parse_manifest(raw_data, skill_path=str(skill_dir))
    except Exception as e:
        logger.error(f"Failed to validate manifest in {skill_dir}: {e}")
        return None

    # Load associated prompts
    manifest.prompts = _load_prompts(skill_dir)

    logger.info(f"Loaded skill: {manifest.name} v{manifest.version} ({manifest.tool_count} tools)")
    return manifest


def load_skills_from_directory(base_dir: Path) -> List[SkillManifest]:
    """Scan a directory for skill packages and load all valid ones.

    Each immediate subdirectory containing SKILL.yaml is treated as a skill.
    """
    base_dir = Path(base_dir)
    skills = []

    if not base_dir.is_dir():
        logger.debug(f"Skills directory does not exist: {base_dir}")
        return skills

    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            manifest = load_skill(child)
            if manifest is not None:
                skills.append(manifest)

    return skills


def load_builtin_skills() -> List[SkillManifest]:
    """Load all built-in skill packages shipped with PRSM."""
    return load_skills_from_directory(BUILTINS_DIR)
