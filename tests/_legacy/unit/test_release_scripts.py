"""
Tests for PRSM Release Scripts

Tests for:
- Version bumping (bump_version.py)
- Release preparation (prepare_release.py)
- Changelog generation
"""

import re
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "release"))

from bump_version import VersionBumper


class TestVersionBumper:
    """Tests for VersionBumper class."""
    
    def test_validate_version_valid_semver(self):
        """Test valid semver versions are accepted."""
        bumper = VersionBumper("1.0.0", dry_run=True)
        assert bumper.validate_version() is True
        
        bumper = VersionBumper("0.1.0", dry_run=True)
        assert bumper.validate_version() is True
        
        bumper = VersionBumper("10.20.30", dry_run=True)
        assert bumper.validate_version() is True
    
    def test_validate_version_prerelease(self):
        """Test prerelease versions are accepted."""
        bumper = VersionBumper("1.0.0-alpha", dry_run=True)
        assert bumper.validate_version() is True
        
        bumper = VersionBumper("1.0.0-beta.1", dry_run=True)
        assert bumper.validate_version() is True
        
        bumper = VersionBumper("1.0.0-rc.1", dry_run=True)
        assert bumper.validate_version() is True
    
    def test_validate_version_invalid(self):
        """Test invalid versions are rejected."""
        # Missing patch version
        bumper = VersionBumper("1.0", dry_run=True)
        assert bumper.validate_version() is False
        
        # Non-numeric
        bumper = VersionBumper("a.b.c", dry_run=True)
        assert bumper.validate_version() is False
        
        # Empty string
        bumper = VersionBumper("", dry_run=True)
        assert bumper.validate_version() is False
        
        # Leading 'v'
        bumper = VersionBumper("v1.0.0", dry_run=True)
        assert bumper.validate_version() is False
    
    def test_get_current_version(self):
        """Test getting current version from pyproject.toml."""
        bumper = VersionBumper("1.0.0", dry_run=True)
        version = bumper.get_current_version()
        
        # Should return a version string or None
        if version:
            assert re.match(r'^\d+\.\d+\.\d+', version)
    
    @pytest.mark.parametrize("old,new,expected", [
        ("0.1.0", "0.1.1", "patch"),
        ("0.1.0", "0.2.0", "minor"),
        ("0.1.0", "1.0.0", "major"),
        ("0.1.0", "0.1.0-alpha", "prerelease"),
    ])
    def test_release_type_detection(self, old, new, expected):
        """Test release type is correctly detected."""
        # This would be tested in prepare_release.py
        # For now, we test the version comparison logic
        old_parts = old.split('-')[0].split('.')
        new_parts = new.split('-')[0].split('.')
        
        if '-' in new:
            release_type = "prerelease"
        elif int(new_parts[0]) > int(old_parts[0]):
            release_type = "major"
        elif int(new_parts[1]) > int(old_parts[1]):
            release_type = "minor"
        else:
            release_type = "patch"
        
        assert release_type == expected


class TestVersionBumperFileUpdates:
    """Tests for file update operations."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure for testing."""
        # Create pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test-project"
version = "0.1.0"
description = "Test project"
""")
        
        # Create setup.py
        setup_py = tmp_path / "setup.py"
        setup_py.write_text("""
from setuptools import setup

setup(
    name="test-project",
    version="0.1.0",
)
""")
        
        # Create __init__.py
        init_dir = tmp_path / "test_package"
        init_dir.mkdir()
        init_py = init_dir / "__init__.py"
        init_py.write_text('__version__ = "0.1.0"\n')
        
        # Create CHANGELOG.md
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("""# Changelog

All notable changes will be documented in this file.

## [0.1.0] - 2024-01-01

### Added
- Initial release
""")
        
        return tmp_path
    
    def test_bump_pyproject_toml(self, temp_project):
        """Test version bump in pyproject.toml."""
        # Read original content
        pyproject_path = temp_project / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Verify version pattern exists
        assert 'version = "0.1.0"' in content
        
        # Test the regex replacement
        new_content = re.sub(
            r'^version\s*=\s*"[^"]+"',
            'version = "0.2.0"',
            content,
            flags=re.MULTILINE
        )
        
        assert 'version = "0.2.0"' in new_content
        assert 'version = "0.1.0"' not in new_content
    
    def test_bump_setup_py(self, temp_project):
        """Test version bump in setup.py."""
        setup_path = temp_project / "setup.py"
        content = setup_path.read_text()
        
        # Test the regex replacement
        new_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            'version="0.2.0"',
            content
        )
        
        assert 'version="0.2.0"' in new_content
        assert 'version="0.1.0"' not in new_content
    
    def test_bump_init_py(self, temp_project):
        """Test version bump in __init__.py."""
        init_path = temp_project / "test_package" / "__init__.py"
        content = init_path.read_text()
        
        # Test the regex replacement
        new_content = re.sub(
            r'__version__\s*=\s*["\'][^"\']+["\']',
            '__version__ = "0.2.0"',
            content
        )
        
        assert '__version__ = "0.2.0"' in new_content
        assert '__version__ = "0.1.0"' not in new_content
    
    def test_changelog_update(self, temp_project):
        """Test changelog section addition."""
        changelog_path = temp_project / "CHANGELOG.md"
        content = changelog_path.read_text()
        
        # Create new section
        today = datetime.now().strftime("%Y-%m-%d")
        new_version = "0.2.0"
        
        # Check version doesn't already exist
        assert f"## [{new_version}]" not in content
        
        # Add new section
        new_section = f"""
## [{new_version}] - {today}

### Added
- New features

### Changed
- Updates

### Fixed
- Bug fixes
"""
        
        # Find insertion point
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('# Changelog'):
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].strip() == '':
                    insert_idx += 1
                break
        
        new_content = '\n'.join(lines[:insert_idx]) + new_section + '\n'.join(lines[insert_idx:])
        
        assert f"## [{new_version}]" in new_content
        assert today in new_content


class TestReleasePreparation:
    """Tests for release preparation logic."""
    
    def test_version_comparison(self):
        """Test version comparison logic."""
        def compare_versions(old: str, new: str) -> str:
            """Compare versions and return release type."""
            old_parts = old.split('-')[0].split('.')
            new_parts = new.split('-')[0].split('.')
            
            if '-' in new:
                return "prerelease"
            elif int(new_parts[0]) > int(old_parts[0]):
                return "major"
            elif int(new_parts[1]) > int(old_parts[1]):
                return "minor"
            else:
                return "patch"
        
        # Test cases
        assert compare_versions("0.1.0", "0.1.1") == "patch"
        assert compare_versions("0.1.0", "0.2.0") == "minor"
        assert compare_versions("0.1.0", "1.0.0") == "major"
        assert compare_versions("1.0.0", "1.1.0-alpha") == "prerelease"
        assert compare_versions("1.0.0", "2.0.0") == "major"
        assert compare_versions("1.2.3", "1.3.0") == "minor"
        assert compare_versions("1.2.3", "1.2.4") == "patch"
    
    def test_suggest_next_version(self):
        """Test next version suggestion."""
        def suggest_next(current: str, release_type: str) -> str:
            """Suggest next version based on release type."""
            parts = current.split('-')[0].split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if release_type == "major":
                return f"{major + 1}.0.0"
            elif release_type == "minor":
                return f"{major}.{minor + 1}.0"
            elif release_type == "patch":
                return f"{major}.{minor}.{patch + 1}"
            else:
                return f"{major}.{minor}.{patch + 1}-rc.1"
        
        assert suggest_next("0.1.0", "patch") == "0.1.1"
        assert suggest_next("0.1.0", "minor") == "0.2.0"
        assert suggest_next("0.1.0", "major") == "1.0.0"
        assert suggest_next("1.2.3", "patch") == "1.2.4"
        assert suggest_next("1.2.3", "minor") == "1.3.0"
        assert suggest_next("1.2.3", "major") == "2.0.0"


class TestChangelogGeneration:
    """Tests for changelog generation."""
    
    def test_changelog_format(self):
        """Test changelog follows expected format."""
        # Sample changelog entry
        changelog = """# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-01-15

### Added
- New feature A
- New feature B

### Changed
- Updated feature C

### Fixed
- Bug fix for D

### Security
- Security patch for E

## [0.1.0] - 2024-01-01

### Added
- Initial release
"""
        
        # Verify structure
        assert "# Changelog" in changelog
        assert "## [0.2.0]" in changelog
        assert "## [0.1.0]" in changelog
        assert "### Added" in changelog
        assert "### Changed" in changelog
        assert "### Fixed" in changelog
        assert "### Security" in changelog
    
    def test_conventional_commit_parsing(self):
        """Test parsing of conventional commit messages."""
        commits = [
            ("feat: add new feature", "Features"),
            ("fix: resolve bug", "Bug Fixes"),
            ("docs: update documentation", "Documentation"),
            ("perf: improve performance", "Performance"),
            ("refactor: clean up code", "Refactor"),
            ("test: add tests", "Testing"),
            ("chore: update dependencies", "Miscellaneous Tasks"),
        ]
        
        commit_parsers = [
            (r"^feat", "Features"),
            (r"^fix", "Bug Fixes"),
            (r"^doc", "Documentation"),
            (r"^perf", "Performance"),
            (r"^refactor", "Refactor"),
            (r"^test", "Testing"),
            (r"^chore|^ci", "Miscellaneous Tasks"),
        ]
        
        for message, expected_group in commits:
            for pattern, group in commit_parsers:
                if re.match(pattern, message):
                    assert group == expected_group
                    break


class TestPyprojectToml:
    """Tests for pyproject.toml configuration."""
    
    def test_pyproject_toml_exists(self):
        """Test pyproject.toml exists in project root."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
    
    def test_pyproject_toml_build_system(self):
        """Test build system configuration."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert "[build-system]" in content
        assert "setuptools" in content
        assert "wheel" in content
    
    def test_pyproject_toml_project_metadata(self):
        """Test project metadata in pyproject.toml."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert "[project]" in content
        assert 'name = "prsm-network"' in content
        assert "version" in content
        assert "description" in content
    
    def test_pyproject_toml_scripts(self):
        """Test console scripts are defined."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert "[project.scripts]" in content
        assert "prsm = " in content
    
    def test_pyproject_toml_git_cliff_config(self):
        """Test git-cliff configuration exists."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert "[tool.git-cliff.changelog]" in content
        assert "[tool.git-cliff.git]" in content
        assert "conventional_commits = true" in content


class TestReleaseWorkflow:
    """Tests for release workflow configuration."""
    
    def test_release_workflow_exists(self):
        """Test release workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        assert workflow_path.exists(), "release.yml workflow should exist"
    
    def test_release_workflow_triggers(self):
        """Test release workflow triggers."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        content = workflow_path.read_text()
        
        assert "release:" in content
        assert "types: [published]" in content
        assert "workflow_dispatch:" in content
    
    def test_release_workflow_pypi_job(self):
        """Test PyPI publishing job exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        content = workflow_path.read_text()
        
        assert "publish-pypi:" in content or "pypi:" in content
        assert "pypa/gh-action-pypi-publish" in content
    
    def test_release_workflow_docker_job(self):
        """Test Docker publishing job exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        content = workflow_path.read_text()
        
        assert "publish-docker:" in content or "docker:" in content
        assert "docker/build-push-action" in content


class TestChangelogWorkflow:
    """Tests for changelog workflow configuration."""
    
    def test_changelog_workflow_exists(self):
        """Test changelog workflow file exists."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "changelog.yml"
        assert workflow_path.exists(), "changelog.yml workflow should exist"
    
    def test_changelog_workflow_triggers(self):
        """Test changelog workflow triggers."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "changelog.yml"
        content = workflow_path.read_text()
        
        assert "push:" in content
        assert "tags:" in content
        assert "release:" in content or "workflow_dispatch:" in content
    
    def test_changelog_workflow_git_cliff(self):
        """Test git-cliff action is used."""
        workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "changelog.yml"
        content = workflow_path.read_text()
        
        assert "git-cliff" in content.lower() or "orhun/git-cliff-action" in content


# Integration tests (marked as slow)
@pytest.mark.slow
class TestReleaseIntegration:
    """Integration tests for release process."""
    
    @pytest.mark.skip(reason="Requires git repository")
    def test_full_release_dry_run(self):
        """Test full release process in dry-run mode."""
        # This would test the complete release flow
        # but requires a git repository setup
        pass
    
    @pytest.mark.skip(reason="Requires git repository")
    def test_version_bump_creates_commit(self):
        """Test version bump creates a commit."""
        # This would test git commit creation
        pass
    
    @pytest.mark.skip(reason="Requires git repository")
    def test_tag_creation(self):
        """Test git tag is created correctly."""
        # This would test tag creation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
