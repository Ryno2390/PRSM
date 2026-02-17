#!/usr/bin/env python3
"""
PRSM Project Foundation Test Suite
Tests the basic foundation structure and core imports of PRSM implementation
"""

import pytest
import sys
import os
from pathlib import Path
from typing import List

# Add the project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectStructure:
    """Test suite for verifying project directory structure"""
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Fixture providing the project root directory"""
        return PROJECT_ROOT
    
    @pytest.mark.parametrize("expected_dir", [
        "prsm",
        "prsm/core",
        "prsm/compute",
        "prsm/compute/agents",
        "prsm/compute/nwtn",
        "prsm/compute/teachers",
        "prsm/compute/federation",
        "prsm/economy",
        "prsm/economy/tokenomics",
        "prsm/data",
        "prsm/interface",
        "prsm/sdks",
        "tests",
        "docs",
        "scripts",
    ])
    def test_expected_directories_exist(self, project_root, expected_dir):
        """Test that all expected project directories exist"""
        dir_path = project_root / expected_dir
        assert dir_path.exists(), f"Expected directory {expected_dir} does not exist"
        assert dir_path.is_dir(), f"Expected {expected_dir} to be a directory"
    
    @pytest.mark.parametrize("expected_file", [
        "README.md",
        "requirements.txt",
        "prsm/__init__.py",
        "prsm/core/__init__.py",
        "prsm/compute/__init__.py",
    ])
    def test_expected_files_exist(self, project_root, expected_file):
        """Test that critical project files exist"""
        file_path = project_root / expected_file
        assert file_path.exists(), f"Expected file {expected_file} does not exist"
        assert file_path.is_file(), f"Expected {expected_file} to be a file"
    
    def test_python_package_structure(self, project_root):
        """Test that Python packages are properly structured with __init__.py files"""
        prsm_dir = project_root / "prsm"
        assert prsm_dir.exists(), "PRSM package directory must exist"

        # Directories to exclude from __init__.py requirement
        exclude_patterns = {"archive", "experiments", "unused_files", "__pycache__"}

        # Find all Python package directories (directories containing Python files)
        python_dirs = []
        for root, dirs, files in os.walk(prsm_dir):
            root_path = Path(root)
            # Skip excluded directories
            if any(part in exclude_patterns for part in root_path.parts):
                continue
            if any(f.endswith('.py') and f != '__init__.py' for f in files):
                python_dirs.append(root_path)

        # Each directory with Python files should have an __init__.py
        missing_init_files = []
        for py_dir in python_dirs:
            init_file = py_dir / "__init__.py"
            if not init_file.exists():
                missing_init_files.append(py_dir.relative_to(project_root))

        assert len(missing_init_files) == 0, f"Missing __init__.py files in: {missing_init_files}"


class TestCoreImports:
    """Test suite for verifying core module imports work correctly"""
    
    def test_core_models_import(self):
        """Test that core models can be imported"""
        try:
            from prsm.core.models import PeerNode
            assert PeerNode is not None
        except ImportError as e:
            pytest.fail(f"Failed to import core models: {e}")
    
    def test_core_config_import(self):
        """Test that core configuration can be imported"""
        try:
            from prsm.core.config import PRSMConfig
            assert PRSMConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import core config: {e}")
    
    def test_agent_framework_imports(self):
        """Test that agent framework components can be imported"""
        try:
            from prsm.compute.agents.base import BaseAgent
            assert BaseAgent is not None
        except ImportError as e:
            pytest.fail(f"Failed to import agent framework: {e}")
    
    def test_tokenomics_imports(self):
        """Test that tokenomics components can be imported"""
        try:
            from prsm.economy.tokenomics.ftns_service import FTNSService
            assert FTNSService is not None
        except ImportError as e:
            pytest.fail(f"Failed to import tokenomics: {e}")
    
    def test_federation_imports(self):
        """Test that federation components can be imported"""
        try:
            from prsm.compute.federation.consensus import DistributedConsensus
            assert DistributedConsensus is not None
        except ImportError as e:
            pytest.fail(f"Failed to import federation: {e}")


class TestSystemRequirements:
    """Test suite for verifying system requirements and dependencies"""
    
    def test_python_version(self):
        """Test that Python version meets minimum requirements"""
        import sys
        major, minor = sys.version_info[:2]
        assert major == 3, "Python 3 is required"
        assert minor >= 11, f"Python 3.11+ required, found {major}.{minor}"
    
    def test_required_packages_available(self):
        """Test that critical packages are available for import"""
        critical_packages = [
            'asyncio',
            'pathlib',
            'json',
            'logging',
            'dataclasses',
            'typing'
        ]
        
        missing_packages = []
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        assert len(missing_packages) == 0, f"Missing critical packages: {missing_packages}"
    
    def test_optional_packages_importable(self):
        """Test that optional packages can be imported if available"""
        optional_packages = {
            'fastapi': 'FastAPI web framework',
            'pydantic': 'Data validation',
            'sqlalchemy': 'Database ORM',
            'redis': 'Caching',
            'pytest': 'Testing framework'
        }
        
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                pytest.skip(f"Optional package {package} ({description}) not available")


class TestProjectConfiguration:
    """Test suite for project configuration files"""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and is readable"""
        req_file = PROJECT_ROOT / "requirements.txt"
        assert req_file.exists(), "requirements.txt file must exist"
        
        # Try to read the file
        with open(req_file, 'r') as f:
            content = f.read()
        assert len(content) > 0, "requirements.txt should not be empty"
    
    def test_readme_file_exists(self):
        """Test that README.md exists and has content"""
        readme_file = PROJECT_ROOT / "README.md"
        assert readme_file.exists(), "README.md file must exist"
        
        with open(readme_file, 'r') as f:
            content = f.read()
        assert len(content) > 100, "README.md should contain substantial content"
        assert "PRSM" in content, "README should mention PRSM"
    
    def test_git_repository_structure(self):
        """Test that this is a proper git repository"""
        git_dir = PROJECT_ROOT / ".git"
        if git_dir.exists():
            assert git_dir.is_dir(), ".git should be a directory"
            # Basic git files that should exist
            assert (git_dir / "config").exists(), "Git config should exist"
            assert (git_dir / "HEAD").exists(), "Git HEAD should exist"


class TestCodeQuality:
    """Test suite for basic code quality checks"""
    
    def test_no_syntax_errors_in_core_modules(self):
        """Test that core Python modules have no syntax errors"""
        core_modules = [
            "prsm/__init__.py",
            "prsm/core/__init__.py",
            "prsm/core/models.py",
            "prsm/core/config.py"
        ]
        
        for module_path in core_modules:
            file_path = PROJECT_ROOT / module_path
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        source_code = f.read()
                    compile(source_code, str(file_path), 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {module_path}: {e}")
    
    def test_python_files_have_proper_encoding(self):
        """Test that Python files have proper UTF-8 encoding"""
        prsm_dir = PROJECT_ROOT / "prsm"
        
        for py_file in prsm_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                pytest.fail(f"File {py_file.relative_to(PROJECT_ROOT)} has encoding issues")


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])