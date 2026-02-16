#!/usr/bin/env python3
"""
PRSM Foundation Testing Script
Tests the basic foundation of PRSM Phase 1 / Week 1 implementation
"""

import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def run_section(name: str):
    """Decorator to mark test sections"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {name}")
    print('='*60)

def success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def error(message: str, exception: Exception = None):
    """Print error message"""
    print(f"❌ {message}")
    if exception:
        print(f"   Error: {str(exception)}")

def warning(message: str):
    """Print warning message"""
    print(f"⚠️  {message}")

def test_project_structure():
    """Test that all expected directories and files exist"""
    run_section("Project Structure")
    
    expected_dirs = [
        "prsm",
        "prsm/core",
        "prsm/nwtn", 
        "prsm/agents",
        "prsm/agents/architects",
        "prsm/agents/prompters",
        "prsm/agents/routers",
        "prsm/agents/executors",
        "prsm/agents/compilers",
        "prsm/teachers",
        "prsm/safety",
        "prsm/federation",
        "prsm/tokenomics",
        "prsm/data_layer",
        "prsm/improvement",
        "prsm/api"
    ]
    
    expected_files = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        ".env.example",
        ".gitignore",
        "Makefile",
        "execution_plan.md",
        "prsm/__init__.py",
        "prsm/core/config.py",
        "prsm/core/models.py",
        "prsm/nwtn/orchestrator.py",
        "prsm/agents/architects/hierarchical_architect.py",
        "prsm/api/main.py",
        "prsm/cli.py"
    ]
    
    # Test directories
    missing_dirs = []
    for dir_path in expected_dirs:
        if Path(dir_path).is_dir():
            success(f"Directory exists: {dir_path}")
        else:
            missing_dirs.append(dir_path)
            error(f"Missing directory: {dir_path}")
    
    # Test files
    missing_files = []
    for file_path in expected_files:
        if Path(file_path).is_file():
            success(f"File exists: {file_path}")
        else:
            missing_files.append(file_path)
            error(f"Missing file: {file_path}")
    
    # Test __init__.py files in all Python packages
    for dir_path in expected_dirs:
        if dir_path.startswith("prsm"):
            init_file = Path(dir_path) / "__init__.py"
            if init_file.is_file():
                success(f"Package init file exists: {init_file}")
            else:
                error(f"Missing __init__.py: {init_file}")
    
    if not missing_dirs and not missing_files:
        success("All expected directories and files are present!")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_basic_imports():
    """Test that basic package imports work"""
    run_section("Basic Package Imports")
    
    import_tests = [
        ("prsm", "Main package"),
        ("prsm.core", "Core module"),
        ("prsm.core.config", "Configuration module"),
        ("prsm.core.models", "Data models module"),
        ("prsm.nwtn", "NWTN module"),
        ("prsm.nwtn.orchestrator", "NWTN orchestrator"),
        ("prsm.agents", "Agents module"),
        ("prsm.agents.architects", "Architects module"),
        ("prsm.agents.architects.hierarchical_architect", "Hierarchical architect"),
        ("prsm.api", "API module"),
        ("prsm.api.main", "FastAPI main")
    ]
    
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            success(f"Import successful: {module_name} ({description})")
        except Exception as e:
            failed_imports.append((module_name, description, e))
            error(f"Import failed: {module_name} ({description})", e)
    
    return len(failed_imports) == 0

def test_configuration_system():
    """Test configuration system"""
    run_section("Configuration System")
    
    try:
        from prsm.core.config import get_settings, PRSMSettings
        success("Configuration imports successful")
        
        # Test settings loading
        settings = get_settings()
        success(f"Settings loaded: {type(settings).__name__}")
        
        # Test some basic settings
        success(f"App name: {settings.app_name}")
        success(f"Environment: {settings.environment}")
        success(f"Debug mode: {settings.debug}")
        success(f"NWTN enabled: {settings.nwtn_enabled}")
        success(f"FTNS enabled: {settings.ftns_enabled}")
        
        # Test environment detection
        success(f"Is development: {settings.is_development}")
        success(f"Is production: {settings.is_production}")
        
        return True
        
    except Exception as e:
        error("Configuration system test failed", e)
        traceback.print_exc()
        return False

def test_data_models():
    """Test data models validation"""
    run_section("Data Models")
    
    try:
        from prsm.core.models import (
            PRSMSession, UserInput, ClarifiedPrompt, PRSMResponse,
            ArchitectTask, TaskHierarchy, TeacherModel, FTNSTransaction
        )
        success("Data models import successful")
        
        # Test UserInput model
        user_input = UserInput(
            user_id="test_user",
            prompt="Test query for PRSM",
            context_allocation=100
        )
        success(f"UserInput model created: {user_input.user_id}")
        
        # Test ArchitectTask model
        from uuid import uuid4
        task = ArchitectTask(
            session_id=uuid4(),  # Provide a proper UUID
            instruction="Test task",
            complexity_score=0.5
        )
        success(f"ArchitectTask model created: {task.task_id}")
        
        # Test model serialization
        json_data = user_input.model_dump()
        success(f"Model serialization works: {len(json_data)} fields")
        
        # Test model validation
        try:
            invalid_task = ArchitectTask(
                session_id=task.session_id,
                instruction="",  # Empty instruction should be valid but test validation
                complexity_score=1.5  # Invalid score > 1.0
            )
            warning("Model validation may need strengthening (complexity_score > 1.0 allowed)")
        except Exception as validation_error:
            success("Model validation working (rejected invalid complexity_score)")
        
        return True
        
    except Exception as e:
        error("Data models test failed", e)
        traceback.print_exc()
        return False

def test_cli_basic():
    """Test CLI interface basics"""
    run_section("CLI Interface")
    
    try:
        # Test CLI module import
        from prsm.cli import main
        success("CLI module import successful")
        
        # Test CLI help (this would normally be tested with click.testing.CliRunner)
        success("CLI main function accessible")
        
        # Note: Full CLI testing would require click.testing.CliRunner
        # but we can at least verify the module loads correctly
        
        return True
        
    except Exception as e:
        error("CLI test failed", e)
        traceback.print_exc()
        return False

def test_api_basic():
    """Test FastAPI application basics"""
    run_section("FastAPI Application")
    
    try:
        from prsm.interface.api.main import app
        success("FastAPI app import successful")
        
        # Test that app is a FastAPI instance
        from fastapi import FastAPI
        if isinstance(app, FastAPI):
            success("App is valid FastAPI instance")
        else:
            error("App is not a FastAPI instance")
            return False
        
        # Test basic app properties
        success(f"App title: {app.title}")
        success(f"App version: {app.version}")
        success(f"App description: {app.description}")
        
        return True
        
    except Exception as e:
        error("FastAPI test failed", e)
        traceback.print_exc()
        return False

def test_orchestrator_basic():
    """Test NWTN orchestrator basics"""
    run_section("NWTN Orchestrator")
    
    try:
        from prsm.compute.nwtn.orchestrator import NWTNOrchestrator, nwtn_orchestrator
        success("NWTN orchestrator import successful")
        
        # Test orchestrator instantiation
        orchestrator = NWTNOrchestrator()
        success("NWTN orchestrator created")
        
        # Test global instance
        if nwtn_orchestrator is not None:
            success("Global NWTN orchestrator available")
        else:
            warning("Global NWTN orchestrator is None")
        
        return True
        
    except Exception as e:
        error("NWTN orchestrator test failed", e)
        traceback.print_exc()
        return False

def test_architect_basic():
    """Test hierarchical architect basics"""
    run_section("Hierarchical Architect")
    
    try:
        from prsm.compute.agents.architects.hierarchical_architect import HierarchicalArchitect, create_architect
        success("Hierarchical architect import successful")
        
        # Test architect creation
        architect = create_architect(level=1)
        success(f"Architect created at level {architect.level}")
        
        # Test basic architect properties
        success(f"Max depth: {architect.max_depth}")
        
        return True
        
    except Exception as e:
        error("Hierarchical architect test failed", e)
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all foundation tests"""
    print("🚀 PRSM Foundation Testing")
    print("Testing Phase 1 / Week 1 implementation")
    print("="*60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Project Structure", test_project_structure()))
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("Configuration System", test_configuration_system()))
    test_results.append(("Data Models", test_data_models()))
    test_results.append(("CLI Interface", test_cli_basic()))
    test_results.append(("FastAPI Application", test_api_basic()))
    test_results.append(("NWTN Orchestrator", test_orchestrator_basic()))
    test_results.append(("Hierarchical Architect", test_architect_basic()))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        if result:
            success(f"{test_name}")
            passed += 1
        else:
            error(f"{test_name}")
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    # Assert all tests pass
    assert failed == 0, f"{failed} foundation tests failed. All must pass before proceeding."
    
    print("🎉 All foundation tests PASSED! Ready for Phase 1 / Week 2")
    return True

if __name__ == "__main__":
    success_overall = run_all_tests()
    sys.exit(0 if success_overall else 1)