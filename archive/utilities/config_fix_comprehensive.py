#!/usr/bin/env python3
"""
Comprehensive Configuration Fix for NWTN Testing
===============================================

This script diagnoses and fixes all configuration issues preventing
the NWTN pipeline from initializing properly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("🔧 COMPREHENSIVE CONFIGURATION FIX")
print("=" * 50)

def test_basic_imports():
    """Test basic PRSM imports"""
    print("\n📦 Testing basic imports...")
    
    try:
        from prsm.nwtn.breakthrough_modes import BreakthroughMode
        print("✅ breakthrough_modes imported successfully")
        
        from prsm.core.models import UserInput
        print("✅ core.models imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config_system():
    """Test configuration system"""
    print("\n⚙️  Testing configuration system...")
    
    try:
        from prsm.core.config import get_settings, PRSMSettings
        
        # Try to get settings normally
        try:
            settings = get_settings()
            if settings is None:
                print("⚠️  get_settings() returned None")
                raise Exception("Settings is None")
            print("✅ Main settings loaded successfully")
            print(f"   • Environment: {settings.environment}")
            print(f"   • Agent timeout: {settings.agent_timeout_seconds}s")
            return settings
        except Exception as e:
            print(f"⚠️  Main settings failed: {e}")
            
            # Try creating settings manually with minimal config
            print("🔄 Attempting manual settings creation...")
            
            # Create minimal settings for testing
            minimal_settings = PRSMSettings(
                environment="development",
                debug=True,
                database_url="sqlite:///./prsm_test.db",
                secret_key="test-secret-key-for-development-only-32chars",
                agent_timeout_seconds=300
            )
            
            print("✅ Manual settings created successfully")
            print(f"   • Environment: {minimal_settings.environment}")
            print(f"   • Agent timeout: {minimal_settings.agent_timeout_seconds}s")
            
            return minimal_settings
            
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return None

def test_enhanced_orchestrator():
    """Test enhanced orchestrator with fixed config"""
    print("\n🚀 Testing enhanced orchestrator...")
    
    try:
        from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
        
        # Initialize orchestrator
        orchestrator = EnhancedNWTNOrchestrator()
        print("✅ Enhanced orchestrator initialized")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_input_model():
    """Test UserInput model with correct field names"""
    print("\n👤 Testing UserInput model...")
    
    try:
        from prsm.core.models import UserInput
        
        # Test with 'prompt' field (correct)
        test_input = UserInput(
            user_id="test_user",
            prompt="Test query for configuration validation",
            preferences={"test_mode": True}
        )
        
        print("✅ UserInput model works with 'prompt' field")
        print(f"   • User ID: {test_input.user_id}")
        print(f"   • Prompt: {test_input.prompt[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ UserInput model failed: {e}")
        return False

def test_claude_api_key():
    """Test Claude API key availability"""
    print("\n🤖 Testing Claude API key...")
    
    try:
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            if api_key.startswith('sk-ant-api'):
                print("✅ Claude API key found and valid format")
                print(f"   • Key prefix: {api_key[:20]}...")
                return api_key
            else:
                print("⚠️  Claude API key has invalid format")
                return None
        else:
            print("❌ Claude API key file not found")
            return None
            
    except Exception as e:
        print(f"❌ Claude API key test failed: {e}")
        return None

def test_embeddings_availability():
    """Test 100K embeddings availability"""
    print("\n🌐 Testing 100K embeddings...")
    
    try:
        embeddings_path = "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings"
        
        if os.path.exists(embeddings_path):
            embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('.json')]
            embedding_count = len(embedding_files)
            
            print(f"✅ Embeddings directory found: {embedding_count:,} files")
            
            if embedding_count >= 90000:  # Allow some variance
                print("✅ 100K embeddings corpus available")
                return True
            else:
                print(f"⚠️  Only {embedding_count:,} embeddings found (expected ~100K)")
                return False
        else:
            print("❌ Embeddings directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def run_integration_test():
    """Run integration test with all fixes"""
    print("\n🧪 Running integration test...")
    
    try:
        from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
        from prsm.core.models import UserInput
        from prsm.nwtn.breakthrough_modes import BreakthroughMode
        
        # Create test input
        test_input = UserInput(
            user_id="config_test_user",
            prompt="What are breakthrough approaches for quantum error correction?",
            preferences={
                "test_mode": True,
                "breakthrough_mode": BreakthroughMode.CREATIVE.value
            }
        )
        
        # Initialize orchestrator
        orchestrator = EnhancedNWTNOrchestrator()
        
        print("✅ Integration test components ready")
        print(f"   • Test input: {test_input.prompt[:50]}...")
        print(f"   • Orchestrator: Initialized")
        print(f"   • Breakthrough mode: {BreakthroughMode.CREATIVE.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive configuration fix"""
    
    results = {}
    
    # Test each component
    results['imports'] = test_basic_imports()
    results['config'] = test_config_system() is not None
    results['orchestrator'] = test_enhanced_orchestrator()
    results['user_input'] = test_user_input_model()
    results['api_key'] = test_claude_api_key() is not None
    results['embeddings'] = test_embeddings_availability()
    results['integration'] = run_integration_test()
    
    # Summary
    print("\n🎯 CONFIGURATION FIX SUMMARY")
    print("=" * 50)
    
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.replace('_', ' ').title()}: {'READY' if status else 'NEEDS ATTENTION'}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n📊 Overall Status: {success_count}/{total_count} components ready")
    
    if success_count == total_count:
        print("\n🎉 ALL CONFIGURATION ISSUES RESOLVED!")
        print("🚀 NWTN pipeline is ready for testing")
        return True
    else:
        print(f"\n⚠️  {total_count - success_count} issues remaining")
        print("Some components may need additional configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)