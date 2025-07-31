#!/usr/bin/env python3
"""
Configuration Fix for NWTN Testing
==================================

Creates a minimal configuration to resolve the agent timeout issues.
"""

from pydantic import BaseModel
from typing import Optional

class MinimalConfig:
    """Minimal configuration for testing"""
    
    def __init__(self):
        self.agent_timeout_seconds = 300
        self.environment = "development"
        self.debug = True
        
    def __getattr__(self, name):
        # Return reasonable defaults for any missing attributes
        defaults = {
            'agent_timeout_seconds': 300,
            'database_url': 'sqlite:///./test.db',
            'secret_key': 'test-secret-key-for-development-only-32chars',
            'api_host': '127.0.0.1',
            'api_port': 8000,
            'debug': True,
            'environment': 'development'
        }
        return defaults.get(name, None)

# Create a minimal settings instance for testing
minimal_settings = MinimalConfig()

def get_test_settings():
    """Get minimal settings for testing"""
    return minimal_settings

if __name__ == "__main__":
    settings = get_test_settings()
    print(f"Agent timeout: {settings.agent_timeout_seconds}")
    print(f"Environment: {settings.environment}")
    print("✅ Minimal config working")