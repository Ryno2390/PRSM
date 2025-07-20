#!/usr/bin/env python3
"""
Setup Anthropic API Credentials
===============================

Register the Anthropic API key in the secure credential system for NWTN testing.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from prsm.integrations.security.secure_config_manager import SecureConfigManager


async def setup_anthropic_credentials():
    """Setup Anthropic API credentials for testing"""
    
    print("🔐 Setting up Anthropic API credentials...")
    
    # Initialize secure config manager
    config_manager = SecureConfigManager()
    
    # Initialize secure configuration system
    print("🔧 Initializing secure configuration system...")
    init_success = await config_manager.initialize_secure_configuration()
    
    if not init_success:
        print("❌ Failed to initialize secure configuration system")
        return False
    
    print("✅ Secure configuration system initialized")
    
    # Register Anthropic API credentials
    anthropic_api_key = "your-api-key-here"
    
    credentials = {
        "api_key": anthropic_api_key
    }
    
    print("🔑 Registering Anthropic API credentials...")
    success = await config_manager.register_api_credentials(
        platform="anthropic",
        credentials=credentials,
        user_id="system"  # System-level credentials
    )
    
    if success:
        print("✅ Anthropic API credentials registered successfully!")
        return True
    else:
        print("❌ Failed to register Anthropic API credentials")
        return False


async def main():
    """Main setup function"""
    print("🚀 ANTHROPIC API CREDENTIAL SETUP")
    print("=" * 50)
    
    success = await setup_anthropic_credentials()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SETUP COMPLETE!")
        print("✅ Anthropic API credentials are now available for NWTN")
        print("🧠 Ready for full end-to-end NWTN testing")
    else:
        print("❌ SETUP FAILED!")
        print("🔧 Please check error messages above")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())