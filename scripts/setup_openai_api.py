#!/usr/bin/env python3
"""
OpenAI API Setup Script
=======================

Quick setup script to configure OpenAI API credentials for PRSM testing.
This script helps you get started with real OpenAI integration testing.

Usage:
    python scripts/setup_openai_api.py --interactive
    python scripts/setup_openai_api.py --api-key YOUR_KEY
"""

import os
import sys
from pathlib import Path

import click

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_env_file(api_key: str):
    """Create or update .env file with API key"""
    env_file = Path.cwd() / ".env"
    
    # Read existing .env if it exists
    existing_lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            existing_lines = f.readlines()
    
    # Check if OPENAI_API_KEY already exists
    openai_key_exists = any(line.startswith('OPENAI_API_KEY=') for line in existing_lines)
    
    if openai_key_exists:
        # Update existing line
        updated_lines = []
        for line in existing_lines:
            if line.startswith('OPENAI_API_KEY='):
                updated_lines.append(f'OPENAI_API_KEY={api_key}\n')
            else:
                updated_lines.append(line)
        
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        click.echo(f"✅ Updated OPENAI_API_KEY in {env_file}")
    else:
        # Add new line
        with open(env_file, 'a') as f:
            if existing_lines and not existing_lines[-1].endswith('\n'):
                f.write('\n')
            f.write(f'OPENAI_API_KEY={api_key}\n')
        
        click.echo(f"✅ Added OPENAI_API_KEY to {env_file}")


def validate_api_key_format(api_key: str) -> bool:
    """Basic validation of API key format"""
    return (
        api_key.startswith('sk-') and 
        len(api_key) > 20 and 
        all(c.isalnum() or c in '-_' for c in api_key)
    )


@click.command()
@click.option('--api-key', '-k', help='OpenAI API key')
@click.option('--interactive', '-i', is_flag=True, help='Interactive setup')
@click.option('--test', '-t', is_flag=True, help='Test API key after setup')
@click.option('--env-only', is_flag=True, help='Only set up .env file (skip PRSM credential system)')
def main(api_key: str, interactive: bool, test: bool, env_only: bool):
    """Set up OpenAI API credentials for PRSM"""
    
    click.echo("🔧 OpenAI API Setup for PRSM")
    click.echo("=" * 40)
    
    # Get API key
    if interactive and not api_key:
        click.echo("\n📝 To get an OpenAI API key:")
        click.echo("1. Go to https://platform.openai.com/api-keys")
        click.echo("2. Sign in or create an account")
        click.echo("3. Click 'Create new secret key'")
        click.echo("4. Copy the key (starts with 'sk-')")
        
        api_key = click.prompt('\n🔑 Enter your OpenAI API key', hide_input=True)
    
    if not api_key:
        # Try to get from environment
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            click.echo("❌ No API key provided. Use --api-key or --interactive")
            sys.exit(1)
        else:
            click.echo("✅ Found API key in environment")
    
    # Validate API key format
    if not validate_api_key_format(api_key):
        click.echo("⚠️  Warning: API key format doesn't look correct")
        click.echo("   OpenAI keys should start with 'sk-' and be ~51 characters")
        
        if interactive:
            if not click.confirm("Continue anyway?"):
                sys.exit(1)
    
    # Set up .env file
    setup_env_file(api_key)
    
    # Set up PRSM credential system (unless env-only)
    if not env_only:
        try:
            click.echo("\n🔒 Setting up PRSM credential system...")
            
            # This is where we'd integrate with the PRSM credential system
            # For now, we'll just set the environment variable
            os.environ['OPENAI_API_KEY'] = api_key
            click.echo("✅ Environment variable set for current session")
            
        except Exception as e:
            click.echo(f"⚠️  PRSM credential setup failed: {e}")
            click.echo("   You can still use the .env file for testing")
    
    # Test the API key if requested
    if test:
        click.echo("\n🧪 Testing API key...")
        
        try:
            import asyncio
            from prsm.agents.executors.enhanced_openai_client import create_enhanced_openai_client
            from prsm.agents.executors.api_clients import ModelExecutionRequest, ModelProvider
            
            async def test_api():
                client = await create_enhanced_openai_client(api_key, budget_limit_usd=0.10)
                
                try:
                    request = ModelExecutionRequest(
                        prompt="Say 'Hello, PRSM!' in exactly three words.",
                        model_id="gpt-3.5-turbo",
                        provider=ModelProvider.OPENAI,
                        max_tokens=10,
                        temperature=0.0
                    )
                    
                    response = await client.execute(request)
                    
                    if response.success:
                        click.echo(f"✅ API test successful!")
                        click.echo(f"   Response: {response.content}")
                        click.echo(f"   Cost: ${response.metadata.get('cost_usd', 0):.4f}")
                    else:
                        click.echo(f"❌ API test failed: {response.error}")
                        return False
                    
                finally:
                    await client.close()
                
                return True
            
            result = asyncio.run(test_api())
            
            if result:
                click.echo("🎉 OpenAI API is working correctly with PRSM!")
            else:
                click.echo("❌ API test failed. Please check your API key.")
                sys.exit(1)
                
        except ImportError as e:
            click.echo(f"⚠️  Cannot test API - missing dependencies: {e}")
            click.echo("   Run 'pip install -r requirements.txt' first")
        except Exception as e:
            click.echo(f"❌ API test failed with error: {e}")
            sys.exit(1)
    
    click.echo("\n🎯 Next Steps:")
    click.echo("1. Test the basic integration:")
    click.echo("   python scripts/test_openai_integration.py --batch-test")
    click.echo("2. Test the enhanced client:")
    click.echo("   python scripts/test_enhanced_openai_client.py --test-all")
    click.echo("3. Start using OpenAI in your PRSM applications!")


if __name__ == '__main__':
    main()