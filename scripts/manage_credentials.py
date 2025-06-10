#!/usr/bin/env python3
"""
PRSM Credential Management CLI
==============================

Command-line tool for managing API credentials securely.
Provides easy credential registration, validation, and status checking.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import click
import structlog

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.integrations.security.secure_config_manager import secure_config_manager
from prsm.integrations.security.secure_api_client_factory import SecureClientType, secure_client_factory

# Set up logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """PRSM Credential Management CLI"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--platform', '-p', required=True, 
              type=click.Choice([e.value for e in SecureClientType]),
              help='Platform to register credentials for')
@click.option('--api-key', '-k', help='API key for the platform')
@click.option('--token', '-t', help='OAuth token (for GitHub)')
@click.option('--url', '-u', help='Service URL (for Weaviate, Ollama)')
@click.option('--environment', '-e', help='Environment (for Pinecone)')
@click.option('--user-id', help='User ID (defaults to system)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive credential entry')
def register(platform: str, api_key: str, token: str, url: str, environment: str, user_id: str, interactive: bool):
    """Register API credentials for a platform"""
    
    async def _register():
        try:
            click.echo(f"🔐 Registering credentials for {platform}")
            
            # Build credentials based on platform
            credentials = {}
            
            if platform == SecureClientType.GITHUB:
                if interactive and not token:
                    token = click.prompt("GitHub Access Token", hide_input=True)
                if token:
                    credentials["access_token"] = token
                else:
                    click.echo("❌ GitHub requires an access token")
                    return False
                    
            elif platform == SecureClientType.PINECONE:
                if interactive and not api_key:
                    api_key = click.prompt("Pinecone API Key", hide_input=True)
                if interactive and not environment:
                    environment = click.prompt("Pinecone Environment", default="us-west1-gcp")
                
                if api_key:
                    credentials["api_key"] = api_key
                    credentials["environment"] = environment or "us-west1-gcp"
                else:
                    click.echo("❌ Pinecone requires an API key")
                    return False
                    
            elif platform in [SecureClientType.WEAVIATE, SecureClientType.OLLAMA]:
                if interactive and not url:
                    default_url = "http://localhost:8080" if platform == SecureClientType.WEAVIATE else "http://localhost:11434"
                    url = click.prompt("Service URL", default=default_url)
                
                credentials["url"] = url or ("http://localhost:8080" if platform == SecureClientType.WEAVIATE else "http://localhost:11434")
                
                if platform == SecureClientType.WEAVIATE and (interactive or api_key):
                    if interactive and not api_key:
                        api_key = click.prompt("Weaviate API Key (optional)", default="", hide_input=True)
                    if api_key:
                        credentials["api_key"] = api_key
                        
            else:
                # Standard API key platforms (OpenAI, Anthropic, HuggingFace)
                if interactive and not api_key:
                    api_key = click.prompt(f"{platform.title()} API Key", hide_input=True)
                
                if api_key:
                    credentials["api_key"] = api_key
                else:
                    click.echo(f"❌ {platform} requires an API key")
                    return False
            
            if not credentials:
                click.echo("❌ No credentials provided")
                return False
            
            # Register credentials
            success = await secure_config_manager.register_api_credentials(
                platform=platform,
                credentials=credentials,
                user_id=user_id
            )
            
            if success:
                click.echo(f"✅ Credentials registered successfully for {platform}")
                if user_id:
                    click.echo(f"   User: {user_id}")
                else:
                    click.echo("   Level: System")
                return True
            else:
                click.echo(f"❌ Failed to register credentials for {platform}")
                return False
                
        except Exception as e:
            logger.error("Failed to register credentials", error=str(e))
            click.echo(f"❌ Error: {e}")
            return False
    
    result = asyncio.run(_register())
    sys.exit(0 if result else 1)


@cli.command()
@click.option('--platform', '-p', 
              type=click.Choice([e.value for e in SecureClientType]),
              help='Platform to validate (validates all if not specified)')
@click.option('--user-id', help='User ID (defaults to system)')
def validate(platform: str, user_id: str):
    """Validate API credentials for platforms"""
    
    async def _validate():
        try:
            if platform:
                platforms = [SecureClientType(platform)]
            else:
                platforms = list(SecureClientType)
            
            click.echo("🔍 Validating credentials...")
            
            results = {}
            for p in platforms:
                is_valid = await secure_client_factory.validate_client_credentials(
                    p, user_id or "system"
                )
                results[p.value] = is_valid
                
                status = "✅ Valid" if is_valid else "❌ Invalid/Missing"
                click.echo(f"   {p.value:12} {status}")
            
            valid_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            
            click.echo(f"\n📊 Summary: {valid_count}/{total_count} platforms have valid credentials")
            
            return valid_count > 0
            
        except Exception as e:
            logger.error("Failed to validate credentials", error=str(e))
            click.echo(f"❌ Error: {e}")
            return False
    
    result = asyncio.run(_validate())
    sys.exit(0 if result else 1)


@cli.command()
@click.option('--user-id', help='User ID (defaults to system)')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
def status(user_id: str, json_output: bool):
    """Get credential status"""
    
    async def _status():
        try:
            status_data = await secure_config_manager.get_secure_configuration_status()
            
            if json_output:
                click.echo(json.dumps(status_data, indent=2, default=str))
                return True
            
            click.echo("🔐 PRSM Credential Management Status")
            click.echo("=" * 50)
            
            # System status
            click.echo(f"Migration Completed: {'✅ Yes' if status_data.get('migration_completed') else '❌ No'}")
            click.echo(f"System Secrets Secure: {'✅ Yes' if status_data.get('system_secrets_secure') else '❌ No'}")
            click.echo(f"Credential Manager Available: {'✅ Yes' if status_data.get('credential_manager_available') else '❌ No'}")
            
            # Platform status
            click.echo("\n📱 Platform Credentials:")
            platform_creds = status_data.get('platform_credentials', {})
            
            for platform, info in platform_creds.items():
                has_creds = info.get('credentials_available', False)
                status_icon = "✅" if has_creds else "❌"
                click.echo(f"   {platform:12} {status_icon}")
            
            # Statistics
            total_platforms = len(platform_creds)
            platforms_with_creds = sum(1 for info in platform_creds.values() if info.get('credentials_available'))
            
            click.echo(f"\n📊 Statistics:")
            click.echo(f"   Total Platforms: {total_platforms}")
            click.echo(f"   With Credentials: {platforms_with_creds}")
            click.echo(f"   Coverage: {platforms_with_creds/total_platforms*100:.1f}%" if total_platforms > 0 else "   Coverage: 0%")
            
            return True
            
        except Exception as e:
            logger.error("Failed to get status", error=str(e))
            click.echo(f"❌ Error: {e}")
            return False
    
    result = asyncio.run(_status())
    sys.exit(0 if result else 1)


@cli.command()
def initialize():
    """Initialize secure configuration system"""
    
    async def _initialize():
        try:
            click.echo("🔧 Initializing secure configuration system...")
            
            success = await secure_config_manager.initialize_secure_configuration()
            
            if success:
                click.echo("✅ Secure configuration system initialized successfully")
                click.echo("\n📋 Next Steps:")
                click.echo("1. Register your API credentials using:")
                click.echo("   python manage_credentials.py register -p openai -i")
                click.echo("2. Validate credentials using:")
                click.echo("   python manage_credentials.py validate")
                click.echo("3. Check status using:")
                click.echo("   python manage_credentials.py status")
                return True
            else:
                click.echo("❌ Failed to initialize secure configuration system")
                return False
                
        except Exception as e:
            logger.error("Failed to initialize", error=str(e))
            click.echo(f"❌ Error: {e}")
            return False
    
    result = asyncio.run(_initialize())
    sys.exit(0 if result else 1)


@cli.command()
def migrate():
    """Migrate environment variables to secure storage"""
    
    async def _migrate():
        try:
            click.echo("🔄 Migrating environment variables to secure storage...")
            
            # Check for environment variables
            env_vars = {
                'OPENAI_API_KEY': 'openai',
                'ANTHROPIC_API_KEY': 'anthropic', 
                'HUGGINGFACE_API_KEY': 'huggingface',
                'GITHUB_ACCESS_TOKEN': 'github',
                'PINECONE_API_KEY': 'pinecone'
            }
            
            found_vars = {}
            for env_var, platform in env_vars.items():
                value = os.getenv(env_var)
                if value:
                    found_vars[platform] = value
            
            if not found_vars:
                click.echo("ℹ️  No environment variables found to migrate")
                return True
            
            click.echo(f"🔍 Found {len(found_vars)} environment variables:")
            for platform in found_vars.keys():
                click.echo(f"   {platform}")
            
            if not click.confirm("\nProceed with migration?"):
                click.echo("Migration cancelled")
                return False
            
            # Perform migration via initialization
            success = await secure_config_manager.initialize_secure_configuration()
            
            if success:
                click.echo("✅ Migration completed successfully")
                click.echo("\n⚠️  Security Recommendation:")
                click.echo("   Remove environment variables from your .env file")
                click.echo("   Credentials are now managed securely")
                return True
            else:
                click.echo("❌ Migration failed")
                return False
                
        except Exception as e:
            logger.error("Failed to migrate", error=str(e))
            click.echo(f"❌ Error: {e}")
            return False
    
    result = asyncio.run(_migrate())
    sys.exit(0 if result else 1)


@cli.command()
def list_platforms():
    """List all supported platforms"""
    
    click.echo("🌐 Supported Platforms:")
    click.echo("=" * 30)
    
    for client_type in SecureClientType:
        # Get required fields
        if client_type == SecureClientType.GITHUB:
            required = "access_token"
        elif client_type == SecureClientType.PINECONE:
            required = "api_key, environment"
        elif client_type in [SecureClientType.WEAVIATE, SecureClientType.OLLAMA]:
            required = "url"
        else:
            required = "api_key"
        
        click.echo(f"   {client_type.value:12} (requires: {required})")
    
    click.echo(f"\n📊 Total: {len(SecureClientType)} platforms supported")


if __name__ == '__main__':
    cli()