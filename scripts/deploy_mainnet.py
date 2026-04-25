#!/usr/bin/env python3
"""
PRSM Polygon Mainnet Deployment Script
======================================

Production deployment script for deploying FTNS smart contracts
to Polygon mainnet with comprehensive security and monitoring.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

import click
import structlog
from getpass import getpass

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.web3.mainnet_deployer import get_mainnet_deployer
from prsm.web3.mainnet_config import get_mainnet_config_manager

# Set up logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
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
    """PRSM Polygon Mainnet Deployment Tool"""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--private-key', '-k', help='Deployment wallet private key (will prompt if not provided)')
@click.option('--contracts', '-c', help='Comma-separated list of contracts to deploy (default: all)')
@click.option('--verify/--no-verify', default=True, help='Enable/disable contract verification')
@click.option('--gas-limit', type=int, default=300, help='Maximum gas price limit in Gwei')
@click.option('--dry-run', is_flag=True, help='Perform a dry run without actual deployment')
def deploy(private_key, contracts, verify, gas_limit, dry_run):
    """Deploy FTNS contracts to Polygon mainnet"""
    
    async def _deploy():
        try:
            click.echo("🚀 PRSM Polygon Mainnet Deployment")
            click.echo("=" * 60)
            
            if dry_run:
                click.echo("🧪 DRY RUN MODE - No actual deployment will occur")
                click.echo()
            
            # Get private key securely
            if not private_key:
                private_key = getpass("Enter deployment wallet private key: ")
            
            if not private_key or len(private_key) != 66 or not private_key.startswith('0x'):
                click.echo("❌ Invalid private key format. Must be 66 characters starting with 0x")
                return False
            
            # Parse contracts list
            contracts_list = None
            if contracts:
                contracts_list = [c.strip() for c in contracts.split(',')]
                click.echo(f"📋 Deploying specific contracts: {contracts_list}")
            else:
                click.echo("📋 Deploying all contracts: FTNS Token, Marketplace, Governance, Timelock")
            
            # Initialize deployer
            deployer = get_mainnet_deployer()
            
            if dry_run:
                click.echo("✅ Dry run completed - deployment configuration validated")
                return True
            
            # Confirm deployment
            click.echo()
            click.echo("⚠️  MAINNET DEPLOYMENT CONFIRMATION")
            click.echo("This will deploy contracts to Polygon mainnet using real MATIC tokens.")
            click.echo("Gas costs will be incurred and contracts will be permanently deployed.")
            click.echo()
            
            if not click.confirm("Are you sure you want to proceed with mainnet deployment?"):
                click.echo("❌ Deployment cancelled by user")
                return False
            
            click.echo()
            click.echo("🚀 Starting mainnet deployment...")
            
            # Execute deployment
            deployment_result = await deployer.deploy_to_mainnet(
                deployer_private_key=private_key,
                contracts_to_deploy=contracts_list,
                verification_enabled=verify
            )
            
            # Display results
            if deployment_result["status"] == "completed":
                click.echo()
                click.echo("🎉 MAINNET DEPLOYMENT SUCCESSFUL!")
                click.echo("=" * 50)
                
                # Display deployed contracts
                deployed_contracts = deployment_result.get("contracts_deployed", {})
                for contract_name, contract_info in deployed_contracts.items():
                    click.echo(f"✅ {contract_name}")
                    click.echo(f"   Address: {contract_info['address']}")
                    click.echo(f"   TX Hash: {contract_info['transaction_hash']}")
                    click.echo(f"   Gas Used: {contract_info['gas_used']:,}")
                    click.echo()
                
                # Display verification results
                verification_results = deployment_result.get("verification_results", {})
                if verification_results:
                    click.echo("🔍 Contract Verification Results:")
                    for contract_name, verification_info in verification_results.items():
                        status = "✅ Verified" if verification_info.get("verified") else "❌ Failed"
                        click.echo(f"   {contract_name}: {status}")
                        if verification_info.get("verification_url"):
                            click.echo(f"     URL: {verification_info['verification_url']}")
                    click.echo()
                
                # Display deployment summary
                click.echo("📊 Deployment Summary:")
                click.echo(f"   Deployment ID: {deployment_result['deployment_id']}")
                click.echo(f"   Duration: {deployment_result.get('duration_seconds', 0):.1f} seconds")
                click.echo(f"   Total Contracts: {len(deployed_contracts)}")
                click.echo(f"   Status: {deployment_result['status']}")
                
                # Initialize configuration
                if deployed_contracts:
                    click.echo()
                    click.echo("⚙️ Initializing mainnet configuration...")
                    
                    config_manager = get_mainnet_config_manager()
                    
                    # Extract deployer address safely
                    from eth_account import Account
                    deployer_address = Account.from_key(private_key).address
                    
                    config_result = await config_manager.initialize_mainnet_config(
                        deployment_results=deployment_result,
                        deployer_address=deployer_address
                    )
                    
                    if config_result["success"]:
                        click.echo("✅ Mainnet configuration initialized")
                        click.echo(f"   Config ID: {config_result['config_id']}")
                        click.echo(f"   Contracts Configured: {config_result['contracts_configured']}")
                    else:
                        click.echo(f"❌ Configuration failed: {config_result['error']}")
                
                click.echo()
                click.echo("🎯 NEXT STEPS:")
                click.echo("1. Update your environment variables with the new contract addresses")
                click.echo("2. Configure your frontend/applications to use mainnet")
                click.echo("3. Run health checks to verify deployment")
                click.echo("4. Begin governance token distribution")
                
                return True
                
            elif deployment_result["status"] == "partially_failed":
                click.echo()
                click.echo("⚠️ DEPLOYMENT PARTIALLY FAILED")
                click.echo("Some contracts deployed successfully, but issues occurred:")
                
                errors = deployment_result.get("errors", [])
                for error in errors:
                    click.echo(f"❌ {error}")
                
                warnings = deployment_result.get("warnings", [])
                for warning in warnings:
                    click.echo(f"⚠️ {warning}")
                
                return False
                
            else:
                click.echo()
                click.echo("❌ DEPLOYMENT FAILED")
                click.echo("Deployment encountered critical errors:")
                
                errors = deployment_result.get("errors", [])
                for error in errors:
                    click.echo(f"❌ {error}")
                
                return False
            
        except Exception as e:
            logger.error("Deployment failed", error=str(e))
            click.echo(f"❌ Deployment failed: {e}")
            return False
    
    success = asyncio.run(_deploy())
    sys.exit(0 if success else 1)


@cli.command()
def status():
    """Check mainnet deployment status"""
    
    async def _status():
        try:
            click.echo("📊 PRSM Mainnet Deployment Status")
            click.echo("=" * 40)
            
            deployer = get_mainnet_deployer()
            deployment_status = await deployer.get_deployment_status()
            
            click.echo(f"Deployment ID: {deployment_status.get('deployment_id', 'N/A')}")
            click.echo(f"Contracts Deployed: {deployment_status.get('contracts_deployed', 0)}")
            
            # Display deployment addresses
            addresses = deployment_status.get('deployment_addresses', {})
            if addresses:
                click.echo()
                click.echo("📋 Deployed Contract Addresses:")
                for contract_name, address in addresses.items():
                    if address:
                        click.echo(f"   {contract_name}: {address}")
                        click.echo(f"      PolygonScan: https://polygonscan.com/address/{address}")
            else:
                click.echo()
                click.echo("ℹ️ No contracts deployed yet")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get status: {e}")
            return False
    
    success = asyncio.run(_status())
    sys.exit(0 if success else 1)


@cli.command()
def config():
    """View mainnet configuration"""
    
    async def _config():
        try:
            click.echo("⚙️ PRSM Mainnet Configuration")
            click.echo("=" * 35)
            
            config_manager = get_mainnet_config_manager()
            config_result = await config_manager.get_mainnet_config()
            
            if config_result["success"]:
                config = config_result["config"]
                
                click.echo(f"Config ID: {config_result['config_id']}")
                click.echo(f"Last Updated: {config_result['last_updated']}")
                click.echo(f"Contracts Count: {config_result['contracts_count']}")
                click.echo()
                
                # Network configuration
                network = config.get("network", {})
                click.echo("🌐 Network Configuration:")
                click.echo(f"   Name: {network.get('name', 'N/A')}")
                click.echo(f"   Chain ID: {network.get('chain_id', 'N/A')}")
                click.echo(f"   RPC URL: {network.get('rpc_url', 'N/A')}")
                click.echo(f"   Explorer: {network.get('explorer_url', 'N/A')}")
                click.echo()
                
                # Contracts
                contracts = config.get("contracts", {})
                if contracts:
                    click.echo("📜 Smart Contracts:")
                    for contract_name, address in contracts.items():
                        click.echo(f"   {contract_name}: {address}")
                click.echo()
                
                # Deployment info
                deployment = config.get("deployment", {})
                if deployment:
                    click.echo("🚀 Deployment Information:")
                    click.echo(f"   Deployment ID: {deployment.get('deployment_id', 'N/A')}")
                    click.echo(f"   Deployed At: {deployment.get('deployed_at', 'N/A')}")
                    click.echo(f"   Deployer: {deployment.get('deployer_address', 'N/A')}")
                    click.echo(f"   Status: {deployment.get('status', 'N/A')}")
                
            else:
                click.echo("ℹ️ No mainnet configuration found")
                click.echo("Run deployment first to create configuration")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to get configuration: {e}")
            return False
    
    success = asyncio.run(_config())
    sys.exit(0 if success else 1)


@cli.command()
def validate():
    """Validate mainnet configuration"""
    
    async def _validate():
        try:
            click.echo("✅ PRSM Mainnet Configuration Validation")
            click.echo("=" * 45)
            
            config_manager = get_mainnet_config_manager()
            validation_result = await config_manager.validate_configuration()
            
            # Display validation results
            click.echo(f"Overall Status: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
            click.echo(f"Total Checks: {validation_result.get('total_checks', 0)}")
            click.echo(f"Issues Found: {validation_result.get('issues_count', 0)}")
            click.echo(f"Warnings: {validation_result.get('warnings_count', 0)}")
            click.echo()
            
            # Display checks performed
            checks = validation_result.get('checks_performed', [])
            if checks:
                click.echo("🔍 Checks Performed:")
                for check in checks:
                    click.echo(f"   ✓ {check.replace('_', ' ').title()}")
                click.echo()
            
            # Display issues
            issues = validation_result.get('issues', [])
            if issues:
                click.echo("❌ Issues Found:")
                for issue in issues:
                    click.echo(f"   • {issue}")
                click.echo()
            
            # Display warnings
            warnings = validation_result.get('warnings', [])
            if warnings:
                click.echo("⚠️ Warnings:")
                for warning in warnings:
                    click.echo(f"   • {warning}")
                click.echo()
            
            if validation_result['valid']:
                click.echo("🎉 Configuration is valid and ready for production!")
            else:
                click.echo("🔧 Please address the issues above before using in production")
            
            return validation_result['valid']
            
        except Exception as e:
            click.echo(f"❌ Validation failed: {e}")
            return False
    
    success = asyncio.run(_validate())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--contract', '-c', required=True, help='Contract name to update')
@click.option('--address', '-a', required=True, help='New contract address')
@click.option('--reason', '-r', required=True, help='Reason for the update')
def update_address(contract, address, reason):
    """Update a contract address in configuration"""
    
    async def _update():
        try:
            click.echo(f"🔧 Updating {contract} address to {address}")
            click.echo(f"Reason: {reason}")
            click.echo()
            
            if not click.confirm("Are you sure you want to update this contract address?"):
                click.echo("❌ Update cancelled")
                return False
            
            config_manager = get_mainnet_config_manager()
            
            update_result = await config_manager.update_contract_address(
                contract_name=contract,
                new_address=address,
                update_reason=reason
            )
            
            if update_result["success"]:
                click.echo("✅ Contract address updated successfully!")
                click.echo(f"   Contract: {contract}")
                click.echo(f"   Old Address: {update_result.get('old_address', 'N/A')}")
                click.echo(f"   New Address: {update_result['new_address']}")
                click.echo(f"   Updated At: {update_result['update_record']['updated_at']}")
                
                return True
            else:
                click.echo(f"❌ Update failed: {update_result['error']}")
                return False
            
        except Exception as e:
            click.echo(f"❌ Update failed: {e}")
            return False
    
    success = asyncio.run(_update())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    cli()