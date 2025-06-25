#!/usr/bin/env python3
"""
PRSM FTNS Contract Deployment CLI

Simple command-line interface for deploying FTNS smart contracts
to Polygon networks.
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.web3.contract_deployer import FTNSContractDeployer, deploy_to_mumbai, get_latest_deployment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main deployment CLI"""
    parser = argparse.ArgumentParser(description='Deploy PRSM FTNS smart contracts')
    parser.add_argument('action', choices=['deploy', 'status', 'instructions'], 
                       help='Action to perform')
    parser.add_argument('--network', default='polygon_mumbai', 
                       choices=['polygon_mumbai', 'polygon_mainnet'],
                       help='Target network')
    parser.add_argument('--private-key', 
                       help='Deployment wallet private key (or set PRIVATE_KEY env var)')
    parser.add_argument('--treasury', 
                       help='Treasury address (defaults to deployer address)')
    parser.add_argument('--mock', action='store_true',
                       help='Create mock deployment for testing')
    
    args = parser.parse_args()
    
    if args.action == 'deploy':
        await handle_deploy(args)
    elif args.action == 'status':
        await handle_status(args)
    elif args.action == 'instructions':
        await handle_instructions(args)

async def handle_deploy(args):
    """Handle deployment command"""
    try:
        # Get private key
        private_key = args.private_key or os.getenv('PRIVATE_KEY')
        if not private_key:
            logger.error("Private key required. Use --private-key or set PRIVATE_KEY environment variable")
            return False
        
        logger.info(f"🚀 Starting deployment to {args.network}")
        
        if args.network == 'polygon_mumbai':
            deployment = await deploy_to_mumbai(private_key, args.treasury)
        else:
            logger.error("Mainnet deployment not yet implemented")
            return False
        
        if deployment:
            logger.info("✅ Deployment completed successfully!")
            print("\n" + "="*60)
            print("🎉 DEPLOYMENT SUMMARY")
            print("="*60)
            print(f"Network: {deployment['network']}")
            print(f"Chain ID: {deployment['chain_id']}")
            print(f"Deployer: {deployment['deployer']}")
            print(f"Treasury: {deployment['treasury']}")
            print(f"FTNS Token: {deployment['contracts']['ftns_token']}")
            print(f"Deployment Type: {deployment['deployment_type']}")
            print(f"Timestamp: {deployment['timestamp']}")
            
            print("\n📋 Next Steps:")
            print("1. Update PRSM .env file:")
            print(f"   FTNS_TOKEN_ADDRESS={deployment['contracts']['ftns_token']}")
            print(f"   WEB3_NETWORK={args.network}")
            print("   WEB3_MONITORING_ENABLED=true")
            
            print("\n2. Test integration:")
            print("   python scripts/test_web3_integration.py")
            
            print("\n3. Start PRSM with Web3 support:")
            print("   python -m uvicorn prsm.api.main:app --reload")
            
            if deployment['deployment_type'] == 'mock':
                print("\n⚠️  Note: This is a MOCK deployment for development.")
                print("   For production, implement actual smart contract deployment.")
            
            return True
        else:
            logger.error("❌ Deployment failed")
            return False
            
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        return False

async def handle_status(args):
    """Handle status command"""
    try:
        logger.info(f"📊 Checking deployment status for {args.network}")
        
        deployment = await get_latest_deployment(args.network)
        
        if deployment:
            print("\n" + "="*60)
            print("📊 DEPLOYMENT STATUS")
            print("="*60)
            print(f"Network: {deployment['network']}")
            print(f"Contract Address: {deployment['contracts']['ftns_token']}")
            print(f"Deployer: {deployment['deployer']}")
            print(f"Deployed: {deployment['timestamp']}")
            print(f"Type: {deployment['deployment_type']}")
            print(f"Verified: {deployment.get('verified', False)}")
            
            if deployment['deployment_type'] == 'mock':
                print("\n⚠️  This is a mock deployment for testing")
            
            # Check if integrated with PRSM
            env_token = os.getenv('FTNS_TOKEN_ADDRESS')
            if env_token and env_token.lower() == deployment['contracts']['ftns_token'].lower():
                print("✅ Integrated with PRSM")
            else:
                print("⚠️  Not integrated with PRSM (update .env file)")
                
        else:
            print(f"❌ No deployments found for {args.network}")
            
    except Exception as e:
        logger.error(f"Status check error: {e}")

async def handle_instructions(args):
    """Handle instructions command"""
    try:
        deployer = FTNSContractDeployer()
        instructions = deployer.get_deployment_instructions(args.network)
        print(instructions)
        
    except Exception as e:
        logger.error(f"Error getting instructions: {e}")

def check_environment():
    """Check if environment is properly setup"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check required modules
    try:
        import web3
    except ImportError:
        issues.append("web3 module not installed (pip install web3)")
    
    try:
        import eth_account
    except ImportError:
        issues.append("eth_account module not installed (pip install eth-account)")
    
    # Check environment variables
    if not os.getenv('POLYGON_MUMBAI_RPC_URL'):
        issues.append("POLYGON_MUMBAI_RPC_URL not set (will use default)")
    
    if issues:
        print("⚠️  Environment Issues:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    
    return len(issues) == 0

if __name__ == '__main__':
    print("🚀 PRSM FTNS Contract Deployment CLI")
    print("="*60)
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print("Some environment issues detected (see above)")
        print("Continuing anyway...\n")
    
    # Run main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)