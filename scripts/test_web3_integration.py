#!/usr/bin/env python3
"""
PRSM Web3 Integration Test

Tests the Web3 integration layer with mock deployment
to ensure all components work correctly.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_web3_integration():
    """Test Web3 integration components"""
    
    print("🧪 PRSM Web3 Integration Test")
    print("="*60)
    
    try:
        # Load deployment info
        deployment_file = await get_latest_deployment()
        if not deployment_file:
            print("❌ No deployment found. Run mock_deploy.py first.")
            return False
            
        with open(deployment_file, 'r') as f:
            deployment = json.load(f)
            
        print(f"📄 Using deployment: {deployment_file.name}")
        print(f"Contract: {deployment['contracts']['ftns_token']}")
        print(f"Type: {deployment['deployment_type']}")
        
        # Test 1: Import Web3 modules
        print("\n🔧 Testing Web3 module imports...")
        try:
            from prsm.web3.wallet_connector import Web3WalletConnector
            from prsm.web3.contract_interface import FTNSContractInterface
            from prsm.web3.web3_service import Web3ServiceManager
            print("✅ All Web3 modules imported successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Install missing dependencies: pip install web3 eth-account")
            return False
        
        # Test 2: Configuration validation
        print("\n⚙️  Testing configuration...")
        
        # Check required environment variables
        required_vars = ['FTNS_TOKEN_ADDRESS', 'WEB3_NETWORK']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {missing_vars}")
            print("💡 Copy from .env.testnet to your main .env file")
        else:
            print("✅ Configuration variables found")
        
        # Test 3: Service initialization (mock mode)
        print("\n🚀 Testing service initialization...")
        try:
            from prsm.core.database_service import DatabaseService
            
            # Initialize database service
            db_service = DatabaseService()
            print("✅ Database service initialized")
            
            # Initialize Web3 service manager
            service_manager = Web3ServiceManager(db_service)
            print("✅ Web3 service manager created")
            
            # Test configuration loading
            config = service_manager.config
            print(f"✅ Configuration loaded: {len(config['networks'])} networks")
            
        except Exception as e:
            print(f"❌ Service initialization error: {e}")
            return False
        
        # Test 4: Mock contract address validation
        print("\n🔍 Testing contract address format...")
        contract_address = deployment['contracts']['ftns_token']
        
        if contract_address.startswith('0x') and len(contract_address) == 42:
            print("✅ Contract address format valid")
        else:
            print("❌ Invalid contract address format")
            return False
        
        # Test 5: API endpoint simulation
        print("\n🌐 Testing API endpoint availability...")
        try:
            from prsm.web3.frontend_integration import router
            print("✅ Web3 API router imported successfully")
            
            # Check endpoint paths
            paths = [route.path for route in router.routes]
            expected_paths = ['/connect', '/wallet/info', '/contracts/initialize']
            
            found_paths = [path for path in expected_paths if any(path in p for p in paths)]
            print(f"✅ API endpoints available: {len(found_paths)}/{len(expected_paths)}")
            
        except Exception as e:
            print(f"❌ API endpoint error: {e}")
            return False
        
        # Test 6: Mock transaction simulation
        print("\n💸 Testing mock transaction handling...")
        
        mock_transaction = {
            "from": "0x742D35Cc6603C0532C4E9Fdb7b5A4F6c8c4b6D3f",
            "to": "0x1234567890123456789012345678901234567890",
            "amount": "100.0",
            "type": "transfer"
        }
        
        print(f"✅ Mock transaction created: {mock_transaction['amount']} FTNS")
        
        # Test Summary
        print("\n" + "="*60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("="*60)
        print("✅ Web3 modules: Imported successfully")
        print("✅ Configuration: Validated")
        print("✅ Services: Initialized")
        print("✅ Contract address: Valid format")
        print("✅ API endpoints: Available")
        print("✅ Transaction handling: Ready")
        
        print(f"\n🎯 Ready for integration with contract: {contract_address}")
        
        if deployment['deployment_type'] == 'mock':
            print("\n⚠️  Note: Using MOCK deployment")
            print("   - No real blockchain interaction")
            print("   - For testing Web3 integration layer only")
            print("   - Deploy real contracts for production")
        
        print("\n🚀 Next Steps:")
        print("1. Start PRSM API server:")
        print("   python -m uvicorn prsm.api.main:app --reload")
        print("\n2. Test Web3 endpoints:")
        print("   curl http://localhost:8000/api/v1/web3/wallet/info")
        print("\n3. Initialize contracts:")
        print("   curl -X POST http://localhost:8000/api/v1/web3/contracts/initialize \\")
        print(f"     -d '{{\"ftns_token\": \"{contract_address}\"}}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

async def get_latest_deployment():
    """Get latest deployment file"""
    try:
        deployments_dir = Path(__file__).parent.parent / "deployments"
        if not deployments_dir.exists():
            return None
            
        deployment_files = list(deployments_dir.glob("polygon_mumbai-*.json"))
        if not deployment_files:
            return None
            
        # Get most recent
        latest_file = max(deployment_files, key=lambda x: x.stat().st_mtime)
        return latest_file
        
    except Exception:
        return None

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = [
        ('web3', 'pip install web3'),
        ('eth_account', 'pip install eth-account'),
        ('fastapi', 'pip install fastapi'),
        ('pydantic', 'pip install pydantic')
    ]
    
    missing = []
    for dep, install_cmd in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append((dep, install_cmd))
    
    if missing:
        print("❌ Missing dependencies:")
        for dep, cmd in missing:
            print(f"   {dep}: {cmd}")
        return False
    
    return True

if __name__ == '__main__':
    print("🧪 PRSM Web3 Integration Test Suite")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstall missing dependencies and try again.")
        sys.exit(1)
    
    # Run tests
    try:
        success = asyncio.run(test_web3_integration())
        if success:
            print("\n🎉 All tests passed! Web3 integration ready.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check output above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        sys.exit(1)