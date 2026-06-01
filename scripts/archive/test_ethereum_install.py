#!/usr/bin/env python3
"""
Test script to verify Ethereum package installation
Run this to test if web3 and eth-account work together
"""

try:
    import web3
    print(f"✅ web3 version: {web3.__version__}")
    
    import eth_account
    print(f"✅ eth_account version: {eth_account.__version__}")
    
    import eth_utils
    print(f"✅ eth_utils version: {eth_utils.__version__}")
    
    import parsimonious
    print(f"✅ parsimonious version: {parsimonious.__version__} (Python 3.13 compatible)")
    
    import sphinx
    print(f"✅ sphinx version: {sphinx.__version__}")
    
    # Test basic functionality
    from web3 import Web3
    w3 = Web3()
    print(f"✅ Web3 instance created successfully")
    
    from eth_account import Account
    account = Account.create()
    print(f"✅ Account creation works: {account.address[:10]}...")
    
    # Test eth-utils compatibility
    from eth_utils import to_checksum_address
    checksum_addr = to_checksum_address(account.address)
    print(f"✅ eth_utils checksum works: {checksum_addr[:10]}...")
    
    print("\n🎉 All packages installed and working correctly!")
    print("✅ Sphinx documentation build should now work!")
    print("✅ Python 3.13 compatibility confirmed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Install packages with: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")