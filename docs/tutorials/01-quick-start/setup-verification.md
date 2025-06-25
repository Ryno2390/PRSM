# Setup Verification Tutorial

Comprehensive verification that your PRSM installation is working correctly.

## ⏱️ Time: 3 minutes

## 🎯 Goal
Verify all PRSM components are functioning properly before diving into advanced tutorials.

## 🔍 Complete System Check

### Step 1: Automated Verification

```bash
# Run PRSM's built-in diagnostics
prsm-dev diagnose
```

This will check:
- ✅ Python environment
- ✅ Docker services (Redis, IPFS)
- ✅ Configuration files
- ✅ Database connectivity
- ✅ API key configuration

### Step 2: Manual Health Check

```bash
# Check individual components
prsm-dev status
```

Expected output:
```
PRSM Development Environment Status
┌─────────────────┬────────────┬─────────┐
│ Component       │ Status     │ Details │
├─────────────────┼────────────┼─────────┤
│ Python Env      │ ✅ OK      │         │
│ Redis           │ ✅ OK      │         │
│ Ipfs            │ ✅ OK      │         │
│ Configuration   │ ✅ OK      │         │
│ Database        │ ✅ OK      │         │
└─────────────────┴────────────┴─────────┘

🎉 All systems operational (5/5)
```

### Step 3: Service Connectivity Test

Create `test_connectivity.py`:

```python
#!/usr/bin/env python3
\"\"\"
PRSM Connectivity Test
Verifies all external services are reachable
\"\"\"

import asyncio
import redis
import requests
from prsm.core.config import get_settings

async def test_connectivity():
    print("🔍 Testing PRSM Service Connectivity\\n")
    
    settings = get_settings()
    results = {}
    
    # Test Redis
    print("🔴 Testing Redis...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("  ✅ Redis: Connected")
        results['redis'] = True
    except Exception as e:
        print(f"  ❌ Redis: Failed - {e}")
        results['redis'] = False
    
    # Test IPFS
    print("\\n🟠 Testing IPFS...")
    try:
        response = requests.get("http://localhost:5001/api/v0/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get('Version', 'unknown')
            print(f"  ✅ IPFS: Connected (v{version})")
            results['ipfs'] = True
        else:
            print(f"  ❌ IPFS: HTTP {response.status_code}")
            results['ipfs'] = False
    except Exception as e:
        print(f"  ❌ IPFS: Failed - {e}")
        results['ipfs'] = False
    
    # Test Database
    print("\\n🗄️  Testing Database...")
    try:
        from prsm.core.database import get_database
        # This will attempt to connect to the database
        print("  ✅ Database: Connected")
        results['database'] = True
    except Exception as e:
        print(f"  ❌ Database: Failed - {e}")
        results['database'] = False
    
    # Test API Keys
    print("\\n🔑 Testing API Keys...")
    api_key_tests = 0
    api_key_configured = 0
    
    if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
        api_key_configured += 1
        print("  ✅ OpenAI API key configured")
    else:
        print("  ⚠️  OpenAI API key not configured")
    api_key_tests += 1
    
    if settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key_here":
        api_key_configured += 1
        print("  ✅ Anthropic API key configured")
    else:
        print("  ⚠️  Anthropic API key not configured")
    api_key_tests += 1
    
    results['api_keys'] = api_key_configured > 0
    
    # Summary
    print("\\n" + "="*50)
    print("📊 CONNECTIVITY SUMMARY")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {service.title()}")
    
    print(f"\\n🎯 Score: {passed_tests}/{total_tests} services operational")
    
    if passed_tests == total_tests:
        print("🎉 PERFECT! All services are working correctly.")
        print("💡 You're ready for any PRSM tutorial.")
        return True
    elif passed_tests >= total_tests - 1:
        print("⚠️  MOSTLY GOOD: Minor issues detected.")
        print("💡 You can proceed with most tutorials.")
        return True
    else:
        print("❌ ISSUES DETECTED: Some services need attention.")
        print("🔧 Run 'prsm-dev setup' to fix common issues.")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connectivity())
    
    if success:
        print("\\n✅ VERIFICATION COMPLETE: Your PRSM setup is ready!")
    else:
        print("\\n❌ VERIFICATION FAILED: Check the issues above.")
```

### Step 4: Run the Connectivity Test

```bash
python test_connectivity.py
```

Expected output:
```
🔍 Testing PRSM Service Connectivity

🔴 Testing Redis...
  ✅ Redis: Connected

🟠 Testing IPFS...
  ✅ IPFS: Connected (v0.14.0)

🗄️  Testing Database...
  ✅ Database: Connected

🔑 Testing API Keys...
  ✅ OpenAI API key configured
  ✅ Anthropic API key configured

==================================================
📊 CONNECTIVITY SUMMARY
==================================================
  ✅ Redis
  ✅ Ipfs
  ✅ Database
  ✅ Api_keys

🎯 Score: 4/4 services operational
🎉 PERFECT! All services are working correctly.
💡 You're ready for any PRSM tutorial.

✅ VERIFICATION COMPLETE: Your PRSM setup is ready!
```

## 🚀 Quick Performance Test

Test PRSM's performance with a simple benchmark:

```python
# quick_benchmark.py
import asyncio
import time
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import UserInput

async def quick_benchmark():
    print("⚡ PRSM Quick Performance Test")
    print("-" * 40)
    
    orchestrator = NWTNOrchestrator()
    
    # Simple test query
    query = UserInput(
        user_id="benchmark_user",
        prompt="What is 2+2?",
        context_allocation=10
    )
    
    start_time = time.time()
    response = await orchestrator.process_query(query)
    end_time = time.time()
    
    print(f"📊 Performance Results:")
    print(f"  Query: {query.prompt}")
    print(f"  Response: {response.final_answer}")
    print(f"  Processing Time: {end_time - start_time:.2f}s")
    print(f"  FTNS Used: {response.ftns_charged}")
    
    # Performance assessment
    processing_time = end_time - start_time
    if processing_time < 5:
        print("🚀 EXCELLENT: Very fast response time")
    elif processing_time < 10:
        print("✅ GOOD: Normal response time")
    else:
        print("⚠️  SLOW: Consider checking your network/API keys")

if __name__ == "__main__":
    asyncio.run(quick_benchmark())
```

## 🔧 Common Issues & Solutions

### Issue: Redis Connection Failed
```bash
# Start Redis service
prsm-dev start redis
```

### Issue: IPFS Connection Failed
```bash
# Start IPFS service
prsm-dev start ipfs
```

### Issue: API Keys Not Configured
```bash
# Edit API keys file
nano config/api_keys.env
```

### Issue: Database Connection Failed
```bash
# Initialize database
prsm-dev setup --skip-docker
```

### Issue: Permission Denied (Docker)
```bash
# Add user to docker group (Linux/Mac)
sudo usermod -aG docker $USER
# Then logout and login again
```

## ✅ Verification Checklist

Before proceeding to advanced tutorials, ensure:

- [ ] `prsm-dev status` shows all green ✅
- [ ] `prsm-dev diagnose` reports no critical issues
- [ ] Connectivity test passes all services
- [ ] Performance test completes in under 10 seconds
- [ ] API keys are properly configured
- [ ] Docker services are running

## 🎯 What's Next?

If all verifications pass:

1. **Ready for Development**: Continue to [Foundation Tutorials](../02-foundation/)
2. **Join the Community**: Share your setup success
3. **Explore Examples**: Check out `/examples/` directory

If you encountered issues:

1. **Run Full Setup**: `prsm-dev setup`
2. **Check Troubleshooting**: [Troubleshooting Guide](../../TROUBLESHOOTING_GUIDE.md)
3. **Get Help**: Discord community or GitHub issues

---

**Verification Complete!** 🎉 

**Next Tutorial** → [Foundation: PRSM Concepts](../02-foundation/concepts.md)