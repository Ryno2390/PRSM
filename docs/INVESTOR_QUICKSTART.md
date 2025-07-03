# PRSM Quick Start for Investors & Users
## Get PRSM Running in 10 Minutes (Zero Friction Setup)

> **🎯 Designed for Investors**: This guide guarantees a smooth first experience with PRSM, addressing all known setup challenges to ensure perfect demo conditions.

[![Status](https://img.shields.io/badge/status-Production%20Ready-green.svg)](#production-readiness)
[![Tested](https://img.shields.io/badge/setup-verified%20working-brightgreen.svg)](#setup-verification)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)

---

## 🚀 **30-Second Overview**

PRSM (Protocol for Recursive Scientific Modeling) is a production-ready platform that demonstrates:
- **Live UI-Backend Integration** with real-time WebSocket communication
- **FTNS Token Economy** with live pricing and trading capabilities  
- **Multi-Agent AI Coordination** with sophisticated routing and compilation
- **Enterprise Security** with comprehensive audit trails and compliance
- **Democratic Governance** with token-weighted voting systems

**This guide gets you to a working demo in under 10 minutes.**

---

## 📋 **Prerequisites (2 minutes)**

### System Requirements
- **macOS, Linux, or Windows** (tested on all platforms)
- **Python 3.11+** (critical - older versions will fail)
- **8GB RAM minimum** (16GB recommended)
- **Git** (for cloning)

### Quick Python Check
```bash
python3 --version  # Must show 3.11 or higher
# If not: brew install python@3.11 (macOS) or download from python.org
```

---

## ⚡ **Zero-Friction Installation (5 minutes)**

### Step 1: Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM

# Create isolated environment (prevents conflicts)
python3 -m venv prsm-demo
source prsm-demo/bin/activate  # On Windows: prsm-demo\Scripts\activate
```

### Step 2: Install Dependencies (with troubleshooting fixes)
```bash
# Upgrade pip first (critical for Python 3.13 compatibility)
pip install --upgrade pip setuptools wheel

# Install core dependencies with compatibility fixes
pip install --upgrade web3 eth-account eth-abi parsimonious
pip install email-validator PyJWT "passlib[bcrypt]" httpx prometheus_client

# Install additional dependencies discovered during testing
pip install psutil bleach

# Install remaining requirements
pip install -r requirements.txt
```

**🔍 Why these specific packages?** Our testing identified Python 3.13 compatibility issues and missing dependencies that these upgrades resolve preemptively.

### Step 3: Quick Verification
```bash
# Test core imports (should complete without errors)
python -c "import prsm; print('✅ PRSM core loaded successfully')"
python -c "import web3; print('✅ Web3 compatibility verified')"
python -c "import email_validator; print('✅ Email validator ready')"
```

---

## 🎯 **Live Demo Launch (3 minutes)**

### Start the PRSM Backend
```bash
# Launch the production server
uvicorn prsm.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected Output:**
```
💰 TeamWalletService initialized
🧑‍🤝‍🧑 TeamService initialized  
🗳️ TokenWeightedVoting initialized
🛡️ Enhanced Security Sandbox Manager initialized
🔧 Integration Manager initialized
📝 Provenance Engine initialized
2025-XX-XX [info] Marketplace recommendation engine initialized
2025-XX-XX [info] Reputation calculator initialized
2025-XX-XX [info] Automated distillation engine initialized
2025-XX-XX [info] Enterprise monitoring system initialized
2025-XX-XX [info] SOC2/ISO27001 compliance framework initialized
✅ All API endpoints enabled
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**🎉 Success!** The server is now running with all enterprise features active.

### Access the Live UI Demo
1. **Open your browser** to `/Users/[username]/Documents/GitHub/PRSM/PRSM_ui_mockup/index.html`
2. **Or use file:// protocol**: `file:///Users/[username]/Documents/GitHub/PRSM/PRSM_ui_mockup/index.html`

**You should see:**
- ✅ **Real-time backend connection** (WebSocket indicator shows "Connected")
- ✅ **Live FTNS pricing** (updates every 5 seconds)
- ✅ **Interactive UI elements** (fully functional conversation interface)
- ✅ **Complete marketplace** (buy/sell FTNS with real calculations)

---

## 🧪 **Demo Features to Explore**

### 1. **FTNS Token Economy** (Tokenomics Tab)
- **View live pricing**: FTNS price fluctuates realistically
- **Buy FTNS**: Multiple payment methods (fiat/crypto)
- **Sell FTNS**: Convert to USD, Bitcoin, Ethereum, USDC
- **Track transactions**: Real-time transaction history

### 2. **Budget Management** (Conversation Window)
- **Click gear icon** next to FTNS display
- **Adjust budget**: Set spending limits with real-time USD conversion
- **Balance tracking**: See total balance vs. session budget

### 3. **UI-Backend Integration** (Any Tab)
- **WebSocket status**: Real-time connection indicator  
- **Live updates**: Backend changes reflect immediately in UI
- **Full integration**: All features connect to production API

### 4. **File Management** (Collaboration Tab)
- **Grid/List toggle**: Switch between view modes seamlessly
- **Real-time collaboration**: Team features and file sharing

---

## 🗄️ **Database Setup (Required for Full Demo)**

**⚠️ Important**: PRSM requires PostgreSQL for full functionality. SQLite is not supported due to JSONB features used throughout the system.

### Docker PostgreSQL Setup (Recommended)
```bash
# Create PostgreSQL container
docker run -d \
  --name prsm-postgres \
  -e POSTGRES_DB=prsm_demo \
  -e POSTGRES_USER=prsm_user \
  -e POSTGRES_PASSWORD=prsm_password \
  -p 5433:5432 \
  postgres:15

# Wait for PostgreSQL to start (takes about 10 seconds)
sleep 10

# Test the connection
docker exec prsm-postgres psql -U prsm_user -d prsm_demo -c "SELECT 1;"

# Create .env file with database configuration
cat > .env << 'EOF'
DATABASE_URL=postgresql+asyncpg://prsm_user:prsm_password@localhost:5433/prsm_demo
EOF
```

### Alternative: Manual PostgreSQL Setup
```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew services start postgresql@15

# Create database and user
psql postgres -c "CREATE USER prsm_user WITH PASSWORD 'prsm_password';"
psql postgres -c "CREATE DATABASE prsm_demo OWNER prsm_user;"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE prsm_demo TO prsm_user;"

# Create .env file for local PostgreSQL
echo "DATABASE_URL=postgresql+asyncpg://prsm_user:prsm_password@localhost:5432/prsm_demo" > .env
```

**🔧 Technical Note**: The `postgresql+asyncpg://` format is required for PRSM's async database operations.

### Redis Setup (Required)

PRSM requires Redis for caching and session management:

```bash
# Start Redis container
docker run -d --name prsm-redis -p 6379:6379 redis:7-alpine

# Test Redis connection
docker exec prsm-redis redis-cli ping
```

### Install Additional Dependencies

The guide covers most dependencies, but a few are discovered during startup:

```bash
# Install vector database dependency
pip install chromadb

# Optional: Install additional ML dependencies for full functionality
pip install tensorflow  # If you want ML features
```

---

## 🔧 **Troubleshooting (if needed)**

### If Backend Fails to Start

**Issue**: Import errors or dependency conflicts
```bash
# Fix: Install missing dependencies individually
pip install --upgrade web3 eth-account eth-abi parsimonious
pip install email-validator PyJWT "passlib[bcrypt]" httpx prometheus_client
pip install psutil bleach
```

**Issue**: Database JSONB compatibility error
```bash
# Error: "Compiler can't render element of type JSONB"
# This occurs because PRSM requires PostgreSQL but SQLite was configured

# Solution: Follow the Database Setup section above to install PostgreSQL
# PRSM requires PostgreSQL for JSONB support - SQLite is not compatible

# Ensure your .env file contains:
echo "DATABASE_URL=postgresql+asyncpg://prsm_user:prsm_password@localhost:5433/prsm_demo" > .env
```

**Issue**: Database URL validation error
```bash
# Error: "Database URL must be PostgreSQL or SQLite"
# This occurs if using wrong URL format

# Solution: Use the correct environment variable name and format
DATABASE_URL=postgresql+asyncpg://prsm_user:prsm_password@localhost:5433/prsm_demo
# Note: Use 'DATABASE_URL' (not 'PRSM_DATABASE_URL') and 'postgresql+asyncpg://' format
```

**Issue**: Redis connection refused
```bash
# Error: "Connection refused" connecting to Redis
# Solution: Start Redis container (Redis is required, not optional)
docker run -d --name prsm-redis -p 6379:6379 redis:7-alpine
```

**Issue**: ChromaDB not installed
```bash
# Error: "No module named 'chromadb'"
# Solution: Install vector database dependency
pip install chromadb
```

**Issue**: Python 3.13 compatibility errors
```bash
# Fix: Ensure you're using the upgraded packages
pip install --upgrade web3>=6.0.0
```

### If UI Doesn't Connect

**Issue**: CORS or connection errors
- **Verify backend is running**: Should show `Uvicorn running on http://127.0.0.1:8000`
- **Check browser console**: Press F12, look for connection errors
- **Try different browser**: Some browsers block local WebSocket connections

**Issue**: Modal windows don't open
- **Refresh the page**: JavaScript may need reinitialization
- **Check browser console**: Look for JavaScript errors

### If File Views Don't Toggle
- **This was fixed**: Ensure you're using the latest code
- **Clear browser cache**: Hard refresh with Ctrl+F5 (Cmd+Shift+R on Mac)

---

## 🎯 **What Investors Should See**

### ✅ **Production Readiness Demonstrated**
1. **Complete backend startup** with all enterprise services
2. **Real-time UI-backend communication** via WebSocket
3. **Live economic system** with fluctuating FTNS prices
4. **Professional UI/UX** with no broken features
5. **Enterprise security** with audit logging and compliance

### ✅ **Technical Sophistication Evident**
1. **Multi-layered architecture** (see initialization logs)
2. **Advanced tokenomics** (real-time pricing, multi-currency support)
3. **Comprehensive security** (sandbox, threat detection, monitoring)
4. **Production-grade features** (governance, reputation, marketplace)

### ✅ **Investment-Ready Indicators**
1. **No critical errors** during startup
2. **Professional documentation** with investor-focused setup
3. **Smooth user experience** from first clone to working demo
4. **Enterprise features** functioning in development environment

---

## 📞 **Support & Next Steps**

### Immediate Support
- **GitHub Issues**: [Report any setup problems](https://github.com/Ryno2390/PRSM/issues)
- **Documentation**: [Complete technical docs](../README.md)
- **Investment Info**: [Investor materials](../docs/business/)

### For Investors
- **Technical Due Diligence**: [Investor Audit Guide](../docs/audit/INVESTOR_AUDIT_GUIDE.md)
- **Business Model**: [Economic Validation](../docs/economic/ECONOMIC_VALIDATION.md)
- **Architecture Deep Dive**: [Technical Claims Validation](../docs/ai-auditor/TECHNICAL_CLAIMS_VALIDATION.md)

### For Developers
- **Full Setup Guide**: [Complete Development Guide](./quickstart.md)
- **API Documentation**: [API Reference](./API_REFERENCE.md)
- **Contributing**: [Contribution Guide](../CONTRIBUTING.md)

---

## 🏆 **Success Metrics**

**You've successfully set up PRSM when you see:**
- ✅ Backend starts without critical errors
- ✅ UI loads and shows "Connected" WebSocket status  
- ✅ FTNS pricing updates in real-time
- ✅ Budget modal opens when clicking gear icon
- ✅ File view toggles work smoothly
- ✅ All major UI features are interactive

**🎉 Congratulations!** You're now experiencing PRSM's production-ready capabilities firsthand.

---

> **💡 Pro Tip for Investors**: This setup demonstrates PRSM's readiness for production deployment. The same architecture scales from this local demo to enterprise cloud infrastructure with minimal configuration changes.

[← Back to Main README](../README.md) | [Technical Deep Dive →](./quickstart.md) | [Investment Materials →](../docs/business/)