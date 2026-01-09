# PRSM P2P Network & Tokenomics Demos

This directory contains two critical deliverables demonstrating PRSM's core capabilities:

## 🌐 Task 1: Minimal P2P Network Demo

**Objective**: Demonstrate PRSM's decentralized peer-to-peer architecture with node discovery, secure messaging, and consensus mechanisms.

### Files
- **`p2p_network_demo.py`** - Core P2P network simulation
- **`p2p_dashboard.py`** - Interactive Streamlit monitoring dashboard

### Features Demonstrated
- ✅ **Node Discovery & Registration**: Automatic peer discovery with handshaking
- ✅ **Secure Message Sharing**: Cryptographically signed message exchange
- ✅ **Consensus Simulation**: Byzantine fault tolerant consensus proposals
- ✅ **Failure Recovery**: Node failure simulation and auto-recovery
- ✅ **Real-time Monitoring**: Live network topology and metrics

### Quick Start

#### Run Basic P2P Demo
```bash
# Install dependencies
pip install asyncio hashlib uuid

# Run the demo
cd demos/
python p2p_network_demo.py
```

#### Launch Interactive Dashboard
```bash
# Install Streamlit and Plotly
pip install streamlit plotly pandas

# Launch dashboard
streamlit run p2p_dashboard.py
```

### Demo Output
The P2P demo simulates a 3-node network demonstrating:
1. **Peer Discovery**: Nodes automatically find and connect to each other
2. **Consensus Proposals**: Coordinator nodes propose timestamp synchronization
3. **File Sharing**: Secure data distribution with hash verification
4. **Network Resilience**: Node failure and recovery simulation

**Example Output:**
```
🚀 PRSM P2P Network Demo Starting...
📊 Initial Network Status: 3/3 nodes active, 6 connections
🤝 Demonstrating Consensus Mechanism...
📁 Demonstrating File Sharing...
⚠️ Simulating Node Failure and Recovery...
✅ P2P Network Demo Complete!
```

---

## 💰 Task 2: Tokenomics Simulation & Stress Test

**Objective**: Validate FTNS token economy sustainability under various market conditions with multi-agent economic simulation.

### Files
- **`tokenomics_simulation.py`** - Core economic simulation engine
- **`tokenomics_dashboard.py`** - Interactive analysis dashboard

### Features Demonstrated
- ✅ **Multi-Agent Economy**: 30-50 agents with 5 behavioral types
- ✅ **Quality-Based Rewards**: Contribution quality affects token distribution
- ✅ **Economic Stress Testing**: 4 market scenarios (normal, bull, bear, volatile)
- ✅ **Fairness Analysis**: Gini coefficient and wealth distribution metrics
- ✅ **Real-time Visualization**: Interactive charts and economic indicators

### Agent Types
1. **Data Contributors** (30%): Upload datasets, earn based on quality
2. **Model Creators** (20%): Develop AI models, higher rewards/costs
3. **Query Users** (30%): Consume services, pay transaction fees
4. **Validators** (15%): Validate content quality, earn consistent rewards
5. **Freeloaders** (5%): Bad actors attempting to game the system

### Quick Start

#### Run Stress Test Scenarios
```bash
# Install dependencies
pip install mesa numpy pandas matplotlib plotly seaborn

# Run comprehensive stress tests
cd demos/
python tokenomics_simulation.py
```

#### Launch Interactive Dashboard
```bash
# Launch Streamlit dashboard
streamlit run tokenomics_dashboard.py
```

### Stress Test Scenarios

#### 1. Normal Growth
- **Condition**: Stable market conditions
- **Duration**: 30 days
- **Validation**: Baseline performance metrics

#### 2. Market Volatility
- **Condition**: High price volatility spikes
- **Duration**: Days 5-10 and 20-25
- **Validation**: Price stability under stress

#### 3. Economic Shock
- **Condition**: Bear market + compute shortage
- **Duration**: Bear (days 3-15), Shortage (days 16-25)
- **Validation**: Network resilience to economic downturns

#### 4. Data Oversupply
- **Condition**: Market flooded with low-quality data
- **Duration**: Days 8-22
- **Validation**: Quality maintenance mechanisms

### Validation Criteria
✅ **Wealth Distribution Fair**: Gini coefficient ≤ 0.7  
✅ **Quality Maintained**: Average quality ≥ 60%  
✅ **Price Stable**: Price variance ≤ 20%  
✅ **High Participation**: Daily activity ≥ 50%  

### Expected Results
- **Price Stability**: 85%+ scenarios maintain stable pricing
- **Wealth Distribution**: Gini coefficient typically 0.3-0.6 (fair)
- **Quality Scores**: Average 70-85% across all scenarios
- **Network Participation**: 60-80% daily activity rates

---

## 🎯 Investment Evaluation Criteria

Both demos address key technical due diligence concerns:

### P2P Network Demo Validation
- ✅ **Observable Behavior**: Real-time network topology and message flow
- ✅ **Measurable Metrics**: Connection success rates, consensus participation
- ✅ **Clear Documentation**: Step-by-step execution with logged outputs
- ✅ **Failure Scenarios**: Node failure and recovery demonstration

### Tokenomics Validation
- ✅ **Economic Viability**: Multi-scenario stress testing
- ✅ **Fairness Metrics**: Gini coefficient and wealth distribution analysis
- ✅ **Attack Resistance**: Bad actor (freeloader) behavior modeling
- ✅ **Scalability Evidence**: Performance with 10-50 agents

### Technical Strengths Demonstrated
1. **Solo+AI Development**: Complex systems built with AI assistance
2. **Production Thinking**: Enterprise-grade logging, monitoring, dashboards
3. **Investor-Friendly**: Clear metrics, validation criteria, visual reporting
4. **Realistic Modeling**: Network delays, message loss, economic psychology

### Limitations Acknowledged
1. **Simulation vs Production**: Network uses local processes, not real P2P
2. **Simplified Economics**: Real tokenomics would have more complex mechanisms
3. **Scale Constraints**: Demo limited to 10-100 agents vs thousands in production
4. **Security Simulation**: Cryptography and consensus are simplified for demonstration

---

## 🚀 Running the Complete Demo Suite

### System Requirements
- Python 3.9+
- 8GB RAM (for larger simulations)
- Modern web browser (for Streamlit dashboards)

### Installation
```bash
# Clone PRSM repository
git clone https://github.com/PRSM-AI/PRSM.git
cd PRSM/demos/

# Install all dependencies
pip install -r requirements.txt

# Alternative: Install individually
pip install streamlit plotly pandas numpy matplotlib seaborn asyncio mesa
```

### Execution Sequence
```bash
# 1. Run P2P Network Demo
python p2p_network_demo.py

# 2. Run Tokenomics Stress Tests
python tokenomics_simulation.py

# 3. Launch Interactive Dashboards (in separate terminals)
streamlit run p2p_dashboard.py --server.port 8501
streamlit run tokenomics_dashboard.py --server.port 8502
```

### Demo Presentation Flow
1. **P2P Network** (10 minutes)
   - Start `p2p_network_demo.py` to show basic functionality
   - Launch `p2p_dashboard.py` for visual network topology
   - Demonstrate node discovery, messaging, consensus

2. **Tokenomics Analysis** (15 minutes)
   - Run `tokenomics_simulation.py` for stress test results
   - Launch `tokenomics_dashboard.py` for interactive analysis
   - Show economic fairness, stress test scenarios, agent behavior

3. **Q&A Discussion** (10 minutes)
   - Address technical architecture questions
   - Discuss production scaling plans
   - Review investment readiness metrics

---

## 📊 Success Metrics & Validation

### P2P Network Success Criteria
- ✅ All 3 nodes successfully discover peers
- ✅ >95% message delivery success rate
- ✅ Consensus proposals achieve majority approval
- ✅ Network recovers from node failures within 30 seconds

### Tokenomics Success Criteria  
- ✅ >75% of stress scenarios pass all validation criteria
- ✅ Gini coefficient remains ≤ 0.7 in normal conditions
- ✅ Average contribution quality ≥ 60% across scenarios
- ✅ Price stability ≥ 80% (variance ≤ 20%)

### Investment Readiness Indicators
- ✅ **Technical Feasibility**: Working demos with measurable performance
- ✅ **Economic Viability**: Stress-tested tokenomics with fairness validation  
- ✅ **Scalability Evidence**: Performance analysis across agent populations
- ✅ **Risk Assessment**: Failure scenarios and recovery mechanisms
- ✅ **Solo Capability**: Advanced systems built with AI assistance

---

## 🛠️ Technical Architecture

### P2P Network Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Coordinator   │    │     Worker      │    │   Validator     │
│      Node       │◄──►│      Node       │◄──►│      Node       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │    Consensus    │
                    │    Mechanism    │
                    └─────────────────┘
```

### Tokenomics Simulation Flow
```
Economic Agents → Market Actions → Token Distribution → Quality Validation
       ↓               ↓                 ↓                    ↓
   Behavioral      Supply/Demand     Reward Calculation   Reputation Update
   Parameters      Dynamics          Based on Quality     & Network Effects
       ↓               ↓                 ↓                    ↓
   Activity Level → Price Discovery → FTNS Balance → Performance Metrics
```

---

## 📝 Next Steps for Production

### P2P Network Evolution
1. **Real Network Protocol**: Replace simulation with libp2p implementation
2. **Enhanced Security**: Full cryptographic signatures and message encryption
3. **Byzantine Tolerance**: Production-grade consensus (PBFT/Tendermint)
4. **Geographic Distribution**: Multi-region node deployment
5. **Performance Optimization**: Connection pooling, message batching

### Tokenomics Enhancement
1. **On-Chain Implementation**: Deploy FTNS contracts to Polygon/Ethereum
2. **Advanced Economics**: Liquidity pools, yield farming, governance tokens
3. **Real-World Integration**: Fiat gateways, exchange listings
4. **Regulatory Compliance**: KYC/AML integration, legal framework
5. **Scale Testing**: 10,000+ agent simulations, load testing

### Investment Timeline
- **Phase 1** (Completed): Solo demonstration and validation
- **Phase 2** (Next 60 days): Team expansion, security audit, production deployment
- **Phase 3** (90-180 days): Alpha user onboarding, token launch, network scaling

This demonstration validates PRSM's core technical feasibility and economic viability, providing investors with concrete evidence of the platform's potential for successful deployment at scale.