# PRSM FTNS Token-Based Pricing System
**Advanced Tokenomics with Computational Complexity and Market-Responsive Pricing**

## 🎯 Executive Summary

The PRSM Fungible Tokens for Node Support (FTNS) implements a sophisticated token-based pricing system that directly correlates costs with computational complexity, similar to LLM token pricing models. This system creates fair, transparent, and market-responsive pricing for NWTN deep reasoning queries while maintaining sustainable economic incentives.

**Key Innovation**: FTNS pricing maps directly to computational units (tokens) with floating market rates, reasoning complexity multipliers, and quality-based bonuses - creating a self-regulating economy where complex multi-modal queries naturally cost more while market forces determine fair pricing.

## 🪙 FTNS Token Fundamentals

### Token Specifications
```
Token Name: PRSM Fungible Tokens for Node Support
Symbol: FTNS
Decimals: 18
Total Supply: 1,000,000,000 FTNS (1 billion)
Initial Supply: 100,000,000 FTNS (100 million)
Network: Polygon (MATIC)
Standard: ERC-20 with governance extensions
```

### Multi-Tier Balance System

The FTNS token implements a sophisticated three-tier balance system:

#### 1. **Liquid Balance** - Immediately Transferable
- **Purpose**: Day-to-day transactions and trading
- **Characteristics**: No restrictions, full transferability
- **Use Cases**: 
  - Marketplace purchases
  - Peer-to-peer transfers
  - Exchange trading
  - Service payments

#### 2. **Locked Balance** - Time-Based Restrictions
- **Purpose**: Rewards with vesting periods
- **Characteristics**: Automatically unlocks based on time schedule
- **Use Cases**:
  - Research publication rewards (6-month vesting)
  - Model contribution bonuses (3-month vesting)
  - Data sharing incentives (1-month vesting)
  - Development team allocations (48-month vesting)

#### 3. **Staked Balance** - Governance Participation
- **Purpose**: Democratic governance and consensus participation
- **Characteristics**: Locked during staking period, earns voting power
- **Benefits**:
  - Governance voting rights (1 FTNS = 1 vote)
  - Additional reward multipliers (5-25% APY)
  - Priority access to premium features
  - Reduced marketplace fees (up to 50% discount)

### Token Flow Example

```
User Research Contribution (1,000 FTNS reward)
├── 400 FTNS → Liquid Balance (immediate)
├── 400 FTNS → Locked Balance (3-month vesting)
└── 200 FTNS → Staked Balance (optional governance participation)

After 3 months:
├── 800 FTNS → Liquid Balance
└── 200 FTNS → Staked Balance (if chosen)
```

## 🧮 Token-Based Pricing Framework

### Computational Token Mapping

FTNS pricing directly correlates with computational complexity using token-equivalent units, similar to OpenAI/Anthropic pricing models:

```
FTNS_Cost = Base_Computational_Tokens × Reasoning_Multiplier × Market_Rate × Quality_Bonus × Verbosity_Factor

Where:
Base_Computational_Tokens = Estimated processing units for query execution
Reasoning_Multiplier = Complexity scaling based on reasoning depth
Market_Rate = Floating rate based on network supply/demand
Quality_Bonus = Performance-based multiplier (0.8x - 1.5x)
Verbosity_Factor = Output complexity scaling (0.5x - 3.0x)
```

### Reasoning Complexity Multipliers

| ThinkingMode | Base Multiplier | Description | Typical Use Cases |
|--------------|----------------|-------------|-------------------|
| **QUICK** | 1.0x | Fast inference, minimal reasoning | Simple queries, fact lookup |
| **INTERMEDIATE** | 2.5x | Multi-step reasoning, moderate analysis | Research synthesis, comparison |
| **DEEP** | 5.0x | Complex reasoning, multi-modal analysis | Novel research, deep analysis |

### Verbosity Scaling Factors

| Verbosity Level | Token Target | Cost Multiplier | Output Characteristics |
|----------------|--------------|-----------------|----------------------|
| **BRIEF** | 100-300 tokens | 0.5x | Concise, key points only |
| **STANDARD** | 300-800 tokens | 1.0x | Balanced detail level |
| **DETAILED** | 800-1,500 tokens | 1.5x | Comprehensive explanations |
| **COMPREHENSIVE** | 1,500-3,000 tokens | 2.0x | Full analysis with examples |
| **ACADEMIC** | 3,000+ tokens | 3.0x | Research-grade thoroughness |

### Market-Responsive Rate System

#### Floating Market Rates

**Base Rate Calculation**:
```
Market_Rate = Base_Rate × (1 + Supply_Demand_Factor + Network_Congestion_Factor)

Supply_Demand_Factor = (Active_Queries - Available_Capacity) / Available_Capacity
Network_Congestion_Factor = Current_Load / Max_Capacity
```

**Rate Ranges**:
- **Low Demand**: 0.8x - 1.0x base rate
- **Normal Demand**: 1.0x - 1.2x base rate  
- **High Demand**: 1.2x - 2.0x base rate
- **Peak Congestion**: 2.0x - 3.0x base rate

#### Quality Performance Bonuses

| Performance Tier | Multiplier | Criteria |
|-----------------|------------|----------|
| **Standard** | 1.0x | Baseline performance |
| **High Quality** | 1.15x | >90% accuracy, <5s response time |
| **Premium** | 1.3x | >95% accuracy, <3s response time |
| **Excellence** | 1.5x | >98% accuracy, <2s response time |

### Example Pricing Calculations

#### Simple Query Example
```
Query: "What is quantum computing?"
- Base_Tokens: 50 (simple factual query)
- ThinkingMode: QUICK (1.0x)
- Verbosity: STANDARD (1.0x)
- Market_Rate: 1.1x (normal demand)
- Quality_Bonus: 1.0x (standard)

FTNS_Cost = 50 × 1.0 × 1.1 × 1.0 × 1.0 = 55 FTNS
```

#### Complex Deep Reasoning Example
```
Query: "Analyze the implications of quantum error correction on 
       near-term quantum advantage in cryptography"
- Base_Tokens: 200 (complex analysis required)
- ThinkingMode: DEEP (5.0x)
- Verbosity: COMPREHENSIVE (2.0x)  
- Market_Rate: 1.4x (high demand)
- Quality_Bonus: 1.15x (high quality service)

FTNS_Cost = 200 × 5.0 × 2.0 × 1.4 × 1.15 = 3,220 FTNS
```

### Revenue Distribution Model

#### Primary Revenue Streams (Updated)

1. **NWTN Query Processing** (60% of revenue)
   - Token-based pricing for all reasoning queries
   - Dynamic pricing based on complexity and demand
   - Premium quality tiers with enhanced performance

2. **Context Grounding Services** (25% of revenue)  
   - PDF processing and embedding generation
   - Paper retrieval and synthesis
   - External knowledge integration

3. **Marketplace Transactions** (10% of revenue)
   - Model and data trading
   - Computational resource rental
   - Research collaboration tools

4. **Governance & Staking** (5% of revenue)
   - Governance participation fees
   - Validator operations
   - Staking rewards distribution

### Economic Sustainability Mechanisms

#### Token Burn Mechanisms (Deflationary Pressure)
```
Context Allocation Burns:
├── AI Query Processing: 0.5-5 FTNS burned per query
├── Model Training: 10-100 FTNS burned per training session
└── Data Processing: 1-20 FTNS burned per dataset

Annual Burn Estimate: 2-5% of circulating supply
```

#### Reward Distribution (Inflationary Balance)
```
Research Rewards Pool: 40% of inflation
├── Data Contribution: 0.05 FTNS per MB contributed
├── Model Publication: 100-10,000 FTNS per model
├── Paper Publication: 500-5,000 FTNS per citation
└── Teaching Success: 20 FTNS + improvement multiplier

Maximum Annual Inflation: 8% decreasing by 0.5% yearly
```

## 🏪 Marketplace Economics

### Resource Pricing Model

The PRSM marketplace operates with dynamic, market-driven pricing:

#### Resource Categories & Typical Pricing
| Resource Type | Unit | Typical Range (FTNS) | Quality Premium |
|---------------|------|---------------------|-----------------|
| **AI Inference** | Per query | 0.001 - 0.1 | 15-25% |
| **GPU Compute** | Per hour | 0.5 - 50 | 20-40% |
| **CPU Compute** | Per hour | 0.01 - 1 | 10-20% |
| **Storage SSD** | Per GB/month | 0.0001 - 0.01 | 5-15% |
| **Data Processing** | Per GB processed | 0.1 - 5 | 25-35% |
| **Network Bandwidth** | Per GB transferred | 0.001 - 0.1 | 10-20% |
| **AI Training** | Per model | 10 - 1000 | 30-50% |
| **Data Licensing** | Per dataset | 5 - 5000 | Variable |

#### Price Discovery Mechanism
1. **Initial Pricing**: Providers set base prices
2. **Quality Adjustment**: Reputation system applies multipliers
3. **Supply/Demand**: Market forces adjust pricing
4. **Platform Optimization**: Algorithm suggests competitive pricing

### Transaction Flow Example

```
User wants to rent GPU for AI training (24 hours @ 10 FTNS/hour)

Transaction Breakdown:
├── Base Cost: 240 FTNS (24 hours × 10 FTNS)
├── Platform Fee: 6 FTNS (2.5% of 240 FTNS)
├── Quality Premium: 36 FTNS (15% for high-quality provider)
└── Total Cost: 282 FTNS

Payment Distribution:
├── Provider Receives: 270 FTNS (240 + 36 - 6)
├── Platform Revenue: 6 FTNS
├── Quality Bonus Pool: 4 FTNS (redistributed to high-quality providers)
└── Burn Mechanism: 2 FTNS (0.7% deflation)
```

## 🗳️ Governance & Staking

### Democratic Governance System

#### Governance Mechanics
- **Voting Power**: 1 FTNS staked = 1 vote
- **Proposal Threshold**: 10,000 FTNS to submit proposals
- **Voting Period**: 7 days per proposal
- **Quorum Requirement**: 10% of staked supply must participate
- **Approval Threshold**: 60% approval needed

#### Governance Categories
1. **Technical Proposals**: Protocol upgrades, new features
2. **Economic Proposals**: Fee adjustments, reward changes
3. **Treasury Proposals**: Fund allocation, partnerships
4. **Emergency Proposals**: Security responses (24-hour voting)

#### Staking Rewards Structure
```
Base Staking APY: 5%
├── Governance Participation Bonus: +2% APY
├── Long-term Staking Bonus: +3% APY (12+ months)
├── Validator Operations Bonus: +5% APY
└── Maximum Staking APY: 15%

Reward Distribution:
├── 50% from transaction fees
├── 30% from new token emissions
└── 20% from penalty redistributions
```

### Staking Example

```
User stakes 10,000 FTNS for governance
├── Lock Period: 12 months minimum
├── Voting Power: 10,000 votes
├── Base Rewards: 500 FTNS/year (5% APY)
├── Participation Bonus: 200 FTNS/year (active voting)
├── Long-term Bonus: 300 FTNS/year (12+ month stake)
└── Total Annual Rewards: 1,000 FTNS (10% effective APY)
```

## 📊 Validated Economic Performance

### Simulation Results (7-Day Test)

Based on agent-based economic modeling with 500 participants:

```
Transaction Metrics:
├── Total Transactions: 2,740
├── Completion Rate: 86.1%
├── Daily Volume: 365,854 FTNS
└── Daily Revenue: 8,920 FTNS

Market Distribution:
├── AI Inference: 40.3% of transactions
├── CPU Compute: 38.7% of transactions
├── GPU Compute: 8.6% of transactions
└── Other Resources: 12.4% of transactions
```

### Revenue Scaling Projections

#### Conservative Growth (10x - 5,000 users)
- **Daily Revenue**: 89,203 FTNS (~$4,460 at $0.05/FTNS)
- **Monthly Revenue**: 2.7M FTNS (~$134K)
- **Annual Revenue**: 32.6M FTNS (~$1.6M)

#### Aggressive Growth (100x - 50,000 users)
- **Daily Revenue**: 892,030 FTNS (~$44,600)
- **Monthly Revenue**: 26.8M FTNS (~$1.34M)
- **Annual Revenue**: 325.6M FTNS (~$16.3M)

#### Enterprise Scale (1000x - 500,000 users)
- **Daily Revenue**: 8.9M FTNS (~$446K)
- **Monthly Revenue**: 267.6M FTNS (~$13.4M)
- **Annual Revenue**: 3.26B FTNS (~$163M)

## 🔗 Smart Contract Integration

### Contract Architecture

The FTNS ecosystem consists of four primary smart contracts:

#### 1. **FTNSToken.sol** - Core Token Contract
```solidity
// Key Functions
function liquidBalance(address account) → uint256
function lockedBalance(address account) → uint256  
function stakedBalance(address account) → uint256
function allocateContext(uint256 amount) → bool
function stakeForGovernance(uint256 amount, uint256 duration) → bool
```

#### 2. **FTNSMarketplace.sol** - Resource Trading
```solidity
// Key Functions
function createListing(string calldata ipfsHash, string calldata title, 
                     string calldata description, uint256 pricePerHour) → uint256
function rentModel(uint256 listingId, uint256 duration) → bool
function completeTransaction(uint256 transactionId) → bool
```

#### 3. **FTNSGovernance.sol** - Democratic Voting
```solidity
// Key Functions
function proposeWithDetails(address[] calldata targets, uint256[] calldata values,
                          bytes[] calldata calldatas, string calldata description) → uint256
function castVoteWithReason(uint256 proposalId, uint8 support, 
                          string calldata reason) → uint256
```

#### 4. **TimelockController.sol** - Secure Execution
```solidity
// Automated execution of approved governance proposals
// 48-hour delay for security
```

### Integration Example

```python
# Python integration with PRSM backend
from prsm.web3.ftns_service import FTNSService

# Initialize service
ftns = FTNSService(network="polygon_mumbai")

# Check multi-tier balances
balances = await ftns.get_account_info(user_address)
print(f"Liquid: {balances.liquid} FTNS")
print(f"Locked: {balances.locked} FTNS") 
print(f"Staked: {balances.staked} FTNS")

# Allocate context for AI processing
await ftns.allocate_context(
    user_id="user123",
    amount=100,  # 100 FTNS
    purpose="AI_model_training"
)

# Participate in governance
await ftns.stake_for_governance(
    user_id="user123",
    amount=5000,  # 5,000 FTNS
    duration_months=12
)
```

## 🛡️ Security & Compliance

### Access Control & Roles

The smart contracts implement role-based access control:

```solidity
// Contract Roles
DEFAULT_ADMIN_ROLE     // Contract administration
MINTER_ROLE           // Token minting for rewards
BURNER_ROLE           // Token burning for context allocation
PAUSER_ROLE           // Emergency pause functionality
GOVERNANCE_ROLE       // Governance operations
```

### Security Features

- ✅ **ReentrancyGuard**: Protection against reentrancy attacks
- ✅ **Pausable**: Emergency stop functionality
- ✅ **AccessControl**: Role-based permissions
- ✅ **TimelockController**: Delayed execution for governance
- ✅ **UUPS Upgrades**: Secure upgradeability pattern

### Compliance Framework

- ✅ **SOC2 Ready**: Security controls implementation
- ✅ **Polygon Network**: Ethereum-compatible, enterprise-ready
- ✅ **Transparent Operations**: All transactions publicly auditable
- ✅ **KYC Integration**: Optional identity verification for large transactions

## 🎯 Token Utility Summary

### Primary Use Cases

1. **Resource Payment**: Pay for computational resources and AI services
2. **Governance Participation**: Vote on protocol upgrades and decisions
3. **Reward Distribution**: Earn tokens for valuable contributions
4. **Context Allocation**: Burn tokens for AI model processing
5. **Staking Benefits**: Earn yield and reduced fees through staking
6. **Marketplace Transactions**: Buy/sell AI models and datasets

### Token Flow Cycle

```
Research Contribution → FTNS Rewards → Multi-tier Balances
                                    ↓
Liquid Balance → Marketplace Transactions → Platform Fees
     ↓                                            ↓
Staking → Governance Participation → Protocol Improvements
     ↓                                            ↓
Reward Multipliers → Increased Earnings → Token Burns (Context)
                                          ↓
                               Deflationary Pressure → Value Appreciation
```

## 📈 Investment & Growth Metrics

### Key Performance Indicators

**Token Metrics:**
- Circulating Supply Growth: Target 5-8% annually
- Burn Rate: Target 2-5% of transactions
- Staking Ratio: Target 30-50% of supply staked
- Price Stability: Target <20% monthly volatility

**Network Metrics:**
- Daily Active Users: Target 25% monthly growth
- Transaction Volume: Target 15% monthly growth
- Average Transaction Value: Monitor for ecosystem health
- Platform Revenue: Target 20% monthly growth

**Quality Metrics:**
- Transaction Success Rate: Target >90%
- User Retention: Target >80% monthly
- Provider Quality Score: Target >0.7/1.0 average
- Governance Participation: Target >15% eligible voters

### Risk Mitigation

**Economic Risks:**
- Market volatility mitigated by utility value
- Regulatory compliance through transparent operations
- Technology risk addressed by production-ready infrastructure

**Operational Risks:**
- Smart contract security through audits and testing
- Liquidity risk managed through market makers
- Scaling risk validated through economic simulation

## 🚀 Getting Started

### For Users
1. **Acquire FTNS**: Purchase on supported exchanges or earn through contributions
2. **Set Up Wallet**: Use MetaMask or similar Web3 wallet with Polygon network
3. **Join Marketplace**: Browse and purchase computational resources
4. **Participate in Governance**: Stake tokens and vote on proposals

### For Developers
1. **Contract Integration**: Use provided SDKs for Web3 integration
2. **API Access**: Leverage PRSM API for token operations
3. **Testing Environment**: Deploy and test on Polygon Mumbai testnet
4. **Production Deployment**: Launch with audited smart contracts

### For Investors
1. **Economic Validation**: Review simulation results and projections
2. **Technical Assessment**: Evaluate smart contract security and functionality
3. **Market Analysis**: Understand TAM/SAM/SOM and competitive position
4. **Growth Potential**: Assess network effects and scaling opportunities

---

## 📞 Support & Resources

**Technical Documentation:**
- [Smart Contract Code](../contracts/README.md)
- [Deployment Guide](../contracts/DEPLOYMENT_GUIDE.md)
- [API Reference](../API_REFERENCE.md)

**Economic Analysis:**
- [Economic Validation Report](../economic/ECONOMIC_VALIDATION.md)
- [Investment Materials](../business/INVESTOR_MATERIALS.md)

**Community:**
- GitHub Issues for technical support
- Discord for community discussion
- Documentation wiki for comprehensive guides

---

**Document Version:** 1.0.0  
**Last Updated:** July 2025  
**Next Review:** October 2025

*This document provides comprehensive clarity on FTNS tokenomics. For specific implementation details, refer to the smart contract documentation and API references.*