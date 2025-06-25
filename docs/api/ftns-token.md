# FTNS Token API

Manage FTNS (Fractal Time Network Synchronization) tokens, economic incentives, and network participation rewards.

## 🎯 Overview

The FTNS Token API provides comprehensive token management functionality including wallet operations, staking, rewards distribution, governance participation, and economic coordination within the PRSM network.

## 📋 Base URL

```
https://api.prsm.ai/v1/tokens
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/tokens
```

## 🚀 Quick Start

### Check Token Balance

```python
import prsm

client = prsm.Client(api_key="your-api-key")

# Check FTNS token balance
balance = client.tokens.get_balance()
print(f"FTNS Balance: {balance.ftns_tokens}")
print(f"Staked Tokens: {balance.staked_tokens}")
print(f"Pending Rewards: {balance.pending_rewards}")
```

## 📊 Endpoints

### GET /tokens/balance
Get current token balance and staking information.

**Response:**
```json
{
  "wallet_address": "0x742d35Cc6084C7CBF8D7BB0eB8A9A5Bf9aB9F7E2",
  "ftns_tokens": 15000.75,
  "staked_tokens": 5000.00,
  "locked_tokens": 1000.00,
  "pending_rewards": 125.50,
  "total_earned": 2450.75,
  "staking_apy": 0.12,
  "last_reward_claim": "2024-01-15T10:30:00Z",
  "next_reward_distribution": "2024-01-16T00:00:00Z"
}
```

### POST /tokens/stake
Stake FTNS tokens to participate in network consensus and earn rewards.

**Request Body:**
```json
{
  "amount": 1000.0,
  "staking_period": "flexible",
  "validator_preference": "auto_select",
  "reinvest_rewards": true
}
```

**Response:**
```json
{
  "transaction_id": "tx_stake_abc123",
  "staked_amount": 1000.0,
  "staking_period": "flexible",
  "estimated_apy": 0.12,
  "estimated_rewards_per_day": 0.33,
  "validator_assigned": "validator_xyz789",
  "staking_start_date": "2024-01-15T11:00:00Z",
  "minimum_stake_duration": 0,
  "early_unstake_penalty": 0.0
}
```

### POST /tokens/unstake
Unstake FTNS tokens and claim rewards.

**Request Body:**
```json
{
  "amount": 500.0,
  "claim_rewards": true,
  "force_immediate": false
}
```

**Response:**
```json
{
  "transaction_id": "tx_unstake_def456",
  "unstaked_amount": 500.0,
  "rewards_claimed": 45.75,
  "penalty_applied": 0.0,
  "available_immediately": 500.0,
  "processing_time_hours": 0,
  "new_staked_balance": 4500.0
}
```

### POST /tokens/claim-rewards
Claim accumulated staking rewards.

**Response:**
```json
{
  "transaction_id": "tx_claim_ghi789",
  "rewards_claimed": 125.50,
  "total_rewards_to_date": 2576.25,
  "next_reward_available": "2024-01-16T00:00:00Z",
  "claiming_fee": 0.0,
  "net_rewards_received": 125.50
}
```

### GET /tokens/transactions
Get transaction history for token operations.

**Query Parameters:**
- `type`: Filter by transaction type (stake, unstake, reward, transfer)
- `from_date`: Start date for transaction history
- `to_date`: End date for transaction history
- `limit`: Maximum number of transactions to return

**Response:**
```json
{
  "transactions": [
    {
      "id": "tx_abc123",
      "type": "stake",
      "amount": 1000.0,
      "timestamp": "2024-01-15T11:00:00Z",
      "status": "confirmed",
      "gas_fee": 0.05,
      "block_number": 12345678,
      "transaction_hash": "0x1234567890abcdef..."
    },
    {
      "id": "tx_def456",
      "type": "reward_claim",
      "amount": 45.75,
      "timestamp": "2024-01-14T00:00:00Z",
      "status": "confirmed",
      "gas_fee": 0.02,
      "block_number": 12345123
    }
  ],
  "total_transactions": 47,
  "page": 1,
  "pages": 5
}
```

## 💰 Economic Participation

### Network Contribution Rewards

```python
# Earn tokens by contributing to network
contribution = client.tokens.contribute(
    contribution_type="inference_compute",
    resources_offered={
        "cpu_cores": 8,
        "ram_gb": 32,
        "gpu_memory_gb": 12
    },
    availability_hours=24,
    pricing={
        "cpu_per_hour": 0.10,
        "ram_per_gb_hour": 0.05,
        "gpu_per_hour": 0.50
    }
)
```

### Usage-Based Rewards

```python
# Track rewards for network usage
usage_rewards = client.tokens.get_usage_rewards(
    timeframe="last_30_days",
    breakdown=True
)

print(f"Inference rewards: {usage_rewards.inference_rewards}")
print(f"Data sharing rewards: {usage_rewards.data_sharing_rewards}")
print(f"Network maintenance rewards: {usage_rewards.maintenance_rewards}")
```

### Referral Program

```python
# Participate in referral program
referral = client.tokens.create_referral_code(
    reward_percentage=0.05,
    max_uses=100,
    expiry_date="2024-12-31T23:59:59Z"
)

# Check referral earnings
referral_earnings = client.tokens.get_referral_earnings()
```

## 🗳️ Governance Participation

### Voting on Proposals

```python
# Vote on governance proposals
vote = client.tokens.vote_on_proposal(
    proposal_id="prop_123",
    vote="yes",
    voting_power=1000.0,  # Based on staked tokens
    justification="This proposal improves network efficiency"
)
```

### Create Governance Proposal

```python
# Create new governance proposal (requires minimum stake)
proposal = client.tokens.create_proposal(
    title="Increase Staking Rewards",
    description="Proposal to increase staking APY from 12% to 15%",
    proposal_type="parameter_change",
    parameters={
        "staking_apy": 0.15,
        "effective_date": "2024-02-01T00:00:00Z"
    },
    voting_period_days=7,
    minimum_participation=0.3
)
```

### Delegation

```python
# Delegate voting power to another address
delegation = client.tokens.delegate_voting_power(
    delegate_address="0x742d35Cc6084C7CBF8D7BB0eB8A9A5Bf9aB9F7E2",
    amount=500.0,
    delegation_period_days=30
)
```

## 💱 Token Utilities

### Payment for Services

```python
# Pay for PRSM services with FTNS tokens
payment = client.tokens.pay_for_service(
    service_type="model_inference",
    service_details={
        "model": "gpt-4",
        "token_count": 1000
    },
    payment_amount=2.50,
    discount_applied=0.15  # Discount for paying with FTNS
)
```

### Resource Market

```python
# Purchase computational resources with tokens
resource_purchase = client.tokens.purchase_resources(
    resource_type="gpu_compute",
    duration_hours=4,
    specifications={
        "gpu_type": "A100",
        "gpu_count": 2,
        "ram_gb": 64
    },
    max_price_per_hour=1.50
)
```

### Premium Features

```python
# Access premium features with token payment
premium_access = client.tokens.activate_premium(
    feature="advanced_analytics",
    duration_days=30,
    auto_renew=True
)
```

## 📊 Token Economics Analytics

### Personal Economics Dashboard

```python
# Get comprehensive token economics overview
economics = client.tokens.economics_dashboard(
    timeframe="last_90_days"
)

print(f"Total tokens earned: {economics.total_earned}")
print(f"Total tokens spent: {economics.total_spent}")
print(f"Net token flow: {economics.net_flow}")
print(f"ROI percentage: {economics.roi_percentage}")
```

### Network Economics Metrics

```python
# Get network-wide token metrics
network_metrics = client.tokens.network_metrics()

print(f"Total tokens in circulation: {network_metrics.total_supply}")
print(f"Total staked percentage: {network_metrics.staked_percentage}")
print(f"Average staking APY: {network_metrics.avg_staking_apy}")
print(f"Transaction volume 24h: {network_metrics.volume_24h}")
```

### Yield Optimization

```python
# Get recommendations for yield optimization
optimization = client.tokens.yield_optimization(
    risk_tolerance="moderate",
    time_horizon_days=180,
    current_allocation={
        "staking": 0.6,
        "liquidity_provision": 0.2,
        "governance_participation": 0.2
    }
)
```

## 🔄 Token Transfers and Trading

### Transfer Tokens

```python
# Transfer FTNS tokens to another address
transfer = client.tokens.transfer(
    recipient_address="0x742d35Cc6084C7CBF8D7BB0eB8A9A5Bf9aB9F7E2",
    amount=100.0,
    memo="Payment for services",
    gas_price="standard"
)
```

### Exchange Integration

```python
# Get current exchange rates and trading options
exchange_info = client.tokens.exchange_info()

print(f"FTNS/USD rate: ${exchange_info.usd_rate}")
print(f"FTNS/ETH rate: {exchange_info.eth_rate} ETH")
print(f"24h trading volume: {exchange_info.volume_24h}")
```

### Liquidity Provision

```python
# Provide liquidity to token pools
liquidity = client.tokens.provide_liquidity(
    pool="FTNS/USDC",
    ftns_amount=1000.0,
    usdc_amount=2000.0,
    slippage_tolerance=0.005
)
```

## 🔐 Security and Compliance

### Multi-Signature Wallets

```python
# Create multi-signature wallet for enhanced security
multisig = client.tokens.create_multisig_wallet(
    owners=["0xAddress1", "0xAddress2", "0xAddress3"],
    required_signatures=2,
    daily_limit=1000.0
)
```

### Audit Trail

```python
# Get comprehensive audit trail
audit_trail = client.tokens.audit_trail(
    from_date="2024-01-01T00:00:00Z",
    include_transaction_details=True,
    include_smart_contract_interactions=True
)
```

### Compliance Reporting

```python
# Generate compliance reports for tax purposes
tax_report = client.tokens.generate_tax_report(
    tax_year=2024,
    jurisdiction="US",
    include_staking_rewards=True,
    include_governance_rewards=True
)
```

## 🎯 Advanced Token Features

### Token Vesting

```python
# Manage token vesting schedules
vesting = client.tokens.get_vesting_schedule()

print(f"Total vested: {vesting.total_vested}")
print(f"Total unvested: {vesting.total_unvested}")
print(f"Next vesting date: {vesting.next_vesting_date}")
print(f"Next vesting amount: {vesting.next_vesting_amount}")
```

### Burn Mechanisms

```python
# Participate in token burn events
burn_participation = client.tokens.participate_in_burn(
    amount=50.0,
    burn_reason="network_optimization",
    receive_credits=True
)
```

### Token Locking

```python
# Lock tokens for enhanced rewards
token_lock = client.tokens.lock_tokens(
    amount=2000.0,
    lock_period_days=90,
    enhanced_apy=0.18,
    early_unlock_penalty=0.05
)
```

## 📈 DeFi Integration

### Lending and Borrowing

```python
# Lend FTNS tokens for yield
lending = client.tokens.lend_tokens(
    amount=1500.0,
    term_days=30,
    interest_rate=0.08,
    collateral_required=False
)

# Borrow against FTNS collateral
borrowing = client.tokens.borrow_against_ftns(
    collateral_amount=2000.0,
    borrow_currency="USDC",
    borrow_amount=1000.0,
    loan_term_days=60
)
```

### Yield Farming

```python
# Participate in yield farming
yield_farming = client.tokens.join_yield_farm(
    farm_id="farm_ftns_usdc",
    lp_token_amount=500.0,
    auto_compound=True
)
```

## 🛠️ Developer Tools

### Smart Contract Interaction

```python
# Interact with FTNS smart contracts directly
contract_call = client.tokens.call_smart_contract(
    contract_address="0x...",
    function_name="transfer",
    parameters={
        "to": "0x742d35Cc6084C7CBF8D7BB0eB8A9A5Bf9aB9F7E2",
        "amount": 100.0
    },
    gas_limit=21000
)
```

### Custom Token Operations

```python
# Implement custom token logic
custom_operation = client.tokens.custom_operation(
    operation_type="batch_transfer",
    recipients=[
        {"address": "0xAddr1", "amount": 50.0},
        {"address": "0xAddr2", "amount": 75.0},
        {"address": "0xAddr3", "amount": 25.0}
    ],
    execution_strategy="atomic"
)
```

## 🧪 Testing and Simulation

### Testnet Operations

```python
# Use testnet for development and testing
testnet_client = prsm.Client(
    api_key="test-api-key",
    network="testnet"
)

# Get test tokens
test_tokens = testnet_client.tokens.get_test_tokens(
    amount=10000.0
)
```

### Economic Simulation

```python
# Simulate economic scenarios
simulation = client.tokens.simulate_economics(
    scenario="increased_network_usage",
    parameters={
        "usage_multiplier": 2.0,
        "new_user_growth": 0.5,
        "staking_participation_change": 0.1
    },
    simulation_days=365
)
```

## 📞 Support

- **Token Issues**: tokens@prsm.ai
- **Staking Support**: staking@prsm.ai
- **DeFi Integration**: defi@prsm.ai
- **Governance**: governance@prsm.ai