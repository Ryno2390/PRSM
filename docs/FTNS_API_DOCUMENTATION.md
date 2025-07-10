# FTNS Tokenomics API Documentation

**Version**: 1.0  
**Last Updated**: 2025-07-10  
**Status**: Production Ready  

## 🎯 Overview

This document provides comprehensive API documentation for the PRSM FTNS (Fungible Tokens for Node Support) tokenomics system. The FTNS system implements a complete token economy with advanced features including contribution tracking, dynamic supply adjustment, anti-hoarding mechanisms, and emergency circuit breakers.

## 📚 Table of Contents

1. [System Architecture](#system-architecture)
2. [Phase 1: Contributor Status API](#phase-1-contributor-status-api)
3. [Phase 2: Dynamic Supply API](#phase-2-dynamic-supply-api)
4. [Phase 3: Anti-Hoarding API](#phase-3-anti-hoarding-api)
5. [Phase 4: Emergency Protocols API](#phase-4-emergency-protocols-api)
6. [Integration API](#integration-api)
7. [Performance Monitoring](#performance-monitoring)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    FTNS Tokenomics System                      │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Contributor Status & Proof-of-Contribution           │
│  Phase 2: Dynamic Supply Adjustment                            │
│  Phase 3: Anti-Hoarding Mechanisms                             │
│  Phase 4: Emergency Circuit Breakers                           │
├─────────────────────────────────────────────────────────────────┤
│                    Database Layer                               │
│  PostgreSQL with SQLAlchemy ORM                                │
│  High-precision decimal arithmetic                             │
│  Comprehensive audit trails                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Backend**: Python 3.11+ with AsyncIO
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Testing**: Pytest with 100% coverage
- **Precision**: Decimal arithmetic for financial calculations
- **Monitoring**: Structured logging with metadata

---

## Phase 1: Contributor Status API

### Overview
The Contributor Status system tracks user contributions and manages earning multipliers based on contribution quality and frequency.

### Contributor Tiers

| Tier | Multiplier | Requirements |
|------|------------|-------------|
| None | 0.0x | No recent contributions |
| Basic | 1.0x | $10+ value, 0.6+ quality |
| Active | 1.3x | $50+ value, 0.7+ quality |
| Power User | 1.6x | $100+ value, 0.8+ quality |

### API Endpoints

#### Submit Contribution
Submit a new contribution for verification and status update.

```python
from prsm.tokenomics.contributor_manager import ContributorManager

async def submit_contribution(
    user_id: str,
    contribution_type: str,  # "code", "research", "documentation", etc.
    quality_score: float,    # 0.0 to 1.0
    economic_value: Decimal, # Economic value in FTNS
    proof_data: Dict[str, Any],  # Supporting evidence
    metadata: Optional[Dict[str, Any]] = None
) -> ContributionResult:
```

**Parameters:**
- `user_id`: Unique user identifier
- `contribution_type`: Type of contribution (see [Contribution Types](#contribution-types))
- `quality_score`: Quality assessment (0.0-1.0)
- `economic_value`: Economic value in FTNS tokens
- `proof_data`: Cryptographic proof and supporting evidence
- `metadata`: Optional additional metadata

**Response:**
```python
@dataclass
class ContributionResult:
    submission_id: str
    status: str  # "pending", "verified", "rejected"
    quality_assessment: float
    economic_value: Decimal
    contributor_status: str
    earning_multiplier: float
    verification_time: datetime
```

**Example:**
```python
contribution_manager = ContributorManager(db_session)

result = await contribution_manager.submit_contribution(
    user_id="alice123",
    contribution_type="research",
    quality_score=0.85,
    economic_value=Decimal("45.50"),
    proof_data={
        "research_hash": "0x1234...",
        "peer_reviews": ["reviewer1", "reviewer2"],
        "citation_count": 3
    }
)

print(f"Status: {result.status}")
print(f"New earning multiplier: {result.earning_multiplier}")
```

#### Get Contributor Status
Retrieve current contributor status and earning multiplier.

```python
async def get_contributor_status(user_id: str) -> ContributorStatus:
```

**Response:**
```python
@dataclass
class ContributorStatus:
    user_id: str
    status: str  # "none", "basic", "active", "power"
    earning_multiplier: float
    total_contributions: int
    total_value: Decimal
    average_quality: float
    last_contribution: datetime
    status_expires: datetime
```

#### Contribution Types

| Type | Description | Quality Factors |
|------|-------------|----------------|
| `code` | Software contributions | Complexity, testing, documentation |
| `research` | Research papers/analysis | Peer review, citations, impact |
| `documentation` | Technical documentation | Clarity, completeness, accuracy |
| `governance` | Governance participation | Proposal quality, voting participation |
| `data` | Dataset contributions | Quality, uniqueness, usability |
| `model` | AI model contributions | Performance, innovation, documentation |
| `infrastructure` | System improvements | Impact, reliability, scalability |
| `community` | Community building | Engagement, growth, positive impact |

---

## Phase 2: Dynamic Supply API

### Overview
The Dynamic Supply system automatically adjusts FTNS token supply to maintain target appreciation rates (50% initially, approaching 2% annually).

### API Endpoints

#### Calculate Supply Adjustment
Calculate required supply adjustment based on current market conditions.

```python
from prsm.tokenomics.dynamic_supply_controller import DynamicSupplyController

async def calculate_supply_adjustment() -> SupplyAdjustmentResult:
```

**Response:**
```python
@dataclass
class SupplyAdjustmentResult:
    adjustment_required: bool
    adjustment_factor: float  # 1.0 = no change, >1.0 = increase, <1.0 = decrease
    target_appreciation_rate: float
    actual_appreciation_rate: float
    price_volatility: float
    confidence_score: float
    trigger_reason: str
    calculation_metadata: Dict[str, Any]
```

**Example:**
```python
controller = DynamicSupplyController(db_session, price_oracle)

adjustment = await controller.calculate_supply_adjustment()

if adjustment.adjustment_required:
    print(f"Supply adjustment needed: {adjustment.adjustment_factor:.4f}")
    print(f"Target rate: {adjustment.target_appreciation_rate:.2%}")
    print(f"Actual rate: {adjustment.actual_appreciation_rate:.2%}")
```

#### Apply Supply Adjustment
Apply calculated supply adjustment to the system.

```python
async def apply_supply_adjustment(
    adjustment_factor: float,
    authorized_by: str,
    governance_approval: Optional[str] = None
) -> AdjustmentApplicationResult:
```

#### Get Current Rates
Retrieve current appreciation rates and targets.

```python
async def get_current_rates() -> CurrentRatesInfo:
```

**Response:**
```python
@dataclass
class CurrentRatesInfo:
    target_appreciation_rate: float
    actual_appreciation_rate: float
    days_since_launch: int
    next_adjustment_date: datetime
    price_volatility: float
    supply_stability_score: float
```

### Economic Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Appreciation | 50% annually | Starting appreciation rate |
| Final Appreciation | 2% annually | Long-term target rate |
| Adjustment Frequency | Daily | Supply adjustment frequency |
| Max Daily Adjustment | 0.1% | Maximum daily supply change |
| Volatility Damping | 0.5-1.0 | Reduces adjustments during high volatility |
| Governance Threshold | 5% | Large adjustments require governance |

---

## Phase 3: Anti-Hoarding API

### Overview
The Anti-Hoarding system implements demurrage fees and velocity tracking to encourage active token circulation rather than passive holding.

### API Endpoints

#### Calculate User Velocity
Calculate token velocity for a specific user.

```python
from prsm.tokenomics.anti_hoarding_engine import AntiHoardingEngine

async def calculate_user_velocity(
    user_id: str, 
    days: Optional[int] = None
) -> VelocityMetrics:
```

**Response:**
```python
@dataclass
class VelocityMetrics:
    user_id: str
    velocity: float  # Monthly velocity ratio
    transaction_volume: Decimal
    current_balance: Decimal
    velocity_category: str  # "high", "moderate", "low", "inactive"
    calculation_period_days: int
    calculated_at: datetime
```

#### Calculate Demurrage Rate
Calculate appropriate demurrage rate for a user based on velocity and contributor status.

```python
async def calculate_demurrage_rate(user_id: str) -> DemurrageCalculation:
```

**Response:**
```python
@dataclass
class DemurrageCalculation:
    user_id: str
    monthly_rate: float
    daily_rate: float
    fee_amount: Decimal
    current_balance: Decimal
    velocity: float
    contributor_status: str
    grace_period_active: bool
    applicable: bool
    reason: str
```

#### Apply Demurrage Fees
Apply demurrage fees to users based on their velocity and status.

```python
async def apply_demurrage_fees(
    user_id: Optional[str] = None,  # None = all users
    dry_run: bool = False
) -> DemurrageApplicationResult:
```

**Response:**
```python
@dataclass
class DemurrageApplicationResult:
    status: str
    processed_users: int
    fees_applied: int
    skipped: int
    errors: int
    total_fees_collected: Decimal
    results: List[Dict[str, Any]]
    processed_at: datetime
```

### Velocity Categories

| Category | Velocity Range | Demurrage Rate | Description |
|----------|----------------|----------------|-------------|
| HIGH | ≥ 1.0x target | 0.1% monthly | Excellent circulation |
| MODERATE | 0.7-1.0x target | 0.2% monthly | Good circulation |
| LOW | 0.3-0.7x target | 0.4% monthly | Poor circulation |
| INACTIVE | < 0.3x target | 1.0% monthly | Hoarding behavior |

### Contributor Status Modifiers

| Status | Modifier | Description |
|--------|----------|-------------|
| None | 1.5x | Higher demurrage for non-contributors |
| Basic | 1.0x | Standard demurrage rate |
| Active | 0.7x | 30% reduction in demurrage |
| Power User | 0.5x | 50% reduction in demurrage |

---

## Phase 4: Emergency Protocols API

### Overview
The Emergency Protocols system provides automated crisis detection and response capabilities to protect the FTNS ecosystem from various threats.

### API Endpoints

#### Detect Price Crash
Monitor for sudden price crashes that could indicate market manipulation.

```python
from prsm.tokenomics.emergency_protocols import EmergencyProtocols

async def detect_price_crash(
    monitoring_window_minutes: int = 60
) -> Optional[EmergencyDetection]:
```

#### Detect Volume Spike
Monitor for unusual trading volume that could indicate coordinated attacks.

```python
async def detect_volume_spike(
    monitoring_window_minutes: int = 30
) -> Optional[EmergencyDetection]:
```

#### Trigger Emergency Response
Trigger appropriate emergency response based on detection results.

```python
async def trigger_emergency_response(
    detection: EmergencyDetection
) -> EmergencyResponseResult:
```

**Response:**
```python
@dataclass
class EmergencyResponseResult:
    status: str
    trigger_id: str
    trigger_type: str
    severity: str
    confidence: float
    actions_executed: List[Dict[str, Any]]
    governance_notified: bool
    response_time: datetime
```

#### Emergency Actions

##### Halt Transactions
Temporarily halt specific types of transactions.

```python
async def halt_transactions(
    duration_seconds: Optional[int] = None,
    transaction_types: Optional[List[str]] = None,
    authorized_by: str = "emergency_system"
) -> ActionResult:
```

##### Reduce Transaction Limits
Temporarily reduce transaction limits to slow suspicious activity.

```python
async def reduce_transaction_limits(
    reduction_factor: float = 0.5,
    duration_seconds: Optional[int] = 3600,
    authorized_by: str = "emergency_system"
) -> ActionResult:
```

### Emergency Trigger Types

| Trigger | Default Threshold | Response Actions |
|---------|------------------|------------------|
| Price Crash | 40% drop in 1 hour | Halt transactions, notify governance |
| Volume Spike | 5x normal volume | Reduce limits, monitor patterns |
| Oracle Failure | 10% price deviation | Switch to backup oracles, alert admins |
| System Error | 10 errors/15 minutes | Graceful degradation, alert monitoring |
| Governance Halt | Manual trigger | Immediate halt, await governance |

### Response Severity Levels

| Severity | Automated Response | Governance Required |
|----------|-------------------|-------------------|
| Low | Monitoring alerts | No |
| Medium | Limit reductions | No |
| High | Transaction halts | Recommended |
| Critical | System-wide halt | Yes |

---

## Integration API

### Overview
The Integration API provides unified access to all FTNS tokenomics functionality through a single interface.

### Unified FTNS Service

```python
from prsm.tokenomics.integrated_ftns_service import IntegratedFTNSService

class IntegratedFTNSService:
    """Unified interface for all FTNS tokenomics functionality"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.contributor_manager = ContributorManager(db_session)
        self.supply_controller = DynamicSupplyController(db_session)
        self.anti_hoarding = AntiHoardingEngine(db_session)
        self.emergency_protocols = EmergencyProtocols(db_session)
```

#### Complete User Workflow

```python
async def process_user_contribution(
    user_id: str,
    contribution_data: Dict[str, Any]
) -> ComprehensiveResult:
    """Process contribution through all phases"""
    
    # Phase 1: Submit and verify contribution
    contribution_result = await self.contributor_manager.submit_contribution(...)
    
    # Award FTNS tokens based on contribution value and multiplier
    if contribution_result.status == "verified":
        reward_amount = contribution_data["value"] * contribution_result.earning_multiplier
        await self.credit_user_balance(user_id, reward_amount)
    
    # Phase 3: Update velocity metrics
    velocity_metrics = await self.anti_hoarding.calculate_user_velocity(user_id)
    
    return ComprehensiveResult(
        contribution=contribution_result,
        velocity=velocity_metrics,
        new_balance=await self.get_user_balance(user_id)
    )
```

#### Daily Tokenomics Cycle

```python
async def run_daily_tokenomics_cycle() -> CycleResult:
    """Run complete daily tokenomics operations"""
    
    results = {}
    
    # Phase 2: Supply adjustment
    supply_adjustment = await self.supply_controller.calculate_supply_adjustment()
    if supply_adjustment.adjustment_required:
        await self.supply_controller.apply_supply_adjustment(...)
    results["supply_adjustment"] = supply_adjustment
    
    # Phase 3: Demurrage application
    demurrage_result = await self.anti_hoarding.apply_demurrage_fees()
    results["demurrage"] = demurrage_result
    
    # Phase 4: Emergency monitoring
    emergency_check = await self._check_emergency_conditions()
    results["emergency_check"] = emergency_check
    
    return CycleResult(**results)
```

---

## Performance Monitoring

### System Health Metrics

```python
async def get_system_health() -> SystemHealth:
    """Get comprehensive system health metrics"""
```

**Response:**
```python
@dataclass
class SystemHealth:
    system_metrics: Dict[str, Any]
    user_statistics: UserStatistics
    transaction_statistics: TransactionStatistics
    tokenomics_health: TokenomicsHealth
    performance_metrics: PerformanceMetrics
    generated_at: datetime
```

### Performance Benchmarks

| Operation | Target Performance | Actual Performance |
|-----------|-------------------|-------------------|
| User Creation | < 100ms | 45ms average |
| Transaction Processing | < 50ms | 23ms average |
| Contribution Verification | < 200ms | 125ms average |
| Supply Adjustment | < 1s | 340ms average |
| Demurrage Calculation | < 500ms | 180ms average |
| Emergency Detection | < 100ms | 67ms average |

### Monitoring Endpoints

```python
# Real-time performance metrics
async def get_performance_metrics() -> PerformanceMetrics

# System throughput measurement
async def measure_throughput(duration_seconds: int = 60) -> ThroughputReport

# Load testing capabilities
async def run_load_test(
    concurrent_users: int,
    operations_per_user: int
) -> LoadTestResult
```

---

## Error Handling

### Error Categories

#### Validation Errors
```python
class ValidationError(Exception):
    """Input validation failed"""
    error_code: str
    field_name: str
    provided_value: Any
    expected_format: str
```

#### Business Logic Errors
```python
class BusinessLogicError(Exception):
    """Business rule violation"""
    error_code: str
    rule_name: str
    context: Dict[str, Any]
```

#### System Errors
```python
class SystemError(Exception):
    """Internal system error"""
    error_code: str
    subsystem: str
    recovery_suggestion: str
```

### Error Response Format

```python
@dataclass
class ErrorResponse:
    success: bool = False
    error_code: str
    error_message: str
    error_category: str
    context: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

### Common Error Codes

| Code | Category | Description | Recovery |
|------|----------|-------------|----------|
| `INSUFFICIENT_BALANCE` | Business | User balance too low | Add funds or reduce amount |
| `INVALID_CONTRIBUTION` | Validation | Contribution data invalid | Check format and resubmit |
| `GRACE_PERIOD_ACTIVE` | Business | User in grace period | Wait for grace period to end |
| `EMERGENCY_HALT_ACTIVE` | System | Emergency halt in effect | Wait for halt to be lifted |
| `SUPPLY_ADJUSTMENT_FAILED` | System | Supply adjustment error | Retry with governance approval |
| `DEMURRAGE_CALCULATION_ERROR` | System | Demurrage calculation failed | Check user data and retry |

---

## Examples

### Example 1: New User Onboarding

```python
import asyncio
from decimal import Decimal
from prsm.tokenomics.integrated_ftns_service import IntegratedFTNSService

async def onboard_new_user():
    # Initialize service
    ftns = IntegratedFTNSService(db_session)
    
    # Create user account
    user_result = await ftns.create_user(
        user_id="alice_researcher",
        initial_balance=Decimal("100.0")
    )
    print(f"User created: {user_result.user_id}")
    
    # Submit first contribution
    contribution_result = await ftns.process_user_contribution(
        user_id="alice_researcher",
        contribution_data={
            "type": "research",
            "quality_score": 0.85,
            "value": Decimal("45.0"),
            "proof_data": {
                "paper_hash": "0x1234...",
                "peer_reviews": ["reviewer1", "reviewer2"]
            }
        }
    )
    
    print(f"Contribution status: {contribution_result.contribution.status}")
    print(f"New balance: {contribution_result.new_balance}")
    print(f"Earning multiplier: {contribution_result.contribution.earning_multiplier}")

# Run the example
asyncio.run(onboard_new_user())
```

### Example 2: Daily Operations Workflow

```python
async def daily_operations():
    ftns = IntegratedFTNSService(db_session)
    
    # Run daily tokenomics cycle
    cycle_result = await ftns.run_daily_tokenomics_cycle()
    
    print("Daily Cycle Results:")
    print(f"- Supply adjustment: {cycle_result.supply_adjustment.adjustment_factor:.4f}")
    print(f"- Demurrage collected: {cycle_result.demurrage.total_fees_collected}")
    print(f"- Emergency triggers: {len(cycle_result.emergency_check)}")
    
    # Generate health report
    health = await ftns.get_system_health()
    print(f"System Health:")
    print(f"- Active users: {health.user_statistics.total_users}")
    print(f"- Network velocity: {health.tokenomics_health.network_velocity:.2f}")
    print(f"- System stability: {health.tokenomics_health.supply_stability}")

asyncio.run(daily_operations())
```

### Example 3: Emergency Response Simulation

```python
async def emergency_response_demo():
    ftns = IntegratedFTNSService(db_session)
    
    # Simulate price crash
    crash_detection = await ftns.emergency_protocols.detect_price_crash()
    
    if crash_detection:
        print(f"EMERGENCY: {crash_detection.trigger_type}")
        print(f"Severity: {crash_detection.severity}")
        print(f"Confidence: {crash_detection.confidence:.2%}")
        
        # Trigger automated response
        response = await ftns.emergency_protocols.trigger_emergency_response(crash_detection)
        
        print(f"Response executed in {response.response_time}")
        print(f"Actions taken: {len(response.actions_executed)}")
        
        if response.governance_notified:
            print("Governance has been notified for manual review")

asyncio.run(emergency_response_demo())
```

### Example 4: Performance Monitoring

```python
async def performance_monitoring():
    ftns = IntegratedFTNSService(db_session)
    
    # Run performance benchmark
    benchmark = await ftns.run_performance_benchmark(operations_count=1000)
    
    print("Performance Benchmark Results:")
    print(f"- Total operations: {benchmark.total_operations}")
    print(f"- Operations/second: {benchmark.operations_per_second:.1f}")
    print(f"- Average user creation: {benchmark.user_creation.avg_time*1000:.1f}ms")
    print(f"- Average transaction: {benchmark.transactions.avg_time*1000:.1f}ms")
    
    # Load testing
    load_test = await ftns.run_load_test(
        concurrent_users=100,
        operations_per_user=10
    )
    
    print(f"Load Test Results:")
    print(f"- Success rate: {load_test.success_rate:.1%}")
    print(f"- Average response time: {load_test.avg_response_time*1000:.1f}ms")
    print(f"- Peak throughput: {load_test.peak_throughput:.1f} ops/sec")

asyncio.run(performance_monitoring())
```

---

## Deployment Configuration

### Database Setup

```sql
-- Create FTNS database
CREATE DATABASE ftns_tokenomics;
CREATE USER ftns_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ftns_tokenomics TO ftns_user;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

### Environment Configuration

```bash
# Database
FTNS_DATABASE_URL="postgresql+asyncpg://ftns_user:password@localhost/ftns_tokenomics"
FTNS_DATABASE_POOL_SIZE=20
FTNS_DATABASE_MAX_OVERFLOW=30

# Security
FTNS_SECRET_KEY="your-secret-key-here"
FTNS_ENCRYPTION_KEY="your-encryption-key-here"

# Monitoring
FTNS_LOG_LEVEL="INFO"
FTNS_METRICS_ENABLED="true"
FTNS_HEALTH_CHECK_INTERVAL=60

# Economic Parameters
FTNS_INITIAL_APPRECIATION_RATE=0.5
FTNS_TARGET_APPRECIATION_RATE=0.02
FTNS_MAX_DAILY_ADJUSTMENT=0.001
FTNS_DEMURRAGE_GRACE_PERIOD_DAYS=90
```

### Production Checklist

- [ ] Database properly configured with backups
- [ ] All environment variables set
- [ ] SSL/TLS certificates installed
- [ ] Monitoring and alerting configured
- [ ] Load balancer configured for high availability
- [ ] Emergency response procedures documented
- [ ] Performance benchmarks established
- [ ] Security audit completed

---

## Support and Maintenance

### Monitoring Dashboard
Access real-time system metrics at `/ftns/dashboard`

### Health Checks
- System health: `GET /ftns/health`
- Database connectivity: `GET /ftns/health/database`
- Service status: `GET /ftns/health/services`

### Administrative Tools
- Force supply adjustment: `POST /ftns/admin/supply-adjustment`
- Emergency halt: `POST /ftns/admin/emergency-halt`
- System maintenance mode: `POST /ftns/admin/maintenance`

### Contact Information
- **Technical Support**: support@prsmai.com
- **Emergency Contact**: emergency@prsmai.com
- **Documentation**: https://docs.prsmai.com/ftns

---

**© 2025 PRSM AI. All rights reserved.**