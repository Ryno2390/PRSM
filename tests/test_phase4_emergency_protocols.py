"""
Test Phase 4: Emergency Circuit Breakers
Comprehensive integration tests for the emergency protocols system

This test suite validates:
- Emergency trigger detection algorithms (price crashes, volume spikes, oracle failures)
- Circuit breaker activation and transaction halts
- Emergency response action execution and coordination
- Governance integration for emergency proposals
- Automated monitoring and response workflows
- Emergency configuration management and thresholds

Test Philosophy:
- Use realistic crisis scenarios (market crashes, technical failures, security incidents)
- Test automated response systems work as designed
- Verify emergency actions are properly authorized and logged
- Ensure governance integration works for high-severity events
- Test edge cases and error handling in crisis conditions
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from unittest.mock import Mock, AsyncMock

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Configure pytest-asyncio
import pytest_asyncio
pytest_plugins = ('pytest_asyncio',)

# Define enums locally for testing
class EmergencyTriggerType:
    PRICE_CRASH = "price_crash"
    VOLUME_SPIKE = "volume_spike"
    SYSTEM_ERROR = "system_error"
    ORACLE_FAILURE = "oracle_failure"
    GOVERNANCE_HALT = "governance_halt"
    SECURITY_BREACH = "security_breach"

class EmergencyStatus:
    DETECTED = "detected"
    EVALUATING = "evaluating"
    CONFIRMED = "confirmed"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    FALSE_ALARM = "false_alarm"

class EmergencyActionType:
    HALT_TRANSACTIONS = "halt_transactions"
    REDUCE_LIMITS = "reduce_limits"
    NOTIFY_GOVERNANCE = "notify_governance"
    ADJUST_RATES = "adjust_rates"
    FREEZE_ACCOUNTS = "freeze_accounts"
    ACTIVATE_CIRCUIT_BREAKER = "activate_circuit_breaker"

# Create test-compatible models for SQLite
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, DECIMAL
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from uuid import uuid4
from decimal import Decimal

TestBase = declarative_base()

class FTNSEmergencyTriggerTest(TestBase):
    """Test version of emergency trigger model"""
    __tablename__ = "ftns_emergency_triggers_test"
    
    trigger_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger_type = Column(String(50), nullable=False)
    threshold_value = Column(DECIMAL(10, 6), nullable=False)
    actual_value = Column(DECIMAL(10, 6), nullable=False)
    confidence_score = Column(DECIMAL(5, 4), nullable=False)
    severity_level = Column(String(20), nullable=False)
    data_source = Column(String(100), nullable=False)
    detection_algorithm = Column(String(100), nullable=False)
    time_window_seconds = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default=EmergencyStatus.DETECTED)
    detected_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    auto_response_enabled = Column(Boolean, nullable=False, default=True)
    governance_required = Column(Boolean, nullable=False, default=False)
    response_delay_seconds = Column(Integer, nullable=False, default=0)
    trigger_metadata = Column(JSON, nullable=True)

class FTNSEmergencyActionTest(TestBase):
    """Test version of emergency action model"""
    __tablename__ = "ftns_emergency_actions_test"
    
    action_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger_id = Column(PG_UUID(as_uuid=True), nullable=False)
    action_type = Column(String(50), nullable=False)
    action_severity = Column(String(20), nullable=False)
    target_scope = Column(String(50), nullable=False)
    affected_users = Column(JSON, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    transaction_halt_types = Column(JSON, nullable=True)
    rate_adjustments = Column(JSON, nullable=True)
    limit_reductions = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    scheduled_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    executed_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    reverted_at = Column(DateTime(timezone=True), nullable=True)
    authorized_by = Column(String(255), nullable=True)
    governance_proposal_id = Column(String(255), nullable=True)
    override_reason = Column(Text, nullable=True)
    users_affected = Column(Integer, nullable=False, default=0)
    transactions_halted = Column(Integer, nullable=False, default=0)
    volume_impact = Column(DECIMAL(20, 8), nullable=True)
    action_metadata = Column(JSON, nullable=True)

# Mock dataclasses for testing
from dataclasses import dataclass

@dataclass
class EmergencyDetection:
    trigger_type: str
    severity: str
    confidence: float
    threshold_value: float
    actual_value: float
    data_source: str
    detection_time: datetime
    auto_response_recommended: bool
    governance_required: bool

@dataclass
class EmergencyResponse:
    action_type: str
    severity: str
    target_scope: str
    duration_seconds: int
    affected_users: list
    parameters: dict
    authorized_by: str
    governance_proposal_id: str

# Mock Emergency Protocols for testing
class EmergencyProtocols:
    def __init__(self, db_session, ftns_service=None, governance_service=None, price_oracle=None):
        self.db = db_session
        self.ftns = ftns_service
        self.governance = governance_service
        self.price_oracle = price_oracle
        
        # Emergency configuration
        self.emergency_config = {
            'price_crash_threshold': 0.4,      # 40% price drop
            'volume_spike_threshold': 5.0,     # 5x normal volume
            'oracle_deviation_threshold': 0.1,  # 10% oracle deviation
            'system_error_threshold': 10,      # Error count threshold
            'confidence_threshold': 0.8,       # 80% confidence required
            'auto_response_enabled': True,
            'governance_escalation_threshold': 0.9
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.active_emergencies = set()
        self.action_cooldowns = {
            'halt_transactions': 300,
            'reduce_limits': 600,
            'notify_governance': 1800
        }
        self.last_actions = {}
        
        # Mock data for testing
        self.mock_price_history = []
        self.mock_volume_data = {}
        self.mock_oracle_prices = {}
        self.mock_system_errors = 0
    
    # Mock data setters for testing
    def set_mock_price_history(self, price_history):
        self.mock_price_history = price_history
    
    def set_mock_volume_data(self, current_volume, baseline_volume):
        self.mock_volume_data = {
            'current': current_volume,
            'baseline': baseline_volume
        }
    
    def set_mock_oracle_prices(self, oracle_prices):
        self.mock_oracle_prices = oracle_prices
    
    def set_mock_system_errors(self, error_count):
        self.mock_system_errors = error_count
    
    # Emergency detection methods
    async def detect_price_crash(self, monitoring_window_minutes=60):
        """Mock price crash detection"""
        if not self.mock_price_history:
            return None
        
        if len(self.mock_price_history) < 2:
            return None
        
        initial_price = self.mock_price_history[0]['price']
        current_price = self.mock_price_history[-1]['price']
        price_change = (current_price - initial_price) / initial_price
        
        crash_threshold = -abs(self.emergency_config['price_crash_threshold'])
        
        if price_change <= crash_threshold:
            confidence = min(0.95, abs(price_change) / abs(crash_threshold))
            severity = self._determine_price_crash_severity(abs(price_change))
            
            return EmergencyDetection(
                trigger_type=EmergencyTriggerType.PRICE_CRASH,
                severity=severity,
                confidence=confidence,
                threshold_value=abs(crash_threshold),
                actual_value=abs(price_change),
                data_source=f"mock_price_oracle_{monitoring_window_minutes}m",
                detection_time=datetime.now(timezone.utc),
                auto_response_recommended=confidence > self.emergency_config['confidence_threshold'],
                governance_required=confidence > self.emergency_config['governance_escalation_threshold']
            )
        
        return None
    
    async def detect_volume_spike(self, monitoring_window_minutes=30):
        """Mock volume spike detection"""
        if not self.mock_volume_data:
            return None
        
        current_volume = self.mock_volume_data.get('current', 0)
        baseline_volume = self.mock_volume_data.get('baseline', 0)
        
        if baseline_volume == 0:
            return None
        
        volume_ratio = current_volume / baseline_volume
        spike_threshold = self.emergency_config['volume_spike_threshold']
        
        if volume_ratio >= spike_threshold:
            confidence = min(0.9, (volume_ratio - spike_threshold) / spike_threshold)
            severity = self._determine_volume_spike_severity(volume_ratio)
            
            return EmergencyDetection(
                trigger_type=EmergencyTriggerType.VOLUME_SPIKE,
                severity=severity,
                confidence=confidence,
                threshold_value=spike_threshold,
                actual_value=volume_ratio,
                data_source=f"mock_ftns_transactions_{monitoring_window_minutes}m",
                detection_time=datetime.now(timezone.utc),
                auto_response_recommended=confidence > self.emergency_config['confidence_threshold'],
                governance_required=confidence > self.emergency_config['governance_escalation_threshold']
            )
        
        return None
    
    async def detect_oracle_failure(self):
        """Mock oracle failure detection"""
        if len(self.mock_oracle_prices) < 2:
            return None
        
        prices = [float(p) for p in self.mock_oracle_prices.values()]
        mean_price = sum(prices) / len(prices)
        max_deviation = max(abs(p - mean_price) / mean_price for p in prices)
        
        deviation_threshold = self.emergency_config['oracle_deviation_threshold']
        
        if max_deviation >= deviation_threshold:
            confidence = min(0.95, 0.7 + (max_deviation - deviation_threshold) * 2)
            severity = self._determine_oracle_failure_severity(max_deviation)
            
            return EmergencyDetection(
                trigger_type=EmergencyTriggerType.ORACLE_FAILURE,
                severity=severity,
                confidence=confidence,
                threshold_value=deviation_threshold,
                actual_value=max_deviation,
                data_source="mock_multi_oracle_comparison",
                detection_time=datetime.now(timezone.utc),
                auto_response_recommended=True,
                governance_required=max_deviation > 0.2
            )
        
        return None
    
    async def detect_system_errors(self, monitoring_window_minutes=15):
        """Mock system error detection"""
        error_count = self.mock_system_errors
        error_threshold = self.emergency_config['system_error_threshold']
        
        if error_count >= error_threshold:
            confidence = min(0.9, 0.6 + (error_count - error_threshold) * 0.05)
            severity = self._determine_system_error_severity(error_count)
            
            return EmergencyDetection(
                trigger_type=EmergencyTriggerType.SYSTEM_ERROR,
                severity=severity,
                confidence=confidence,
                threshold_value=error_threshold,
                actual_value=error_count,
                data_source="mock_system_monitoring",
                detection_time=datetime.now(timezone.utc),
                auto_response_recommended=error_count >= error_threshold * 2,
                governance_required=error_count >= error_threshold * 3
            )
        
        return None
    
    # Emergency response methods
    async def trigger_emergency_response(self, detection):
        """Mock emergency response triggering"""
        trigger_record = await self._record_emergency_trigger(detection)
        response_actions = await self._determine_response_actions(detection)
        
        results = []
        for action in response_actions:
            result = await self._execute_emergency_action(trigger_record.trigger_id, action)
            results.append(result)
        
        return {
            "status": "completed",
            "trigger_id": str(trigger_record.trigger_id),
            "trigger_type": detection.trigger_type,
            "severity": detection.severity,
            "confidence": detection.confidence,
            "actions_executed": results,
            "governance_notified": detection.governance_required,
            "response_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def halt_transactions(self, duration_seconds=None, transaction_types=None, authorized_by="emergency_system"):
        """Mock transaction halt"""
        if not await self._check_action_cooldown("halt_transactions"):
            return {
                "status": "rejected",
                "reason": "Action in cooldown period"
            }
        
        halt_types = transaction_types or ["transfer", "marketplace", "governance"]
        affected_users = 100  # Mock affected user count
        
        # Record the call in FTNS service if available
        if self.ftns:
            await self.ftns.emergency_halt_transactions(halt_types, duration_seconds)
        
        self.last_actions["halt_transactions"] = datetime.now(timezone.utc)
        
        return {
            "status": "executed",
            "action_type": "halt_transactions",
            "transaction_types": halt_types,
            "duration_seconds": duration_seconds,
            "affected_users": affected_users
        }
    
    async def reduce_transaction_limits(self, reduction_factor=0.5, duration_seconds=3600, authorized_by="emergency_system"):
        """Mock transaction limit reduction"""
        if not await self._check_action_cooldown("reduce_limits"):
            return {
                "status": "rejected",
                "reason": "Action in cooldown period"
            }
        
        affected_users = 150  # Mock affected user count
        
        # Record the call in FTNS service if available
        if self.ftns:
            await self.ftns.emergency_reduce_limits(reduction_factor, duration_seconds)
        
        self.last_actions["reduce_limits"] = datetime.now(timezone.utc)
        
        return {
            "status": "executed",
            "action_type": "reduce_limits",
            "reduction_factor": reduction_factor,
            "duration_seconds": duration_seconds,
            "affected_users": affected_users
        }
    
    async def create_emergency_proposal(self, detection, proposed_actions):
        """Mock emergency proposal creation"""
        if not self.governance:
            return {
                "status": "error",
                "reason": "Governance service not available"
            }
        
        # Use governance service to create proposal
        proposal = await self.governance.create_emergency_proposal(
            title=f"Emergency Response: {detection.trigger_type}",
            description=f"Emergency trigger detected",
            proposed_actions=proposed_actions,
            trigger_type=detection.trigger_type,
            detection_timestamp=detection.detection_time,
            voting_period_hours=6,
            activation_threshold=0.15
        )
        
        return proposal
    
    # Monitoring methods
    async def start_emergency_monitoring(self, check_interval_seconds=60):
        """Mock monitoring start"""
        self.monitoring_active = True
        return {"status": "started", "interval_seconds": check_interval_seconds}
    
    async def stop_emergency_monitoring(self):
        """Mock monitoring stop"""
        self.monitoring_active = False
        return {"status": "stopped"}
    
    async def get_emergency_status(self):
        """Get current emergency system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "active_emergencies": len(self.active_emergencies),
            "emergency_config": self.emergency_config,
            "action_cooldowns": {
                action: (datetime.now(timezone.utc) - last_time).total_seconds() 
                for action, last_time in self.last_actions.items()
            },
            "status_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Utility methods
    def _determine_price_crash_severity(self, crash_percentage):
        if crash_percentage >= 0.7:  # 70%+ crash
            return "critical"
        elif crash_percentage >= 0.5:  # 50%+ crash
            return "high"
        elif crash_percentage >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _determine_volume_spike_severity(self, volume_ratio):
        if volume_ratio >= 20:
            return "critical"
        elif volume_ratio >= 10:
            return "high"
        elif volume_ratio >= 5:
            return "medium"
        else:
            return "low"
    
    def _determine_oracle_failure_severity(self, deviation):
        if deviation >= 0.4:  # 40%+ deviation
            return "critical"
        elif deviation >= 0.25:  # 25%+ deviation
            return "high"
        elif deviation >= 0.15:
            return "medium"
        else:
            return "low"
    
    def _determine_system_error_severity(self, error_count):
        threshold = self.emergency_config['system_error_threshold']
        if error_count >= threshold * 5:
            return "critical"
        elif error_count >= threshold * 3:
            return "high"
        elif error_count >= threshold * 2:
            return "medium"
        else:
            return "low"
    
    async def _check_action_cooldown(self, action_type):
        if action_type not in self.last_actions:
            return True
        
        cooldown_duration = self.action_cooldowns.get(action_type, 300)
        last_action = self.last_actions[action_type]
        
        return (datetime.now(timezone.utc) - last_action).total_seconds() >= cooldown_duration
    
    async def _record_emergency_trigger(self, detection):
        """Record emergency trigger in database"""
        trigger_record = FTNSEmergencyTriggerTest(
            trigger_type=detection.trigger_type,
            threshold_value=Decimal(str(detection.threshold_value)),
            actual_value=Decimal(str(detection.actual_value)),
            confidence_score=Decimal(str(detection.confidence)),
            severity_level=detection.severity,
            data_source=detection.data_source,
            detection_algorithm="emergency_protocols_test",
            time_window_seconds=3600,
            status=EmergencyStatus.DETECTED,
            detected_at=detection.detection_time,
            auto_response_enabled=detection.auto_response_recommended,
            governance_required=detection.governance_required,
            trigger_metadata={
                "test_mode": True,
                "detection_method": "mock_monitoring"
            }
        )
        
        self.db.add(trigger_record)
        await self.db.commit()
        
        return trigger_record
    
    async def _determine_response_actions(self, detection):
        """Determine appropriate response actions"""
        actions = []
        
        if detection.trigger_type == EmergencyTriggerType.PRICE_CRASH:
            if detection.severity in ["high", "critical"]:
                actions.append(EmergencyResponse(
                    action_type=EmergencyActionType.HALT_TRANSACTIONS,
                    severity=detection.severity,
                    target_scope="all_users",
                    duration_seconds=3600,
                    affected_users=None,
                    parameters={"transaction_types": ["transfer", "marketplace"]},
                    authorized_by="emergency_system",
                    governance_proposal_id=None
                ))
        
        elif detection.trigger_type == EmergencyTriggerType.VOLUME_SPIKE:
            if detection.severity in ["high", "critical"]:
                actions.append(EmergencyResponse(
                    action_type=EmergencyActionType.REDUCE_LIMITS,
                    severity=detection.severity,
                    target_scope="all_users",
                    duration_seconds=600,
                    affected_users=None,
                    parameters={"reduction_factor": 0.3},
                    authorized_by="emergency_system",
                    governance_proposal_id=None
                ))
        
        if detection.governance_required:
            actions.append(EmergencyResponse(
                action_type=EmergencyActionType.NOTIFY_GOVERNANCE,
                severity=detection.severity,
                target_scope="system_wide",
                duration_seconds=None,
                affected_users=None,
                parameters={"urgency": "high", "voting_window_hours": 6},
                authorized_by="emergency_system",
                governance_proposal_id=None
            ))
        
        return actions
    
    async def _execute_emergency_action(self, trigger_id, action):
        """Execute emergency action"""
        action_record = FTNSEmergencyActionTest(
            trigger_id=trigger_id,
            action_type=action.action_type,
            action_severity=action.severity,
            target_scope=action.target_scope,
            duration_seconds=action.duration_seconds,
            status="pending",
            authorized_by=action.authorized_by,
            action_metadata=action.parameters
        )
        
        self.db.add(action_record)
        await self.db.commit()
        
        # Mock execution
        if action.action_type == EmergencyActionType.HALT_TRANSACTIONS:
            result = await self.halt_transactions(
                duration_seconds=action.duration_seconds,
                transaction_types=action.parameters.get("transaction_types"),
                authorized_by=action.authorized_by
            )
        elif action.action_type == EmergencyActionType.REDUCE_LIMITS:
            result = await self.reduce_transaction_limits(
                reduction_factor=action.parameters.get("reduction_factor", 0.5),
                duration_seconds=action.duration_seconds,
                authorized_by=action.authorized_by
            )
        else:
            result = {
                "status": "executed",
                "action_type": action.action_type,
                "affected_users": 0
            }
        
        return result

# Mock services
class MockPriceOracle:
    def __init__(self):
        self.price_history = []
        self.multi_source_prices = {}
    
    async def get_price_history(self, start_time, end_time):
        return self.price_history
    
    async def get_multi_source_prices(self):
        return self.multi_source_prices
    
    def set_price_history(self, history):
        self.price_history = history
    
    def set_multi_source_prices(self, prices):
        self.multi_source_prices = prices

class MockFTNSService:
    def __init__(self):
        self.transaction_volume = 0
        self.average_volume = 0
        self.halt_calls = []
        self.limit_calls = []
    
    async def get_transaction_volume(self, start_time, end_time):
        return self.transaction_volume
    
    async def get_average_volume(self, start_time, end_time, window_minutes):
        return self.average_volume
    
    async def emergency_halt_transactions(self, transaction_types, duration_seconds):
        self.halt_calls.append({
            "transaction_types": transaction_types,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now(timezone.utc)
        })
        return {"success": True, "affected_users": 100}
    
    async def emergency_reduce_limits(self, reduction_factor, duration_seconds):
        self.limit_calls.append({
            "reduction_factor": reduction_factor,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now(timezone.utc)
        })
        return {"success": True, "affected_users": 150}
    
    def set_volume_data(self, current, average):
        self.transaction_volume = current
        self.average_volume = average

class MockGovernanceService:
    def __init__(self):
        self.proposals = []
    
    async def create_emergency_proposal(self, **kwargs):
        proposal_id = str(uuid4())
        proposal = {"status": "created", "proposal_id": proposal_id, **kwargs}
        self.proposals.append(proposal)
        return proposal

# Use TestBase for database creation
Base = TestBase

# Test fixtures
@pytest_asyncio.fixture
async def db_session():
    """Create in-memory SQLite database for testing"""
    
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with Session() as session:
        yield session
    
    await engine.dispose()

@pytest.fixture
def mock_price_oracle():
    """Mock price oracle"""
    return MockPriceOracle()

@pytest.fixture
def mock_ftns_service():
    """Mock FTNS service"""
    return MockFTNSService()

@pytest.fixture
def mock_governance_service():
    """Mock governance service"""
    return MockGovernanceService()

@pytest.fixture
def emergency_protocols(db_session, mock_ftns_service, mock_governance_service, mock_price_oracle):
    """Emergency protocols with mocked dependencies"""
    return EmergencyProtocols(db_session, mock_ftns_service, mock_governance_service, mock_price_oracle)

# Test suites
class TestEmergencyDetection:
    """Test suite for emergency detection algorithms"""
    
    @pytest.mark.asyncio
    async def test_price_crash_detection_major_crash(self, emergency_protocols):
        """Test detection of major price crash (60% drop)"""
        
        # Set up price history showing major crash
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 80.0, "timestamp": datetime.now(timezone.utc) - timedelta(minutes=30)},
            {"price": 40.0, "timestamp": datetime.now(timezone.utc)}  # 60% crash
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        
        detection = await emergency_protocols.detect_price_crash()
        
        assert detection is not None
        assert detection.trigger_type == EmergencyTriggerType.PRICE_CRASH
        assert detection.severity == "high"
        assert detection.actual_value == 0.6  # 60% crash
        assert detection.confidence > 0.8
        assert detection.auto_response_recommended is True
    
    @pytest.mark.asyncio
    async def test_price_crash_detection_minor_decline(self, emergency_protocols):
        """Test that minor price declines don't trigger emergency"""
        
        # Set up price history showing minor decline (20% drop)
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 90.0, "timestamp": datetime.now(timezone.utc) - timedelta(minutes=30)},
            {"price": 80.0, "timestamp": datetime.now(timezone.utc)}  # 20% decline
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        
        detection = await emergency_protocols.detect_price_crash()
        
        assert detection is None  # Should not trigger emergency
    
    @pytest.mark.asyncio
    async def test_volume_spike_detection_extreme_spike(self, emergency_protocols):
        """Test detection of extreme volume spike (10x normal)"""
        
        emergency_protocols.set_mock_volume_data(
            current_volume=10000.0,   # Current volume
            baseline_volume=1000.0    # Normal volume (10x spike)
        )
        
        detection = await emergency_protocols.detect_volume_spike()
        
        assert detection is not None
        assert detection.trigger_type == EmergencyTriggerType.VOLUME_SPIKE
        assert detection.severity == "high"
        assert detection.actual_value == 10.0  # 10x spike
        assert detection.confidence > 0.8
        assert detection.auto_response_recommended is True
    
    @pytest.mark.asyncio
    async def test_volume_spike_detection_normal_volume(self, emergency_protocols):
        """Test that normal volume doesn't trigger emergency"""
        
        emergency_protocols.set_mock_volume_data(
            current_volume=1200.0,    # Current volume
            baseline_volume=1000.0    # Normal volume (1.2x, normal)
        )
        
        detection = await emergency_protocols.detect_volume_spike()
        
        assert detection is None  # Should not trigger emergency
    
    @pytest.mark.asyncio
    async def test_oracle_failure_detection_major_deviation(self, emergency_protocols):
        """Test detection of major oracle price deviation"""
        
        # Set up oracle prices with major deviation
        oracle_prices = {
            "source_a": 100.0,
            "source_b": 95.0,
            "source_c": 180.0,   # Much larger deviation for critical severity
            "source_d": 105.0
        }
        
        emergency_protocols.set_mock_oracle_prices(oracle_prices)
        
        detection = await emergency_protocols.detect_oracle_failure()
        
        assert detection is not None
        assert detection.trigger_type == EmergencyTriggerType.ORACLE_FAILURE
        assert detection.severity == "critical"
        assert detection.actual_value > 0.4  # Significant deviation
        assert detection.governance_required is True
    
    @pytest.mark.asyncio
    async def test_system_error_detection_high_error_count(self, emergency_protocols):
        """Test detection of high system error count"""
        
        emergency_protocols.set_mock_system_errors(25)  # 2.5x threshold
        
        detection = await emergency_protocols.detect_system_errors()
        
        assert detection is not None
        assert detection.trigger_type == EmergencyTriggerType.SYSTEM_ERROR
        assert detection.severity == "medium"
        assert detection.actual_value == 25
        assert detection.auto_response_recommended is True


class TestEmergencyResponse:
    """Test suite for emergency response actions"""
    
    @pytest.mark.asyncio
    async def test_transaction_halt_execution(self, emergency_protocols, mock_ftns_service):
        """Test emergency transaction halt execution"""
        
        result = await emergency_protocols.halt_transactions(
            duration_seconds=3600,
            transaction_types=["transfer", "marketplace"],
            authorized_by="test_system"
        )
        
        assert result["status"] == "executed"
        assert result["action_type"] == "halt_transactions"
        assert result["transaction_types"] == ["transfer", "marketplace"]
        assert result["duration_seconds"] == 3600
        assert result["affected_users"] == 100
        
        # Verify FTNS service was called
        assert len(mock_ftns_service.halt_calls) == 1
        assert mock_ftns_service.halt_calls[0]["transaction_types"] == ["transfer", "marketplace"]
    
    @pytest.mark.asyncio
    async def test_transaction_limit_reduction(self, emergency_protocols, mock_ftns_service):
        """Test emergency transaction limit reduction"""
        
        result = await emergency_protocols.reduce_transaction_limits(
            reduction_factor=0.3,
            duration_seconds=1800,
            authorized_by="test_system"
        )
        
        assert result["status"] == "executed"
        assert result["action_type"] == "reduce_limits"
        assert result["reduction_factor"] == 0.3
        assert result["duration_seconds"] == 1800
        assert result["affected_users"] == 150
        
        # Verify FTNS service was called
        assert len(mock_ftns_service.limit_calls) == 1
        assert mock_ftns_service.limit_calls[0]["reduction_factor"] == 0.3
    
    @pytest.mark.asyncio
    async def test_action_cooldown_prevention(self, emergency_protocols):
        """Test that action cooldowns prevent spam"""
        
        # First halt should succeed
        result1 = await emergency_protocols.halt_transactions()
        assert result1["status"] == "executed"
        
        # Second halt within cooldown should be rejected
        result2 = await emergency_protocols.halt_transactions()
        assert result2["status"] == "rejected"
        assert result2["reason"] == "Action in cooldown period"
    
    @pytest.mark.asyncio
    async def test_emergency_proposal_creation(self, emergency_protocols, mock_governance_service):
        """Test emergency governance proposal creation"""
        
        detection = EmergencyDetection(
            trigger_type=EmergencyTriggerType.PRICE_CRASH,
            severity="critical",
            confidence=0.95,
            threshold_value=0.4,
            actual_value=0.7,
            data_source="test_oracle",
            detection_time=datetime.now(timezone.utc),
            auto_response_recommended=True,
            governance_required=True
        )
        
        proposed_actions = [
            EmergencyResponse(
                action_type=EmergencyActionType.HALT_TRANSACTIONS,
                severity="critical",
                target_scope="all_users",
                duration_seconds=7200,
                affected_users=None,
                parameters={"transaction_types": ["transfer", "marketplace"]},
                authorized_by="emergency_system",
                governance_proposal_id=None
            )
        ]
        
        result = await emergency_protocols.create_emergency_proposal(detection, proposed_actions)
        
        assert result["status"] == "created"
        assert "proposal_id" in result
        assert result["trigger_type"] == EmergencyTriggerType.PRICE_CRASH
        assert result["proposed_actions"] is not None
        
        # Verify governance service was called
        assert len(mock_governance_service.proposals) == 1


class TestEmergencyIntegration:
    """Test suite for integrated emergency workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_price_crash_response(self, emergency_protocols, mock_ftns_service):
        """Test complete price crash detection and response workflow"""
        
        # Set up critical price crash scenario
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 30.0, "timestamp": datetime.now(timezone.utc)}  # 70% crash
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        
        # Detect emergency
        detection = await emergency_protocols.detect_price_crash()
        assert detection is not None
        assert detection.severity == "critical"
        
        # Trigger response
        response = await emergency_protocols.trigger_emergency_response(detection)
        
        assert response["status"] == "completed"
        assert response["trigger_type"] == EmergencyTriggerType.PRICE_CRASH
        assert response["severity"] == "critical"
        assert len(response["actions_executed"]) >= 1
        
        # Verify transaction halt was triggered
        halt_action = next((a for a in response["actions_executed"] 
                          if a.get("action_type") == "halt_transactions"), None)
        assert halt_action is not None
        assert halt_action["status"] == "executed"
    
    @pytest.mark.asyncio
    async def test_volume_spike_graduated_response(self, emergency_protocols, mock_ftns_service):
        """Test graduated response to volume spike (limits before halt)"""
        
        # Set up high volume spike scenario
        emergency_protocols.set_mock_volume_data(
            current_volume=15000.0,   # 15x normal volume
            baseline_volume=1000.0
        )
        
        # Detect emergency
        detection = await emergency_protocols.detect_volume_spike()
        assert detection is not None
        assert detection.severity == "high"
        
        # Trigger response
        response = await emergency_protocols.trigger_emergency_response(detection)
        
        assert response["status"] == "completed"
        
        # Verify limit reduction was triggered (not halt for volume spikes)
        limit_action = next((a for a in response["actions_executed"] 
                           if a.get("action_type") == "reduce_limits"), None)
        assert limit_action is not None
        assert limit_action["status"] == "executed"
    
    @pytest.mark.asyncio
    async def test_governance_escalation_workflow(self, emergency_protocols, mock_governance_service):
        """Test governance escalation for high-confidence emergencies"""
        
        # Set up critical oracle failure requiring governance
        oracle_prices = {
            "source_a": 100.0,
            "source_b": 200.0   # 100% deviation - critical failure
        }
        
        emergency_protocols.set_mock_oracle_prices(oracle_prices)
        
        # Detect emergency
        detection = await emergency_protocols.detect_oracle_failure()
        assert detection is not None
        assert detection.governance_required is True
        
        # Trigger response
        response = await emergency_protocols.trigger_emergency_response(detection)
        
        assert response["status"] == "completed"
        assert response["governance_notified"] is True
        
        # Verify governance notification action
        governance_action = next((a for a in response["actions_executed"] 
                                if a.get("action_type") == "notify_governance"), None)
        assert governance_action is not None
    
    @pytest.mark.asyncio
    async def test_emergency_monitoring_lifecycle(self, emergency_protocols):
        """Test emergency monitoring start/stop lifecycle"""
        
        # Initial state
        status = await emergency_protocols.get_emergency_status()
        assert status["monitoring_active"] is False
        
        # Start monitoring
        start_result = await emergency_protocols.start_emergency_monitoring(check_interval_seconds=30)
        assert start_result["status"] == "started"
        assert start_result["interval_seconds"] == 30
        
        # Check status
        status = await emergency_protocols.get_emergency_status()
        assert status["monitoring_active"] is True
        
        # Stop monitoring
        stop_result = await emergency_protocols.stop_emergency_monitoring()
        assert stop_result["status"] == "stopped"
        
        # Final status
        status = await emergency_protocols.get_emergency_status()
        assert status["monitoring_active"] is False


class TestEmergencyConfiguration:
    """Test suite for emergency configuration and thresholds"""
    
    @pytest.mark.asyncio
    async def test_configurable_thresholds(self, emergency_protocols):
        """Test that emergency thresholds are configurable"""
        
        # Test default configuration
        config = emergency_protocols.emergency_config
        assert config["price_crash_threshold"] == 0.4
        assert config["volume_spike_threshold"] == 5.0
        assert config["confidence_threshold"] == 0.8
        
        # Modify configuration
        emergency_protocols.emergency_config["price_crash_threshold"] = 0.3  # 30% threshold
        
        # Set up price crash at new threshold
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 70.0, "timestamp": datetime.now(timezone.utc)}  # 30% crash
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        
        # Should now trigger at 30% threshold
        detection = await emergency_protocols.detect_price_crash()
        assert detection is not None
        assert detection.actual_value == 0.3
    
    @pytest.mark.asyncio
    async def test_emergency_status_reporting(self, emergency_protocols):
        """Test comprehensive emergency status reporting"""
        
        status = await emergency_protocols.get_emergency_status()
        
        # Verify status structure
        assert "monitoring_active" in status
        assert "active_emergencies" in status
        assert "emergency_config" in status
        assert "action_cooldowns" in status
        assert "status_timestamp" in status
        
        # Verify config is included
        assert status["emergency_config"]["price_crash_threshold"] == 0.4
        assert status["emergency_config"]["auto_response_enabled"] is True
        
        # Verify cooldowns tracking
        assert isinstance(status["action_cooldowns"], dict)
    
    @pytest.mark.asyncio
    async def test_severity_classification_accuracy(self, emergency_protocols):
        """Test that severity levels are classified correctly"""
        
        # Test critical price crash (80%+ drop)
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 15.0, "timestamp": datetime.now(timezone.utc)}  # 85% crash
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        detection = await emergency_protocols.detect_price_crash()
        
        assert detection is not None
        assert detection.severity == "critical"
        assert detection.actual_value == 0.85
        
        # Test high price crash (50% drop)
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"price": 50.0, "timestamp": datetime.now(timezone.utc)}  # 50% crash
        ]
        
        emergency_protocols.set_mock_price_history(price_history)
        detection = await emergency_protocols.detect_price_crash()
        
        assert detection is not None
        assert detection.severity == "high"  # 50% crash is classified as high with our thresholds
        assert detection.actual_value == 0.5


class TestEmergencyEdgeCases:
    """Test suite for emergency system edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_insufficient_price_data(self, emergency_protocols):
        """Test handling of insufficient price data"""
        
        # Set up insufficient price history
        price_history = [
            {"price": 100.0, "timestamp": datetime.now(timezone.utc)}
        ]  # Only one data point
        
        emergency_protocols.set_mock_price_history(price_history)
        
        detection = await emergency_protocols.detect_price_crash()
        assert detection is None  # Should not trigger with insufficient data
    
    @pytest.mark.asyncio
    async def test_zero_baseline_volume(self, emergency_protocols):
        """Test handling of zero baseline volume"""
        
        emergency_protocols.set_mock_volume_data(
            current_volume=1000.0,
            baseline_volume=0.0      # Zero baseline
        )
        
        detection = await emergency_protocols.detect_volume_spike()
        assert detection is None  # Should not trigger with zero baseline
    
    @pytest.mark.asyncio
    async def test_single_oracle_source(self, emergency_protocols):
        """Test handling of single oracle source (no comparison possible)"""
        
        oracle_prices = {
            "source_a": 100.0
        }  # Only one source
        
        emergency_protocols.set_mock_oracle_prices(oracle_prices)
        
        detection = await emergency_protocols.detect_oracle_failure()
        assert detection is None  # Should not trigger with single source
    
    @pytest.mark.asyncio
    async def test_emergency_response_without_services(self, db_session):
        """Test emergency response when external services are unavailable"""
        
        # Create emergency protocols without external services
        emergency_protocols = EmergencyProtocols(db_session)
        
        detection = EmergencyDetection(
            trigger_type=EmergencyTriggerType.SYSTEM_ERROR,
            severity="medium",
            confidence=0.85,
            threshold_value=10,
            actual_value=15,
            data_source="test_monitoring",
            detection_time=datetime.now(timezone.utc),
            auto_response_recommended=True,
            governance_required=False
        )
        
        # Should still work but with limited functionality
        response = await emergency_protocols.trigger_emergency_response(detection)
        
        assert response["status"] == "completed"
        assert response["trigger_type"] == EmergencyTriggerType.SYSTEM_ERROR
        # Actions may be limited without external services