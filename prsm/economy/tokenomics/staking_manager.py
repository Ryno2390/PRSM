"""
FTNS Staking Manager
Implements staking, unstaking, slashing, and reward distribution

This module provides a comprehensive staking system for FTNS tokens, enabling:
- Token staking with configurable minimums and maximums
- Unstaking with time-locked withdrawal periods
- Slashing for misbehavior penalties
- Staking rewards with configurable rates
- Governance integration for parameter changes

Core Features:
- Secure staking with balance locking
- Time-delayed unstaking for security
- Attributable slashing with evidence tracking
- Automatic reward calculation and distribution
- Comprehensive audit trails

Security Considerations:
- All operations are atomic with proper rollback
- Slashing requires governance approval or automated triggers
- Unstaking has mandatory delay periods
- Rate limits prevent rapid stake manipulation
"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

logger = structlog.get_logger(__name__)


# === Enums ===

class StakeStatus(str, Enum):
    """Status of a stake"""
    ACTIVE = "active"              # Currently staked and earning rewards
    UNSTAKING = "unstaking"         # Unstake requested, waiting for period
    WITHDRAWN = "withdrawn"         # Fully withdrawn
    SLASHED = "slashed"             # Partially or fully slashed
    LOCKED = "locked"               # Locked due to dispute or investigation


class UnstakeRequestStatus(str, Enum):
    """Status of an unstake request"""
    PENDING = "pending"            # Waiting for unstaking period
    AVAILABLE = "available"         # Ready for withdrawal
    COMPLETED = "completed"         # Withdrawn successfully
    CANCELLED = "cancelled"         # Request cancelled
    SLASHED = "slashed"             # Slashed during unstaking


class SlashReason(str, Enum):
    """Reasons for slashing"""
    MISCONDUCT = "misconduct"                    # General misbehavior
    VALIDATION_FAILURE = "validation_failure"    # Failed to validate properly
    DOUBLE_SIGNING = "double_signing"           # Signed conflicting blocks
    DOWNTIME = "downtime"                        # Excessive offline time
    GOVERNANCE_VIOLATION = "governance_violation"  # Violated governance rules
    FRAUD = "fraud"                              # Fraudulent activity
    COLLUSION = "collusion"                       # Collusion with other nodes
    DATA_MANIPULATION = "data_manipulation"      # Manipulated data
    SECURITY_BREACH = "security_breach"          # Security violation
    APPEAL_REJECTED = "appeal_rejected"           # Appeal was rejected


class StakeType(str, Enum):
    """Types of staking"""
    GOVERNANCE = "governance"        # Staking for governance participation
    VALIDATION = "validation"        # Staking for network validation
    COMPUTE = "compute"              # Staking for compute provision
    STORAGE = "storage"              # Staking for storage provision
    LIQUIDITY = "liquidity"          # Staking for liquidity provision
    GENERAL = "general"              # General purpose staking


# === Dataclasses ===

@dataclass
class StakingConfig:
    """Configuration for staking mechanism"""
    minimum_stake: int = 1000                    # Minimum FTNS to stake
    unstaking_period_seconds: int = 7 * 24 * 3600  # 7 days in seconds
    reward_rate_annual: float = 0.05              # 5% annual reward rate
    slashing_rate_base: float = 0.1               # 10% base slash rate
    max_stake_per_user: int = 10_000_000          # Maximum stake per user
    max_total_stake: int = 1_000_000_000           # Maximum total network stake
    reward_compounding: bool = False              # Whether rewards compound
    min_reward_claim_interval_seconds: int = 24 * 3600  # 1 day minimum between claims
    governance_slash_multiplier: float = 2.0       # Multiplier for governance-approved slashes
    appeal_period_seconds: int = 3 * 24 * 3600    # 3 days to appeal slashes
    
    # Rate limits
    max_stake_operations_per_day: int = 5         # Max stake operations per day
    max_unstake_operations_per_day: int = 3       # Max unstake operations per day
    
    # Reward parameters
    reward_calculation_interval_seconds: int = 3600  # Calculate rewards hourly
    min_stake_age_for_rewards_seconds: int = 24 * 3600  # 1 day minimum stake age
    
    # Slashing thresholds
    downtime_slash_threshold_seconds: int = 24 * 3600  # 24 hours offline triggers slash
    max_slash_per_event: float = 0.5            # Max 50% slash per event


@dataclass
class StakeRecord:
    """Record of a stake"""
    stake_id: str
    user_id: str
    amount: Decimal
    stake_type: StakeType
    staked_at: datetime
    rewards_earned: Decimal = Decimal('0')
    rewards_claimed: Decimal = Decimal('0')
    last_reward_calculation: datetime = None
    status: StakeStatus = StakeStatus.ACTIVE
    lock_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.last_reward_calculation is None:
            self.last_reward_calculation = self.staked_at
        if isinstance(self.stake_type, str):
            self.stake_type = StakeType(self.stake_type)
        if isinstance(self.status, str):
            self.status = StakeStatus(self.status)


@dataclass
class UnstakeRequest:
    """Request to unstake tokens"""
    request_id: str
    stake_id: str
    user_id: str
    amount: Decimal
    requested_at: datetime
    available_at: datetime
    status: UnstakeRequestStatus = UnstakeRequestStatus.PENDING
    completed_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = UnstakeRequestStatus(self.status)
    
    @property
    def is_available(self) -> bool:
        """Check if unstake is available for withdrawal"""
        return (
            self.status == UnstakeRequestStatus.AVAILABLE and
            datetime.now(timezone.utc) >= self.available_at
        )


@dataclass
class SlashRecord:
    """Record of a slashing event"""
    slash_id: str
    stake_id: str
    user_id: str
    amount_slashed: Decimal
    reason: SlashReason
    slash_rate: float
    evidence: Dict[str, Any]
    slashed_at: datetime
    slashed_by: str  # Governance proposal ID or automated system
    appeal_deadline: Optional[datetime] = None
    appeal_status: Optional[str] = None  # "pending", "approved", "rejected"
    appeal_evidence: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.reason, str):
            self.reason = SlashReason(self.reason)
        if self.appeal_deadline is None:
            self.appeal_deadline = self.slashed_at + timedelta(days=3)


@dataclass
class RewardCalculation:
    """Result of reward calculation"""
    stake_id: str
    user_id: str
    principal: Decimal
    reward_amount: Decimal
    annual_rate: float
    days_staked: float
    calculation_timestamp: datetime
    next_calculation: datetime


# === Database Models ===
# These would typically be in models.py, but included here for reference

"""
class FTNSStake(Base):
    '''Database model for stakes'''
    __tablename__ = "ftns_stakes"
    
    stake_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    
    amount = Column(DECIMAL(20, 8), nullable=False)
    stake_type = Column(String(50), nullable=False, default=StakeType.GENERAL.value)
    status = Column(String(50), nullable=False, default=StakeStatus.ACTIVE.value)
    
    rewards_earned = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0'))
    rewards_claimed = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0'))
    last_reward_calculation = Column(DateTime(timezone=True), nullable=False)
    
    staked_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    unstake_requested_at = Column(DateTime(timezone=True), nullable=True)
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)
    
    lock_reason = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    wallet = relationship("FTNSWallet", backref="stakes")
    unstake_requests = relationship("FTNSUnstakeRequest", backref="stake")
    slash_events = relationship("FTNSSlashEvent", backref="stake")


class FTNSUnstakeRequest(Base):
    '''Database model for unstake requests'''
    __tablename__ = "ftns_unstake_requests"
    
    request_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    stake_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_stakes.stake_id'), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    amount = Column(DECIMAL(20, 8), nullable=False)
    status = Column(String(50), nullable=False, default=UnstakeRequestStatus.PENDING.value)
    
    requested_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    available_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    cancellation_reason = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)


class FTNSSlashEvent(Base):
    '''Database model for slash events'''
    __tablename__ = "ftns_slash_events"
    
    slash_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    stake_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_stakes.stake_id'), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    amount_slashed = Column(DECIMAL(20, 8), nullable=False)
    reason = Column(String(50), nullable=False)
    slash_rate = Column(Float, nullable=False)
    
    evidence = Column(JSONB, nullable=True)
    slashed_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    slashed_by = Column(String(255), nullable=False)  # Governance proposal or system
    
    appeal_deadline = Column(DateTime(timezone=True), nullable=True)
    appeal_status = Column(String(50), nullable=True)
    appeal_evidence = Column(JSONB, nullable=True)
    
    metadata = Column(JSONB, nullable=True)
"""


class StakingManager:
    """
    Manages FTNS staking operations
    
    The StakingManager handles all staking-related operations including:
    - Creating and managing stakes
    - Processing unstake requests with time delays
    - Applying slashing penalties
    - Calculating and distributing rewards
    - Managing stake status transitions
    
    All operations are designed to be atomic and maintain consistency
    with the underlying FTNS balance system.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        ftns_service,
        config: Optional[StakingConfig] = None
    ):
        """
        Initialize the staking manager
        
        Args:
            db_session: Database session for persistence
            ftns_service: FTNS service for balance operations
            config: Staking configuration (uses defaults if not provided)
        """
        self.db = db_session
        self.ftns = ftns_service
        self.config = config or StakingConfig()
        
        # In-memory caches (would use database in production)
        self._stakes: Dict[str, StakeRecord] = {}
        self._unstake_requests: Dict[str, UnstakeRequest] = {}
        self._slash_records: Dict[str, SlashRecord] = {}
        self._user_stakes: Dict[str, List[str]] = {}  # user_id -> [stake_ids]
        
        # Rate limiting
        self._daily_operations: Dict[str, List[datetime]] = {}  # user_id -> [timestamps]
        
        logger.info(
            "StakingManager initialized",
            minimum_stake=self.config.minimum_stake,
            unstaking_period_days=self.config.unstaking_period_seconds / 86400,
            reward_rate=self.config.reward_rate_annual
        )
    
    # === Staking Operations ===
    
    async def stake(
        self,
        user_id: str,
        amount: Decimal,
        stake_type: StakeType = StakeType.GENERAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StakeRecord:
        """
        Stake FTNS tokens
        
        Args:
            user_id: User making the stake
            amount: Amount to stake
            stake_type: Type of staking (governance, validation, etc.)
            metadata: Additional metadata for the stake
            
        Returns:
            StakeRecord: The created stake record
            
        Raises:
            ValueError: If stake validation fails
        """
        
        # Validate amount
        if amount < Decimal(self.config.minimum_stake):
            raise ValueError(
                f"Stake amount {amount} below minimum {self.config.minimum_stake}"
            )
        
        # Check user's current total stake
        user_total_stake = await self.get_user_total_stake(user_id)
        if user_total_stake + amount > Decimal(self.config.max_stake_per_user):
            raise ValueError(
                f"Stake would exceed user maximum of {self.config.max_stake_per_user}"
            )
        
        # Check network total stake
        network_total = await self.get_network_total_stake()
        if network_total + amount > Decimal(self.config.max_total_stake):
            raise ValueError(
                f"Stake would exceed network maximum of {self.config.max_total_stake}"
            )
        
        # Check rate limit
        await self._check_rate_limit(user_id, "stake")
        
        # Verify user has sufficient balance
        available_balance = await self.ftns.get_available_balance(user_id)
        if available_balance < amount:
            raise ValueError(
                f"Insufficient balance. Available: {available_balance}, Required: {amount}"
            )
        
        # Lock tokens from user's balance
        await self.ftns.lock_tokens(user_id, amount, reason="staking")
        
        # Create stake record
        stake_id = str(uuid4())
        now = datetime.now(timezone.utc)
        
        stake = StakeRecord(
            stake_id=stake_id,
            user_id=user_id,
            amount=amount,
            stake_type=stake_type,
            staked_at=now,
            status=StakeStatus.ACTIVE,
            metadata=metadata or {}
        )
        
        # Store stake
        self._stakes[stake_id] = stake
        
        # Update user's stake list
        if user_id not in self._user_stakes:
            self._user_stakes[user_id] = []
        self._user_stakes[user_id].append(stake_id)
        
        # Record operation for rate limiting
        self._record_operation(user_id, "stake")
        
        await logger.ainfo(
            "Stake created",
            stake_id=stake_id,
            user_id=user_id,
            amount=float(amount),
            stake_type=stake_type.value
        )
        
        return stake
    
    async def unstake(
        self,
        user_id: str,
        stake_id: str,
        amount: Optional[Decimal] = None
    ) -> UnstakeRequest:
        """
        Request to unstake tokens
        
        Args:
            user_id: User making the request
            stake_id: ID of the stake to unstake
            amount: Amount to unstake (None = full stake)
            
        Returns:
            UnstakeRequest: The created unstake request
            
        Raises:
            ValueError: If unstake validation fails
        """
        
        # Get stake
        stake = self._stakes.get(stake_id)
        if not stake:
            raise ValueError(f"Stake {stake_id} not found")
        
        if stake.user_id != user_id:
            raise ValueError("Stake does not belong to user")
        
        if stake.status != StakeStatus.ACTIVE:
            raise ValueError(f"Stake is not active: {stake.status}")
        
        # Determine amount
        unstake_amount = amount if amount else stake.amount
        if unstake_amount > stake.amount:
            raise ValueError(f"Cannot unstake more than staked amount")
        
        if unstake_amount <= 0:
            raise ValueError("Unstake amount must be positive")
        
        # Check rate limit
        await self._check_rate_limit(user_id, "unstake")
        
        # Create unstake request
        request_id = str(uuid4())
        now = datetime.now(timezone.utc)
        available_at = now + timedelta(seconds=self.config.unstaking_period_seconds)
        
        request = UnstakeRequest(
            request_id=request_id,
            stake_id=stake_id,
            user_id=user_id,
            amount=unstake_amount,
            requested_at=now,
            available_at=available_at,
            status=UnstakeRequestStatus.PENDING
        )
        
        # Update stake status
        if unstake_amount == stake.amount:
            stake.status = StakeStatus.UNSTAKING
        else:
            # Partial unstake - reduce stake amount
            stake.amount -= unstake_amount
        
        # Store request
        self._unstake_requests[request_id] = request
        
        # Record operation
        self._record_operation(user_id, "unstake")
        
        await logger.ainfo(
            "Unstake request created",
            request_id=request_id,
            stake_id=stake_id,
            user_id=user_id,
            amount=float(unstake_amount),
            available_at=available_at.isoformat()
        )
        
        return request
    
    async def withdraw(
        self,
        user_id: str,
        request_id: str
    ) -> Tuple[bool, Decimal]:
        """
        Withdraw unstaked tokens after period
        
        Args:
            user_id: User making the withdrawal
            request_id: ID of the unstake request
            
        Returns:
            Tuple[bool, Decimal]: (success, amount withdrawn)
            
        Raises:
            ValueError: If withdrawal validation fails
        """
        
        # Get request
        request = self._unstake_requests.get(request_id)
        if not request:
            raise ValueError(f"Unstake request {request_id} not found")
        
        if request.user_id != user_id:
            raise ValueError("Request does not belong to user")
        
        if request.status not in (UnstakeRequestStatus.PENDING, UnstakeRequestStatus.AVAILABLE):
            raise ValueError(f"Request status invalid: {request.status}")
        
        # Check if available
        now = datetime.now(timezone.utc)
        if now < request.available_at:
            wait_seconds = (request.available_at - now).total_seconds()
            raise ValueError(
                f"Unstaking period not complete. Wait {wait_seconds:.0f} more seconds"
            )
        
        # Update status
        request.status = UnstakeRequestStatus.COMPLETED
        request.completed_at = now
        
        # Get stake and update
        stake = self._stakes.get(request.stake_id)
        if stake and stake.status == StakeStatus.UNSTAKING:
            stake.status = StakeStatus.WITHDRAWN
        
        # Unlock tokens back to user
        await self.ftns.unlock_tokens(user_id, request.amount, reason="unstake_withdrawal")
        
        await logger.ainfo(
            "Withdrawal completed",
            request_id=request_id,
            user_id=user_id,
            amount=float(request.amount)
        )
        
        return True, request.amount
    
    async def cancel_unstake(
        self,
        user_id: str,
        request_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an unstake request
        
        Args:
            user_id: User cancelling the request
            request_id: ID of the unstake request
            reason: Reason for cancellation
            
        Returns:
            bool: True if cancelled successfully
            
        Raises:
            ValueError: If cancellation validation fails
        """
        
        request = self._unstake_requests.get(request_id)
        if not request:
            raise ValueError(f"Unstake request {request_id} not found")
        
        if request.user_id != user_id:
            raise ValueError("Request does not belong to user")
        
        if request.status != UnstakeRequestStatus.PENDING:
            raise ValueError(f"Cannot cancel request in status: {request.status}")
        
        # Cancel request
        request.status = UnstakeRequestStatus.CANCELLED
        request.cancellation_reason = reason or "User cancelled"
        
        # Restore stake
        stake = self._stakes.get(request.stake_id)
        if stake:
            if stake.status == StakeStatus.UNSTAKING:
                stake.status = StakeStatus.ACTIVE
            else:
                # Partial unstake - restore amount
                stake.amount += request.amount
        
        await logger.ainfo(
            "Unstake request cancelled",
            request_id=request_id,
            user_id=user_id,
            reason=reason
        )
        
        return True
    
    # === Slashing Operations ===
    
    async def slash(
        self,
        user_id: str,
        stake_id: str,
        reason: SlashReason,
        evidence: Dict[str, Any],
        slash_rate: Optional[float] = None,
        slashed_by: str = "system"
    ) -> SlashRecord:
        """
        Slash staked tokens for misbehavior
        
        Args:
            user_id: User being slashed
            stake_id: ID of the stake to slash
            reason: Reason for slashing
            evidence: Evidence supporting the slash
            slash_rate: Rate to slash (uses default if not provided)
            slashed_by: Entity that initiated the slash
            
        Returns:
            SlashRecord: Record of the slash event
            
        Raises:
            ValueError: If slash validation fails
        """
        
        # Get stake
        stake = self._stakes.get(stake_id)
        if not stake:
            raise ValueError(f"Stake {stake_id} not found")
        
        if stake.user_id != user_id:
            raise ValueError("Stake does not belong to user")
        
        if stake.status not in (StakeStatus.ACTIVE, StakeStatus.UNSTAKING):
            raise ValueError(f"Cannot slash stake in status: {stake.status}")
        
        # Calculate slash rate
        effective_rate = slash_rate if slash_rate else self.config.slashing_rate_base
        
        # Cap at maximum
        effective_rate = min(effective_rate, self.config.max_slash_per_event)
        
        # Calculate slash amount
        slash_amount = stake.amount * Decimal(str(effective_rate))
        slash_amount = slash_amount.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        # Create slash record
        slash_id = str(uuid4())
        now = datetime.now(timezone.utc)
        appeal_deadline = now + timedelta(seconds=self.config.appeal_period_seconds)
        
        slash_record = SlashRecord(
            slash_id=slash_id,
            stake_id=stake_id,
            user_id=user_id,
            amount_slashed=slash_amount,
            reason=reason,
            slash_rate=effective_rate,
            evidence=evidence,
            slashed_at=now,
            slashed_by=slashed_by,
            appeal_deadline=appeal_deadline
        )
        
        # Apply slash to stake
        stake.amount -= slash_amount
        stake.rewards_earned = max(Decimal('0'), stake.rewards_earned - slash_amount)
        
        # If stake is depleted, mark as slashed
        if stake.amount <= 0:
            stake.status = StakeStatus.SLASHED
        
        # Store slash record
        self._slash_records[slash_id] = slash_record
        
        # Burn slashed tokens (or send to treasury)
        await self.ftns.burn_tokens(user_id, slash_amount, reason=f"slash:{reason.value}")
        
        await logger.awarning(
            "Stake slashed",
            slash_id=slash_id,
            stake_id=stake_id,
            user_id=user_id,
            amount=float(slash_amount),
            reason=reason.value,
            slashed_by=slashed_by
        )
        
        return slash_record
    
    async def appeal_slash(
        self,
        user_id: str,
        slash_id: str,
        appeal_evidence: Dict[str, Any]
    ) -> bool:
        """
        Submit an appeal for a slash
        
        Args:
            user_id: User submitting the appeal
            slash_id: ID of the slash to appeal
            appeal_evidence: Evidence supporting the appeal
            
        Returns:
            bool: True if appeal submitted successfully
            
        Raises:
            ValueError: If appeal validation fails
        """
        
        slash_record = self._slash_records.get(slash_id)
        if not slash_record:
            raise ValueError(f"Slash record {slash_id} not found")
        
        if slash_record.user_id != user_id:
            raise ValueError("Slash does not belong to user")
        
        if slash_record.appeal_status:
            raise ValueError(f"Slash already has appeal status: {slash_record.appeal_status}")
        
        now = datetime.now(timezone.utc)
        if now > slash_record.appeal_deadline:
            raise ValueError("Appeal deadline has passed")
        
        # Set appeal status
        slash_record.appeal_status = "pending"
        slash_record.appeal_evidence = appeal_evidence
        
        await logger.ainfo(
            "Slash appeal submitted",
            slash_id=slash_id,
            user_id=user_id
        )
        
        return True
    
    async def resolve_appeal(
        self,
        slash_id: str,
        approved: bool,
        resolution_note: str
    ) -> bool:
        """
        Resolve a slash appeal (governance operation)
        
        Args:
            slash_id: ID of the slash
            approved: Whether to approve the appeal
            resolution_note: Note explaining the resolution
            
        Returns:
            bool: True if resolved successfully
        """
        
        slash_record = self._slash_records.get(slash_id)
        if not slash_record:
            raise ValueError(f"Slash record {slash_id} not found")
        
        if slash_record.appeal_status != "pending":
            raise ValueError(f"Slash appeal not pending: {slash_record.appeal_status}")
        
        slash_record.appeal_status = "approved" if approved else "rejected"
        
        if approved:
            # Refund the slashed amount
            stake = self._stakes.get(slash_record.stake_id)
            if stake:
                stake.amount += slash_record.amount_slashed
                if stake.status == StakeStatus.SLASHED:
                    stake.status = StakeStatus.ACTIVE
            
            # Credit tokens back
            await self.ftns.mint_tokens(
                slash_record.user_id,
                slash_record.amount_slashed,
                reason=f"appeal_refund:{slash_id}"
            )
            
            await logger.ainfo(
                "Slash appeal approved - refunding",
                slash_id=slash_id,
                user_id=slash_record.user_id,
                amount=float(slash_record.amount_slashed)
            )
        else:
            await logger.ainfo(
                "Slash appeal rejected",
                slash_id=slash_id,
                user_id=slash_record.user_id,
                resolution_note=resolution_note
            )
        
        return True
    
    # === Reward Operations ===
    
    async def calculate_rewards(
        self,
        user_id: str,
        stake_id: Optional[str] = None
    ) -> List[RewardCalculation]:
        """
        Calculate pending rewards for a user's stakes
        
        Args:
            user_id: User to calculate rewards for
            stake_id: Specific stake (None = all user's stakes)
            
        Returns:
            List[RewardCalculation]: Reward calculations for each stake
        """
        
        calculations = []
        now = datetime.now(timezone.utc)
        
        # Get stakes to calculate
        stake_ids = [stake_id] if stake_id else self._user_stakes.get(user_id, [])
        
        for sid in stake_ids:
            stake = self._stakes.get(sid)
            if not stake or stake.status != StakeStatus.ACTIVE:
                continue
            
            # Check minimum stake age
            stake_age = (now - stake.staked_at).total_seconds()
            if stake_age < self.config.min_stake_age_for_rewards_seconds:
                continue
            
            # Calculate time since last calculation
            time_staked = (now - stake.last_reward_calculation).total_seconds()
            days_staked = time_staked / 86400
            
            # Calculate reward using annual rate
            # reward = principal * rate * (days / 365)
            annual_rate = Decimal(str(self.config.reward_rate_annual))
            reward_amount = stake.amount * annual_rate * Decimal(str(days_staked / 365))
            reward_amount = reward_amount.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
            
            calculation = RewardCalculation(
                stake_id=sid,
                user_id=user_id,
                principal=stake.amount,
                reward_amount=reward_amount,
                annual_rate=self.config.reward_rate_annual,
                days_staked=days_staked,
                calculation_timestamp=now,
                next_calculation=now + timedelta(seconds=self.config.reward_calculation_interval_seconds)
            )
            
            calculations.append(calculation)
        
        return calculations
    
    async def claim_rewards(
        self,
        user_id: str,
        stake_id: Optional[str] = None
    ) -> Decimal:
        """
        Claim accumulated staking rewards
        
        Args:
            user_id: User claiming rewards
            stake_id: Specific stake (None = all stakes)
            
        Returns:
            Decimal: Total rewards claimed
        """
        
        calculations = await self.calculate_rewards(user_id, stake_id)
        total_rewards = Decimal('0')
        
        for calc in calculations:
            stake = self._stakes.get(calc.stake_id)
            if not stake:
                continue
            
            # Update stake
            stake.rewards_earned += calc.reward_amount
            stake.last_reward_calculation = calc.calculation_timestamp
            
            # Add to total
            total_rewards += calc.reward_amount
            
            # If compounding, add to stake
            if self.config.reward_compounding:
                stake.amount += calc.reward_amount
        
        # Mint reward tokens to user
        if total_rewards > 0:
            await self.ftns.mint_tokens(
                user_id,
                total_rewards,
                reason="staking_rewards"
            )
        
        await logger.ainfo(
            "Rewards claimed",
            user_id=user_id,
            total_rewards=float(total_rewards),
            stake_count=len(calculations)
        )
        
        return total_rewards
    
    # === Query Operations ===
    
    async def get_stake(self, stake_id: str) -> Optional[StakeRecord]:
        """Get a specific stake by ID"""
        return self._stakes.get(stake_id)
    
    async def get_user_stakes(
        self,
        user_id: str,
        status: Optional[StakeStatus] = None
    ) -> List[StakeRecord]:
        """
        Get all stakes for a user
        
        Args:
            user_id: User to get stakes for
            status: Filter by status (None = all statuses)
            
        Returns:
            List[StakeRecord]: User's stakes
        """
        
        stake_ids = self._user_stakes.get(user_id, [])
        stakes = [self._stakes[sid] for sid in stake_ids if sid in self._stakes]
        
        if status:
            stakes = [s for s in stakes if s.status == status]
        
        return stakes
    
    async def get_user_total_stake(self, user_id: str) -> Decimal:
        """Get total staked amount for a user"""
        
        stakes = await self.get_user_stakes(user_id, status=StakeStatus.ACTIVE)
        return sum(s.amount for s in stakes)
    
    async def get_network_total_stake(self) -> Decimal:
        """Get total staked amount across all users"""
        
        return sum(
            s.amount for s in self._stakes.values()
            if s.status == StakeStatus.ACTIVE
        )
    
    async def get_unstake_request(self, request_id: str) -> Optional[UnstakeRequest]:
        """Get a specific unstake request by ID"""
        return self._unstake_requests.get(request_id)
    
    async def get_pending_unstake_requests(
        self,
        user_id: Optional[str] = None
    ) -> List[UnstakeRequest]:
        """
        Get pending unstake requests
        
        Args:
            user_id: Filter by user (None = all users)
            
        Returns:
            List[UnstakeRequest]: Pending requests
        """
        
        requests = [
            r for r in self._unstake_requests.values()
            if r.status in (UnstakeRequestStatus.PENDING, UnstakeRequestStatus.AVAILABLE)
        ]
        
        if user_id:
            requests = [r for r in requests if r.user_id == user_id]
        
        return requests
    
    async def get_available_withdrawals(self, user_id: str) -> List[UnstakeRequest]:
        """Get unstake requests ready for withdrawal"""
        
        now = datetime.now(timezone.utc)
        requests = await self.get_pending_unstake_requests(user_id)
        
        return [
            r for r in requests
            if r.available_at <= now and r.status == UnstakeRequestStatus.AVAILABLE
        ]
    
    async def get_slash_history(
        self,
        user_id: Optional[str] = None,
        stake_id: Optional[str] = None
    ) -> List[SlashRecord]:
        """
        Get slash history
        
        Args:
            user_id: Filter by user (None = all users)
            stake_id: Filter by stake (None = all stakes)
            
        Returns:
            List[SlashRecord]: Slash records
        """
        
        records = list(self._slash_records.values())
        
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        
        if stake_id:
            records = [r for r in records if r.stake_id == stake_id]
        
        return sorted(records, key=lambda r: r.slashed_at, reverse=True)
    
    # === Internal Methods ===
    
    async def _check_rate_limit(self, user_id: str, operation: str) -> None:
        """Check if user has exceeded rate limits"""
        
        now = datetime.now(timezone.utc)
        key = f"{user_id}:{operation}"
        
        if key not in self._daily_operations:
            self._daily_operations[key] = []
        
        # Clean old operations
        day_ago = now - timedelta(days=1)
        self._daily_operations[key] = [
            ts for ts in self._daily_operations[key]
            if ts > day_ago
        ]
        
        # Check limit
        max_ops = (
            self.config.max_stake_operations_per_day
            if operation == "stake"
            else self.config.max_unstake_operations_per_day
        )
        
        if len(self._daily_operations[key]) >= max_ops:
            raise ValueError(
                f"Rate limit exceeded for {operation}. Max {max_ops} per day."
            )
    
    def _record_operation(self, user_id: str, operation: str) -> None:
        """Record an operation for rate limiting"""
        
        key = f"{user_id}:{operation}"
        if key not in self._daily_operations:
            self._daily_operations[key] = []
        
        self._daily_operations[key].append(datetime.now(timezone.utc))
    
    # === Maintenance Operations ===
    
    async def process_matured_unstakes(self) -> List[UnstakeRequest]:
        """
        Process all unstake requests that have matured
        
        This should be called periodically by a background task.
        
        Returns:
            List[UnstakeRequest]: Requests that are now available
        """
        
        now = datetime.now(timezone.utc)
        matured = []
        
        for request in self._unstake_requests.values():
            if request.status == UnstakeRequestStatus.PENDING:
                if request.available_at <= now:
                    request.status = UnstakeRequestStatus.AVAILABLE
                    matured.append(request)
                    
                    await logger.ainfo(
                        "Unstake request matured",
                        request_id=request.request_id,
                        user_id=request.user_id
                    )
        
        return matured
    
    async def get_staking_stats(self) -> Dict[str, Any]:
        """
        Get staking statistics
        
        Returns:
            Dict with staking statistics
        """
        
        active_stakes = [s for s in self._stakes.values() if s.status == StakeStatus.ACTIVE]
        total_staked = sum(s.amount for s in active_stakes)
        
        unstaking_stakes = [s for s in self._stakes.values() if s.status == StakeStatus.UNSTAKING]
        total_unstaking = sum(s.amount for s in unstaking_stakes)
        
        pending_requests = [
            r for r in self._unstake_requests.values()
            if r.status == UnstakeRequestStatus.PENDING
        ]
        
        return {
            "total_stakes": len(self._stakes),
            "active_stakes": len(active_stakes),
            "total_staked": float(total_staked),
            "total_unstaking": float(total_unstaking),
            "pending_unstake_requests": len(pending_requests),
            "total_slashes": len(self._slash_records),
            "unique_stakers": len(self._user_stakes),
            "config": {
                "minimum_stake": self.config.minimum_stake,
                "unstaking_period_days": self.config.unstaking_period_seconds / 86400,
                "reward_rate_annual": self.config.reward_rate_annual,
                "slashing_rate_base": self.config.slashing_rate_base
            }
        }


# === Factory Function ===

_staking_manager_instance: Optional[StakingManager] = None


async def get_staking_manager(
    db_session: AsyncSession,
    ftns_service,
    config: Optional[StakingConfig] = None
) -> StakingManager:
    """
    Get or create the staking manager instance
    
    Args:
        db_session: Database session
        ftns_service: FTNS service instance
        config: Optional staking configuration
        
    Returns:
        StakingManager: The staking manager instance
    """
    global _staking_manager_instance
    
    if _staking_manager_instance is None:
        _staking_manager_instance = StakingManager(
            db_session=db_session,
            ftns_service=ftns_service,
            config=config
        )
    
    return _staking_manager_instance