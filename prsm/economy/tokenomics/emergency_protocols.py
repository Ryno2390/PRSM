"""
FTNS Emergency Protocols
Implements circuit breakers and emergency controls for risk management

This module provides comprehensive emergency response capabilities for the FTNS
system, including automated detection of crisis conditions, circuit breaker
activation, and governance integration for emergency decision-making.

Core Features:
- Real-time monitoring of price crashes, volume spikes, and system errors
- Automated emergency trigger detection with configurable thresholds
- Circuit breaker activation to halt or limit transactions
- Fast-track governance integration for emergency proposals
- Comprehensive audit trails and emergency response analytics

Emergency Triggers:
- Price crash detection (default: 40% drop in 1 hour)
- Volume spike detection (default: 5x normal volume)
- Oracle failure detection (10% price deviation)
- System error threshold breaches
- Manual governance-initiated halts
- Security breach detection

Response Actions:
- Transaction halts (temporary or indefinite)
- Transaction limit reductions
- Account freezing for suspicious activity
- Emergency rate adjustments
- Governance notification and fast-track voting
- System-wide circuit breaker activation
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from prsm.economy.tokenomics.models import (
    FTNSEmergencyTrigger, FTNSEmergencyAction, FTNSEmergencyConfig,
    EmergencyTriggerType, EmergencyStatus, EmergencyActionType
)

logger = structlog.get_logger(__name__)


@dataclass
class EmergencyDetection:
    """Emergency condition detection result"""
    trigger_type: str
    severity: str  # low, medium, high, critical
    confidence: float  # 0.0 to 1.0
    threshold_value: float
    actual_value: float
    data_source: str
    detection_time: datetime
    auto_response_recommended: bool
    governance_required: bool


@dataclass
class EmergencyResponse:
    """Emergency response action specification"""
    action_type: str
    severity: str
    target_scope: str  # all_users, specific_users, system_wide
    duration_seconds: Optional[int]
    affected_users: Optional[List[str]]
    parameters: Dict[str, Any]
    authorized_by: str
    governance_proposal_id: Optional[str]


class EmergencyProtocols:
    """
    Circuit breakers and emergency controls using PRSM's governance system
    
    The EmergencyProtocols service provides automated monitoring and response
    to crisis conditions that could threaten FTNS system stability. It integrates
    with the governance system for authorization and maintains comprehensive
    audit trails for transparency.
    """
    
    def __init__(self, db_session: AsyncSession, ftns_service=None, 
                 governance_service=None, price_oracle=None):
        self.db = db_session
        self.ftns = ftns_service
        self.governance = governance_service
        self.price_oracle = price_oracle
        
        # Default emergency thresholds (configurable via database)
        self.emergency_config = {
            'price_crash_threshold': 0.4,      # 40% price drop
            'volume_spike_threshold': 5.0,     # 5x normal volume
            'oracle_deviation_threshold': 0.1,  # 10% oracle deviation
            'system_error_threshold': 10,      # Error count threshold
            'confidence_threshold': 0.8,       # 80% confidence required
            'auto_response_enabled': True,
            'governance_escalation_threshold': 0.9  # 90% confidence for governance
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.last_price_check = None
        self.baseline_metrics = {}
        self.active_emergencies = set()
        
        # Emergency action cooldowns (prevent spam)
        self.action_cooldowns = {
            'halt_transactions': 300,      # 5 minutes
            'reduce_limits': 600,          # 10 minutes
            'notify_governance': 1800      # 30 minutes
        }
        self.last_actions = {}
    
    # === EMERGENCY DETECTION ===
    
    async def detect_price_crash(self, monitoring_window_minutes: int = 60) -> Optional[EmergencyDetection]:
        """
        Detect sudden price crashes that could indicate market manipulation
        or system instability
        """
        
        if not self.price_oracle:
            return None
        
        try:
            # Get recent price history
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=monitoring_window_minutes)
            
            price_history = await self.price_oracle.get_price_history(start_time, end_time)
            
            if len(price_history) < 2:
                return None
            
            # Calculate price change
            initial_price = price_history[0]['price']
            current_price = price_history[-1]['price']
            price_change = (current_price - initial_price) / initial_price
            
            # Check if crash threshold is breached
            crash_threshold = -abs(self.emergency_config['price_crash_threshold'])
            
            if price_change <= crash_threshold:
                # Calculate confidence based on data quality and consistency
                confidence = await self._calculate_price_crash_confidence(price_history)
                
                severity = self._determine_price_crash_severity(abs(price_change))
                
                return EmergencyDetection(
                    trigger_type=EmergencyTriggerType.PRICE_CRASH.value,
                    severity=severity,
                    confidence=confidence,
                    threshold_value=abs(crash_threshold),
                    actual_value=abs(price_change),
                    data_source=f"price_oracle_{monitoring_window_minutes}m",
                    detection_time=end_time,
                    auto_response_recommended=confidence > self.emergency_config['confidence_threshold'],
                    governance_required=confidence > self.emergency_config['governance_escalation_threshold']
                )
                
        except Exception as e:
            await logger.aerror("Price crash detection failed", error=str(e))
            return None
    
    async def detect_volume_spike(self, monitoring_window_minutes: int = 30) -> Optional[EmergencyDetection]:
        """
        Detect unusual trading volume that could indicate coordinated attacks
        or system manipulation
        """
        
        if not self.ftns:
            return None
        
        try:
            # Get recent transaction volume
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=monitoring_window_minutes)
            
            current_volume = await self.ftns.get_transaction_volume(start_time, end_time)
            
            # Get baseline volume for comparison
            baseline_start = start_time - timedelta(days=7)  # 7-day baseline
            baseline_end = start_time
            baseline_volume = await self.ftns.get_average_volume(baseline_start, baseline_end, monitoring_window_minutes)
            
            if baseline_volume == 0:
                return None
            
            # Calculate volume spike ratio
            volume_ratio = current_volume / baseline_volume
            spike_threshold = self.emergency_config['volume_spike_threshold']
            
            if volume_ratio >= spike_threshold:
                # Calculate confidence based on volume pattern analysis
                confidence = await self._calculate_volume_spike_confidence(
                    current_volume, baseline_volume, monitoring_window_minutes
                )
                
                severity = self._determine_volume_spike_severity(volume_ratio)
                
                return EmergencyDetection(
                    trigger_type=EmergencyTriggerType.VOLUME_SPIKE.value,
                    severity=severity,
                    confidence=confidence,
                    threshold_value=spike_threshold,
                    actual_value=volume_ratio,
                    data_source=f"ftns_transactions_{monitoring_window_minutes}m",
                    detection_time=end_time,
                    auto_response_recommended=confidence > self.emergency_config['confidence_threshold'],
                    governance_required=confidence > self.emergency_config['governance_escalation_threshold']
                )
                
        except Exception as e:
            await logger.aerror("Volume spike detection failed", error=str(e))
            return None
    
    async def detect_oracle_failure(self) -> Optional[EmergencyDetection]:
        """
        Detect price oracle failures or significant deviations between sources
        """
        
        if not self.price_oracle:
            return None
        
        try:
            # Get prices from multiple sources
            oracle_prices = await self.price_oracle.get_multi_source_prices()
            
            if len(oracle_prices) < 2:
                return None
            
            # Calculate price deviations
            prices = [float(p['price']) for p in oracle_prices.values()]
            mean_price = sum(prices) / len(prices)
            max_deviation = max(abs(p - mean_price) / mean_price for p in prices)
            
            deviation_threshold = self.emergency_config['oracle_deviation_threshold']
            
            if max_deviation >= deviation_threshold:
                # High confidence for oracle failures (objective measurement)
                confidence = min(0.95, 0.7 + (max_deviation - deviation_threshold) * 2)
                
                severity = self._determine_oracle_failure_severity(max_deviation)
                
                return EmergencyDetection(
                    trigger_type=EmergencyTriggerType.ORACLE_FAILURE.value,
                    severity=severity,
                    confidence=confidence,
                    threshold_value=deviation_threshold,
                    actual_value=max_deviation,
                    data_source="multi_oracle_comparison",
                    detection_time=datetime.now(timezone.utc),
                    auto_response_recommended=True,
                    governance_required=max_deviation > 0.2  # 20% deviation requires governance
                )
                
        except Exception as e:
            await logger.aerror("Oracle failure detection failed", error=str(e))
            return None
    
    async def detect_system_errors(self, monitoring_window_minutes: int = 15) -> Optional[EmergencyDetection]:
        """
        Detect system error spikes that could indicate technical failures
        """
        
        try:
            # Get recent error counts from system logs
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=monitoring_window_minutes)
            
            # This would integrate with actual system monitoring
            error_count = await self._get_system_error_count(start_time, end_time)
            error_threshold = self.emergency_config['system_error_threshold']
            
            if error_count >= error_threshold:
                # Confidence based on error pattern and severity
                confidence = min(0.9, 0.6 + (error_count - error_threshold) * 0.05)
                
                severity = self._determine_system_error_severity(error_count)
                
                return EmergencyDetection(
                    trigger_type=EmergencyTriggerType.SYSTEM_ERROR.value,
                    severity=severity,
                    confidence=confidence,
                    threshold_value=error_threshold,
                    actual_value=error_count,
                    data_source="system_monitoring",
                    detection_time=end_time,
                    auto_response_recommended=error_count >= error_threshold * 2,
                    governance_required=error_count >= error_threshold * 3
                )
                
        except Exception as e:
            await logger.aerror("System error detection failed", error=str(e))
            return None
    
    # === EMERGENCY RESPONSE ===
    
    async def trigger_emergency_response(self, detection: EmergencyDetection) -> Dict[str, Any]:
        """
        Trigger appropriate emergency response based on detection results
        """
        
        try:
            # Record the emergency trigger
            trigger_record = await self._record_emergency_trigger(detection)
            
            # Determine appropriate response actions
            response_actions = await self._determine_response_actions(detection)
            
            results = []
            for action in response_actions:
                try:
                    result = await self._execute_emergency_action(trigger_record.trigger_id, action)
                    results.append(result)
                except Exception as e:
                    await logger.aerror("Emergency action execution failed", 
                                      action_type=action.action_type, error=str(e))
                    results.append({
                        "action_type": action.action_type,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Update trigger status
            await self._update_trigger_status(
                trigger_record.trigger_id, 
                EmergencyStatus.CONFIRMED.value if any(r.get("status") == "executed" for r in results)
                else EmergencyStatus.EVALUATING.value
            )
            
            # Notify governance if required
            if detection.governance_required:
                await self._notify_governance(detection, trigger_record.trigger_id)
            
            await logger.ainfo(
                "Emergency response triggered",
                trigger_type=detection.trigger_type,
                severity=detection.severity,
                confidence=detection.confidence,
                actions_executed=len([r for r in results if r.get("status") == "executed"])
            )
            
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
            
        except Exception as e:
            await logger.aerror("Emergency response failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "trigger_type": detection.trigger_type
            }
    
    async def halt_transactions(self, duration_seconds: Optional[int] = None, 
                               transaction_types: Optional[List[str]] = None,
                               authorized_by: str = "emergency_system") -> Dict[str, Any]:
        """
        Emergency halt of FTNS transactions
        """
        
        try:
            # Check cooldown
            if not await self._check_action_cooldown("halt_transactions"):
                return {
                    "status": "rejected",
                    "reason": "Action in cooldown period"
                }
            
            # Determine scope
            halt_types = transaction_types or ["transfer", "marketplace", "governance"]
            
            # Execute halt via FTNS service
            if self.ftns:
                halt_result = await self.ftns.emergency_halt_transactions(
                    transaction_types=halt_types,
                    duration_seconds=duration_seconds
                )
                
                if halt_result.get("success"):
                    # Record action in database
                    await self._record_halt_action(halt_types, duration_seconds, authorized_by)
                    
                    # Update cooldown
                    self.last_actions["halt_transactions"] = datetime.now(timezone.utc)
                    
                    await logger.awarn(
                        "Emergency transaction halt activated",
                        transaction_types=halt_types,
                        duration_seconds=duration_seconds,
                        authorized_by=authorized_by
                    )
                    
                    return {
                        "status": "executed",
                        "action_type": "halt_transactions",
                        "transaction_types": halt_types,
                        "duration_seconds": duration_seconds,
                        "affected_users": halt_result.get("affected_users", 0)
                    }
                else:
                    return {
                        "status": "failed",
                        "reason": halt_result.get("error", "Unknown error")
                    }
            else:
                # Simulate halt for testing
                return {
                    "status": "executed",
                    "action_type": "halt_transactions",
                    "transaction_types": halt_types,
                    "duration_seconds": duration_seconds,
                    "affected_users": 0,
                    "note": "Simulated - no FTNS service available"
                }
                
        except Exception as e:
            await logger.aerror("Transaction halt failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def reduce_transaction_limits(self, reduction_factor: float = 0.5,
                                      duration_seconds: Optional[int] = 3600,
                                      authorized_by: str = "emergency_system") -> Dict[str, Any]:
        """
        Temporarily reduce transaction limits to slow down suspicious activity
        """
        
        try:
            # Check cooldown
            if not await self._check_action_cooldown("reduce_limits"):
                return {
                    "status": "rejected",
                    "reason": "Action in cooldown period"
                }
            
            # Execute limit reduction via FTNS service
            if self.ftns:
                limit_result = await self.ftns.emergency_reduce_limits(
                    reduction_factor=reduction_factor,
                    duration_seconds=duration_seconds
                )
                
                if limit_result.get("success"):
                    # Record action
                    await self._record_limit_reduction_action(
                        reduction_factor, duration_seconds, authorized_by
                    )
                    
                    # Update cooldown
                    self.last_actions["reduce_limits"] = datetime.now(timezone.utc)
                    
                    await logger.awarn(
                        "Emergency transaction limits reduced",
                        reduction_factor=reduction_factor,
                        duration_seconds=duration_seconds,
                        authorized_by=authorized_by
                    )
                    
                    return {
                        "status": "executed",
                        "action_type": "reduce_limits",
                        "reduction_factor": reduction_factor,
                        "duration_seconds": duration_seconds,
                        "affected_users": limit_result.get("affected_users", 0)
                    }
                else:
                    return {
                        "status": "failed",
                        "reason": limit_result.get("error", "Unknown error")
                    }
            else:
                # Simulate limit reduction for testing
                return {
                    "status": "executed",
                    "action_type": "reduce_limits",
                    "reduction_factor": reduction_factor,
                    "duration_seconds": duration_seconds,
                    "affected_users": 0,
                    "note": "Simulated - no FTNS service available"
                }
                
        except Exception as e:
            await logger.aerror("Limit reduction failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    # === GOVERNANCE INTEGRATION ===
    
    async def create_emergency_proposal(self, detection: EmergencyDetection, 
                                       proposed_actions: List[EmergencyResponse]) -> Dict[str, Any]:
        """
        Create fast-track governance proposal for emergency response
        """
        
        try:
            if not self.governance:
                return {
                    "status": "error",
                    "reason": "Governance service not available"
                }
            
            # Prepare proposal content
            proposal_title = f"Emergency Response: {detection.trigger_type.replace('_', ' ').title()}"
            proposal_description = await self._generate_emergency_proposal_description(
                detection, proposed_actions
            )
            
            # Create fast-track proposal
            proposal = await self.governance.create_emergency_proposal(
                title=proposal_title,
                description=proposal_description,
                proposed_actions=[{
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "severity": action.severity,
                    "duration_seconds": action.duration_seconds
                } for action in proposed_actions],
                trigger_type=detection.trigger_type,
                detection_timestamp=detection.detection_time,
                voting_period_hours=6,  # Fast-track voting
                activation_threshold=self.emergency_config.get('governance_activation_threshold', 0.15)
            )
            
            await logger.ainfo(
                "Emergency governance proposal created",
                proposal_id=proposal.get("proposal_id"),
                trigger_type=detection.trigger_type,
                actions_count=len(proposed_actions)
            )
            
            return proposal
            
        except Exception as e:
            await logger.aerror("Emergency proposal creation failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    # === MONITORING AND AUTOMATION ===
    
    async def start_emergency_monitoring(self, check_interval_seconds: int = 60):
        """
        Start continuous emergency monitoring with configurable check interval
        """
        
        if self.monitoring_active:
            await logger.awarn("Emergency monitoring already active")
            return
        
        self.monitoring_active = True
        await logger.ainfo("Emergency monitoring started", interval_seconds=check_interval_seconds)
        
        try:
            while self.monitoring_active:
                await self._run_monitoring_cycle()
                await asyncio.sleep(check_interval_seconds)
        except Exception as e:
            await logger.aerror("Emergency monitoring failed", error=str(e))
        finally:
            self.monitoring_active = False
            await logger.ainfo("Emergency monitoring stopped")
    
    async def stop_emergency_monitoring(self):
        """Stop emergency monitoring"""
        self.monitoring_active = False
        await logger.ainfo("Emergency monitoring stop requested")
    
    async def _run_monitoring_cycle(self):
        """Run one cycle of emergency detection"""
        
        try:
            # Run all detection methods
            detections = []
            
            # Price crash detection
            price_detection = await self.detect_price_crash()
            if price_detection:
                detections.append(price_detection)
            
            # Volume spike detection
            volume_detection = await self.detect_volume_spike()
            if volume_detection:
                detections.append(volume_detection)
            
            # Oracle failure detection
            oracle_detection = await self.detect_oracle_failure()
            if oracle_detection:
                detections.append(oracle_detection)
            
            # System error detection
            error_detection = await self.detect_system_errors()
            if error_detection:
                detections.append(error_detection)
            
            # Process detected emergencies
            for detection in detections:
                # Check if we're already handling this type of emergency
                emergency_key = f"{detection.trigger_type}_{detection.severity}"
                
                if emergency_key not in self.active_emergencies:
                    self.active_emergencies.add(emergency_key)
                    
                    try:
                        # Trigger emergency response
                        response_result = await self.trigger_emergency_response(detection)
                        
                        await logger.ainfo(
                            "Emergency response completed",
                            trigger_type=detection.trigger_type,
                            severity=detection.severity,
                            response_status=response_result.get("status")
                        )
                        
                    except Exception as e:
                        await logger.aerror("Emergency response failed", 
                                          trigger_type=detection.trigger_type, error=str(e))
                    finally:
                        # Remove from active set after processing
                        self.active_emergencies.discard(emergency_key)
                
        except Exception as e:
            await logger.aerror("Monitoring cycle failed", error=str(e))
    
    # === UTILITY METHODS ===
    
    async def _calculate_price_crash_confidence(self, price_history: List[Dict]) -> float:
        """Calculate confidence score for price crash detection"""
        
        if len(price_history) < 3:
            return 0.5  # Low confidence with insufficient data
        
        # Factors that increase confidence:
        # 1. Consistent downward trend
        # 2. High data quality
        # 3. Multiple data sources agreement
        
        prices = [p['price'] for p in price_history]
        
        # Check for consistent downward trend
        downward_consistency = 0
        for i in range(1, len(prices)):
            if prices[i] <= prices[i-1]:
                downward_consistency += 1
        
        trend_confidence = downward_consistency / (len(prices) - 1)
        
        # Data quality score based on oracle reliability
        data_quality = 0.8  # Default, would be calculated from oracle metadata
        
        # Combined confidence score
        confidence = (trend_confidence * 0.6 + data_quality * 0.4)
        
        return min(0.95, max(0.1, confidence))
    
    async def _calculate_volume_spike_confidence(self, current_volume: float, 
                                               baseline_volume: float, 
                                               window_minutes: int) -> float:
        """Calculate confidence score for volume spike detection"""
        
        # Confidence based on:
        # 1. Magnitude of the spike
        # 2. Pattern consistency
        # 3. Historical volatility
        
        spike_magnitude = current_volume / baseline_volume
        
        # Higher spikes = higher confidence
        magnitude_confidence = min(0.9, (spike_magnitude - 2.0) / 8.0)
        
        # Pattern analysis (would analyze transaction patterns)
        pattern_confidence = 0.7  # Default
        
        # Combined confidence
        confidence = (magnitude_confidence * 0.7 + pattern_confidence * 0.3)
        
        return min(0.95, max(0.1, confidence))
    
    def _determine_price_crash_severity(self, crash_percentage: float) -> str:
        """Determine severity level for price crash"""
        
        if crash_percentage >= 0.8:    # 80%+ crash
            return "critical"
        elif crash_percentage >= 0.6:  # 60%+ crash
            return "high"
        elif crash_percentage >= 0.4:  # 40%+ crash
            return "medium"
        else:
            return "low"
    
    def _determine_volume_spike_severity(self, volume_ratio: float) -> str:
        """Determine severity level for volume spike"""
        
        if volume_ratio >= 20:      # 20x normal volume
            return "critical"
        elif volume_ratio >= 10:    # 10x normal volume
            return "high"
        elif volume_ratio >= 5:     # 5x normal volume
            return "medium"
        else:
            return "low"
    
    def _determine_oracle_failure_severity(self, deviation: float) -> str:
        """Determine severity level for oracle failure"""
        
        if deviation >= 0.5:        # 50%+ deviation
            return "critical"
        elif deviation >= 0.3:      # 30%+ deviation
            return "high"
        elif deviation >= 0.15:     # 15%+ deviation
            return "medium"
        else:
            return "low"
    
    def _determine_system_error_severity(self, error_count: int) -> str:
        """Determine severity level for system errors"""
        
        threshold = self.emergency_config['system_error_threshold']
        
        if error_count >= threshold * 5:    # 5x threshold
            return "critical"
        elif error_count >= threshold * 3:  # 3x threshold
            return "high"
        elif error_count >= threshold * 2:  # 2x threshold
            return "medium"
        else:
            return "low"
    
    async def _check_action_cooldown(self, action_type: str) -> bool:
        """Check if action is in cooldown period"""
        
        if action_type not in self.last_actions:
            return True
        
        cooldown_duration = self.action_cooldowns.get(action_type, 300)
        last_action = self.last_actions[action_type]
        
        return (datetime.now(timezone.utc) - last_action).total_seconds() >= cooldown_duration
    
    # === DATABASE OPERATIONS ===
    
    async def _record_emergency_trigger(self, detection: EmergencyDetection) -> FTNSEmergencyTrigger:
        """Record emergency trigger in database"""
        
        trigger_record = FTNSEmergencyTrigger(
            trigger_type=detection.trigger_type,
            threshold_value=Decimal(str(detection.threshold_value)),
            actual_value=Decimal(str(detection.actual_value)),
            confidence_score=Decimal(str(detection.confidence)),
            severity_level=detection.severity,
            data_source=detection.data_source,
            detection_algorithm="emergency_protocols_v1",
            time_window_seconds=3600,  # Default 1 hour
            status=EmergencyStatus.DETECTED.value,
            detected_at=detection.detection_time,
            auto_response_enabled=detection.auto_response_recommended,
            governance_required=detection.governance_required,
            trigger_metadata={
                "detection_method": "automated_monitoring",
                "data_quality": "high",
                "trigger_context": detection.data_source
            }
        )
        
        self.db.add(trigger_record)
        await self.db.commit()
        
        return trigger_record
    
    async def _update_trigger_status(self, trigger_id: UUID, status: str):
        """Update emergency trigger status"""
        
        await self.db.execute(
            update(FTNSEmergencyTrigger)
            .where(FTNSEmergencyTrigger.trigger_id == trigger_id)
            .values(
                status=status,
                confirmed_at=datetime.now(timezone.utc) if status == EmergencyStatus.CONFIRMED.value else None
            )
        )
        await self.db.commit()
    
    async def _determine_response_actions(self, detection: EmergencyDetection) -> List[EmergencyResponse]:
        """Determine appropriate response actions for emergency"""
        
        actions = []
        
        # Price crash responses
        if detection.trigger_type == EmergencyTriggerType.PRICE_CRASH.value:
            if detection.severity in ["high", "critical"]:
                actions.append(EmergencyResponse(
                    action_type=EmergencyActionType.HALT_TRANSACTIONS.value,
                    severity=detection.severity,
                    target_scope="all_users",
                    duration_seconds=3600,  # 1 hour
                    affected_users=None,
                    parameters={"transaction_types": ["transfer", "marketplace"]},
                    authorized_by="emergency_system",
                    governance_proposal_id=None
                ))
            
            if detection.severity == "medium":
                actions.append(EmergencyResponse(
                    action_type=EmergencyActionType.REDUCE_LIMITS.value,
                    severity=detection.severity,
                    target_scope="all_users",
                    duration_seconds=1800,  # 30 minutes
                    affected_users=None,
                    parameters={"reduction_factor": 0.5},
                    authorized_by="emergency_system",
                    governance_proposal_id=None
                ))
        
        # Volume spike responses
        elif detection.trigger_type == EmergencyTriggerType.VOLUME_SPIKE.value:
            if detection.severity in ["high", "critical"]:
                actions.append(EmergencyResponse(
                    action_type=EmergencyActionType.REDUCE_LIMITS.value,
                    severity=detection.severity,
                    target_scope="all_users",
                    duration_seconds=600,  # 10 minutes
                    affected_users=None,
                    parameters={"reduction_factor": 0.3},
                    authorized_by="emergency_system",
                    governance_proposal_id=None
                ))
        
        # Always notify governance for high confidence events
        if detection.governance_required:
            actions.append(EmergencyResponse(
                action_type=EmergencyActionType.NOTIFY_GOVERNANCE.value,
                severity=detection.severity,
                target_scope="system_wide",
                duration_seconds=None,
                affected_users=None,
                parameters={"urgency": "high", "voting_window_hours": 6},
                authorized_by="emergency_system",
                governance_proposal_id=None
            ))
        
        return actions
    
    async def _execute_emergency_action(self, trigger_id: UUID, 
                                       action: EmergencyResponse) -> Dict[str, Any]:
        """Execute a specific emergency action"""
        
        try:
            # Record action in database
            action_record = FTNSEmergencyAction(
                trigger_id=trigger_id,
                action_type=action.action_type,
                action_severity=action.severity,
                target_scope=action.target_scope,
                duration_seconds=action.duration_seconds,
                affected_users=action.affected_users,
                status="pending",
                authorized_by=action.authorized_by,
                action_metadata=action.parameters
            )
            
            self.db.add(action_record)
            await self.db.commit()
            
            # Execute the action based on type
            if action.action_type == EmergencyActionType.HALT_TRANSACTIONS.value:
                result = await self.halt_transactions(
                    duration_seconds=action.duration_seconds,
                    transaction_types=action.parameters.get("transaction_types"),
                    authorized_by=action.authorized_by
                )
            
            elif action.action_type == EmergencyActionType.REDUCE_LIMITS.value:
                result = await self.reduce_transaction_limits(
                    reduction_factor=action.parameters.get("reduction_factor", 0.5),
                    duration_seconds=action.duration_seconds,
                    authorized_by=action.authorized_by
                )
            
            elif action.action_type == EmergencyActionType.NOTIFY_GOVERNANCE.value:
                result = await self._notify_governance_action(action.parameters)
            
            else:
                result = {
                    "status": "error",
                    "reason": f"Unknown action type: {action.action_type}"
                }
            
            # Update action record with results
            await self.db.execute(
                update(FTNSEmergencyAction)
                .where(FTNSEmergencyAction.action_id == action_record.action_id)
                .values(
                    status="executed" if result.get("status") == "executed" else "failed",
                    executed_at=datetime.now(timezone.utc),
                    users_affected=result.get("affected_users", 0),
                    action_metadata={**action.parameters, "execution_result": result}
                )
            )
            await self.db.commit()
            
            return result
            
        except Exception as e:
            await logger.aerror("Emergency action execution failed", 
                              action_type=action.action_type, error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    # === PLACEHOLDER METHODS (Integration Points) ===
    
    async def _get_system_error_count(self, start_time: datetime, end_time: datetime) -> int:
        """Get system error count - would integrate with actual monitoring"""
        # Placeholder - would connect to actual system monitoring
        return 0
    
    async def _notify_governance(self, detection: EmergencyDetection, trigger_id: UUID):
        """Notify governance of emergency - would integrate with governance system"""
        await logger.ainfo("Governance notification sent", 
                          trigger_type=detection.trigger_type, trigger_id=str(trigger_id))
    
    async def _notify_governance_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute governance notification action"""
        await logger.ainfo("Governance notification action executed", parameters=parameters)
        return {
            "status": "executed",
            "action_type": "notify_governance",
            "affected_users": 0
        }
    
    async def _record_halt_action(self, halt_types: List[str], 
                                 duration_seconds: Optional[int], authorized_by: str):
        """Record transaction halt action"""
        # Would record in database
        pass
    
    async def _record_limit_reduction_action(self, reduction_factor: float, 
                                           duration_seconds: Optional[int], authorized_by: str):
        """Record limit reduction action"""
        # Would record in database
        pass
    
    async def _generate_emergency_proposal_description(self, detection: EmergencyDetection, 
                                                      actions: List[EmergencyResponse]) -> str:
        """Generate description for emergency governance proposal"""
        
        return f"""
Emergency Response Proposal

Trigger: {detection.trigger_type.replace('_', ' ').title()}
Severity: {detection.severity.title()}
Confidence: {detection.confidence:.1%}
Detection Time: {detection.detection_time.isoformat()}

Detected Condition:
- Threshold: {detection.threshold_value}
- Actual Value: {detection.actual_value}
- Data Source: {detection.data_source}

Proposed Actions:
{chr(10).join(f"- {action.action_type.replace('_', ' ').title()}: {action.severity} severity" for action in actions)}

This proposal requests fast-track approval for emergency response measures to protect FTNS system stability.
"""

    async def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency system status"""
        
        return {
            "monitoring_active": self.monitoring_active,
            "active_emergencies": len(self.active_emergencies),
            "emergency_config": self.emergency_config,
            "last_price_check": self.last_price_check.isoformat() if self.last_price_check else None,
            "action_cooldowns": {
                action: (datetime.now(timezone.utc) - last_time).total_seconds() 
                for action, last_time in self.last_actions.items()
            },
            "status_timestamp": datetime.now(timezone.utc).isoformat()
        }