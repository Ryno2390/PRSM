"""
Distributed Consensus Mechanisms for PRSM
Byzantine fault tolerance with safety oversight and advanced agreement protocols

IMPLEMENTATION STATUS:
- Byzantine Fault Tolerance: ✅ Core algorithms implemented
- Consensus Protocols: ✅ PBFT and voting mechanisms complete
- P2P Network Integration: ✅ Basic networking layer implemented
- Testing: ⚠️ Unit tests exist, large-scale validation not yet performed
- Performance Metrics: ⚠️ Success rates not yet benchmarked in production
- Fault Injection Testing: ❌ Not yet implemented (planned feature)

NOTE: Success rates and fault tolerance percentages will be determined through 
comprehensive testing once deployed in production environment.
"""

import asyncio
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.core.models import (
    PeerNode, AgentResponse, SafetyFlag, SafetyLevel
)
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from prsm.economy.tokenomics.ftns_service import get_ftns_service


# === Consensus Configuration ===

# Byzantine fault tolerance settings
BYZANTINE_FAULT_TOLERANCE = float(getattr(settings, "PRSM_BYZANTINE_THRESHOLD", 0.33))  # 33% Byzantine nodes tolerated
MIN_CONSENSUS_PARTICIPANTS = int(getattr(settings, "PRSM_MIN_CONSENSUS_PEERS", 4))
CONSENSUS_TIMEOUT_SECONDS = int(getattr(settings, "PRSM_CONSENSUS_TIMEOUT", 30))
MAX_CONSENSUS_ROUNDS = int(getattr(settings, "PRSM_MAX_CONSENSUS_ROUNDS", 5))

# Agreement thresholds
STRONG_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_STRONG_CONSENSUS", 0.80))  # 80% agreement
WEAK_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_WEAK_CONSENSUS", 0.67))    # 67% agreement
SAFETY_CONSENSUS_THRESHOLD = float(getattr(settings, "PRSM_SAFETY_CONSENSUS", 0.90)) # 90% for safety decisions

# Validation settings
ENABLE_SAFETY_CONSENSUS = getattr(settings, "PRSM_SAFETY_CONSENSUS_ENABLED", True)
EXECUTION_LOG_VERIFICATION = getattr(settings, "PRSM_EXECUTION_LOG_VERIFICATION", True)
PEER_REPUTATION_WEIGHTING = getattr(settings, "PRSM_REPUTATION_WEIGHTING", True)


class ConsensusType:
    """Types of consensus mechanisms available"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    SAFETY_CRITICAL = "safety_critical"
    ZK_SNARK = "zk_snark"
    ZK_SNARK = "zk_snark"


class ConsensusResult:
    """Result of a consensus operation"""
    def __init__(self, 
                 agreed_value: Any = None,
                 consensus_achieved: bool = False,
                 consensus_type: str = ConsensusType.SIMPLE_MAJORITY,
                 agreement_ratio: float = 0.0,
                 participating_peers: List[str] = None,
                 failed_peers: List[str] = None,
                 consensus_rounds: int = 1,
                 execution_time: float = 0.0):
        self.agreed_value = agreed_value
        self.consensus_achieved = consensus_achieved
        self.consensus_type = consensus_type
        self.agreement_ratio = agreement_ratio
        self.participating_peers = participating_peers or []
        self.failed_peers = failed_peers or []
        self.consensus_rounds = consensus_rounds
        self.execution_time = execution_time
        self.timestamp = datetime.now(timezone.utc)


class DistributedConsensus:
    """
    Distributed consensus mechanisms with Byzantine fault tolerance and safety oversight
    Implements multiple consensus algorithms with safety framework integration
    """
    
    def __init__(self):
        # Core consensus state
        self.active_consensus_sessions: Dict[str, Dict[str, Any]] = {}
        self.peer_reputations: Dict[str, float] = {}
        self.consensus_history: List[ConsensusResult] = []
        self.ftns_service = get_ftns_service()
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Performance tracking
        self.consensus_metrics: Dict[str, Any] = {
            "total_consensus_attempts": 0,
            "successful_consensus": 0,
            "byzantine_failures_detected": 0,
            "average_consensus_time": 0.0,
            "consensus_types_used": Counter()
        }
        
        # Synchronization
        self._consensus_lock = asyncio.Lock()
        self._reputation_lock = asyncio.Lock()
        
    
    async def achieve_result_consensus(self, 
                                     peer_results: List[Dict[str, Any]], 
                                     consensus_type: str = ConsensusType.BYZANTINE_FAULT_TOLERANT,
                                     session_id: Optional[str] = None) -> ConsensusResult:
        """
        Achieve consensus on peer results using specified consensus mechanism
        
        Args:
            peer_results: List of results from different peers
            consensus_type: Type of consensus mechanism to use
            session_id: Optional session identifier for tracking
            
        Returns:
            ConsensusResult with agreed value and consensus details
        """
        start_time = datetime.now(timezone.utc)
        session_id = session_id or str(uuid4())
        
        async with self._consensus_lock:
            try:
                self.consensus_metrics["total_consensus_attempts"] += 1
                self.consensus_metrics["consensus_types_used"][consensus_type] += 1
                
                # Safety validation for all peer results
                if ENABLE_SAFETY_CONSENSUS:
                    peer_results = await self._safety_validate_peer_results(peer_results)
                
                # Check minimum participants (Bypass for ZK proofs)
                if consensus_type != ConsensusType.ZK_SNARK and len(peer_results) < MIN_CONSENSUS_PARTICIPANTS:
                    return ConsensusResult(
                        consensus_achieved=False,
                        consensus_type=consensus_type,
                        participating_peers=[r.get("peer_id", "") for r in peer_results],
                        execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                    )
                
                # Execute consensus based on type
                if consensus_type == ConsensusType.SIMPLE_MAJORITY:
                    result = await self._simple_majority_consensus(peer_results, session_id)
                elif consensus_type == ConsensusType.WEIGHTED_MAJORITY:
                    result = await self._weighted_majority_consensus(peer_results, session_id)
                elif consensus_type == ConsensusType.BYZANTINE_FAULT_TOLERANT:
                    result = await self._byzantine_fault_tolerant_consensus(peer_results, session_id)
                elif consensus_type == ConsensusType.SAFETY_CRITICAL:
                    result = await self._safety_critical_consensus(peer_results, session_id)
                elif consensus_type == ConsensusType.ZK_SNARK:
                    result = await self._zk_snark_verification(peer_results, session_id)
                else:
                    raise ValueError(f"Unknown consensus type: {consensus_type}")
                
                # Update execution time
                result.execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Update metrics
                if result.consensus_achieved:
                    self.consensus_metrics["successful_consensus"] += 1
                    
                    # Update average consensus time
                    total_attempts = self.consensus_metrics["total_consensus_attempts"]
                    current_avg = self.consensus_metrics["average_consensus_time"]
                    self.consensus_metrics["average_consensus_time"] = (
                        (current_avg * (total_attempts - 1) + result.execution_time) / total_attempts
                    )
                
                # Store in history
                self.consensus_history.append(result)
                
                # Limit history size
                if len(self.consensus_history) > 1000:
                    self.consensus_history = self.consensus_history[-500:]
                
                print(f"🤝 Consensus {'achieved' if result.consensus_achieved else 'failed'}: "
                      f"{result.agreement_ratio:.2%} agreement ({consensus_type})")
                
                return result
                
            except Exception as e:
                print(f"❌ Consensus error: {str(e)}")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type=consensus_type,
                    execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
    
    
    async def validate_execution_integrity(self, execution_log: List[Dict[str, Any]]) -> bool:
        """
        Validate the integrity of distributed execution logs
        
        Args:
            execution_log: List of execution log entries from peers
            
        Returns:
            True if execution integrity is validated
        """
        if not EXECUTION_LOG_VERIFICATION:
            return True
            
        try:
            if not execution_log:
                return False
            
            # Group logs by peer
            peer_logs = defaultdict(list)
            for log_entry in execution_log:
                peer_id = log_entry.get("peer_id")
                if peer_id:
                    peer_logs[peer_id].append(log_entry)
            
            # Validate each peer's log consistency
            for peer_id, logs in peer_logs.items():
                if not await self._validate_peer_log_consistency(peer_id, logs):
                    print(f"⚠️ Log inconsistency detected for peer {peer_id}")
                    return False
            
            # Cross-validate between peers
            if not await self._cross_validate_peer_logs(peer_logs):
                print("⚠️ Cross-peer log validation failed")
                return False
            
            # Safety validation of execution sequence
            if ENABLE_SAFETY_CONSENSUS:
                safety_valid = await self.safety_monitor.validate_model_output(
                    {"execution_log": execution_log},
                    ["validate_execution_sequence", "check_safety_compliance"]
                )
                if not safety_valid:
                    print("⚠️ Safety validation of execution log failed")
                    return False
            
            print(f"✅ Execution integrity validated for {len(peer_logs)} peers")
            return True
            
        except Exception as e:
            print(f"❌ Execution integrity validation error: {str(e)}")
            return False
    
    
    async def handle_byzantine_failures(self, failed_peers: List[str], session_id: Optional[str] = None):
        """
        Handle Byzantine failures with safety framework integration
        
        Args:
            failed_peers: List of peer IDs that exhibited Byzantine behavior
            session_id: Optional session identifier for tracking
        """
        try:
            session_id = session_id or str(uuid4())
            
            print(f"🚨 Handling Byzantine failures for {len(failed_peers)} peers")
            
            # Update metrics
            self.consensus_metrics["byzantine_failures_detected"] += len(failed_peers)
            
            # Update peer reputations
            async with self._reputation_lock:
                for peer_id in failed_peers:
                    if peer_id in self.peer_reputations:
                        # Significant reputation penalty for Byzantine behavior
                        self.peer_reputations[peer_id] = max(0.0, self.peer_reputations[peer_id] - 0.5)
                    else:
                        self.peer_reputations[peer_id] = 0.0
                    
                    print(f"📉 Peer {peer_id} reputation reduced to {self.peer_reputations[peer_id]:.2f}")
            
            # Safety framework integration
            if ENABLE_SAFETY_CONSENSUS:
                for peer_id in failed_peers:
                    # Report to circuit breaker
                    assessment = await self.circuit_breaker.monitor_model_behavior(
                        peer_id,
                        {
                            "behavior": "byzantine_failure",
                            "session_id": session_id,
                            "timestamp": datetime.now(timezone.utc)
                        }
                    )
                    
                    # Trigger emergency halt if threat level is high
                    if assessment and hasattr(assessment, 'threat_level'):
                        threat_value = getattr(assessment.threat_level, 'value', 0)
                        if threat_value >= ThreatLevel.HIGH.value:
                            await self.circuit_breaker.trigger_emergency_halt(
                                ThreatLevel.HIGH.value,
                                f"Byzantine failure detected in peer {peer_id}"
                            )
            
            # FTNS penalty (economic disincentive)
            for peer_id in failed_peers:
                try:
                    # Penalty for Byzantine behavior
                    self.ftns_service.deduct_tokens(peer_id, Decimal('1000'), description="Byzantine failure penalty")
                except Exception as e:
                    print(f"⚠️ FTNS penalty failed for peer {peer_id}: {str(e)}")
            
            print(f"✅ Byzantine failure handling completed for session {session_id}")
            
        except Exception as e:
            print(f"❌ Byzantine failure handling error: {str(e)}")
    
    
    async def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get current consensus performance metrics"""
        return {
            **self.consensus_metrics,
            "active_sessions": len(self.active_consensus_sessions),
            "peer_count": len(self.peer_reputations),
            "average_peer_reputation": (
                statistics.mean(self.peer_reputations.values()) 
                if self.peer_reputations else 0.0
            ),
            "consensus_history_size": len(self.consensus_history),
            "byzantine_fault_tolerance": BYZANTINE_FAULT_TOLERANCE,
            "consensus_thresholds": {
                "strong": STRONG_CONSENSUS_THRESHOLD,
                "weak": WEAK_CONSENSUS_THRESHOLD,
                "safety": SAFETY_CONSENSUS_THRESHOLD
            }
        }
    
    
    async def update_peer_reputation(self, peer_id: str, reputation_delta: float):
        """Update peer reputation based on consensus participation"""
        async with self._reputation_lock:
            if peer_id not in self.peer_reputations:
                self.peer_reputations[peer_id] = 0.5  # Starting reputation
            
            # Apply delta with bounds
            new_reputation = self.peer_reputations[peer_id] + reputation_delta
            self.peer_reputations[peer_id] = max(0.0, min(1.0, new_reputation))
    
    
    # === Private Helper Methods ===
    
    async def _simple_majority_consensus(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Simple majority consensus mechanism"""
        if not peer_results:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.SIMPLE_MAJORITY)
        
        # Extract result values and count occurrences
        result_values = []
        peer_ids = []
        
        for result in peer_results:
            if "result" in result:
                result_hash = self._hash_result(result["result"])
                result_values.append(result_hash)
                peer_ids.append(result.get("peer_id", "unknown"))
        
        if not result_values:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.SIMPLE_MAJORITY)
        
        # Count votes
        vote_counts = Counter(result_values)
        winning_hash, winning_count = vote_counts.most_common(1)[0]
        
        # Calculate agreement ratio
        agreement_ratio = winning_count / len(result_values)
        
        # Check if majority is achieved
        consensus_achieved = agreement_ratio > 0.5
        
        # Find the actual result value for the winning hash
        agreed_value = None
        for result in peer_results:
            if "result" in result and self._hash_result(result["result"]) == winning_hash:
                agreed_value = result["result"]
                break
        
        return ConsensusResult(
            agreed_value=agreed_value,
            consensus_achieved=consensus_achieved,
            consensus_type=ConsensusType.SIMPLE_MAJORITY,
            agreement_ratio=agreement_ratio,
            participating_peers=peer_ids
        )
    
    
    async def _weighted_majority_consensus(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Weighted majority consensus using peer reputations"""
        if not peer_results:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.WEIGHTED_MAJORITY)
        
        # Extract results with reputation weights
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        peer_ids = []
        
        for result in peer_results:
            peer_id = result.get("peer_id", "unknown")
            peer_ids.append(peer_id)
            
            if "result" in result:
                result_hash = self._hash_result(result["result"])
                reputation = self.peer_reputations.get(peer_id, 0.5)
                
                weighted_votes[result_hash] += reputation
                total_weight += reputation
        
        if not weighted_votes or total_weight == 0:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.WEIGHTED_MAJORITY)
        
        # Find winning result
        winning_hash = max(weighted_votes, key=weighted_votes.get)
        winning_weight = weighted_votes[winning_hash]
        
        # Calculate weighted agreement ratio
        agreement_ratio = winning_weight / total_weight
        
        # Check consensus threshold
        consensus_achieved = agreement_ratio >= WEAK_CONSENSUS_THRESHOLD
        
        # Find the actual result value
        agreed_value = None
        for result in peer_results:
            if "result" in result and self._hash_result(result["result"]) == winning_hash:
                agreed_value = result["result"]
                break
        
        return ConsensusResult(
            agreed_value=agreed_value,
            consensus_achieved=consensus_achieved,
            consensus_type=ConsensusType.WEIGHTED_MAJORITY,
            agreement_ratio=agreement_ratio,
            participating_peers=peer_ids
        )
    
    
    async def _byzantine_fault_tolerant_consensus(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Byzantine fault tolerant consensus mechanism"""
        if not peer_results:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT)
        
        total_peers = len(peer_results)
        max_byzantine = int(total_peers * BYZANTINE_FAULT_TOLERANCE)
        min_honest = total_peers - max_byzantine
        
        # Require sufficient honest nodes
        if min_honest < 1:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT)
        
        # Multiple rounds of consensus with Byzantine detection
        consensus_rounds = 0
        current_results = peer_results.copy()
        failed_peers = []
        
        for round_num in range(MAX_CONSENSUS_ROUNDS):
            consensus_rounds += 1
            
            # Attempt weighted consensus
            round_result = await self._weighted_majority_consensus(current_results, session_id)
            
            # Check if strong consensus is achieved
            if round_result.consensus_achieved and round_result.agreement_ratio >= STRONG_CONSENSUS_THRESHOLD:
                return ConsensusResult(
                    agreed_value=round_result.agreed_value,
                    consensus_achieved=True,
                    consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT,
                    agreement_ratio=round_result.agreement_ratio,
                    participating_peers=round_result.participating_peers,
                    failed_peers=failed_peers,
                    consensus_rounds=consensus_rounds
                )
            
            # Detect and remove Byzantine peers
            byzantine_peers = await self._detect_byzantine_peers(current_results, round_result.agreed_value)
            
            if not byzantine_peers:
                # No Byzantine peers detected, consensus failed
                break
            
            # Remove Byzantine peers and retry
            failed_peers.extend(byzantine_peers)
            current_results = [r for r in current_results if r.get("peer_id") not in byzantine_peers]
            
            # Handle Byzantine failures
            await self.handle_byzantine_failures(byzantine_peers, session_id)
            
            # Check if we still have enough peers
            if len(current_results) < min_honest:
                break
        
        # Final attempt with remaining peers
        if current_results:
            final_result = await self._weighted_majority_consensus(current_results, session_id)
            if final_result.consensus_achieved:
                return ConsensusResult(
                    agreed_value=final_result.agreed_value,
                    consensus_achieved=True,
                    consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT,
                    agreement_ratio=final_result.agreement_ratio,
                    participating_peers=final_result.participating_peers,
                    failed_peers=failed_peers,
                    consensus_rounds=consensus_rounds
                )
        
        return ConsensusResult(
            consensus_achieved=False,
            consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT,
            failed_peers=failed_peers,
            consensus_rounds=consensus_rounds
        )
    
    
    async def _safety_critical_consensus(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Safety-critical consensus requiring very high agreement"""
        if not peer_results:
            return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.SAFETY_CRITICAL)
        
        # Use weighted consensus as base
        base_result = await self._weighted_majority_consensus(peer_results, session_id)
        
        # Require safety consensus threshold
        consensus_achieved = (base_result.consensus_achieved and 
                            base_result.agreement_ratio >= SAFETY_CONSENSUS_THRESHOLD)
        
        # Additional safety validation
        if consensus_achieved and ENABLE_SAFETY_CONSENSUS:
            safety_valid = await self.safety_monitor.validate_model_output(
                base_result.agreed_value,
                ["safety_critical_validation", "high_confidence_check"]
            )
            consensus_achieved = consensus_achieved and safety_valid
        
        return ConsensusResult(
            agreed_value=base_result.agreed_value if consensus_achieved else None,
            consensus_achieved=consensus_achieved,
            consensus_type=ConsensusType.SAFETY_CRITICAL,
            agreement_ratio=base_result.agreement_ratio,
            participating_peers=base_result.participating_peers
        )

    async def _zk_snark_verification(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Reach consensus via Zero-Knowledge proof verification"""
        from prsm.core.cryptography.zk_proofs import get_zk_proof_system
        zk_system = await get_zk_proof_system()
        
        # In ZK mode, we only need ONE valid proof to reach consensus
        for result in peer_results:
            proof_id = result.get("zk_proof_id")
            if not proof_id:
                continue
                
            # Verify the proof (Mock implementation uses the proof_id directly for lookup)
            # In a real SNARK, we would verify the proof bytes against public inputs
            is_valid = await zk_system.verify_proof(proof_id, "consensus_engine")
            
            if is_valid:
                return ConsensusResult(
                    agreed_value=result.get("result"),
                    consensus_achieved=True,
                    consensus_type=ConsensusType.ZK_SNARK,
                    agreement_ratio=1.0,
                    participating_peers=[result.get("peer_id", "prover")]
                )
                
        return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.ZK_SNARK)
    
    
    async def _safety_validate_peer_results(self, peer_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Safety validate all peer results, filtering out unsafe ones"""
        validated_results = []
        
        for result in peer_results:
            try:
                # Safety validation
                is_safe = await self.safety_monitor.validate_model_output(
                    result,
                    ["validate_peer_result", "check_safety_compliance"]
                )
                
                if is_safe:
                    validated_results.append(result)
                else:
                    peer_id = result.get("peer_id", "unknown")
                    print(f"⚠️ Peer {peer_id} result failed safety validation")
                    
                    # Report unsafe result
                    await self.circuit_breaker.monitor_model_behavior(
                        peer_id,
                        {"unsafe_result": True, "result": result}
                    )
                    
            except Exception as e:
                print(f"⚠️ Safety validation error for peer result: {str(e)}")
        
        return validated_results
    
    
    async def _detect_byzantine_peers(self, peer_results: List[Dict[str, Any]], consensus_value: Any) -> List[str]:
        """Detect peers exhibiting Byzantine behavior"""
        byzantine_peers = []
        
        if consensus_value is None:
            return byzantine_peers
        
        consensus_hash = self._hash_result(consensus_value)
        
        for result in peer_results:
            peer_id = result.get("peer_id")
            if not peer_id:
                continue
            
            # Check if peer result significantly differs from consensus
            if "result" in result:
                peer_hash = self._hash_result(result["result"])
                
                # If peer consistently disagrees with consensus, may be Byzantine
                if peer_hash != consensus_hash:
                    peer_reputation = self.peer_reputations.get(peer_id, 0.5)
                    
                    # Lower reputation peers more likely to be Byzantine
                    if peer_reputation < 0.3:
                        byzantine_peers.append(peer_id)
        
        return byzantine_peers
    
    
    async def _validate_peer_log_consistency(self, peer_id: str, logs: List[Dict[str, Any]]) -> bool:
        """Validate consistency of a peer's execution logs"""
        if not logs:
            return False
        
        # Check temporal consistency
        timestamps = []
        for log in logs:
            if "timestamp" in log:
                try:
                    timestamps.append(datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')))
                except:
                    return False
        
        # Timestamps should be in order
        if timestamps != sorted(timestamps):
            return False
        
        # Check log sequence consistency
        sequence_numbers = [log.get("sequence", 0) for log in logs]
        if sequence_numbers != list(range(len(sequence_numbers))):
            return False
        
        return True
    
    
    async def _cross_validate_peer_logs(self, peer_logs: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Cross-validate execution logs between peers"""
        if len(peer_logs) < 2:
            return True  # Cannot cross-validate with single peer
        
        # Compare key execution milestones across peers
        milestone_hashes = defaultdict(list)
        
        for peer_id, logs in peer_logs.items():
            for log in logs:
                if log.get("milestone"):
                    milestone = log["milestone"]
                    milestone_hash = hashlib.sha256(json.dumps(milestone, sort_keys=True).encode()).hexdigest()
                    milestone_hashes[log.get("milestone_type", "default")].append(milestone_hash)
        
        # Check for consensus on milestones
        for milestone_type, hashes in milestone_hashes.items():
            if len(set(hashes)) > 1:  # Disagreement on milestone
                hash_counts = Counter(hashes)
                most_common_hash, count = hash_counts.most_common(1)[0]
                agreement_ratio = count / len(hashes)
                
                if agreement_ratio < 0.67:  # Less than 67% agreement
                    return False
        
        return True
    
    
    def _hash_result(self, result: Any) -> str:
        """Create hash of result for comparison"""
        try:
            if isinstance(result, dict):
                # Remove non-deterministic fields
                clean_result = {k: v for k, v in result.items() 
                             if k not in ["timestamp", "execution_time", "peer_id"]}
                result_str = json.dumps(clean_result, sort_keys=True)
            else:
                result_str = str(result)
            
            return hashlib.sha256(result_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(result).encode()).hexdigest()

    async def _zk_snark_verification(self, peer_results: List[Dict[str, Any]], session_id: str) -> ConsensusResult:
        """Reach consensus via Zero-Knowledge proof verification"""
        from prsm.core.cryptography.zk_proofs import get_zk_proof_system
        zk_system = await get_zk_proof_system()
        
        # In ZK mode, we only need ONE valid proof to reach consensus
        for result in peer_results:
            proof_id = result.get("zk_proof_id")
            if not proof_id:
                continue
                
            is_valid = await zk_system.verify_proof(proof_id, "consensus_engine")
            
            if is_valid:
                return ConsensusResult(
                    agreed_value=result.get("result"),
                    consensus_achieved=True,
                    consensus_type=ConsensusType.ZK_SNARK,
                    agreement_ratio=1.0,
                    participating_peers=[result.get("peer_id", "prover")]
                )
                
        return ConsensusResult(consensus_achieved=False, consensus_type=ConsensusType.ZK_SNARK)


# === Global Consensus Instance ===

_consensus_instance: Optional[DistributedConsensus] = None

def get_consensus() -> DistributedConsensus:
    """Get or create the global distributed consensus instance"""
    global _consensus_instance
    if _consensus_instance is None:
        _consensus_instance = DistributedConsensus()
    return _consensus_instance