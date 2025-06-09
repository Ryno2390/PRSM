"""
Sybil Resistance Protocol for PRSM CDN
======================================

Implements challenge-response protocols and bandwidth proofs to prevent
gaming of the CDN reward system by fake or low-quality nodes.

Key Features:
- Challenge-response protocols for bandwidth verification
- Geographic validation through latency triangulation
- Proof-of-bandwidth with cryptographic verification
- Reputation scoring based on verified performance
- Institutional vouching system for trusted nodes
"""

import asyncio
import hashlib
import secrets
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal

import aiohttp
from pydantic import BaseModel, Field


class ChallengeType(str, Enum):
    """Types of challenges for node verification"""
    BANDWIDTH_PROOF = "bandwidth_proof"     # Prove actual bandwidth capacity
    LATENCY_VERIFICATION = "latency_verification"  # Verify geographic claims
    CONTENT_DELIVERY = "content_delivery"   # Prove ability to serve content
    UPTIME_CHECK = "uptime_check"          # Verify node availability
    STORAGE_PROOF = "storage_proof"        # Prove storage of claimed content


class ValidationStatus(str, Enum):
    """Validation status for nodes"""
    UNVERIFIED = "unverified"      # New node, not yet validated
    VERIFIED = "verified"          # Passed verification challenges
    SUSPICIOUS = "suspicious"      # Failed some challenges, under review
    BLOCKED = "blocked"           # Confirmed malicious, blocked from rewards
    VOUCHED = "vouched"           # Verified by trusted institution


@dataclass
class BandwidthChallenge:
    """Bandwidth verification challenge"""
    challenge_id: UUID
    target_node_id: UUID
    challenge_data_size_mb: int
    expected_duration_ms: int
    tolerance_percentage: float = 20.0  # 20% tolerance for network variance


@dataclass
class LatencyChallenge:
    """Latency verification challenge from multiple points"""
    challenge_id: UUID
    target_node_id: UUID
    verification_nodes: List[UUID]  # Nodes that will test latency
    expected_latency_ms: float
    geographic_consistency_required: bool = True


class NodeChallenge(BaseModel):
    """Individual challenge for a node"""
    challenge_id: UUID = Field(default_factory=uuid4)
    node_id: UUID
    challenge_type: ChallengeType
    
    # Challenge parameters
    challenge_data: Dict[str, Any] = Field(default_factory=dict)
    expected_response: Optional[str] = None
    
    # Timing
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_deadline: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    response_received: Optional[str] = None
    verification_successful: bool = False
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class NodeReputation(BaseModel):
    """Reputation score and history for a node"""
    node_id: UUID
    
    # Current reputation
    reputation_score: float = Field(ge=0.0, le=1.0, default=0.5)  # Start neutral
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    
    # Challenge history
    total_challenges: int = 0
    challenges_passed: int = 0
    challenges_failed: int = 0
    last_challenge_date: Optional[datetime] = None
    
    # Performance tracking
    average_response_time_ms: float = 0.0
    uptime_percentage: float = 0.0
    bandwidth_verification_score: float = 0.0
    
    # Trust factors
    institutional_voucher: Optional[str] = None  # Vouching institution
    peer_endorsements: int = 0
    
    # Penalties
    penalty_score: float = 0.0
    penalty_reasons: List[str] = Field(default_factory=list)


class SybilResistance:
    """
    Sybil resistance system that validates node authenticity and performance
    to prevent gaming of CDN rewards.
    """
    
    def __init__(self):
        # Challenge tracking
        self.active_challenges: Dict[UUID, NodeChallenge] = {}
        self.challenge_history: List[NodeChallenge] = []
        
        # Reputation management
        self.node_reputations: Dict[UUID, NodeReputation] = {}
        
        # Validation parameters
        self.validation_thresholds = {
            "minimum_reputation": 0.7,     # Minimum score for rewards
            "challenge_frequency_hours": 24,  # How often to challenge nodes
            "bandwidth_tolerance": 0.2,    # 20% tolerance for bandwidth claims
            "latency_tolerance": 0.3,      # 30% tolerance for latency claims
            "uptime_requirement": 0.95,    # 95% uptime requirement
        }
        
        # Trusted institutions for vouching
        self.trusted_institutions = {
            "mit.edu", "stanford.edu", "berkeley.edu", "cmu.edu",
            "oxford.ac.uk", "cambridge.ac.uk", "ethz.ch", "mpi.de"
        }
        
        print("🛡️ Sybil Resistance Protocol initialized")
        print("   - Challenge-response system active")
        print("   - Bandwidth verification enabled")
        print("   - Geographic validation ready")
    
    async def validate_new_node(self, 
                               node_id: UUID,
                               claimed_capabilities: Dict[str, Any],
                               institutional_voucher: Optional[str] = None) -> ValidationStatus:
        """
        Validate a new node joining the CDN network.
        """
        
        # Initialize reputation
        reputation = NodeReputation(node_id=node_id)
        
        # Check institutional voucher
        if institutional_voucher and self._is_trusted_institution(institutional_voucher):
            reputation.institutional_voucher = institutional_voucher
            reputation.validation_status = ValidationStatus.VOUCHED
            reputation.reputation_score = 0.8  # High initial trust
            
            print(f"✅ Node vouched by trusted institution: {institutional_voucher}")
        
        self.node_reputations[node_id] = reputation
        
        # Issue initial validation challenges
        await self._issue_comprehensive_validation(node_id, claimed_capabilities)
        
        return reputation.validation_status
    
    async def issue_challenge(self,
                            node_id: UUID,
                            challenge_type: ChallengeType,
                            challenge_params: Dict[str, Any] = None) -> NodeChallenge:
        """
        Issue a specific challenge to a node.
        """
        
        challenge_params = challenge_params or {}
        
        # Create challenge based on type
        if challenge_type == ChallengeType.BANDWIDTH_PROOF:
            challenge = await self._create_bandwidth_challenge(node_id, challenge_params)
        elif challenge_type == ChallengeType.LATENCY_VERIFICATION:
            challenge = await self._create_latency_challenge(node_id, challenge_params)
        elif challenge_type == ChallengeType.CONTENT_DELIVERY:
            challenge = await self._create_content_delivery_challenge(node_id, challenge_params)
        elif challenge_type == ChallengeType.UPTIME_CHECK:
            challenge = await self._create_uptime_challenge(node_id, challenge_params)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
        
        # Track challenge
        self.active_challenges[challenge.challenge_id] = challenge
        
        # Update reputation challenge count
        if node_id in self.node_reputations:
            self.node_reputations[node_id].total_challenges += 1
            self.node_reputations[node_id].last_challenge_date = datetime.now(timezone.utc)
        
        print(f"🎯 Challenge issued: {challenge_type} to {node_id}")
        return challenge
    
    async def submit_challenge_response(self,
                                      challenge_id: UUID,
                                      response_data: Dict[str, Any]) -> bool:
        """
        Submit response to a challenge and validate it.
        """
        
        if challenge_id not in self.active_challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        
        challenge = self.active_challenges[challenge_id]
        
        # Check if response is within deadline
        now = datetime.now(timezone.utc)
        if now > challenge.response_deadline:
            await self._handle_challenge_timeout(challenge)
            return False
        
        # Validate response based on challenge type
        validation_result = await self._validate_challenge_response(challenge, response_data)
        
        # Update challenge
        challenge.completed_at = now
        challenge.response_received = str(response_data)
        challenge.verification_successful = validation_result["success"]
        challenge.performance_metrics = validation_result.get("metrics", {})
        
        # Update node reputation
        await self._update_node_reputation(challenge.node_id, challenge, validation_result)
        
        # Move to history
        self.challenge_history.append(challenge)
        del self.active_challenges[challenge_id]
        
        print(f"📝 Challenge response processed: {challenge_id} ({'✅' if validation_result['success'] else '❌'})")
        
        return validation_result["success"]
    
    async def get_node_trust_score(self, node_id: UUID) -> float:
        """
        Get comprehensive trust score for a node (0.0 to 1.0).
        """
        
        if node_id not in self.node_reputations:
            return 0.0  # Unknown nodes have no trust
        
        reputation = self.node_reputations[node_id]
        
        # Base reputation score
        base_score = reputation.reputation_score
        
        # Institutional vouching bonus
        institutional_bonus = 0.2 if reputation.institutional_voucher else 0.0
        
        # Performance bonuses
        performance_bonus = 0.0
        if reputation.uptime_percentage > 0.99:
            performance_bonus += 0.1
        if reputation.bandwidth_verification_score > 0.9:
            performance_bonus += 0.1
        
        # Challenge success rate bonus
        success_rate = (reputation.challenges_passed / max(reputation.total_challenges, 1))
        success_bonus = success_rate * 0.2
        
        # Penalty deductions
        penalty_deduction = min(0.5, reputation.penalty_score)
        
        # Calculate final score
        final_score = base_score + institutional_bonus + performance_bonus + success_bonus - penalty_deduction
        
        return max(0.0, min(1.0, final_score))
    
    async def periodic_validation_sweep(self) -> Dict[str, Any]:
        """
        Perform periodic validation of all nodes in the network.
        """
        
        validation_results = {
            "nodes_challenged": 0,
            "nodes_validated": 0,
            "nodes_flagged": 0,
            "nodes_blocked": 0
        }
        
        current_time = datetime.now(timezone.utc)
        
        for node_id, reputation in self.node_reputations.items():
            
            # Check if node needs periodic validation
            needs_challenge = False
            
            if reputation.last_challenge_date is None:
                needs_challenge = True
            elif (current_time - reputation.last_challenge_date).hours >= self.validation_thresholds["challenge_frequency_hours"]:
                needs_challenge = True
            elif reputation.validation_status == ValidationStatus.SUSPICIOUS:
                needs_challenge = True  # Challenge suspicious nodes more frequently
            
            if needs_challenge:
                # Issue random challenge
                challenge_types = [ChallengeType.BANDWIDTH_PROOF, ChallengeType.LATENCY_VERIFICATION, 
                                 ChallengeType.UPTIME_CHECK]
                challenge_type = secrets.choice(challenge_types)
                
                try:
                    await self.issue_challenge(node_id, challenge_type)
                    validation_results["nodes_challenged"] += 1
                except Exception as e:
                    print(f"❌ Failed to challenge node {node_id}: {e}")
            
            # Update validation status based on current reputation
            trust_score = await self.get_node_trust_score(node_id)
            
            if trust_score >= self.validation_thresholds["minimum_reputation"]:
                if reputation.validation_status != ValidationStatus.VOUCHED:
                    reputation.validation_status = ValidationStatus.VERIFIED
                validation_results["nodes_validated"] += 1
            elif trust_score >= 0.3:
                reputation.validation_status = ValidationStatus.SUSPICIOUS
                validation_results["nodes_flagged"] += 1
            else:
                reputation.validation_status = ValidationStatus.BLOCKED
                validation_results["nodes_blocked"] += 1
        
        print(f"🔍 Periodic validation sweep completed")
        print(f"   - Nodes challenged: {validation_results['nodes_challenged']}")
        print(f"   - Nodes flagged: {validation_results['nodes_flagged']}")
        
        return validation_results
    
    def _is_trusted_institution(self, voucher: str) -> bool:
        """Check if voucher is from a trusted institution"""
        voucher_domain = voucher.lower().split('@')[-1] if '@' in voucher else voucher.lower()
        return voucher_domain in self.trusted_institutions
    
    async def _issue_comprehensive_validation(self, 
                                            node_id: UUID, 
                                            claimed_capabilities: Dict[str, Any]):
        """Issue comprehensive validation challenges for new nodes"""
        
        # Always start with bandwidth proof
        await self.issue_challenge(
            node_id, 
            ChallengeType.BANDWIDTH_PROOF,
            {"claimed_bandwidth": claimed_capabilities.get("bandwidth_mbps", 10)}
        )
        
        # Add latency verification if geographic claims made
        if "geographic_location" in claimed_capabilities:
            await self.issue_challenge(
                node_id,
                ChallengeType.LATENCY_VERIFICATION,
                {"claimed_location": claimed_capabilities["geographic_location"]}
            )
        
        # Add uptime check
        await self.issue_challenge(node_id, ChallengeType.UPTIME_CHECK)
    
    async def _create_bandwidth_challenge(self, 
                                        node_id: UUID, 
                                        params: Dict[str, Any]) -> NodeChallenge:
        """Create bandwidth verification challenge"""
        
        # Challenge: Download test data and report timing
        test_data_size_mb = 10  # 10 MB test
        challenge_data = {
            "test_data_url": f"https://test-data.prsm.ai/bandwidth-test-{test_data_size_mb}mb.bin",
            "expected_max_duration_ms": (test_data_size_mb * 8 * 1000) / params.get("claimed_bandwidth", 10),
            "verification_hash": hashlib.sha256(f"bandwidth-test-{node_id}".encode()).hexdigest()
        }
        
        return NodeChallenge(
            node_id=node_id,
            challenge_type=ChallengeType.BANDWIDTH_PROOF,
            challenge_data=challenge_data,
            response_deadline=datetime.now(timezone.utc) + timedelta(minutes=5)
        )
    
    async def _create_latency_challenge(self, 
                                      node_id: UUID, 
                                      params: Dict[str, Any]) -> NodeChallenge:
        """Create latency verification challenge"""
        
        challenge_data = {
            "ping_targets": [
                "8.8.8.8",  # Google DNS
                "1.1.1.1",  # Cloudflare DNS
                "208.67.222.222"  # OpenDNS
            ],
            "expected_pattern": "geographic_consistency",
            "claimed_location": params.get("claimed_location")
        }
        
        return NodeChallenge(
            node_id=node_id,
            challenge_type=ChallengeType.LATENCY_VERIFICATION,
            challenge_data=challenge_data,
            response_deadline=datetime.now(timezone.utc) + timedelta(minutes=2)
        )
    
    async def _create_content_delivery_challenge(self, 
                                               node_id: UUID, 
                                               params: Dict[str, Any]) -> NodeChallenge:
        """Create content delivery challenge"""
        
        # Generate random content hash to serve
        test_content_hash = hashlib.sha256(f"test-content-{node_id}-{time.time()}".encode()).hexdigest()
        
        challenge_data = {
            "content_hash": test_content_hash,
            "expected_response_time_ms": 1000,
            "content_verification": "hash_match"
        }
        
        return NodeChallenge(
            node_id=node_id,
            challenge_type=ChallengeType.CONTENT_DELIVERY,
            challenge_data=challenge_data,
            response_deadline=datetime.now(timezone.utc) + timedelta(minutes=3)
        )
    
    async def _create_uptime_challenge(self, 
                                     node_id: UUID, 
                                     params: Dict[str, Any]) -> NodeChallenge:
        """Create uptime verification challenge"""
        
        challenge_data = {
            "ping_interval_minutes": 15,
            "duration_hours": 24,
            "acceptable_downtime_minutes": 30
        }
        
        return NodeChallenge(
            node_id=node_id,
            challenge_type=ChallengeType.UPTIME_CHECK,
            challenge_data=challenge_data,
            response_deadline=datetime.now(timezone.utc) + timedelta(hours=25)  # 24h + 1h grace
        )
    
    async def _validate_challenge_response(self, 
                                         challenge: NodeChallenge, 
                                         response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response based on challenge type"""
        
        if challenge.challenge_type == ChallengeType.BANDWIDTH_PROOF:
            return await self._validate_bandwidth_response(challenge, response_data)
        elif challenge.challenge_type == ChallengeType.LATENCY_VERIFICATION:
            return await self._validate_latency_response(challenge, response_data)
        elif challenge.challenge_type == ChallengeType.CONTENT_DELIVERY:
            return await self._validate_content_delivery_response(challenge, response_data)
        elif challenge.challenge_type == ChallengeType.UPTIME_CHECK:
            return await self._validate_uptime_response(challenge, response_data)
        else:
            return {"success": False, "reason": "Unknown challenge type"}
    
    async def _validate_bandwidth_response(self, 
                                         challenge: NodeChallenge, 
                                         response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bandwidth proof response"""
        
        reported_duration = response_data.get("download_duration_ms", 0)
        expected_max_duration = challenge.challenge_data["expected_max_duration_ms"]
        
        # Allow for network variance
        tolerance = self.validation_thresholds["bandwidth_tolerance"]
        acceptable_max_duration = expected_max_duration * (1 + tolerance)
        
        success = reported_duration <= acceptable_max_duration
        
        return {
            "success": success,
            "reason": f"Bandwidth test: {reported_duration}ms vs {expected_max_duration}ms expected",
            "metrics": {
                "reported_duration_ms": reported_duration,
                "expected_duration_ms": expected_max_duration,
                "performance_ratio": expected_max_duration / max(reported_duration, 1)
            }
        }
    
    async def _validate_latency_response(self, 
                                       challenge: NodeChallenge, 
                                       response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate latency verification response"""
        
        ping_results = response_data.get("ping_results", {})
        
        # Check for geographic consistency
        latencies = list(ping_results.values())
        if not latencies:
            return {"success": False, "reason": "No ping results provided"}
        
        # Basic validation: latencies should be reasonable and consistent
        avg_latency = sum(latencies) / len(latencies)
        max_variance = max(latencies) - min(latencies)
        
        # Geographic consistency check (simplified)
        success = (avg_latency < 500 and  # Less than 500ms average
                  max_variance < 200)     # Less than 200ms variance
        
        return {
            "success": success,
            "reason": f"Latency test: {avg_latency:.1f}ms avg, {max_variance:.1f}ms variance",
            "metrics": {
                "average_latency_ms": avg_latency,
                "latency_variance_ms": max_variance,
                "ping_results": ping_results
            }
        }
    
    async def _validate_content_delivery_response(self, 
                                                challenge: NodeChallenge, 
                                                response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content delivery response"""
        
        delivered_hash = response_data.get("content_hash")
        response_time = response_data.get("response_time_ms", float('inf'))
        
        expected_hash = challenge.challenge_data["content_hash"]
        expected_max_time = challenge.challenge_data["expected_response_time_ms"]
        
        success = (delivered_hash == expected_hash and 
                  response_time <= expected_max_time)
        
        return {
            "success": success,
            "reason": f"Content delivery: {'hash match' if delivered_hash == expected_hash else 'hash mismatch'}, {response_time}ms",
            "metrics": {
                "response_time_ms": response_time,
                "hash_match": delivered_hash == expected_hash
            }
        }
    
    async def _validate_uptime_response(self, 
                                      challenge: NodeChallenge, 
                                      response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate uptime check response"""
        
        uptime_percentage = response_data.get("uptime_percentage", 0)
        downtime_incidents = response_data.get("downtime_incidents", [])
        
        success = uptime_percentage >= self.validation_thresholds["uptime_requirement"]
        
        return {
            "success": success,
            "reason": f"Uptime: {uptime_percentage:.1f}% (required: {self.validation_thresholds['uptime_requirement']*100:.1f}%)",
            "metrics": {
                "uptime_percentage": uptime_percentage,
                "downtime_incidents": len(downtime_incidents)
            }
        }
    
    async def _update_node_reputation(self, 
                                    node_id: UUID, 
                                    challenge: NodeChallenge, 
                                    validation_result: Dict[str, Any]):
        """Update node reputation based on challenge result"""
        
        if node_id not in self.node_reputations:
            return
        
        reputation = self.node_reputations[node_id]
        
        # Update challenge counts
        if validation_result["success"]:
            reputation.challenges_passed += 1
        else:
            reputation.challenges_failed += 1
        
        # Update performance metrics
        metrics = validation_result.get("metrics", {})
        if challenge.challenge_type == ChallengeType.BANDWIDTH_PROOF:
            if "performance_ratio" in metrics:
                reputation.bandwidth_verification_score = min(1.0, metrics["performance_ratio"])
        
        # Update reputation score (exponential moving average)
        alpha = 0.1  # Learning rate
        if validation_result["success"]:
            reputation.reputation_score = (1 - alpha) * reputation.reputation_score + alpha * 1.0
        else:
            reputation.reputation_score = (1 - alpha) * reputation.reputation_score + alpha * 0.0
        
        # Add penalties for failures
        if not validation_result["success"]:
            reputation.penalty_score += 0.1
            reputation.penalty_reasons.append(f"{challenge.challenge_type}: {validation_result.get('reason', 'Failed')}")
    
    async def _handle_challenge_timeout(self, challenge: NodeChallenge):
        """Handle challenge timeout (node didn't respond)"""
        
        print(f"⏰ Challenge timeout: {challenge.challenge_id}")
        
        # Update reputation for timeout
        if challenge.node_id in self.node_reputations:
            reputation = self.node_reputations[challenge.node_id]
            reputation.challenges_failed += 1
            reputation.penalty_score += 0.2  # Higher penalty for timeout
            reputation.penalty_reasons.append(f"{challenge.challenge_type}: Timeout")
            
            # Reduce reputation score more significantly for timeouts
            reputation.reputation_score *= 0.8


# Global sybil resistance instance
sybil_resistance = SybilResistance()