"""
Post-Quantum Consensus Integration
Integrates CRYSTALS-Dilithium / ML-DSA signatures into PRSM consensus mechanisms

This module extends the consensus system with quantum-resistant digital signatures
for secure peer authentication and message integrity in distributed consensus.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from enum import Enum

# Import post-quantum crypto directly
import importlib.util
from pathlib import Path

# Load post-quantum module
pq_module_path = Path(__file__).parent.parent / "cryptography" / "post_quantum.py"
spec = importlib.util.spec_from_file_location("post_quantum", pq_module_path)
pq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq_module)

# Import post-quantum classes
PostQuantumCrypto = pq_module.PostQuantumCrypto
PostQuantumKeyPair = pq_module.PostQuantumKeyPair
PostQuantumSignature = pq_module.PostQuantumSignature
SecurityLevel = pq_module.SecurityLevel
SignatureType = pq_module.SignatureType


class ConsensusMessageType(str, Enum):
    """Types of consensus messages"""
    PROPOSAL = "proposal"
    VOTE = "vote"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    SAFETY_ALERT = "safety_alert"


@dataclass
class PostQuantumConsensusMessage:
    """Consensus message with post-quantum signature"""
    message_id: str
    message_type: ConsensusMessageType
    sender_id: str
    content: Dict[str, Any]
    timestamp: datetime
    round_number: int
    security_level: SecurityLevel
    signature: Optional[PostQuantumSignature] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "round_number": self.round_number,
            "security_level": self.security_level.value
        }
        
        if self.signature:
            data["signature"] = self.signature.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostQuantumConsensusMessage':
        """Create from dictionary"""
        message = cls(
            message_id=data["message_id"],
            message_type=ConsensusMessageType(data["message_type"]),
            sender_id=data["sender_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            round_number=data["round_number"],
            security_level=SecurityLevel(data["security_level"])
        )
        
        if "signature" in data:
            message.signature = PostQuantumSignature.from_dict(data["signature"])
        
        return message
    
    def get_message_hash(self) -> str:
        """Get hash of message content for signing"""
        message_data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "round_number": self.round_number
        }
        
        message_json = json.dumps(message_data, sort_keys=True)
        return hashlib.sha256(message_json.encode()).hexdigest()


@dataclass
class ConsensusRound:
    """Post-quantum consensus round"""
    round_id: str
    round_number: int
    proposal: Optional[PostQuantumConsensusMessage] = None
    votes: Dict[str, PostQuantumConsensusMessage] = field(default_factory=dict)
    commits: Dict[str, PostQuantumConsensusMessage] = field(default_factory=dict)
    participants: Set[str] = field(default_factory=set)
    required_votes: int = 0
    consensus_achieved: bool = False
    result: Optional[Dict[str, Any]] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None


class PostQuantumConsensusEngine:
    """
    Post-Quantum Consensus Engine
    
    Provides quantum-resistant consensus using ML-DSA signatures
    """
    
    def __init__(self, 
                 node_id: str,
                 security_level: SecurityLevel = SecurityLevel.LEVEL_1,
                 byzantine_threshold: float = 0.33):
        """
        Initialize post-quantum consensus engine
        
        Args:
            node_id: This node's identifier
            security_level: Post-quantum security level
            byzantine_threshold: Byzantine fault tolerance threshold
        """
        self.node_id = node_id
        self.security_level = security_level
        self.byzantine_threshold = byzantine_threshold
        
        # Cryptography
        self.pq_crypto = PostQuantumCrypto(security_level)
        self.node_keypair = self.pq_crypto.generate_keypair(security_level)
        
        # Consensus state
        self.current_round_number = 0
        self.active_rounds: Dict[str, ConsensusRound] = {}
        self.peer_public_keys: Dict[str, bytes] = {}
        
        # Performance tracking
        self.consensus_metrics = {
            "rounds_completed": 0,
            "rounds_failed": 0,
            "total_consensus_time": 0.0,
            "signature_verification_time": 0.0,
            "average_round_time": 0.0
        }
    
    def register_peer_public_key(self, peer_id: str, public_key: bytes):
        """Register a peer's post-quantum public key"""
        self.peer_public_keys[peer_id] = public_key
    
    def create_consensus_message(self,
                               message_type: ConsensusMessageType,
                               content: Dict[str, Any],
                               round_number: Optional[int] = None) -> PostQuantumConsensusMessage:
        """
        Create a consensus message with post-quantum signature
        
        Args:
            message_type: Type of consensus message
            content: Message content
            round_number: Consensus round number
            
        Returns:
            Signed PostQuantumConsensusMessage
        """
        if round_number is None:
            round_number = self.current_round_number
        
        # Create message
        message = PostQuantumConsensusMessage(
            message_id=str(uuid4()),
            message_type=message_type,
            sender_id=self.node_id,
            content=content,
            timestamp=datetime.now(timezone.utc),
            round_number=round_number,
            security_level=self.security_level
        )
        
        # Sign message
        message_hash = message.get_message_hash()
        signature = self.pq_crypto.sign_message(message_hash, self.node_keypair)
        message.signature = signature
        
        return message
    
    def verify_consensus_message(self, message: PostQuantumConsensusMessage) -> bool:
        """
        Verify post-quantum signature on consensus message
        
        Args:
            message: Message to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not message.signature:
            return False
        
        # Get sender's public key
        sender_public_key = self.peer_public_keys.get(message.sender_id)
        if not sender_public_key:
            return False
        
        # Verify signature
        start_time = time.perf_counter()
        
        message_hash = message.get_message_hash()
        is_valid = self.pq_crypto.verify_signature(
            message_hash,
            message.signature,
            sender_public_key
        )
        
        # Track verification time
        verification_time = time.perf_counter() - start_time
        self.consensus_metrics["signature_verification_time"] += verification_time
        
        return is_valid
    
    async def propose_consensus(self, 
                              proposal_content: Dict[str, Any],
                              participants: List[str]) -> str:
        """
        Initiate a new consensus round
        
        Args:
            proposal_content: Content to achieve consensus on
            participants: List of peer IDs to participate
            
        Returns:
            Round ID for tracking consensus
        """
        self.current_round_number += 1
        round_id = f"round_{self.current_round_number}_{uuid4().hex[:8]}"
        
        # Create consensus round
        consensus_round = ConsensusRound(
            round_id=round_id,
            round_number=self.current_round_number,
            participants=set(participants),
            required_votes=len(participants) - int(len(participants) * self.byzantine_threshold)
        )
        
        # Create proposal message
        proposal_message = self.create_consensus_message(
            ConsensusMessageType.PROPOSAL,
            proposal_content,
            self.current_round_number
        )
        
        consensus_round.proposal = proposal_message
        self.active_rounds[round_id] = consensus_round
        
        return round_id
    
    async def vote_on_proposal(self, 
                             round_id: str,
                             vote_content: Dict[str, Any]) -> PostQuantumConsensusMessage:
        """
        Vote on a consensus proposal
        
        Args:
            round_id: Consensus round ID
            vote_content: Vote content (typically {"vote": True/False, "reason": "..."})
            
        Returns:
            Signed vote message
        """
        consensus_round = self.active_rounds.get(round_id)
        if not consensus_round:
            raise ValueError(f"Consensus round {round_id} not found")
        
        # Create vote message
        vote_message = self.create_consensus_message(
            ConsensusMessageType.VOTE,
            vote_content,
            consensus_round.round_number
        )
        
        # Add vote to round
        consensus_round.votes[self.node_id] = vote_message
        
        return vote_message
    
    async def process_vote(self, 
                         round_id: str,
                         vote_message: PostQuantumConsensusMessage) -> bool:
        """
        Process a vote message from another peer
        
        Args:
            round_id: Consensus round ID
            vote_message: Vote message to process
            
        Returns:
            True if vote was accepted, False otherwise
        """
        consensus_round = self.active_rounds.get(round_id)
        if not consensus_round:
            return False
        
        # Verify message signature
        if not self.verify_consensus_message(vote_message):
            return False
        
        # Check if sender is authorized to vote
        if vote_message.sender_id not in consensus_round.participants:
            return False
        
        # Add vote to round
        consensus_round.votes[vote_message.sender_id] = vote_message
        
        # Check if consensus achieved
        await self._check_consensus_completion(round_id)
        
        return True
    
    async def _check_consensus_completion(self, round_id: str):
        """Check if consensus has been achieved for a round"""
        consensus_round = self.active_rounds.get(round_id)
        if not consensus_round or consensus_round.consensus_achieved:
            return
        
        # Count votes
        yes_votes = 0
        no_votes = 0
        
        for vote_message in consensus_round.votes.values():
            vote_content = vote_message.content
            if vote_content.get("vote") is True:
                yes_votes += 1
            elif vote_content.get("vote") is False:
                no_votes += 1
        
        total_votes = len(consensus_round.votes)
        
        # Check if we have enough votes for consensus
        if total_votes >= consensus_round.required_votes:
            # Simple majority consensus
            if yes_votes > no_votes:
                consensus_round.consensus_achieved = True
                consensus_round.result = {
                    "consensus": True,
                    "yes_votes": yes_votes,
                    "no_votes": no_votes,
                    "total_votes": total_votes
                }
            else:
                consensus_round.consensus_achieved = True
                consensus_round.result = {
                    "consensus": False,
                    "yes_votes": yes_votes,
                    "no_votes": no_votes,
                    "total_votes": total_votes
                }
            
            consensus_round.end_time = datetime.now(timezone.utc)
            
            # Update metrics
            round_time = (consensus_round.end_time - consensus_round.start_time).total_seconds()
            self.consensus_metrics["rounds_completed"] += 1
            self.consensus_metrics["total_consensus_time"] += round_time
            self.consensus_metrics["average_round_time"] = (
                self.consensus_metrics["total_consensus_time"] / 
                self.consensus_metrics["rounds_completed"]
            )
    
    def get_consensus_result(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get consensus result for a round"""
        consensus_round = self.active_rounds.get(round_id)
        if consensus_round and consensus_round.consensus_achieved:
            return consensus_round.result
        return None
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get consensus performance metrics"""
        return {
            **self.consensus_metrics,
            "node_id": self.node_id,
            "security_level": self.security_level.value,
            "byzantine_threshold": self.byzantine_threshold,
            "registered_peers": len(self.peer_public_keys),
            "active_rounds": len(self.active_rounds)
        }
    
    def cleanup_completed_rounds(self, max_age_minutes: int = 60):
        """Clean up old completed consensus rounds"""
        cutoff_time = datetime.now(timezone.utc).replace(
            minute=datetime.now(timezone.utc).minute - max_age_minutes
        )
        
        rounds_to_remove = []
        for round_id, consensus_round in self.active_rounds.items():
            if (consensus_round.consensus_achieved and 
                consensus_round.end_time and 
                consensus_round.end_time < cutoff_time):
                rounds_to_remove.append(round_id)
        
        for round_id in rounds_to_remove:
            del self.active_rounds[round_id]


# Example usage and testing
async def example_post_quantum_consensus():
    """Example post-quantum consensus between three nodes"""
    print("🔐 PRSM Post-Quantum Consensus Example")
    print("=" * 50)
    
    # Create three consensus nodes
    nodes = []
    for i in range(3):
        node = PostQuantumConsensusEngine(
            node_id=f"node_{i}",
            security_level=SecurityLevel.LEVEL_1
        )
        nodes.append(node)
    
    # Exchange public keys between nodes
    print("1. Exchanging post-quantum public keys...")
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i != j:
                node.register_peer_public_key(
                    other_node.node_id,
                    other_node.node_keypair.public_key
                )
    print(f"   ✅ {len(nodes)} nodes registered with each other")
    
    # Node 0 proposes consensus
    print("\n2. Initiating consensus proposal...")
    proposer = nodes[0]
    participants = [node.node_id for node in nodes]
    
    proposal_content = {
        "proposal_type": "model_validation",
        "model_id": "test_model_123",
        "validation_score": 0.95,
        "proposed_by": proposer.node_id
    }
    
    round_id = await proposer.propose_consensus(proposal_content, participants)
    print(f"   ✅ Consensus round started: {round_id}")
    print(f"   Proposal: {proposal_content['proposal_type']}")
    
    # Share the consensus round with all nodes (in real implementation, this would be via network)
    proposal_message = proposer.active_rounds[round_id].proposal
    for node in nodes[1:]:  # Skip proposer
        # Create consensus round on each node
        consensus_round = ConsensusRound(
            round_id=round_id,
            round_number=proposer.current_round_number,
            participants=set(participants),
            required_votes=len(participants) - int(len(participants) * node.byzantine_threshold)
        )
        consensus_round.proposal = proposal_message
        node.active_rounds[round_id] = consensus_round
        node.current_round_number = proposer.current_round_number
    
    # Each node votes
    print("\n3. Collecting votes...")
    votes = []
    
    for i, node in enumerate(nodes):
        # Simulate different voting patterns
        vote_decision = True if i < 2 else False  # 2 yes, 1 no
        vote_content = {
            "vote": vote_decision,
            "reason": f"Node {i} analysis",
            "confidence": 0.8 + (i * 0.1)
        }
        
        vote_message = await node.vote_on_proposal(round_id, vote_content)
        votes.append(vote_message)
        print(f"   Node {i}: {'YES' if vote_decision else 'NO'} (confidence: {vote_content['confidence']})")
    
    # Process votes on proposer
    print("\n4. Processing votes...")
    for vote_message in votes:
        if vote_message.sender_id != proposer.node_id:  # Don't process own vote
            success = await proposer.process_vote(round_id, vote_message)
            print(f"   Processed vote from {vote_message.sender_id}: {'✅' if success else '❌'}")
    
    # Check consensus result
    print("\n5. Consensus result:")
    result = proposer.get_consensus_result(round_id)
    if result:
        print(f"   Consensus achieved: {'YES' if result['consensus'] else 'NO'}")
        print(f"   Yes votes: {result['yes_votes']}")
        print(f"   No votes: {result['no_votes']}")
        print(f"   Total votes: {result['total_votes']}")
    else:
        print("   ❌ Consensus not yet achieved")
    
    # Display metrics
    print("\n6. Performance metrics:")
    metrics = proposer.get_consensus_metrics()
    print(f"   Rounds completed: {metrics['rounds_completed']}")
    print(f"   Average round time: {metrics['average_round_time']:.2f}s")
    print(f"   Signature verification time: {metrics['signature_verification_time']*1000:.2f}ms")
    print(f"   Security level: {metrics['security_level']}")
    
    print("\n✅ Post-quantum consensus demonstration complete!")


if __name__ == "__main__":
    asyncio.run(example_post_quantum_consensus())