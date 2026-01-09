"""
Consensus Fault Injection Testing for PRSM
Implements comprehensive fault injection scenarios for consensus resilience testing
"""

import asyncio
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from uuid import UUID, uuid4
from enum import Enum
import statistics

from prsm.core.config import settings
from prsm.core.models import PeerNode, AgentResponse, SafetyFlag, SafetyLevel
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from .consensus import DistributedConsensus, ConsensusResult, ConsensusType


# === Fault Injection Configuration ===

# Fault injection settings
ENABLE_FAULT_INJECTION = getattr(settings, "PRSM_FAULT_INJECTION", False)
FAULT_INJECTION_RATE = float(getattr(settings, "PRSM_FAULT_INJECTION_RATE", 0.1))  # 10%
MAX_CONCURRENT_FAULTS = int(getattr(settings, "PRSM_MAX_CONCURRENT_FAULTS", 3))
FAULT_DURATION_SECONDS = int(getattr(settings, "PRSM_FAULT_DURATION", 30))

# Byzantine behavior settings
BYZANTINE_NODE_RATIO = float(getattr(settings, "PRSM_BYZANTINE_RATIO", 0.2))  # 20%
BYZANTINE_BEHAVIOR_PROBABILITY = float(getattr(settings, "PRSM_BYZANTINE_PROBABILITY", 0.3))  # 30%

# Network partition settings
PARTITION_PROBABILITY = float(getattr(settings, "PRSM_PARTITION_PROBABILITY", 0.05))  # 5%
PARTITION_DURATION = int(getattr(settings, "PRSM_PARTITION_DURATION", 45))  # seconds


class FaultType(Enum):
    """Types of faults that can be injected into consensus"""
    NODE_CRASH = "node_crash"                 # Node becomes unresponsive
    NODE_SLOW = "node_slow"                   # Node responds slowly
    BYZANTINE_BEHAVIOR = "byzantine_behavior" # Node sends conflicting messages
    NETWORK_PARTITION = "network_partition"   # Network splits into partitions
    MESSAGE_LOSS = "message_loss"             # Messages are dropped
    MESSAGE_DELAY = "message_delay"           # Messages are delayed
    MEMORY_PRESSURE = "memory_pressure"       # High memory usage
    CPU_OVERLOAD = "cpu_overload"            # High CPU usage
    CONSENSUS_TIMEOUT = "consensus_timeout"   # Consensus operations timeout
    INCONSISTENT_STATE = "inconsistent_state" # Node has inconsistent state


class FaultSeverity(Enum):
    """Severity levels for fault injection"""
    LOW = "low"           # Minor impact, should be recoverable
    MEDIUM = "medium"     # Moderate impact, may affect performance
    HIGH = "high"         # Major impact, may cause consensus failure
    CRITICAL = "critical" # Severe impact, likely to cause system failure


class FaultScenario:
    """Represents a fault injection scenario"""
    
    def __init__(self, name: str, description: str, fault_type: FaultType, 
                 severity: FaultSeverity, target_nodes: List[str],
                 duration: int = 30, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.fault_type = fault_type
        self.severity = severity
        self.target_nodes = target_nodes
        self.duration = duration
        self.parameters = parameters or {}
        
        self.scenario_id = str(uuid4())
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.active = False
        
        # Results tracking
        self.consensus_attempts = 0
        self.consensus_successes = 0
        self.consensus_failures = 0
        self.recovery_time: Optional[float] = None
        self.impact_metrics: Dict[str, Any] = {}


class FaultInjector:
    """Comprehensive fault injection system for consensus testing"""
    
    def __init__(self):
        # Fault management
        self.active_faults: Dict[str, FaultScenario] = {}
        self.fault_history: List[FaultScenario] = []
        self.fault_queue: deque = deque()
        
        # Node state tracking
        self.node_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.byzantine_nodes: Set[str] = set()
        self.partitioned_nodes: Dict[str, Set[str]] = {}  # partition_id -> nodes
        
        # Network simulation
        self.message_delays: Dict[Tuple[str, str], float] = {}
        self.message_drop_rates: Dict[Tuple[str, str], float] = {}
        self.network_partitions: List[Set[str]] = []
        
        # Performance tracking
        self.fault_metrics: Dict[str, Any] = {
            "total_faults_injected": 0,
            "active_fault_count": 0,
            "consensus_success_rate": 0.0,
            "average_recovery_time": 0.0,
            "fault_type_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int)
        }
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Injection control
        self.injection_active = False
        self.injection_rate = FAULT_INJECTION_RATE
        self.last_injection = datetime.now(timezone.utc)
    
    async def initialize_fault_injection(self, peer_nodes: List[PeerNode]) -> bool:
        """Initialize fault injection system with peer nodes"""
        try:
            print(f"🔧 Initializing fault injection system with {len(peer_nodes)} nodes")
            
            # Initialize node states
            for node in peer_nodes:
                self.node_states[node.peer_id] = {
                    "status": "healthy",
                    "last_seen": datetime.now(timezone.utc),
                    "response_time": 0.1,
                    "error_count": 0,
                    "reputation": node.reputation_score,
                    "active_faults": []
                }
            
            # Identify potential byzantine nodes (lowest reputation)
            sorted_nodes = sorted(peer_nodes, key=lambda n: n.reputation_score)
            byzantine_count = max(1, int(len(peer_nodes) * BYZANTINE_NODE_RATIO))
            self.byzantine_nodes = {node.peer_id for node in sorted_nodes[:byzantine_count]}
            
            print(f"✅ Fault injection initialized:")
            print(f"   - Monitoring {len(peer_nodes)} nodes")
            print(f"   - Potential byzantine nodes: {len(self.byzantine_nodes)}")
            print(f"   - Fault injection rate: {self.injection_rate:.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing fault injection: {e}")
            return False
    
    async def inject_fault_scenario(self, scenario: FaultScenario) -> bool:
        """Inject a specific fault scenario"""
        try:
            print(f"💉 Injecting fault: {scenario.name}")
            print(f"   - Type: {scenario.fault_type.value}")
            print(f"   - Severity: {scenario.severity.value}")
            print(f"   - Target nodes: {len(scenario.target_nodes)}")
            print(f"   - Duration: {scenario.duration}s")
            
            # Validate target nodes
            valid_targets = [node for node in scenario.target_nodes 
                           if node in self.node_states]
            
            if not valid_targets:
                print(f"❌ No valid target nodes for fault injection")
                return False
            
            scenario.target_nodes = valid_targets
            scenario.start_time = datetime.now(timezone.utc)
            scenario.active = True
            
            # Apply fault based on type
            success = await self._apply_fault(scenario)
            
            if success:
                self.active_faults[scenario.scenario_id] = scenario
                self.fault_metrics["total_faults_injected"] += 1
                self.fault_metrics["active_fault_count"] = len(self.active_faults)
                self.fault_metrics["fault_type_distribution"][scenario.fault_type.value] += 1
                self.fault_metrics["severity_distribution"][scenario.severity.value] += 1
                
                # Schedule fault recovery
                asyncio.create_task(self._schedule_fault_recovery(scenario))
                
                print(f"✅ Fault injection successful: {scenario.name}")
                return True
            else:
                print(f"❌ Fault injection failed: {scenario.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error injecting fault: {e}")
            return False
    
    async def _apply_fault(self, scenario: FaultScenario) -> bool:
        """Apply specific fault type to target nodes"""
        try:
            if scenario.fault_type == FaultType.NODE_CRASH:
                return await self._inject_node_crash(scenario)
            elif scenario.fault_type == FaultType.NODE_SLOW:
                return await self._inject_node_slow(scenario)
            elif scenario.fault_type == FaultType.BYZANTINE_BEHAVIOR:
                return await self._inject_byzantine_behavior(scenario)
            elif scenario.fault_type == FaultType.NETWORK_PARTITION:
                return await self._inject_network_partition(scenario)
            elif scenario.fault_type == FaultType.MESSAGE_LOSS:
                return await self._inject_message_loss(scenario)
            elif scenario.fault_type == FaultType.MESSAGE_DELAY:
                return await self._inject_message_delay(scenario)
            elif scenario.fault_type == FaultType.MEMORY_PRESSURE:
                return await self._inject_memory_pressure(scenario)
            elif scenario.fault_type == FaultType.CPU_OVERLOAD:
                return await self._inject_cpu_overload(scenario)
            elif scenario.fault_type == FaultType.CONSENSUS_TIMEOUT:
                return await self._inject_consensus_timeout(scenario)
            elif scenario.fault_type == FaultType.INCONSISTENT_STATE:
                return await self._inject_inconsistent_state(scenario)
            else:
                print(f"⚠️ Unknown fault type: {scenario.fault_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error applying fault {scenario.fault_type}: {e}")
            return False
    
    async def _inject_node_crash(self, scenario: FaultScenario) -> bool:
        """Simulate node crash - node becomes completely unresponsive"""
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "crashed"
            self.node_states[node_id]["response_time"] = float('inf')
            self.node_states[node_id]["active_faults"].append("node_crash")
        
        print(f"   💥 Crashed {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_node_slow(self, scenario: FaultScenario) -> bool:
        """Simulate slow node - increased response times"""
        slowdown_factor = scenario.parameters.get("slowdown_factor", 5.0)
        
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "slow"
            current_time = self.node_states[node_id]["response_time"]
            self.node_states[node_id]["response_time"] = current_time * slowdown_factor
            self.node_states[node_id]["active_faults"].append("node_slow")
        
        print(f"   🐌 Slowed down {len(scenario.target_nodes)} nodes by {slowdown_factor}x")
        return True
    
    async def _inject_byzantine_behavior(self, scenario: FaultScenario) -> bool:
        """Simulate byzantine behavior - conflicting/malicious responses"""
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "byzantine"
            self.node_states[node_id]["active_faults"].append("byzantine_behavior")
            self.byzantine_nodes.add(node_id)
        
        print(f"   🎭 Activated byzantine behavior in {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_network_partition(self, scenario: FaultScenario) -> bool:
        """Simulate network partition - split network into isolated groups"""
        partition_size = scenario.parameters.get("partition_size", len(scenario.target_nodes) // 2)
        
        # Create two partitions
        partition_1 = set(scenario.target_nodes[:partition_size])
        partition_2 = set(scenario.target_nodes[partition_size:])
        
        partition_id = str(uuid4())[:8]
        self.partitioned_nodes[f"partition_1_{partition_id}"] = partition_1
        self.partitioned_nodes[f"partition_2_{partition_id}"] = partition_2
        
        # Update node states
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "partitioned"
            self.node_states[node_id]["active_faults"].append("network_partition")
        
        print(f"   🌐 Created network partition: {len(partition_1)} | {len(partition_2)} nodes")
        return True
    
    async def _inject_message_loss(self, scenario: FaultScenario) -> bool:
        """Simulate message loss - drop percentage of messages"""
        drop_rate = scenario.parameters.get("drop_rate", 0.3)  # 30% message loss
        
        for node_id in scenario.target_nodes:
            # Apply message loss to all connections involving this node
            for other_node in self.node_states.keys():
                if other_node != node_id:
                    edge = (min(node_id, other_node), max(node_id, other_node))
                    self.message_drop_rates[edge] = drop_rate
            
            self.node_states[node_id]["active_faults"].append("message_loss")
        
        print(f"   📦 Applied {drop_rate:.1%} message loss to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_message_delay(self, scenario: FaultScenario) -> bool:
        """Simulate message delays - add latency to communications"""
        delay_ms = scenario.parameters.get("delay_ms", 500)  # 500ms delay
        
        for node_id in scenario.target_nodes:
            # Apply delays to all connections involving this node
            for other_node in self.node_states.keys():
                if other_node != node_id:
                    edge = (min(node_id, other_node), max(node_id, other_node))
                    self.message_delays[edge] = delay_ms / 1000.0  # Convert to seconds
            
            self.node_states[node_id]["active_faults"].append("message_delay")
        
        print(f"   ⏱️ Added {delay_ms}ms message delay to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_memory_pressure(self, scenario: FaultScenario) -> bool:
        """Simulate memory pressure - high memory usage affecting performance"""
        memory_load = scenario.parameters.get("memory_load", 0.9)  # 90% memory usage
        
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "memory_pressure"
            self.node_states[node_id]["active_faults"].append("memory_pressure")
            # Increase response time due to memory pressure
            self.node_states[node_id]["response_time"] *= (1 + memory_load)
        
        print(f"   🧠 Applied {memory_load:.1%} memory pressure to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_cpu_overload(self, scenario: FaultScenario) -> bool:
        """Simulate CPU overload - high CPU usage affecting performance"""
        cpu_load = scenario.parameters.get("cpu_load", 0.95)  # 95% CPU usage
        
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "cpu_overload"
            self.node_states[node_id]["active_faults"].append("cpu_overload")
            # Significantly increase response time due to CPU overload
            self.node_states[node_id]["response_time"] *= (1 + cpu_load * 2)
        
        print(f"   💻 Applied {cpu_load:.1%} CPU overload to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_consensus_timeout(self, scenario: FaultScenario) -> bool:
        """Simulate consensus timeouts - consensus operations timeout frequently"""
        timeout_rate = scenario.parameters.get("timeout_rate", 0.5)  # 50% timeout rate
        
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "timeout_prone"
            self.node_states[node_id]["active_faults"].append("consensus_timeout")
            self.node_states[node_id]["timeout_rate"] = timeout_rate
        
        print(f"   ⏰ Applied {timeout_rate:.1%} consensus timeout rate to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _inject_inconsistent_state(self, scenario: FaultScenario) -> bool:
        """Simulate inconsistent state - node has outdated or conflicting state"""
        for node_id in scenario.target_nodes:
            self.node_states[node_id]["status"] = "inconsistent"
            self.node_states[node_id]["active_faults"].append("inconsistent_state")
            self.node_states[node_id]["state_lag"] = scenario.parameters.get("state_lag", 5)
        
        print(f"   🔄 Applied inconsistent state to {len(scenario.target_nodes)} nodes")
        return True
    
    async def _schedule_fault_recovery(self, scenario: FaultScenario):
        """Schedule automatic recovery from fault"""
        try:
            await asyncio.sleep(scenario.duration)
            await self.recover_fault(scenario.scenario_id)
        except Exception as e:
            print(f"❌ Error in fault recovery scheduling: {e}")
    
    async def recover_fault(self, scenario_id: str) -> bool:
        """Recover from a specific fault scenario"""
        try:
            if scenario_id not in self.active_faults:
                return False
            
            scenario = self.active_faults[scenario_id]
            print(f"🔄 Recovering from fault: {scenario.name}")
            
            # Recovery based on fault type
            success = await self._apply_fault_recovery(scenario)
            
            if success:
                scenario.end_time = datetime.now(timezone.utc)
                scenario.active = False
                
                # Calculate recovery metrics
                if scenario.start_time:
                    scenario.recovery_time = (scenario.end_time - scenario.start_time).total_seconds()
                
                # Move to history
                self.fault_history.append(scenario)
                del self.active_faults[scenario_id]
                
                self.fault_metrics["active_fault_count"] = len(self.active_faults)
                
                print(f"✅ Fault recovery successful: {scenario.name}")
                return True
            else:
                print(f"❌ Fault recovery failed: {scenario.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error recovering fault: {e}")
            return False
    
    async def _apply_fault_recovery(self, scenario: FaultScenario) -> bool:
        """Apply recovery actions for specific fault type"""
        try:
            # Remove fault from node states
            for node_id in scenario.target_nodes:
                if node_id in self.node_states:
                    # Remove fault from active faults list
                    fault_name = scenario.fault_type.value
                    if fault_name in self.node_states[node_id].get("active_faults", []):
                        self.node_states[node_id]["active_faults"].remove(fault_name)
                    
                    # Reset node state if no other faults
                    if not self.node_states[node_id].get("active_faults", []):
                        self.node_states[node_id]["status"] = "healthy"
                        self.node_states[node_id]["response_time"] = 0.1  # Reset to normal
                        
                        # Remove byzantine status
                        if node_id in self.byzantine_nodes:
                            self.byzantine_nodes.discard(node_id)
            
            # Clear network-level effects
            if scenario.fault_type == FaultType.NETWORK_PARTITION:
                # Remove partitions involving target nodes
                partitions_to_remove = []
                for partition_id, nodes in self.partitioned_nodes.items():
                    if any(node in scenario.target_nodes for node in nodes):
                        partitions_to_remove.append(partition_id)
                
                for partition_id in partitions_to_remove:
                    del self.partitioned_nodes[partition_id]
            
            elif scenario.fault_type in [FaultType.MESSAGE_LOSS, FaultType.MESSAGE_DELAY]:
                # Clear message effects
                edges_to_clear = []
                for edge in self.message_delays.keys():
                    if any(node in scenario.target_nodes for node in edge):
                        edges_to_clear.append(edge)
                
                for edge in edges_to_clear:
                    self.message_delays.pop(edge, None)
                    self.message_drop_rates.pop(edge, None)
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying fault recovery: {e}")
            return False
    
    def simulate_consensus_under_faults(self, proposal: Dict[str, Any], 
                                      participating_nodes: List[str]) -> ConsensusResult:
        """Simulate consensus execution under current fault conditions"""
        try:
            start_time = time.time()
            
            # Check node availability
            available_nodes = []
            for node_id in participating_nodes:
                node_state = self.node_states.get(node_id, {})
                status = node_state.get("status", "healthy")
                
                if status == "crashed":
                    continue  # Node is completely unavailable
                elif status in ["slow", "memory_pressure", "cpu_overload"]:
                    # Node is slow but available
                    available_nodes.append(node_id)
                elif status == "byzantine":
                    # Byzantine node may participate but with malicious behavior
                    if random.random() > BYZANTINE_BEHAVIOR_PROBABILITY:
                        available_nodes.append(node_id)
                else:
                    # Healthy node
                    available_nodes.append(node_id)
            
            # Check for network partitions
            largest_partition = self._get_largest_connected_partition(available_nodes)
            consensus_nodes = largest_partition
            
            # Simulate consensus process
            consensus_achieved = False
            execution_time = 0.0
            agreement_ratio = 0.0
            
            if len(consensus_nodes) >= len(participating_nodes) * 0.5:  # Majority available
                # Calculate consensus probability based on faults
                success_probability = self._calculate_consensus_probability(consensus_nodes)
                
                # Simulate execution time with faults
                base_execution_time = 1.0  # 1 second base time
                fault_delay = self._calculate_fault_delay(consensus_nodes)
                execution_time = base_execution_time + fault_delay
                
                # Determine consensus success
                if random.random() < success_probability:
                    consensus_achieved = True
                    agreement_ratio = len(consensus_nodes) / len(participating_nodes)
                else:
                    consensus_achieved = False
                    agreement_ratio = 0.0
            
            # Update scenario metrics
            for scenario in self.active_faults.values():
                scenario.consensus_attempts += 1
                if consensus_achieved:
                    scenario.consensus_successes += 1
                else:
                    scenario.consensus_failures += 1
            
            return ConsensusResult(
                agreed_value=proposal if consensus_achieved else None,
                consensus_achieved=consensus_achieved,
                consensus_type="fault_injected",
                agreement_ratio=agreement_ratio,
                participating_peers=consensus_nodes,
                execution_time=execution_time
            )
            
        except Exception as e:
            print(f"❌ Error simulating consensus under faults: {e}")
            return ConsensusResult(
                consensus_achieved=False,
                consensus_type="fault_simulation_error",
                execution_time=time.time() - start_time
            )
    
    def _get_largest_connected_partition(self, nodes: List[str]) -> List[str]:
        """Get the largest connected partition of nodes"""
        if not self.partitioned_nodes:
            return nodes  # No partitions, all nodes connected
        
        # Find which partition each node belongs to
        node_partitions = {}
        for partition_id, partition_nodes in self.partitioned_nodes.items():
            for node in nodes:
                if node in partition_nodes:
                    node_partitions[node] = partition_id
        
        # Group nodes by partition
        partition_groups = defaultdict(list)
        for node in nodes:
            partition = node_partitions.get(node, "main")
            partition_groups[partition].append(node)
        
        # Return largest partition
        if partition_groups:
            return max(partition_groups.values(), key=len)
        else:
            return nodes
    
    def _calculate_consensus_probability(self, nodes: List[str]) -> float:
        """Calculate consensus success probability based on current faults"""
        if not nodes:
            return 0.0
        
        base_probability = 0.9  # 90% base success rate
        
        # Reduce probability based on active faults
        for node_id in nodes:
            node_state = self.node_states.get(node_id, {})
            active_faults = node_state.get("active_faults", [])
            
            for fault in active_faults:
                if fault == "byzantine_behavior":
                    base_probability *= 0.7  # 30% reduction per byzantine node
                elif fault in ["memory_pressure", "cpu_overload"]:
                    base_probability *= 0.8  # 20% reduction
                elif fault == "consensus_timeout":
                    timeout_rate = node_state.get("timeout_rate", 0.1)
                    base_probability *= (1.0 - timeout_rate)
                elif fault == "inconsistent_state":
                    base_probability *= 0.6  # 40% reduction
        
        return max(0.1, min(1.0, base_probability))  # Clamp between 10% and 100%
    
    def _calculate_fault_delay(self, nodes: List[str]) -> float:
        """Calculate additional delay caused by faults"""
        total_delay = 0.0
        
        for node_id in nodes:
            node_state = self.node_states.get(node_id, {})
            response_time = node_state.get("response_time", 0.1)
            
            # Add response time delay
            total_delay += max(0, response_time - 0.1)  # Subtract base response time
            
            # Add network delays
            for other_node in nodes:
                if other_node != node_id:
                    edge = (min(node_id, other_node), max(node_id, other_node))
                    delay = self.message_delays.get(edge, 0.0)
                    total_delay += delay
        
        return total_delay / len(nodes) if nodes else 0.0  # Average delay
    
    async def create_fault_scenarios(self, peer_nodes: List[PeerNode]) -> List[FaultScenario]:
        """Create comprehensive fault scenarios for testing"""
        scenarios = []
        node_ids = [node.peer_id for node in peer_nodes]
        
        # 1. Single node crash
        scenarios.append(FaultScenario(
            name="Single Node Crash",
            description="Single node becomes completely unresponsive",
            fault_type=FaultType.NODE_CRASH,
            severity=FaultSeverity.MEDIUM,
            target_nodes=[node_ids[0]],
            duration=30
        ))
        
        # 2. Multiple node slowdown
        scenarios.append(FaultScenario(
            name="Multiple Node Slowdown",
            description="Multiple nodes respond slowly",
            fault_type=FaultType.NODE_SLOW,
            severity=FaultSeverity.MEDIUM,
            target_nodes=node_ids[:min(3, len(node_ids))],
            duration=45,
            parameters={"slowdown_factor": 3.0}
        ))
        
        # 3. Byzantine behavior
        byzantine_count = max(1, len(node_ids) // 4)  # 25% of nodes
        scenarios.append(FaultScenario(
            name="Byzantine Node Attack",
            description="Nodes exhibit malicious byzantine behavior",
            fault_type=FaultType.BYZANTINE_BEHAVIOR,
            severity=FaultSeverity.HIGH,
            target_nodes=node_ids[:byzantine_count],
            duration=60
        ))
        
        # 4. Network partition
        if len(node_ids) >= 4:
            scenarios.append(FaultScenario(
                name="Network Partition",
                description="Network splits into two partitions",
                fault_type=FaultType.NETWORK_PARTITION,
                severity=FaultSeverity.HIGH,
                target_nodes=node_ids,
                duration=60,
                parameters={"partition_size": len(node_ids) // 2}
            ))
        
        # 5. Message loss
        scenarios.append(FaultScenario(
            name="High Message Loss",
            description="Significant message loss between nodes",
            fault_type=FaultType.MESSAGE_LOSS,
            severity=FaultSeverity.MEDIUM,
            target_nodes=node_ids[:min(2, len(node_ids))],
            duration=40,
            parameters={"drop_rate": 0.4}
        ))
        
        # 6. CPU overload
        scenarios.append(FaultScenario(
            name="CPU Overload",
            description="Nodes experience high CPU load",
            fault_type=FaultType.CPU_OVERLOAD,
            severity=FaultSeverity.MEDIUM,
            target_nodes=node_ids[:min(2, len(node_ids))],
            duration=35,
            parameters={"cpu_load": 0.9}
        ))
        
        # 7. Consensus timeouts
        scenarios.append(FaultScenario(
            name="Consensus Timeouts",
            description="Frequent consensus operation timeouts",
            fault_type=FaultType.CONSENSUS_TIMEOUT,
            severity=FaultSeverity.HIGH,
            target_nodes=node_ids[:min(2, len(node_ids))],
            duration=50,
            parameters={"timeout_rate": 0.6}
        ))
        
        return scenarios
    
    async def get_fault_metrics(self) -> Dict[str, Any]:
        """Get comprehensive fault injection metrics"""
        try:
            # Calculate success rates
            total_attempts = sum(s.consensus_attempts for s in self.fault_history)
            total_successes = sum(s.consensus_successes for s in self.fault_history)
            
            success_rate = total_successes / max(1, total_attempts)
            
            # Calculate average recovery time
            recovery_times = [s.recovery_time for s in self.fault_history if s.recovery_time]
            avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0
            
            # Current fault status
            active_fault_types = [f.fault_type.value for f in self.active_faults.values()]
            active_severities = [f.severity.value for f in self.active_faults.values()]
            
            return {
                **self.fault_metrics,
                "consensus_success_rate": success_rate,
                "average_recovery_time": avg_recovery_time,
                "total_scenarios_completed": len(self.fault_history),
                "current_active_faults": len(self.active_faults),
                "active_fault_types": active_fault_types,
                "active_severities": active_severities,
                "byzantine_node_count": len(self.byzantine_nodes),
                "network_partitions": len(self.partitioned_nodes),
                "nodes_under_faults": len([node for node, state in self.node_states.items() 
                                        if state.get("active_faults", [])]),
                "injection_active": self.injection_active,
                "injection_rate": self.injection_rate
            }
            
        except Exception as e:
            print(f"❌ Error getting fault metrics: {e}")
            return self.fault_metrics


# === Global Fault Injection Instance ===

_fault_injector_instance: Optional[FaultInjector] = None

def get_fault_injector() -> FaultInjector:
    """Get or create the global fault injector instance"""
    global _fault_injector_instance
    if _fault_injector_instance is None:
        _fault_injector_instance = FaultInjector()
    return _fault_injector_instance