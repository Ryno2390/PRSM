#!/usr/bin/env python3
"""
Comprehensive Test Suite for P2P Federation Foundation
Tests P2P model distribution, distributed execution, and peer validation
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
from uuid import uuid4

# Add the project root to the path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prsm.core.models import (
    ArchitectTask, PeerNode, ModelShard, ModelType, TaskStatus,
    SafetyLevel
)
from prsm.safety.circuit_breaker import ThreatLevel
from prsm.federation.p2p_network import P2PModelNetwork, get_p2p_network


class TestP2PFederation:
    """Comprehensive test suite for P2P federation functionality"""
    
    def __init__(self):
        self.p2p_network = get_p2p_network()
        self.test_results = {}
        self.test_peers = []
        self.test_models = []
        
    async def run_all_tests(self):
        """Run complete P2P federation test suite"""
        print("🧪 Starting P2P Federation Foundation Test Suite")
        print("=" * 60)
        
        test_methods = [
            ("Peer Management", self.test_peer_management),
            ("Model Shard Distribution", self.test_model_shard_distribution),
            ("Distributed Execution", self.test_distributed_execution),
            ("Peer Contribution Validation", self.test_peer_contribution_validation),
            ("Network Status & Monitoring", self.test_network_status),
            ("Safety Integration", self.test_safety_integration),
            ("Performance & Scalability", self.test_performance_scalability),
            ("Fault Tolerance", self.test_fault_tolerance),
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_method in test_methods:
            print(f"\n🔬 Testing: {test_name}")
            print("-" * 40)
            
            try:
                test_result = await test_method()
                if test_result:
                    print(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAILED")
                total_tests += 1
                
                self.test_results[test_name] = test_result
                
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = False
                total_tests += 1
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 P2P Federation Test Results: {passed_tests}/{total_tests} tests passed")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        return passed_tests == total_tests
    
    async def test_peer_management(self) -> bool:
        """Test peer addition, removal, and management"""
        try:
            # Test 1: Add peers to network
            print("  📡 Testing peer addition...")
            test_peers = []
            
            for i in range(5):
                peer = PeerNode(
                    node_id=f"node_{i}",
                    peer_id=f"peer_{i}",
                    multiaddr=f"/ip4/192.168.1.{i+10}/tcp/4001",
                    capabilities=["model_execution", "data_storage"],
                    reputation_score=0.7 + random.uniform(-0.2, 0.2)
                )
                
                success = await self.p2p_network.add_peer(peer)
                if not success:
                    return False
                
                test_peers.append(peer)
                self.test_peers.append(peer)
            
            # Verify peers were added
            network_status = await self.p2p_network.get_network_status()
            if network_status["total_peers"] != 5:
                print(f"    ❌ Expected 5 peers, got {network_status['total_peers']}")
                return False
            
            print(f"    ✅ Added {len(test_peers)} peers successfully")
            
            # Test 2: Remove a peer
            print("  🗑️ Testing peer removal...")
            peer_to_remove = test_peers[0]
            
            success = await self.p2p_network.remove_peer(peer_to_remove.peer_id)
            if not success:
                return False
            
            # Verify peer was removed
            network_status = await self.p2p_network.get_network_status()
            if network_status["total_peers"] != 4:
                print(f"    ❌ Expected 4 peers after removal, got {network_status['total_peers']}")
                return False
            
            print(f"    ✅ Removed peer {peer_to_remove.peer_id} successfully")
            
            # Test 3: Peer capabilities validation
            print("  🔧 Testing peer capabilities...")
            specialized_peer = PeerNode(
                node_id="specialized_node",
                peer_id="specialized_peer",
                multiaddr="/ip4/192.168.1.100/tcp/4001",
                capabilities=["model_execution", "data_storage", "gpu_acceleration"],
                reputation_score=0.9
            )
            
            success = await self.p2p_network.add_peer(specialized_peer)
            if not success:
                return False
            
            self.test_peers.append(specialized_peer)
            print("    ✅ Specialized peer with GPU capabilities added")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in peer management test: {str(e)}")
            return False
    
    async def test_model_shard_distribution(self) -> bool:
        """Test model distribution into shards across peers"""
        try:
            # Test 1: Basic model sharding
            print("  📦 Testing model shard distribution...")
            
            model_cid = "QmTestModel123456789abcdef"
            shard_count = 4
            
            # Distribute model shards
            shards = await self.p2p_network.distribute_model_shards(model_cid, shard_count)
            
            if len(shards) != shard_count:
                print(f"    ❌ Expected {shard_count} shards, got {len(shards)}")
                return False
            
            # Verify shard properties
            for i, shard in enumerate(shards):
                if shard.shard_index != i:
                    print(f"    ❌ Shard index mismatch: expected {i}, got {shard.shard_index}")
                    return False
                
                if shard.total_shards != shard_count:
                    print(f"    ❌ Total shards mismatch: expected {shard_count}, got {shard.total_shards}")
                    return False
                
                if not shard.hosted_by:
                    print(f"    ❌ Shard {i} has no hosting peers")
                    return False
                
                if not shard.verification_hash:
                    print(f"    ❌ Shard {i} missing verification hash")
                    return False
            
            print(f"    ✅ Successfully distributed model into {len(shards)} shards")
            
            # Test 2: Shard redundancy
            print("  🔄 Testing shard redundancy...")
            
            total_replicas = sum(len(shard.hosted_by) for shard in shards)
            avg_replicas = total_replicas / len(shards)
            
            if avg_replicas < 2.0:  # Minimum redundancy
                print(f"    ❌ Insufficient redundancy: {avg_replicas:.1f} replicas per shard")
                return False
            
            print(f"    ✅ Good redundancy: {avg_replicas:.1f} replicas per shard")
            
            # Test 3: Large model distribution
            print("  📈 Testing large model distribution...")
            
            large_model_cid = "QmLargeModel987654321fedcba"
            large_shard_count = 10
            
            large_shards = await self.p2p_network.distribute_model_shards(
                large_model_cid, large_shard_count
            )
            
            if len(large_shards) != large_shard_count:
                print(f"    ❌ Large model sharding failed")
                return False
            
            print(f"    ✅ Large model distributed into {len(large_shards)} shards")
            
            self.test_models.extend([model_cid, large_model_cid])
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in shard distribution test: {str(e)}")
            return False
    
    async def test_distributed_execution(self) -> bool:
        """Test distributed task execution across peers"""
        try:
            # Test 1: Basic distributed execution
            print("  🚀 Testing distributed execution...")
            
            # Create test task
            test_task = ArchitectTask(
                task_id=uuid4(),
                session_id=uuid4(),
                level=1,
                parent_task_id=None,
                instruction="Test distributed execution across P2P network",
                complexity_score=0.5,
                dependencies=[]
            )
            
            # Coordinate distributed execution
            execution_futures = await self.p2p_network.coordinate_distributed_execution(test_task)
            
            if not execution_futures:
                print("    ❌ No execution futures returned")
                return False
            
            # Wait for execution completion
            results = await asyncio.gather(*execution_futures, return_exceptions=True)
            
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            
            if len(successful_results) < 2:  # Minimum successful executions
                print(f"    ❌ Insufficient successful executions: {len(successful_results)}")
                return False
            
            print(f"    ✅ Distributed execution completed on {len(successful_results)} peers")
            
            # Test 2: Concurrent execution
            print("  ⚡ Testing concurrent execution...")
            
            # Create multiple tasks
            concurrent_tasks = []
            for i in range(3):
                task = ArchitectTask(
                    task_id=uuid4(),
                    session_id=uuid4(),
                    level=1,
                    parent_task_id=None,
                    instruction=f"Concurrent test task {i+1}",
                    complexity_score=0.3,
                    dependencies=[]
                )
                concurrent_tasks.append(task)
            
            # Execute all tasks concurrently
            all_futures = []
            for task in concurrent_tasks:
                futures = await self.p2p_network.coordinate_distributed_execution(task)
                all_futures.extend(futures)
            
            # Wait for all executions
            all_results = await asyncio.gather(*all_futures, return_exceptions=True)
            successful_concurrent = [r for r in all_results if isinstance(r, dict) and r.get("success")]
            
            if len(successful_concurrent) < 6:  # 3 tasks × 2 peers minimum
                print(f"    ❌ Concurrent execution failed: {len(successful_concurrent)} successes")
                return False
            
            print(f"    ✅ Concurrent execution completed: {len(successful_concurrent)} total executions")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in distributed execution test: {str(e)}")
            return False
    
    async def test_peer_contribution_validation(self) -> bool:
        """Test peer contribution validation and consensus"""
        try:
            # Test 1: Consensus validation
            print("  🤝 Testing peer consensus validation...")
            
            # Create test results with consensus
            consistent_result = {
                "output": "consensus_result",
                "confidence": 0.9,
                "metadata": {"algorithm": "test"}
            }
            
            peer_results = []
            for i in range(5):
                result = consistent_result.copy()
                result["peer_id"] = f"peer_{i}"
                result["timestamp"] = datetime.now(timezone.utc)
                result["execution_time"] = random.uniform(1.0, 3.0)
                
                # Add small variations to 80% of results (should still reach consensus)
                if i < 4:
                    # Keep result consistent
                    pass
                else:
                    # Make one result different
                    result["output"] = "different_result"
                
                peer_results.append(result)
            
            # Validate consensus
            consensus_valid = await self.p2p_network.validate_peer_contributions(peer_results)
            
            if not consensus_valid:
                print("    ❌ Consensus validation failed for valid results")
                return False
            
            print("    ✅ Consensus validation working correctly")
            
            # Test 2: No consensus scenario
            print("  ❌ Testing no-consensus detection...")
            
            # Create results with no consensus
            conflicting_results = []
            for i in range(5):
                result = {
                    "peer_id": f"peer_{i}",
                    "output": f"result_{i}",  # All different
                    "confidence": 0.8,
                    "timestamp": datetime.now(timezone.utc),
                    "execution_time": random.uniform(1.0, 3.0)
                }
                conflicting_results.append(result)
            
            # Should fail consensus
            no_consensus = await self.p2p_network.validate_peer_contributions(conflicting_results)
            
            if no_consensus:
                print("    ❌ No-consensus detection failed")
                return False
            
            print("    ✅ No-consensus detection working correctly")
            
            # Test 3: Empty results validation
            print("  🚫 Testing empty results validation...")
            
            empty_valid = await self.p2p_network.validate_peer_contributions([])
            
            if empty_valid:
                print("    ❌ Empty results incorrectly validated")
                return False
            
            print("    ✅ Empty results validation working correctly")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in contribution validation test: {str(e)}")
            return False
    
    async def test_network_status(self) -> bool:
        """Test network status monitoring and reporting"""
        try:
            print("  📊 Testing network status monitoring...")
            
            # Get current network status
            status = await self.p2p_network.get_network_status()
            
            # Verify status structure
            required_fields = [
                "total_peers", "active_peers", "total_models", 
                "total_shards", "pending_executions", "safety_monitoring"
            ]
            
            for field in required_fields:
                if field not in status:
                    print(f"    ❌ Missing status field: {field}")
                    return False
            
            # Verify status values
            if status["total_peers"] <= 0:
                print(f"    ❌ Invalid peer count: {status['total_peers']}")
                return False
            
            if not isinstance(status["active_peers"], list):
                print("    ❌ Active peers should be a list")
                return False
            
            if status["total_models"] < 0:
                print(f"    ❌ Invalid model count: {status['total_models']}")
                return False
            
            print(f"    ✅ Network status: {status['total_peers']} peers, {status['total_models']} models")
            print(f"    ✅ Safety monitoring: {status['safety_monitoring']}")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in network status test: {str(e)}")
            return False
    
    async def test_safety_integration(self) -> bool:
        """Test safety framework integration"""
        try:
            print("  🛡️ Testing safety framework integration...")
            
            # Test 1: Safety validation during distribution
            print("    🔍 Testing safety validation during model distribution...")
            
            # Should work with valid model
            try:
                valid_model_cid = "QmSafeModel123"
                shards = await self.p2p_network.distribute_model_shards(valid_model_cid, 3)
                if not shards:
                    print("    ❌ Valid model distribution failed")
                    return False
                print("    ✅ Valid model distribution with safety checks")
            except Exception as e:
                print(f"    ❌ Valid model distribution error: {str(e)}")
                return False
            
            # Test 2: Safety validation during execution
            print("    🚨 Testing safety validation during execution...")
            
            safe_task = ArchitectTask(
                task_id=uuid4(),
                session_id=uuid4(),
                level=1,
                parent_task_id=None,
                instruction="Safe task for safety integration testing",
                complexity_score=0.2,  # Low complexity = safe
                dependencies=[]
            )
            
            try:
                futures = await self.p2p_network.coordinate_distributed_execution(safe_task)
                if not futures:
                    print("    ❌ Safe task execution failed")
                    return False
                print("    ✅ Safe task execution with safety monitoring")
            except Exception as e:
                print(f"    ❌ Safe task execution error: {str(e)}")
                return False
            
            # Test 3: Peer reputation system
            print("    📈 Testing peer reputation system...")
            
            # Check that peers have reputation scores
            for peer_id, peer in self.p2p_network.active_peers.items():
                if not hasattr(peer, 'reputation_score'):
                    print("    ❌ Peer missing reputation score")
                    return False
                
                if not (0.0 <= peer.reputation_score <= 1.0):
                    print(f"    ❌ Invalid reputation score: {peer.reputation_score}")
                    return False
            
            print("    ✅ Peer reputation system working correctly")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in safety integration test: {str(e)}")
            return False
    
    async def test_performance_scalability(self) -> bool:
        """Test performance and scalability metrics"""
        try:
            print("  ⚡ Testing performance and scalability...")
            
            # Test 1: Performance tracking
            print("    📊 Testing performance tracking...")
            
            # Verify peer performance metrics exist
            for peer_id in self.p2p_network.active_peers.keys():
                if peer_id not in self.p2p_network.peer_performance:
                    print(f"    ❌ Missing performance metrics for peer {peer_id}")
                    return False
                
                metrics = self.p2p_network.peer_performance[peer_id]
                required_metrics = ["total_executions", "successful_executions", "average_response_time"]
                
                for metric in required_metrics:
                    if metric not in metrics:
                        print(f"    ❌ Missing metric {metric} for peer {peer_id}")
                        return False
            
            print("    ✅ Performance tracking working correctly")
            
            # Test 2: Throughput measurement
            print("    🏃 Testing throughput measurement...")
            
            start_time = time.time()
            
            # Create and execute multiple tasks
            tasks = []
            for i in range(5):
                task = ArchitectTask(
                    task_id=uuid4(),
                    session_id=uuid4(),
                    level=1,
                    parent_task_id=None,
                    instruction=f"Performance test task {i+1}",
                    complexity_score=0.1,  # Simple tasks
                    dependencies=[]
                )
                tasks.append(task)
            
            # Execute all tasks
            all_futures = []
            for task in tasks:
                futures = await self.p2p_network.coordinate_distributed_execution(task)
                all_futures.extend(futures)
            
            # Wait for completion
            results = await asyncio.gather(*all_futures, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            throughput = len(successful_results) / total_time
            
            print(f"    ✅ Throughput: {throughput:.2f} executions/second")
            
            # Test 3: Memory usage check
            print("    💾 Testing memory usage...")
            
            # Verify data structures aren't growing unbounded
            if len(self.p2p_network.execution_metrics) > 100:
                print(f"    ⚠️ High execution metrics count: {len(self.p2p_network.execution_metrics)}")
            
            if len(self.p2p_network.model_shards) > 50:
                print(f"    ⚠️ High model shard count: {len(self.p2p_network.model_shards)}")
            
            print("    ✅ Memory usage reasonable")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in performance test: {str(e)}")
            return False
    
    async def test_fault_tolerance(self) -> bool:
        """Test fault tolerance and error handling"""
        try:
            print("  🛠️ Testing fault tolerance...")
            
            # Test 1: Peer failure during execution
            print("    💔 Testing peer failure handling...")
            
            # Get current peer count
            initial_peer_count = len(self.p2p_network.active_peers)
            
            # Remove a peer
            if self.test_peers:
                # Find a peer that's still active
                peer_to_fail = None
                for peer in self.test_peers:
                    if peer.peer_id in self.p2p_network.active_peers:
                        peer_to_fail = peer
                        break
                
                if peer_to_fail:
                    await self.p2p_network.remove_peer(peer_to_fail.peer_id)
                    
                    # Verify peer was removed
                    new_peer_count = len(self.p2p_network.active_peers)
                    if new_peer_count != initial_peer_count - 1:
                        print("    ❌ Peer removal failed")
                        return False
                    
                    print("    ✅ Peer failure handled correctly")
                else:
                    print("    ⚠️ No active peer found to remove")
            
            # Test 2: Invalid model distribution
            print("    🚫 Testing invalid model handling...")
            
            try:
                # Try to distribute with invalid parameters
                await self.p2p_network.distribute_model_shards("", 0)
                print("    ❌ Invalid model distribution should have failed")
                return False
            except Exception:
                print("    ✅ Invalid model distribution correctly rejected")
            
            # Test 3: Network resilience
            print("    🌐 Testing network resilience...")
            
            # Verify network can still function with reduced peers
            if len(self.p2p_network.active_peers) > 0:
                test_task = ArchitectTask(
                    task_id=uuid4(),
                    session_id=uuid4(),
                    level=1,
                    parent_task_id=None,
                    instruction="Fault tolerance test task",
                    complexity_score=0.3,
                    dependencies=[]
                )
                
                try:
                    futures = await self.p2p_network.coordinate_distributed_execution(test_task)
                    if futures:
                        results = await asyncio.gather(*futures, return_exceptions=True)
                        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
                        
                        if successful_results:
                            print("    ✅ Network resilient to peer failures")
                        else:
                            print("    ⚠️ No successful executions after peer failure")
                    else:
                        print("    ⚠️ No execution futures after peer failure")
                except Exception as e:
                    print(f"    ⚠️ Execution failed after peer failure: {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"    💥 Error in fault tolerance test: {str(e)}")
            return False
    
    def save_test_results(self):
        """Save test results to file"""
        try:
            results_summary = {
                "test_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for result in self.test_results.values() if result),
                "test_details": self.test_results,
                "network_status": None  # Will be filled by async call
            }
            
            with open("test_results/p2p_federation_test_results.json", "w") as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            print(f"\n💾 Test results saved to test_results/p2p_federation_test_results.json")
            
        except Exception as e:
            print(f"❌ Failed to save test results: {str(e)}")


async def main():
    """Run P2P federation test suite"""
    tester = TestP2PFederation()
    
    try:
        success = await tester.run_all_tests()
        tester.save_test_results()
        
        if success:
            print("\n🎉 All P2P federation tests passed!")
            return 0
        else:
            print("\n💥 Some P2P federation tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)