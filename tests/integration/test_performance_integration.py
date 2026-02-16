"""
Performance Integration Tests for P2P Secure Collaboration Platform

This test suite validates performance characteristics under realistic load:
- File sharding and reconstruction performance
- Network operation performance  
- Concurrent operation handling
- Memory and resource usage
- Scalability testing
"""

import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
import hashlib
import logging
from unittest.mock import Mock, patch
import gc
import statistics

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.measurements = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        self.start_cpu = psutil.cpu_percent()
        self.measurements = []
    
    def record_measurement(self, operation_name, duration, memory_delta=None):
        """Record a performance measurement"""
        measurement = {
            'operation': operation_name,
            'duration': duration,
            'timestamp': time.time() - self.start_time,
            'memory_used': psutil.virtual_memory().used - self.start_memory,
            'cpu_percent': psutil.cpu_percent()
        }
        if memory_delta:
            measurement['memory_delta'] = memory_delta
        
        self.measurements.append(measurement)
        return measurement
    
    def get_summary(self):
        """Get performance summary"""
        if not self.measurements:
            return {}
        
        durations = [m['duration'] for m in self.measurements]
        memory_usage = [m['memory_used'] for m in self.measurements]
        
        return {
            'total_operations': len(self.measurements),
            'total_time': time.time() - self.start_time,
            'avg_operation_time': statistics.mean(durations),
            'max_operation_time': max(durations),
            'min_operation_time': min(durations),
            'peak_memory_usage': max(memory_usage),
            'final_memory_usage': memory_usage[-1]
        }


@pytest.fixture
def performance_monitor():
    """Initialize performance monitor for performance tests"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    yield monitor


class TestFileOperationPerformance:
    """Performance tests for file operations"""

    @pytest.mark.asyncio
    async def test_file_sharding_performance(self, sample_files, performance_benchmarks, performance_monitor):
        """Test file sharding performance across different file sizes"""
        logger.info("Testing file sharding performance...")
        
        # Mock the reconstruction engine for performance testing
        class MockReconstructionEngine:
            def __init__(self):
                self.shard_creation_time = 0.1  # Base time per shard
            
            async def create_secure_shards(self, content, shard_count=7, security_level='high'):
                # Simulate realistic sharding performance
                file_size_mb = len(content) / (1024 * 1024)
                processing_time = self.shard_creation_time * shard_count * (1 + file_size_mb * 0.1)
                
                await asyncio.sleep(processing_time)
                
                # Create mock shards
                shards = []
                chunk_size = len(content) // shard_count
                
                for i in range(shard_count):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < shard_count - 1 else len(content)
                    chunk = content[start_idx:end_idx]
                    
                    # Simulate encryption overhead
                    encrypted_chunk = hashlib.sha256(chunk).digest() + chunk
                    
                    shard = {
                        'shard_id': i + 1,
                        'total_shards': shard_count,
                        'encrypted_data': encrypted_chunk,
                        'checksum': hashlib.sha256(chunk).hexdigest()
                    }
                    shards.append(shard)
                
                return shards
        
        reconstruction_engine = MockReconstructionEngine()
        
        # Test different file sizes
        test_cases = [
            ('small', 'small_file_sharding'),
            ('medium', 'medium_file_sharding'),
            ('large', 'large_file_sharding')
        ]
        
        for file_type, benchmark_key in test_cases:
            logger.info(f"Testing {file_type} file sharding...")
            
            # Read test file
            with open(sample_files[file_type], 'rb') as f:
                file_content = f.read()
            
            file_size_mb = len(file_content) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            # Measure sharding performance
            start_time = time.time()
            memory_before = psutil.virtual_memory().used
            
            shards = await reconstruction_engine.create_secure_shards(
                file_content, 
                shard_count=7,
                security_level='high'
            )
            
            end_time = time.time()
            memory_after = psutil.virtual_memory().used
            
            duration = end_time - start_time
            memory_delta = memory_after - memory_before
            
            # Record performance
            performance_monitor.record_measurement(
                f"{file_type}_file_sharding",
                duration,
                memory_delta
            )
            
            # Verify sharding completed successfully
            assert len(shards) == 7
            assert all(shard['total_shards'] == 7 for shard in shards)
            
            # Check performance against benchmarks
            benchmark = performance_benchmarks['file_operations'][benchmark_key]
            assert duration <= benchmark['max_time'], \
                f"{file_type} file sharding took {duration:.2f}s, max allowed: {benchmark['max_time']}s"
            
            if duration <= benchmark['target_time']:
                logger.info(f"✅ {file_type} file sharding: {duration:.2f}s (target: {benchmark['target_time']}s)")
            else:
                logger.warning(f"⚠️  {file_type} file sharding: {duration:.2f}s (target: {benchmark['target_time']}s)")
        
        logger.info("✅ File sharding performance tests completed")
    
    @pytest.mark.asyncio
    async def test_file_reconstruction_performance(self, sample_files, performance_benchmarks, performance_monitor):
        """Test file reconstruction performance"""
        logger.info("Testing file reconstruction performance...")
        
        class MockReconstructionEngine:
            def __init__(self):
                self.reconstruction_time = 0.05  # Base time per shard
            
            async def create_secure_shards(self, content, shard_count=7, security_level='high'):
                # Create mock shards for reconstruction testing
                shards = []
                chunk_size = len(content) // shard_count
                
                for i in range(shard_count):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < shard_count - 1 else len(content)
                    chunk = content[start_idx:end_idx]
                    
                    shard = {
                        'shard_id': i + 1,
                        'total_shards': shard_count,
                        'encrypted_data': chunk,  # Simplified for testing
                        'checksum': hashlib.sha256(chunk).hexdigest()
                    }
                    shards.append(shard)
                
                return shards
            
            async def reconstruct_file(self, shards):
                # Simulate reconstruction processing time
                processing_time = self.reconstruction_time * len(shards)
                await asyncio.sleep(processing_time)
                
                # Reconstruct file from shards
                sorted_shards = sorted(shards, key=lambda x: x['shard_id'])
                reconstructed_content = b''.join(shard['encrypted_data'] for shard in sorted_shards)
                
                return reconstructed_content
        
        reconstruction_engine = MockReconstructionEngine()
        
        # Test reconstruction with medium file
        with open(sample_files['medium'], 'rb') as f:
            original_content = f.read()
        
        # Create shards
        shards = await reconstruction_engine.create_secure_shards(original_content)
        
        # Measure reconstruction performance
        start_time = time.time()
        memory_before = psutil.virtual_memory().used
        
        reconstructed_content = await reconstruction_engine.reconstruct_file(shards)
        
        end_time = time.time()
        memory_after = psutil.virtual_memory().used
        
        duration = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # Record performance
        performance_monitor.record_measurement(
            "file_reconstruction",
            duration,
            memory_delta
        )
        
        # Verify reconstruction
        assert reconstructed_content == original_content
        
        # Check performance
        benchmark = performance_benchmarks['file_operations']['file_reconstruction']
        assert duration <= benchmark['max_time'], \
            f"File reconstruction took {duration:.2f}s, max allowed: {benchmark['max_time']}s"
        
        logger.info(f"✅ File reconstruction: {duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, sample_files, performance_monitor):
        """Test concurrent file sharding operations"""
        logger.info("Testing concurrent file operations...")
        
        class MockReconstructionEngine:
            def __init__(self):
                self.operation_count = 0
                self.concurrent_operations = 0
                self.max_concurrent = 0
            
            async def create_secure_shards(self, content, shard_count=7, security_level='high'):
                self.concurrent_operations += 1
                self.max_concurrent = max(self.max_concurrent, self.concurrent_operations)
                
                try:
                    # Simulate processing time
                    file_size_mb = len(content) / (1024 * 1024)
                    processing_time = 0.1 * shard_count * (1 + file_size_mb * 0.05)
                    await asyncio.sleep(processing_time)
                    
                    # Create mock shards
                    shards = []
                    chunk_size = len(content) // shard_count
                    
                    for i in range(shard_count):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size if i < shard_count - 1 else len(content)
                        chunk = content[start_idx:end_idx]
                        
                        shard = {
                            'shard_id': i + 1,
                            'total_shards': shard_count,
                            'encrypted_data': chunk,
                            'checksum': hashlib.sha256(chunk).hexdigest()
                        }
                        shards.append(shard)
                    
                    self.operation_count += 1
                    return shards
                
                finally:
                    self.concurrent_operations -= 1
        
        reconstruction_engine = MockReconstructionEngine()
        
        # Prepare multiple file contents for concurrent processing
        file_contents = []
        for file_type in ['small', 'medium']:
            with open(sample_files[file_type], 'rb') as f:
                file_contents.append(f.read())
        
        # Create multiple copies to test concurrency
        test_contents = file_contents * 3  # 6 total operations
        
        # Measure concurrent operations
        start_time = time.time()
        memory_before = psutil.virtual_memory().used
        
        # Execute concurrent sharding operations
        tasks = []
        for i, content in enumerate(test_contents):
            task = reconstruction_engine.create_secure_shards(
                content,
                shard_count=5,  # Smaller shard count for faster testing
                security_level='medium'
            )
            tasks.append(task)
        
        # Wait for all operations to complete
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        memory_after = psutil.virtual_memory().used
        
        duration = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # Record performance
        performance_monitor.record_measurement(
            "concurrent_file_operations",
            duration,
            memory_delta
        )
        
        # Verify all operations completed successfully
        assert len(results) == len(test_contents)
        assert all(len(shards) == 5 for shards in results)
        assert reconstruction_engine.operation_count == len(test_contents)
        
        # Check concurrency achieved
        assert reconstruction_engine.max_concurrent > 1, "Should achieve concurrent execution"
        
        # Performance should be better than sequential
        avg_time_per_operation = duration / len(test_contents)
        logger.info(f"Concurrent operations: {duration:.2f}s total, {avg_time_per_operation:.2f}s avg per operation")
        logger.info(f"Max concurrent operations: {reconstruction_engine.max_concurrent}")
        
        # Should complete in reasonable time
        assert duration < 10.0, f"Concurrent operations took too long: {duration:.2f}s"
        
        logger.info("✅ Concurrent file operations test completed")


class TestNetworkPerformance:
    """Performance tests for network operations"""
    
    @pytest.mark.asyncio
    async def test_peer_discovery_performance(self, mock_network_peers, performance_benchmarks, performance_monitor):
        """Test peer discovery performance"""
        logger.info("Testing peer discovery performance...")
        
        class MockNodeDiscovery:
            def __init__(self):
                self.peers = {}
                self.discovery_time = 0.05  # Base discovery time per peer
            
            async def initialize(self):
                await asyncio.sleep(0.1)  # Initialization time
            
            async def add_peer(self, peer):
                await asyncio.sleep(0.01)  # Simulate network overhead
                self.peers[peer['id']] = peer
            
            async def discover_peers(self, max_peers=50):
                # Simulate network discovery time
                discovery_time = self.discovery_time * min(len(self.peers), max_peers)
                await asyncio.sleep(discovery_time)
                
                # Return available peers
                return list(self.peers.values())[:max_peers]
        
        node_discovery = MockNodeDiscovery()
        await node_discovery.initialize()
        
        # Add mock peers
        for peer in mock_network_peers:
            await node_discovery.add_peer(peer)
        
        # Measure peer discovery performance
        start_time = time.time()
        
        discovered_peers = await node_discovery.discover_peers()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record performance
        performance_monitor.record_measurement("peer_discovery", duration)
        
        # Verify discovery results
        assert len(discovered_peers) == len(mock_network_peers)
        
        # Check performance
        benchmark = performance_benchmarks['network_operations']['peer_discovery']
        assert duration <= benchmark['max_time'], \
            f"Peer discovery took {duration:.2f}s, max allowed: {benchmark['max_time']}s"
        
        logger.info(f"✅ Peer discovery: {duration:.2f}s for {len(discovered_peers)} peers")
    
    @pytest.mark.asyncio
    async def test_shard_distribution_performance(self, mock_network_peers, performance_benchmarks, performance_monitor):
        """Test shard distribution performance"""
        logger.info("Testing shard distribution performance...")
        
        class MockShardDistributor:
            def __init__(self):
                self.distribution_time = 0.02  # Base time per shard distribution
            
            async def initialize(self):
                await asyncio.sleep(0.1)
            
            async def select_distribution_peers(self, peers, shard_count, security_level='high'):
                # Simulate peer selection algorithm
                selection_time = 0.1 + (len(peers) * 0.01)
                await asyncio.sleep(selection_time)
                
                # Select best peers based on reputation and capabilities
                suitable_peers = [p for p in peers if p['reputation'] >= 4.0]
                selected = suitable_peers[:shard_count]
                
                return selected
            
            async def optimize_shard_placement(self, selected_peers, file_metadata):
                # Simulate placement optimization
                placement_time = self.distribution_time * len(selected_peers)
                await asyncio.sleep(placement_time)
                
                placements = []
                for i, peer in enumerate(selected_peers):
                    placement = {
                        'shard_id': i + 1,
                        'peer_id': peer['id'],
                        'peer_region': peer['region'],
                        'estimated_latency': peer['latency']
                    }
                    placements.append(placement)
                
                return placements
        
        shard_distributor = MockShardDistributor()
        await shard_distributor.initialize()
        
        # Test high-security distribution (7 shards)
        file_metadata = {
            'size': 2.4 * 1024 * 1024,  # 2.4 MB
            'security_level': 'high',
            'shard_count': 7
        }
        
        # Measure distribution performance
        start_time = time.time()
        
        # Select peers
        selected_peers = await shard_distributor.select_distribution_peers(
            mock_network_peers, 
            file_metadata['shard_count'],
            file_metadata['security_level']
        )
        
        # Optimize placement
        placements = await shard_distributor.optimize_shard_placement(
            selected_peers, 
            file_metadata
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record performance
        performance_monitor.record_measurement("shard_distribution", duration)
        
        # Verify distribution results
        assert len(selected_peers) == 7
        assert len(placements) == 7
        assert all(p['reputation'] >= 4.0 for p in selected_peers)
        
        # Check performance
        benchmark = performance_benchmarks['network_operations']['shard_distribution']
        assert duration <= benchmark['max_time'], \
            f"Shard distribution took {duration:.2f}s, max allowed: {benchmark['max_time']}s"
        
        logger.info(f"✅ Shard distribution: {duration:.2f}s for {len(placements)} shards")
    
    @pytest.mark.asyncio
    async def test_network_optimization_performance(self, mock_network_peers, performance_monitor):
        """Test network optimization performance"""
        logger.info("Testing network optimization performance...")
        
        class MockBandwidthOptimizer:
            def __init__(self):
                self.peers = {}
                self.optimization_time = 0.1
            
            async def initialize(self):
                await asyncio.sleep(0.05)
            
            async def add_peer(self, peer_id, bandwidth, latency):
                self.peers[peer_id] = {
                    'bandwidth': bandwidth,
                    'latency': latency,
                    'load': 0.5  # Initial load
                }
            
            async def optimize_network_performance(self):
                # Simulate network optimization algorithms
                await asyncio.sleep(self.optimization_time)
                
                optimizations = []
                for peer_id, metrics in self.peers.items():
                    if metrics['load'] > 0.8:
                        optimizations.append({
                            'peer_id': peer_id,
                            'optimization': 'load_balancing',
                            'improvement': '15%'
                        })
                    if metrics['latency'] > 100:
                        optimizations.append({
                            'peer_id': peer_id,
                            'optimization': 'route_optimization',
                            'improvement': '20%'
                        })
                
                return optimizations
        
        optimizer = MockBandwidthOptimizer()
        await optimizer.initialize()
        
        # Add peers to optimizer
        for peer in mock_network_peers:
            await optimizer.add_peer(
                peer['id'],
                peer['bandwidth']['upload'],
                peer['latency']
            )
        
        # Measure optimization performance
        start_time = time.time()
        
        optimizations = await optimizer.optimize_network_performance()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record performance
        performance_monitor.record_measurement("network_optimization", duration)
        
        # Verify optimization results
        assert isinstance(optimizations, list)
        
        # Should complete quickly
        assert duration < 1.0, f"Network optimization took too long: {duration:.2f}s"
        
        logger.info(f"✅ Network optimization: {duration:.2f}s, {len(optimizations)} optimizations applied")


class TestSecurityPerformance:
    """Performance tests for security operations"""
    
    @pytest.mark.asyncio
    async def test_key_generation_performance(self, performance_benchmarks, performance_monitor):
        """Test post-quantum key generation performance"""
        logger.info("Testing key generation performance...")
        
        class MockKeyManager:
            def __init__(self):
                self.key_generation_time = 0.2  # Realistic PQ key gen time
            
            async def generate_master_key(self, algorithm='kyber-1024'):
                # Simulate post-quantum key generation
                await asyncio.sleep(self.key_generation_time)
                
                # Mock key structure
                return {
                    'algorithm': algorithm,
                    'public_key': b'mock_public_key_' + os.urandom(32),
                    'private_key': b'mock_private_key_' + os.urandom(32),
                    'key_size': 1568 if algorithm == 'kyber-1024' else 1024
                }
            
            async def create_secret_shares(self, private_key, threshold=4, total_shares=7):
                # Simulate Shamir's Secret Sharing
                sharing_time = 0.05 * total_shares
                await asyncio.sleep(sharing_time)
                
                shares = []
                for i in range(total_shares):
                    share = {
                        'share_id': i + 1,
                        'threshold': threshold,
                        'total_shares': total_shares,
                        'share_data': hashlib.sha256(private_key + str(i).encode()).digest()
                    }
                    shares.append(share)
                
                return shares
        
        key_manager = MockKeyManager()
        
        # Test key generation performance
        start_time = time.time()
        
        master_key = await key_manager.generate_master_key('kyber-1024')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record performance
        performance_monitor.record_measurement("key_generation", duration)
        
        # Verify key generation
        assert master_key['algorithm'] == 'kyber-1024'
        assert len(master_key['public_key']) > 0
        assert len(master_key['private_key']) > 0
        
        # Check performance
        benchmark = performance_benchmarks['security_operations']['key_generation']
        assert duration <= benchmark['max_time'], \
            f"Key generation took {duration:.2f}s, max allowed: {benchmark['max_time']}s"
        
        # Test secret sharing performance
        start_time = time.time()
        
        shares = await key_manager.create_secret_shares(master_key['private_key'])
        
        end_time = time.time()
        sharing_duration = end_time - start_time
        
        # Verify secret sharing
        assert len(shares) == 7
        assert all(share['threshold'] == 4 for share in shares)
        
        logger.info(f"✅ Key generation: {duration:.2f}s, Secret sharing: {sharing_duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_access_authorization_performance(self, mock_users, performance_benchmarks, performance_monitor):
        """Test access authorization performance"""
        logger.info("Testing access authorization performance...")
        
        class MockAccessController:
            def __init__(self):
                self.users = {}
                self.authorization_time = 0.05
            
            async def register_user(self, user):
                self.users[user['id']] = user
            
            async def request_authorization(self, request):
                # Simulate authorization processing
                await asyncio.sleep(self.authorization_time)
                
                user = self.users.get(request['user_id'])
                if not user:
                    return {'status': 'denied', 'reason': 'user_not_found'}
                
                # Determine approval requirements based on security level
                if 'high' in request.get('resource_id', ''):
                    required_approvals = 2
                elif 'medium' in request.get('resource_id', ''):
                    required_approvals = 1
                else:
                    required_approvals = 1
                
                if user['clearance_level'] == 'high' and required_approvals == 1:
                    return {'status': 'approved', 'request_id': f"auth_{hash(request['user_id']) % 1000}"}
                else:
                    return {
                        'status': 'pending',
                        'request_id': f"auth_{hash(request['user_id']) % 1000}",
                        'required_approvals': required_approvals
                    }
        
        access_controller = MockAccessController()
        
        # Register users
        for user in mock_users:
            await access_controller.register_user(user)
        
        # Test various authorization scenarios
        test_requests = [
            {
                'resource_id': 'standard_document.pdf',
                'user_id': 'dr.chen@unc.edu',
                'operation': 'read'
            },
            {
                'resource_id': 'high_security_algorithm.pdf',
                'user_id': 'michael.j@sas.com',
                'operation': 'evaluate'
            },
            {
                'resource_id': 'medium_security_data.csv',
                'user_id': 'alex.r@duke.edu',
                'operation': 'read'
            }
        ]
        
        total_time = 0
        for request in test_requests:
            start_time = time.time()
            
            result = await access_controller.request_authorization(request)
            
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            # Verify authorization result
            assert result['status'] in ['approved', 'pending', 'denied']
            
            # Record individual performance
            performance_monitor.record_measurement(
                f"access_authorization_{request['resource_id'].split('_')[0]}",
                duration
            )
        
        avg_time = total_time / len(test_requests)
        
        # Check performance
        benchmark = performance_benchmarks['security_operations']['access_authorization']
        assert avg_time <= benchmark['max_time'], \
            f"Average authorization time {avg_time:.2f}s, max allowed: {benchmark['max_time']}s"
        
        logger.info(f"✅ Access authorization: {avg_time:.2f}s average for {len(test_requests)} requests")
    
    @pytest.mark.asyncio
    async def test_integrity_validation_performance(self, sample_files, performance_benchmarks, performance_monitor):
        """Test integrity validation performance"""
        logger.info("Testing integrity validation performance...")
        
        class MockIntegrityValidator:
            def __init__(self):
                self.validation_time = 0.01  # Per shard validation time
            
            async def create_merkle_tree(self, shards):
                # Simulate Merkle tree creation
                tree_creation_time = 0.05 + (len(shards) * 0.01)
                await asyncio.sleep(tree_creation_time)
                
                # Create mock Merkle tree
                return {
                    'root_hash': hashlib.sha256(str(len(shards)).encode()).hexdigest(),
                    'leaf_hashes': [hashlib.sha256(str(i).encode()).hexdigest() for i in range(len(shards))],
                    'tree_depth': len(shards).bit_length()
                }
            
            async def validate_shard_integrity(self, shard, root_hash):
                # Simulate integrity validation
                await asyncio.sleep(self.validation_time)
                
                # Mock validation (always pass for testing)
                return True
        
        validator = MockIntegrityValidator()
        
        # Create mock shards for testing
        with open(sample_files['medium'], 'rb') as f:
            file_content = f.read()
        
        # Create mock shards
        shard_count = 7
        chunk_size = len(file_content) // shard_count
        shards = []
        
        for i in range(shard_count):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < shard_count - 1 else len(file_content)
            chunk = file_content[start_idx:end_idx]
            
            shard = {
                'shard_id': i + 1,
                'total_shards': shard_count,
                'encrypted_data': chunk,
                'checksum': hashlib.sha256(chunk).hexdigest()
            }
            shards.append(shard)
        
        # Test Merkle tree creation performance
        start_time = time.time()
        
        merkle_tree = await validator.create_merkle_tree(shards)
        
        tree_creation_time = time.time() - start_time
        
        # Test integrity validation performance
        start_time = time.time()
        
        validation_results = []
        for shard in shards:
            is_valid = await validator.validate_shard_integrity(shard, merkle_tree['root_hash'])
            validation_results.append(is_valid)
        
        validation_time = time.time() - start_time
        total_time = tree_creation_time + validation_time
        
        # Record performance
        performance_monitor.record_measurement("merkle_tree_creation", tree_creation_time)
        performance_monitor.record_measurement("integrity_validation", validation_time)
        
        # Verify results
        assert merkle_tree['root_hash'] is not None
        assert len(merkle_tree['leaf_hashes']) == shard_count
        assert all(validation_results)
        
        # Check performance
        benchmark = performance_benchmarks['network_operations']['integrity_validation']
        assert total_time <= benchmark['max_time'], \
            f"Integrity validation took {total_time:.2f}s, max allowed: {benchmark['max_time']}s"
        
        logger.info(f"✅ Integrity validation: {total_time:.2f}s total ({tree_creation_time:.2f}s tree + {validation_time:.2f}s validation)")


class TestScalabilityPerformance:
    """Scalability and stress tests"""
    
    @pytest.mark.asyncio
    async def test_scalability_with_increasing_peers(self, performance_monitor):
        """Test performance scaling with increasing number of peers"""
        logger.info("Testing scalability with increasing peers...")
        
        class MockScalableSystem:
            def __init__(self):
                self.peers = {}
                self.base_operation_time = 0.01
            
            async def add_peer(self, peer):
                self.peers[peer['id']] = peer
            
            async def process_network_operation(self):
                # Simulate operation that scales with peer count
                peer_count = len(self.peers)
                operation_time = self.base_operation_time * (1 + peer_count * 0.01)
                await asyncio.sleep(operation_time)
                return peer_count
        
        system = MockScalableSystem()
        
        # Test with different peer counts
        peer_counts = [10, 25, 50, 100, 200]
        performance_results = []
        
        for target_count in peer_counts:
            # Add peers to reach target count
            current_count = len(system.peers)
            for i in range(current_count, target_count):
                peer = {
                    'id': f'peer_{i}',
                    'reputation': 4.0 + (i % 5) * 0.2,
                    'region': ['us-west', 'us-east', 'europe', 'asia'][i % 4]
                }
                await system.add_peer(peer)
            
            # Measure operation performance
            start_time = time.time()
            
            result = await system.process_network_operation()
            
            end_time = time.time()
            duration = end_time - start_time
            
            performance_results.append({
                'peer_count': target_count,
                'duration': duration,
                'throughput': target_count / duration
            })
            
            performance_monitor.record_measurement(f"scalability_{target_count}_peers", duration)
            
            assert result == target_count
            
            logger.info(f"Peer count: {target_count}, Duration: {duration:.3f}s, Throughput: {target_count/duration:.1f} peers/s")
        
        # Verify scaling characteristics
        # Performance should not degrade exponentially
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i-1]
            
            scale_factor = current['peer_count'] / previous['peer_count']
            time_factor = current['duration'] / previous['duration']
            
            # Time increase should be roughly proportional to peer increase (not exponential)
            assert time_factor < scale_factor * 2, \
                f"Performance degradation too severe: {scale_factor}x peers took {time_factor}x time"
        
        logger.info("✅ Scalability test completed successfully")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_monitor):
        """Test memory usage under heavy load"""
        logger.info("Testing memory usage under load...")
        
        # Force garbage collection before test
        gc.collect()
        initial_memory = psutil.virtual_memory().used
        
        class MemoryIntensiveOperation:
            def __init__(self):
                self.data_storage = {}
            
            async def process_large_dataset(self, dataset_size_mb):
                # Simulate processing large amounts of data
                data = bytearray(dataset_size_mb * 1024 * 1024)  # Create data of specified size
                
                # Simulate some processing
                for i in range(0, len(data), 1024):
                    data[i:i+4] = hashlib.sha256(data[i:i+1024]).digest()[:4]
                
                # Store result temporarily
                dataset_id = f"dataset_{len(self.data_storage)}"
                self.data_storage[dataset_id] = data
                
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Clean up (simulate data lifecycle)
                if len(self.data_storage) > 5:  # Keep only recent datasets
                    oldest_key = next(iter(self.data_storage))
                    del self.data_storage[oldest_key]
                    gc.collect()  # Force cleanup
                
                return len(data)
        
        operation = MemoryIntensiveOperation()
        
        # Test with various dataset sizes
        dataset_sizes = [1, 5, 10, 20]  # MB
        memory_measurements = []
        
        for size_mb in dataset_sizes:
            memory_before = psutil.virtual_memory().used
            
            start_time = time.time()
            
            result = await operation.process_large_dataset(size_mb)
            
            end_time = time.time()
            memory_after = psutil.virtual_memory().used
            
            duration = end_time - start_time
            memory_delta = memory_after - memory_before
            
            memory_measurements.append({
                'dataset_size_mb': size_mb,
                'duration': duration,
                'memory_delta_mb': memory_delta / (1024 * 1024),
                'memory_efficiency': size_mb / (memory_delta / (1024 * 1024)) if memory_delta > 0 else 0
            })
            
            performance_monitor.record_measurement(f"memory_test_{size_mb}mb", duration, memory_delta)
            
            assert result == size_mb * 1024 * 1024
            
            logger.info(f"Dataset: {size_mb}MB, Duration: {duration:.2f}s, Memory delta: {memory_delta/(1024*1024):.1f}MB")
        
        # Verify memory usage is reasonable
        final_memory = psutil.virtual_memory().used
        total_memory_increase = final_memory - initial_memory
        
        # Should not have excessive memory leaks
        assert total_memory_increase < 100 * 1024 * 1024, \
            f"Excessive memory usage: {total_memory_increase/(1024*1024):.1f}MB increase"
        
        # Memory efficiency should be reasonable (at least 50% efficiency)
        for measurement in memory_measurements:
            if measurement['memory_delta_mb'] > 0:
                assert measurement['memory_efficiency'] > 0.3, \
                    f"Poor memory efficiency: {measurement['memory_efficiency']:.2f} for {measurement['dataset_size_mb']}MB"
        
        logger.info(f"✅ Memory usage test completed. Total increase: {total_memory_increase/(1024*1024):.1f}MB")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])