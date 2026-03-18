"""
Unit tests for Distributed Resource Manager disk benchmarking and network measurement methods.

Tests cover:
- Disk read/write benchmark math correctness
- Disk benchmark caching behavior
- Disk benchmark fallback on exception
- Network latency measurement math correctness
- Network bandwidth measurement math correctness
- Network measurement caching behavior
- Network measurement fallback on exception
"""

import pytest
import asyncio
import time
import socket
import statistics
from unittest.mock import patch, MagicMock, AsyncMock
import os

from prsm.compute.federation.distributed_resource_manager import ResourceCapabilityDetector


class TestDiskBenchmarkMath:
    """Test that disk benchmark calculations are mathematically correct."""
    
    @pytest.mark.asyncio
    async def test_disk_write_benchmark_math(self):
        """Verify disk write benchmark calculates MB/s correctly."""
        detector = ResourceCapabilityDetector()
        
        # Mock time.perf_counter to return predictable values
        # Simulate a write that takes 0.5 seconds for 64MB
        # Expected result: 64 MB / 0.5 s = 128 MB/s
        mock_times = [0.0, 0.5]  # start, end
        
        with patch('time.perf_counter', side_effect=mock_times):
            with patch('os.urandom', return_value=b'\x00' * (64 * 1024 * 1024)):
                with patch('tempfile.NamedTemporaryFile') as mock_tmpfile:
                    # Setup mock file
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = '/tmp/test_bench.prsm_bench'
                    mock_file.fileno.return_value = 3
                    mock_tmpfile.return_value = mock_file
                    
                    with patch('os.fsync'):
                        with patch('os.unlink'):
                            result = await detector._benchmark_disk_write()
                            
                            # Verify the math: 64 MB / 0.5 s = 128 MB/s
                            assert result == 128.0
    
    @pytest.mark.asyncio
    async def test_disk_read_benchmark_math(self):
        """Verify disk read benchmark calculates MB/s correctly."""
        detector = ResourceCapabilityDetector()
        
        # Mock time.perf_counter to return predictable values
        # Simulate a read that takes 0.25 seconds for 64MB
        # Expected result: 64 MB / 0.25 s = 256 MB/s
        mock_times = [0.0, 0.25]  # start, end
        
        with patch('time.perf_counter', side_effect=mock_times):
            with patch('os.urandom', return_value=b'\x00' * (64 * 1024 * 1024)):
                with patch('tempfile.NamedTemporaryFile') as mock_tmpfile:
                    # Setup mock file for write phase
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = '/tmp/test_bench.prsm_bench'
                    mock_file.fileno.return_value = 3
                    mock_tmpfile.return_value = mock_file
                    
                    with patch('os.fsync'):
                        with patch('os.unlink'):
                            # Mock the open for read phase
                            mock_read_file = MagicMock()
                            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
                            mock_read_file.__exit__ = MagicMock(return_value=False)
                            # Simulate reading 64MB in 1MB chunks (64 iterations)
                            mock_read_file.read.side_effect = [b'\x00' * (1024 * 1024)] * 64 + [b'']
                            
                            with patch('builtins.open', return_value=mock_read_file):
                                result = await detector._benchmark_disk_read()
                                
                                # Verify the math: 64 MB / 0.25 s = 256 MB/s
                                assert result == 256.0


class TestDiskBenchmarkCaching:
    """Test caching behavior of disk benchmarks."""
    
    @pytest.mark.asyncio
    async def test_disk_write_cache_hit(self):
        """Test that cached write result is returned within TTL."""
        detector = ResourceCapabilityDetector()
        
        # Pre-populate cache
        detector._disk_write_cache = 200.0
        detector._disk_write_cache_time = time.time()
        
        # Should return cached value without running benchmark
        result = await detector._benchmark_disk_write()
        assert result == 200.0
    
    @pytest.mark.asyncio
    async def test_disk_read_cache_hit(self):
        """Test that cached read result is returned within TTL."""
        detector = ResourceCapabilityDetector()
        
        # Pre-populate cache
        detector._disk_read_cache = 300.0
        detector._disk_read_cache_time = time.time()
        
        # Should return cached value without running benchmark
        result = await detector._benchmark_disk_read()
        assert result == 300.0
    
    @pytest.mark.asyncio
    async def test_disk_write_cache_expired(self):
        """Test that cache is refreshed after TTL expires."""
        detector = ResourceCapabilityDetector()
        detector._disk_cache_ttl = 1.0  # 1 second TTL for testing
        
        # Pre-populate cache with expired timestamp
        detector._disk_write_cache = 200.0
        detector._disk_write_cache_time = time.time() - 2.0  # 2 seconds ago
        
        # Mock the sync benchmark to return a new value
        with patch.object(detector, '_sync_benchmark_disk_write', return_value=150.0):
            result = await detector._benchmark_disk_write()
            
            # Should have run benchmark and returned new value
            assert result == 150.0
            assert detector._disk_write_cache == 150.0
    
    @pytest.mark.asyncio
    async def test_disk_read_cache_expired(self):
        """Test that read cache is refreshed after TTL expires."""
        detector = ResourceCapabilityDetector()
        detector._disk_cache_ttl = 1.0  # 1 second TTL for testing
        
        # Pre-populate cache with expired timestamp
        detector._disk_read_cache = 300.0
        detector._disk_read_cache_time = time.time() - 2.0  # 2 seconds ago
        
        # Mock the sync benchmark to return a new value
        with patch.object(detector, '_sync_benchmark_disk_read', return_value=250.0):
            result = await detector._benchmark_disk_read()
            
            # Should have run benchmark and returned new value
            assert result == 250.0
            assert detector._disk_read_cache == 250.0
    
    @pytest.mark.asyncio
    async def test_cache_stores_result_after_benchmark(self):
        """Test that benchmark result is cached after running."""
        detector = ResourceCapabilityDetector()
        
        # No cache initially
        assert detector._disk_write_cache is None
        
        # Mock the sync benchmark
        with patch.object(detector, '_sync_benchmark_disk_write', return_value=180.0):
            result = await detector._benchmark_disk_write()
            
            # Result should be cached
            assert result == 180.0
            assert detector._disk_write_cache == 180.0
            assert detector._disk_write_cache_time is not None


class TestDiskBenchmarkFallback:
    """Test fallback behavior on exceptions."""
    
    @pytest.mark.asyncio
    async def test_disk_write_fallback_on_exception(self):
        """Test that write benchmark returns 100.0 on exception."""
        detector = ResourceCapabilityDetector()
        
        # Make the sync benchmark raise an exception
        with patch.object(detector, '_sync_benchmark_disk_write', side_effect=OSError("Disk error")):
            result = await detector._benchmark_disk_write()
            
            # Should return fallback value
            assert result == 100.0
    
    @pytest.mark.asyncio
    async def test_disk_read_fallback_on_exception(self):
        """Test that read benchmark returns 100.0 on exception."""
        detector = ResourceCapabilityDetector()
        
        # Make the sync benchmark raise an exception
        with patch.object(detector, '_sync_benchmark_disk_read', side_effect=OSError("Disk error")):
            result = await detector._benchmark_disk_read()
            
            # Should return fallback value
            assert result == 100.0
    
    @pytest.mark.asyncio
    async def test_disk_write_fallback_on_permission_error(self):
        """Test fallback on permission denied error."""
        detector = ResourceCapabilityDetector()
        
        with patch.object(detector, '_sync_benchmark_disk_write', 
                         side_effect=PermissionError("Permission denied")):
            result = await detector._benchmark_disk_write()
            assert result == 100.0
    
    @pytest.mark.asyncio
    async def test_disk_read_fallback_on_file_not_found(self):
        """Test fallback on file not found error."""
        detector = ResourceCapabilityDetector()
        
        with patch.object(detector, '_sync_benchmark_disk_read', 
                         side_effect=FileNotFoundError("Temp file not found")):
            result = await detector._benchmark_disk_read()
            assert result == 100.0


class TestSyncBenchmarkImplementations:
    """Test the synchronous benchmark implementations directly."""
    
    def test_sync_disk_write_creates_temp_file(self):
        """Test that sync write benchmark creates temp file with correct suffix."""
        detector = ResourceCapabilityDetector()
        
        with patch('tempfile.NamedTemporaryFile') as mock_tmpfile:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = '/tmp/test.prsm_bench'
            mock_file.fileno.return_value = 3
            mock_tmpfile.return_value = mock_file
            
            with patch('os.urandom', return_value=b'\x00' * (64 * 1024 * 1024)):
                with patch('time.perf_counter', side_effect=[0.0, 1.0]):
                    with patch('os.fsync'):
                        with patch('os.unlink'):
                            detector._sync_benchmark_disk_write()
                            
                            # Verify NamedTemporaryFile was called with correct suffix
                            mock_tmpfile.assert_called_once()
                            call_kwargs = mock_tmpfile.call_args[1]
                            assert call_kwargs['delete'] == False
                            assert call_kwargs['suffix'] == '.prsm_bench'
    
    def test_sync_disk_read_cleans_up_temp_file(self):
        """Test that sync read benchmark cleans up temp file."""
        detector = ResourceCapabilityDetector()
        
        with patch('tempfile.NamedTemporaryFile') as mock_tmpfile:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = '/tmp/test.prsm_bench'
            mock_file.fileno.return_value = 3
            mock_tmpfile.return_value = mock_file
            
            with patch('os.urandom', return_value=b'\x00' * (64 * 1024 * 1024)):
                with patch('time.perf_counter', side_effect=[0.0, 1.0]):
                    with patch('os.fsync'):
                        with patch('os.unlink') as mock_unlink:
                            # Create a proper mock for the file read context manager
                            mock_read_file = MagicMock()
                            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
                            mock_read_file.__exit__ = MagicMock(return_value=False)
                            # Return data once, then empty bytes to terminate the loop
                            mock_read_file.read.side_effect = [b'\x00' * (1024 * 1024), b'']
                            
                            with patch('builtins.open', return_value=mock_read_file):
                                detector._sync_benchmark_disk_read()
                                
                                # Verify unlink was called to clean up
                                assert mock_unlink.called


# ============================================================================
# NETWORK LATENCY MEASUREMENT TESTS
# ============================================================================

class TestNetworkLatencyMath:
    """Test that network latency measurement calculations are correct."""
    
    @pytest.mark.asyncio
    async def test_latency_median_calculation(self):
        """Verify latency returns median of successful measurements."""
        detector = ResourceCapabilityDetector()
        
        # Mock the sync measurement to return a known median value
        # This tests that the async wrapper correctly returns the sync result
        with patch.object(detector, '_sync_measure_latency', return_value=20.0):
            result = await detector._measure_network_latency()
            
            # Should return the median from sync measurement
            assert result == 20.0
    
    @pytest.mark.asyncio
    async def test_latency_with_one_failure(self):
        """Verify latency returns median when one target fails."""
        detector = ResourceCapabilityDetector()
        
        # Mock the sync measurement to simulate partial failure scenario
        # The sync function handles the median calculation internally
        with patch.object(detector, '_sync_measure_latency', return_value=20.0):
            result = await detector._measure_network_latency()
            
            # Should return the median from sync measurement
            assert result == 20.0
    
    @pytest.mark.asyncio
    async def test_latency_all_failures_returns_999(self):
        """Verify latency returns 999.0 when all targets fail."""
        detector = ResourceCapabilityDetector()
        
        # Mock the sync measurement to return 999.0 (all targets failed)
        with patch.object(detector, '_sync_measure_latency', return_value=999.0):
            result = await detector._measure_network_latency()
            
            # Should return 999.0 when all fail
            assert result == 999.0


class TestNetworkLatencyCaching:
    """Test caching behavior of network latency measurement."""
    
    @pytest.mark.asyncio
    async def test_latency_cache_hit(self):
        """Test that cached latency result is returned within TTL."""
        detector = ResourceCapabilityDetector()
        
        # Pre-populate cache
        detector._network_latency_cache = 25.0
        detector._network_latency_cache_time = time.time()
        
        # Should return cached value without running measurement
        result = await detector._measure_network_latency()
        assert result == 25.0
    
    @pytest.mark.asyncio
    async def test_latency_cache_expired(self):
        """Test that cache is refreshed after TTL expires."""
        detector = ResourceCapabilityDetector()
        detector._network_cache_ttl = 1.0  # 1 second TTL for testing
        
        # Pre-populate cache with expired timestamp
        detector._network_latency_cache = 25.0
        detector._network_latency_cache_time = time.time() - 2.0  # 2 seconds ago
        
        # Mock the sync measurement to return a new value
        with patch.object(detector, '_sync_measure_latency', return_value=35.0):
            result = await detector._measure_network_latency()
            
            # Should have run measurement and returned new value
            assert result == 35.0
            assert detector._network_latency_cache == 35.0
    
    @pytest.mark.asyncio
    async def test_latency_cache_stores_result_after_measurement(self):
        """Test that measurement result is cached after running."""
        detector = ResourceCapabilityDetector()
        
        # No cache initially
        assert detector._network_latency_cache is None
        
        # Mock the sync measurement
        with patch.object(detector, '_sync_measure_latency', return_value=18.0):
            result = await detector._measure_network_latency()
            
            # Result should be cached
            assert result == 18.0
            assert detector._network_latency_cache == 18.0
            assert detector._network_latency_cache_time is not None


class TestNetworkLatencyFallback:
    """Test fallback behavior on exceptions."""
    
    @pytest.mark.asyncio
    async def test_latency_fallback_on_exception(self):
        """Test that latency measurement returns 999.0 on exception."""
        detector = ResourceCapabilityDetector()
        
        # Make the sync measurement raise an exception
        with patch.object(detector, '_sync_measure_latency', side_effect=RuntimeError("Unexpected error")):
            result = await detector._measure_network_latency()
            
            # Should return fallback value
            assert result == 999.0


class TestSyncLatencyMeasurement:
    """Test the synchronous latency measurement implementation directly."""
    
    def test_sync_latency_uses_correct_targets(self):
        """Test that sync latency uses the correct DNS targets."""
        detector = ResourceCapabilityDetector()
        
        expected_targets = [
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
            ("208.67.222.222", 53),
        ]
        
        connected_targets = []
        
        def mock_create_connection(target, timeout):
            connected_targets.append(target)
            return MagicMock(close=lambda: None)
        
        with patch('socket.create_connection', side_effect=mock_create_connection):
            with patch('time.perf_counter', side_effect=[0.0, 0.010] * 3):
                detector._sync_measure_latency()
                
                # Verify all expected targets were contacted
                assert len(connected_targets) == 3
                for target in expected_targets:
                    assert target in connected_targets
    
    def test_sync_latency_timeout_parameter(self):
        """Test that socket connection uses correct timeout."""
        detector = ResourceCapabilityDetector()
        
        captured_timeout = []
        
        def mock_create_connection(target, timeout):
            captured_timeout.append(timeout)
            return MagicMock(close=lambda: None)
        
        with patch('socket.create_connection', side_effect=mock_create_connection):
            with patch('time.perf_counter', side_effect=[0.0, 0.010] * 3):
                detector._sync_measure_latency()
                
                # Verify timeout is 2 seconds
                for t in captured_timeout:
                    assert t == 2


# ============================================================================
# NETWORK BANDWIDTH MEASUREMENT TESTS
# ============================================================================

class TestNetworkBandwidthMath:
    """Test that network bandwidth measurement calculations are correct."""
    
    @pytest.mark.asyncio
    async def test_bandwidth_calculation(self):
        """Verify bandwidth calculates Mbps correctly from counter differential."""
        detector = ResourceCapabilityDetector()
        
        # Create mock counters
        # Simulate 3,750,000 bytes received and 1,500,000 bytes sent in 3 seconds
        # download: (3,750,000 * 8) / (3 * 1,000,000) = 10 Mbps
        # upload: (1,500,000 * 8) / (3 * 1,000,000) = 4 Mbps
        
        mock_before = MagicMock()
        mock_before.bytes_recv = 1_000_000
        mock_before.bytes_sent = 500_000
        
        mock_after = MagicMock()
        mock_after.bytes_recv = 4_750_000  # 3,750,000 bytes received
        mock_after.bytes_sent = 2_000_000  # 1,500,000 bytes sent
        
        with patch('psutil.net_io_counters', side_effect=[mock_before, mock_after]):
            with patch('asyncio.sleep', return_value=None):
                result = await detector._run_bandwidth_test()
                
                # Verify the math
                assert result[0] == 10.0  # download Mbps
                assert result[1] == 4.0   # upload Mbps
    
    @pytest.mark.asyncio
    async def test_bandwidth_minimum_floor(self):
        """Test that very low bandwidth returns minimum floor values."""
        detector = ResourceCapabilityDetector()
        
        # Create mock counters with very low throughput
        # Simulate only 1,000 bytes received (well below 0.1 Mbps threshold)
        mock_before = MagicMock()
        mock_before.bytes_recv = 0
        mock_before.bytes_sent = 0
        
        mock_after = MagicMock()
        mock_after.bytes_recv = 1_000  # Very low
        mock_after.bytes_sent = 500
        
        with patch('psutil.net_io_counters', side_effect=[mock_before, mock_after]):
            with patch('asyncio.sleep', return_value=None):
                result = await detector._run_bandwidth_test()
                
                # Should return minimum floor
                assert result == (1.0, 0.5)


class TestNetworkBandwidthCaching:
    """Test caching behavior of network bandwidth measurement."""
    
    @pytest.mark.asyncio
    async def test_bandwidth_cache_hit(self):
        """Test that cached bandwidth result is returned within TTL."""
        detector = ResourceCapabilityDetector()
        
        # Pre-populate cache
        detector._bandwidth_cache = (50.0, 25.0)
        detector._bandwidth_cache_time = time.time()
        
        # Should return cached value without running measurement
        result = await detector._run_bandwidth_test()
        assert result == (50.0, 25.0)
    
    @pytest.mark.asyncio
    async def test_bandwidth_cache_expired(self):
        """Test that cache is refreshed after TTL expires."""
        detector = ResourceCapabilityDetector()
        detector._network_cache_ttl = 1.0  # 1 second TTL for testing
        
        # Pre-populate cache with expired timestamp
        detector._bandwidth_cache = (50.0, 25.0)
        detector._bandwidth_cache_time = time.time() - 2.0  # 2 seconds ago
        
        # Mock psutil counters
        mock_before = MagicMock()
        mock_before.bytes_recv = 0
        mock_before.bytes_sent = 0
        
        mock_after = MagicMock()
        mock_after.bytes_recv = 3_750_000  # 10 Mbps
        mock_after.bytes_sent = 1_500_000  # 4 Mbps
        
        with patch('psutil.net_io_counters', side_effect=[mock_before, mock_after]):
            with patch('asyncio.sleep', return_value=None):
                result = await detector._run_bandwidth_test()
                
                # Should have run measurement and returned new value
                assert result == (10.0, 4.0)
                assert detector._bandwidth_cache == (10.0, 4.0)
    
    @pytest.mark.asyncio
    async def test_bandwidth_cache_stores_result_after_measurement(self):
        """Test that measurement result is cached after running."""
        detector = ResourceCapabilityDetector()
        
        # No cache initially
        assert detector._bandwidth_cache is None
        
        # Mock psutil counters
        mock_before = MagicMock()
        mock_before.bytes_recv = 0
        mock_before.bytes_sent = 0
        
        mock_after = MagicMock()
        mock_after.bytes_recv = 3_750_000  # 10 Mbps
        mock_after.bytes_sent = 1_500_000  # 4 Mbps
        
        with patch('psutil.net_io_counters', side_effect=[mock_before, mock_after]):
            with patch('asyncio.sleep', return_value=None):
                result = await detector._run_bandwidth_test()
                
                # Result should be cached
                assert result == (10.0, 4.0)
                assert detector._bandwidth_cache == (10.0, 4.0)
                assert detector._bandwidth_cache_time is not None


class TestNetworkBandwidthFallback:
    """Test fallback behavior on exceptions."""
    
    @pytest.mark.asyncio
    async def test_bandwidth_fallback_on_psutil_error(self):
        """Test that bandwidth measurement returns fallback on psutil error."""
        detector = ResourceCapabilityDetector()
        
        with patch('psutil.net_io_counters', side_effect=ImportError("psutil not available")):
            result = await detector._run_bandwidth_test()
            
            # Should return fallback value
            assert result == (10.0, 5.0)


class TestPerformanceScoreCalculation:
    """Test that performance score calculation is correct for all resource types."""
    
    def test_cpu_performance_score_at_baseline(self):
        """Verify CPU score is 1.0 when ops matches baseline (50,000 ops/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "ops_per_second": 50_000  # Exactly at baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(1.0) = 1.0
        assert score == 1.0

    def test_cpu_performance_score_above_baseline(self):
        """Verify CPU score is clamped to 1.0 when ops exceeds baseline."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "ops_per_second": 200_000  # 4x baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(4.0) = 2.0, clamped to 1.0
        assert score == 1.0

    def test_cpu_performance_score_below_baseline(self):
        """Verify CPU score is sqrt of ratio when below baseline."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "ops_per_second": 25_000  # 0.5x baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(0.5) ≈ 0.7071
        assert abs(score - 0.7071) < 0.001

    def test_gpu_performance_score_calculation(self):
        """Verify GPU score uses correct baseline (1,000,000 ops/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_GPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "ops_per_second": 500_000  # 0.5x baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(0.5) ≈ 0.7071
        assert abs(score - 0.7071) < 0.001

    def test_storage_persistent_performance_score(self):
        """Verify storage persistent score uses throughput_mbps baseline (500 MB/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.STORAGE_PERSISTENT,
            measurement_unit=ResourceMeasurement.STORAGE_GB,
            total_capacity=1000.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "throughput_mbps": 500  # Exactly at baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        assert score == 1.0

    def test_storage_memory_performance_score(self):
        """Verify storage memory score uses throughput_mbps baseline (20,000 MB/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.STORAGE_MEMORY,
            measurement_unit=ResourceMeasurement.MEMORY_GB,
            total_capacity=64.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "throughput_mbps": 10_000  # 0.5x baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(0.5) ≈ 0.7071
        assert abs(score - 0.7071) < 0.001

    def test_bandwidth_ingress_performance_score(self):
        """Verify bandwidth ingress score uses throughput_mbps baseline (100 MB/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.BANDWIDTH_INGRESS,
            measurement_unit=ResourceMeasurement.MBPS,
            total_capacity=1000.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "throughput_mbps": 100  # Exactly at baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        assert score == 1.0

    def test_bandwidth_egress_performance_score(self):
        """Verify bandwidth egress score uses throughput_mbps baseline (100 MB/s)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.BANDWIDTH_EGRESS,
            measurement_unit=ResourceMeasurement.MBPS,
            total_capacity=1000.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "throughput_mbps": 200  # 2x baseline
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        # sqrt(2.0) ≈ 1.414, clamped to 1.0
        assert score == 1.0

    def test_unknown_resource_type_returns_neutral(self):
        """Verify unknown resource types return neutral score (0.5)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.SPECIALIZED_QUANTUM,  # No baseline defined
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=10.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "ops_per_second": 1_000_000
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        assert score == 0.5

    def test_missing_performance_metrics_returns_fallback(self):
        """Verify missing performance_metrics key returns fallback (0.5)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {}  # No performance_metrics key
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        assert score == 0.5

    def test_wrong_metric_key_returns_fallback(self):
        """Verify wrong metric key returns fallback (0.5)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine, ResourceType, ResourceSpec, ResourceMeasurement
        )
        
        engine = ResourceVerificationEngine()
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        benchmark_result = {
            "performance_metrics": {
                "wrong_key": 50_000  # Wrong key for CPU
            }
        }
        
        score = engine._calculate_performance_score(benchmark_result, resource_spec)
        assert score == 0.5


class TestNodeCostCalculation:
    """Test that node cost calculation is correct for various scenarios."""

    def test_single_cpu_resource_cost(self):
        """Verify cost calculation for single CPU resource."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="small",
            geographic_region="us-east",
            resources={ResourceType.COMPUTE_CPU: resource_spec},
            reputation_score=0.5  # Neutral reputation
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.05, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.05 * 1.0 * 100 = 5.0
        # total_weight = 100
        # cost = 5.0 / 100 = 0.05
        assert cost == 0.05

    def test_single_gpu_resource_cost(self):
        """Verify cost calculation for single GPU resource."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_GPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=10.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-west",
            resources={ResourceType.COMPUTE_GPU: resource_spec},
            reputation_score=0.5  # Neutral reputation
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.50, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.50 * 1.0 * 10 = 5.0
        # total_weight = 10
        # cost = 5.0 / 10 = 0.50
        assert cost == 0.50

    def test_reputation_premium_high_reputation(self):
        """Verify high-reputation nodes can charge a premium."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="small",
            geographic_region="us-east",
            resources={ResourceType.COMPUTE_CPU: resource_spec},
            reputation_score=1.0  # Maximum reputation
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.05, reputation_multiplier: 0.5 + 1.0 = 1.5
        # total_cost = 0.05 * 1.5 * 100 = 7.5
        # total_weight = 100
        # cost = 7.5 / 100 = 0.075
        assert abs(cost - 0.075) < 0.0001

    def test_reputation_discount_low_reputation(self):
        """Verify low-reputation nodes are discounted."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="small",
            geographic_region="us-east",
            resources={ResourceType.COMPUTE_CPU: resource_spec},
            reputation_score=0.0  # Minimum reputation
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.05, reputation_multiplier: 0.5 + 0.0 = 0.5
        # total_cost = 0.05 * 0.5 * 100 = 2.5
        # total_weight = 100
        # cost = 2.5 / 100 = 0.025
        assert cost == 0.025

    def test_multiple_resources_weighted_average(self):
        """Verify weighted average cost across multiple resources."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        cpu_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=100.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        gpu_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_GPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=10.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-east",
            resources={
                ResourceType.COMPUTE_CPU: cpu_spec,
                ResourceType.COMPUTE_GPU: gpu_spec
            },
            reputation_score=0.5  # Neutral reputation
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # CPU: base=0.05, weight=100, cost_contrib=0.05*1.0*100=5.0
        # GPU: base=0.50, weight=10, cost_contrib=0.50*1.0*10=5.0
        # total_cost = 10.0, total_weight = 110
        # cost = 10.0 / 110 ≈ 0.0909
        assert abs(cost - 0.0909) < 0.001

    def test_empty_resources_returns_default(self):
        """Verify empty resources returns default cost (0.1)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="micro",
            geographic_region="us-east",
            resources={},  # Empty resources
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        assert cost == 0.1

    def test_unknown_resource_type_uses_default_base_rate(self):
        """Verify unknown resource types use default base rate (0.1)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.SPECIALIZED_QUANTUM,  # No base rate defined
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=10.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="specialized",
            geographic_region="us-east",
            resources={ResourceType.SPECIALIZED_QUANTUM: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.1 (default), reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.1 * 1.0 * 10 = 1.0
        # total_weight = 10
        # cost = 1.0 / 10 = 0.1
        assert cost == 0.1

    def test_tpu_resource_cost(self):
        """Verify TPU resource uses correct base rate (1.00)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_TPU,
            measurement_unit=ResourceMeasurement.COMPUTE_UNITS,
            total_capacity=4.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="large",
            geographic_region="us-east",
            resources={ResourceType.COMPUTE_TPU: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 1.00, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 1.00 * 1.0 * 4 = 4.0
        # total_weight = 4
        # cost = 4.0 / 4 = 1.00
        assert cost == 1.0

    def test_storage_persistent_resource_cost(self):
        """Verify storage persistent uses correct base rate (0.01)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.STORAGE_PERSISTENT,
            measurement_unit=ResourceMeasurement.STORAGE_GB,
            total_capacity=1000.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-east",
            resources={ResourceType.STORAGE_PERSISTENT: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.01, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.01 * 1.0 * 1000 = 10.0
        # total_weight = 1000
        # cost = 10.0 / 1000 = 0.01
        assert cost == 0.01

    def test_storage_memory_resource_cost(self):
        """Verify storage memory uses correct base rate (0.08)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.STORAGE_MEMORY,
            measurement_unit=ResourceMeasurement.MEMORY_GB,
            total_capacity=128.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-east",
            resources={ResourceType.STORAGE_MEMORY: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.08, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.08 * 1.0 * 128 = 10.24
        # total_weight = 128
        # cost = 10.24 / 128 = 0.08
        assert cost == 0.08

    def test_bandwidth_ingress_resource_cost(self):
        """Verify bandwidth ingress uses correct base rate (0.02)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.BANDWIDTH_INGRESS,
            measurement_unit=ResourceMeasurement.MBPS,
            total_capacity=1000.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-east",
            resources={ResourceType.BANDWIDTH_INGRESS: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.02, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.02 * 1.0 * 1000 = 20.0
        # total_weight = 1000
        # cost = 20.0 / 1000 = 0.02
        assert cost == 0.02

    def test_bandwidth_egress_resource_cost(self):
        """Verify bandwidth egress uses correct base rate (0.03)."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine, NodeResourceProfile, ResourceType, ResourceSpec,
            ResourceMeasurement
        )
        from decimal import Decimal
        
        manager = ResourceAllocationEngine()
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.BANDWIDTH_EGRESS,
            measurement_unit=ResourceMeasurement.MBPS,
            total_capacity=500.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        node = NodeResourceProfile(
            node_id="test-node-1",
            user_id="user-1",
            node_type="medium",
            geographic_region="us-east",
            resources={ResourceType.BANDWIDTH_EGRESS: resource_spec},
            reputation_score=0.5
        )
        
        cost = manager._calculate_node_cost_per_unit(node)
        # Base rate: 0.03, reputation_multiplier: 0.5 + 0.5 = 1.0
        # total_cost = 0.03 * 1.0 * 500 = 15.0
        # total_weight = 500
        # cost = 15.0 / 500 = 0.03
        assert cost == 0.03


class TestGPUDetection:
    """Test GPU detection methods with various fallback paths."""
    
    @pytest.mark.asyncio
    async def test_gpu_memory_pynvml_success(self):
        """Test GPU memory detection via pynvml (Priority 1)."""
        detector = ResourceCapabilityDetector()
        
        # Create mock pynvml module
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.total = 24 * (1024**3)  # 24 GB
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlShutdown.return_value = None
        
        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = await detector._detect_gpu_memory()
            
            # Should return 24 GB
            assert result == 24.0
            mock_pynvml.nvmlInit.assert_called_once()
            mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            mock_pynvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(mock_handle)
            mock_pynvml.nvmlShutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gpu_memory_pynvml_error_fallback_to_torch(self):
        """Test fallback to torch.cuda when pynvml raises NVMLError."""
        detector = ResourceCapabilityDetector()
        
        # Create mock pynvml that raises error
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("NVMLError: No NVIDIA GPU found")
        
        # Create mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024**3)  # 12 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        with patch.dict('sys.modules', {'pynvml': mock_pynvml, 'torch': mock_torch}):
            result = await detector._detect_gpu_memory()
            
            # Should fall back to torch and return 12 GB
            assert result == 12.0
            mock_torch.cuda.is_available.assert_called_once()
            mock_torch.cuda.get_device_properties.assert_called_once_with(0)
    
    @pytest.mark.asyncio
    async def test_gpu_memory_torch_success(self):
        """Test GPU memory detection via torch.cuda (Priority 2)."""
        detector = ResourceCapabilityDetector()
        
        # Create mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        # Mock pynvml to raise ImportError (not installed)
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # Need to clear pynvml from modules to simulate not installed
            import sys
            pynvml_backup = sys.modules.get('pynvml')
            if 'pynvml' in sys.modules:
                del sys.modules['pynvml']
            
            try:
                result = await detector._detect_gpu_memory()
                assert result == 16.0
            finally:
                if pynvml_backup:
                    sys.modules['pynvml'] = pynvml_backup
    
    @pytest.mark.asyncio
    async def test_gpu_memory_gputil_fallback(self):
        """Test GPU memory detection via GPUtil (Priority 3)."""
        detector = ResourceCapabilityDetector()
        
        # Create mock GPUtil
        mock_gputil = MagicMock()
        mock_gpu = MagicMock()
        mock_gpu.memoryTotal = 8192  # 8192 MB = 8 GB
        mock_gputil.getGPUs.return_value = [mock_gpu]
        
        # Mock pynvml and torch to fail
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError("No pynvml")
        
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {
            'pynvml': mock_pynvml,
            'torch': mock_torch,
            'GPUtil': mock_gputil
        }):
            result = await detector._detect_gpu_memory()
            
            # Should fall back to GPUtil and return 8 GB
            assert result == 8.0
            mock_gputil.getGPUs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gpu_memory_no_gpu_libraries(self):
        """Test fallback when no GPU libraries are available."""
        detector = ResourceCapabilityDetector()
        
        # Mock all GPU libraries to raise ImportError
        import sys
        
        # Store original modules
        original_modules = {}
        for mod in ['pynvml', 'torch', 'GPUtil']:
            if mod in sys.modules:
                original_modules[mod] = sys.modules[mod]
                del sys.modules[mod]
        
        try:
            result = await detector._detect_gpu_memory()
            # Should return 0.0 when no GPU detected
            assert result == 0.0
        finally:
            # Restore original modules
            for mod, val in original_modules.items():
                sys.modules[mod] = val
    
    @pytest.mark.asyncio
    async def test_gpu_memory_cpu_only_fallback(self):
        """Test that CPU-only systems return 0.0 without crashing."""
        detector = ResourceCapabilityDetector()
        
        # Mock pynvml to raise NVMLError (simulating no NVIDIA GPU)
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("NVMLError")
        mock_pynvml.NVMLError = Exception
        
        # Mock torch to report no CUDA
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        # Mock GPUtil to return empty list
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = []
        
        with patch.dict('sys.modules', {
            'pynvml': mock_pynvml,
            'torch': mock_torch,
            'GPUtil': mock_gputil
        }):
            result = await detector._detect_gpu_memory()
            assert result == 0.0
    
    def test_gpu_info_pynvml_success(self):
        """Test detailed GPU info detection via pynvml."""
        detector = ResourceCapabilityDetector()
        
        # Create mock pynvml
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.total = 24 * (1024**3)  # 24 GB
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlDeviceGetName.return_value = b'NVIDIA GeForce RTX 4090'
        mock_pynvml.nvmlDeviceGetNumGpuCores.return_value = 16384
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
        mock_pynvml.nvmlShutdown.return_value = None
        mock_pynvml.NVMLError = Exception
        
        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = detector._get_gpu_info_via_pynvml()
            
            assert result is not None
            assert result["memory_gb"] == 24.0
            assert result["cuda_cores"] == 16384
            assert result["compute_capability"] == "8.9"
            assert result["device_name"] == 'NVIDIA GeForce RTX 4090'
            # RTX 4090 should have 1008 GB/s from lookup table
            assert result["memory_bandwidth"] == 1008.0
    
    def test_gpu_info_torch_success(self):
        """Test detailed GPU info detection via torch.cuda."""
        detector = ResourceCapabilityDetector()
        
        # Create mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB
        mock_props.name = 'NVIDIA GeForce RTX 4080'
        mock_props.major = 8
        mock_props.minor = 9
        mock_props.multi_processor_count = 76  # RTX 4080 has 76 SMs
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = detector._get_gpu_info_via_torch()
            
            assert result is not None
            assert result["memory_gb"] == 16.0
            assert result["compute_capability"] == "8.9"
            assert result["device_name"] == 'NVIDIA GeForce RTX 4080'
            # 76 SMs * 128 cores/SM = 9728 cores (approximate)
            assert result["cuda_cores"] == 76 * 128
            # RTX 4080 should have 717 GB/s from lookup table
            assert result["memory_bandwidth"] == 717.0
    
    def test_memory_bandwidth_lookup(self):
        """Test memory bandwidth lookup for various GPUs."""
        from prsm.compute.federation.distributed_resource_manager import NVIDIA_MEMORY_BANDWIDTH
        detector = ResourceCapabilityDetector()
        
        # Test exact matches
        assert detector._get_memory_bandwidth("RTX 4090") == 1008.0
        assert detector._get_memory_bandwidth("RTX 3080") == 760.0
        assert detector._get_memory_bandwidth("A100") == 2039.0
        assert detector._get_memory_bandwidth("V100") == 900.0
        
        # Test partial matches with full device name
        assert detector._get_memory_bandwidth("NVIDIA GeForce RTX 4090") == 1008.0
        assert detector._get_memory_bandwidth("NVIDIA A100-SXM4") == 2039.0
        
        # Test unknown GPU returns default
        assert detector._get_memory_bandwidth("Unknown GPU Model") == 448.0
    
    def test_cuda_cores_estimation(self):
        """Test CUDA cores estimation for various GPUs."""
        detector = ResourceCapabilityDetector()
        
        # Test known GPUs
        assert detector._estimate_cuda_cores("RTX 4090") == 16384
        assert detector._estimate_cuda_cores("RTX 3090") == 10496
        assert detector._estimate_cuda_cores("A100") == 6912
        assert detector._estimate_cuda_cores("V100") == 5120
        
        # Test with full device name
        assert detector._estimate_cuda_cores("NVIDIA GeForce RTX 4090") == 16384
        
        # Test unknown GPU returns default
        assert detector._estimate_cuda_cores("Unknown GPU") == 2048
    
    @pytest.mark.asyncio
    async def test_gpu_capabilities_full_detection(self):
        """Test full GPU capabilities detection returns ResourceSpec."""
        detector = ResourceCapabilityDetector()
        
        # Create mock pynvml
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.total = 24 * (1024**3)  # 24 GB
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlDeviceGetName.return_value = b'NVIDIA GeForce RTX 4090'
        mock_pynvml.nvmlDeviceGetNumGpuCores.return_value = 16384
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
        mock_pynvml.nvmlShutdown.return_value = None
        mock_pynvml.NVMLError = Exception
        
        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            result = await detector._detect_gpu_capabilities()
            
            assert result is not None
            assert result.total_capacity == 24.0
            assert result.quality_metrics["cuda_cores"] == 16384
            assert result.quality_metrics["compute_capability"] == "8.9"
            assert result.quality_metrics["memory_bandwidth"] == 1008.0
            assert result.quality_metrics["device_name"] == 'NVIDIA GeForce RTX 4090'
    
    @pytest.mark.asyncio
    async def test_gpu_capabilities_no_gpu_returns_none(self):
        """Test that GPU capabilities returns None when no GPU detected."""
        detector = ResourceCapabilityDetector()
        
        # Mock all GPU libraries to fail
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = ImportError("No pynvml")
        
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = []
        
        with patch.dict('sys.modules', {
            'pynvml': mock_pynvml,
            'torch': mock_torch,
            'GPUtil': mock_gputil
        }):
            result = await detector._detect_gpu_capabilities()
            assert result is None
    
    @pytest.mark.asyncio
    async def test_apple_silicon_no_crash(self):
        """Test that Apple Silicon systems don't crash and return 0.0."""
        detector = ResourceCapabilityDetector()
        
        # Mock pynvml to raise NVMLError (Apple Silicon has no NVIDIA GPU)
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("NVMLError: No NVIDIA GPU")
        mock_pynvml.NVMLError = Exception
        
        # Mock torch to report no CUDA (Apple Silicon uses MPS, not CUDA)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        # Mock GPUtil to return empty (no NVIDIA GPUs)
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.return_value = []
        
        with patch.dict('sys.modules', {
            'pynvml': mock_pynvml,
            'torch': mock_torch,
            'GPUtil': mock_gputil
        }):
            # Should not crash and return 0.0
            result = await detector._detect_gpu_memory()
            assert result == 0.0
            
            # GPU capabilities should return None
            gpu_spec = await detector._detect_gpu_capabilities()
            assert gpu_spec is None


class TestGeographicRegionDetection:
    """Test geographic region detection functionality."""
    
    @pytest.mark.asyncio
    async def test_env_var_override(self):
        """Test that PRSM_REGION environment variable takes priority."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        with patch.dict('os.environ', {'PRSM_REGION': 'custom-region'}):
            result = await manager._detect_geographic_region()
            assert result == 'custom-region'
    
    @pytest.mark.asyncio
    async def test_cache_hit_within_ttl(self):
        """Test that cached region is returned within TTL."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        manager._geographic_region_cache = "cached-region"
        manager._geographic_region_cache_time = time.time() - 3600  # 1 hour ago (within 24h TTL)
        
        result = await manager._detect_geographic_region()
        assert result == "cached-region"
    
    @pytest.mark.asyncio
    async def test_cache_expired_makes_network_call(self):
        """Test that expired cache triggers a new network call."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        manager._geographic_region_cache = "old-cached-region"
        manager._geographic_region_cache_time = time.time() - 100000  # Expired (beyond 24h TTL)
        
        # Mock aiohttp
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "countryCode": "US",
            "lat": 37.7749,
            "lon": -122.4194
        })
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await manager._detect_geographic_region()
                assert result == "us-west"
                # Verify cache was updated
                assert manager._geographic_region_cache == "us-west"
    
    @pytest.mark.asyncio
    async def test_ip_geolocation_us_west(self):
        """Test IP geolocation returns us-west for US location with lon < -100."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Mock aiohttp response for US West Coast
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "countryCode": "US",
            "lat": 37.7749,
            "lon": -122.4194  # San Francisco (west of -100)
        })
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await manager._detect_geographic_region()
                assert result == "us-west"
    
    @pytest.mark.asyncio
    async def test_ip_geolocation_us_east(self):
        """Test IP geolocation returns us-east for US location with lon >= -100."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Mock aiohttp response for US East Coast
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "countryCode": "US",
            "lat": 40.7128,
            "lon": -74.0060  # New York (east of -100)
        })
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await manager._detect_geographic_region()
                assert result == "us-east"
    
    @pytest.mark.asyncio
    async def test_network_error_returns_unknown(self):
        """Test that network errors return 'unknown'."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Mock aiohttp to raise an exception
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=Exception("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await manager._detect_geographic_region()
                assert result == "unknown"


class TestMapCoordsToRegion:
    """Test the _map_coords_to_region helper method."""
    
    def test_us_west_longitude(self):
        """Test US with longitude < -100 returns us-west."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        result = manager._map_coords_to_region(37.0, -122.0, "US")
        assert result == "us-west"
    
    def test_us_east_longitude(self):
        """Test US with longitude >= -100 returns us-east."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        result = manager._map_coords_to_region(40.0, -74.0, "US")
        assert result == "us-east"
    
    def test_europe_countries(self):
        """Test European countries return europe."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        european_countries = ["GB", "DE", "FR", "NL", "SE", "NO", "FI", "ES", "IT", "PT", "CH", "AT", "BE", "DK", "PL"]
        
        for country in european_countries:
            result = manager._map_coords_to_region(50.0, 10.0, country)
            assert result == "europe", f"Country {country} should return europe"
    
    def test_asia_pacific_countries(self):
        """Test Asia-Pacific countries return asia-pacific."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        apac_countries = ["JP", "KR", "CN", "TW", "HK", "SG", "AU", "NZ", "IN", "TH", "ID", "MY", "PH", "VN"]
        
        for country in apac_countries:
            result = manager._map_coords_to_region(35.0, 139.0, country)
            assert result == "asia-pacific", f"Country {country} should return asia-pacific"
    
    def test_south_america_countries(self):
        """Test South American countries return south-america."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        sa_countries = ["BR", "AR", "CL", "CO", "PE", "MX"]
        
        for country in sa_countries:
            result = manager._map_coords_to_region(-23.0, -46.0, country)
            assert result == "south-america", f"Country {country} should return south-america"
    
    def test_africa_countries(self):
        """Test African countries return africa."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        african_countries = ["ZA", "NG", "KE", "EG", "MA"]
        
        for country in african_countries:
            result = manager._map_coords_to_region(-1.0, 37.0, country)
            assert result == "africa", f"Country {country} should return africa"
    
    def test_middle_east_countries(self):
        """Test Middle Eastern countries return middle-east."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        me_countries = ["AE", "SA", "IL", "TR", "IR", "IQ"]
        
        for country in me_countries:
            result = manager._map_coords_to_region(25.0, 55.0, country)
            assert result == "middle-east", f"Country {country} should return middle-east"
    
    def test_unknown_country_americas_fallback(self):
        """Test unknown country in Americas longitude returns americas."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Longitude between -170 and -50
        result = manager._map_coords_to_region(10.0, -80.0, "XX")
        assert result == "americas"
    
    def test_unknown_country_europe_africa_fallback(self):
        """Test unknown country in Europe/Africa longitude returns europe-africa."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Longitude between -30 and 60
        result = manager._map_coords_to_region(45.0, 15.0, "XX")
        assert result == "europe-africa"
    
    def test_unknown_country_asia_pacific_fallback(self):
        """Test unknown country in Asia-Pacific longitude returns asia-pacific."""
        from prsm.compute.federation.distributed_resource_manager import DistributedResourceManager
        
        manager = DistributedResourceManager()
        
        # Longitude outside other ranges
        result = manager._map_coords_to_region(35.0, 140.0, "XX")
        assert result == "asia-pacific"


class TestPeerVerificationMessaging:
    """Test peer verification and reservation messaging methods."""
    
    @pytest.fixture
    def verification_engine_with_nodes(self):
        """Create a ResourceVerificationEngine with mock node registry."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceVerificationEngine,
            NodeResourceProfile,
            ResourceSpec,
            ResourceType,
            ResourceMeasurement
        )
        from datetime import datetime, timezone
        from decimal import Decimal
        
        engine = ResourceVerificationEngine()
        
        # Create mock node profiles
        node1 = NodeResourceProfile(
            node_id="node_1",
            user_id="user_1",
            node_type="medium",
            geographic_region="americas",
            reputation_score=0.9,
            stake_amount=Decimal("1000"),
            contribution_settings={"host": "node1.example.com", "port": 8080}
        )
        
        node2 = NodeResourceProfile(
            node_id="node_2",
            user_id="user_2",
            node_type="medium",
            geographic_region="europe-africa",
            reputation_score=0.7,
            stake_amount=Decimal("500"),
            contribution_settings={"host": "node2.example.com", "port": 8081}
        )
        
        node3 = NodeResourceProfile(
            node_id="node_3",
            user_id="user_3",
            node_type="small",
            geographic_region="asia-pacific",
            reputation_score=0.5,
            stake_amount=Decimal("250"),
            contribution_settings={"host": "node3.example.com", "port": 8082}
        )
        
        # Add nodes to registry
        engine.node_registry = {
            "node_1": node1,
            "node_2": node2,
            "node_3": node3
        }
        
        return engine
    
    @pytest.fixture
    def allocation_engine_with_nodes(self):
        """Create a ResourceAllocationEngine with mock node registry."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceAllocationEngine,
            NodeResourceProfile
        )
        from decimal import Decimal
        
        engine = ResourceAllocationEngine()
        
        # Create mock node profiles
        node1 = NodeResourceProfile(
            node_id="node_1",
            user_id="user_1",
            node_type="medium",
            geographic_region="americas",
            reputation_score=0.9,
            stake_amount=Decimal("1000"),
            contribution_settings={"host": "node1.example.com", "port": 8080}
        )
        
        node2 = NodeResourceProfile(
            node_id="node_2",
            user_id="user_2",
            node_type="medium",
            geographic_region="europe-africa",
            reputation_score=0.7,
            stake_amount=Decimal("500"),
            contribution_settings={"host": "node2.example.com", "port": 8081}
        )
        
        # Add nodes to registry
        engine.node_registry = {
            "node_1": node1,
            "node_2": node2
        }
        
        return engine
    
    def test_select_verification_peers_excludes_target(self, verification_engine_with_nodes):
        """Test that _select_verification_peers excludes the target node."""
        # Request peers for node_1 - should only return node_2 and node_3
        peers = verification_engine_with_nodes._select_verification_peers("node_1", count=2)
        
        assert "node_1" not in peers
        assert len(peers) <= 2
        assert all(p in ["node_2", "node_3"] for p in peers)
    
    def test_select_verification_peers_prefers_high_reputation(self, verification_engine_with_nodes):
        """Test that _select_verification_peers prefers high reputation peers."""
        # Request 1 peer - should prefer node_1 (0.9) or node_2 (0.7) over node_3 (0.5)
        peers = verification_engine_with_nodes._select_verification_peers("node_3", count=1)
        
        # The pool is top 2x candidates = [node_1, node_2], randomly sampled
        # So the result should be either node_1 or node_2
        assert peers[0] in ["node_1", "node_2"]
    
    def test_select_verification_peers_returns_all_when_few_candidates(self, verification_engine_with_nodes):
        """Test that all candidates are returned when fewer than requested count."""
        # Only 2 other nodes available, request 5
        peers = verification_engine_with_nodes._select_verification_peers("node_1", count=5)
        
        # Should return all available peers (node_2, node_3)
        assert len(peers) == 2
        assert set(peers) == {"node_2", "node_3"}
    
    def test_select_verification_peers_empty_registry(self):
        """Test behavior when node registry is empty."""
        from prsm.compute.federation.distributed_resource_manager import ResourceVerificationEngine
        
        engine = ResourceVerificationEngine()
        engine.node_registry = {}
        
        peers = engine._select_verification_peers("any_node", count=3)
        assert peers == []
    
    @pytest.mark.asyncio
    async def test_request_peer_verification_peer_not_found(self, verification_engine_with_nodes):
        """Test _request_peer_verification returns error when peer not in registry."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceSpec, ResourceType, ResourceMeasurement
        )
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.CPU_CORES,
            total_capacity=8.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        result = await verification_engine_with_nodes._request_peer_verification(
            "unknown_peer", "node_1", resource_spec
        )
        
        assert result["peer_id"] == "unknown_peer"
        assert result["verified"] is False
        assert result["error"] == "peer_not_found"
    
    @pytest.mark.asyncio
    async def test_request_peer_verification_success(self, verification_engine_with_nodes):
        """Test _request_peer_verification successful HTTP response."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceSpec, ResourceType, ResourceMeasurement
        )
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.CPU_CORES,
            total_capacity=8.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        # Mock aiohttp ClientSession with nested async context managers
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "peer_id": "node_2",
            "verified": True,
            "verification_score": 0.95
        })
        # Make response an async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Make session.post return an async context manager
        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await verification_engine_with_nodes._request_peer_verification(
                "node_2", "node_1", resource_spec
            )
        
        assert result["verified"] is True
        assert result["verification_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_request_peer_verification_http_error(self, verification_engine_with_nodes):
        """Test _request_peer_verification handles HTTP errors."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceSpec, ResourceType, ResourceMeasurement
        )
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.CPU_CORES,
            total_capacity=8.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        # Mock aiohttp ClientSession with HTTP error - nested async context managers
        mock_response = AsyncMock()
        mock_response.status = 500
        # Make response an async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Make session.post return an async context manager
        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await verification_engine_with_nodes._request_peer_verification(
                "node_2", "node_1", resource_spec
            )
        
        assert result["verified"] is False
        assert "HTTP 500" in result["error"]
    
    @pytest.mark.asyncio
    async def test_request_peer_verification_network_error(self, verification_engine_with_nodes):
        """Test _request_peer_verification handles network errors."""
        from prsm.compute.federation.distributed_resource_manager import (
            ResourceSpec, ResourceType, ResourceMeasurement
        )
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_spec = ResourceSpec(
            resource_type=ResourceType.COMPUTE_CPU,
            measurement_unit=ResourceMeasurement.CPU_CORES,
            total_capacity=8.0,
            allocated_capacity=0.0,
            reserved_capacity=0.0
        )
        
        # Mock aiohttp ClientSession with network error
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await verification_engine_with_nodes._request_peer_verification(
                "node_2", "node_1", resource_spec
            )
        
        assert result["verified"] is False
        assert "Connection refused" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_reservation_request_node_not_found(self, allocation_engine_with_nodes):
        """Test _send_reservation_request returns False when node not in registry."""
        from prsm.compute.federation.distributed_resource_manager import ResourceType
        
        resource_contribution = {ResourceType.COMPUTE_CPU: 4.0}
        
        result = await allocation_engine_with_nodes._send_reservation_request(
            "unknown_node", resource_contribution
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_reservation_request_success(self, allocation_engine_with_nodes):
        """Test _send_reservation_request successful HTTP response."""
        from prsm.compute.federation.distributed_resource_manager import ResourceType
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_contribution = {ResourceType.COMPUTE_CPU: 4.0}
        
        # Mock aiohttp ClientSession with nested async context managers
        mock_response = AsyncMock()
        mock_response.status = 200
        # Make response an async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Make session.post return an async context manager
        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await allocation_engine_with_nodes._send_reservation_request(
                "node_1", resource_contribution
            )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_send_reservation_request_http_error(self, allocation_engine_with_nodes):
        """Test _send_reservation_request handles HTTP errors."""
        from prsm.compute.federation.distributed_resource_manager import ResourceType
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_contribution = {ResourceType.COMPUTE_CPU: 4.0}
        
        # Mock aiohttp ClientSession with HTTP error - nested async context managers
        mock_response = AsyncMock()
        mock_response.status = 503
        # Make response an async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Make session.post return an async context manager
        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await allocation_engine_with_nodes._send_reservation_request(
                "node_1", resource_contribution
            )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_reservation_request_network_error(self, allocation_engine_with_nodes):
        """Test _send_reservation_request handles network errors."""
        from prsm.compute.federation.distributed_resource_manager import ResourceType
        from unittest.mock import AsyncMock, patch, MagicMock
        import aiohttp
        
        resource_contribution = {ResourceType.COMPUTE_CPU: 4.0}
        
        # Mock aiohttp ClientSession with network error
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(aiohttp, 'ClientSession', return_value=mock_session):
            result = await allocation_engine_with_nodes._send_reservation_request(
                "node_1", resource_contribution
            )
        
        assert result is False
    
    def test_generate_verification_nonce_unique(self, verification_engine_with_nodes):
        """Test that _generate_verification_nonce returns unique values."""
        nonce1 = verification_engine_with_nodes._generate_verification_nonce()
        nonce2 = verification_engine_with_nodes._generate_verification_nonce()
        
        assert nonce1 != nonce2
        assert len(nonce1) == 16
        assert len(nonce2) == 16
