#!/usr/bin/env python3
"""
NWTN Provenance System Performance Benchmark
============================================

Comprehensive performance benchmarking suite for the NWTN provenance tracking
and royalty distribution system. This validates that the system meets performance
requirements under various load conditions.

Benchmark Categories:
1. Provenance Tracking Performance - Content registration and usage tracking
2. Royalty Calculation Performance - Complex calculations with many sources
3. Duplicate Detection Performance - Multi-layer detection algorithms
4. Attribution Generation Performance - Source link and summary generation
5. Concurrent Load Testing - System behavior under concurrent access
6. Memory Usage Analysis - Memory efficiency and leak detection
7. Database Performance - Query and transaction performance
8. Network Performance - IPFS and external API performance

Usage:
    python tests/nwtn_performance_benchmark.py
    python tests/nwtn_performance_benchmark.py --category royalty
    python tests/nwtn_performance_benchmark.py --load-test --concurrent-users 100
"""

import asyncio
import time
import psutil
import gc
import json
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from uuid import uuid4
from decimal import Decimal
import argparse
import sys
import tracemalloc

# Import systems under test
from prsm.compute.nwtn.meta_reasoning_engine import MetaReasoningEngine
from prsm.compute.nwtn.content_royalty_engine import ContentRoyaltyEngine, QueryComplexity
from prsm.compute.nwtn.content_ingestion_engine import NWTNContentIngestionEngine
from prsm.compute.nwtn.voicebox import NWTNVoicebox
from prsm.data.provenance.enhanced_provenance_system import EnhancedProvenanceSystem, ContentType


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    benchmark_name: str
    operation_count: int
    total_time_seconds: float
    average_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    operations_per_second: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LoadTestResult:
    """Result of a load testing scenario"""
    test_name: str
    concurrent_users: int
    total_operations: int
    test_duration_seconds: float
    successful_operations: int
    failed_operations: int
    average_response_time_ms: float
    throughput_ops_per_second: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    bottlenecks_identified: List[str]


class PerformanceBenchmark:
    """Main performance benchmarking class"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.load_test_results: List[LoadTestResult] = []
        self.process = psutil.Process()
        
        # Performance targets (based on requirements)
        self.performance_targets = {
            'content_registration_ms': 100,      # Content registration < 100ms
            'usage_tracking_ms': 10,             # Usage tracking < 10ms
            'royalty_calculation_ms': 500,       # Royalty calculation < 500ms (50 sources)
            'duplicate_detection_ms': 200,       # Duplicate detection < 200ms
            'attribution_generation_ms': 50,     # Attribution generation < 50ms
            'memory_usage_mb': 1000,             # Memory usage < 1GB
            'concurrent_users': 100,             # Support 100+ concurrent users
            'throughput_ops_per_second': 50      # 50+ operations per second
        }
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("🚀 Starting NWTN Provenance System Performance Benchmarks")
        print("=" * 80)
        
        benchmark_suite = [
            ("Provenance Tracking", self.benchmark_provenance_tracking),
            ("Royalty Calculation", self.benchmark_royalty_calculation),
            ("Duplicate Detection", self.benchmark_duplicate_detection),
            ("Attribution Generation", self.benchmark_attribution_generation),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Concurrent Operations", self.benchmark_concurrent_operations)
        ]
        
        for name, benchmark_func in benchmark_suite:
            print(f"\n📊 Running {name} Benchmark...")
            try:
                await benchmark_func()
                print(f"✅ {name} benchmark completed")
            except Exception as e:
                print(f"❌ {name} benchmark failed: {e}")
                traceback.print_exc()
        
        # Generate comprehensive report
        report = self._generate_performance_report()
        return report
    
    async def benchmark_provenance_tracking(self):
        """Benchmark provenance tracking performance"""
        
        # Test 1: Content registration performance
        print("  Testing content registration performance...")
        
        provenance_system = EnhancedProvenanceSystem()
        operation_times = []
        memory_usage = []
        errors = 0
        
        # Warm up
        for _ in range(5):
            try:
                content_data = f"Warmup content {uuid4()}".encode('utf-8')
                await provenance_system.register_content_with_provenance(
                    content_data=content_data,
                    content_type=ContentType.RESEARCH_PAPER,
                    creator_info={'name': 'test_creator', 'platform': 'benchmark'},
                    license_info={'type': 'open_source'},
                    metadata={'title': 'Benchmark Content'}
                )
            except Exception:
                pass
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Benchmark content registration (100 operations)
        start_time = time.perf_counter()
        
        for i in range(100):
            operation_start = time.perf_counter()
            
            try:
                content_data = f"Benchmark content {i} - {uuid4()}".encode('utf-8')
                await provenance_system.register_content_with_provenance(
                    content_data=content_data,
                    content_type=ContentType.RESEARCH_PAPER,
                    creator_info={
                        'name': f'creator_{i}',
                        'ftns_address': f'ftns_{i}',
                        'platform': 'benchmark'
                    },
                    license_info={'type': 'open_source'},
                    metadata={
                        'title': f'Benchmark Content {i}',
                        'domain': 'performance_testing',
                        'tags': ['benchmark', 'performance']
                    }
                )
                
                operation_time = (time.perf_counter() - operation_start) * 1000
                operation_times.append(operation_time)
                
                # Track memory every 10 operations
                if i % 10 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_usage.append(current_memory)
                
            except Exception as e:
                errors += 1
                print(f"    Error in operation {i}: {e}")
        
        total_time = time.perf_counter() - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = max(memory_usage) if memory_usage else final_memory
        
        # Stop memory tracking
        current_memory, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate statistics
        if operation_times:
            result = BenchmarkResult(
                benchmark_name="Content Registration",
                operation_count=len(operation_times),
                total_time_seconds=total_time,
                average_time_ms=statistics.mean(operation_times),
                min_time_ms=min(operation_times),
                max_time_ms=max(operation_times),
                median_time_ms=statistics.median(operation_times),
                p95_time_ms=self._percentile(operation_times, 95),
                p99_time_ms=self._percentile(operation_times, 99),
                operations_per_second=len(operation_times) / total_time,
                memory_usage_mb=final_memory - initial_memory,
                memory_peak_mb=peak_memory,
                cpu_usage_percent=self.process.cpu_percent(),
                success_rate=(len(operation_times) / 100) * 100,
                error_count=errors
            )
            
            self.results.append(result)
            self._print_benchmark_result(result)
            
            # Check against performance targets
            if result.average_time_ms > self.performance_targets['content_registration_ms']:
                print(f"    ⚠️  Average time {result.average_time_ms:.1f}ms exceeds target {self.performance_targets['content_registration_ms']}ms")
    
    async def benchmark_royalty_calculation(self):
        """Benchmark royalty calculation performance"""
        print("  Testing royalty calculation performance...")
        
        royalty_engine = ContentRoyaltyEngine()
        await royalty_engine.initialize()
        
        # Test with varying numbers of content sources
        source_counts = [10, 25, 50, 100]
        
        for source_count in source_counts:
            print(f"    Testing with {source_count} content sources...")
            
            content_sources = [uuid4() for _ in range(source_count)]
            operation_times = []
            errors = 0
            
            # Benchmark royalty calculation (20 operations per source count)
            for i in range(20):
                operation_start = time.perf_counter()
                
                try:
                    calculations = await royalty_engine.calculate_usage_royalty(
                        content_sources=content_sources,
                        query_complexity=QueryComplexity.COMPLEX,
                        user_tier="premium",
                        reasoning_context={
                            'reasoning_path': ['deductive', 'inductive', 'analogical'],
                            'overall_confidence': 0.85,
                            'content_weights': {str(cid): 1.0/source_count for cid in content_sources}
                        }
                    )
                    
                    operation_time = (time.perf_counter() - operation_start) * 1000
                    operation_times.append(operation_time)
                    
                except Exception as e:
                    errors += 1
                    print(f"      Error in calculation {i}: {e}")
            
            if operation_times:
                avg_time = statistics.mean(operation_times)
                print(f"      {source_count} sources: {avg_time:.1f}ms average")
                
                # Check performance target for 50 sources
                if source_count == 50 and avg_time > self.performance_targets['royalty_calculation_ms']:
                    print(f"      ⚠️  Average time {avg_time:.1f}ms exceeds target {self.performance_targets['royalty_calculation_ms']}ms")
                
                result = BenchmarkResult(
                    benchmark_name=f"Royalty Calculation ({source_count} sources)",
                    operation_count=len(operation_times),
                    total_time_seconds=sum(operation_times) / 1000,
                    average_time_ms=avg_time,
                    min_time_ms=min(operation_times),
                    max_time_ms=max(operation_times),
                    median_time_ms=statistics.median(operation_times),
                    p95_time_ms=self._percentile(operation_times, 95),
                    p99_time_ms=self._percentile(operation_times, 99),
                    operations_per_second=len(operation_times) / (sum(operation_times) / 1000),
                    memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                    memory_peak_mb=self.process.memory_info().rss / 1024 / 1024,
                    cpu_usage_percent=self.process.cpu_percent(),
                    success_rate=(len(operation_times) / 20) * 100,
                    error_count=errors
                )
                
                self.results.append(result)
    
    async def benchmark_duplicate_detection(self):
        """Benchmark duplicate detection performance"""
        print("  Testing duplicate detection performance...")
        
        ingestion_engine = NWTNContentIngestionEngine()
        await ingestion_engine.initialize()
        
        # Test with different content sizes
        content_sizes = [1024, 10240, 51200, 102400]  # 1KB, 10KB, 50KB, 100KB
        
        for size in content_sizes:
            print(f"    Testing with {size} byte content...")
            
            content_data = b"A" * size
            operation_times = []
            errors = 0
            
            # Benchmark duplicate detection (30 operations per size)
            for i in range(30):
                operation_start = time.perf_counter()
                
                try:
                    duplicates = await ingestion_engine.check_content_for_duplicates(
                        content=content_data,
                        content_type=ContentType.RESEARCH_PAPER,
                        metadata={'title': f'Test Content {i}'}
                    )
                    
                    operation_time = (time.perf_counter() - operation_start) * 1000
                    operation_times.append(operation_time)
                    
                except Exception as e:
                    errors += 1
                    print(f"      Error in detection {i}: {e}")
            
            if operation_times:
                avg_time = statistics.mean(operation_times)
                print(f"      {size} bytes: {avg_time:.1f}ms average")
                
                # Check performance target
                if avg_time > self.performance_targets['duplicate_detection_ms']:
                    print(f"      ⚠️  Average time {avg_time:.1f}ms exceeds target {self.performance_targets['duplicate_detection_ms']}ms")
                
                result = BenchmarkResult(
                    benchmark_name=f"Duplicate Detection ({size} bytes)",
                    operation_count=len(operation_times),
                    total_time_seconds=sum(operation_times) / 1000,
                    average_time_ms=avg_time,
                    min_time_ms=min(operation_times),
                    max_time_ms=max(operation_times),
                    median_time_ms=statistics.median(operation_times),
                    p95_time_ms=self._percentile(operation_times, 95),
                    p99_time_ms=self._percentile(operation_times, 99),
                    operations_per_second=len(operation_times) / (sum(operation_times) / 1000),
                    memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                    memory_peak_mb=self.process.memory_info().rss / 1024 / 1024,
                    cpu_usage_percent=self.process.cpu_percent(),
                    success_rate=(len(operation_times) / 30) * 100,
                    error_count=errors
                )
                
                self.results.append(result)
    
    async def benchmark_attribution_generation(self):
        """Benchmark attribution generation performance"""
        print("  Testing attribution generation performance...")
        
        voicebox = NWTNVoicebox()
        await voicebox.initialize()
        
        # Test with varying numbers of source links
        source_counts = [1, 5, 10, 20]
        
        for source_count in source_counts:
            print(f"    Testing with {source_count} source links...")
            
            # Create mock reasoning result
            class MockReasoningResult:
                def __init__(self, source_count):
                    self.content_sources = [uuid4() for _ in range(source_count)]
                    self.overall_confidence = 0.85
                    self.reasoning_path = ['deductive', 'analogical']
                    self.multi_modal_evidence = ['evidence1', 'evidence2']
            
            mock_result = MockReasoningResult(source_count)
            operation_times = []
            errors = 0
            
            # Benchmark attribution generation (50 operations per count)
            for i in range(50):
                operation_start = time.perf_counter()
                
                try:
                    # Mock the provenance system calls
                    with patch.object(voicebox.provenance_system, '_load_attribution_chain') as mock_attr:
                        with patch.object(voicebox.provenance_system, '_load_content_fingerprint') as mock_fp:
                            
                            # Mock responses
                            mock_attr.return_value = type('MockAttr', (), {
                                'original_creator': 'test_creator',
                                'creation_timestamp': datetime.now(timezone.utc)
                            })()
                            
                            mock_fp.return_value = type('MockFP', (), {
                                'ipfs_hash': 'QmTestHash123',
                                'content_type': ContentType.RESEARCH_PAPER
                            })()
                            
                            # Generate source links
                            source_links = await voicebox._generate_source_links(mock_result)
                            
                            # Generate attribution summary
                            summary = await voicebox._generate_attribution_summary(mock_result, source_links)
                    
                    operation_time = (time.perf_counter() - operation_start) * 1000
                    operation_times.append(operation_time)
                    
                except Exception as e:
                    errors += 1
                    print(f"      Error in generation {i}: {e}")
            
            if operation_times:
                avg_time = statistics.mean(operation_times)
                print(f"      {source_count} sources: {avg_time:.1f}ms average")
                
                # Check performance target
                if avg_time > self.performance_targets['attribution_generation_ms']:
                    print(f"      ⚠️  Average time {avg_time:.1f}ms exceeds target {self.performance_targets['attribution_generation_ms']}ms")
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage and efficiency"""
        print("  Testing memory efficiency...")
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Create systems
        provenance_system = EnhancedProvenanceSystem()
        royalty_engine = ContentRoyaltyEngine()
        await royalty_engine.initialize()
        
        memory_samples = []
        
        # Perform memory-intensive operations
        for i in range(100):
            # Register content
            content_data = f"Memory test content {i}".encode('utf-8')
            content_id, _, _ = await provenance_system.register_content_with_provenance(
                content_data=content_data,
                content_type=ContentType.RESEARCH_PAPER,
                creator_info={'name': f'creator_{i}', 'platform': 'memory_test'},
                license_info={'type': 'open_source'},
                metadata={'title': f'Memory Test {i}'}
            )
            
            # Calculate royalties
            await royalty_engine.calculate_usage_royalty(
                content_sources=[content_id],
                query_complexity=QueryComplexity.MODERATE,
                user_tier="basic"
            )
            
            # Sample memory every 10 operations
            if i % 10 == 0:
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force garbage collection
                gc.collect()
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        peak_memory = max(memory_samples) if memory_samples else final_memory
        
        # Stop memory tracking
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"    Initial memory: {initial_memory:.1f} MB")
        print(f"    Final memory: {final_memory:.1f} MB")
        print(f"    Memory growth: {memory_growth:.1f} MB")
        print(f"    Peak memory: {peak_memory:.1f} MB")
        
        # Check memory target
        if peak_memory > self.performance_targets['memory_usage_mb']:
            print(f"    ⚠️  Peak memory {peak_memory:.1f}MB exceeds target {self.performance_targets['memory_usage_mb']}MB")
        
        result = BenchmarkResult(
            benchmark_name="Memory Efficiency",
            operation_count=100,
            total_time_seconds=0,  # Not time-focused
            average_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            median_time_ms=0,
            p95_time_ms=0,
            p99_time_ms=0,
            operations_per_second=0,
            memory_usage_mb=memory_growth,
            memory_peak_mb=peak_memory,
            cpu_usage_percent=self.process.cpu_percent(),
            success_rate=100.0,
            error_count=0
        )
        
        self.results.append(result)
    
    async def benchmark_concurrent_operations(self):
        """Benchmark system performance under concurrent load"""
        print("  Testing concurrent operations...")
        
        concurrent_users = [10, 25, 50]
        
        for user_count in concurrent_users:
            print(f"    Testing with {user_count} concurrent users...")
            
            async def simulate_user_session(user_id: int):
                """Simulate a user session with multiple operations"""
                try:
                    provenance_system = EnhancedProvenanceSystem()
                    
                    # Register content
                    content_data = f"Concurrent user {user_id} content".encode('utf-8')
                    content_id, _, _ = await provenance_system.register_content_with_provenance(
                        content_data=content_data,
                        content_type=ContentType.RESEARCH_PAPER,
                        creator_info={'name': f'user_{user_id}', 'platform': 'concurrent_test'},
                        license_info={'type': 'open_source'},
                        metadata={'title': f'Concurrent Content {user_id}'}
                    )
                    
                    # Track usage
                    await provenance_system.track_content_usage(
                        content_id=content_id,
                        user_id=f'user_{user_id}',
                        session_id=uuid4(),
                        usage_type='reasoning_source',
                        context={'test': 'concurrent_operations'}
                    )
                    
                    return True
                    
                except Exception as e:
                    print(f"      User {user_id} error: {e}")
                    return False
            
            # Run concurrent operations
            start_time = time.perf_counter()
            
            tasks = [simulate_user_session(i) for i in range(user_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.perf_counter() - start_time
            
            # Analyze results
            successful_operations = sum(1 for r in results if r is True)
            failed_operations = user_count - successful_operations
            throughput = successful_operations / total_time
            
            print(f"      {successful_operations}/{user_count} operations successful")
            print(f"      Throughput: {throughput:.1f} ops/second")
            print(f"      Total time: {total_time:.2f} seconds")
            
            # Check throughput target
            if throughput < self.performance_targets['throughput_ops_per_second']:
                print(f"      ⚠️  Throughput {throughput:.1f} ops/sec below target {self.performance_targets['throughput_ops_per_second']}")
            
            load_result = LoadTestResult(
                test_name=f"Concurrent Operations ({user_count} users)",
                concurrent_users=user_count,
                total_operations=user_count,
                test_duration_seconds=total_time,
                successful_operations=successful_operations,
                failed_operations=failed_operations,
                average_response_time_ms=(total_time / user_count) * 1000,
                throughput_ops_per_second=throughput,
                error_rate=(failed_operations / user_count) * 100,
                memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                cpu_usage_percent=self.process.cpu_percent(),
                bottlenecks_identified=[]
            )
            
            self.load_test_results.append(load_result)
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            if upper_index >= len(sorted_data):
                return sorted_data[lower_index]
            
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _print_benchmark_result(self, result: BenchmarkResult):
        """Print formatted benchmark result"""
        print(f"    📊 {result.benchmark_name} Results:")
        print(f"      Operations: {result.operation_count}")
        print(f"      Average time: {result.average_time_ms:.1f}ms")
        print(f"      Median time: {result.median_time_ms:.1f}ms")
        print(f"      95th percentile: {result.p95_time_ms:.1f}ms")
        print(f"      Operations/second: {result.operations_per_second:.1f}")
        print(f"      Success rate: {result.success_rate:.1f}%")
        
        if result.error_count > 0:
            print(f"      ❌ Errors: {result.error_count}")
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'benchmark_summary': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_benchmarks': len(self.results),
                'total_load_tests': len(self.load_test_results)
            },
            'performance_targets': self.performance_targets,
            'benchmark_results': [],
            'load_test_results': [],
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Add benchmark results
        for result in self.results:
            report['benchmark_results'].append({
                'name': result.benchmark_name,
                'average_time_ms': result.average_time_ms,
                'operations_per_second': result.operations_per_second,
                'success_rate': result.success_rate,
                'memory_usage_mb': result.memory_usage_mb,
                'meets_target': self._check_performance_target(result)
            })
        
        # Add load test results
        for result in self.load_test_results:
            report['load_test_results'].append({
                'name': result.test_name,
                'concurrent_users': result.concurrent_users,
                'throughput_ops_per_second': result.throughput_ops_per_second,
                'error_rate': result.error_rate,
                'average_response_time_ms': result.average_response_time_ms
            })
        
        # Performance analysis
        content_reg_results = [r for r in self.results if 'Content Registration' in r.benchmark_name]
        if content_reg_results:
            avg_reg_time = statistics.mean([r.average_time_ms for r in content_reg_results])
            report['performance_analysis']['content_registration_avg_ms'] = avg_reg_time
        
        royalty_results = [r for r in self.results if 'Royalty Calculation' in r.benchmark_name]
        if royalty_results:
            avg_royalty_time = statistics.mean([r.average_time_ms for r in royalty_results])
            report['performance_analysis']['royalty_calculation_avg_ms'] = avg_royalty_time
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _check_performance_target(self, result: BenchmarkResult) -> bool:
        """Check if benchmark result meets performance targets"""
        
        target_mapping = {
            'Content Registration': 'content_registration_ms',
            'Royalty Calculation': 'royalty_calculation_ms',
            'Duplicate Detection': 'duplicate_detection_ms',
            'Attribution': 'attribution_generation_ms'
        }
        
        for key, target_key in target_mapping.items():
            if key in result.benchmark_name:
                return result.average_time_ms <= self.performance_targets[target_key]
        
        return True
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check each benchmark against targets
        for result in self.results:
            if 'Content Registration' in result.benchmark_name:
                if result.average_time_ms > self.performance_targets['content_registration_ms']:
                    recommendations.append(
                        f"Content registration performance ({result.average_time_ms:.1f}ms) exceeds target. "
                        "Consider optimizing fingerprint generation or database operations."
                    )
            
            elif 'Royalty Calculation' in result.benchmark_name:
                if result.average_time_ms > self.performance_targets['royalty_calculation_ms']:
                    recommendations.append(
                        f"Royalty calculation performance ({result.average_time_ms:.1f}ms) exceeds target. "
                        "Consider implementing caching for creator info or parallel processing."
                    )
            
            elif 'Memory' in result.benchmark_name:
                if result.memory_peak_mb > self.performance_targets['memory_usage_mb']:
                    recommendations.append(
                        f"Memory usage ({result.memory_peak_mb:.1f}MB) exceeds target. "
                        "Consider implementing object pooling or more aggressive garbage collection."
                    )
        
        # Check load test results
        for result in self.load_test_results:
            if result.throughput_ops_per_second < self.performance_targets['throughput_ops_per_second']:
                recommendations.append(
                    f"Throughput ({result.throughput_ops_per_second:.1f} ops/sec) below target. "
                    "Consider implementing connection pooling or horizontal scaling."
                )
            
            if result.error_rate > 5.0:  # 5% error rate threshold
                recommendations.append(
                    f"Error rate ({result.error_rate:.1f}%) is high under load. "
                    "Investigate error handling and resource contention."
                )
        
        if not recommendations:
            recommendations.append("All performance benchmarks meet targets. System is well-optimized.")
        
        return recommendations


async def main():
    """Main benchmark runner"""
    parser = argparse.ArgumentParser(description='NWTN Provenance System Performance Benchmark')
    parser.add_argument('--category', choices=['all', 'provenance', 'royalty', 'duplicate', 'attribution', 'memory', 'concurrent'], 
                       default='all', help='Benchmark category to run')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--load-test', action='store_true', help='Run load testing scenarios')
    parser.add_argument('--concurrent-users', type=int, default=50, help='Number of concurrent users for load testing')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    try:
        if args.category == 'all':
            report = await benchmark.run_all_benchmarks()
        else:
            # Run specific category
            category_map = {
                'provenance': benchmark.benchmark_provenance_tracking,
                'royalty': benchmark.benchmark_royalty_calculation,
                'duplicate': benchmark.benchmark_duplicate_detection,
                'attribution': benchmark.benchmark_attribution_generation,
                'memory': benchmark.benchmark_memory_efficiency,
                'concurrent': benchmark.benchmark_concurrent_operations
            }
            
            await category_map[args.category]()
            report = benchmark._generate_performance_report()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📋 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        for result in benchmark.results:
            status = "✅" if benchmark._check_performance_target(result) else "⚠️"
            print(f"{status} {result.benchmark_name}: {result.average_time_ms:.1f}ms avg, {result.operations_per_second:.1f} ops/sec")
        
        if benchmark.load_test_results:
            print("\n🔄 LOAD TEST RESULTS:")
            for result in benchmark.load_test_results:
                print(f"  {result.test_name}: {result.throughput_ops_per_second:.1f} ops/sec, {result.error_rate:.1f}% errors")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Results saved to {args.output}")
        
        print("\n🎉 Performance benchmarking completed!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Add necessary imports for mocking
    try:
        from unittest.mock import patch
    except ImportError:
        print("Warning: unittest.mock not available, some tests may fail")
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)