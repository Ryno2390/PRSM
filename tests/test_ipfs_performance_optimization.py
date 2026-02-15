#!/usr/bin/env python3
"""
IPFS Performance Benchmarks and Optimization Tests
Comprehensive testing for IPFS performance optimization, caching, and production tuning
"""

import asyncio
import json
import sys
import time
import hashlib
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import random

async def test_ipfs_performance_optimization():
    """Test IPFS performance optimization and benchmarking"""
    
    print("⚡ Testing PRSM IPFS Performance & Optimization...")
    print("=" * 80)
    
    try:
        # Import IPFS clients
        from prsm.core.ipfs_client import (
            IPFSClient, IPFSNode, IPFSResult, IPFSConnectionType
        )
        from prsm.data.data_layer.enhanced_ipfs import (
            PRSMIPFSClient, get_ipfs_client
        )
        
        print("✅ IPFS performance testing imports successful")
        
        # === Initialize Clients ===
        
        print(f"\n🚀 Initializing IPFS clients for performance testing...")
        
        core_client = IPFSClient()
        await core_client.initialize()
        
        enhanced_client = PRSMIPFSClient()
        await enhanced_client._ensure_initialized()
        
        print(f"✅ Clients initialized")
        print(f"   - Core client connected: {core_client.connected}")
        print(f"   - Enhanced client connected: {enhanced_client.connected}")
        print(f"   - Available nodes: {len(core_client.nodes) if core_client.connected else 'Simulated'}")
        
        # === Performance Benchmark: Upload Throughput ===
        
        print(f"\n📈 Testing upload throughput performance...")
        
        # Test different content sizes
        size_tests = [
            (1024, "1KB"),          # Small content
            (10240, "10KB"),        # Medium content  
            (102400, "100KB"),      # Large content
            (1048576, "1MB")        # Very large content
        ]
        
        upload_results = {}
        
        for size_bytes, size_label in size_tests:
            print(f"\n   📦 Testing {size_label} uploads...")
            
            # Generate test content
            test_content = f"PERFORMANCE_TEST_{size_label}_".encode('utf-8') + b'x' * (size_bytes - 50)
            
            # Measure upload performance
            upload_times = []
            successful_uploads = 0
            
            for i in range(3):  # Test multiple uploads
                start_time = time.time()
                
                result = await core_client.upload_content(
                    content=test_content,
                    filename=f"perf_test_{size_label}_{i}.bin",
                    pin=False  # Skip pinning for performance
                )
                
                elapsed = time.time() - start_time
                upload_times.append(elapsed)
                
                if result.success:
                    successful_uploads += 1
                
                print(f"      Upload {i+1}: {elapsed:.3f}s {'✅' if result.success else '❌'}")
            
            # Calculate statistics
            if upload_times:
                avg_time = statistics.mean(upload_times)
                min_time = min(upload_times)
                max_time = max(upload_times)
                throughput_mbps = (size_bytes / avg_time) / (1024 * 1024) if avg_time > 0 else 0
                
                upload_results[size_label] = {
                    "size_bytes": size_bytes,
                    "successful_uploads": successful_uploads,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "throughput_mbps": throughput_mbps
                }
                
                print(f"      📊 {size_label} Results:")
                print(f"         - Success rate: {successful_uploads}/3")
                print(f"         - Average time: {avg_time:.3f}s")
                print(f"         - Throughput: {throughput_mbps:.2f} MB/s")
        
        # === Performance Benchmark: Concurrent Uploads ===
        
        print(f"\n🔄 Testing concurrent upload performance...")
        
        concurrent_test_content = b"CONCURRENT_TEST_CONTENT_" + b'x' * 1000
        concurrency_levels = [1, 2, 5, 10]
        
        concurrency_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n   🎯 Testing {concurrency} concurrent uploads...")
            
            async def upload_task(task_id):
                content = concurrent_test_content + f"_TASK_{task_id}".encode('utf-8')
                start = time.time()
                result = await core_client.upload_content(
                    content=content,
                    filename=f"concurrent_{task_id}.bin",
                    pin=False
                )
                elapsed = time.time() - start
                return result.success, elapsed
            
            # Run concurrent uploads
            start_time = time.time()
            
            tasks = [upload_task(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, tuple) and r[0]]
            failed_results = [r for r in results if isinstance(r, tuple) and not r[0]]
            error_results = [r for r in results if isinstance(r, Exception)]
            
            success_rate = len(successful_results) / concurrency if concurrency > 0 else 0
            throughput_ops_per_sec = concurrency / total_time if total_time > 0 else 0
            
            if successful_results:
                avg_individual_time = statistics.mean([r[1] for r in successful_results])
            else:
                avg_individual_time = 0
            
            concurrency_results[concurrency] = {
                "total_time": total_time,
                "success_rate": success_rate,
                "throughput_ops_per_sec": throughput_ops_per_sec,
                "avg_individual_time": avg_individual_time,
                "successful": len(successful_results),
                "failed": len(failed_results),
                "errors": len(error_results)
            }
            
            print(f"      📊 Concurrency {concurrency} Results:")
            print(f"         - Success rate: {success_rate:.1%}")
            print(f"         - Total time: {total_time:.3f}s")
            print(f"         - Throughput: {throughput_ops_per_sec:.2f} ops/s")
            print(f"         - Avg individual time: {avg_individual_time:.3f}s")
        
        # === Enhanced Client Performance Tests ===
        
        print(f"\n🎯 Testing enhanced client performance features...")
        
        # Test provenance tracking overhead
        print(f"\n   📋 Testing provenance tracking performance...")
        
        test_model_data = b"MODEL_PERFORMANCE_TEST_" + b'x' * 10000
        model_metadata = {
            "uploader_id": "perf_test_user",
            "model_type": "neural_network",
            "framework": "pytorch",
            "version": "1.0.0"
        }
        
        # Test with provenance tracking
        start_time = time.time()
        
        model_cid = await enhanced_client.store_model(test_model_data, model_metadata)
        
        provenance_time = time.time() - start_time
        
        print(f"      ✅ Model storage with provenance: {provenance_time:.3f}s")
        print(f"      📦 Generated CID: {model_cid[:20]}..." if model_cid else "      ❌ Storage failed")
        
        # Test access tracking performance
        if model_cid:
            print(f"\n   📊 Testing access tracking performance...")
            
            access_times = []
            for i in range(10):
                start = time.time()
                await enhanced_client.track_access(model_cid, f"user_{i}")
                elapsed = time.time() - start
                access_times.append(elapsed)
            
            avg_access_time = statistics.mean(access_times)
            print(f"      ✅ Average access tracking time: {avg_access_time:.4f}s")
            
            # Test metrics calculation performance
            start = time.time()
            metrics = await enhanced_client.calculate_usage_metrics(model_cid)
            metrics_time = time.time() - start
            
            print(f"      ✅ Metrics calculation time: {metrics_time:.3f}s")
            print(f"      📊 Tracked accesses: {metrics['total_accesses']}")
        
        # === Cache Performance Testing ===
        
        print(f"\n🗄️ Testing cache performance...")
        
        if model_cid:
            # Test repeated retrieval (should hit cache)
            retrieval_times = []
            
            for i in range(5):
                start = time.time()
                content, metadata = await enhanced_client.retrieve_with_provenance(model_cid)
                elapsed = time.time() - start
                retrieval_times.append(elapsed)
                print(f"      Retrieval {i+1}: {elapsed:.3f}s")
            
            if retrieval_times:
                print(f"      📊 Cache Performance:")
                print(f"         - First retrieval: {retrieval_times[0]:.3f}s")
                print(f"         - Average subsequent: {statistics.mean(retrieval_times[1:]):.3f}s")
                print(f"         - Cache speedup: {retrieval_times[0] / statistics.mean(retrieval_times[1:]):.2f}x" if len(retrieval_times) > 1 and statistics.mean(retrieval_times[1:]) > 0 else "N/A")
        
        # === Memory Usage Testing ===
        
        print(f"\n💾 Testing memory usage optimization...")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Measure memory before large operations
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Perform memory-intensive operations
            large_contents = []
            for i in range(10):
                large_content = f"MEMORY_TEST_{i}_".encode('utf-8') + b'x' * 100000  # 100KB each
                large_contents.append(large_content)
                
                # Upload but don't store reference (test garbage collection)
                await core_client.upload_content(
                    content=large_content,
                    filename=f"memory_test_{i}.bin",
                    pin=False
                )
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Measure memory after
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = memory_after - memory_before
            
            print(f"      📊 Memory Usage:")
            print(f"         - Before: {memory_before:.1f} MB")
            print(f"         - After: {memory_after:.1f} MB")
            print(f"         - Increase: {memory_increase:.1f} MB")
            print(f"         - Per operation: {memory_increase/10:.2f} MB")
            
        except ImportError:
            print(f"      ⚠️  psutil not available, skipping memory tests")
        
        # === Network Efficiency Testing ===
        
        print(f"\n🌐 Testing network efficiency...")
        
        # Test connection reuse
        node_usage = {}
        
        for i in range(10):
            test_content = f"NETWORK_TEST_{i}".encode('utf-8')
            result = await core_client.upload_content(content=test_content, pin=False)
            
            if result.connection_type:
                conn_type = result.connection_type.value
                node_usage[conn_type] = node_usage.get(conn_type, 0) + 1
        
        print(f"      📊 Connection Distribution:")
        for conn_type, count in node_usage.items():
            print(f"         - {conn_type}: {count} operations")
        
        # Test retry efficiency
        print(f"\n   🔄 Testing retry mechanism efficiency...")
        
        # Simulate network issues by using invalid content
        retry_start = time.time()
        
        invalid_result = await core_client.upload_content(content=b"", pin=False)
        
        retry_time = time.time() - retry_start
        
        print(f"      📊 Retry Performance:")
        print(f"         - Total retry time: {retry_time:.3f}s")
        print(f"         - Retry attempts: {invalid_result.retry_count}")
        print(f"         - Time per retry: {retry_time/max(invalid_result.retry_count, 1):.3f}s")
        
        # === Configuration Optimization Tests ===
        
        print(f"\n⚙️ Testing configuration optimization...")
        
        # Test different timeout configurations
        timeout_configs = [10, 30, 60]
        
        for timeout in timeout_configs:
            print(f"\n   ⏱️  Testing {timeout}s timeout...")
            
            test_node = IPFSNode(
                url="https://ipfs.io",
                connection_type=IPFSConnectionType.GATEWAY,
                timeout=timeout
            )
            
            await test_node.initialize()
            
            start = time.time()
            health = await test_node.health_check()
            elapsed = time.time() - start
            
            print(f"      📊 Timeout {timeout}s:")
            print(f"         - Health check: {elapsed:.3f}s ({'✅' if health else '❌'})")
            print(f"         - Efficiency: {elapsed/timeout:.2%} of timeout used")
            
            await test_node.cleanup()
        
        # === Generate Performance Report ===
        
        print(f"\n📋 Performance Optimization Report")
        print("=" * 50)
        
        print(f"\n📦 Upload Performance:")
        for size_label, results in upload_results.items():
            if results['successful_uploads'] > 0:
                print(f"   {size_label}:")
                print(f"      - Throughput: {results['throughput_mbps']:.2f} MB/s")
                print(f"      - Success rate: {results['successful_uploads']}/3")
                print(f"      - Latency: {results['min_time']:.3f}s - {results['max_time']:.3f}s")
        
        print(f"\n🔄 Concurrency Performance:")
        for concurrency, results in concurrency_results.items():
            print(f"   {concurrency} concurrent:")
            print(f"      - Success rate: {results['success_rate']:.1%}")
            print(f"      - Throughput: {results['throughput_ops_per_sec']:.2f} ops/s")
        
        print(f"\n🏆 Optimization Recommendations:")
        
        # Analyze results and provide recommendations
        recommendations = []
        
        # Check upload performance
        if upload_results:
            large_throughput = upload_results.get('1MB', {}).get('throughput_mbps', 0)
            if large_throughput < 1.0:
                recommendations.append("📈 Consider increasing timeout for large uploads")
            
            small_success = upload_results.get('1KB', {}).get('successful_uploads', 0)
            if small_success < 3:
                recommendations.append("🔧 Improve small content upload reliability")
        
        # Check concurrency performance
        if concurrency_results:
            best_concurrency = max(concurrency_results.keys(), 
                                 key=lambda k: concurrency_results[k]['throughput_ops_per_sec'])
            recommendations.append(f"⚡ Optimal concurrency level: {best_concurrency}")
        
        # Check error rates
        total_errors = sum(r.get('errors', 0) for r in concurrency_results.values())
        if total_errors > 0:
            recommendations.append("🛡️ Implement additional error handling")
        
        if not recommendations:
            recommendations.append("✨ IPFS performance is already optimized!")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        # === Performance Optimization Suggestions ===
        
        print(f"\n🔧 Production Optimization Settings:")
        print(f"   - Recommended timeout: 60s for reliable operations")
        print(f"   - Optimal concurrency: {best_concurrency if 'best_concurrency' in locals() else '5-10'} operations")
        print(f"   - Cache TTL: 24 hours for metadata")
        print(f"   - Retry attempts: 3 with exponential backoff")
        print(f"   - Node redundancy: 3+ gateway nodes + 1 local node")
        print(f"   - Content verification: Enable for production data")
        print(f"   - Monitoring: Track response times and success rates")
        
        # === Cleanup ===
        
        print(f"\n🧹 Cleaning up performance test resources...")
        
        await core_client.cleanup()
        
        print(f"✅ Performance test cleanup completed")
        
        print("\n" + "=" * 80)
        print("🎉 ALL IPFS PERFORMANCE TESTS COMPLETED!")
        
        return True
        
    except Exception as e:
        print(f"❌ IPFS performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ipfs_performance_optimization())
    sys.exit(0 if success else 1)