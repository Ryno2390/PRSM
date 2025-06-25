#!/usr/bin/env python3
"""
PRSM Persistent Test Environment Demo
====================================

🎯 DEMONSTRATION:
Shows how to use the persistent test environment for ongoing PRSM development
and validation. This demo creates an environment, runs comprehensive tests,
and demonstrates the monitoring capabilities.

🚀 RUN THIS DEMO:
python tests/environment/demo_persistent_environment.py
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from persistent_test_environment import (
    PersistentTestEnvironment, TestEnvironmentConfig,
    create_test_environment
)

from test_runner import TestRunner

async def demo_persistent_environment():
    """Demonstrate the persistent test environment capabilities"""
    
    print("🚀 PRSM Persistent Test Environment Demo")
    print("=" * 50)
    
    # 1. Create a test environment
    print("\n📋 Step 1: Creating Persistent Test Environment")
    
    config = TestEnvironmentConfig(
        environment_id="demo_environment",
        persistent_data=True,
        auto_cleanup=False,  # Keep environment for inspection
        performance_monitoring=True,
        health_check_interval=10  # Check every 10 seconds for demo
    )
    
    print(f"   🆔 Environment ID: {config.environment_id}")
    print(f"   📁 Base Directory: {config.base_directory}")
    print(f"   ⚙️  Persistent Data: {config.persistent_data}")
    print(f"   📊 Performance Monitoring: {config.performance_monitoring}")
    
    try:
        env = await create_test_environment(config)
        print(f"   ✅ Environment created successfully!")
        print(f"   📍 Environment Path: {env.env_dir}")
        
        # 2. Show environment status
        print("\n📊 Step 2: Environment Status")
        status = await env.get_environment_status()
        
        print(f"   🔄 Status: {status['status']}")
        print(f"   📅 Created: {status['created_at']}")
        print(f"   🔧 Components:")
        for component, healthy in status['components_status'].items():
            health_emoji = "✅" if healthy else "❌"
            print(f"      {health_emoji} {component.replace('_', ' ').title()}")
        
        print(f"   📁 Directories created:")
        for name, path in status['directories'].items():
            print(f"      📂 {name.title()}: {path}")
        
        # 3. Demonstrate test isolation
        print("\n🧪 Step 3: Test Isolation Demo")
        
        async with env.test_context("demo_test_1") as test_ctx:
            print(f"   🔬 Running isolated test: {test_ctx['test_id']}")
            print(f"   📁 Test temp directory: {test_ctx['temp_dir']}")
            
            # Create some test files
            test_file = test_ctx['temp_dir'] / "demo_data.json"
            with open(test_file, 'w') as f:
                json.dump({"demo": "data", "timestamp": datetime.now().isoformat()}, f)
            
            print(f"   📄 Created test file: {test_file}")
            await asyncio.sleep(2)  # Simulate test work
            print(f"   ✅ Test completed - cleanup will happen automatically")
        
        print(f"   🧹 Test context cleaned up automatically")
        
        # 4. Show test data
        print("\n💾 Step 4: Test Data Management")
        test_data_info = status.get('test_data_info', {})
        if test_data_info:
            print(f"   📊 Test data generated at: {test_data_info.get('generated_at', 'Unknown')}")
            print(f"   🌱 Data seed: {test_data_info.get('seed', 'Unknown')}")
            print(f"   📝 Datasets:")
            for dataset_name, dataset_info in test_data_info.get('datasets', {}).items():
                print(f"      📋 {dataset_name.title()}: {dataset_info.get('count', 0)} records")
        else:
            print("   ℹ️  No test data info available yet")
        
        # 5. Run comprehensive tests using the test runner
        print("\n🔧 Step 5: Running Comprehensive Tests")
        
        runner = TestRunner()
        runner.environments[config.environment_id] = env
        
        # Run a subset of tests for demo
        test_suites = ["system_health", "integration"]
        print(f"   🧪 Running test suites: {', '.join(test_suites)}")
        
        results = await runner.run_comprehensive_tests(config.environment_id, test_suites)
        
        print(f"   📊 Test Results Summary:")
        summary = results.get('summary', {})
        print(f"      ✅ Passed: {summary.get('passed_suites', 0)}")
        print(f"      ❌ Failed: {summary.get('failed_suites', 0)}")
        print(f"      📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"      ⏱️  Duration: {summary.get('total_duration', 0):.2f}s")
        
        # 6. Show monitoring data
        print("\n📈 Step 6: Performance Monitoring")
        
        if status.get('performance_metrics'):
            metrics = status['performance_metrics']
            print(f"   💾 Memory Usage: {metrics.get('memory_usage_mb', 0):.1f} MB")
            print(f"   🖥️  CPU Usage: {metrics.get('cpu_percent', 0):.1f}%")
            
            if 'database_response_time' in metrics:
                print(f"   🗄️  Database Response: {metrics['database_response_time']:.3f}s")
            if 'redis_response_time' in metrics:
                print(f"   🔴 Redis Response: {metrics['redis_response_time']:.3f}s")
        else:
            print("   ℹ️  Performance metrics will be available after monitoring period")
        
        # 7. Demonstrate environment persistence
        print("\n💾 Step 7: Environment Persistence")
        
        print(f"   📁 Environment files saved to: {env.env_dir}")
        print(f"   📊 Configuration: {env.config_dir / 'environment.json'}")
        print(f"   📝 Logs directory: {env.logs_dir}")
        print(f"   💾 Data directory: {env.data_dir}")
        
        # Show what files were created
        config_file = env.config_dir / "environment.json"
        if config_file.exists():
            with open(config_file) as f:
                env_config = json.load(f)
            print(f"   ✅ Environment config saved (created: {env_config.get('created_at', 'Unknown')})")
        
        # 8. Let monitoring run for a bit
        print("\n⏱️  Step 8: Monitoring Demonstration")
        print("   📊 Letting environment run for 30 seconds to show monitoring...")
        
        for i in range(6):  # 30 seconds in 5-second intervals
            await asyncio.sleep(5)
            current_status = await env.get_environment_status()
            healthy_count = sum(1 for healthy in current_status['components_status'].values() if healthy)
            total_count = len(current_status['components_status'])
            print(f"   📈 Health check {i+1}/6: {healthy_count}/{total_count} components healthy")
        
        # 9. Show how to stop the environment
        print("\n🛑 Step 9: Stopping Environment")
        print("   ⚠️  Note: Environment will be stopped but data preserved (auto_cleanup=False)")
        
        await env.stop()
        final_status = await env.get_environment_status()
        print(f"   ✅ Environment stopped: {final_status['status']}")
        
        # 10. Summary
        print("\n📋 Step 10: Demo Summary")
        print("   🎉 Persistent Test Environment Demo Completed!")
        print(f"   📁 Environment preserved at: {env.env_dir}")
        print("   💡 Key Benefits Demonstrated:")
        print("      ✅ Persistent state across test runs")
        print("      ✅ Isolated test execution with cleanup")
        print("      ✅ Comprehensive health monitoring")
        print("      ✅ Performance metrics collection")
        print("      ✅ Test data management")
        print("      ✅ Automated service orchestration")
        
        print("\n🔄 Next Steps:")
        print("   1. Inspect the environment directory")
        print("   2. Run individual test suites")
        print("   3. Use the environment for development")
        print("   4. Extend with custom test scenarios")
        
        return env.config.environment_id
        
    except Exception as e:
        print(f"   ❌ Demo failed: {str(e)}")
        raise

async def cleanup_demo_environment(environment_id: str):
    """Clean up the demo environment"""
    print(f"\n🧹 Cleaning up demo environment: {environment_id}")
    
    import shutil
    env_dir = Path.cwd() / "test_environments" / environment_id
    
    if env_dir.exists():
        shutil.rmtree(env_dir, ignore_errors=True)
        print(f"   ✅ Environment directory removed: {env_dir}")
    else:
        print(f"   ℹ️  Environment directory not found: {env_dir}")

async def main():
    """Main demo function"""
    print("🎯 Starting PRSM Persistent Test Environment Demo")
    print("   This will demonstrate all key features of the persistent test environment")
    print("   including environment creation, test execution, and monitoring.\n")
    
    try:
        environment_id = await demo_persistent_environment()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Environment preserved at: test_environments/{environment_id}")
        
        # Ask if user wants to clean up
        response = input("\n🗑️  Clean up demo environment? (y/N): ").strip().lower()
        if response == 'y':
            await cleanup_demo_environment(environment_id)
        else:
            print(f"📁 Environment preserved for inspection: test_environments/{environment_id}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())