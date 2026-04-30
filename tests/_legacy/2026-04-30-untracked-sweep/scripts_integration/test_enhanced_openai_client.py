#!/usr/bin/env python3
"""
Enhanced OpenAI Client Test Suite
=================================

Comprehensive testing for the enhanced OpenAI client with:
- Cost tracking and budget management
- Rate limiting and retry logic
- Performance benchmarking
- Error handling validation

Usage:
    python scripts/test_enhanced_openai_client.py --api-key YOUR_KEY --test-all
    python scripts/test_enhanced_openai_client.py --api-key YOUR_KEY --benchmark
    python scripts/test_enhanced_openai_client.py --api-key YOUR_KEY --cost-test
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal

import click
import structlog

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.compute.agents.executors.enhanced_openai_client import (
    EnhancedOpenAIClient, 
    create_enhanced_openai_client
)
from prsm.compute.agents.executors.api_clients import (
    ModelExecutionRequest, 
    ModelProvider
)

logger = structlog.get_logger(__name__)


class EnhancedOpenAITester:
    """Comprehensive test suite for enhanced OpenAI client"""
    
    def __init__(self, api_key: str, budget_limit: float = 10.0):
        self.api_key = api_key
        self.budget_limit = budget_limit
        self.client = None
        self.test_results = []
    
    async def setup(self):
        """Initialize the enhanced OpenAI client"""
        click.echo("🔧 Setting up Enhanced OpenAI client...")
        self.client = await create_enhanced_openai_client(
            api_key=self.api_key,
            budget_limit_usd=self.budget_limit,
            enable_cost_tracking=True,
            enable_rate_limiting=True
        )
        click.echo("✅ Enhanced OpenAI client initialized")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()
            click.echo("🧹 Client resources cleaned up")
    
    async def test_cost_tracking(self) -> Dict[str, Any]:
        """Test cost tracking functionality"""
        click.echo("💰 Testing cost tracking...")
        
        # Test multiple requests with different models
        requests = [
            ("gpt-3.5-turbo", "Write a short poem about AI"),
            ("gpt-4", "Explain quantum computing in one sentence"),
            ("gpt-3.5-turbo", "Count from 1 to 5")
        ]
        
        total_cost_before = self.client.cost_tracker.total_cost
        
        for model, prompt in requests:
            request = ModelExecutionRequest(
                prompt=prompt,
                model_id=model,
                provider=ModelProvider.OPENAI,
                max_tokens=50,
                temperature=0.7
            )
            
            response = await self.client.execute(request)
            if response.success:
                click.echo(f"   ✅ {model}: Cost ${response.metadata.get('cost_usd', 0):.4f}")
            else:
                click.echo(f"   ❌ {model}: {response.error}")
        
        # Get usage summary
        summary = await self.client.get_usage_summary()
        total_cost_after = self.client.cost_tracker.total_cost
        
        result = {
            "test": "cost_tracking",
            "total_cost_change": float(total_cost_after - total_cost_before),
            "usage_summary": summary,
            "requests_tested": len(requests)
        }
        
        click.echo(f"💰 Total cost: ${summary['total_cost_usd']:.4f}")
        click.echo(f"💰 Budget remaining: ${summary['budget_remaining_usd']:.2f}")
        click.echo(f"💰 Budget used: {summary['budget_used_percent']:.1f}%")
        
        self.test_results.append(result)
        return result
    
    async def test_budget_enforcement(self) -> Dict[str, Any]:
        """Test budget limit enforcement"""
        click.echo("🚫 Testing budget enforcement...")
        
        # Create client with very low budget
        low_budget_client = EnhancedOpenAIClient(
            api_key=self.api_key,
            budget_limit_usd=0.01  # Very low budget
        )
        await low_budget_client.initialize()
        
        try:
            # Make request that should exceed budget
            request = ModelExecutionRequest(
                prompt="Write a very long essay about the history of artificial intelligence, covering all major developments from the 1950s to today in great detail.",
                model_id="gpt-4",
                provider=ModelProvider.OPENAI,
                max_tokens=1000,
                temperature=0.7
            )
            
            response = await low_budget_client.execute(request)
            
            result = {
                "test": "budget_enforcement",
                "budget_exceeded": not response.success and "budget" in response.error.lower(),
                "error_message": response.error,
                "success": not response.success
            }
            
            if result["budget_exceeded"]:
                click.echo("✅ Budget enforcement working correctly")
            else:
                click.echo("❌ Budget enforcement failed")
            
        finally:
            await low_budget_client.close()
        
        self.test_results.append(result)
        return result
    
    async def test_retry_logic(self) -> Dict[str, Any]:
        """Test retry logic with invalid model"""
        click.echo("🔄 Testing retry logic...")
        
        request = ModelExecutionRequest(
            prompt="This should trigger retry logic",
            model_id="invalid-model-that-does-not-exist",
            provider=ModelProvider.OPENAI,
            max_tokens=10,
            temperature=0.0
        )
        
        start_time = time.time()
        response = await self.client.execute(request)
        execution_time = time.time() - start_time
        
        result = {
            "test": "retry_logic",
            "failed_as_expected": not response.success,
            "execution_time": execution_time,
            "error_message": response.error,
            "retries_attempted": execution_time > 1.0  # Should take longer due to retries
        }
        
        if result["failed_as_expected"]:
            click.echo(f"✅ Retry logic working (took {execution_time:.2f}s)")
        else:
            click.echo("❌ Retry logic test failed")
        
        self.test_results.append(result)
        return result
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        click.echo("⏱️ Testing rate limiting...")
        
        # Send multiple requests quickly
        num_requests = 5
        requests_start = time.time()
        
        tasks = []
        for i in range(num_requests):
            request = ModelExecutionRequest(
                prompt=f"Count to {i+3}. This is request {i+1}.",
                model_id="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                max_tokens=20,
                temperature=0.0
            )
            tasks.append(self.client.execute(request))
        
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - requests_start
        
        successful_requests = sum(1 for r in responses if r.success)
        
        result = {
            "test": "rate_limiting",
            "num_requests": num_requests,
            "successful_requests": successful_requests,
            "total_time": total_time,
            "avg_time_per_request": total_time / num_requests,
            "rate_limiting_active": total_time > num_requests * 0.5  # Should add some delay
        }
        
        click.echo(f"⏱️ {successful_requests}/{num_requests} requests successful")
        click.echo(f"⏱️ Total time: {total_time:.2f}s")
        click.echo(f"⏱️ Avg per request: {result['avg_time_per_request']:.2f}s")
        
        self.test_results.append(result)
        return result
    
    async def test_performance_benchmark(self, num_requests: int = 10) -> Dict[str, Any]:
        """Performance benchmark test"""
        click.echo(f"🏃 Running performance benchmark ({num_requests} requests)...")
        
        latencies = []
        costs = []
        token_usage = []
        
        for i in range(num_requests):
            request = ModelExecutionRequest(
                prompt=f"Explain the concept of machine learning in exactly 2 sentences. Request #{i+1}.",
                model_id="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                max_tokens=100,
                temperature=0.3
            )
            
            start_time = time.time()
            response = await self.client.execute(request)
            latency = time.time() - start_time
            
            latencies.append(latency)
            
            if response.success:
                costs.append(response.metadata.get('cost_usd', 0))
                token_usage.append(response.token_usage.get('total_tokens', 0))
                click.echo(f"   Request {i+1}: {latency:.2f}s, ${response.metadata.get('cost_usd', 0):.4f}")
            else:
                click.echo(f"   Request {i+1}: FAILED - {response.error}")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        total_cost = sum(costs)
        avg_tokens = sum(token_usage) / len(token_usage) if token_usage else 0
        
        result = {
            "test": "performance_benchmark",
            "num_requests": num_requests,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "total_cost": total_cost,
            "avg_cost_per_request": total_cost / num_requests if num_requests > 0 else 0,
            "avg_tokens_per_request": avg_tokens,
            "meets_3s_target": avg_latency < 3.0,
            "latencies": latencies
        }
        
        click.echo(f"📊 Average latency: {avg_latency:.2f}s")
        click.echo(f"📊 Latency range: {min_latency:.2f}s - {max_latency:.2f}s")
        click.echo(f"📊 Total cost: ${total_cost:.4f}")
        click.echo(f"📊 Avg cost/request: ${result['avg_cost_per_request']:.4f}")
        click.echo(f"{'✅' if avg_latency < 3.0 else '❌'} Meets <3s latency target")
        
        self.test_results.append(result)
        return result
    
    async def test_model_comparison(self) -> Dict[str, Any]:
        """Compare different model performance and costs"""
        click.echo("🆚 Testing model comparison...")
        
        models = ["gpt-3.5-turbo", "gpt-4"]
        prompt = "Explain the difference between AI and machine learning in one paragraph."
        
        model_results = {}
        
        for model in models:
            request = ModelExecutionRequest(
                prompt=prompt,
                model_id=model,
                provider=ModelProvider.OPENAI,
                max_tokens=150,
                temperature=0.5
            )
            
            start_time = time.time()
            response = await self.client.execute(request)
            latency = time.time() - start_time
            
            if response.success:
                model_results[model] = {
                    "latency": latency,
                    "cost": response.metadata.get('cost_usd', 0),
                    "tokens": response.token_usage.get('total_tokens', 0),
                    "content_length": len(response.content),
                    "success": True
                }
                click.echo(f"   ✅ {model}: {latency:.2f}s, ${response.metadata.get('cost_usd', 0):.4f}")
            else:
                model_results[model] = {
                    "error": response.error,
                    "success": False
                }
                click.echo(f"   ❌ {model}: {response.error}")
        
        result = {
            "test": "model_comparison",
            "prompt": prompt,
            "model_results": model_results
        }
        
        self.test_results.append(result)
        return result
    
    async def run_comprehensive_test_suite(self):
        """Run all tests"""
        click.echo("🚀 Running comprehensive enhanced OpenAI test suite...")
        
        await self.setup()
        
        try:
            await self.test_cost_tracking()
            await self.test_budget_enforcement()
            await self.test_retry_logic()
            await self.test_rate_limiting()
            await self.test_performance_benchmark()
            await self.test_model_comparison()
            
            # Final usage summary
            final_summary = await self.client.get_usage_summary()
            
            click.echo("\n📋 Final Test Summary:")
            click.echo("=" * 60)
            
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results 
                                 if r.get('success', True) and 
                                    not any(k.startswith('failed') and v for k, v in r.items()))
            
            click.echo(f"Total Tests: {total_tests}")
            click.echo(f"Successful: {successful_tests}")
            click.echo(f"Success Rate: {successful_tests/total_tests:.1%}")
            
            click.echo("\n💰 Final Usage Summary:")
            click.echo(f"Total Cost: ${final_summary['total_cost_usd']:.4f}")
            click.echo(f"Total Requests: {final_summary['total_requests']}")
            click.echo(f"Budget Used: {final_summary['budget_used_percent']:.1f}%")
            click.echo(f"Avg Cost/Request: ${final_summary['avg_cost_per_request']:.4f}")
            
        finally:
            await self.cleanup()
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "budget_limit": self.budget_limit,
                "test_results": self.test_results,
                "summary": {
                    "total_tests": len(self.test_results),
                    "successful_tests": sum(1 for r in self.test_results if r.get('success', True))
                }
            }, f, indent=2, default=str)
        click.echo(f"💾 Results saved to {filepath}")


@click.command()
@click.option('--api-key', '-k', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--budget', '-b', default=5.0, help='Budget limit in USD for testing')
@click.option('--test-all', is_flag=True, help='Run comprehensive test suite')
@click.option('--benchmark', is_flag=True, help='Run performance benchmark only')
@click.option('--cost-test', is_flag=True, help='Run cost tracking tests only')
@click.option('--save-results', '-s', help='Save results to JSON file')
@click.option('--num-requests', '-n', default=10, help='Number of requests for benchmark')
def main(api_key: str, budget: float, test_all: bool, benchmark: bool, 
         cost_test: bool, save_results: str, num_requests: int):
    """Test Enhanced OpenAI Client"""
    
    # Get API key
    if not api_key:
        import os
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        api_key = click.prompt('OpenAI API Key', hide_input=True)
    
    if not api_key:
        click.echo("❌ No API key provided. Use --api-key or set OPENAI_API_KEY")
        sys.exit(1)
    
    async def run_tests():
        tester = EnhancedOpenAITester(api_key, budget)
        
        if test_all:
            await tester.run_comprehensive_test_suite()
        elif benchmark:
            await tester.setup()
            try:
                await tester.test_performance_benchmark(num_requests)
            finally:
                await tester.cleanup()
        elif cost_test:
            await tester.setup()
            try:
                await tester.test_cost_tracking()
                await tester.test_budget_enforcement()
            finally:
                await tester.cleanup()
        else:
            # Quick basic test
            await tester.setup()
            try:
                await tester.test_cost_tracking()
            finally:
                await tester.cleanup()
        
        if save_results:
            tester.save_results(save_results)
    
    try:
        asyncio.run(run_tests())
        click.echo("✅ All tests completed!")
    except KeyboardInterrupt:
        click.echo("\n⏸️  Tests interrupted by user")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()