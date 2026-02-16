#!/usr/bin/env python3
"""
Test Script for OpenAI Integration
==================================

This script tests the real OpenAI API integration in PRSM, replacing
mock responses with actual GPT-4 calls. It validates the full pipeline
from API key management to response processing.

Usage:
    python scripts/test_openai_integration.py --interactive
    python scripts/test_openai_integration.py --batch-test
"""

import pytest
pytest.skip('Module dependencies not yet fully implemented', allow_module_level=True)

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import click
import structlog

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.compute.agents.executors.api_clients import (
    OpenAIClient, 
    ModelExecutionRequest, 
    ModelProvider,
    ModelClientRegistry
)

logger = structlog.get_logger(__name__)


class OpenAIIntegrationTester:
    """Test suite for OpenAI API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.test_results = []
    
    async def setup(self):
        """Initialize the OpenAI client"""
        click.echo("🔧 Setting up OpenAI client...")
        self.client = OpenAIClient(api_key=self.api_key)
        await self.client.initialize()
        click.echo("✅ OpenAI client initialized")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()
            click.echo("🧹 Client resources cleaned up")
    
    async def test_basic_gpt4_call(self) -> Dict[str, Any]:
        """Test basic GPT-4 API call"""
        click.echo("🧪 Testing basic GPT-4 call...")
        
        request = ModelExecutionRequest(
            prompt="What is the capital of France? Respond in exactly one word.",
            model_id="gpt-4",
            provider=ModelProvider.OPENAI,
            max_tokens=10,
            temperature=0.0
        )
        
        start_time = time.time()
        response = await self.client.execute(request)
        execution_time = time.time() - start_time
        
        result = {
            "test": "basic_gpt4_call",
            "success": response.success,
            "execution_time": execution_time,
            "latency_ms": execution_time * 1000,
            "content": response.content,
            "token_usage": response.token_usage,
            "error": response.error
        }
        
        if response.success:
            click.echo(f"✅ Success: {response.content} (took {execution_time:.2f}s)")
        else:
            click.echo(f"❌ Failed: {response.error}")
        
        self.test_results.append(result)
        return result
    
    async def test_gpt35_turbo_call(self) -> Dict[str, Any]:
        """Test GPT-3.5 Turbo for cost comparison"""
        click.echo("🧪 Testing GPT-3.5 Turbo call...")
        
        request = ModelExecutionRequest(
            prompt="Write a haiku about artificial intelligence.",
            model_id="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            max_tokens=50,
            temperature=0.7
        )
        
        start_time = time.time()
        response = await self.client.execute(request)
        execution_time = time.time() - start_time
        
        result = {
            "test": "gpt35_turbo_call",
            "success": response.success,
            "execution_time": execution_time,
            "latency_ms": execution_time * 1000,
            "content": response.content,
            "token_usage": response.token_usage,
            "error": response.error
        }
        
        if response.success:
            click.echo(f"✅ Success (took {execution_time:.2f}s):")
            click.echo(f"   {response.content}")
        else:
            click.echo(f"❌ Failed: {response.error}")
        
        self.test_results.append(result)
        return result
    
    async def test_system_prompt_usage(self) -> Dict[str, Any]:
        """Test system prompt functionality"""
        click.echo("🧪 Testing system prompt usage...")
        
        request = ModelExecutionRequest(
            prompt="What should I do today?",
            model_id="gpt-4",
            provider=ModelProvider.OPENAI,
            max_tokens=100,
            temperature=0.5,
            system_prompt="You are a helpful AI assistant that always responds with exactly 3 bullet points."
        )
        
        start_time = time.time()
        response = await self.client.execute(request)
        execution_time = time.time() - start_time
        
        result = {
            "test": "system_prompt_usage",
            "success": response.success,
            "execution_time": execution_time,
            "latency_ms": execution_time * 1000,
            "content": response.content,
            "token_usage": response.token_usage,
            "error": response.error
        }
        
        if response.success:
            click.echo(f"✅ Success (took {execution_time:.2f}s):")
            click.echo(f"   {response.content[:100]}...")
        else:
            click.echo(f"❌ Failed: {response.error}")
        
        self.test_results.append(result)
        return result
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid model"""
        click.echo("🧪 Testing error handling...")
        
        request = ModelExecutionRequest(
            prompt="This should fail",
            model_id="invalid-model-name",
            provider=ModelProvider.OPENAI,
            max_tokens=10,
            temperature=0.0
        )
        
        start_time = time.time()
        response = await self.client.execute(request)
        execution_time = time.time() - start_time
        
        result = {
            "test": "error_handling",
            "success": response.success,
            "execution_time": execution_time,
            "latency_ms": execution_time * 1000,
            "content": response.content,
            "token_usage": response.token_usage,
            "error": response.error
        }
        
        if not response.success:
            click.echo(f"✅ Error handling works: {response.error}")
        else:
            click.echo(f"❌ Expected error but got success")
        
        self.test_results.append(result)
        return result
    
    async def test_latency_benchmark(self, num_requests: int = 5) -> Dict[str, Any]:
        """Test latency with multiple requests"""
        click.echo(f"🧪 Testing latency with {num_requests} requests...")
        
        latencies = []
        successes = 0
        
        for i in range(num_requests):
            request = ModelExecutionRequest(
                prompt=f"Count from 1 to 5. This is request #{i+1}.",
                model_id="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                max_tokens=30,
                temperature=0.0
            )
            
            start_time = time.time()
            response = await self.client.execute(request)
            execution_time = time.time() - start_time
            
            latencies.append(execution_time)
            if response.success:
                successes += 1
            
            click.echo(f"   Request {i+1}: {execution_time:.2f}s")
        
        avg_latency = sum(latencies) / len(latencies)
        success_rate = successes / num_requests
        
        result = {
            "test": "latency_benchmark",
            "num_requests": num_requests,
            "avg_latency": avg_latency,
            "avg_latency_ms": avg_latency * 1000,
            "success_rate": success_rate,
            "latencies": latencies,
            "meets_3s_target": avg_latency < 3.0
        }
        
        click.echo(f"📊 Average latency: {avg_latency:.2f}s")
        click.echo(f"📊 Success rate: {success_rate:.1%}")
        click.echo(f"{'✅' if avg_latency < 3.0 else '❌'} Meets <3s latency target")
        
        self.test_results.append(result)
        return result
    
    async def run_full_test_suite(self):
        """Run all tests"""
        click.echo("🚀 Running full OpenAI integration test suite...")
        
        await self.setup()
        
        try:
            await self.test_basic_gpt4_call()
            await self.test_gpt35_turbo_call()
            await self.test_system_prompt_usage()
            await self.test_error_handling()
            await self.test_latency_benchmark()
            
            # Generate summary
            click.echo("\n📋 Test Summary:")
            click.echo("=" * 50)
            
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results if r.get('success', False) or r['test'] == 'error_handling')
            
            click.echo(f"Total Tests: {total_tests}")
            click.echo(f"Successful: {successful_tests}")
            click.echo(f"Success Rate: {successful_tests/total_tests:.1%}")
            
            # Performance metrics
            latency_tests = [r for r in self.test_results if 'execution_time' in r and r.get('success')]
            if latency_tests:
                avg_latency = sum(r['execution_time'] for r in latency_tests) / len(latency_tests)
                click.echo(f"Average Latency: {avg_latency:.2f}s")
            
        finally:
            await self.cleanup()
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "test_results": self.test_results,
                "summary": {
                    "total_tests": len(self.test_results),
                    "successful_tests": sum(1 for r in self.test_results if r.get('success', False))
                }
            }, f, indent=2)
        click.echo(f"💾 Results saved to {filepath}")


@click.command()
@click.option('--api-key', '-k', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode with prompts')
@click.option('--batch-test', '-b', is_flag=True, help='Run full automated test suite')
@click.option('--save-results', '-s', help='Save results to JSON file')
def main(api_key: str, interactive: bool, batch_test: bool, save_results: str):
    """Test OpenAI integration in PRSM"""
    
    # Get API key
    if not api_key:
        import os
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        if interactive:
            api_key = click.prompt('OpenAI API Key', hide_input=True)
        else:
            click.echo("❌ No API key provided. Use --api-key or set OPENAI_API_KEY")
            sys.exit(1)
    
    async def run_tests():
        tester = OpenAIIntegrationTester(api_key)
        
        if batch_test:
            await tester.run_full_test_suite()
        elif interactive:
            click.echo("🤖 Interactive OpenAI Testing")
            await tester.setup()
            
            try:
                while True:
                    prompt = click.prompt("Enter a prompt (or 'quit' to exit)")
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    model = click.prompt("Model", default="gpt-4")
                    max_tokens = click.prompt("Max tokens", default=100, type=int)
                    temperature = click.prompt("Temperature", default=0.7, type=float)
                    
                    request = ModelExecutionRequest(
                        prompt=prompt,
                        model_id=model,
                        provider=ModelProvider.OPENAI,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    click.echo("🔄 Processing...")
                    start_time = time.time()
                    response = await tester.client.execute(request)
                    execution_time = time.time() - start_time
                    
                    if response.success:
                        click.echo(f"✅ Response (took {execution_time:.2f}s):")
                        click.echo(f"   {response.content}")
                        click.echo(f"💰 Token usage: {response.token_usage}")
                    else:
                        click.echo(f"❌ Error: {response.error}")
            finally:
                await tester.cleanup()
        else:
            # Quick test
            await tester.test_basic_gpt4_call()
        
        if save_results:
            tester.save_results(save_results)
    
    try:
        asyncio.run(run_tests())
        click.echo("✅ All tests completed!")
    except KeyboardInterrupt:
        click.echo("\n⏸️  Tests interrupted by user")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()