#!/usr/bin/env python3
"""
OpenAI Integration Demo
======================

Quick demo of the OpenAI integration without requiring API keys.
Shows the structure and capabilities of our enhanced client.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

import click

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prsm.agents.executors.api_clients import (
    OpenAIClient,
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)


class MockOpenAIClient(OpenAIClient):
    """Mock client for demo purposes"""
    
    def __init__(self, **kwargs):
        # Initialize without requiring API key
        self.api_key = "demo-key"
        self.base_url = "https://api.openai.com/v1"
        self.session = None
    
    async def initialize(self):
        """Mock initialization"""
        click.echo("🔧 Mock OpenAI client initialized for demo")
    
    async def close(self):
        """Mock cleanup"""
        click.echo("🧹 Mock client closed")
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Mock execution that shows the structure"""
        start_time = time.time()
        
        # Simulate API call delay
        await asyncio.sleep(0.5 + (len(request.prompt) / 1000))
        
        # Generate mock response based on request
        if "error" in request.prompt.lower():
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.OPENAI,
                model_id=request.model_id,
                execution_time=time.time() - start_time,
                token_usage={},
                success=False,
                error="Mock error for demonstration"
            )
        
        # Generate realistic mock content
        mock_responses = {
            "gpt-4": f"GPT-4 response to: {request.prompt[:50]}... [Generated using advanced reasoning and context understanding]",
            "gpt-3.5-turbo": f"GPT-3.5-Turbo response: {request.prompt[:30]}... [Fast and efficient response]"
        }
        
        content = mock_responses.get(request.model_id, f"Mock response from {request.model_id}")
        
        # Estimate realistic token usage
        prompt_tokens = len(request.prompt.split()) + (len(request.system_prompt.split()) if request.system_prompt else 0)
        completion_tokens = len(content.split())
        
        execution_time = time.time() - start_time
        
        return ModelExecutionResponse(
            content=content,
            provider=ModelProvider.OPENAI,
            model_id=request.model_id,
            execution_time=execution_time,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            success=True,
            metadata={
                "finish_reason": "stop",
                "mock_demo": True,
                "estimated_cost_usd": (prompt_tokens * 0.00003 + completion_tokens * 0.00006) if request.model_id == "gpt-4" else (prompt_tokens * 0.000001 + completion_tokens * 0.000002)
            }
        )


async def demo_basic_integration():
    """Demo basic OpenAI integration"""
    click.echo("\n🤖 DEMO: Basic OpenAI Integration")
    click.echo("=" * 50)
    
    client = MockOpenAIClient()
    await client.initialize()
    
    try:
        # Test different scenarios
        test_cases = [
            {
                "name": "Basic GPT-4 Query",
                "request": ModelExecutionRequest(
                    prompt="What is the capital of France?",
                    model_id="gpt-4",
                    provider=ModelProvider.OPENAI,
                    max_tokens=50,
                    temperature=0.0
                )
            },
            {
                "name": "GPT-3.5-Turbo with System Prompt",
                "request": ModelExecutionRequest(
                    prompt="Write a haiku about AI",
                    model_id="gpt-3.5-turbo",
                    provider=ModelProvider.OPENAI,
                    max_tokens=100,
                    temperature=0.7,
                    system_prompt="You are a creative poet who writes beautiful haikus"
                )
            },
            {
                "name": "Error Handling Test",
                "request": ModelExecutionRequest(
                    prompt="This should trigger an error response",
                    model_id="gpt-4",
                    provider=ModelProvider.OPENAI,
                    max_tokens=10,
                    temperature=0.0
                )
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            click.echo(f"\n📝 Test {i}: {test_case['name']}")
            click.echo("-" * 30)
            
            request = test_case['request']
            click.echo(f"Model: {request.model_id}")
            click.echo(f"Prompt: {request.prompt}")
            if request.system_prompt:
                click.echo(f"System: {request.system_prompt}")
            
            click.echo("🔄 Processing...")
            response = await client.execute(request)
            
            if response.success:
                click.echo(f"✅ Success (took {response.execution_time:.2f}s)")
                click.echo(f"📄 Response: {response.content}")
                click.echo(f"🔢 Tokens: {response.token_usage.get('total_tokens', 0)}")
                click.echo(f"💰 Est. Cost: ${response.metadata.get('estimated_cost_usd', 0):.4f}")
            else:
                click.echo(f"❌ Error: {response.error}")
    
    finally:
        await client.close()


async def demo_cost_tracking():
    """Demo cost tracking capabilities"""
    click.echo("\n💰 DEMO: Cost Tracking & Budget Management")
    click.echo("=" * 50)
    
    total_cost = 0.0
    total_tokens = 0
    request_count = 0
    
    client = MockOpenAIClient()
    await client.initialize()
    
    try:
        requests = [
            ("gpt-3.5-turbo", "Explain machine learning in one sentence"),
            ("gpt-4", "Write a short poem about technology"),
            ("gpt-3.5-turbo", "What is Python programming?"),
            ("gpt-4", "Describe quantum computing briefly"),
        ]
        
        click.echo("📊 Processing multiple requests to track costs...")
        
        for model, prompt in requests:
            request = ModelExecutionRequest(
                prompt=prompt,
                model_id=model,
                provider=ModelProvider.OPENAI,
                max_tokens=100,
                temperature=0.5
            )
            
            response = await client.execute(request)
            request_count += 1
            
            if response.success:
                tokens = response.token_usage.get('total_tokens', 0)
                cost = response.metadata.get('estimated_cost_usd', 0)
                
                total_cost += cost
                total_tokens += tokens
                
                click.echo(f"✅ {model}: ${cost:.4f} ({tokens} tokens)")
            else:
                click.echo(f"❌ {model}: Failed")
        
        # Show cost summary
        click.echo("\n📈 Cost Summary:")
        click.echo(f"Total Requests: {request_count}")
        click.echo(f"Total Tokens: {total_tokens:,}")
        click.echo(f"Total Cost: ${total_cost:.4f}")
        click.echo(f"Avg Cost/Request: ${total_cost/max(request_count,1):.4f}")
        
        # Budget demo
        budget_limit = 0.10  # $0.10
        budget_used_percent = (total_cost / budget_limit) * 100
        
        click.echo(f"\n🎯 Budget Status (Demo: ${budget_limit:.2f} limit):")
        click.echo(f"Used: {budget_used_percent:.1f}%")
        click.echo(f"Remaining: ${budget_limit - total_cost:.4f}")
        
        if total_cost > budget_limit:
            click.echo("⚠️  Budget exceeded! Real client would block further requests.")
        else:
            click.echo("✅ Within budget limits")
    
    finally:
        await client.close()


async def demo_performance_benchmark():
    """Demo performance benchmarking"""
    click.echo("\n🏃 DEMO: Performance Benchmarking")
    click.echo("=" * 50)
    
    client = MockOpenAIClient()
    await client.initialize()
    
    try:
        num_requests = 5
        latencies = []
        
        click.echo(f"📊 Running {num_requests} requests to measure performance...")
        
        for i in range(num_requests):
            request = ModelExecutionRequest(
                prompt=f"Count from 1 to {i+3}. Request #{i+1}",
                model_id="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                max_tokens=30,
                temperature=0.0
            )
            
            start_time = time.time()
            response = await client.execute(request)
            total_time = time.time() - start_time
            
            latencies.append(response.execution_time)
            
            if response.success:
                click.echo(f"   Request {i+1}: {response.execution_time:.2f}s")
            else:
                click.echo(f"   Request {i+1}: FAILED")
        
        # Calculate performance stats
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        click.echo(f"\n📈 Performance Results:")
        click.echo(f"Average Latency: {avg_latency:.2f}s")
        click.echo(f"Min Latency: {min_latency:.2f}s")
        click.echo(f"Max Latency: {max_latency:.2f}s")
        click.echo(f"Target (<3s): {'✅ PASS' if avg_latency < 3.0 else '❌ FAIL'}")
        
        # Show what real performance would look like
        click.echo(f"\n🎯 Expected Real Performance:")
        click.echo(f"GPT-3.5-Turbo: ~1.2-2.0s per request")
        click.echo(f"GPT-4: ~2.0-4.0s per request")
        click.echo(f"Rate Limiting: 3,500 requests/min")
        click.echo(f"Token Limiting: 90,000 tokens/min")
    
    finally:
        await client.close()


def demo_integration_architecture():
    """Show how the integration fits into PRSM"""
    click.echo("\n🏗️ DEMO: PRSM Architecture Integration")
    click.echo("=" * 50)
    
    click.echo("📋 Integration Points:")
    click.echo("1. ModelExecutionRequest/Response - Standard PRSM interfaces")
    click.echo("2. Cost tracking feeds into FTNS token system")
    click.echo("3. Performance metrics integrate with monitoring")
    click.echo("4. Error handling works with PRSM safety systems")
    
    click.echo("\n🔄 PRSM Agent Pipeline:")
    click.echo("Architect → Prompter → Router → [ENHANCED OPENAI CLIENT] → Compiler")
    
    click.echo("\n🌟 Enhanced Features:")
    click.echo("✅ Real-time cost tracking with budget limits")
    click.echo("✅ Automatic retry with exponential backoff")
    click.echo("✅ Rate limiting (3,500 RPM, 90K TPM)")
    click.echo("✅ Multi-model support (GPT-4, GPT-3.5-turbo)")
    click.echo("✅ Performance monitoring and analytics")
    click.echo("✅ Production-grade error handling")
    
    click.echo("\n🎯 Next Steps:")
    click.echo("1. Add real API key to test with live OpenAI API")
    click.echo("2. Run benchmarks against centralized alternatives")
    click.echo("3. Integrate with PRSM's multi-agent architecture")
    click.echo("4. Deploy in production with real users")


@click.command()
@click.option('--full', '-f', is_flag=True, help='Run full demo suite')
@click.option('--basic', '-b', is_flag=True, help='Run basic integration demo only')
@click.option('--cost', '-c', is_flag=True, help='Run cost tracking demo only')
@click.option('--performance', '-p', is_flag=True, help='Run performance demo only')
def main(full: bool, basic: bool, cost: bool, performance: bool):
    """Demo OpenAI Integration in PRSM"""
    
    click.echo("🎭 OpenAI Integration Demo for PRSM")
    click.echo("=" * 60)
    click.echo("This demo shows the structure and capabilities of our")
    click.echo("enhanced OpenAI integration using mock responses.")
    click.echo("With a real API key, this would connect to OpenAI's API.")
    
    async def run_demos():
        if full or not any([basic, cost, performance]):
            # Run all demos
            await demo_basic_integration()
            await demo_cost_tracking() 
            await demo_performance_benchmark()
            demo_integration_architecture()
        else:
            if basic:
                await demo_basic_integration()
            if cost:
                await demo_cost_tracking()
            if performance:
                await demo_performance_benchmark()
    
    try:
        asyncio.run(run_demos())
        
        click.echo("\n🎉 Demo completed!")
        click.echo("\n🚀 To test with real OpenAI API:")
        click.echo("1. Get API key from https://platform.openai.com/api-keys")
        click.echo("2. Run: python scripts/setup_openai_api.py --interactive --test")
        click.echo("3. Run: python scripts/test_openai_integration.py --batch-test")
        
    except KeyboardInterrupt:
        click.echo("\n⏸️  Demo interrupted by user")
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()