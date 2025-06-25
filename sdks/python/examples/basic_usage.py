#!/usr/bin/env python3
"""
PRSM Python SDK - Basic Usage Example

This example demonstrates the fundamental capabilities of the PRSM Python SDK,
including model inference, cost optimization, and error handling.
"""

import asyncio
import os
from prsm_sdk import PRSMClient, PRSMError, BudgetExceededError, RateLimitError


async def basic_inference_example():
    """Demonstrate basic model inference"""
    print("=== Basic Model Inference ===")
    
    # Initialize client
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Simple text generation
        result = await client.models.infer(
            model="gpt-4",
            prompt="Explain quantum computing in simple terms",
            max_tokens=150
        )
        
        print(f"Model: {result.model}")
        print(f"Response: {result.content}")
        print(f"Tokens used: {result.usage.total_tokens}")
        print(f"Cost: ${result.cost:.4f}")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def cost_optimization_example():
    """Demonstrate cost-aware request routing"""
    print("\n=== Cost Optimization ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Optimize request for cost efficiency
        optimized = await client.cost_optimization.optimize_request(
            prompt="Write a Python function to sort a list",
            constraints={
                "max_cost": 0.02,
                "min_quality": 0.8,
                "max_latency": 5.0
            }
        )
        
        print(f"Selected Model: {optimized.selected_model}")
        print(f"Estimated Cost: ${optimized.estimated_cost:.4f}")
        print(f"Expected Quality: {optimized.quality_score:.2f}")
        print(f"Cost Savings: ${optimized.cost_savings:.4f}")
        
        # Execute the optimized request
        result = await client.models.infer(
            model=optimized.selected_model,
            prompt="Write a Python function to sort a list",
            max_tokens=200
        )
        
        print(f"\nOptimized Response: {result.content[:100]}...")
        
    except BudgetExceededError as e:
        print(f"Budget exceeded: {e.message}")
        print(f"Remaining budget: ${e.remaining_budget:.2f}")
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def model_comparison_example():
    """Demonstrate model comparison capabilities"""
    print("\n=== Model Comparison ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Compare models for a specific task
        comparison = await client.models.compare(
            models=["gpt-4", "claude-3", "gpt-3.5-turbo"],
            task_type="code_generation",
            test_prompts=[
                "Write a Python function to calculate fibonacci numbers",
                "Create a REST API endpoint using FastAPI"
            ]
        )
        
        print("Model Comparison Results:")
        for model_name, metrics in comparison.results.items():
            print(f"\n{model_name}:")
            print(f"  Overall Score: {metrics.overall_score:.2f}")
            print(f"  Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}")
            print(f"  Average Latency: {metrics.avg_latency_ms}ms")
            print(f"  Quality Score: {metrics.quality_score:.2f}")
        
        print(f"\nRecommended Model: {comparison.recommended_model}")
        print(f"Recommendation Reason: {comparison.recommendation_reason}")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def streaming_example():
    """Demonstrate streaming responses"""
    print("\n=== Streaming Response ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        print("Streaming response for: 'Write a short story about AI'")
        print("Response: ", end="", flush=True)
        
        async for chunk in client.models.stream(
            model="gpt-4",
            prompt="Write a short story about AI and humans working together",
            max_tokens=300
        ):
            print(chunk.content, end="", flush=True)
        
        print("\n[Streaming completed]")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def error_handling_example():
    """Demonstrate proper error handling"""
    print("\n=== Error Handling ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    # Example 1: Rate limiting
    try:
        # This might trigger rate limiting
        tasks = []
        for i in range(100):  # Many concurrent requests
            task = client.models.infer(
                model="gpt-4",
                prompt=f"Say hello #{i}",
                max_tokens=10
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count
        
        print(f"Concurrent requests: {len(tasks)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        
    except RateLimitError as e:
        print(f"Rate limited: {e.message}")
        print(f"Retry after: {e.retry_after_seconds} seconds")
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")
    
    # Example 2: Invalid model
    try:
        await client.models.infer(
            model="nonexistent-model",
            prompt="Hello world"
        )
    except PRSMError as e:
        print(f"Expected error for invalid model: {e.message}")


async def budget_management_example():
    """Demonstrate budget monitoring and management"""
    print("\n=== Budget Management ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Check current budget status
        budget = await client.cost_optimization.get_budget()
        
        print(f"Total Monthly Budget: ${budget.total_budget:.2f}")
        print(f"Amount Spent: ${budget.spent:.2f}")
        print(f"Remaining Budget: ${budget.remaining:.2f}")
        print(f"Budget Utilization: {(budget.spent / budget.total_budget) * 100:.1f}%")
        
        if budget.projected_exhaustion:
            print(f"Projected Exhaustion: {budget.projected_exhaustion}")
        
        # Set budget alerts
        await client.cost_optimization.set_budget_alerts(
            thresholds=[0.5, 0.8, 0.95],  # 50%, 80%, 95%
            notification_channels=["email", "webhook"]
        )
        
        print("Budget alerts configured for 50%, 80%, and 95% thresholds")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def main():
    """Run all examples"""
    # Check for API key
    if not os.getenv("PRSM_API_KEY"):
        print("Please set PRSM_API_KEY environment variable")
        return
    
    print("PRSM Python SDK - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    await basic_inference_example()
    await cost_optimization_example()
    await model_comparison_example()
    await streaming_example()
    await error_handling_example()
    await budget_management_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())