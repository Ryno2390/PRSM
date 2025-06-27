#!/usr/bin/env python3
"""
PRSM Python SDK - Cost Management Example
Demonstrates budget control, cost tracking, and optimization features
"""

import os
import asyncio
from typing import Optional
from datetime import datetime, timedelta

# Import PRSM SDK (assuming it's in parent directory)
import sys
sys.path.append('..')
from prsm_sdk import Client, BudgetManager, CostTracker

async def cost_management_example():
    """Demonstrate cost management features with PRSM Python SDK"""
    print("🚀 PRSM Python SDK - Cost Management")
    
    # Initialize client with budget controls
    client = Client(
        api_key=os.getenv('PRSM_API_KEY', 'demo-key'),
        endpoint=os.getenv('PRSM_ENDPOINT', 'http://localhost:8000')
    )
    
    # Set up budget manager
    budget_manager = BudgetManager(
        daily_limit=100.0,  # 100 FTNS per day
        monthly_limit=2000.0,  # 2000 FTNS per month
        alert_threshold=0.8  # Alert at 80% of budget
    )
    
    # Initialize cost tracker
    cost_tracker = CostTracker()
    
    try:
        # Check current balance and budget status
        balance = await client.marketplace.get_balance()
        budget_status = budget_manager.get_status()
        
        print(f"💰 Current balance: {balance.ftns:.2f} FTNS")
        print(f"📊 Daily budget: {budget_status.daily_used:.2f}/{budget_status.daily_limit:.2f} FTNS")
        print(f"📈 Monthly budget: {budget_status.monthly_used:.2f}/{budget_status.monthly_limit:.2f} FTNS")
        
        # Set cost-per-query limit
        max_cost_per_query = 0.5  # Maximum 0.5 FTNS per query
        
        # Example queries with cost tracking
        queries = [
            "Explain PRSM's consensus mechanism",
            "How do FTNS token economics work?",
            "What are the benefits of distributed AI?",
            "Describe PRSM's security features in detail"
        ]
        
        total_cost = 0.0
        
        for i, prompt in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {prompt[:50]}...")
            
            # Check budget before query
            if not budget_manager.can_spend(max_cost_per_query):
                print("⚠️ Budget limit reached, skipping remaining queries")
                break
            
            # Estimate cost before query
            cost_estimate = await client.estimate_cost(
                prompt=prompt,
                max_tokens=150
            )
            
            print(f"💡 Estimated cost: {cost_estimate.ftns:.3f} FTNS")
            
            if cost_estimate.ftns > max_cost_per_query:
                print(f"⚠️ Query too expensive ({cost_estimate.ftns:.3f} > {max_cost_per_query}), skipping")
                continue
            
            # Execute query with cost tracking
            start_time = datetime.now()
            
            response = await client.query(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            # Track costs
            cost_tracker.add_query(
                cost=response.cost,
                tokens=response.token_count,
                latency=query_time,
                model=response.model_used
            )
            
            budget_manager.record_spend(response.cost)
            total_cost += response.cost
            
            print(f"✅ Response: {response.text[:100]}...")
            print(f"💸 Actual cost: {response.cost:.3f} FTNS")
            print(f"⏱️ Latency: {query_time:.2f}s")
            
            # Check for budget alerts
            if budget_manager.should_alert():
                print("🚨 Budget alert: Approaching daily/monthly limit!")
        
        # Generate cost analysis report
        print("\n📋 Cost Analysis Report")
        print("=" * 40)
        
        stats = cost_tracker.get_statistics()
        print(f"Total queries: {stats.query_count}")
        print(f"Total cost: {stats.total_cost:.3f} FTNS")
        print(f"Average cost per query: {stats.avg_cost:.3f} FTNS")
        print(f"Cost per token: {stats.cost_per_token:.6f} FTNS")
        print(f"Average latency: {stats.avg_latency:.2f}s")
        
        # Cost optimization suggestions
        print(f"\n💡 Optimization Suggestions:")
        if stats.avg_cost > 0.3:
            print("- Consider using smaller max_tokens for shorter responses")
        if stats.avg_latency > 2.0:
            print("- Consider using faster models for time-sensitive queries")
        if stats.cost_per_token > 0.001:
            print("- Batch similar queries together for better efficiency")
        
        # Export cost data for analysis
        cost_data = cost_tracker.export_data()
        print(f"📄 Cost data exported: {len(cost_data)} entries")
        
        # Set up budget alerts for future
        await setup_budget_alerts(client, budget_manager)
        
    except Exception as error:
        print(f"❌ Cost management error: {error}")

async def setup_budget_alerts(client: Client, budget_manager: BudgetManager):
    """Set up automated budget alerts and controls"""
    print("\n🔔 Setting up budget alerts...")
    
    # Configure webhook for budget alerts (if supported)
    try:
        await client.configure_alerts({
            'budget_threshold': 0.8,  # 80% threshold
            'webhook_url': os.getenv('BUDGET_WEBHOOK_URL'),
            'email_alerts': os.getenv('BUDGET_ALERT_EMAIL')
        })
        print("✅ Budget alerts configured")
    except Exception as e:
        print(f"⚠️ Could not configure alerts: {e}")

class BudgetManager:
    """Simple budget management class"""
    def __init__(self, daily_limit: float, monthly_limit: float, alert_threshold: float = 0.8):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.alert_threshold = alert_threshold
        self.daily_spent = 0.0
        self.monthly_spent = 0.0
        
    def can_spend(self, amount: float) -> bool:
        return (self.daily_spent + amount <= self.daily_limit and 
                self.monthly_spent + amount <= self.monthly_limit)
    
    def record_spend(self, amount: float):
        self.daily_spent += amount
        self.monthly_spent += amount
    
    def should_alert(self) -> bool:
        return (self.daily_spent >= self.daily_limit * self.alert_threshold or
                self.monthly_spent >= self.monthly_limit * self.alert_threshold)
    
    def get_status(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            daily_used=self.daily_spent,
            daily_limit=self.daily_limit,
            monthly_used=self.monthly_spent,
            monthly_limit=self.monthly_limit
        )

class CostTracker:
    """Simple cost tracking class"""
    def __init__(self):
        self.queries = []
    
    def add_query(self, cost: float, tokens: int, latency: float, model: str):
        self.queries.append({
            'cost': cost,
            'tokens': tokens,
            'latency': latency,
            'model': model,
            'timestamp': datetime.now()
        })
    
    def get_statistics(self):
        if not self.queries:
            from types import SimpleNamespace
            return SimpleNamespace(
                query_count=0, total_cost=0, avg_cost=0, 
                cost_per_token=0, avg_latency=0
            )
        
        total_cost = sum(q['cost'] for q in self.queries)
        total_tokens = sum(q['tokens'] for q in self.queries)
        total_latency = sum(q['latency'] for q in self.queries)
        
        from types import SimpleNamespace
        return SimpleNamespace(
            query_count=len(self.queries),
            total_cost=total_cost,
            avg_cost=total_cost / len(self.queries),
            cost_per_token=total_cost / max(total_tokens, 1),
            avg_latency=total_latency / len(self.queries)
        )
    
    def export_data(self):
        return self.queries

def run_sync_example():
    """Synchronous wrapper for the async example"""
    try:
        asyncio.run(cost_management_example())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    run_sync_example()