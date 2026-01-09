"""
PRSM FTNS Budget Management Example

🎯 BUDGET CONTROL DEMONSTRATION:
This example shows how to use PRSM's sophisticated budget management system
to control FTNS spending for open-ended queries like Prismatica's Feynman tree analysis.

Key Features Demonstrated:
1. Predictive cost estimation before execution
2. Session budget creation with user-defined limits
3. Real-time spending tracking during execution
4. Budget expansion requests and authorization
5. Multi-resource budget breakdown and analytics
"""

import asyncio
import json
from decimal import Decimal
from typing import Dict, Any

# Import PRSM budget components
from prsm.core.models import UserInput, PRSMSession
from prsm.tokenomics.ftns_budget_manager import (
    FTNSBudgetManager, SpendingCategory, BudgetStatus
)
from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator


async def demonstrate_budget_workflow():
    """
    Demonstrate complete budget workflow for a complex Prismatica query
    
    This example simulates the scenario you described where Prismatica
    needs to perform critical virtual testing on a Feynman tree branch
    for APM development - an open-ended task that could consume significant FTNS.
    """
    print("🎯 PRSM Budget Management Demonstration")
    print("=" * 60)
    
    # Initialize budget manager
    budget_manager = FTNSBudgetManager()
    orchestrator = EnhancedNWTNOrchestrator()
    
    # Example complex query from Prismatica
    complex_query = """
    Perform comprehensive virtual testing analysis on Feynman tree branch 
    Alpha-7 for APM development. Analyze quantum field interactions, 
    calculate dimensional resonance patterns, validate theoretical models 
    against experimental data, and generate optimization recommendations 
    for enhanced photonic coupling efficiency. Include Monte Carlo 
    simulations for edge cases and provide detailed mathematical proofs 
    for all derived equations.
    """
    
    print(f"📝 Query: {complex_query[:100]}...")
    print()
    
    # Step 1: Predict costs before execution
    print("🔮 STEP 1: Predictive Cost Estimation")
    print("-" * 40)
    
    user_input = UserInput(
        user_id="prismatica_user_001",
        prompt=complex_query
    )
    
    session = PRSMSession(user_id="prismatica_user_001")
    
    # Generate cost prediction
    prediction = await budget_manager.predict_session_cost(user_input, session)
    
    print(f"📊 Estimated Total Cost: {float(prediction.estimated_total_cost):.2f} FTNS")
    print(f"🎯 Confidence Score: {prediction.confidence_score:.2f}")
    print(f"📈 Query Complexity: {prediction.query_complexity:.2f}")
    print()
    
    print("💰 Category Breakdown:")
    for category, amount in prediction.category_estimates.items():
        print(f"   • {category.value.replace('_', ' ').title()}: {float(amount):.2f} FTNS")
    print()
    
    recommended_budget = prediction.get_recommended_budget()
    print(f"💡 Recommended Budget (with safety margin): {float(recommended_budget):.2f} FTNS")
    print()
    
    # Step 2: Create session budget with user-defined limit
    print("💳 STEP 2: Session Budget Creation")
    print("-" * 40)
    
    # User sets budget limit (maybe more conservative than recommended)
    user_budget_limit = 150.0  # User decides 150 FTNS is their comfort limit
    
    budget_config = {
        "total_budget": user_budget_limit,
        "auto_expand_enabled": True,
        "max_auto_expand": 100.0,  # Allow up to 100 FTNS auto-expansion
        "expansion_increment": 25.0,  # Expand in 25 FTNS increments
        "category_allocations": {
            "model_inference": {"percentage": 60},  # 60% for model costs
            "tool_execution": {"percentage": 25},   # 25% for tools/analysis
            "agent_coordination": {"percentage": 10}, # 10% for agent coordination
            "context_processing": {"percentage": 5}   # 5% for context management
        }
    }
    
    session_budget = await budget_manager.create_session_budget(
        session, user_input, budget_config
    )
    
    print(f"✅ Budget Created: {session_budget.budget_id}")
    print(f"💰 Total Budget: {float(session_budget.total_budget):.2f} FTNS")
    print(f"🔄 Auto-expand: {'Enabled' if session_budget.auto_expand_enabled else 'Disabled'}")
    print(f"📈 Max Auto-expand: {float(session_budget.max_auto_expand):.2f} FTNS")
    print()
    
    print("📊 Category Allocations:")
    for category, allocation in session_budget.category_allocations.items():
        print(f"   • {category.value.replace('_', ' ').title()}: {float(allocation.allocated_amount):.2f} FTNS")
    print()
    
    # Step 3: Simulate query execution with budget tracking
    print("⚡ STEP 3: Query Execution with Budget Tracking")
    print("-" * 40)
    
    # Simulate spending during execution
    execution_steps = [
        (SpendingCategory.AGENT_COORDINATION, 8.5, "Initial query analysis and task decomposition"),
        (SpendingCategory.MODEL_INFERENCE, 35.0, "Primary model inference for quantum field analysis"),
        (SpendingCategory.TOOL_EXECUTION, 22.0, "Monte Carlo simulation tools"),
        (SpendingCategory.MODEL_INFERENCE, 28.0, "Mathematical proof generation"),
        (SpendingCategory.TOOL_EXECUTION, 18.5, "Dimensional resonance calculations"),
        (SpendingCategory.MODEL_INFERENCE, 31.0, "Optimization recommendation synthesis"),
    ]
    
    total_spent = 0
    for i, (category, amount, description) in enumerate(execution_steps, 1):
        success = await budget_manager.spend_budget_amount(
            session_budget.budget_id,
            Decimal(str(amount)),
            category,
            description
        )
        
        total_spent += amount
        
        if success:
            budget_status = await budget_manager.get_budget_status(session_budget.budget_id)
            utilization = budget_status["utilization_percentage"]
            remaining = budget_status["available_budget"]
            
            print(f"✅ Step {i}: Spent {amount:.1f} FTNS on {description}")
            print(f"   📊 Budget Utilization: {utilization:.1f}% | Remaining: {remaining:.1f} FTNS")
        else:
            print(f"❌ Step {i}: Failed to spend {amount:.1f} FTNS - Budget limit reached!")
            break
        print()
    
    # Step 4: Handle budget expansion if needed
    print("📈 STEP 4: Budget Expansion Handling")
    print("-" * 40)
    
    # Simulate needing more budget for additional analysis
    additional_analysis_cost = 45.0
    
    print(f"🔍 Additional analysis needed: {additional_analysis_cost:.1f} FTNS")
    
    # Check if budget expansion is needed
    final_budget_status = await budget_manager.get_budget_status(session_budget.budget_id)
    remaining_budget = final_budget_status["available_budget"]
    
    if remaining_budget < additional_analysis_cost:
        print(f"⚠️ Insufficient budget remaining ({remaining_budget:.1f} FTNS < {additional_analysis_cost:.1f} FTNS)")
        print("📤 Requesting budget expansion...")
        
        # Request budget expansion
        expansion_request = await budget_manager.request_budget_expansion(
            session_budget.budget_id,
            Decimal(str(additional_analysis_cost + 10)),  # Request a bit extra
            "Additional deep analysis for photonic coupling optimization",
            {
                SpendingCategory.MODEL_INFERENCE: Decimal(str(additional_analysis_cost * 0.7)),
                SpendingCategory.TOOL_EXECUTION: Decimal(str(additional_analysis_cost * 0.3))
            }
        )
        
        if expansion_request.approved:
            print(f"✅ Budget expansion auto-approved: {float(expansion_request.approved_amount):.1f} FTNS")
        else:
            print(f"⏳ Budget expansion pending user approval: {float(expansion_request.requested_amount):.1f} FTNS")
            print(f"📝 Request ID: {expansion_request.request_id}")
            
            # Simulate user approval
            approval_success = await budget_manager.approve_budget_expansion(
                expansion_request.request_id,
                approved=True,
                approved_amount=expansion_request.requested_amount,
                reason="Approved for critical APM development analysis"
            )
            
            if approval_success:
                print("✅ Budget expansion manually approved by user")
    else:
        print(f"✅ Sufficient budget remaining: {remaining_budget:.1f} FTNS")
    
    print()
    
    # Step 5: Final analytics and insights
    print("📊 STEP 5: Final Budget Analytics")
    print("-" * 40)
    
    final_status = await budget_manager.get_budget_status(session_budget.budget_id)
    
    print(f"💰 Final Budget Status:")
    print(f"   • Total Budget: {final_status['total_budget']:.2f} FTNS")
    print(f"   • Total Spent: {final_status['total_spent']:.2f} FTNS")
    print(f"   • Available: {final_status['available_budget']:.2f} FTNS")
    print(f"   • Utilization: {final_status['utilization_percentage']:.1f}%")
    print()
    
    print("📈 Category Spending Breakdown:")
    for category, data in final_status["category_breakdown"].items():
        if data["spent"] > 0:
            print(f"   • {category.replace('_', ' ').title()}:")
            print(f"     - Allocated: {data['allocated']:.1f} FTNS")
            print(f"     - Spent: {data['spent']:.1f} FTNS")
            print(f"     - Utilization: {data['utilization']:.1f}%")
    
    print()
    print("🎯 Budget Management Benefits Demonstrated:")
    print("   ✅ Predictive cost estimation prevents budget surprises")
    print("   ✅ Real-time spending tracking provides transparency")
    print("   ✅ Automatic expansion requests maintain workflow continuity")
    print("   ✅ Category-based budgeting enables granular control")
    print("   ✅ User authorization ensures spending approval for overages")


async def demonstrate_marketplace_budget_integration():
    """
    Demonstrate budget integration with marketplace transactions
    
    Shows how budgets work with U2U and A2A marketplace transactions
    for models, agents, tools, and datasets.
    """
    print("\n🏪 MARKETPLACE BUDGET INTEGRATION")
    print("=" * 60)
    
    budget_manager = FTNSBudgetManager()
    
    # Create a budget specifically for marketplace activities
    marketplace_user_input = UserInput(
        user_id="marketplace_user_001",
        prompt="Purchase specialized quantum simulation models and tools for research"
    )
    
    marketplace_session = PRSMSession(user_id="marketplace_user_001")
    
    marketplace_budget_config = {
        "total_budget": 200.0,
        "category_allocations": {
            "marketplace_trading": {"percentage": 80},  # 80% for marketplace purchases
            "model_inference": {"percentage": 15},      # 15% for testing purchased models
            "agent_coordination": {"percentage": 5}     # 5% for coordination
        }
    }
    
    marketplace_budget = await budget_manager.create_session_budget(
        marketplace_session, marketplace_user_input, marketplace_budget_config
    )
    
    print(f"🛒 Marketplace Budget Created: {marketplace_budget.budget_id}")
    print(f"💰 Total Budget: {float(marketplace_budget.total_budget):.2f} FTNS")
    print()
    
    # Simulate marketplace transactions
    marketplace_transactions = [
        (SpendingCategory.MARKETPLACE_TRADING, 45.0, "Purchase quantum simulation model from UserABC"),
        (SpendingCategory.MARKETPLACE_TRADING, 25.0, "Buy advanced calculation tools from AgentXYZ"),
        (SpendingCategory.MODEL_INFERENCE, 15.0, "Test purchased quantum model"),
        (SpendingCategory.MARKETPLACE_TRADING, 30.0, "Purchase research dataset from DataProvider123"),
        (SpendingCategory.AGENT_COORDINATION, 8.0, "Coordinate with purchased agent services"),
    ]
    
    print("🔄 Marketplace Transaction Simulation:")
    for i, (category, amount, description) in enumerate(marketplace_transactions, 1):
        success = await budget_manager.spend_budget_amount(
            marketplace_budget.budget_id,
            Decimal(str(amount)),
            category,
            description
        )
        
        if success:
            budget_status = await budget_manager.get_budget_status(marketplace_budget.budget_id)
            print(f"✅ Transaction {i}: {amount:.1f} FTNS - {description}")
            print(f"   💳 Remaining Budget: {budget_status['available_budget']:.1f} FTNS")
        else:
            print(f"❌ Transaction {i}: Failed - Insufficient budget")
        print()
    
    final_marketplace_status = await budget_manager.get_budget_status(marketplace_budget.budget_id)
    
    print("📊 Marketplace Budget Summary:")
    print(f"   💰 Total Marketplace Spending: {final_marketplace_status['category_breakdown']['marketplace_trading']['spent']:.1f} FTNS")
    print(f"   📈 Budget Utilization: {final_marketplace_status['utilization_percentage']:.1f}%")
    print()


async def main():
    """Run the complete budget management demonstration"""
    try:
        await demonstrate_budget_workflow()
        await demonstrate_marketplace_budget_integration()
        
        print("\n🎉 Budget Management Demonstration Complete!")
        print("\nKey Takeaways:")
        print("• FTNS budgets provide predictable cost control for open-ended queries")
        print("• Real-time tracking prevents runaway spending")
        print("• Category-based allocation enables granular resource management")
        print("• Automatic expansion with user approval maintains workflow continuity")
        print("• Marketplace integration supports U2U and A2A transaction budgeting")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())