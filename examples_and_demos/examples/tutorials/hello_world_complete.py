#!/usr/bin/env python3
"""
Complete Hello World Tutorial Example
All-in-one example demonstrating PRSM basics with comprehensive error handling
"""

import asyncio
import sys
import time
from pathlib import Path

# Add PRSM to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import UserInput, PRSMSession

class HelloWorldDemo:
    """Complete PRSM Hello World demonstration"""
    
    def __init__(self):
        self.orchestrator = NWTNOrchestrator()
        self.session = None
        
    async def setup_session(self):
        """Initialize a PRSM session"""
        print("🔧 Setting up PRSM session...")
        
        self.session = PRSMSession(
            user_id="hello_world_demo",
            nwtn_context_allocation=100,
            session_config={
                "max_query_cost": 50,
                "quality_threshold": 80,
                "timeout": 30
            }
        )
        print("✅ Session ready!")
        
    async def basic_query(self):
        """Demonstrate a basic PRSM query"""
        print("\n📝 Running Basic Query...")
        print("-" * 40)
        
        query = UserInput(
            user_id="hello_world_demo",
            prompt="Hello PRSM! Please introduce yourself and explain what you do.",
            context_allocation=25
        )
        
        print(f"Query: {query.prompt}")
        print(f"FTNS Budget: {query.context_allocation}")
        print("Processing...\n")
        
        start_time = time.time()
        response = await self.orchestrator.process_query(query)
        end_time = time.time()
        
        print("✅ RESPONSE RECEIVED:")
        print("=" * 50)
        print(f"🤖 Answer: {response.final_answer}")
        print(f"💰 FTNS Used: {response.ftns_charged}")
        print(f"⏱️  Time: {end_time - start_time:.2f}s")
        print(f"🎯 Quality: {response.quality_score}/100")
        
        if response.reasoning_trace:
            print("\n🧭 Reasoning Steps:")
            for i, step in enumerate(response.reasoning_trace, 1):
                print(f"   {i}. {step}")
                
        return response
        
    async def scientific_query(self):
        """Demonstrate a scientific query"""
        print("\n🔬 Running Scientific Query...")
        print("-" * 40)
        
        query = UserInput(
            user_id="hello_world_demo",
            prompt="""
            Explain how photosynthesis works at the molecular level, 
            including the light-dependent and light-independent reactions.
            Make it accessible to a high school student.
            """,
            context_allocation=40
        )
        
        print(f"Query: Scientific explanation of photosynthesis")
        print(f"FTNS Budget: {query.context_allocation}")
        print("Processing...\n")
        
        response = await self.orchestrator.process_query(query)
        
        print("✅ SCIENTIFIC RESPONSE:")
        print("=" * 50)
        print(f"🤖 Answer: {response.final_answer}")
        print(f"💰 FTNS Used: {response.ftns_charged}")
        print(f"🎯 Quality: {response.quality_score}/100")
        
        return response
        
    async def creative_query(self):
        """Demonstrate a creative query"""
        print("\n🎨 Running Creative Query...")
        print("-" * 40)
        
        query = UserInput(
            user_id="hello_world_demo",
            prompt="""
            Write a short poem about artificial intelligence that also 
            explains how neural networks learn. Make it both beautiful 
            and educational.
            """,
            context_allocation=35
        )
        
        print(f"Query: Creative AI poem with educational content")
        print(f"FTNS Budget: {query.context_allocation}")
        print("Processing...\n")
        
        response = await self.orchestrator.process_query(query)
        
        print("✅ CREATIVE RESPONSE:")
        print("=" * 50)
        print(f"🎨 Poem:\n{response.final_answer}")
        print(f"💰 FTNS Used: {response.ftns_charged}")
        print(f"🎯 Quality: {response.quality_score}/100")
        
        return response
        
    async def budget_demo(self):
        """Demonstrate FTNS budget management"""
        print("\n💰 FTNS Budget Management Demo...")
        print("-" * 40)
        
        # Show current budget
        if self.session:
            print(f"💳 Session Budget: {self.session.nwtn_context_allocation} FTNS")
            
        # Low-cost query
        cheap_query = UserInput(
            user_id="hello_world_demo",
            prompt="What is 2+2?",
            context_allocation=5
        )
        
        print("\n💸 Low-cost query (5 FTNS):")
        response1 = await self.orchestrator.process_query(cheap_query)
        print(f"   Answer: {response1.final_answer}")
        print(f"   Actual cost: {response1.ftns_charged} FTNS")
        
        # Medium-cost query
        medium_query = UserInput(
            user_id="hello_world_demo",
            prompt="Explain the theory of relativity in simple terms",
            context_allocation=20
        )
        
        print("\n💳 Medium-cost query (20 FTNS):")
        response2 = await self.orchestrator.process_query(medium_query)
        print(f"   Answer: {response2.final_answer[:100]}...")
        print(f"   Actual cost: {response2.ftns_charged} FTNS")
        
        # Cost comparison
        print(f"\n📊 Cost Comparison:")
        print(f"   Simple math: {response1.ftns_charged} FTNS")
        print(f"   Complex explanation: {response2.ftns_charged} FTNS")
        print(f"   💡 Complex queries cost more but provide better answers")
        
    async def run_complete_demo(self):
        """Run the complete Hello World demonstration"""
        print("🌟 PRSM Complete Hello World Demo")
        print("=" * 50)
        print("This demo shows PRSM's capabilities across different query types")
        print()
        
        try:
            # Setup
            await self.setup_session()
            
            # Run different types of queries
            await self.basic_query()
            await self.scientific_query() 
            await self.creative_query()
            await self.budget_demo()
            
            print("\n🎉 DEMO COMPLETE!")
            print("=" * 50)
            print("✅ You've successfully:")
            print("   • Initialized PRSM and created a session")
            print("   • Run basic, scientific, and creative queries")
            print("   • Explored FTNS budget management")
            print("   • Seen PRSM's reasoning and quality scoring")
            print()
            print("🚀 Next Steps:")
            print("   • Try the Foundation tutorials for deeper concepts")
            print("   • Explore API usage and SDK integration")
            print("   • Build your own PRSM applications")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Ensure PRSM is properly installed: prsm-dev setup")
            print("   2. Check services are running: prsm-dev status")
            print("   3. Verify API keys: config/api_keys.env")
            print("   4. Run diagnostics: prsm-dev diagnose")

async def main():
    """Main entry point"""
    demo = HelloWorldDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    # Run the complete demonstration
    asyncio.run(main())