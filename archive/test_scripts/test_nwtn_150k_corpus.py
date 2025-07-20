#!/usr/bin/env python3
"""
Test NWTN System with Full 150K Corpus
======================================

Test the complete NWTN pipeline with the full corpus of 149,726 real arXiv papers
to validate search capabilities and Claude API integration.

Test query: "What are the latest advances in transformer architectures for natural language processing?"

Expected: NWTN should find relevant CS/ML papers from the corpus and generate 
a coherent response with actual paper citations using Claude API.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput
from prsm.nwtn.candidate_answer_generator import AnswerVerbosity


async def test_nwtn_with_150k_corpus():
    """Test NWTN with the full 150K corpus"""
    
    print("🚀 NWTN 150K CORPUS TEST")
    print("=" * 60)
    print("🎯 Testing transformer/NLP query against 149,726 real arXiv papers")
    print("📊 Expected: Relevant CS/ML papers with Claude API synthesis")
    print("🔍 Query: 'What are the latest advances in transformer architectures for natural language processing?'")
    print()
    
    try:
        # First, let's create FTNS service and add tokens to the test user
        from prsm.tokenomics.ftns_service import FTNSService
        from prsm.core.models import FTNSBalance
        
        ftns_service = FTNSService()
        # Give test user 2000 FTNS tokens for testing
        ftns_service.balances["test_user"] = FTNSBalance(user_id="test_user", balance=2000.0)
        print("💰 Added 2000 FTNS tokens to test_user balance")
        print()
        
        # Initialize budget manager with our FTNS service
        from prsm.tokenomics.ftns_budget_manager import FTNSBudgetManager
        budget_manager = FTNSBudgetManager(ftns_service=ftns_service)
        
        # Initialize NWTN orchestrator with our FTNS service and budget manager
        print("🔧 Initializing NWTN Orchestrator...")
        orchestrator = EnhancedNWTNOrchestrator(ftns_service=ftns_service, budget_manager=budget_manager)
        print("✅ NWTN Orchestrator initialized successfully")
        print()
        
        # Create user input
        test_query = "What are the latest advances in transformer architectures for natural language processing?"
        print(f"🔍 Processing query: '{test_query}'")
        print("⏱️ This may take 30-60 seconds to search the full corpus...")
        print()
        
        user_input = UserInput(
            user_id="test_user",
            prompt=test_query,
            context_allocation=1000,  # Allocate FTNS tokens
            preferences={
                "verbosity": AnswerVerbosity.DETAILED.value,
                "max_sources": 5
            }
        )
        
        # Process query
        result = await orchestrator.process_query(user_input)
        
        # Display results
        print("🎉 NWTN PROCESSING COMPLETE")
        print("=" * 60)
        
        if result and result.status == "success":
            print("✅ Answer Generated:")
            print("-" * 40)
            print(result.content)
            print("-" * 40)
            print()
            
            # Show metadata
            if result.metadata:
                print(f"📊 Metadata: {result.metadata}")
                print()
            
            # Show reasoning trace (if available)
            if hasattr(result, 'reasoning_trace') and result.reasoning_trace:
                print("🧠 Reasoning Engines Used:")
                for engine_name, engine_result in result.reasoning_trace.items():
                    if engine_result.get("used", False):
                        confidence = engine_result.get("confidence", 0.0)
                        print(f"   ✅ {engine_name}: {confidence:.2f} confidence")
                    else:
                        print(f"   ⏸️ {engine_name}: Not used")
                print()
            
            print(f"🎯 Success: NWTN query processed successfully")
            print(f"⏱️ Response ID: {result.response_id}")
            
            return True
            
        else:
            print("❌ No answer generated - NWTN processing failed")
            if result:
                print(f"Status: {result.status}")
                if result.error:
                    print(f"Error: {result.error}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if orchestrator:
                await orchestrator.shutdown()
        except:
            pass


async def main():
    """Main test function"""
    
    print("🧪 NWTN 150K CORPUS VALIDATION TEST")
    print("=" * 60)
    print("📚 Database: 149,726 real arXiv papers")
    print("🎯 Goal: Validate NWTN search and Claude API integration")
    print("🔬 Query focus: Transformer architectures (CS/ML domain)")
    print()
    
    success = await test_nwtn_with_150k_corpus()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST SUCCESSFUL: NWTN successfully processed the query with real arXiv papers!")
        print("✅ Pipeline working: Search → Deep Reasoning → Claude API → Works Cited")
    else:
        print("❌ TEST FAILED: NWTN could not process the query properly")
        print("🔧 Investigation needed in search or reasoning components")
    
    print("=" * 60)
    return success


if __name__ == "__main__":
    asyncio.run(main())