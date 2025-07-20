#!/usr/bin/env python3
"""
Test Ferrari Deep Reasoning Integration
======================================

This script tests that NWTN's deep reasoning engines are actually using the
external knowledge base during reasoning, not just during source link generation.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ferrari_deep_reasoning():
    """Test that reasoning engines use external knowledge base"""
    print("🏎️  Testing Ferrari Deep Reasoning Integration...")
    print("=" * 80)
    
    # Set up environment
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    os.environ["PRSM_NWTN_MODEL"] = "claude-3-5-sonnet-20241022"
    
    try:
        # Test 1: Initialize NWTN with Ferrari fuel line
        print("\n🔧 Test 1: NWTN MetaReasoningEngine Ferrari Integration")
        print("-" * 60)
        
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider
        
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Configure API key
        await voicebox.configure_api_key(
            user_id="test_user_deep",
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Add FTNS balance
        from prsm.tokenomics.ftns_service import get_ftns_service
        ftns_service = await get_ftns_service()
        await ftns_service.reward_contribution("test_user_deep", "data", 3000.0)
        
        # Check if MetaReasoningEngine has external knowledge base
        has_external_kb = (voicebox.meta_reasoning_engine.external_knowledge_base is not None and
                          voicebox.meta_reasoning_engine.external_knowledge_base.initialized)
        
        print(f"✓ MetaReasoningEngine External KB: {has_external_kb}")
        print(f"✓ Voicebox External KB: {voicebox.external_knowledge_base.initialized if voicebox.external_knowledge_base else False}")
        
        # Test 2: Direct MetaReasoningEngine Test
        print("\n🔧 Test 2: Direct MetaReasoningEngine Test")
        print("-" * 60)
        
        test_query = "What is quantum tunneling and how does it work?"
        
        # Test the meta reasoning engine directly
        reasoning_result = await voicebox.meta_reasoning_engine.meta_reason(
            query=test_query,
            context={"user_id": "test_user_deep"}
        )
        
        print(f"✓ Reasoning Result ID: {reasoning_result.id}")
        print(f"✓ Query: {reasoning_result.query}")
        print(f"✓ Original Query: {reasoning_result.original_query}")
        print(f"✓ Content Sources: {len(reasoning_result.content_sources)} sources")
        print(f"✓ Meta Confidence: {reasoning_result.meta_confidence}")
        
        # Display content sources
        if reasoning_result.content_sources:
            print("\n📄 CONTENT SOURCES FOUND BY REASONING:")
            for i, source in enumerate(reasoning_result.content_sources):
                print(f"  {i+1}. {source}")
        else:
            print("\n❌ No content sources found by reasoning engine")
        
        # Test 3: Full Voicebox Test with Source Links
        print("\n🔧 Test 3: Full Voicebox Test")
        print("-" * 60)
        
        print(f"🔍 Query: {test_query}")
        print("🏎️  Running through complete Ferrari pipeline...")
        
        # Process query through voicebox
        response = await voicebox.process_query(
            user_id="test_user_deep",
            query=test_query,
            context={}
        )
        
        print("\n" + "="*80)
        print("🎯 FERRARI DEEP REASONING RESULTS:")
        print("="*80)
        print(f"📝 Natural Language Response: {response.natural_language_response[:200]}...")
        print(f"🧠 Reasoning Engines Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print(f"🔗 Source Links: {len(response.source_links)} sources")
        
        # Display source links
        if response.source_links:
            print("\n📄 SOURCE LINKS FROM FERRARI REASONING:")
            for i, link in enumerate(response.source_links):
                print(f"  {i+1}. {link.title}")
                print(f"     Creator: {link.creator}")
                print(f"     Link: {link.ipfs_link}")
                print(f"     Relevance: {link.relevance_score}")
        
        print(f"\n📊 Attribution Summary: {response.attribution_summary}")
        
        # Test 4: Verify Deep Integration
        print("\n🔧 Test 4: Deep Integration Verification")
        print("-" * 60)
        
        # Check if the reasoning actually found sources (not just fallback search)
        reasoning_found_sources = hasattr(response, 'structured_insights') and \
                                 hasattr(response.structured_insights, 'content_sources') and \
                                 len(response.structured_insights.content_sources) > 0
        
        voicebox_found_sources = len(response.source_links) > 0
        
        print(f"✓ Reasoning Engine Found Sources: {reasoning_found_sources}")
        print(f"✓ Voicebox Generated Source Links: {voicebox_found_sources}")
        
        # Check reasoning trace for external knowledge usage
        if hasattr(response, 'structured_insights') and response.structured_insights:
            result = response.structured_insights
            if hasattr(result, 'content_sources') and result.content_sources:
                print(f"✓ Reasoning Used {len(result.content_sources)} external sources")
                
                # Show which sources were used
                for i, source in enumerate(result.content_sources[:3]):
                    print(f"  {i+1}. {source}")
            else:
                print("⚠️  Reasoning did not use external sources")
        
        # Test 5: Success Criteria
        print("\n🔧 Test 5: Success Criteria Verification")
        print("-" * 60)
        
        success_criteria = {
            "MetaReasoningEngine has External KB": has_external_kb,
            "Reasoning found content sources": reasoning_found_sources or len(reasoning_result.content_sources) > 0,
            "Source links generated": voicebox_found_sources,
            "Natural language response": len(response.natural_language_response) > 100,
            "Confidence score": response.confidence_score > 0.0
        }
        
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 FERRARI DEEP REASONING INTEGRATION SUCCESSFUL!")
            print("🏎️  The Ferrari is using the external knowledge base during reasoning!")
            print("🔥 Deep reasoning engines are connected to 150K+ papers")
            return True
        else:
            print("\n🚨 FERRARI DEEP REASONING INTEGRATION ISSUES DETECTED")
            print("⚠️  Some integration points are not working correctly")
            return False
        
    except Exception as e:
        print(f"❌ Error during Ferrari deep reasoning test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ferrari_deep_reasoning())
    if success:
        print("\n🏆 FERRARI DEEP REASONING TEST PASSED!")
        print("🎯 The Ferrari is fully integrated with external knowledge!")
    else:
        print("\n💥 FERRARI DEEP REASONING TEST FAILED!")
        print("🔧 The reasoning engines need better integration")