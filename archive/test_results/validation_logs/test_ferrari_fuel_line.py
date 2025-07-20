#!/usr/bin/env python3
"""
Test Ferrari Fuel Line Connection
=================================

This script tests the connection between NWTN and the external drive containing
150K+ papers and 4,727 embedding batch files.

This implements the roadmap from NWTN_FERRARI_FUEL_LINE_ROADMAP.md
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ferrari_fuel_line():
    """Test the Ferrari fuel line connection"""
    print("🏎️  Testing Ferrari Fuel Line Connection...")
    print("=" * 80)
    
    # Set up environment
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    os.environ["PRSM_NWTN_MODEL"] = "claude-3-5-sonnet-20241022"
    
    try:
        # Test 1: External Storage Configuration
        print("\n🔧 Test 1: External Storage Configuration")
        print("-" * 40)
        
        from prsm.nwtn.external_storage_config import ExternalStorageConfig, ExternalStorageManager
        
        config = ExternalStorageConfig()
        print(f"✓ External Drive Path: {config.external_drive_path}")
        print(f"✓ Storage Path: {config.storage_path}")
        print(f"✓ Available: {config.is_available}")
        print(f"✓ Embeddings Count: {config.embeddings_count}")
        print(f"✓ Total Size: {config.total_size_gb:.2f} GB")
        
        if not config.is_available:
            print("❌ External storage not available - please ensure drive is connected")
            return False
        
        # Test 2: External Storage Manager
        print("\n🔧 Test 2: External Storage Manager")
        print("-" * 40)
        
        storage_manager = ExternalStorageManager(config)
        initialized = await storage_manager.initialize()
        
        if initialized:
            print("✅ External storage manager initialized successfully")
            
            # Get storage stats
            stats = await storage_manager.get_storage_stats()
            print(f"✓ Papers Count: {stats.get('papers_count', 'Unknown')}")
            print(f"✓ Embeddings Count: {stats.get('embeddings_count', 0)}")
            print(f"✓ Total Size: {stats.get('total_size_gb', 0):.2f} GB")
        else:
            print("❌ Failed to initialize external storage manager")
            return False
        
        # Test 3: External Knowledge Base
        print("\n🔧 Test 3: External Knowledge Base")
        print("-" * 40)
        
        from prsm.nwtn.external_storage_config import ExternalKnowledgeBase
        
        kb = ExternalKnowledgeBase(storage_manager)
        kb_initialized = await kb.initialize()
        
        if kb_initialized:
            print("✅ External knowledge base initialized successfully")
            
            # Test search functionality
            print("\n🔍 Testing knowledge base search...")
            papers = await kb.search_papers("quantum mechanics", max_results=5)
            print(f"✓ Found {len(papers)} papers for 'quantum mechanics'")
            
            for i, paper in enumerate(papers[:3]):
                print(f"  {i+1}. {paper.get('title', 'No title')[:60]}...")
                print(f"     Authors: {paper.get('authors', 'Unknown')}")
                print(f"     ArXiv ID: {paper.get('arxiv_id', 'N/A')}")
        else:
            print("❌ Failed to initialize external knowledge base")
            return False
        
        # Test 4: NWTN Voicebox Integration
        print("\n🔧 Test 4: NWTN Voicebox Integration")
        print("-" * 40)
        
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider
        
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Configure API key
        await voicebox.configure_api_key(
            user_id="test_user_ferrari",
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Add FTNS balance
        from prsm.tokenomics.ftns_service import get_ftns_service
        ftns_service = await get_ftns_service()
        await ftns_service.reward_contribution("test_user_ferrari", "data", 2000.0)
        
        print("✅ NWTN Voicebox initialized with Ferrari fuel line")
        print(f"✓ External Knowledge Base Available: {voicebox.external_knowledge_base.initialized if voicebox.external_knowledge_base else False}")
        
        # Test 5: Ferrari Fuel Line Query
        print("\n🔧 Test 5: Ferrari Fuel Line Query")
        print("-" * 40)
        
        test_query = "What is the uncertainty principle in quantum mechanics?"
        
        print(f"🔍 Query: {test_query}")
        print("🏎️  Running through Ferrari (NWTN + External Knowledge Base)...")
        
        # Process query through voicebox
        response = await voicebox.process_query(
            user_id="test_user_ferrari",
            query=test_query,
            context={}
        )
        
        print("\n" + "="*80)
        print("🎯 FERRARI FUEL LINE RESULTS:")
        print("="*80)
        print(f"📝 Natural Language Response: {response.natural_language_response}")
        print(f"🧠 Reasoning Engines Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print(f"🔗 Source Links: {len(response.source_links)} sources")
        
        # Display source links (the Ferrari fuel!)
        if response.source_links:
            print("\n📄 SOURCE LINKS FROM FERRARI FUEL LINE:")
            for i, link in enumerate(response.source_links):
                print(f"  {i+1}. {link.title}")
                print(f"     Creator: {link.creator}")
                print(f"     Link: {link.ipfs_link}")
                print(f"     Relevance: {link.relevance_score}")
        else:
            print("\n⚠️  No source links generated - fuel line may not be connected")
        
        print(f"\n📊 Attribution Summary: {response.attribution_summary}")
        
        # Test 6: Verify Ferrari Performance
        print("\n🔧 Test 6: Ferrari Performance Verification")
        print("-" * 40)
        
        success_criteria = {
            "Natural Language Response": len(response.natural_language_response) > 100,
            "Confidence Score": response.confidence_score > 0.0,
            "Processing Time": response.processing_time_seconds > 0.5,
            "Source Links": len(response.source_links) > 0,
            "Attribution Summary": len(response.attribution_summary) > 0
        }
        
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 FERRARI FUEL LINE CONNECTION SUCCESSFUL!")
            print("🏎️  The Ferrari has a FULL tank of gas and is ready to race!")
            print("🔥 150K+ papers are now connected to live attribution system")
            return True
        else:
            print("\n🚨 FERRARI FUEL LINE CONNECTION ISSUES DETECTED")
            print("⚠️  Some components are not working as expected")
            return False
        
    except Exception as e:
        print(f"❌ Error during Ferrari fuel line test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ferrari_fuel_line())
    if success:
        print("\n🏆 FERRARI FUEL LINE TEST PASSED!")
        print("🎯 Phase 1 of the roadmap is complete")
    else:
        print("\n💥 FERRARI FUEL LINE TEST FAILED!")
        print("🔧 Check the connection and try again")