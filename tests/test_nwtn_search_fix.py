#!/usr/bin/env python3
"""
Test script to debug the NWTN search method reference error
"""

import asyncio
import sys
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.compute.nwtn.external_storage_config import ExternalStorageConfig, ExternalKnowledgeBase

async def test_search_method():
    """Test the search method reference issue"""
    print("Testing NWTN search method reference...")
    
    # Initialize the knowledge base
    kb = ExternalKnowledgeBase()
    
    try:
        await kb.initialize()
        print("✓ Knowledge base initialized successfully")
        
        # Test the search method that was failing
        results = await kb.search_papers("quantum gravity", max_results=5)
        print(f"✓ Search completed successfully, found {len(results)} results")
        
        # Print first result if available
        if results:
            print(f"First result: {results[0].get('title', 'No title')}")
        
    except Exception as e:
        print(f"✗ Search failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_method())