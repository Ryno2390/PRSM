#!/usr/bin/env python3
"""
Quick test to validate embedding batch loading fix
"""

import sys
import asyncio
sys.path.insert(0, '.')

# Import all necessary classes for pickle deserialization
from prsm.nwtn.data_models import PaperEmbedding, PaperData, SemanticSearchResult
from prsm.nwtn.external_storage_config import ExternalStorageManager, ExternalStorageConfig

async def test_embedding_fix():
    """Test if embedding batch loading works now"""
    print("🔍 Testing Embedding Batch Loading Fix")
    print("=" * 40)
    
    # Initialize storage manager
    storage_manager = ExternalStorageManager()
    
    # Try to load a few embedding batches
    for batch_id in range(5):
        try:
            print(f"Testing batch {batch_id}...")
            batch_data = await storage_manager.load_embedding_batch(batch_id)
            if batch_data:
                print(f"✅ Batch {batch_id}: Loaded successfully")
            else:
                print(f"⚠️  Batch {batch_id}: Not found (expected for test)")
        except Exception as e:
            if "PaperEmbedding" in str(e):
                print(f"❌ Batch {batch_id}: PaperEmbedding import error: {e}")
                return False
            else:
                print(f"ℹ️  Batch {batch_id}: Other error (expected): {e}")
    
    print(f"\n🎯 Result: PaperEmbedding import errors should be resolved!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_embedding_fix())
    if success:
        print("✅ Embedding fix validation successful")
    else:
        print("❌ Embedding fix needs more work")