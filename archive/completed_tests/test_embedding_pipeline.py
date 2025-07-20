#!/usr/bin/env python3
"""
Test Embedding Pipeline - Process a small batch first
"""

import asyncio
import sys
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from build_paper_embeddings import PaperEmbeddingPipeline

async def main():
    # Test with smaller batch size first
    pipeline = PaperEmbeddingPipeline(
        papers_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot",
        output_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings_Test",
        batch_size=16,  # Smaller batch for testing
        max_workers=2
    )
    
    # Override the process_all_papers method to process only first 100 papers
    paper_files = pipeline.discover_paper_files()
    
    if not paper_files:
        print("No paper files found!")
        return
    
    print(f"Testing with first 100 papers out of {len(paper_files):,} total")
    
    # Process only first 100 papers
    test_files = paper_files[:100]
    
    # Initialize model
    pipeline.initialize_embedding_model()
    
    # Process in batches
    batch_id = 0
    for i in range(0, len(test_files), pipeline.batch_size):
        batch_files = test_files[i:i + pipeline.batch_size]
        
        print(f"Processing batch {batch_id + 1}: {len(batch_files)} papers")
        
        # Process batch
        try:
            embeddings = pipeline.process_paper_batch(batch_files)
            
            # Save batch
            if embeddings:
                pipeline.save_embeddings_batch(embeddings, batch_id)
                print(f"✅ Batch {batch_id} completed: {len(embeddings)} embeddings")
            else:
                print(f"❌ Batch {batch_id} failed: no embeddings generated")
                
        except Exception as e:
            print(f"❌ Batch {batch_id} failed with error: {e}")
        
        batch_id += 1
    
    print(f"✅ Test completed: {pipeline.processed_count} papers processed")

if __name__ == "__main__":
    asyncio.run(main())