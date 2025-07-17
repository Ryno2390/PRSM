#!/usr/bin/env python3
"""
Debug paper loading issues
"""

import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from build_paper_embeddings import PaperEmbeddingPipeline

def test_single_paper():
    # Get the first paper file
    papers_dir = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot")
    paper_files = list(papers_dir.rglob("*.dat"))
    
    if not paper_files:
        print("No paper files found!")
        return
    
    test_file = paper_files[0]
    print(f"Testing paper file: {test_file}")
    
    # Try to load it directly
    try:
        # Try different loading methods
        paper_json = None
        try:
            # Method 1: Text mode
            with gzip.open(test_file, 'rt', encoding='utf-8') as f:
                paper_json = json.load(f)
                print("✅ Loaded with text mode")
        except Exception as e1:
            try:
                # Method 2: Binary mode with JSON
                with gzip.open(test_file, 'rb') as f:
                    paper_data = f.read()
                    paper_json = json.loads(paper_data.decode('utf-8'))
                print("✅ Loaded with binary mode")
            except Exception as e2:
                try:
                    # Method 3: Pickle format
                    import pickle
                    with gzip.open(test_file, 'rb') as f:
                        paper_json = pickle.load(f)
                    print("✅ Loaded with pickle format")
                except Exception as e3:
                    print(f"❌ All methods failed: {e1}, {e2}, {e3}")
                    raise
        
        print("✅ Paper loaded successfully")
        print(f"Paper ID: {paper_json.get('id', 'unknown')}")
        print(f"Title: {paper_json.get('title', 'Unknown')}")
        print(f"Abstract length: {len(paper_json.get('abstract', ''))}")
        
        # Test with pipeline
        pipeline = PaperEmbeddingPipeline()
        paper_data = pipeline.load_paper_data(test_file)
        
        if paper_data:
            print("✅ Pipeline loaded paper successfully")
            print(f"Text content length: {len(paper_data.get_text_content())}")
            
            # Test embedding
            pipeline.initialize_embedding_model()
            embeddings = pipeline.process_paper_batch([test_file])
            
            if embeddings:
                print(f"✅ Embedding generated successfully: {len(embeddings)} embeddings")
                print(f"Embedding shape: {embeddings[0].embedding.shape}")
            else:
                print("❌ No embeddings generated")
        else:
            print("❌ Pipeline failed to load paper")
            
    except Exception as e:
        print(f"❌ Error loading paper: {e}")

if __name__ == "__main__":
    test_single_paper()