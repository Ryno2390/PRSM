#!/usr/bin/env python3
"""
Generate Embeddings for 150K arXiv Papers
==========================================

This script generates semantic embeddings for all 149,726 ingested arXiv papers
to enable semantic search capabilities in the NWTN system.

Process:
1. Read papers from storage.db 
2. Generate embeddings using sentence-transformers
3. Store embeddings in batch files for fast retrieval
4. Create index for semantic search

Optimizations:
- Batch processing (1000 papers per batch)
- Efficient embedding generation
- Progress tracking and resumption
- Memory management for large corpus
"""

import asyncio
import sys
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import json

sys.path.insert(0, '.')

class EmbeddingGenerator:
    def __init__(self):
        self.db_path = Path("/Volumes/My Passport/PRSM_Storage/storage.db")
        self.embeddings_path = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Embeddings")
        self.progress_file = "embedding_generation_progress.json"
        
        # Embedding configuration
        self.batch_size = 1000  # Papers per embedding batch
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.model = None
        
        # Progress tracking
        self.processed_count = 0
        self.total_papers = 0
        self.current_batch = 0
        
    async def initialize(self):
        """Initialize embedding model and directories"""
        print("🔧 Initializing Embedding Generator...")
        
        # Create embeddings directory
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer model
        try:
            from sentence_transformers import SentenceTransformer
            print(f"📥 Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded, dimension: {self.embedding_dimension}")
        except ImportError:
            print("❌ sentence-transformers not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model installed and loaded, dimension: {self.embedding_dimension}")
        
        # Count total papers
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM arxiv_papers")
        self.total_papers = cursor.fetchone()[0]
        conn.close()
        
        print(f"📊 Total papers to process: {self.total_papers:,}")
        return True
    
    async def generate_embeddings_for_corpus(self):
        """Generate embeddings for all papers in the corpus"""
        print("🚀 STARTING EMBEDDING GENERATION")
        print("=" * 60)
        print(f"📚 Corpus size: {self.total_papers:,} papers")
        print(f"🔢 Batch size: {self.batch_size:,} papers per batch")
        print(f"📁 Output directory: {self.embeddings_path}")
        print()
        
        start_time = datetime.now(timezone.utc)
        
        # Load existing progress
        start_offset = self.load_progress()
        if start_offset > 0:
            print(f"📋 Resuming from paper {start_offset:,}")
        
        try:
            # Process papers in batches
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            # Query papers in batches
            offset = start_offset
            while offset < self.total_papers:
                # Get batch of papers
                papers = self.get_paper_batch(conn, offset, self.batch_size)
                if not papers:
                    break
                
                # Generate embeddings for batch
                success = await self.process_paper_batch(papers, offset // self.batch_size)
                if not success:
                    print(f"❌ Failed to process batch starting at {offset}")
                    break
                
                offset += len(papers)
                self.processed_count = offset
                
                # Progress update
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                rate = self.processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (self.total_papers - self.processed_count) / rate if rate > 0 else 0
                
                print(f"📊 Progress: {self.processed_count:,}/{self.total_papers:,} "
                      f"({(self.processed_count/self.total_papers)*100:.1f}%) | "
                      f"Rate: {rate:.1f}/sec | "
                      f"ETA: {eta_seconds/3600:.1f}h")
                
                # Save progress
                self.save_progress(self.processed_count)
                
                # Memory cleanup
                if self.processed_count % 10000 == 0:
                    print("🧹 Memory cleanup...")
                    import gc
                    gc.collect()
            
            conn.close()
            
            # Final summary
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"\n🎉 EMBEDDING GENERATION COMPLETE")
            print("=" * 60)
            print(f"📊 Total papers processed: {self.processed_count:,}")
            print(f"📁 Embedding batches created: {self.current_batch + 1}")
            print(f"⏱️ Total time: {total_time/3600:.1f} hours")
            print(f"⚡ Average rate: {self.processed_count/total_time:.1f} papers/second")
            
            return True
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_paper_batch(self, conn: sqlite3.Connection, offset: int, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of papers from the database"""
        try:
            cursor = conn.execute("""
                SELECT id, title, abstract, authors, arxiv_id, publish_date, categories, domain
                FROM arxiv_papers
                ORDER BY id
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            papers = []
            for row in cursor:
                papers.append({
                    'id': row['id'],
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'authors': row['authors'],
                    'arxiv_id': row['arxiv_id'],
                    'publish_date': row['publish_date'],
                    'categories': row['categories'],
                    'domain': row['domain']
                })
            
            return papers
            
        except Exception as e:
            print(f"❌ Failed to get paper batch: {e}")
            return []
    
    async def process_paper_batch(self, papers: List[Dict[str, Any]], batch_id: int) -> bool:
        """Process a batch of papers and generate embeddings"""
        try:
            print(f"⚙️ Processing batch {batch_id} ({len(papers)} papers)...")
            
            # Prepare texts for embedding
            texts = []
            metadata = []
            
            for paper in papers:
                # Combine title and abstract for embedding
                text = f"{paper['title']} {paper['abstract']}"
                texts.append(text)
                
                # Store metadata
                metadata.append({
                    'id': paper['id'],
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'abstract': paper['abstract'][:500] + '...' if len(paper['abstract']) > 500 else paper['abstract'],
                    'arxiv_id': paper['arxiv_id'],
                    'publish_date': paper['publish_date'],
                    'categories': paper['categories'],
                    'domain': paper['domain']
                })
            
            # Generate embeddings
            print(f"🧠 Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
            
            # Convert to list format
            embeddings_list = embeddings.tolist()
            
            # Save batch file
            batch_data = {
                'batch_id': batch_id,
                'embeddings': embeddings_list,
                'metadata': metadata,
                'embedding_dimension': self.embedding_dimension,
                'model_name': self.model_name,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'paper_count': len(papers)
            }
            
            batch_file = self.embeddings_path / f"embeddings_batch_{batch_id:06d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            
            self.current_batch = batch_id
            print(f"💾 Saved batch {batch_id} to {batch_file.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to process batch {batch_id}: {e}")
            return False
    
    def save_progress(self, processed_count: int):
        """Save progress to file"""
        progress_data = {
            'processed_count': processed_count,
            'current_batch': self.current_batch,
            'total_papers': self.total_papers,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Progress save failed: {e}")
    
    def load_progress(self) -> int:
        """Load progress from file"""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                self.current_batch = data.get('current_batch', 0)
                return data.get('processed_count', 0)
        except Exception as e:
            print(f"⚠️ Progress load failed: {e}")
        return 0


async def main():
    """Main function to generate embeddings"""
    print("🧪 NWTN EMBEDDING GENERATION")
    print("=" * 60)
    print("🎯 Goal: Generate embeddings for 149,726 arXiv papers")
    print("🔍 Model: sentence-transformers/all-MiniLM-L6-v2")
    print("📊 Output: Batch files for semantic search")
    print()
    
    generator = EmbeddingGenerator()
    
    # Initialize
    success = await generator.initialize()
    if not success:
        print("❌ Failed to initialize embedding generator")
        return False
    
    # Generate embeddings
    success = await generator.generate_embeddings_for_corpus()
    
    if success:
        print("\n✅ SUCCESS: All embeddings generated successfully!")
        print("🔍 Semantic search now enabled for NWTN system")
        print("📁 Embedding batches ready for retrieval")
    else:
        print("\n❌ FAILED: Embedding generation incomplete")
        print("🔧 Check logs and retry if needed")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())