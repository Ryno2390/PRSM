#!/usr/bin/env python3
"""
Fast Bulk arXiv Ingestion - Optimized for Speed
==============================================

Optimizations:
- Batch processing (1000 papers per batch)
- Bulk database inserts 
- Simplified storage (skip individual compression)
- Direct database storage for faster access
- Progress checkpointing every 10K papers
"""

import sys
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import signal
import re
sys.path.insert(0, '.')

class FastArxivIngestion:
    def __init__(self):
        self.running = True
        self.db_path = Path("/Volumes/My Passport/PRSM_Storage/storage.db")
        self.arxiv_file = Path("/Volumes/My Passport/PRSM_Storage/bulk_datasets/arxiv-metadata-oai-snapshot.json")
        self.progress_file = "fast_arxiv_progress.json"
        self.processed_count = 0
        self.accepted_count = 0
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Graceful shutdown requested...")
        self.running = False
        
    async def run_fast_ingestion(self):
        """Run optimized 150K arXiv ingestion"""
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 FAST BULK ARXIV INGESTION")
        print("=" * 50)
        print("🎯 Target: 150,000 papers")
        print("⚡ Optimized for speed (batched processing)")
        print("💾 Direct database storage")
        print("📊 Progress checkpoints every 10K")
        print()
        
        if not self.arxiv_file.exists():
            print(f"❌ arXiv dataset not found: {self.arxiv_file}")
            return False
            
        # Initialize database
        await self.setup_database()
        
        # Load existing progress
        start_offset = self.load_progress()
        if start_offset > 0:
            print(f"📋 Resuming from paper {start_offset:,}")
            
        # Process in optimized batches
        success = await self.process_in_fast_batches(start_offset)
        
        if success:
            print("🎉 Fast ingestion completed!")
        
        return success
    
    async def setup_database(self):
        """Setup optimized database with indexes"""
        
        conn = sqlite3.connect(str(self.db_path))
        
        # Create optimized arxiv_papers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arxiv_papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT,
                arxiv_id TEXT,
                publish_date TEXT,
                categories TEXT,
                domain TEXT,
                journal_ref TEXT,
                submitter TEXT,
                source TEXT DEFAULT 'fast_bulk',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast searching
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fast_title ON arxiv_papers(title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fast_domain ON arxiv_papers(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fast_categories ON arxiv_papers(categories)")
        
        conn.commit()
        conn.close()
        
        print("✅ Optimized database setup complete")
    
    async def process_in_fast_batches(self, start_offset: int = 0):
        """Process arXiv papers in fast batches"""
        
        print(f"⚡ Starting fast batch processing from offset {start_offset:,}...")
        
        batch_size = 1000  # Process 1000 papers per batch
        target_papers = 150000
        
        start_time = datetime.now(timezone.utc)
        
        try:
            with open(self.arxiv_file, 'r', encoding='utf-8') as f:
                # Skip to start offset
                for _ in range(start_offset):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                current_batch = []
                processed = start_offset
                
                for line in f:
                    if not self.running or processed >= target_papers:
                        break
                    
                    try:
                        paper_data = json.loads(line.strip())
                        standardized = self.standardize_paper(paper_data)
                        
                        # Simple quality check
                        if self.quick_quality_check(standardized):
                            current_batch.append(standardized)
                        
                        processed += 1
                        
                        # Process batch when full
                        if len(current_batch) >= batch_size:
                            accepted_in_batch = await self.store_batch(current_batch)
                            self.accepted_count += accepted_in_batch
                            current_batch = []
                            
                            # Progress report every batch
                            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                            rate = (processed - start_offset) / elapsed if elapsed > 0 else 0
                            eta_seconds = (target_papers - processed) / rate if rate > 0 else 0
                            
                            print(f"📊 Processed: {processed:,} | "
                                  f"Accepted: {self.accepted_count:,} | "
                                  f"Rate: {rate:.0f}/sec | "
                                  f"ETA: {eta_seconds/3600:.1f}h")
                        
                        # Save progress checkpoint every 10K
                        if processed % 10000 == 0:
                            self.save_progress(processed)
                    
                    except (json.JSONDecodeError, Exception) as e:
                        continue
                
                # Process final batch
                if current_batch and self.running:
                    accepted_in_batch = await self.store_batch(current_batch)
                    self.accepted_count += accepted_in_batch
                
                self.processed_count = processed
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
        
        finally:
            # Final progress save
            self.save_progress(self.processed_count)
            
            # Final report
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            final_rate = (self.processed_count - start_offset) / total_time if total_time > 0 else 0
            
            print(f"\n🏁 FAST INGESTION {'COMPLETED' if self.processed_count >= target_papers else 'STOPPED'}")
            print("=" * 50)
            print(f"📊 Processed: {self.processed_count:,}")
            print(f"✅ Accepted: {self.accepted_count:,}")
            print(f"📈 Acceptance rate: {(self.accepted_count/self.processed_count)*100:.1f}%")
            print(f"⏱️ Time: {total_time/3600:.1f} hours")
            print(f"⚡ Rate: {final_rate:.0f} papers/second")
        
        return True
    
    async def store_batch(self, batch):
        """Store batch of papers with bulk insert"""
        
        if not batch:
            return 0
            
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            # Bulk insert using executemany
            conn.executemany("""
                INSERT OR REPLACE INTO arxiv_papers 
                (id, title, abstract, authors, arxiv_id, publish_date, categories, domain, 
                 journal_ref, submitter, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                paper["id"],
                paper["title"],
                paper["abstract"], 
                paper["authors"],
                paper["arxiv_id"],
                paper["published_date"],
                ",".join(paper["categories"]) if paper["categories"] else "",
                paper["domain"],
                paper["journal_ref"],
                paper["submitter"],
                "fast_bulk",
                datetime.now(timezone.utc).isoformat()
            ) for paper in batch])
            
            conn.commit()
            return len(batch)
            
        except Exception as e:
            print(f"⚠️ Batch storage failed: {e}")
            return 0
        finally:
            conn.close()
    
    def standardize_paper(self, paper_data):
        """Convert arXiv paper to standard format"""
        
        categories = paper_data.get("categories", "").split()
        primary_category = categories[0] if categories else "unknown"
        
        # Domain mapping
        domain_mapping = {
            "cs.": "computer_science",
            "math.": "mathematics", 
            "physics.": "physics",
            "stat.": "statistics",
            "q-bio.": "biology",
            "q-fin.": "finance",
            "econ.": "economics",
            "astro-ph": "astronomy",
            "cond-mat": "physics",
            "gr-qc": "physics",
            "hep-": "physics",
            "math-ph": "physics",
            "nlin": "physics",
            "nucl-": "physics",
            "quant-ph": "physics"
        }
        
        domain = "multidisciplinary"
        for prefix, mapped_domain in domain_mapping.items():
            if primary_category.startswith(prefix):
                domain = mapped_domain
                break
        
        return {
            "id": paper_data.get("id", ""),
            "title": paper_data.get("title", "").strip(),
            "abstract": paper_data.get("abstract", "").strip(),
            "authors": paper_data.get("authors", ""),
            "arxiv_id": paper_data.get("id", ""),
            "published_date": paper_data.get("update_date", ""),
            "categories": categories,
            "domain": domain,
            "journal_ref": paper_data.get("journal-ref", ""),
            "submitter": paper_data.get("submitter", "")
        }
    
    def quick_quality_check(self, paper):
        """Fast quality check"""
        return (paper.get("title") and 
                paper.get("abstract") and 
                len(paper["title"]) >= 10 and 
                len(paper["abstract"]) >= 50)
    
    def save_progress(self, processed_count):
        """Save progress to file"""
        progress_data = {
            "processed_count": processed_count,
            "accepted_count": self.accepted_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress" if self.running else "paused"
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Progress save failed: {e}")
    
    def load_progress(self):
        """Load progress from file"""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                self.accepted_count = data.get("accepted_count", 0)
                return data.get("processed_count", 0)
        except Exception as e:
            print(f"⚠️ Progress load failed: {e}")
        return 0

async def main():
    ingestion = FastArxivIngestion()
    await ingestion.run_fast_ingestion()

if __name__ == "__main__":
    asyncio.run(main())