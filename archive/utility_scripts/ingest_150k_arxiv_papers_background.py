#!/usr/bin/env python3
"""
Background 150K arXiv Papers Ingestion
=====================================

This script ingests the complete 150K arXiv dataset in the background
with progress monitoring and can be interrupted/resumed.
"""

import sys
import asyncio
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import signal
sys.path.insert(0, '.')

from prsm.nwtn.bulk_dataset_processor import BulkDatasetProcessor

class BackgroundIngestion:
    def __init__(self):
        self.running = True
        self.processor = None
        self.progress_file = "arxiv_ingestion_progress.json"
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Received interrupt signal. Gracefully shutting down...")
        self.running = False
        
    async def run_ingestion(self):
        """Run the full 150K arXiv ingestion with progress tracking"""
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 STARTING 150K ARXIV PAPERS BACKGROUND INGESTION")
        print("=" * 60)
        print("📊 Target: 150,000 arXiv papers")
        print("⏱️  Estimated time: 2-4 hours")
        print("💾 Storage: External drive with compression")
        print("🔍 Embeddings: High-dimensional semantic vectors")
        print("📋 Progress saved to:", self.progress_file)
        print("\n💡 Press Ctrl+C to gracefully stop and save progress")
        print("=" * 60)
        
        try:
            # Initialize bulk dataset processor
            print("\n📦 Initializing Bulk Dataset Processor...")
            self.processor = BulkDatasetProcessor()
            await self.processor.initialize()
            print("✅ Bulk Dataset Processor initialized")
            
            # Load existing progress if available
            start_offset = self.load_progress()
            if start_offset > 0:
                print(f"📋 Resuming from paper {start_offset:,}")
            
            # Check if arXiv dataset exists
            bulk_data_path = Path("/Volumes/My Passport/PRSM_Storage/bulk_datasets")
            arxiv_file = bulk_data_path / "arxiv-metadata-oai-snapshot.json"
            
            if not arxiv_file.exists():
                print("\n❌ arXiv dataset file not found!")
                print("📥 Please download the arXiv dataset first:")
                print("1. Go to: https://www.kaggle.com/datasets/Cornell-University/arxiv")
                print("2. Download and extract 'arxiv-metadata-oai-snapshot.json'")
                print(f"3. Place it in: {bulk_data_path}/")
                return False
            
            print(f"✅ Found arXiv dataset: {arxiv_file}")
            
            # Process papers with progress tracking
            await self.process_arxiv_with_progress(arxiv_file, start_offset)
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.processor:
                await self.processor.shutdown()
        
        return True
    
    async def process_arxiv_with_progress(self, arxiv_file: Path, start_offset: int = 0):
        """Process arXiv dataset with progress tracking and checkpointing"""
        
        print(f"\n🔄 Processing arXiv dataset from offset {start_offset:,}...")
        
        target_papers = 150000
        batch_size = 1000  # Process in batches for better progress reporting
        processed = start_offset
        accepted = 0
        
        start_time = datetime.now(timezone.utc)
        last_progress_save = start_time
        
        try:
            with open(arxiv_file, 'r', encoding='utf-8') as f:
                # Skip to start offset
                for _ in range(start_offset):
                    try:
                        next(f)
                    except StopIteration:
                        break
                
                current_batch = []
                
                for line_num, line in enumerate(f, start=start_offset):
                    if not self.running:
                        print("\n⚠️ Graceful shutdown requested")
                        break
                    
                    if processed >= target_papers:
                        print(f"\n🎯 Target reached: {target_papers:,} papers processed")
                        break
                    
                    try:
                        paper_data = json.loads(line.strip())
                        standardized = self.processor._standardize_arxiv_paper(paper_data)
                        
                        # Quick quality check
                        if await self.processor._quick_quality_check(standardized):
                            current_batch.append(standardized)
                            accepted += 1
                        
                        processed += 1
                        
                        # Process batch
                        if len(current_batch) >= batch_size:
                            await self.process_batch(current_batch)
                            current_batch = []
                        
                        # Progress reporting every 1000 papers
                        if processed % 1000 == 0:
                            await self.report_progress(processed, accepted, start_time)
                        
                        # Save progress every 5000 papers
                        if processed % 5000 == 0:
                            self.save_progress(processed)
                            last_progress_save = datetime.now(timezone.utc)
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"⚠️ Error processing paper {processed}: {e}")
                        continue
                
                # Process remaining batch
                if current_batch and self.running:
                    await self.process_batch(current_batch)
        
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise
        
        finally:
            # Save final progress
            self.save_progress(processed)
            
            # Final report
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            rate = processed / total_time if total_time > 0 else 0
            
            print(f"\n🏁 INGESTION {'COMPLETED' if processed >= target_papers else 'STOPPED'}")
            print("=" * 60)
            print(f"📊 Total processed: {processed:,}")
            print(f"✅ Total accepted: {accepted:,}")
            print(f"📈 Acceptance rate: {(accepted/processed)*100:.1f}%")
            print(f"⏱️ Total time: {total_time/3600:.1f} hours")
            print(f"⚡ Processing rate: {rate:.0f} papers/second")
            print(f"💾 Progress saved to: {self.progress_file}")
            
            if processed >= target_papers:
                print("🎉 SUCCESS: Full 150K arXiv dataset ingested!")
                print("🔍 NWTN can now search across all papers")
            else:
                print("💾 Progress saved - run again to continue ingestion")
    
    async def process_batch(self, batch):
        """Process a batch of papers"""
        for paper in batch:
            try:
                await self.processor._store_paper(paper)
                self.processor.stats["total_accepted"] += 1
                self.processor.stats["domains_covered"].add(paper.get("domain", "unknown"))
            except Exception as e:
                print(f"⚠️ Failed to store paper: {e}")
    
    async def report_progress(self, processed, accepted, start_time):
        """Report processing progress"""
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        acceptance_rate = (accepted / processed) * 100 if processed > 0 else 0
        eta_seconds = (150000 - processed) / rate if rate > 0 else 0
        eta_hours = eta_seconds / 3600
        
        print(f"📊 Progress: {processed:,}/150,000 | "
              f"Accepted: {accepted:,} ({acceptance_rate:.1f}%) | "
              f"Rate: {rate:.0f}/sec | "
              f"ETA: {eta_hours:.1f}h")
    
    def save_progress(self, processed_count):
        """Save current progress to file"""
        progress_data = {
            "processed_count": processed_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress" if self.running else "paused"
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save progress: {e}")
    
    def load_progress(self):
        """Load existing progress from file"""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return data.get("processed_count", 0)
        except Exception as e:
            print(f"⚠️ Failed to load progress: {e}")
        return 0

async def main():
    ingestion = BackgroundIngestion()
    await ingestion.run_ingestion()

if __name__ == "__main__":
    asyncio.run(main())