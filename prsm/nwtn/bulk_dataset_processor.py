#!/usr/bin/env python3
"""
Bulk Dataset Processor for PRSM
===============================

This processes bulk academic datasets for fast ingestion into NWTN.
Much faster than individual paper downloads.
"""

import asyncio
import json
import gzip
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import aiofiles
from datetime import datetime, timezone
import hashlib
import re
import tarfile
import zipfile

import structlog

from prsm.nwtn.production_storage_manager import StorageManager, StorageConfig
from prsm.nwtn.content_quality_filter import ContentQualityFilter, FilterConfig

logger = structlog.get_logger(__name__)


class BulkDatasetProcessor:
    """Process bulk academic datasets for fast ingestion"""
    
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.bulk_data_path = self.storage_path / "bulk_datasets"
        self.bulk_data_path.mkdir(exist_ok=True)
        
        # Initialize storage and quality systems
        self.storage_manager = None
        self.quality_filter = None
        
        # Processing stats
        self.stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "total_stored": 0,
            "start_time": None,
            "domains_covered": set()
        }
    
    async def initialize(self):
        """Initialize bulk processing systems"""
        
        logger.info("🚀 Initializing Bulk Dataset Processor...")
        
        # Initialize storage (simplified config)
        storage_config = StorageConfig(
            external_drive_path="/Volumes/My Passport",
            max_total_storage=100.0
        )
        self.storage_manager = StorageManager(storage_config)
        await self.storage_manager.initialize()
        
        # Initialize quality filter (relaxed for bulk processing)
        filter_config = FilterConfig(
            min_overall_quality=0.2,  # Very relaxed for bulk
            min_analogical_potential=0.1,
            min_content_length=50,
            max_similarity_threshold=0.98
        )
        self.quality_filter = ContentQualityFilter(filter_config)
        await self.quality_filter.initialize()
        
        logger.info("✅ Bulk Dataset Processor ready")
        return True
    
    async def download_arxiv_dataset(self, target_papers: int = 150000) -> bool:
        """Download and process arXiv dataset"""
        
        print("🚀 ARXIV BULK DATASET DOWNLOAD")
        print("=" * 50)
        print("📄 Target: 150,000 papers from arXiv")
        print("💾 Estimated download: 1-2 GB (compressed)")
        print("⚡ Processing time: 30-60 minutes")
        print("🌍 Domains: Physics, CS, Math, Biology")
        print()
        
        # Initialize the processor first
        await self.initialize()
        
        # Check if dataset already exists
        arxiv_file = self.bulk_data_path / "arxiv-metadata-oai-snapshot.json"
        
        if not arxiv_file.exists():
            print("📥 DOWNLOADING ARXIV DATASET...")
            success = await self._download_arxiv_metadata()
            if not success:
                return False
        else:
            print("✅ ArXiv dataset already downloaded")
        
        # Process the dataset
        print("🔄 PROCESSING ARXIV DATASET...")
        await self._process_arxiv_dataset(arxiv_file, target_papers)
        
        return True
    
    async def _download_arxiv_metadata(self) -> bool:
        """Download arXiv metadata file"""
        
        # ArXiv provides free bulk metadata
        arxiv_url = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
        
        print("📝 MANUAL DOWNLOAD REQUIRED:")
        print("1. Go to: https://www.kaggle.com/datasets/Cornell-University/arxiv")
        print("2. Click 'Download' (requires free Kaggle account)")
        print("3. Extract the downloaded file")
        print("4. Copy 'arxiv-metadata-oai-snapshot.json' to:")
        print(f"   {self.bulk_data_path}/")
        print()
        print("⚡ This is a one-time manual step for massive speed gain!")
        print()
        
        # Wait for user to download
        input("Press Enter after you've downloaded and placed the file...")
        
        arxiv_file = self.bulk_data_path / "arxiv-metadata-oai-snapshot.json"
        if arxiv_file.exists():
            print("✅ ArXiv dataset file found!")
            return True
        else:
            print("❌ ArXiv dataset file not found. Please check the path.")
            return False
    
    async def _process_arxiv_dataset(self, arxiv_file: Path, target_papers: int):
        """Process arXiv dataset file"""
        
        print(f"🔄 Processing {arxiv_file.name}...")
        print("📊 Reading arXiv papers...")
        
        self.stats["start_time"] = datetime.now(timezone.utc)
        processed = 0
        accepted = 0
        
        try:
            with open(arxiv_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if processed >= target_papers:
                        break
                    
                    try:
                        # Parse JSON line
                        paper_data = json.loads(line.strip())
                        
                        # Convert to standard format
                        standardized_paper = self._standardize_arxiv_paper(paper_data)
                        
                        # Quick quality check
                        if await self._quick_quality_check(standardized_paper):
                            # Store paper
                            await self._store_paper(standardized_paper)
                            accepted += 1
                            self.stats["total_accepted"] += 1
                            self.stats["domains_covered"].add(standardized_paper.get("domain", "unknown"))
                        
                        processed += 1
                        self.stats["total_processed"] += 1
                        
                        # Progress reporting
                        if processed % 1000 == 0:
                            acceptance_rate = (accepted / processed) * 100
                            elapsed = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
                            rate = processed / elapsed if elapsed > 0 else 0
                            
                            print(f"📊 Processed: {processed:,} | Accepted: {accepted:,} ({acceptance_rate:.1f}%) | Rate: {rate:.0f}/sec")
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing paper {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading arXiv file: {e}")
            return False
        
        # Final statistics
        elapsed = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        final_rate = processed / elapsed if elapsed > 0 else 0
        
        print("\n🎉 BULK PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"📊 Total Processed: {processed:,}")
        print(f"✅ Total Accepted: {accepted:,}")
        print(f"📈 Acceptance Rate: {(accepted/processed)*100:.1f}%")
        print(f"🌍 Domains Covered: {len(self.stats['domains_covered'])}")
        print(f"⏱️ Processing Time: {elapsed/60:.1f} minutes")
        print(f"⚡ Processing Rate: {final_rate:.0f} papers/second")
        print(f"💾 Storage Path: {self.storage_path}")
        print("=" * 50)
        
        return True
    
    def _standardize_arxiv_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert arXiv paper to standard format"""
        
        # Extract categories for domain classification
        categories = paper_data.get("categories", "").split()
        primary_category = categories[0] if categories else "unknown"
        
        # Map arXiv categories to domains
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
        
        # Clean and structure the paper
        standardized = {
            "id": paper_data.get("id", ""),
            "title": paper_data.get("title", "").strip(),
            "abstract": paper_data.get("abstract", "").strip(),
            "authors": self._parse_authors(paper_data.get("authors", "")),
            "categories": categories,
            "keywords": categories,  # Use categories as keywords
            "domain": domain,
            "type": "preprint",
            "source": "arxiv",
            "url": f"https://arxiv.org/abs/{paper_data.get('id', '')}",
            "published_date": paper_data.get("update_date", ""),
            "submitter": paper_data.get("submitter", ""),
            "journal_ref": paper_data.get("journal-ref", ""),
            "license": paper_data.get("license", ""),
            "versions": paper_data.get("versions", [])
        }
        
        return standardized
    
    def _parse_authors(self, authors_string: str) -> List[str]:
        """Parse author string into list"""
        if not authors_string:
            return []
        
        # Split by common separators and clean
        authors = re.split(r',|\sand\s', authors_string)
        return [author.strip() for author in authors if author.strip()]
    
    async def _quick_quality_check(self, paper: Dict[str, Any]) -> bool:
        """Quick quality check for bulk processing"""
        
        # Basic checks
        if not paper.get("title") or not paper.get("abstract"):
            return False
        
        if len(paper["title"]) < 10 or len(paper["abstract"]) < 50:
            return False
        
        # Check for English content (basic heuristic)
        text = paper["title"] + " " + paper["abstract"]
        if len(text.encode('utf-8')) != len(text):
            # Contains non-ASCII characters - might be non-English
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
            if ascii_ratio < 0.8:
                return False
        
        return True
    
    async def _store_paper(self, paper: Dict[str, Any]):
        """Store paper using storage manager"""
        
        try:
            # Store main content
            content_id = paper.get("id", f"paper_{hash(str(paper))}")
            
            result = await self.storage_manager.store_content(
                content_id=content_id,
                content_data=paper,
                content_type="content"
            )
            
            self.stats["total_stored"] += 1
            
        except Exception as e:
            logger.error(f"Failed to store paper {paper.get('id', 'unknown')}: {e}")
    
    async def process_sample_dataset(self, sample_size: int = 1000):
        """Process a small sample for testing"""
        
        print(f"🧪 PROCESSING SAMPLE DATASET ({sample_size:,} papers)")
        print("=" * 50)
        print("⚡ Quick test of bulk processing system")
        print()
        
        # Create sample arXiv data
        sample_papers = []
        for i in range(sample_size):
            paper = {
                "id": f"sample.{i:04d}",
                "title": f"Sample Paper {i}: Advanced Research in Domain {i % 8}",
                "abstract": f"This is a sample abstract for paper {i}. " * 10,
                "authors": f"Author {i}, Co-Author {i+1}",
                "categories": ["cs.AI", "cs.LG", "math.ST", "physics.data-an"][i % 4],
                "update_date": "2024-01-01",
                "submitter": f"submitter{i}@example.com"
            }
            sample_papers.append(paper)
        
        # Process samples
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        for i, paper_data in enumerate(sample_papers):
            standardized = self._standardize_arxiv_paper(paper_data)
            
            if await self._quick_quality_check(standardized):
                await self._store_paper(standardized)
                self.stats["total_accepted"] += 1
                self.stats["domains_covered"].add(standardized.get("domain", "unknown"))
            
            self.stats["total_processed"] += 1
            
            if (i + 1) % 100 == 0:
                print(f"📊 Sample progress: {i+1:,}/{sample_size:,} processed")
        
        elapsed = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        rate = sample_size / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ Sample processing complete!")
        print(f"📊 Processed: {sample_size:,} papers")
        print(f"⚡ Rate: {rate:.0f} papers/second")
        print(f"🌍 Domains: {len(self.stats['domains_covered'])}")
        print(f"⏱️ Time: {elapsed:.1f} seconds")
        
        return True
    
    async def shutdown(self):
        """Shutdown bulk processor"""
        
        if self.storage_manager:
            await self.storage_manager.shutdown()
        
        logger.info("✅ Bulk Dataset Processor shutdown complete")


async def main():
    """Main function for bulk dataset processing"""
    
    print("🚀 BULK DATASET PROCESSOR")
    print("=" * 50)
    print("⚡ Fast bulk processing of academic datasets")
    print("🎯 Target: 150,000 papers in under 1 hour")
    print("=" * 50)
    
    processor = BulkDatasetProcessor()
    
    try:
        # Initialize
        if await processor.initialize():
            print("\n📊 Choose processing option:")
            print("1. 🧪 Process sample dataset (1,000 papers - for testing)")
            print("2. 🚀 Download and process full arXiv dataset (150,000+ papers)")
            print("3. 📁 Process existing downloaded dataset")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                await processor.process_sample_dataset()
            elif choice == "2":
                await processor.download_arxiv_dataset()
            elif choice == "3":
                # Check for existing dataset
                arxiv_file = processor.bulk_data_path / "arxiv-metadata-oai-snapshot.json"
                if arxiv_file.exists():
                    await processor._process_arxiv_dataset(arxiv_file, 150000)
                else:
                    print("❌ No existing dataset found. Choose option 2 to download first.")
            else:
                print("❌ Invalid choice")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
    finally:
        await processor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())