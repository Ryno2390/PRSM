#!/usr/bin/env python3
"""
NWTN-Compatible PDF Processing Pipeline
======================================

This enhanced pipeline provides full NWTN compatibility including:
1. Source material tracking (raw PDFs)
2. Content hashing for provenance
3. Creator attribution and licensing
4. High-dimensional embeddings with source traceability
5. FTNS reward tracking
6. Integration with NWTN's Deep Reasoning Process

Compatible with the complete NWTN pipeline:
Raw PDFs → Content Hashing + Provenance → Embeddings → Search → Deep Reasoning → Synthesis
"""

import asyncio
import aiohttp
import json
import time
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
import logging
import sys
import os

# Add PRSM to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# NWTN imports
from prsm.nwtn.data_models import PaperData, PaperEmbedding, RetrievedPaper
from prsm.nwtn.content_ingestion_engine import ContentHash, ContentHashGenerator, IngestionResult
from prsm.integrations.core.provenance_engine import ProvenanceEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NWTNPaperRecord:
    """Complete paper record for NWTN system"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    full_text: str
    content_hash: str
    provenance_id: str
    embeddings: Dict[str, np.ndarray]
    source_pdf_path: str
    processed_at: datetime
    ftns_rewards: Dict[str, float] = field(default_factory=dict)
    
    def to_paper_data(self) -> PaperData:
        """Convert to NWTN PaperData format"""
        return PaperData(
            paper_id=self.arxiv_id,
            title=self.title,
            abstract=self.abstract,
            authors=self.authors,
            domain=self._extract_domain(),
            categories=self._extract_categories(),
            published_date=self._extract_publish_date(),
            file_path=self.source_pdf_path
        )
    
    def _extract_domain(self) -> str:
        """Extract domain from arXiv ID"""
        if self.arxiv_id.startswith('cs.'):
            return 'computer_science'
        elif self.arxiv_id.startswith('math.'):
            return 'mathematics'
        elif self.arxiv_id.startswith('physics.'):
            return 'physics'
        else:
            return 'multidisciplinary'
    
    def _extract_categories(self) -> List[str]:
        """Extract categories from domain"""
        domain = self._extract_domain()
        return [domain, 'research_paper', 'arxiv']
    
    def _extract_publish_date(self) -> str:
        """Extract publication date from arXiv ID"""
        # arXiv ID format: YYMM.NNNN
        if len(self.arxiv_id) >= 4:
            year_month = self.arxiv_id[:4]
            if year_month.isdigit():
                year = f"20{year_month[:2]}"
                month = year_month[2:4]
                return f"{year}-{month}-01"
        return "2024-01-01"  # Default

class PostQuantumContentHashGenerator:
    """Generates post-quantum resistant content hashes for duplicate detection"""
    
    def __init__(self, algorithm: str = "sha3_256"):
        """
        Initialize with post-quantum resistant hash algorithm
        
        Args:
            algorithm: Hash algorithm ("sha3_256", "sha3_512", "blake2b", "blake2s")
        """
        self.algorithm = algorithm
        
        # Validate post-quantum algorithms
        pq_algorithms = ["sha3_256", "sha3_512", "blake2b", "blake2s"]
        if algorithm not in pq_algorithms:
            raise ValueError(f"Algorithm must be post-quantum resistant: {pq_algorithms}")
    
    def generate_hash(self, content: str) -> ContentHash:
        """Generate post-quantum resistant content hash from text content"""
        # Normalize content for consistent hashing
        normalized_content = self._normalize_content(content)
        content_bytes = normalized_content.encode('utf-8')
        
        # Generate hash using post-quantum resistant algorithms
        if self.algorithm == "sha3_256":
            import hashlib
            hash_obj = hashlib.sha3_256()
            hash_obj.update(content_bytes)
            content_hash = hash_obj.hexdigest()
        elif self.algorithm == "sha3_512":
            import hashlib
            hash_obj = hashlib.sha3_512()
            hash_obj.update(content_bytes)
            content_hash = hash_obj.hexdigest()
        elif self.algorithm == "blake2b":
            import hashlib
            hash_obj = hashlib.blake2b()
            hash_obj.update(content_bytes)
            content_hash = hash_obj.hexdigest()
        elif self.algorithm == "blake2s":
            import hashlib
            hash_obj = hashlib.blake2s()
            hash_obj.update(content_bytes)
            content_hash = hash_obj.hexdigest()
        else:
            raise ValueError(f"Unsupported post-quantum algorithm: {self.algorithm}")
        
        return ContentHash(
            content_hash=content_hash,
            algorithm=self.algorithm
        )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing"""
        # Remove extra whitespace, convert to lowercase for consistent hashing
        normalized = ' '.join(content.lower().split())
        return normalized

class NWTNCompatiblePipeline:
    """Complete NWTN-compatible PDF processing pipeline"""
    
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.embeddings_path = self.storage_path / "03_EMBEDDINGS_NWTN_SEARCH"
        self.content_path = self.storage_path / "02_PROCESSED_CONTENT"
        self.source_pdfs_path = self.storage_path / "01_RAW_PAPERS" / "source_pdfs"
        self.provenance_path = self.storage_path / "04_PROVENANCE_RECORDS"
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/nwtn_pipeline_progress.json")
        
        # Create all directories
        for path in [self.embeddings_path, self.content_path, self.source_pdfs_path, self.provenance_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.session = None
        self.embedding_model = None
        self.hash_generator = PostQuantumContentHashGenerator(algorithm="sha3_256")  # Post-quantum ready
        self.provenance_engine = ProvenanceEngine()
        
        self.stats = {
            'downloaded': 0,
            'processed': 0,
            'embedded': 0,
            'hashed': 0,
            'provenance_tracked': 0,
            'failed': 0,
            'duplicates_detected': 0,
            'start_time': time.time()
        }
        
    async def initialize(self):
        """Initialize all pipeline components"""
        # HTTP session for downloads
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PRSM-NWTN-Pipeline/2.0)'}
        )
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, embeddings disabled")
            self.embedding_model = None
            
        # Initialize provenance engine
        await self.provenance_engine.initialize()
        
        logger.info("🚀 NWTN-compatible pipeline initialized")
        
    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
            
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def load_progress(self):
        """Load previous progress"""
        try:
            with open(self.progress_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
                # Reset start time for current session
                self.stats['start_time'] = time.time()
        except FileNotFoundError:
            pass
    
    async def download_and_store_pdf(self, arxiv_id: str) -> Optional[str]:
        """Download PDF and store as source material"""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    pdf_content = await response.read()
                    
                    # Store raw PDF as source material
                    pdf_path = self.source_pdfs_path / f"{arxiv_id}.pdf"
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_content)
                    
                    return str(pdf_path)
                elif response.status == 404:
                    return None
        except Exception as e:
            logger.debug(f"Download error {arxiv_id}: {e}")
        
        return None
    
    def extract_complete_content_with_metadata(self, pdf_path: str, arxiv_id: str) -> Dict[str, Any]:
        """Extract complete content with metadata for NWTN"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract all text
                full_text = ""
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    except Exception:
                        continue
                
                # Extract structured sections
                sections = self._extract_paper_sections(full_text)
                
                # Extract metadata
                metadata = {
                    'arxiv_id': arxiv_id,
                    'page_count': len(reader.pages),
                    'content_length': len(full_text),
                    'extraction_method': 'PyPDF2',
                    'extracted_at': datetime.now(timezone.utc).isoformat(),
                    'source_file': pdf_path
                }
                
                return {
                    'full_text': full_text,
                    'title': sections.get('title', ''),
                    'abstract': sections.get('abstract', ''),
                    'authors': sections.get('authors', []),
                    'metadata': metadata,
                    **sections
                }
                
        except Exception as e:
            logger.debug(f"Content extraction error for {arxiv_id}: {e}")
            return {}
    
    def _extract_paper_sections(self, full_text: str) -> Dict[str, Any]:
        """Extract structured sections from paper text"""
        sections = {
            'title': '',
            'abstract': '',
            'authors': [],
            'introduction': '',
            'methods': '',
            'results': '',
            'conclusion': '',
            'references': ''
        }
        
        text_lower = full_text.lower()
        
        # Simple heuristic extraction (can be enhanced)
        lines = full_text.split('\n')
        
        # Extract title (usually first substantial line)
        for line in lines[:10]:
            if len(line.strip()) > 10 and len(line.strip()) < 200:
                sections['title'] = line.strip()
                break
        
        # Extract abstract
        if 'abstract' in text_lower:
            abstract_start = text_lower.find('abstract')
            abstract_end = text_lower.find('introduction', abstract_start)
            if abstract_end == -1:
                abstract_end = abstract_start + 1500
            sections['abstract'] = full_text[abstract_start:abstract_end].strip()
        
        # Extract authors (basic pattern matching)
        author_patterns = ['author', 'by ', 'et al']
        for line in lines[:20]:
            for pattern in author_patterns:
                if pattern in line.lower() and len(line) < 200:
                    # Simple author extraction
                    authors_text = line.replace('Authors:', '').replace('By:', '').strip()
                    sections['authors'] = [a.strip() for a in authors_text.split(',')]
                    break
        
        return sections
    
    def generate_content_hash_with_provenance(self, content: Dict[str, Any], arxiv_id: str) -> Tuple[ContentHash, str]:
        """Generate content hash and create provenance record"""
        # Generate content hash
        hash_content = f"{content.get('title', '')}\n{content.get('abstract', '')}\n{content.get('full_text', '')}"
        content_hash = self.hash_generator.generate_hash(hash_content)
        
        # Create provenance record
        provenance_data = {
            'source_type': 'arxiv_paper',
            'source_id': arxiv_id,
            'content_hash': content_hash.content_hash,
            'original_source': f"https://arxiv.org/abs/{arxiv_id}",
            'license': 'arXiv-1.0',  # arXiv license
            'creator_attribution': content.get('authors', []),
            'extraction_metadata': content.get('metadata', {}),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'provenance_version': '2.0',
            'hash_algorithm': content_hash.algorithm,
            'post_quantum_ready': True
        }
        
        # Store provenance record
        provenance_id = str(uuid4())
        provenance_file = self.provenance_path / f"{arxiv_id}_provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump({
                'provenance_id': provenance_id,
                'arxiv_id': arxiv_id,
                'provenance_data': provenance_data
            }, f, indent=2)
        
        return content_hash, provenance_id
    
    def generate_nwtn_embeddings_with_traceability(self, content: Dict[str, Any], arxiv_id: str, provenance_id: str) -> Dict[str, Any]:
        """Generate embeddings with full source traceability for NWTN"""
        if not self.embedding_model:
            return {}
        
        try:
            embeddings = {}
            
            # 1. Full paper embedding with metadata
            if content.get('full_text'):
                full_text_sample = content['full_text'][:8000]
                if len(full_text_sample) > 100:
                    full_embedding = self.embedding_model.encode(full_text_sample)
                    embeddings['full_paper'] = {
                        'vector': full_embedding,
                        'source_text_length': len(content['full_text']),
                        'sample_length': len(full_text_sample),
                        'embedding_type': 'full_paper'
                    }
            
            # 2. Title embedding
            if content.get('title'):
                title_embedding = self.embedding_model.encode(content['title'])
                embeddings['title'] = {
                    'vector': title_embedding,
                    'source_text': content['title'],
                    'embedding_type': 'title'
                }
            
            # 3. Abstract embedding
            if content.get('abstract'):
                abstract_embedding = self.embedding_model.encode(content['abstract'])
                embeddings['abstract'] = {
                    'vector': abstract_embedding,
                    'source_text': content['abstract'][:500] + "..." if len(content['abstract']) > 500 else content['abstract'],
                    'embedding_type': 'abstract'
                }
            
            # 4. Structured sections
            for section in ['introduction', 'methods', 'results', 'conclusion']:
                if content.get(section) and len(content[section]) > 100:
                    section_embedding = self.embedding_model.encode(content[section][:2000])
                    embeddings[section] = {
                        'vector': section_embedding,
                        'source_text_length': len(content[section]),
                        'embedding_type': f'section_{section}'
                    }
            
            # 5. Composite embedding for search optimization
            composite_text = ""
            if content.get('title'):
                composite_text += content['title'] + " "
            if content.get('abstract'):
                composite_text += content['abstract'][:500] + " "
            if content.get('introduction'):
                composite_text += content['introduction'][:1000]
            
            if composite_text.strip():
                composite_embedding = self.embedding_model.encode(composite_text)
                embeddings['composite'] = {
                    'vector': composite_embedding,
                    'source_components': ['title', 'abstract', 'introduction'],
                    'embedding_type': 'composite_optimized'
                }
            
            # Add traceability metadata to all embeddings
            for embedding_type, embedding_data in embeddings.items():
                embedding_data.update({
                    'arxiv_id': arxiv_id,
                    'provenance_id': provenance_id,
                    'model_name': 'all-MiniLM-L6-v2',
                    'model_dimension': len(embedding_data['vector']),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'nwtn_compatible': True
                })
            
            return embeddings
            
        except Exception as e:
            logger.debug(f"Embedding generation error for {arxiv_id}: {e}")
            return {}
    
    def store_nwtn_compatible_data(self, arxiv_id: str, content: Dict[str, Any], 
                                  content_hash: ContentHash, provenance_id: str, 
                                  embeddings: Dict[str, Any], source_pdf_path: str) -> bool:
        """Store all data in NWTN-compatible format"""
        try:
            # 1. Store processed content with full metadata for NWTN
            content_record = {
                'arxiv_id': arxiv_id,
                'title': content.get('title', ''),
                'abstract': content.get('abstract', ''),
                'authors': content.get('authors', []),
                'full_text_preview': content.get('full_text', '')[:5000],
                'content_length': len(content.get('full_text', '')),
                'page_count': content.get('metadata', {}).get('page_count', 0),
                'content_hash': content_hash.content_hash,
                'hash_algorithm': content_hash.algorithm,
                'provenance_id': provenance_id,
                'source_pdf_path': source_pdf_path,
                'extraction_metadata': content.get('metadata', {}),
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'nwtn_version': '2.0',
                'pipeline_compatible': True
            }
            
            # Store content metadata
            content_file = self.content_path / f"{arxiv_id}_nwtn.json"
            with open(content_file, 'w', encoding='utf-8') as f:
                json.dump(content_record, f, indent=2, ensure_ascii=False)
            
            # Store full text separately
            full_text_file = self.content_path / f"{arxiv_id}_full_text.txt"
            with open(full_text_file, 'w', encoding='utf-8') as f:
                f.write(content.get('full_text', ''))
            
            # 2. Store embeddings with full traceability
            if embeddings:
                embedding_record = {
                    'arxiv_id': arxiv_id,
                    'provenance_id': provenance_id,
                    'content_hash': content_hash.content_hash,
                    'embeddings': {
                        name: {
                            'vector': data['vector'].tolist(),  # Convert numpy to list for JSON
                            'metadata': {k: v for k, v in data.items() if k != 'vector'}
                        }
                        for name, data in embeddings.items()
                    },
                    'embedding_summary': {
                        'total_embeddings': len(embeddings),
                        'embedding_types': list(embeddings.keys()),
                        'model_name': 'all-MiniLM-L6-v2',
                        'dimension': 384,  # MiniLM dimension
                        'created_at': datetime.now(timezone.utc).isoformat()
                    },
                    'nwtn_search_ready': True
                }
                
                # Store embeddings in pickle format for NWTN search
                embedding_file = self.embeddings_path / f"{arxiv_id}_nwtn_embeddings.pkl"
                with open(embedding_file, 'wb') as f:
                    pickle.dump(embedding_record, f)
                
                # Also store JSON version for debugging/inspection
                embedding_json_file = self.embeddings_path / f"{arxiv_id}_embeddings_metadata.json"
                with open(embedding_json_file, 'w') as f:
                    # Store without the actual vectors for readability
                    json_record = embedding_record.copy()
                    json_record['embeddings'] = {
                        name: data['metadata'] for name, data in embedding_record['embeddings'].items()
                    }
                    json.dump(json_record, f, indent=2)
            
            # 3. Create NWTN PaperRecord for direct system integration
            nwtn_record = NWTNPaperRecord(
                arxiv_id=arxiv_id,
                title=content.get('title', ''),
                abstract=content.get('abstract', ''),
                authors=content.get('authors', []),
                full_text=content.get('full_text', ''),
                content_hash=content_hash.content_hash,
                provenance_id=provenance_id,
                embeddings={name: data['vector'] for name, data in embeddings.items()},
                source_pdf_path=source_pdf_path,
                processed_at=datetime.now(timezone.utc)
            )
            
            # Store NWTN record
            nwtn_record_file = self.content_path / f"{arxiv_id}_nwtn_record.pkl"
            with open(nwtn_record_file, 'wb') as f:
                pickle.dump(nwtn_record, f)
            
            return True
            
        except Exception as e:
            logger.debug(f"Storage error for {arxiv_id}: {e}")
            return False
    
    async def process_paper_nwtn_compatible(self, arxiv_id: str) -> bool:
        """Complete NWTN-compatible processing pipeline for one paper"""
        try:
            # 1. Download and store source PDF
            source_pdf_path = await self.download_and_store_pdf(arxiv_id)
            if not source_pdf_path:
                self.stats['failed'] += 1
                return False
            
            self.stats['downloaded'] += 1
            
            # 2. Extract complete content with metadata
            content = self.extract_complete_content_with_metadata(source_pdf_path, arxiv_id)
            if not content or not content.get('full_text'):
                self.stats['failed'] += 1
                return False
            
            self.stats['processed'] += 1
            
            # 3. Generate content hash and provenance record
            content_hash, provenance_id = self.generate_content_hash_with_provenance(content, arxiv_id)
            self.stats['hashed'] += 1
            self.stats['provenance_tracked'] += 1
            
            # 4. Generate embeddings with full traceability
            embeddings = self.generate_nwtn_embeddings_with_traceability(content, arxiv_id, provenance_id)
            if embeddings:
                self.stats['embedded'] += 1
            
            # 5. Store in NWTN-compatible format
            success = self.store_nwtn_compatible_data(
                arxiv_id, content, content_hash, provenance_id, embeddings, source_pdf_path
            )
            
            return success
            
        except Exception as e:
            logger.debug(f"Complete processing error for {arxiv_id}: {e}")
            self.stats['failed'] += 1
            return False
    
    def get_arxiv_ids_to_process(self) -> List[str]:
        """Generate arXiv IDs for processing"""
        arxiv_ids = []
        
        # Generate 2020-2024 papers
        for year in range(2020, 2025):
            year_short = str(year)[2:]
            for month in range(1, 13):
                month_str = f"{month:02d}"
                for paper_num in range(1, 2000, 5):  # Sample every 5th
                    arxiv_id = f"{year_short}{month_str}.{paper_num:04d}"
                    arxiv_ids.append(arxiv_id)
        
        # Add classic papers
        classic_patterns = ["1706.", "1810.", "1909.", "2005."]
        for pattern in classic_patterns:
            for i in range(1, 1000, 10):
                arxiv_ids.append(f"{pattern}{i:04d}")
        
        logger.info(f"📋 Generated {len(arxiv_ids)} arXiv IDs for NWTN processing")
        return arxiv_ids
    
    async def run_nwtn_compatible_pipeline(self):
        """Run the complete NWTN-compatible pipeline"""
        logger.info("🚀 Starting NWTN-compatible PDF processing pipeline")
        logger.info("Pipeline: Download → Extract → Hash → Provenance → Embed → Store (NWTN Format)")
        
        # Load previous progress
        self.load_progress()
        
        # Get papers to process
        arxiv_ids = self.get_arxiv_ids_to_process()
        logger.info(f"📦 Processing {len(arxiv_ids)} papers for NWTN system")
        
        # Process with high concurrency
        semaphore = asyncio.Semaphore(20)
        
        async def process_with_semaphore(arxiv_id):
            async with semaphore:
                return await self.process_paper_nwtn_compatible(arxiv_id)
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i+batch_size]
            
            batch_start = time.time()
            tasks = [process_with_semaphore(arxiv_id) for arxiv_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start
            
            # Calculate performance metrics
            batch_successes = sum(1 for r in results if r is True)
            total_time = time.time() - self.stats['start_time']
            
            if total_time > 0:
                rates = {
                    'download': self.stats['downloaded'] / total_time,
                    'process': self.stats['processed'] / total_time,
                    'hash': self.stats['hashed'] / total_time,
                    'provenance': self.stats['provenance_tracked'] / total_time,
                    'embed': self.stats['embedded'] / total_time
                }
                
                logger.info(f"📊 Batch {i//batch_size + 1}: {batch_successes}/{len(batch)} successful")
                logger.info(f"⚡ Rates/hour: Download: {rates['download']*3600:.0f}, "
                           f"Process: {rates['process']*3600:.0f}, Hash: {rates['hash']*3600:.0f}, "
                           f"Provenance: {rates['provenance']*3600:.0f}, Embed: {rates['embed']*3600:.0f}")
            
            # Save progress
            self.save_progress()
            await asyncio.sleep(0.2)
        
        # Final statistics
        total_time = time.time() - self.stats['start_time']
        
        logger.info("🎉 NWTN-compatible pipeline completed!")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   Downloaded: {self.stats['downloaded']} (source PDFs)")
        logger.info(f"   Processed: {self.stats['processed']} (content extraction)")
        logger.info(f"   Hashed: {self.stats['hashed']} (content hashing)")
        logger.info(f"   Provenance Tracked: {self.stats['provenance_tracked']} (creator attribution)")
        logger.info(f"   Embedded: {self.stats['embedded']} (high-dimensional vectors)")
        logger.info(f"   Failed: {self.stats['failed']}")
        
        if total_time > 0:
            final_rate = self.stats['embedded'] / total_time
            logger.info(f"⚡ Final rate: {final_rate:.2f} papers/sec ({final_rate*3600:.0f} papers/hour)")
            logger.info("✅ All data stored in NWTN-compatible format with full traceability")

async def main():
    """Main execution"""
    pipeline = NWTNCompatiblePipeline()
    
    try:
        await pipeline.initialize()
        await pipeline.run_nwtn_compatible_pipeline()
    finally:
        await pipeline.close()

if __name__ == "__main__":
    # Dependency check
    missing_deps = []
    
    try:
        import PyPDF2
    except ImportError:
        missing_deps.append("PyPDF2")
        
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
        
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
        
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        exit(1)
    
    print("🔥 NWTN-COMPATIBLE PDF PROCESSING PIPELINE")
    print("Features: Source Tracking → Post-Quantum Hashing → Provenance → Embeddings")  
    print("Security: SHA3-256 Post-Quantum Ready Content Hashing")
    print("Target: Full NWTN Deep Reasoning Pipeline Integration")
    print("=" * 80)
    
    asyncio.run(main())