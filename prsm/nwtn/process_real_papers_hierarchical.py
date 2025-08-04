#!/usr/bin/env python3
"""
Real Papers Hierarchical Processing Pipeline
===========================================

Processes the 35,769 real arXiv papers with:
- Hierarchical content extraction preserving natural structure
- Hierarchy-aware chunking (Title → Section → Subsection → Paragraph)  
- High-dimensional embeddings for semantic search
- Content hashing and provenance tracking
- NWTN-ready storage format

This processes REAL papers with valid arXiv IDs that can actually be downloaded.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import sys
from urllib.parse import urlparse
import PyPDF2
import io
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/real_papers_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProvenanceRecord:
    """Complete provenance tracking for processed papers"""
    arxiv_id: str
    source_url: str
    original_citation_count: int
    quality_score: float
    domain: str
    categories: str
    processing_timestamp: str
    content_hash: str
    paper_structure: Dict[str, Any]
    chunk_count: int
    embedding_dimensions: int

@dataclass
class HierarchicalChunk:
    """Represents a hierarchy-aware content chunk"""
    chunk_id: str
    arxiv_id: str
    hierarchy_level: str  # 'title', 'section', 'subsection', 'paragraph'
    parent_section: Optional[str]
    content: str
    content_hash: str
    position_in_paper: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class RealPapersHierarchicalProcessor:
    """Process real papers with hierarchical structure preservation"""
    
    def __init__(self):
        # Storage paths
        self.base_path = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local")
        self.db_path = self.base_path / "01_RAW_PAPERS" / "storage.db"
        self.processed_papers_db = self.base_path / "02_PROCESSED_PAPERS" / "hierarchical_papers.db"
        self.processed_papers_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = Path("/Users/ryneschultz/Documents/GitHub/PRSM/real_papers_progress.json")
        self.processed_papers = set()
        self.stats = {
            'papers_attempted': 0,
            'papers_successfully_downloaded': 0,
            'papers_successfully_processed': 0,
            'papers_with_hierarchy': 0,
            'papers_failed': 0,
            'total_chunks_generated': 0,
            'total_embeddings_created': 0,
            'processing_start_time': 0,
            'processing_end_time': 0
        }
        
        # Session for downloads
        self.session = None
        
        # Load progress
        self.load_progress()
        
        logger.info("🚀 Real Papers Hierarchical Processor initialized")
    
    def load_progress(self):
        """Load processing progress from previous runs"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_papers = set(data.get('processed_papers', []))
                    self.stats.update(data.get('stats', {}))
                logger.info(f"📋 Loaded progress: {len(self.processed_papers)} papers already processed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save current processing progress"""
        try:
            progress_data = {
                'processed_papers': list(self.processed_papers),
                'stats': self.stats,
                'last_updated': time.time()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    async def initialize(self):
        """Initialize HTTP session and embedding model"""
        logger.info("🔧 Initializing processing components...")
        
        # HTTP session for PDF downloads (increased for faster processing)
        connector = aiohttp.TCPConnector(limit=15, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=90, connect=20)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; Academic Research Bot; +mailto:research@academic.edu)',
                'Accept': 'application/pdf,*/*'
            }
        )
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"✅ Embedding model loaded: all-MiniLM-L6-v2 ({self.embedding_model.get_sentence_embedding_dimension()} dimensions)")
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        
        # Initialize database
        self.initialize_database()
        
        logger.info("🎯 Initialization complete")
    
    def initialize_database(self):
        """Initialize processed papers database with proper schema"""
        conn = sqlite3.connect(str(self.processed_papers_db))
        
        # Hierarchical chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hierarchical_chunks (
                chunk_id TEXT PRIMARY KEY,
                arxiv_id TEXT NOT NULL,
                hierarchy_level TEXT NOT NULL,
                parent_section TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                position_in_paper INTEGER NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arxiv_id) REFERENCES papers (arxiv_id)
            )
        """)
        
        # Papers metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT,
                categories TEXT,
                domain TEXT,
                citation_count INTEGER,
                quality_score REAL,
                paper_structure TEXT,
                provenance TEXT,
                processing_status TEXT DEFAULT 'pending',
                processed_at TIMESTAMP,
                total_chunks INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON hierarchical_chunks(arxiv_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy ON hierarchical_chunks(hierarchy_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_domain ON papers(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_citations ON papers(citation_count DESC)")
        
        conn.commit()
        conn.close()
        
        logger.info("💾 Database initialized with hierarchical schema")
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    def get_real_papers_to_process(self, limit: int = 85000) -> List[Dict[str, Any]]:
        """Get comprehensive papers from database that need processing"""
        conn = sqlite3.connect(str(self.db_path))
        
        # Get papers that haven't been processed yet from comprehensive collection
        cursor = conn.execute("""
            SELECT arxiv_id, title, abstract, authors, categories, domain, 
                   citation_count, quality_score
            FROM arxiv_papers 
            WHERE source LIKE 'comprehensive_arxiv_%_2015_2025'
            AND arxiv_id NOT IN ({})
            ORDER BY quality_score DESC, citation_count DESC
            LIMIT ?
        """.format(','.join(['?'] * len(self.processed_papers))), 
        list(self.processed_papers) + [limit])
        
        papers = []
        for row in cursor.fetchall():
            papers.append({
                'arxiv_id': row[0],
                'title': row[1],
                'abstract': row[2],
                'authors': row[3],
                'categories': row[4],
                'domain': row[5],
                'citation_count': row[6],
                'quality_score': row[7]
            })
        
        conn.close()
        return papers
    
    async def download_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF content with proper error handling"""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Verify it's actually a PDF
                    if content.startswith(b'%PDF'):
                        return content
                    else:
                        logger.warning(f"Downloaded content for {arxiv_id} is not a valid PDF")
                        return None
                        
                elif response.status == 404:
                    logger.warning(f"PDF not found for {arxiv_id} (404)")
                    return None
                    
                else:
                    logger.warning(f"HTTP {response.status} for {arxiv_id}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {arxiv_id}")
            return None
        except Exception as e:
            logger.warning(f"Error downloading {arxiv_id}: {e}")
            return None
    
    def extract_hierarchical_content(self, pdf_content: bytes, arxiv_id: str) -> Dict[str, Any]:
        """Extract hierarchical content preserving natural document structure"""
        try:
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                page_texts.append(page_text)
                full_text += page_text + "\n\n"
            
            if not full_text.strip():
                logger.warning(f"No text extracted from {arxiv_id}")
                return {}
            
            # Extract document hierarchy
            hierarchy = self._extract_document_hierarchy(full_text, page_texts)
            
            # Generate hierarchy-aware chunks
            chunks = self._generate_hierarchical_chunks(hierarchy, full_text)
            
            return {
                'full_text': full_text,
                'hierarchy': hierarchy,
                'chunks': chunks,
                'page_count': len(pdf_reader.pages),
                'extraction_success': True
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {arxiv_id}: {e}")
            return {}
    
    def _extract_document_hierarchy(self, full_text: str, page_texts: List[str]) -> Dict[str, Any]:
        """Extract document structure and hierarchy"""
        hierarchy = {
            'title': '',
            'abstract': '',
            'sections': [],
            'references': ''
        }
        
        lines = full_text.split('\n')
        
        # Extract title (usually the first substantial line)
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith(('abstract', 'arxiv:', 'submitted')):
                hierarchy['title'] = line
                break
        
        # Extract abstract
        abstract_started = False
        abstract_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith('abstract'):
                abstract_started = True
                continue
            elif abstract_started and (line.lower().startswith(('1.', 'introduction', '1 introduction')) or len(abstract_lines) > 20):
                break
            elif abstract_started and line:
                abstract_lines.append(line)
        
        hierarchy['abstract'] = ' '.join(abstract_lines)
        
        # Extract sections using common patterns
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Section headers (numbered or capitalized)
            if re.match(r'^(\d+\.?\s+|[A-Z][A-Z\s]+)([A-Za-z].*)', line) and len(line) < 100:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'title': line,
                    'content': [],
                    'subsections': []
                }
            elif current_section and line:
                current_section['content'].append(line)
        
        if current_section:
            sections.append(current_section)
        
        hierarchy['sections'] = sections
        
        return hierarchy
    
    def _generate_hierarchical_chunks(self, hierarchy: Dict[str, Any], full_text: str) -> List[Dict[str, Any]]:
        """Generate hierarchy-aware chunks that preserve document structure"""
        chunks = []
        position = 0
        
        # Chunk 1: Title and Abstract (always together)
        if hierarchy.get('title') and hierarchy.get('abstract'):
            title_abstract = f"Title: {hierarchy['title']}\n\nAbstract: {hierarchy['abstract']}"
            chunks.append({
                'hierarchy_level': 'title_abstract',
                'parent_section': None,
                'content': title_abstract,
                'position_in_paper': position
            })
            position += 1
        
        # Chunk 2+: Section-aware chunking
        for section in hierarchy.get('sections', []):
            section_title = section.get('title', '')
            section_content = ' '.join(section.get('content', []))
            
            # If section is short, keep as one chunk
            if len(section_content) < 1000:
                if section_content.strip():
                    chunks.append({
                        'hierarchy_level': 'section',
                        'parent_section': section_title,
                        'content': f"Section: {section_title}\n\n{section_content}",
                        'position_in_paper': position
                    })
                    position += 1
            else:
                # Split longer sections into paragraphs, preserving context
                paragraphs = section_content.split('. ')
                current_chunk = f"Section: {section_title}\n\n"
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    # If adding this paragraph would make chunk too long, save current chunk
                    if len(current_chunk + paragraph) > 800:
                        if len(current_chunk.strip()) > len(f"Section: {section_title}"):
                            chunks.append({
                                'hierarchy_level': 'subsection',
                                'parent_section': section_title,
                                'content': current_chunk.strip(),
                                'position_in_paper': position
                            })
                            position += 1
                        
                        current_chunk = f"Section: {section_title} (continued)\n\n{paragraph}. "
                    else:
                        current_chunk += paragraph + ". "
                
                # Save final chunk if it has content
                if len(current_chunk.strip()) > len(f"Section: {section_title}"):
                    chunks.append({
                        'hierarchy_level': 'subsection',
                        'parent_section': section_title,
                        'content': current_chunk.strip(),
                        'position_in_paper': position
                    })
                    position += 1
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Generate embeddings for hierarchical chunks"""
        hierarchical_chunks = []
        
        for chunk_data in chunks:
            content = chunk_data['content']
            
            # Generate embedding
            embedding = self.embedding_model.encode([content])[0].tolist()
            
            # Create content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            
            # Create chunk ID
            chunk_id = f"{chunk_data.get('arxiv_id', 'unknown')}_{chunk_data['position_in_paper']:03d}_{content_hash}"
            
            hierarchical_chunk = HierarchicalChunk(
                chunk_id=chunk_id,
                arxiv_id=chunk_data.get('arxiv_id', ''),
                hierarchy_level=chunk_data['hierarchy_level'],
                parent_section=chunk_data['parent_section'],
                content=content,
                content_hash=content_hash,
                position_in_paper=chunk_data['position_in_paper'],
                embedding=embedding,
                metadata={
                    'content_length': len(content),
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'processing_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            hierarchical_chunks.append(hierarchical_chunk)
        
        return hierarchical_chunks
    
    def store_processed_paper(self, paper: Dict[str, Any], chunks: List[HierarchicalChunk], 
                            provenance: ProvenanceRecord):
        """Store processed paper and its hierarchical chunks in database"""
        conn = sqlite3.connect(str(self.processed_papers_db))
        
        try:
            # Store paper metadata
            conn.execute("""
                INSERT OR REPLACE INTO papers 
                (arxiv_id, title, abstract, authors, categories, domain, citation_count, 
                 quality_score, paper_structure, provenance, processing_status, 
                 processed_at, total_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper['arxiv_id'],
                paper['title'],
                paper['abstract'],
                paper['authors'],
                paper['categories'],
                paper['domain'],
                paper['citation_count'],
                paper['quality_score'],
                json.dumps(provenance.paper_structure),
                json.dumps(asdict(provenance)),
                'completed',
                datetime.now(timezone.utc).isoformat(),
                len(chunks)
            ))
            
            # Store hierarchical chunks
            for chunk in chunks:
                # Convert embedding to binary for storage
                embedding_bytes = json.dumps(chunk.embedding).encode('utf-8')
                
                conn.execute("""
                    INSERT OR REPLACE INTO hierarchical_chunks
                    (chunk_id, arxiv_id, hierarchy_level, parent_section, content,
                     content_hash, position_in_paper, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.arxiv_id,
                    chunk.hierarchy_level,
                    chunk.parent_section,
                    chunk.content,
                    chunk.content_hash,
                    chunk.position_in_paper,
                    embedding_bytes,
                    json.dumps(chunk.metadata)
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing paper {paper['arxiv_id']}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def process_paper(self, paper: Dict[str, Any]) -> bool:
        """Process a single paper with hierarchical extraction"""
        arxiv_id = paper['arxiv_id']
        
        try:
            # Download PDF
            pdf_content = await self.download_pdf(arxiv_id)
            if not pdf_content:
                return False
            
            self.stats['papers_successfully_downloaded'] += 1
            
            # Extract hierarchical content
            extraction_result = self.extract_hierarchical_content(pdf_content, arxiv_id)
            if not extraction_result.get('extraction_success'):
                return False
            
            # Prepare chunks for embedding
            chunks_data = extraction_result['chunks']
            for chunk in chunks_data:
                chunk['arxiv_id'] = arxiv_id
            
            # Generate embeddings
            hierarchical_chunks = self.generate_embeddings(chunks_data)
            
            # Create provenance record
            provenance = ProvenanceRecord(
                arxiv_id=arxiv_id,
                source_url=f"https://arxiv.org/pdf/{arxiv_id}",
                original_citation_count=paper['citation_count'],
                quality_score=paper['quality_score'],
                domain=paper['domain'],
                categories=paper['categories'],
                processing_timestamp=datetime.now(timezone.utc).isoformat(),
                content_hash=hashlib.sha256(extraction_result['full_text'].encode('utf-8')).hexdigest(),
                paper_structure=extraction_result['hierarchy'],
                chunk_count=len(hierarchical_chunks),
                embedding_dimensions=384
            )
            
            # Store in database
            self.store_processed_paper(paper, hierarchical_chunks, provenance)
            
            # Update statistics
            self.stats['papers_successfully_processed'] += 1
            self.stats['total_chunks_generated'] += len(hierarchical_chunks)
            self.stats['total_embeddings_created'] += len(hierarchical_chunks)
            
            if len(extraction_result['hierarchy'].get('sections', [])) > 0:
                self.stats['papers_with_hierarchy'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            return False
    
    async def process_real_papers(self, max_papers: int = 5000):
        """Process real papers with hierarchical extraction"""
        logger.info("🚀 Starting real papers hierarchical processing")
        self.stats['processing_start_time'] = time.time()
        
        # Get papers to process
        papers = self.get_real_papers_to_process(max_papers)
        logger.info(f"📊 Processing Status:")
        logger.info(f"   • Total papers available: {len(papers):,}")
        logger.info(f"   • Already processed: {len(self.processed_papers):,}")
        logger.info(f"   • Remaining to process: {len(papers):,}")
        
        if not papers:
            logger.info("✅ All papers already processed!")
            return
        
        processed_count = 0
        failed_count = 0
        
        for i, paper in enumerate(papers):
            arxiv_id = paper['arxiv_id']
            self.stats['papers_attempted'] += 1
            
            logger.info(f"📄 Processing {arxiv_id} ({i+1}/{len(papers)}) - Citations: {paper['citation_count']}")
            
            success = await self.process_paper(paper)
            
            if success:
                self.processed_papers.add(arxiv_id)
                processed_count += 1
                logger.info(f"✅ Successfully processed {arxiv_id}")
            else:
                failed_count += 1
                self.stats['papers_failed'] += 1
                logger.warning(f"❌ Failed to process {arxiv_id}")
            
            # Progress reporting and checkpointing
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - self.stats['processing_start_time']) * 60
                eta_minutes = ((len(papers) - i - 1) / rate) if rate > 0 else 0
                
                logger.info(f"📈 Progress: {i+1}/{len(papers)} ({(i+1)/len(papers)*100:.1f}%)")
                logger.info(f"   • Rate: {rate:.1f} papers/min")
                logger.info(f"   • ETA: {eta_minutes:.0f} minutes")
                logger.info(f"   • Stats: {processed_count} processed, {failed_count} failed")
                
                # Save progress checkpoint
                self.save_progress()
                logger.info("💾 Progress checkpoint saved")
            
            # Reduced pause for faster processing (403 blocks cleared)
            await asyncio.sleep(1.0)
        
        self.stats['processing_end_time'] = time.time()
        self.save_progress()
        
        # Final statistics
        total_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        logger.info("=" * 60)
        logger.info("🎉 REAL PAPERS PROCESSING COMPLETE!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   • Papers processed: {processed_count:,}")
        logger.info(f"   • Papers failed: {failed_count:,}")
        logger.info(f"   • Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
        logger.info(f"   • Total chunks: {self.stats['total_chunks_generated']:,}")
        logger.info(f"   • Total embeddings: {self.stats['total_embeddings_created']:,}")
        logger.info(f"   • Processing time: {total_time/60:.1f} minutes")
        logger.info("=" * 60)
    
    async def process_real_papers_parallel(self, max_papers: int = 100000, batch_size: int = 10):
        """Process real papers with parallel batch processing for much faster throughput"""
        logger.info("🚀 Starting PARALLEL real papers hierarchical processing")
        self.stats['processing_start_time'] = time.time()
        
        # Get papers to process
        papers = self.get_real_papers_to_process(max_papers)
        logger.info(f"📊 Parallel Processing Status:")
        logger.info(f"   • Total papers available: {len(papers):,}")
        logger.info(f"   • Already processed: {len(self.processed_papers):,}")
        logger.info(f"   • Remaining to process: {len(papers):,}")
        logger.info(f"   • Batch size: {batch_size} concurrent papers")
        
        if not papers:
            logger.info("✅ All papers already processed!")
            return
        
        processed_count = 0
        failed_count = 0
        
        # Process papers in parallel batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_tasks = []
            
            # Create concurrent tasks for this batch
            for paper in batch:
                arxiv_id = paper['arxiv_id']
                self.stats['papers_attempted'] += 1
                task = self.process_paper(paper)
                batch_tasks.append((paper, task))
            
            # Execute batch concurrently
            logger.info(f"🔄 Processing batch {i//batch_size + 1} of {len(batch)} papers concurrently...")
            results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
            
            # Process results
            for (paper, _), success in zip(batch_tasks, results):
                arxiv_id = paper['arxiv_id']
                
                if isinstance(success, Exception):
                    logger.warning(f"❌ Exception processing {arxiv_id}: {success}")
                    failed_count += 1
                    self.stats['papers_failed'] += 1
                elif success:
                    self.processed_papers.add(arxiv_id)
                    processed_count += 1
                    logger.info(f"✅ Successfully processed {arxiv_id}")
                else:
                    failed_count += 1
                    self.stats['papers_failed'] += 1
                    logger.warning(f"❌ Failed to process {arxiv_id}")
            
            # Progress reporting and checkpointing
            if (i + batch_size) % (batch_size * 5) == 0:  # Every 5 batches
                rate = (i + batch_size) / (time.time() - self.stats['processing_start_time']) * 60
                eta_minutes = ((len(papers) - i - batch_size) / rate) if rate > 0 else 0
                
                logger.info(f"📈 Parallel Progress: {i+batch_size}/{len(papers)} ({(i+batch_size)/len(papers)*100:.1f}%)")
                logger.info(f"   • Rate: {rate:.1f} papers/min ({rate*60:.0f} papers/hour)")
                logger.info(f"   • ETA: {eta_minutes:.0f} minutes ({eta_minutes/60:.1f} hours)")
                logger.info(f"   • Stats: {processed_count} processed, {failed_count} failed")
                
                # Save progress checkpoint
                self.save_progress()
                logger.info("💾 Progress checkpoint saved")
            
            # Brief pause between batches
            await asyncio.sleep(0.2)
        
        self.stats['processing_end_time'] = time.time()
        self.save_progress()
        
        # Final statistics
        total_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        logger.info("=" * 60)
        logger.info("🎉 PARALLEL PROCESSING COMPLETE!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   • Papers processed: {processed_count:,}")
        logger.info(f"   • Papers failed: {failed_count:,}")
        logger.info(f"   • Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
        logger.info(f"   • Total chunks: {self.stats['total_chunks_generated']:,}")
        logger.info(f"   • Total embeddings: {self.stats['total_embeddings_created']:,}")
        logger.info(f"   • Processing time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        logger.info(f"   • Average rate: {processed_count/(total_time/3600):.0f} papers/hour")
        logger.info("=" * 60)

async def main():
    """Main processing pipeline"""
    processor = RealPapersHierarchicalProcessor()
    
    try:
        await processor.initialize()
        await processor.process_real_papers_parallel(max_papers=100000, batch_size=10)  # Parallel processing for 100K goal
        
    except KeyboardInterrupt:
        logger.info("⚠️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())