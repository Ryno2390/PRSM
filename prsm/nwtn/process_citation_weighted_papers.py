#!/usr/bin/env python3
"""
Citation-Weighted Papers Processing Pipeline
===========================================

Processes our 21,188 high-impact citation-weighted papers with:
- Hierarchical content extraction preserving natural structure  
- Hierarchy-aware chunking (Title → Section → Subsection → Paragraph)
- High-dimensional embeddings for semantic search
- Content hashing and provenance tracking
- NWTN-ready storage format

Key Features:
- Uses existing citation-weighted database as source
- Quality-focused processing (not speed-focused)
- Section-aware chunking to avoid break-in-the-middle problems
- Complete provenance and attribution tracking
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

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/citation_weighted_processing.log'),
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
    processing_timestamp: float
    content_hash: str
    pipeline_version: str = "2.0.0"
    processing_method: str = "citation_weighted_hierarchical"
    creator_attribution: str = "arXiv.org"
    license_info: str = "arXiv non-exclusive license"

@dataclass 
class ProcessingStats:
    """Statistics for the processing pipeline"""
    papers_attempted: int = 0
    papers_successfully_downloaded: int = 0
    papers_successfully_processed: int = 0
    papers_with_hierarchy: int = 0
    papers_failed: int = 0
    total_chunks_generated: int = 0
    total_embeddings_created: int = 0
    processing_start_time: float = 0
    processing_end_time: float = 0

class CitationWeightedPaperProcessor:
    """Processes citation-weighted papers with hierarchical structure preservation"""
    
    def __init__(self):
        # Database paths
        self.citation_db_path = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/01_RAW_PAPERS/storage.db")
        
        # Storage paths for processed content
        self.storage_base = Path("/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local")
        self.processed_content_path = self.storage_base / "02_PROCESSED_CONTENT"
        self.embeddings_path = self.storage_base / "03_NWTN_READY" / "embeddings"
        self.content_hashes_path = self.storage_base / "03_NWTN_READY" / "content_hashes"
        self.provenance_path = self.storage_base / "03_NWTN_READY" / "provenance_2"
        
        # Create directories
        self.processed_content_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True) 
        self.content_hashes_path.mkdir(parents=True, exist_ok=True)
        self.provenance_path.mkdir(parents=True, exist_ok=True)
        
        # Processing state
        self.session = None
        self.embedding_model = None
        self.stats = ProcessingStats()
        self.processed_papers = set()
        
        # Progress tracking
        self.progress_file = Path("citation_weighted_progress.json")
        self.checkpoint_interval = 100  # Save progress every 100 papers
        
        logger.info("🚀 Citation-Weighted Paper Processor initialized")
    
    async def initialize(self):
        """Initialize HTTP session and embedding model"""
        logger.info("🔧 Initializing processing components...")
        
        # HTTP session optimized for quality
        connector = aiohttp.TCPConnector(
            limit=50,           # Moderate concurrency for quality
            limit_per_host=10,  # Conservative per-host limit
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=15)  # Longer timeouts for quality
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'PRSM-Citation-Weighted-Processor/2.0'}
        )
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2 (384 dimensions)")
        except ImportError:
            logger.error("❌ sentence-transformers not available - embeddings will be skipped")
            self.embedding_model = None
        
        # Load existing progress
        self.load_progress()
        
        logger.info("🎯 Initialization complete")
    
    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
        logger.info("🔒 Session closed")
    
    def load_progress(self):
        """Load processing progress"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.processed_papers = set(progress_data.get('processed_papers', []))
                    stats_data = progress_data.get('stats', {})
                    for key, value in stats_data.items():
                        if hasattr(self.stats, key):
                            setattr(self.stats, key, value)
                logger.info(f"📚 Loaded progress: {len(self.processed_papers)} papers already processed")
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save processing progress"""
        try:
            progress_data = {
                'processed_papers': list(self.processed_papers),
                'stats': asdict(self.stats),
                'last_updated': time.time()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def get_citation_weighted_papers(self) -> List[Dict[str, Any]]:
        """Get all citation-weighted papers from our database"""
        logger.info("📊 Loading citation-weighted papers from database...")
        
        try:
            conn = sqlite3.connect(str(self.citation_db_path))
            cursor = conn.execute("""
                SELECT arxiv_id, title, abstract, authors, domain, 
                       citation_count, quality_score, publish_date
                FROM arxiv_papers 
                WHERE source = 'citation_weighted_2015_2025'
                ORDER BY citation_count DESC, quality_score DESC
            """)
            
            papers = []
            for row in cursor.fetchall():
                papers.append({
                    'arxiv_id': row[0],
                    'title': row[1], 
                    'abstract': row[2],
                    'authors': row[3],
                    'domain': row[4],
                    'citation_count': row[5],
                    'quality_score': row[6],
                    'publish_date': row[7]
                })
            
            conn.close()
            logger.info(f"📈 Loaded {len(papers):,} citation-weighted papers")
            return papers
            
        except Exception as e:
            logger.error(f"❌ Failed to load papers from database: {e}")
            return []
    
    async def download_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF with quality-focused retry logic"""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        
        for attempt in range(3):  # Quality-focused: more retries
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        if len(content) > 5000:  # Quality check: reasonable PDF size
                            return content
                    elif response.status == 404:
                        logger.debug(f"PDF not found: {arxiv_id}")
                        return None
                    else:
                        logger.debug(f"HTTP {response.status} for {arxiv_id}")
                        
            except Exception as e:
                logger.debug(f"Download attempt {attempt + 1} failed for {arxiv_id}: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Progressive backoff
        
        return None
    
    def extract_hierarchical_content(self, pdf_content: bytes, arxiv_id: str) -> Dict[str, Any]:
        """Extract hierarchical content preserving document structure"""
        try:
            import PyPDF2
            import io
            import re
            
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract all text
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                        page_texts.append(page_text)
                except Exception as e:
                    logger.debug(f"Page {page_num} extraction error for {arxiv_id}: {e}")
                    continue
            
            if len(full_text) < 1000:  # Quality check
                logger.debug(f"Insufficient content extracted for {arxiv_id}")
                return {}
            
            # Extract document hierarchy
            hierarchy = self._extract_document_hierarchy(full_text, page_texts)
            
            # Generate hierarchy-aware chunks
            chunks = self._generate_hierarchical_chunks(hierarchy, full_text)
            
            return {
                'full_text': full_text,
                'hierarchy': hierarchy,
                'chunks': chunks,
                'content_length': len(full_text),
                'page_count': len(reader.pages),
                'processing_metadata': {
                    'extraction_method': 'hierarchical_pypdf2',
                    'chunk_count': len(chunks),
                    'section_count': len(hierarchy.get('sections', [])),
                    'has_natural_structure': len(hierarchy.get('sections', [])) > 2,
                    'arxiv_id': arxiv_id
                }
            }
            
        except Exception as e:
            logger.debug(f"Content extraction failed for {arxiv_id}: {e}")
            return {}
    
    def _extract_document_hierarchy(self, full_text: str, page_texts: List[str]) -> Dict[str, Any]:
        """Extract document hierarchy using enhanced pattern recognition"""
        import re
        
        hierarchy = {
            'title': '',
            'abstract': '',
            'sections': [],
            'subsections': [],
            'references_start': -1
        }
        
        lines = full_text.split('\n')
        
        # Enhanced title extraction
        for i, line in enumerate(lines[:25]):  # Check more lines for title
            line = line.strip()
            if (len(line) > 15 and len(line) < 250 and 
                not any(skip in line.lower() for skip in ['arxiv', 'pdf', 'submitted', 'preprint', 'abstract', 'keywords']) and
                (line.isupper() or 
                 re.match(r'^[A-Z][A-Za-z\s:,-]+[A-Za-z]$', line) or
                 sum(1 for c in line if c.isupper()) / max(1, len([c for c in line if c.isalpha()])) > 0.3)):
                hierarchy['title'] = line
                break
        
        # Enhanced abstract extraction with multiple patterns
        abstract_patterns = [
            r'(?i)\babstract\b\s*[-:]?\s*(.*?)(?=\n\s*\n|\n\s*\d+\s+introduction|\n\s*introduction|\n\s*keywords|\n\s*1\.)',
            r'(?i)^\s*abstract\s*$(.*?)(?=^\s*\d|\n\s*introduction)',
            r'(?i)abstract[:\.]?\s*(.*?)(?=\n\s*\n\s*[A-Z]|\n\s*keywords)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
            if match:
                abstract_text = match.group(1).strip()
                if len(abstract_text) > 100 and len(abstract_text) < 3000:  # Reasonable abstract length
                    hierarchy['abstract'] = abstract_text
                    break
        
        # Enhanced section detection with academic paper patterns
        section_patterns = [
            # Numbered sections (1. Introduction, 2.1 Methods, etc.)
            r'(?i)^\s*(\d+(?:\.\d+)*\.?)\s+([A-Z][A-Za-z\s&,-]+?)(?=\s*$|\n)',
            # Roman numerals (I. Introduction, II. Methods)
            r'(?i)^\s*([IVX]+\.?)\s+([A-Z][A-Za-z\s&,-]+?)(?=\s*$|\n)',
            # Letter sections (A. Introduction, B. Methods)
            r'(?i)^\s*([A-Z]\.?)\s+([A-Z][A-Za-z\s&,-]+?)(?=\s*$|\n)',
            # Standard academic sections
            r'(?i)^\s*(Introduction|Related Work|Methodology|Methods?|Experiments?|Results?|Discussion|Conclusions?|References?|Acknowledgments?)(?=\s*$|\n)',
            # ALL CAPS sections
            r'^([A-Z][A-Z\s&,-]{2,40})(?=\s*$|\n)'
        ]
        
        sections_found = []
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if len(line_stripped) < 3 or len(line_stripped) > 150:
                continue
            
            for pattern_idx, pattern in enumerate(section_patterns):
                match = re.match(pattern, line_stripped)
                if match:
                    if pattern_idx < 3:  # Numbered, Roman, or Letter sections
                        section_title = match.group(2) if match.lastindex >= 2 else match.group(1)
                    else:  # Standard or ALL CAPS sections
                        section_title = match.group(1) if match.group(1) else match.group(0)
                    
                    if section_title and len(section_title.strip()) > 2:
                        # Validate it's actually a section (not just random text)
                        if (not any(bad_word in section_title.lower() for bad_word in 
                                   ['figure', 'table', 'equation', 'algorithm', 'page', 'see', 'www', 'http']) and
                            len(section_title.split()) < 10):  # Reasonable section title length
                            
                            section_info = {
                                'title': section_title.strip(),
                                'line_number': line_num,
                                'level': 1,
                                'content_start': line_num + 1,  # Content starts after title
                                'pattern_type': pattern_idx
                            }
                            sections_found.append(section_info)
                            break
        
        # Remove duplicate sections (same title close together)
        unique_sections = []
        for section in sections_found:
            if not any(abs(section['line_number'] - existing['line_number']) < 5 and 
                      section['title'].lower() == existing['title'].lower() 
                      for existing in unique_sections):
                unique_sections.append(section)
        
        # Sort sections by line number and add content boundaries
        unique_sections.sort(key=lambda x: x['line_number'])
        
        for i, section in enumerate(unique_sections):
            if i < len(unique_sections) - 1:
                section['content_end'] = unique_sections[i + 1]['line_number']
            else:
                section['content_end'] = len(lines)
            
            # Extract section content
            section_lines = lines[section['content_start']:section['content_end']]
            section['content'] = '\n'.join(section_lines).strip()
        
        hierarchy['sections'] = unique_sections
        
        # Find references section
        for i, line in enumerate(lines):
            if re.match(r'(?i)^\s*(references?|bibliography)\s*$', line.strip()):
                hierarchy['references_start'] = i
                break
        
        return hierarchy
    
    def _generate_hierarchical_chunks(self, hierarchy: Dict[str, Any], full_text: str) -> List[Dict[str, Any]]:
        """Generate hierarchy-aware chunks optimized for retrieval accuracy"""
        chunks = []
        chunk_id = 0
        
        # Chunk 1: Title and Abstract (always together for context)
        title_abstract_content = ""
        if hierarchy.get('title'):
            title_abstract_content += f"Title: {hierarchy['title']}\n\n"
        if hierarchy.get('abstract'):
            title_abstract_content += f"Abstract: {hierarchy['abstract']}\n\n"
        
        if title_abstract_content:
            chunks.append({
                'id': f"chunk_{chunk_id:04d}",
                'type': 'title_abstract',
                'content': title_abstract_content.strip(),
                'hierarchy_level': 0,
                'section': 'Front Matter',
                'length': len(title_abstract_content),
                'importance': 'high'  # Critical for search
            })
            chunk_id += 1
        
        # Process sections with intelligent chunking
        sections = hierarchy.get('sections', [])
        if sections:
            for section in sections:
                section_content = section.get('content', '').strip()
                if len(section_content) < 200:  # Skip very short sections
                    continue
                
                section_title = section['title']
                
                # Determine section importance
                importance = 'medium'
                if any(key_section in section_title.lower() for key_section in 
                       ['introduction', 'abstract', 'conclusion', 'results', 'discussion', 'summary']):
                    importance = 'high'
                elif any(low_section in section_title.lower() for low_section in 
                        ['references', 'acknowledgments', 'appendix']):
                    importance = 'low'
                
                # Smart chunking based on section size
                if len(section_content) > 3000:  # Large section - intelligent subdivision
                    section_chunks = self._chunk_large_section(section_content, section_title)
                    for i, section_chunk in enumerate(section_chunks):
                        chunks.append({
                            'id': f"chunk_{chunk_id:04d}",
                            'type': 'section_part',
                            'content': section_chunk,
                            'hierarchy_level': 1,
                            'section': section_title,
                            'subsection': f"Part {i+1}",  
                            'length': len(section_chunk),
                            'importance': importance
                        })
                        chunk_id += 1
                else:
                    # Keep smaller sections intact to preserve context
                    chunks.append({
                        'id': f"chunk_{chunk_id:04d}",
                        'type': 'section',
                        'content': f"Section: {section_title}\n\n{section_content}",
                        'hierarchy_level': 1,
                        'section': section_title,
                        'length': len(section_content),
                        'importance': importance
                    })
                    chunk_id += 1
        else:
            # Fallback: Intelligent paragraph-based chunking
            logger.debug("No clear sections found, using paragraph-based chunking")
            paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
            
            current_chunk = ""
            for i, para in enumerate(paragraphs):
                if len(current_chunk + para) > 2000 and current_chunk:
                    # Finalize current chunk
                    chunks.append({
                        'id': f"chunk_{chunk_id:04d}",
                        'type': 'content_block',
                        'content': current_chunk.strip(),
                        'hierarchy_level': 2,
                        'section': f'Content Block {chunk_id + 1}',
                        'length': len(current_chunk),
                        'importance': 'medium'
                    })
                    chunk_id += 1
                    current_chunk = para + '\n\n'
                else:
                    current_chunk += para + '\n\n'
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'id': f"chunk_{chunk_id:04d}",
                    'type': 'content_block',
                    'content': current_chunk.strip(),
                    'hierarchy_level': 2,
                    'section': f'Final Content Block',
                    'length': len(current_chunk),
                    'importance': 'medium'
                })
        
        return chunks
    
    def _chunk_large_section(self, section_content: str, section_title: str) -> List[str]:
        """Intelligently chunk large sections with context preservation"""
        target_size = 2000  # Optimal size for embedding models
        overlap_size = 300  # Sufficient overlap to maintain context
        
        chunks = []
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
        
        current_chunk = f"Section: {section_title}\n\n"
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed target size
            if len(current_chunk + para) > target_size and len(current_chunk) > 800:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with context and overlap
                current_chunk = f"Section: {section_title} (continued)\n\n"
                
                # Add meaningful overlap from previous chunk
                prev_sentences = current_chunk.split('. ')[-3:]  # Last 3 sentences
                if len(prev_sentences) > 1:
                    overlap_text = '. '.join(prev_sentences[-2:])  # Keep 2 sentences
                    if len(overlap_text) < overlap_size:
                        current_chunk += overlap_text + '\n\n'
            
            current_chunk += para + '\n\n'
        
        # Add final chunk if it has meaningful content
        if len(current_chunk.strip()) > 500:  # Minimum meaningful chunk size
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_hierarchical_embeddings(self, content: Dict[str, Any], paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hierarchy-aware embeddings with comprehensive metadata"""
        if not self.embedding_model:
            logger.debug("No embedding model available")
            return {}
        
        try:
            arxiv_id = paper_metadata['arxiv_id']
            embeddings_data = {
                'paper_id': arxiv_id,
                'paper_metadata': {
                    'title': paper_metadata.get('title', ''),
                    'domain': paper_metadata.get('domain', ''), 
                    'citation_count': paper_metadata.get('citation_count', 0),
                    'quality_score': paper_metadata.get('quality_score', 0.0)
                },
                'embeddings': {},
                'content_hashes': {},
                'processing_metadata': {
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'embedding_dimension': 384,
                    'processing_timestamp': time.time(),
                    'total_chunks': 0
                }
            }
            
            # Generate content hash for full document
            full_text = content.get('full_text', '')
            if full_text:
                content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
                embeddings_data['content_hashes']['full_document'] = content_hash
            
            # Process hierarchical chunks
            chunks = content.get('chunks', [])
            embeddings_created = 0
            
            for chunk in chunks:
                chunk_id = chunk['id']
                chunk_content = chunk['content']
                
                if len(chunk_content.strip()) < 100:  # Skip very short chunks
                    continue
                
                try:
                    # Generate embedding
                    chunk_embedding = self.embedding_model.encode(chunk_content)
                    
                    embeddings_data['embeddings'][chunk_id] = {
                        'vector': chunk_embedding.tolist(),
                        'type': chunk['type'],
                        'section': chunk.get('section', 'Unknown'),
                        'hierarchy_level': chunk.get('hierarchy_level', 0),
                        'importance': chunk.get('importance', 'medium'),
                        'length': chunk['length'],
                        'content_preview': chunk_content[:250] + '...' if len(chunk_content) > 250 else chunk_content
                    }
                    
                    # Generate content hash for chunk
                    chunk_hash = hashlib.sha256(chunk_content.encode('utf-8')).hexdigest()
                    embeddings_data['content_hashes'][chunk_id] = chunk_hash
                    
                    embeddings_created += 1
                    
                except Exception as e:
                    logger.warning(f"Embedding generation failed for chunk {chunk_id} in {arxiv_id}: {e}")
                    continue
            
            embeddings_data['processing_metadata']['total_chunks'] = embeddings_created
            
            # Add special high-priority embeddings
            hierarchy = content.get('hierarchy', {})
            
            # Title embedding (highest priority)
            if hierarchy.get('title'):
                try:
                    title_embedding = self.embedding_model.encode(hierarchy['title'])
                    embeddings_data['embeddings']['title_priority'] = {
                        'vector': title_embedding.tolist(),
                        'type': 'title',
                        'section': 'Document Title',
                        'hierarchy_level': -2,  # Highest priority
                        'importance': 'critical',
                        'length': len(hierarchy['title']),
                        'content_preview': hierarchy['title']
                    }
                    embeddings_created += 1
                except Exception as e:
                    logger.warning(f"Title embedding failed for {arxiv_id}: {e}")
            
            # Abstract embedding (very high priority)
            if hierarchy.get('abstract'):
                try:
                    abstract_embedding = self.embedding_model.encode(hierarchy['abstract'])
                    embeddings_data['embeddings']['abstract_priority'] = {
                        'vector': abstract_embedding.tolist(),
                        'type': 'abstract',
                        'section': 'Abstract',
                        'hierarchy_level': -1,  # Second highest priority
                        'importance': 'critical',
                        'length': len(hierarchy['abstract']),
                        'content_preview': hierarchy['abstract'][:400] + '...' if len(hierarchy['abstract']) > 400 else hierarchy['abstract']
                    }
                    embeddings_created += 1
                except Exception as e:
                    logger.warning(f"Abstract embedding failed for {arxiv_id}: {e}")
            
            logger.debug(f"Generated {embeddings_created} embeddings for {arxiv_id}")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Embedding generation failed for {paper_metadata.get('arxiv_id', 'unknown')}: {e}")
            return {}
    
    def save_processed_content(self, arxiv_id: str, content: Dict[str, Any], 
                              embeddings: Dict[str, Any], paper_metadata: Dict[str, Any]):
        """Save all processed content with proper organization"""
        try:
            # Save hierarchical content
            content_file = self.processed_content_path / f"{arxiv_id}_hierarchical.json"
            with open(content_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if embeddings:
                embeddings_file = self.embeddings_path / f"{arxiv_id}_embeddings.json"
                with open(embeddings_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings, f, indent=2)
            
            # Save content hashes
            if embeddings and 'content_hashes' in embeddings:
                hashes_file = self.content_hashes_path / f"{arxiv_id}_hashes.json"
                with open(hashes_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings['content_hashes'], f, indent=2)
            
            # Save provenance record
            provenance = ProvenanceRecord(
                arxiv_id=arxiv_id,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                original_citation_count=paper_metadata.get('citation_count', 0),
                quality_score=paper_metadata.get('quality_score', 0.0),
                domain=paper_metadata.get('domain', ''),
                processing_timestamp=time.time(),
                content_hash=embeddings.get('content_hashes', {}).get('full_document', '')
            )
            
            provenance_file = self.provenance_path / f"{arxiv_id}.json"
            with open(provenance_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(provenance), f, indent=2)
            
            logger.debug(f"Saved all processed content for {arxiv_id}")
            
        except Exception as e:
            logger.error(f"Failed to save processed content for {arxiv_id}: {e}")
    
    async def process_paper(self, paper_metadata: Dict[str, Any]) -> bool:
        """Process a single paper with full pipeline"""
        arxiv_id = paper_metadata['arxiv_id']
        
        try:
            # Check if already processed
            if arxiv_id in self.processed_papers:
                logger.debug(f"Skipping already processed paper: {arxiv_id}")
                return True
            
            self.stats.papers_attempted += 1
            
            # Download PDF
            logger.debug(f"Downloading PDF for {arxiv_id}")
            pdf_content = await self.download_pdf(arxiv_id)
            
            if not pdf_content:
                logger.debug(f"Failed to download PDF: {arxiv_id}")
                self.stats.papers_failed += 1
                return False
            
            self.stats.papers_successfully_downloaded += 1
            
            # Extract hierarchical content
            logger.debug(f"Extracting hierarchical content for {arxiv_id}")
            content = self.extract_hierarchical_content(pdf_content, arxiv_id)
            
            if not content or not content.get('full_text'):
                logger.debug(f"Failed to extract content: {arxiv_id}")
                self.stats.papers_failed += 1
                return False
            
            # Track hierarchy success
            if content.get('processing_metadata', {}).get('has_natural_structure', False):
                self.stats.papers_with_hierarchy += 1
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {arxiv_id}")
            embeddings = self.generate_hierarchical_embeddings(content, paper_metadata)
            
            if embeddings:
                self.stats.total_chunks_generated += len(content.get('chunks', []))
                self.stats.total_embeddings_created += len(embeddings.get('embeddings', {}))
            
            # Save all processed content
            self.save_processed_content(arxiv_id, content, embeddings, paper_metadata)
            
            # Mark as processed
            self.processed_papers.add(arxiv_id)
            self.stats.papers_successfully_processed += 1
            
            logger.debug(f"Successfully processed {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed for {arxiv_id}: {e}")
            self.stats.papers_failed += 1
            return False
    
    async def process_all_citation_weighted_papers(self):
        """Process all citation-weighted papers with progress tracking"""
        logger.info("🚀 Starting citation-weighted papers processing")
        
        # Get all papers to process
        papers = self.get_citation_weighted_papers()
        if not papers:
            logger.error("❌ No papers found to process")
            return
        
        # Filter out already processed papers
        papers_to_process = [p for p in papers if p['arxiv_id'] not in self.processed_papers]
        
        logger.info(f"📊 Processing Status:")
        logger.info(f"   • Total papers: {len(papers):,}")
        logger.info(f"   • Already processed: {len(self.processed_papers):,}")
        logger.info(f"   • Remaining to process: {len(papers_to_process):,}")
        
        if not papers_to_process:
            logger.info("✅ All papers already processed!")
            return
        
        self.stats.processing_start_time = time.time()
        
        # Process papers with progress tracking
        processed_count = 0
        for i, paper in enumerate(papers_to_process):
            try:
                success = await self.process_paper(paper)
                processed_count += 1
                
                # Progress reporting
                if processed_count % 50 == 0:
                    elapsed = time.time() - self.stats.processing_start_time
                    rate = processed_count / elapsed * 60  # papers per minute
                    remaining = len(papers_to_process) - processed_count
                    eta_minutes = remaining / (rate + 0.001)  # Avoid division by zero
                    
                    logger.info(f"📈 Progress: {processed_count}/{len(papers_to_process)} ({processed_count/len(papers_to_process)*100:.1f}%)")
                    logger.info(f"   • Rate: {rate:.1f} papers/min")
                    logger.info(f"   • ETA: {eta_minutes:.0f} minutes")
                    logger.info(f"   • Stats: {self.stats.papers_successfully_processed} processed, {self.stats.papers_failed} failed")
                
                # Checkpoint progress
                if processed_count % self.checkpoint_interval == 0:
                    self.save_progress()
                    logger.info(f"💾 Progress checkpoint saved")
                
                # Brief pause to be respectful to arXiv servers
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Unexpected error processing paper {i}: {e}")
                continue
        
        self.stats.processing_end_time = time.time()
        
        # Final progress save
        self.save_progress()
        
        # Generate final report
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate comprehensive final processing report"""
        total_time = self.stats.processing_end_time - self.stats.processing_start_time
        
        logger.info("=" * 60)
        logger.info("🎉 CITATION-WEIGHTED PAPERS PROCESSING COMPLETE")
        logger.info("=" * 60)
        
        logger.info(f"📊 PROCESSING STATISTICS:")
        logger.info(f"   • Papers Attempted: {self.stats.papers_attempted:,}")
        logger.info(f"   • Successfully Downloaded: {self.stats.papers_successfully_downloaded:,}")
        logger.info(f"   • Successfully Processed: {self.stats.papers_successfully_processed:,}")
        logger.info(f"   • With Natural Hierarchy: {self.stats.papers_with_hierarchy:,}")
        logger.info(f"   • Failed: {self.stats.papers_failed:,}")
        
        if self.stats.papers_attempted > 0:
            success_rate = (self.stats.papers_successfully_processed / self.stats.papers_attempted) * 100
            hierarchy_rate = (self.stats.papers_with_hierarchy / max(1, self.stats.papers_successfully_processed)) * 100
            logger.info(f"   • Success Rate: {success_rate:.1f}%")
            logger.info(f"   • Hierarchy Detection Rate: {hierarchy_rate:.1f}%")
        
        logger.info(f"📚 CONTENT STATISTICS:")
        logger.info(f"   • Total Chunks Generated: {self.stats.total_chunks_generated:,}")
        logger.info(f"   • Total Embeddings Created: {self.stats.total_embeddings_created:,}")
        
        if self.stats.papers_successfully_processed > 0:
            avg_chunks = self.stats.total_chunks_generated / self.stats.papers_successfully_processed
            avg_embeddings = self.stats.total_embeddings_created / self.stats.papers_successfully_processed
            logger.info(f"   • Average Chunks per Paper: {avg_chunks:.1f}")
            logger.info(f"   • Average Embeddings per Paper: {avg_embeddings:.1f}")
        
        logger.info(f"⏱️ TIMING STATISTICS:")
        logger.info(f"   • Total Processing Time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
        
        if total_time > 0 and self.stats.papers_successfully_processed > 0:
            rate = self.stats.papers_successfully_processed / total_time
            logger.info(f"   • Average Rate: {rate:.2f} papers/second ({rate*60:.1f} papers/minute)")
        
        logger.info(f"💾 OUTPUT LOCATIONS:")
        logger.info(f"   • Processed Content: {self.processed_content_path}")
        logger.info(f"   • Embeddings: {self.embeddings_path}")
        logger.info(f"   • Content Hashes: {self.content_hashes_path}")
        logger.info(f"   • Provenance Records: {self.provenance_path}")
        
        logger.info("=" * 60)
        logger.info("🚀 READY FOR NWTN INTEGRATION!")
        logger.info("The citation-weighted papers are now processed with:")
        logger.info("✅ Hierarchical structure preservation")
        logger.info("✅ Intelligent chunking for optimal retrieval")
        logger.info("✅ High-dimensional semantic embeddings")
        logger.info("✅ Complete content hashing and provenance")
        logger.info("=" * 60)


async def main():
    """Main processing function"""
    processor = CitationWeightedPaperProcessor()
    
    try:
        await processor.initialize()
        await processor.process_all_citation_weighted_papers()
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())