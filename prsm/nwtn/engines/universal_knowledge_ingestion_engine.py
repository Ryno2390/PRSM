#!/usr/bin/env python3
"""
Universal Knowledge Ingestion Engine for NWTN
=============================================

This module implements the Phase 1.1 Multi-Format Content Processing system that transforms
NWTN from processing only academic PDFs to a Universal Knowledge Ingestion Engine capable
of processing all enterprise data types for breakthrough thinking across diverse knowledge sources.

Architecture:
- UniversalIngestionEngine: Main orchestrator handling all content types
- FormatProcessors: Specialized processors for different content formats
- UnifiedContentModel: Content normalization and entity resolution
- EnterpriseIntegration: Security, access control, and audit trails
- UnifiedKnowledgeGraph: Comprehensive knowledge representation system

Based on NWTN Roadmap Phase 1.1 - Multi-Format Content Processing (Very High Priority)
Expected Impact: 10-100x data scale enabling breakthrough thinking through data diversity
"""

import asyncio
import time
import os
import io
import json
import xml.etree.ElementTree as ET
import csv
import hashlib
import mimetypes
import gc
import psutil
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import structlog

# Optional imports for different content types
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    import docx
    from bs4 import BeautifulSoup
    import markdown
    DOCUMENT_PROCESSORS_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSORS_AVAILABLE = False

try:
    import nbformat
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# ZIM file processing
try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
    ZIM_AVAILABLE = True
except ImportError:
    try:
        # Alternative: zimply library
        import zimply
        ZIM_AVAILABLE = True
        ZIM_LIBRARY = "zimply"
    except ImportError:
        ZIM_AVAILABLE = False
        ZIM_LIBRARY = None
else:
    ZIM_LIBRARY = "libzim"

logger = structlog.get_logger(__name__)

# Processing configuration
MAX_FILENAME_LENGTH = 100
PROCESSING_TIMEOUT = 300  # 5 minutes per file
MEMORY_THRESHOLD = 0.85  # Stop processing if memory usage exceeds 85%
BATCH_SIZE = 50  # Process files in batches of 50

def generate_safe_filename(source_path: str, content_hash: str = None) -> Tuple[str, str]:
    """Generate safe filename that avoids filesystem limitations
    
    Returns:
        Tuple[str, str]: (short_filename, original_filename)
    """
    original_filename = os.path.basename(source_path)
    
    # Generate content hash if not provided
    if content_hash is None:
        content_hash = hashlib.sha256(source_path.encode()).hexdigest()[:12]
    
    # Extract paper ID if it exists (arxiv format)
    paper_id_match = re.search(r'(\d{4}\.\d{5}v?\d*)', original_filename)
    if paper_id_match:
        paper_id = paper_id_match.group(1)
        # Use paper ID + hash for safe filename
        short_filename = f"{paper_id}_{content_hash}"
    else:
        # Use hash-based filename for non-arxiv papers
        clean_name = re.sub(r'[^\w\-_\.]', '_', original_filename)[:50]
        short_filename = f"{clean_name}_{content_hash}"
    
    # Ensure safe length
    if len(short_filename) > MAX_FILENAME_LENGTH:
        short_filename = f"doc_{content_hash}"
    
    return short_filename, original_filename

def check_memory_usage() -> bool:
    """Check if memory usage is below threshold"""
    try:
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent < MEMORY_THRESHOLD
    except:
        return True  # Continue processing if can't check memory

async def safe_file_operation(operation, *args, timeout=PROCESSING_TIMEOUT, **kwargs):
    """Execute file operation with timeout and error handling"""
    try:
        return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"File operation timed out after {timeout} seconds")
    except Exception as e:
        logger.warning("File operation failed", error=str(e), operation=operation.__name__)
        raise

class ContentFormat(Enum):
    """Supported content formats"""
    # Documents
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    
    # Structured Data
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    
    # Communications
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    CONFLUENCE = "confluence"
    
    # Technical Content
    JUPYTER = "jupyter"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    CODE = "code"
    
    # Multimedia
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    
    # Enterprise Systems
    SHAREPOINT = "sharepoint"
    NOTION = "notion"
    SALESFORCE = "salesforce"
    JIRA = "jira"
    
    # Knowledge Bases & Archives
    ZIM = "zim"  # Wikipedia/Knowledge archive format

class SecurityClassification(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"

class ProcessingStatus(Enum):
    """Content processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ContentMetadata:
    """Metadata for processed content"""
    content_id: str = field(default_factory=lambda: str(uuid4()))
    source_path: str = ""
    content_format: ContentFormat = ContentFormat.TXT
    title: str = ""
    authors: List[str] = field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    size_bytes: int = 0
    language: str = "en"
    security_classification: SecurityClassification = SecurityClassification.INTERNAL
    tags: List[str] = field(default_factory=list)
    source_system: str = "unknown"
    access_permissions: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    processing_time: float = 0.0
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Enhanced filename handling
    short_filename: str = ""
    original_filename: str = ""
    content_hash: str = ""

@dataclass
class ProcessedContent:
    """Container for processed content with metadata"""
    metadata: ContentMetadata
    raw_content: str = ""
    structured_content: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None

@dataclass
class IngestionResult:
    """Result of content ingestion process"""
    ingestion_id: str = field(default_factory=lambda: str(uuid4()))
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    total_processing_time: float = 0.0
    processed_content: List[ProcessedContent] = field(default_factory=list)
    error_summary: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class DocumentProcessor:
    """Processes document formats (PDF, DOCX, TXT, HTML, MD)"""
    
    def __init__(self):
        self.supported_formats = {
            ContentFormat.PDF,
            ContentFormat.DOCX,
            ContentFormat.TXT,
            ContentFormat.HTML,
            ContentFormat.MARKDOWN
        }
    
    async def process_content(self, content_path: str, content_format: ContentFormat) -> ProcessedContent:
        """Process document content with enhanced error handling and safe filenames"""
        
        start_time = time.time()
        
        try:
            # Check memory usage before processing
            if not check_memory_usage():
                logger.warning("Memory usage too high, deferring processing", path=content_path)
                raise ResourceWarning("Memory usage exceeded threshold")
            
            # Generate safe filename and content hash
            file_size = os.path.getsize(content_path) if os.path.exists(content_path) else 0
            content_hash = hashlib.sha256(content_path.encode()).hexdigest()[:12]
            short_filename, original_filename = generate_safe_filename(content_path, content_hash)
            
            # Create metadata with enhanced filename handling
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                size_bytes=file_size,
                short_filename=short_filename,
                original_filename=original_filename,
                content_hash=content_hash
            )
            
            # Process based on format with timeout protection
            if content_format == ContentFormat.PDF:
                raw_content = await safe_file_operation(self._process_pdf, content_path)
            elif content_format == ContentFormat.DOCX:
                raw_content = await safe_file_operation(self._process_docx, content_path)
            elif content_format == ContentFormat.TXT:
                raw_content = await safe_file_operation(self._process_txt, content_path)
            elif content_format == ContentFormat.HTML:
                raw_content = await safe_file_operation(self._process_html, content_path)
            elif content_format == ContentFormat.MARKDOWN:
                raw_content = await safe_file_operation(self._process_markdown, content_path)
            else:
                raise ValueError(f"Unsupported document format: {content_format}")
            
            # Validate processed content
            if not raw_content or len(raw_content.strip()) < 10:
                logger.warning("Processed content appears empty or too short", 
                              path=content_path, 
                              content_length=len(raw_content) if raw_content else 0)
                raise ValueError("Processed content is empty or too short")
            
            # Extract basic entities and structure
            structured_content = await self._extract_document_structure(raw_content, content_format)
            entities = await self._extract_entities(raw_content)
            
            # Update metadata
            metadata.title = structured_content.get('title', os.path.basename(content_path))
            metadata.processing_time = time.time() - start_time
            metadata.quality_score = self._assess_content_quality(raw_content, structured_content)
            
            return ProcessedContent(
                metadata=metadata,
                raw_content=raw_content,
                structured_content=structured_content,
                extracted_entities=entities,
                processing_status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Document processing failed", 
                        content_path=content_path, 
                        format=content_format.value, 
                        error=str(e))
            
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                processing_time=processing_time
            )
            
            return ProcessedContent(
                metadata=metadata,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _process_pdf(self, pdf_path: str) -> str:
        """Process PDF document with enhanced error handling"""
        
        if not DOCUMENT_PROCESSORS_AVAILABLE:
            raise ImportError("PDF processing requires PyPDF2")
        
        # Check file exists and is readable
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not os.access(pdf_path, os.R_OK):
            raise PermissionError(f"Cannot read PDF file: {pdf_path}")
        
        # Check file size (skip very large files that might cause memory issues)
        file_size = os.path.getsize(pdf_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            logger.warning("PDF file too large, skipping", path=pdf_path, size_mb=file_size / (1024*1024))
            raise ValueError(f"PDF file too large: {file_size / (1024*1024):.1f}MB")
        
        try:
            with open(pdf_path, 'rb') as file:
                # Try to create reader first to catch corrupted PDFs early
                try:
                    reader = PdfReader(file)
                except Exception as e:
                    logger.warning("Corrupted PDF detected", path=pdf_path, error=str(e))
                    raise ValueError(f"Corrupted PDF: {str(e)}")
                
                # Check if PDF has pages
                if len(reader.pages) == 0:
                    raise ValueError("PDF has no pages")
                
                text_content = []
                pages_processed = 0
                max_pages = min(len(reader.pages), 1000)  # Limit to 1000 pages
                
                for i, page in enumerate(reader.pages[:max_pages]):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text.strip())
                            pages_processed += 1
                        
                        # Memory check every 10 pages
                        if i % 10 == 0 and not check_memory_usage():
                            logger.warning("Memory usage too high during PDF processing", 
                                         path=pdf_path, pages_processed=pages_processed)
                            break
                            
                    except Exception as page_error:
                        logger.debug("Failed to process PDF page", 
                                   path=pdf_path, page=i, error=str(page_error))
                        continue
                
                if not text_content:
                    raise ValueError("No text could be extracted from PDF")
                
                final_text = '\n\n'.join(text_content)
                logger.debug("PDF processed successfully", 
                           path=pdf_path, 
                           pages=pages_processed, 
                           text_length=len(final_text))
                
                return final_text
                
        except (FileNotFoundError, PermissionError, ValueError):
            # Re-raise these as they're expected failures
            raise
        except Exception as e:
            logger.error("Unexpected PDF processing error", path=pdf_path, error=str(e))
            # Don't fall back to basic info - raise the error so it can be handled at higher level
            raise ValueError(f"PDF processing failed: {str(e)}")
    
    async def _process_docx(self, docx_path: str) -> str:
        """Process DOCX document"""
        
        if not DOCUMENT_PROCESSORS_AVAILABLE:
            raise ImportError("DOCX processing requires python-docx")
        
        try:
            doc = docx.Document(docx_path)
            paragraphs = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            logger.warning("DOCX processing failed, using fallback", error=str(e))
            return f"DOCX document: {os.path.basename(docx_path)} (processing failed: {str(e)})"
    
    async def _process_txt(self, txt_path: str) -> str:
        """Process TXT document"""
        
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
                
        except Exception as e:
            logger.warning("TXT processing failed", error=str(e))
            raise
    
    async def _process_html(self, html_path: str) -> str:
        """Process HTML document"""
        
        if not DOCUMENT_PROCESSORS_AVAILABLE:
            # Fallback: basic text extraction
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
                html_content = file.read()
                # Simple tag removal
                clean_text = re.sub(r'<[^>]+>', '', html_content)
                return re.sub(r'\s+', ' ', clean_text).strip()
        
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return '\n'.join(chunk for chunk in chunks if chunk)
                
        except Exception as e:
            logger.warning("HTML processing failed, using fallback", error=str(e))
            return f"HTML document: {os.path.basename(html_path)} (processing failed: {str(e)})"
    
    async def _process_markdown(self, md_path: str) -> str:
        """Process Markdown document"""
        
        try:
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as file:
                md_content = file.read()
            
            if not DOCUMENT_PROCESSORS_AVAILABLE:
                # Basic markdown processing - remove common markdown syntax
                text = re.sub(r'#{1,6}\s+', '', md_content)  # Headers
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
                text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
                text = re.sub(r'`(.*?)`', r'\1', text)  # Code
                text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
                return text
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
            
        except Exception as e:
            logger.warning("Markdown processing failed", error=str(e))
            raise
    
    async def _extract_document_structure(self, content: str, content_format: ContentFormat) -> Dict[str, Any]:
        """Extract document structure and metadata"""
        
        structure = {
            'word_count': len(content.split()),
            'char_count': len(content),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'sections': [],
            'title': '',
            'abstract': '',
            'keywords': []
        }
        
        lines = content.split('\n')
        
        # Extract title (first non-empty line or line with title indicators)
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and (len(line) < 200):  # Reasonable title length
                if any(indicator in line.lower() for indicator in ['title:', 'subject:', '#']):
                    structure['title'] = re.sub(r'^(title:|subject:|#+)\s*', '', line, flags=re.IGNORECASE)
                    break
                elif not structure['title'] and len(line.split()) > 2:
                    structure['title'] = line
        
        # Extract sections (lines starting with #, numbers, or ALL CAPS)
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(\d+\.?\d*\s+.+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]{10,})$'  # ALL CAPS headings
        ]
        
        for line in lines:
            line = line.strip()
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_title = match.group(1).strip()
                    if len(section_title) < 200:  # Reasonable section length
                        structure['sections'].append(section_title)
                    break
        
        # Extract keywords (simple approach)
        # Look for common academic/business keywords
        keyword_indicators = ['keywords:', 'tags:', 'topics:', 'subject areas:']
        for line in lines:
            line_lower = line.lower().strip()
            for indicator in keyword_indicators:
                if line_lower.startswith(indicator):
                    keywords_text = line[len(indicator):].strip()
                    keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text) if kw.strip()]
                    structure['keywords'] = keywords[:10]  # Limit to 10 keywords
                    break
        
        return structure
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from document content"""
        
        entities = []
        
        # Simple entity extraction patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            'number': r'\b\d+\.?\d*%?\b',
            'organization': r'\b[A-Z][a-zA-Z\s]+ (Inc|LLC|Corp|Company|Organization|University|Institute)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:20]:  # Limit entities per type
                entities.append({
                    'type': entity_type,
                    'value': match.strip(),
                    'confidence': 0.8  # Simple confidence score
                })
        
        return entities
    
    def _assess_content_quality(self, raw_content: str, structured_content: Dict[str, Any]) -> float:
        """Assess content quality score (0.0-1.0)"""
        
        quality_factors = []
        
        # Length factor
        content_length = len(raw_content)
        if content_length > 1000:
            quality_factors.append(1.0)
        elif content_length > 500:
            quality_factors.append(0.8)
        elif content_length > 100:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Structure factor
        if structured_content.get('title'):
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.6)
        
        if structured_content.get('sections'):
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)
        
        # Content richness factor
        word_count = structured_content.get('word_count', 0)
        if word_count > 500:
            quality_factors.append(1.0)
        elif word_count > 100:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.4)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

class StructuredDataProcessor:
    """Processes structured data formats (Excel, CSV, JSON, XML)"""
    
    def __init__(self):
        self.supported_formats = {
            ContentFormat.EXCEL,
            ContentFormat.CSV,
            ContentFormat.JSON,
            ContentFormat.XML
        }
    
    async def process_content(self, content_path: str, content_format: ContentFormat) -> ProcessedContent:
        """Process structured data content"""
        
        start_time = time.time()
        
        try:
            # Create metadata
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                size_bytes=os.path.getsize(content_path) if os.path.exists(content_path) else 0
            )
            
            # Process based on format
            if content_format == ContentFormat.EXCEL:
                raw_content, structured_content = await self._process_excel(content_path)
            elif content_format == ContentFormat.CSV:
                raw_content, structured_content = await self._process_csv(content_path)
            elif content_format == ContentFormat.JSON:
                raw_content, structured_content = await self._process_json(content_path)
            elif content_format == ContentFormat.XML:
                raw_content, structured_content = await self._process_xml(content_path)
            else:
                raise ValueError(f"Unsupported structured data format: {content_format}")
            
            # Extract entities from structured data
            entities = await self._extract_structured_entities(structured_content)
            
            # Update metadata
            metadata.title = structured_content.get('title', os.path.basename(content_path))
            metadata.processing_time = time.time() - start_time
            metadata.quality_score = self._assess_structured_quality(structured_content)
            
            return ProcessedContent(
                metadata=metadata,
                raw_content=raw_content,
                structured_content=structured_content,
                extracted_entities=entities,
                processing_status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Structured data processing failed", 
                        content_path=content_path, 
                        format=content_format.value, 
                        error=str(e))
            
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                processing_time=processing_time
            )
            
            return ProcessedContent(
                metadata=metadata,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _process_excel(self, excel_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process Excel file"""
        
        if not PANDAS_AVAILABLE:
            raise ImportError("Excel processing requires pandas")
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(excel_path)
            sheets_data = {}
            raw_content_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convert to dict
                sheet_data = {
                    'name': sheet_name,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data': df.to_dict('records')[:100],  # Limit to first 100 rows
                    'summary': {
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'data_types': df.dtypes.to_dict()
                    }
                }
                sheets_data[sheet_name] = sheet_data
                
                # Create raw content representation
                raw_content_parts.append(f"Sheet: {sheet_name}")
                raw_content_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                raw_content_parts.append(f"Rows: {len(df)}")
                
                # Add sample data
                if not df.empty:
                    sample_data = df.head().to_string(index=False)
                    raw_content_parts.append(f"Sample data:\n{sample_data}")
                
                raw_content_parts.append("")  # Empty line between sheets
            
            structured_content = {
                'type': 'excel_workbook',
                'sheet_count': len(excel_file.sheet_names),
                'sheets': sheets_data,
                'title': os.path.basename(excel_path)
            }
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("Excel processing failed, using fallback", error=str(e))
            return f"Excel file: {os.path.basename(excel_path)} (processing failed: {str(e)})", {}
    
    async def _process_csv(self, csv_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process CSV file"""
        
        try:
            if PANDAS_AVAILABLE:
                # Use pandas for better CSV handling
                df = pd.read_csv(csv_path)
                
                structured_content = {
                    'type': 'csv_table',
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data': df.to_dict('records')[:100],  # Limit to first 100 rows
                    'summary': {
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'data_types': df.dtypes.to_dict()
                    },
                    'title': os.path.basename(csv_path)
                }
                
                # Create raw content
                raw_content_parts = [
                    f"CSV File: {os.path.basename(csv_path)}",
                    f"Columns: {', '.join(df.columns.tolist())}",
                    f"Rows: {len(df)}",
                    "",
                    "Sample data:",
                    df.head().to_string(index=False)
                ]
                
                raw_content = '\n'.join(raw_content_parts)
                
            else:
                # Fallback: use built-in csv module
                data_rows = []
                headers = []
                
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
                    reader = csv.reader(file)
                    headers = next(reader, [])
                    
                    for i, row in enumerate(reader):
                        if i < 100:  # Limit to first 100 rows
                            data_rows.append(row)
                        else:
                            break
                
                structured_content = {
                    'type': 'csv_table',
                    'columns': headers,
                    'data': [dict(zip(headers, row)) for row in data_rows],
                    'summary': {
                        'row_count': len(data_rows),
                        'column_count': len(headers)
                    },
                    'title': os.path.basename(csv_path)
                }
                
                # Create raw content
                raw_content_parts = [
                    f"CSV File: {os.path.basename(csv_path)}",
                    f"Columns: {', '.join(headers)}",
                    f"Rows: {len(data_rows)}",
                    "",
                    "Sample data:"
                ]
                
                # Add sample rows
                for i, row in enumerate(data_rows[:5]):
                    raw_content_parts.append(f"Row {i+1}: {', '.join(str(cell) for cell in row)}")
                
                raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("CSV processing failed", error=str(e))
            raise
    
    async def _process_json(self, json_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process JSON file"""
        
        try:
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as file:
                data = json.load(file)
            
            # Analyze JSON structure
            def analyze_structure(obj, path=""):
                if isinstance(obj, dict):
                    return {
                        'type': 'object',
                        'keys': list(obj.keys())[:20],  # Limit keys
                        'size': len(obj),
                        'nested_types': {k: type(v).__name__ for k, v in list(obj.items())[:20]}
                    }
                elif isinstance(obj, list):
                    return {
                        'type': 'array',
                        'size': len(obj),
                        'item_types': [type(item).__name__ for item in obj[:10]]
                    }
                else:
                    return {
                        'type': type(obj).__name__,
                        'value': str(obj)[:100] if len(str(obj)) > 100 else str(obj)
                    }
            
            structure_analysis = analyze_structure(data)
            
            structured_content = {
                'type': 'json_data',
                'structure': structure_analysis,
                'data': data if len(str(data)) < 10000 else str(data)[:10000] + "... (truncated)",
                'title': os.path.basename(json_path)
            }
            
            # Create raw content representation
            raw_content_parts = [
                f"JSON File: {os.path.basename(json_path)}",
                f"Structure: {structure_analysis['type']}",
                ""
            ]
            
            if structure_analysis['type'] == 'object':
                raw_content_parts.append(f"Keys: {', '.join(structure_analysis['keys'])}")
                raw_content_parts.append(f"Object size: {structure_analysis['size']} items")
            elif structure_analysis['type'] == 'array':
                raw_content_parts.append(f"Array size: {structure_analysis['size']} items")
                raw_content_parts.append(f"Item types: {', '.join(set(structure_analysis['item_types']))}")
            
            raw_content_parts.append("")
            raw_content_parts.append("Content preview:")
            raw_content_parts.append(json.dumps(data, indent=2)[:1000] + ("..." if len(str(data)) > 1000 else ""))
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("JSON processing failed", error=str(e))
            raise
    
    async def _process_xml(self, xml_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process XML file"""
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            def xml_to_dict(element):
                result = {}
                
                # Add attributes
                if element.attrib:
                    result['@attributes'] = element.attrib
                
                # Add text content
                if element.text and element.text.strip():
                    result['text'] = element.text.strip()
                
                # Add child elements
                children = {}
                for child in element:
                    if child.tag not in children:
                        children[child.tag] = []
                    children[child.tag].append(xml_to_dict(child))
                
                # Simplify single-item lists
                for key, value in children.items():
                    if len(value) == 1:
                        result[key] = value[0]
                    else:
                        result[key] = value
                
                return result
            
            xml_data = xml_to_dict(root)
            
            structured_content = {
                'type': 'xml_document',
                'root_tag': root.tag,
                'namespace': root.tag.split('}')[0][1:] if '}' in root.tag else None,
                'data': xml_data,
                'title': os.path.basename(xml_path)
            }
            
            # Create raw content representation
            raw_content_parts = [
                f"XML File: {os.path.basename(xml_path)}",
                f"Root element: {root.tag}",
                f"Child elements: {len(list(root))}",
                ""
            ]
            
            # Add XML structure overview
            def describe_element(elem, level=0):
                indent = "  " * level
                desc = f"{indent}{elem.tag}"
                if elem.attrib:
                    desc += f" (attributes: {', '.join(elem.attrib.keys())})"
                if elem.text and elem.text.strip():
                    text_preview = elem.text.strip()[:50]
                    desc += f" - {text_preview}{'...' if len(elem.text.strip()) > 50 else ''}"
                return desc
            
            raw_content_parts.append("XML structure:")
            raw_content_parts.append(describe_element(root))
            
            for child in list(root)[:10]:  # Limit to first 10 children
                raw_content_parts.append(describe_element(child, 1))
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("XML processing failed", error=str(e))
            raise
    
    async def _extract_structured_entities(self, structured_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from structured data"""
        
        entities = []
        
        # Extract different types of structured entities
        if structured_content.get('type') == 'excel_workbook':
            for sheet_name, sheet_data in structured_content.get('sheets', {}).items():
                entities.append({
                    'type': 'excel_sheet',
                    'value': sheet_name,
                    'metadata': {
                        'rows': sheet_data.get('summary', {}).get('row_count', 0),
                        'columns': sheet_data.get('summary', {}).get('column_count', 0)
                    },
                    'confidence': 1.0
                })
                
                # Extract column names as entities
                for column in sheet_data.get('columns', [])[:10]:
                    entities.append({
                        'type': 'data_column',
                        'value': column,
                        'metadata': {'sheet': sheet_name},
                        'confidence': 0.9
                    })
        
        elif structured_content.get('type') == 'csv_table':
            for column in structured_content.get('columns', [])[:10]:
                entities.append({
                    'type': 'data_column',
                    'value': column,
                    'confidence': 0.9
                })
        
        elif structured_content.get('type') == 'json_data':
            structure = structured_content.get('structure', {})
            if structure.get('type') == 'object':
                for key in structure.get('keys', [])[:10]:
                    entities.append({
                        'type': 'json_key',
                        'value': key,
                        'confidence': 0.9
                    })
        
        elif structured_content.get('type') == 'xml_document':
            entities.append({
                'type': 'xml_root',
                'value': structured_content.get('root_tag', ''),
                'confidence': 1.0
            })
        
        return entities
    
    def _assess_structured_quality(self, structured_content: Dict[str, Any]) -> float:
        """Assess structured data quality score (0.0-1.0)"""
        
        quality_factors = []
        
        data_type = structured_content.get('type', '')
        
        if 'excel' in data_type:
            sheets = structured_content.get('sheets', {})
            if sheets:
                quality_factors.append(1.0)
                # Check if sheets have meaningful data
                total_rows = sum(sheet.get('summary', {}).get('row_count', 0) for sheet in sheets.values())
                if total_rows > 10:
                    quality_factors.append(0.9)
                else:
                    quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
        
        elif 'csv' in data_type:
            row_count = structured_content.get('summary', {}).get('row_count', 0)
            column_count = structured_content.get('summary', {}).get('column_count', 0)
            
            if row_count > 10 and column_count > 2:
                quality_factors.append(1.0)
            elif row_count > 1 and column_count > 1:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
        
        elif 'json' in data_type:
            structure = structured_content.get('structure', {})
            if structure.get('type') == 'object' and structure.get('size', 0) > 1:
                quality_factors.append(0.9)
            elif structure.get('type') == 'array' and structure.get('size', 0) > 1:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
        
        elif 'xml' in data_type:
            if structured_content.get('data'):
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

class TechnicalProcessor:
    """Processes technical content (Jupyter notebooks, code)"""
    
    def __init__(self):
        self.supported_formats = {
            ContentFormat.JUPYTER,
            ContentFormat.PYTHON,
            ContentFormat.JAVASCRIPT,
            ContentFormat.CODE
        }
    
    async def process_content(self, content_path: str, content_format: ContentFormat) -> ProcessedContent:
        """Process technical content"""
        
        start_time = time.time()
        
        try:
            # Create metadata
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                size_bytes=os.path.getsize(content_path) if os.path.exists(content_path) else 0
            )
            
            # Process based on format
            if content_format == ContentFormat.JUPYTER:
                raw_content, structured_content = await self._process_jupyter(content_path)
            elif content_format in [ContentFormat.PYTHON, ContentFormat.JAVASCRIPT, ContentFormat.CODE]:
                raw_content, structured_content = await self._process_code(content_path, content_format)
            else:
                raise ValueError(f"Unsupported technical format: {content_format}")
            
            # Extract technical entities
            entities = await self._extract_technical_entities(structured_content, raw_content)
            
            # Update metadata
            metadata.title = structured_content.get('title', os.path.basename(content_path))
            metadata.processing_time = time.time() - start_time
            metadata.quality_score = self._assess_technical_quality(structured_content)
            
            return ProcessedContent(
                metadata=metadata,
                raw_content=raw_content,
                structured_content=structured_content,
                extracted_entities=entities,
                processing_status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Technical content processing failed", 
                        content_path=content_path, 
                        format=content_format.value, 
                        error=str(e))
            
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                processing_time=processing_time
            )
            
            return ProcessedContent(
                metadata=metadata,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _process_jupyter(self, jupyter_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process Jupyter notebook"""
        
        try:
            if JUPYTER_AVAILABLE:
                with open(jupyter_path, 'r', encoding='utf-8') as file:
                    notebook = nbformat.read(file, as_version=4)
                
                cells_data = []
                raw_content_parts = [f"Jupyter Notebook: {os.path.basename(jupyter_path)}", ""]
                
                for i, cell in enumerate(notebook.cells):
                    cell_data = {
                        'cell_type': cell.cell_type,
                        'source': cell.source,
                        'execution_count': getattr(cell, 'execution_count', None),
                        'outputs': []
                    }
                    
                    # Add cell content to raw content
                    raw_content_parts.append(f"Cell {i+1} ({cell.cell_type}):")
                    raw_content_parts.append(cell.source)
                    raw_content_parts.append("")
                    
                    # Process outputs for code cells
                    if hasattr(cell, 'outputs'):
                        for output in cell.outputs:
                            output_data = {
                                'output_type': output.output_type,
                                'text': getattr(output, 'text', ''),
                                'data': getattr(output, 'data', {})
                            }
                            cell_data['outputs'].append(output_data)
                    
                    cells_data.append(cell_data)
                
                structured_content = {
                    'type': 'jupyter_notebook',
                    'cell_count': len(notebook.cells),
                    'cells': cells_data,
                    'metadata': notebook.metadata,
                    'title': os.path.basename(jupyter_path)
                }
                
                raw_content = '\n'.join(raw_content_parts)
                
            else:
                # Fallback: treat as JSON
                with open(jupyter_path, 'r', encoding='utf-8') as file:
                    notebook_json = json.load(file)
                
                raw_content = f"Jupyter Notebook: {os.path.basename(jupyter_path)}\n"
                raw_content += "Note: Processed as JSON (nbformat not available)\n\n"
                raw_content += json.dumps(notebook_json, indent=2)[:2000] + "..."
                
                structured_content = {
                    'type': 'jupyter_notebook',
                    'cell_count': len(notebook_json.get('cells', [])),
                    'raw_json': notebook_json,
                    'title': os.path.basename(jupyter_path)
                }
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("Jupyter processing failed", error=str(e))
            raise
    
    async def _process_code(self, code_path: str, content_format: ContentFormat) -> Tuple[str, Dict[str, Any]]:
        """Process code files"""
        
        try:
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as file:
                code_content = file.read()
            
            # Basic code analysis
            lines = code_content.split('\n')
            
            # Extract functions/methods
            functions = []
            classes = []
            imports = []
            comments = []
            
            # Language-specific patterns
            if content_format == ContentFormat.PYTHON:
                function_pattern = r'^\s*def\s+(\w+)\s*\('
                class_pattern = r'^\s*class\s+(\w+)'
                import_pattern = r'^\s*(import\s+\w+|from\s+\w+\s+import)'
                comment_pattern = r'^\s*#(.+)'
            elif content_format == ContentFormat.JAVASCRIPT:
                function_pattern = r'^\s*function\s+(\w+)\s*\(|^\s*(\w+)\s*=\s*function'
                class_pattern = r'^\s*class\s+(\w+)'
                import_pattern = r'^\s*(import\s+.+|const\s+.+=\s+require)'
                comment_pattern = r'^\s*//(.+)'
            else:
                # Generic patterns
                function_pattern = r'^\s*(def|function)\s+(\w+)'
                class_pattern = r'^\s*class\s+(\w+)'
                import_pattern = r'^\s*(import|include|require)'
                comment_pattern = r'^\s*(#|//)(.+)'
            
            for line_num, line in enumerate(lines):
                # Extract functions
                match = re.search(function_pattern, line)
                if match:
                    func_name = match.group(1) or match.group(2)
                    if func_name:
                        functions.append({'name': func_name, 'line': line_num + 1})
                
                # Extract classes
                match = re.search(class_pattern, line)
                if match:
                    class_name = match.group(1)
                    classes.append({'name': class_name, 'line': line_num + 1})
                
                # Extract imports
                if re.search(import_pattern, line):
                    imports.append(line.strip())
                
                # Extract comments
                match = re.search(comment_pattern, line)
                if match:
                    comment_text = match.group(1).strip()
                    if len(comment_text) > 10:  # Only meaningful comments
                        comments.append(comment_text)
            
            structured_content = {
                'type': 'code_file',
                'language': content_format.value,
                'line_count': len(lines),
                'functions': functions[:20],  # Limit to first 20
                'classes': classes[:20],
                'imports': imports[:20],
                'comments': comments[:10],
                'title': os.path.basename(code_path)
            }
            
            # Create raw content with structure
            raw_content_parts = [
                f"Code File: {os.path.basename(code_path)}",
                f"Language: {content_format.value}",
                f"Lines: {len(lines)}",
                ""
            ]
            
            if functions:
                raw_content_parts.append("Functions:")
                for func in functions[:10]:
                    raw_content_parts.append(f"  - {func['name']} (line {func['line']})")
                raw_content_parts.append("")
            
            if classes:
                raw_content_parts.append("Classes:")
                for cls in classes[:10]:
                    raw_content_parts.append(f"  - {cls['name']} (line {cls['line']})")
                raw_content_parts.append("")
            
            if imports:
                raw_content_parts.append("Imports:")
                for imp in imports[:10]:
                    raw_content_parts.append(f"  - {imp}")
                raw_content_parts.append("")
            
            raw_content_parts.append("Code content:")
            raw_content_parts.append(code_content[:2000] + ("..." if len(code_content) > 2000 else ""))
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.warning("Code processing failed", error=str(e))
            raise
    
    async def _extract_technical_entities(self, structured_content: Dict[str, Any], raw_content: str) -> List[Dict[str, Any]]:
        """Extract entities from technical content"""
        
        entities = []
        
        content_type = structured_content.get('type', '')
        
        if content_type == 'jupyter_notebook':
            # Extract from Jupyter notebook
            for cell in structured_content.get('cells', []):
                if cell['cell_type'] == 'code':
                    # Extract imports and function calls from code cells
                    code_entities = self._extract_code_entities(cell['source'])
                    entities.extend(code_entities)
        
        elif content_type == 'code_file':
            # Extract from code file
            for func in structured_content.get('functions', []):
                entities.append({
                    'type': 'function',
                    'value': func['name'],
                    'metadata': {'line': func['line']},
                    'confidence': 0.95
                })
            
            for cls in structured_content.get('classes', []):
                entities.append({
                    'type': 'class',
                    'value': cls['name'],
                    'metadata': {'line': cls['line']},
                    'confidence': 0.95
                })
            
            for imp in structured_content.get('imports', []):
                entities.append({
                    'type': 'import',
                    'value': imp,
                    'confidence': 0.9
                })
        
        return entities
    
    def _extract_code_entities(self, code: str) -> List[Dict[str, Any]]:
        """Extract entities from code snippet"""
        
        entities = []
        
        # Extract variable assignments
        var_pattern = r'(\w+)\s*=\s*'
        for match in re.finditer(var_pattern, code):
            var_name = match.group(1)
            if not var_name.isupper():  # Skip constants
                entities.append({
                    'type': 'variable',
                    'value': var_name,
                    'confidence': 0.8
                })
        
        # Extract function calls
        func_call_pattern = r'(\w+)\s*\('
        for match in re.finditer(func_call_pattern, code):
            func_name = match.group(1)
            entities.append({
                'type': 'function_call',
                'value': func_name,
                'confidence': 0.8
            })
        
        return entities[:20]  # Limit to 20 entities
    
    def _assess_technical_quality(self, structured_content: Dict[str, Any]) -> float:
        """Assess technical content quality score (0.0-1.0)"""
        
        quality_factors = []
        
        content_type = structured_content.get('type', '')
        
        if content_type == 'jupyter_notebook':
            cell_count = structured_content.get('cell_count', 0)
            if cell_count > 5:
                quality_factors.append(1.0)
            elif cell_count > 1:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
            
            # Check for mix of code and markdown cells
            cells = structured_content.get('cells', [])
            cell_types = set(cell['cell_type'] for cell in cells)
            if len(cell_types) > 1:
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.6)
        
        elif content_type == 'code_file':
            line_count = structured_content.get('line_count', 0)
            function_count = len(structured_content.get('functions', []))
            
            if line_count > 100 and function_count > 5:
                quality_factors.append(1.0)
            elif line_count > 50 and function_count > 1:
                quality_factors.append(0.8)
            elif line_count > 10:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

class ZIMProcessor:
    """Processes ZIM archive files (Wikipedia and other knowledge bases)"""
    
    def __init__(self):
        self.supported_formats = {ContentFormat.ZIM}
    
    async def process_content(self, content_path: str, content_format: ContentFormat) -> ProcessedContent:
        """Process ZIM archive content for World Model knowledge extraction"""
        
        start_time = time.time()
        
        try:
            if not ZIM_AVAILABLE:
                raise ImportError("ZIM processing requires libzim or zimply library")
            
            # Check memory usage before processing
            if not check_memory_usage():
                logger.warning("Memory usage too high, deferring ZIM processing", path=content_path)
                raise ResourceWarning("Memory usage exceeded threshold")
            
            # Generate safe filename and content hash
            file_size = os.path.getsize(content_path) if os.path.exists(content_path) else 0
            content_hash = hashlib.sha256(content_path.encode()).hexdigest()[:12]
            short_filename, original_filename = generate_safe_filename(content_path, content_hash)
            
            # Create metadata
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                size_bytes=file_size,
                short_filename=short_filename,
                original_filename=original_filename,
                content_hash=content_hash
            )
            
            # Process ZIM file with timeout protection
            raw_content, structured_content = await safe_file_operation(
                self._process_zim, content_path, timeout=600  # 10 minute timeout for large ZIM files
            )
            
            # Extract knowledge entities from ZIM content
            entities = await self._extract_zim_entities(structured_content, raw_content)
            
            # Update metadata
            metadata.title = structured_content.get('title', os.path.basename(content_path))
            metadata.processing_time = time.time() - start_time
            metadata.quality_score = self._assess_zim_quality(structured_content)
            
            return ProcessedContent(
                metadata=metadata,
                raw_content=raw_content,
                structured_content=structured_content,
                extracted_entities=entities,
                processing_status=ProcessingStatus.COMPLETED
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("ZIM processing failed", 
                        content_path=content_path, 
                        format=content_format.value, 
                        error=str(e))
            
            metadata = ContentMetadata(
                source_path=content_path,
                content_format=content_format,
                processing_time=processing_time
            )
            
            return ProcessedContent(
                metadata=metadata,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    async def _process_zim(self, zim_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process ZIM archive file to extract World Model knowledge"""
        
        try:
            if ZIM_LIBRARY == "libzim":
                return await self._process_zim_libzim(zim_path)
            elif ZIM_LIBRARY == "zimply":
                return await self._process_zim_zimply(zim_path)
            else:
                raise ImportError("No ZIM library available")
                
        except Exception as e:
            logger.error("ZIM processing failed", path=zim_path, error=str(e))
            raise
    
    async def _process_zim_libzim(self, zim_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process ZIM file using libzim library"""
        
        try:
            archive = Archive(zim_path)
            
            # Extract basic metadata (fix for libzim API)
            try:
                # Try to get metadata directly from archive
                title = archive.metadata.get('Title', os.path.basename(zim_path))
            except AttributeError:
                # Fallback for different libzim versions
                title = os.path.basename(zim_path)
            
            zim_metadata = {
                'type': 'zim_archive',
                'title': title,
                'description': 'Wikipedia Archive',
                'creator': 'Wikipedia',
                'publisher': 'Kiwix',
                'date': '2025',
                'language': 'en',
                'article_count': archive.entry_count if hasattr(archive, 'entry_count') else 0,
                'size_bytes': os.path.getsize(zim_path)
            }
            
            # Extract key articles for World Model
            world_model_articles = []
            raw_content_parts = [
                f"ZIM Archive: {zim_metadata['title']}",
                f"Description: {zim_metadata['description']}",
                f"Articles: {zim_metadata['article_count']:,}",
                f"Language: {zim_metadata['language']}",
                ""
            ]
            
            # Sample comprehensive articles for robust World Model (increased for production)
            max_articles = 25000  # Allow up to 25k articles per ZIM file
            articles_processed = 0
            
            # Priority patterns for World Model knowledge
            priority_patterns = [
                # Core scientific concepts
                r'^(.*physics.*|.*chemistry.*|.*biology.*|.*mathematics.*)',
                r'^(.*science.*|.*theory.*|.*principle.*|.*law.*)',
                # Fundamental concepts
                r'^(.*definition.*|.*concept.*|.*fundamental.*|.*basic.*)',
                # Important topics  
                r'^(.*important.*|.*major.*|.*significant.*|.*key.*)',
                # Lists and categories
                r'^(list.*|category.*|index.*)'
            ]
            
            # Aggressive approach: extract much more content for comprehensive World Model
            try:
                # Get extensive samples for World Model completeness
                total_entries = archive.entry_count if hasattr(archive, 'entry_count') else 1000
                article_count = getattr(archive, 'article_count', total_entries // 2)
                
                # Scale extraction based on archive size - aim for 10,000-20,000 articles per domain
                if article_count > 100000:  # Large archives (computer, medicine, math)
                    target_articles = min(20000, max_articles)
                    sample_size = min(100000, total_entries)  # Sample up to 100k entries
                elif article_count > 50000:  # Medium archives
                    target_articles = min(15000, max_articles) 
                    sample_size = min(75000, total_entries)
                else:  # Smaller archives
                    target_articles = min(10000, max_articles)
                    sample_size = min(50000, total_entries)
                
                logger.info(f"DEBUG: ZIM archive has {total_entries} entries ({article_count} articles), targeting {target_articles} extractions from {sample_size} samples")
                
                # Try to access articles using known titles first, then random
                known_titles = [
                    'Physics', 'Chemistry', 'Biology', 'Mathematics', 'Computer_science',
                    'Algorithm', 'Machine_learning', 'Artificial_intelligence', 'Quantum_mechanics',
                    'Relativity', 'Evolution', 'DNA', 'Cell', 'Protein', 'Calculus', 'Statistics'
                ]
                
                # First try known article titles with namespace prefix
                for test_title in known_titles:
                    if articles_processed >= max_articles:
                        break
                    try:
                        # Try with 'A/' namespace prefix for articles
                        article_path = f'A/{test_title}'
                        if archive.has_entry_by_path(article_path):
                            entry = archive.get_entry_by_path(article_path)
                            if not entry.is_redirect:
                                item = entry.get_item()
                                if item:
                                    content_bytes = item.content
                                    if content_bytes and len(content_bytes.tobytes()) > 500:
                                        content_text = content_bytes.tobytes().decode('utf-8', errors='ignore')
                                        clean_text = self._extract_text_from_html(content_text)
                                        if len(clean_text.strip()) > 200:
                                            article_data = {
                                                'title': test_title,
                                                'path': article_path,
                                                'content': clean_text[:2000],
                                                'word_count': len(clean_text.split()),
                                                'snippet': clean_text[:200] + '...'
                                            }
                                            world_model_articles.append(article_data)
                                            articles_processed += 1
                                            logger.info(f"DEBUG: Successfully extracted known article '{test_title}' ({len(clean_text)} chars)")
                    except Exception as e:
                        logger.debug(f"Failed to get known article {test_title}: {e}")
                        continue
                
                # Then use aggressive random sampling for comprehensive coverage  
                random_attempts = 0
                max_random_attempts = sample_size  # Use the scaled sample size
                
                logger.info(f"DEBUG: Starting aggressive random sampling - target: {target_articles}, max attempts: {max_random_attempts}")
                
                while articles_processed < target_articles and random_attempts < max_random_attempts:
                    random_attempts += 1
                    try:
                        # Get a random entry from the archive
                        entry = archive.get_random_entry()
                        
                        # Skip redirects and non-content entries
                        if entry.is_redirect:
                            continue
                            
                        title = entry.title
                        if not title or title.startswith('File:') or title.startswith('Template:') or title.startswith('Category:'):
                            continue
                            
                        # Get the item content
                        item = entry.get_item()
                        if not item:
                            continue
                            
                        mimetype = item.mimetype if hasattr(item, 'mimetype') else 'unknown'
                        if mimetype != 'text/html':
                            continue
                            
                        # Extract content
                        content = item.content.tobytes().decode('utf-8', errors='ignore')
                        
                        # Extract clean text from HTML
                        clean_text = self._extract_text_from_html(content)
                        if len(clean_text) > 100:  # Only include substantial articles
                            article_data = {
                                'title': title,
                                'path': entry.get_path() if hasattr(entry, 'get_path') else '',
                                'content': clean_text[:2000],  # Limit content size
                                'word_count': len(clean_text.split()),
                                'snippet': clean_text[:200] + '...'
                            }
                            world_model_articles.append(article_data)
                            articles_processed += 1
                            
                            # Progress logging for aggressive extraction
                            if articles_processed <= 10 or articles_processed % 1000 == 0:
                                logger.info(f"DEBUG: Successfully extracted article '{title}' ({len(clean_text)} chars) - Progress: {articles_processed}/{target_articles}")
                                if articles_processed <= 3:
                                    logger.info(f"DEBUG: Content preview: {clean_text[:200]}...")
                            
                    except Exception as entry_error:
                        # Skip problematic entries
                        continue
                        
            except Exception as iteration_error:
                logger.warning(f"ZIM iteration failed: {str(iteration_error)}")
                # No fallback - we need real data only
                world_model_articles = []
                articles_processed = 0
            
            # Add article summaries to raw content
            raw_content_parts.append(f"Processed Articles ({len(world_model_articles)}):")
            for i, article in enumerate(world_model_articles[:10]):  # Show first 10 in raw content
                raw_content_parts.append(f"{i+1}. {article['title']}")
                raw_content_parts.append(f"   Words: {article['word_count']}")
                raw_content_parts.append(f"   Preview: {article['content'][:200]}...")
                raw_content_parts.append("")
            
            if len(world_model_articles) > 10:
                raw_content_parts.append(f"... and {len(world_model_articles) - 10} more articles")
            
            structured_content = {
                **zim_metadata,
                'world_model_articles': world_model_articles,
                'articles_processed': articles_processed,
                'processing_stats': {
                    'total_articles_available': zim_metadata['article_count'],
                    'articles_processed': len(world_model_articles),
                    'processing_ratio': len(world_model_articles) / max(zim_metadata['article_count'], 1)
                }
            }
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.error("libzim processing failed", error=str(e))
            raise
    
    async def _process_zim_zimply(self, zim_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process ZIM file using zimply library (fallback)"""
        
        try:
            import zimply
            
            zim_file = zimply.ZIMFile(zim_path)
            
            # Extract basic metadata
            zim_metadata = {
                'type': 'zim_archive',
                'title': getattr(zim_file, 'title', os.path.basename(zim_path)),
                'description': getattr(zim_file, 'description', ''),
                'article_count': getattr(zim_file, 'article_count', 0),
                'size_bytes': os.path.getsize(zim_path)
            }
            
            raw_content_parts = [
                f"ZIM Archive: {zim_metadata['title']}",
                f"Articles: {zim_metadata['article_count']:,}",
                ""
            ]
            
            # Extract sample articles
            world_model_articles = []
            articles_to_process = min(100, zim_metadata.get('article_count', 0))  # Limit processing
            
            try:
                for i, article in enumerate(zim_file):
                    if i >= articles_to_process:
                        break
                    
                    try:
                        title = article.title
                        content = article.content
                        
                        if title and content and len(content) > 100:
                            clean_text = self._extract_text_from_html(content)
                            if len(clean_text) > 50:
                                article_data = {
                                    'title': title,
                                    'content': clean_text[:1000],  # Limit content
                                    'word_count': len(clean_text.split())
                                }
                                world_model_articles.append(article_data)
                                
                    except Exception as article_error:
                        logger.debug("Failed to process zimply article", error=str(article_error))
                        continue
                        
            except Exception as iteration_error:
                logger.warning("ZIM iteration failed", error=str(iteration_error))
            
            # Add to raw content
            raw_content_parts.append(f"Sample Articles ({len(world_model_articles)}):")
            for i, article in enumerate(world_model_articles[:5]):
                raw_content_parts.append(f"{i+1}. {article['title']}")
                raw_content_parts.append(f"   Preview: {article['content'][:200]}...")
                raw_content_parts.append("")
            
            structured_content = {
                **zim_metadata,
                'world_model_articles': world_model_articles,
                'articles_processed': len(world_model_articles)
            }
            
            raw_content = '\n'.join(raw_content_parts)
            
            return raw_content, structured_content
            
        except Exception as e:
            logger.error("zimply processing failed", error=str(e))
            raise
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        
        try:
            if DOCUMENT_PROCESSORS_AVAILABLE:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return clean_text
            else:
                # Fallback: simple tag removal
                import re
                clean_text = re.sub(r'<[^>]+>', '', html_content)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                return clean_text
                
        except Exception as e:
            logger.debug("HTML text extraction failed", error=str(e))
            # Return original content with simple cleanup
            import re
            return re.sub(r'<[^>]+>', '', html_content)[:1000]
    
    async def _extract_zim_entities(self, structured_content: Dict[str, Any], raw_content: str) -> List[Dict[str, Any]]:
        """Extract knowledge entities from ZIM content for World Model"""
        
        entities = []
        
        # Extract archive metadata entities
        entities.append({
            'type': 'knowledge_archive',
            'value': structured_content.get('title', ''),
            'metadata': {
                'article_count': structured_content.get('article_count', 0),
                'language': structured_content.get('language', 'en'),
                'domain': self._detect_domain_from_title(structured_content.get('title', ''))
            },
            'confidence': 1.0
        })
        
        # Extract article entities for World Model
        for article in structured_content.get('world_model_articles', [])[:50]:  # Limit entities
            entities.append({
                'type': 'world_model_concept',
                'value': article['title'],
                'metadata': {
                    'word_count': article['word_count'],
                    'content_preview': article['content'][:200],
                    'source_archive': structured_content.get('title', ''),
                    'domain': self._detect_domain_from_title(structured_content.get('title', ''))
                },
                'confidence': 0.9
            })
        
        # Extract domain-specific concepts
        domain = self._detect_domain_from_title(structured_content.get('title', ''))
        if domain:
            entities.append({
                'type': 'knowledge_domain',
                'value': domain,
                'metadata': {
                    'article_count': len(structured_content.get('world_model_articles', [])),
                    'source': 'wikipedia_zim'
                },
                'confidence': 0.95
            })
        
        return entities
    
    def _detect_domain_from_title(self, title: str) -> str:
        """Detect knowledge domain from ZIM archive title"""
        
        title_lower = title.lower()
        
        domain_patterns = {
            'physics': ['physics', 'physical'],
            'chemistry': ['chemistry', 'chemical'],
            'biology': ['biology', 'biological', 'molcell'],
            'medicine': ['medicine', 'medical'],
            'mathematics': ['mathematics', 'mathematical', 'math'],
            'astronomy': ['astronomy', 'astronomical'],
            'computer_science': ['computer', 'computing']
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return domain
        
        return 'general_knowledge'
    
    def _assess_zim_quality(self, structured_content: Dict[str, Any]) -> float:
        """Assess ZIM content quality for World Model (0.0-1.0)"""
        
        quality_factors = []
        
        # Article count factor
        article_count = len(structured_content.get('world_model_articles', []))
        if article_count > 500:
            quality_factors.append(1.0)
        elif article_count > 100:
            quality_factors.append(0.8)
        elif article_count > 10:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Content richness factor
        total_words = sum(article.get('word_count', 0) for article in structured_content.get('world_model_articles', []))
        if total_words > 50000:
            quality_factors.append(1.0)
        elif total_words > 10000:
            quality_factors.append(0.8)
        elif total_words > 1000:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)
        
        # Metadata completeness factor
        metadata_fields = ['title', 'description', 'language', 'article_count']
        present_fields = sum(1 for field in metadata_fields if structured_content.get(field))
        metadata_score = present_fields / len(metadata_fields)
        quality_factors.append(metadata_score)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

class UniversalIngestionEngine:
    """Main orchestrator for universal knowledge ingestion"""
    
    def __init__(self):
        # Initialize format processors
        self.document_processor = DocumentProcessor()
        self.structured_data_processor = StructuredDataProcessor()
        self.technical_processor = TechnicalProcessor()
        self.zim_processor = ZIMProcessor()
        
        # Processor mapping
        self.processors = {
            **{fmt: self.document_processor for fmt in self.document_processor.supported_formats},
            **{fmt: self.structured_data_processor for fmt in self.structured_data_processor.supported_formats},
            **{fmt: self.technical_processor for fmt in self.technical_processor.supported_formats},
            **{fmt: self.zim_processor for fmt in self.zim_processor.supported_formats}
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Universal Ingestion Engine initialized",
                   supported_formats=[fmt.value for fmt in self.processors.keys()])
    
    async def ingest_content(self, 
                           content_paths: List[str], 
                           content_formats: Optional[List[ContentFormat]] = None,
                           batch_size: Optional[int] = None,
                           save_progress: bool = True,
                           output_dir: Optional[str] = None) -> IngestionResult:
        """Ingest multiple content items with batch processing and progress tracking"""
        
        start_time = time.time()
        batch_size = batch_size or BATCH_SIZE
        
        # Auto-detect formats if not provided
        if content_formats is None:
            content_formats = [self._detect_content_format(path) for path in content_paths]
        
        # Ensure equal lengths
        if len(content_paths) != len(content_formats):
            raise ValueError("content_paths and content_formats must have equal lengths")
        
        logger.info("Starting enhanced content ingestion",
                   total_items=len(content_paths),
                   formats=list(set(fmt.value for fmt in content_formats)),
                   batch_size=batch_size,
                   memory_threshold=f"{MEMORY_THRESHOLD*100}%")
        
        # Filter supported formats
        valid_items = []
        for path, fmt in zip(content_paths, content_formats):
            if fmt in self.processors:
                valid_items.append((path, fmt))
            else:
                logger.warning("Unsupported format", path=path, format=fmt.value)
        
        # Process content in batches
        successful_content = []
        failed_content = []
        error_summary = []
        
        total_batches = (len(valid_items) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(valid_items), batch_size)):
            batch_items = valid_items[i:i + batch_size]
            
            logger.info("Processing batch",
                       batch=batch_num + 1,
                       total_batches=total_batches,
                       batch_items=len(batch_items),
                       memory_usage=f"{psutil.virtual_memory().percent:.1f}%")
            
            # Check memory before processing batch
            if not check_memory_usage():
                logger.warning("Memory usage too high, running garbage collection")
                gc.collect()
                
                if not check_memory_usage():
                    logger.error("Memory usage still too high, stopping processing")
                    error_summary.append(f"Processing stopped due to high memory usage at batch {batch_num + 1}")
                    break
            
            # Process batch
            batch_tasks = [self._process_single_content(path, fmt) for path, fmt in batch_items]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    error_summary.append(str(result))
                    failed_content.append(None)
                elif isinstance(result, ProcessedContent):
                    if result.processing_status == ProcessingStatus.COMPLETED:
                        successful_content.append(result)
                        
                        # Save individual processed content if requested
                        if save_progress and output_dir:
                            await self._save_processed_content(result, output_dir)
                    else:
                        failed_content.append(result)
                        if result.error_message:
                            error_summary.append(result.error_message)
                else:
                    error_summary.append(f"Unknown processing result: {type(result)}")
            
            # Clean up memory after each batch
            gc.collect()
            
            # Progress update
            processed_so_far = len(successful_content) + len([f for f in failed_content if f is not None])
            logger.info("Batch completed",
                       batch=batch_num + 1,
                       successful=len(successful_content),
                       failed=len([f for f in failed_content if f is not None]),
                       progress=f"{processed_so_far}/{len(valid_items)} ({processed_so_far*100//len(valid_items)}%)")
        
        # Final processing statistics
        total_failed = len([f for f in failed_content if f is not None]) + len([f for f in failed_content if f is None])
        
        # Calculate metrics
        total_processing_time = time.time() - start_time
        performance_metrics = self._calculate_performance_metrics(successful_content, total_processing_time)
        
        # Create ingestion result
        ingestion_result = IngestionResult(
            total_items=len(content_paths),
            successful_items=len(successful_content),
            failed_items=total_failed,
            skipped_items=len(content_paths) - len(valid_items),  # Unsupported formats
            total_processing_time=total_processing_time,
            processed_content=successful_content,
            error_summary=error_summary,
            performance_metrics=performance_metrics
        )
        
        logger.info("Content ingestion completed",
                   total_items=ingestion_result.total_items,
                   successful=ingestion_result.successful_items,
                   failed=ingestion_result.failed_items,
                   processing_time=total_processing_time)
        
        return ingestion_result
    
    async def _process_single_content(self, content_path: str, content_format: ContentFormat) -> ProcessedContent:
        """Process a single content item"""
        
        try:
            processor = self.processors.get(content_format)
            if not processor:
                raise ValueError(f"No processor available for format: {content_format}")
            
            # Process content using appropriate processor
            processed_content = await processor.process_content(content_path, content_format)
            
            return processed_content
            
        except Exception as e:
            logger.error("Single content processing failed",
                        path=content_path,
                        format=content_format.value,
                        error=str(e))
            raise
    
    async def _save_processed_content(self, content: ProcessedContent, output_dir: str):
        """Save processed content to output directory using safe filename"""
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Use safe filename for saving
            safe_filename = content.metadata.short_filename or f"doc_{content.metadata.content_hash}"
            output_path = os.path.join(output_dir, f"{safe_filename}.json")
            
            # Create save data
            save_data = {
                "metadata": {
                    "content_id": content.metadata.content_id,
                    "source_path": content.metadata.source_path,
                    "original_filename": content.metadata.original_filename,
                    "short_filename": content.metadata.short_filename,
                    "content_format": content.metadata.content_format.value,
                    "title": content.metadata.title,
                    "size_bytes": content.metadata.size_bytes,
                    "quality_score": content.metadata.quality_score,
                    "processing_time": content.metadata.processing_time,
                    "processed_at": content.metadata.processed_at.isoformat(),
                    "content_hash": content.metadata.content_hash
                },
                "raw_content": content.raw_content[:10000] if content.raw_content else "",  # Limit size
                "structured_content": content.structured_content,
                "extracted_entities": content.extracted_entities[:50],  # Limit entities
                "processing_status": content.processing_status.value,
                "error_message": content.error_message
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Processed content saved", 
                        original_file=content.metadata.original_filename,
                        saved_as=safe_filename,
                        output_path=output_path)
                        
        except Exception as e:
            logger.warning("Failed to save processed content", 
                          content_id=content.metadata.content_id,
                          error=str(e))
    
    def _detect_content_format(self, file_path: str) -> ContentFormat:
        """Auto-detect content format from file path"""
        
        file_path = file_path.lower()
        extension = Path(file_path).suffix.lower()
        
        # Extension to format mapping
        extension_map = {
            '.pdf': ContentFormat.PDF,
            '.docx': ContentFormat.DOCX,
            '.doc': ContentFormat.DOCX,
            '.txt': ContentFormat.TXT,
            '.html': ContentFormat.HTML,
            '.htm': ContentFormat.HTML,
            '.md': ContentFormat.MARKDOWN,
            '.markdown': ContentFormat.MARKDOWN,
            '.xlsx': ContentFormat.EXCEL,
            '.xls': ContentFormat.EXCEL,
            '.csv': ContentFormat.CSV,
            '.json': ContentFormat.JSON,
            '.xml': ContentFormat.XML,
            '.ipynb': ContentFormat.JUPYTER,
            '.py': ContentFormat.PYTHON,
            '.js': ContentFormat.JAVASCRIPT,
            '.ts': ContentFormat.JAVASCRIPT,
            '.java': ContentFormat.CODE,
            '.cpp': ContentFormat.CODE,
            '.c': ContentFormat.CODE,
            '.h': ContentFormat.CODE,
            '.rb': ContentFormat.CODE,
            '.php': ContentFormat.CODE,
            '.zim': ContentFormat.ZIM
        }
        
        detected_format = extension_map.get(extension)
        
        if detected_format:
            return detected_format
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith('text/'):
                return ContentFormat.TXT
            elif mime_type == 'application/json':
                return ContentFormat.JSON
            elif mime_type == 'application/xml':
                return ContentFormat.XML
        
        # Default fallback
        logger.warning("Could not detect content format, defaulting to TXT", path=file_path)
        return ContentFormat.TXT
    
    def _calculate_performance_metrics(self, successful_content: List[ProcessedContent], total_time: float) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        if not successful_content:
            return {}
        
        processing_times = [content.metadata.processing_time for content in successful_content]
        quality_scores = [content.metadata.quality_score for content in successful_content]
        
        return {
            'average_processing_time': sum(processing_times) / len(processing_times),
            'max_processing_time': max(processing_times),
            'min_processing_time': min(processing_times),
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'items_per_second': len(successful_content) / total_time if total_time > 0 else 0.0,
            'total_content_size_mb': sum(content.metadata.size_bytes for content in successful_content) / (1024 * 1024)
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported content formats"""
        return [fmt.value for fmt in self.processors.keys()]
    
    def shutdown(self):
        """Shutdown the ingestion engine"""
        self.executor.shutdown(wait=True)
        logger.info("Universal Ingestion Engine shutdown complete")

# Enhanced processing function for corpus processing  
async def process_pdf_corpus(corpus_dir: str,
                           output_dir: str,
                           batch_size: int = 25,
                           max_files: Optional[int] = None,
                           resume_from_failures: bool = True) -> Dict[str, Any]:
    """Process PDF corpus with enhanced error handling and progress tracking"""
    
    logger.info("Starting enhanced PDF corpus processing",
               corpus_dir=corpus_dir,
               output_dir=output_dir,
               batch_size=batch_size)
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    logger.info("Found PDF files to process", count=len(pdf_files))
    
    # Filter out already processed files if resuming
    if resume_from_failures and os.path.exists(output_dir):
        existing_files = set()
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                existing_files.add(file.replace('.json', ''))
        
        # Filter PDFs that haven't been processed
        unprocessed_pdfs = []
        for pdf_path in pdf_files:
            content_hash = hashlib.sha256(pdf_path.encode()).hexdigest()[:12]
            short_filename, _ = generate_safe_filename(pdf_path, content_hash)
            
            if short_filename not in existing_files:
                unprocessed_pdfs.append(pdf_path)
        
        pdf_files = unprocessed_pdfs
        logger.info("Resuming processing", remaining_files=len(pdf_files))
    
    # Create ingestion engine
    engine = UniversalIngestionEngine()
    
    try:
        # Process with enhanced engine
        result = await engine.ingest_content(
            content_paths=pdf_files,
            content_formats=None,  # Auto-detect
            batch_size=batch_size,
            save_progress=True,
            output_dir=output_dir
        )
        
        success_rate = result.successful_items / max(result.total_items, 1)
        avg_quality = sum(content.metadata.quality_score for content in result.processed_content) / max(len(result.processed_content), 1)
        
        return {
            "conclusion": f"Enhanced corpus processing: {result.successful_items}/{result.total_items} PDFs processed with {success_rate:.1%} success rate",
            "total_items": result.total_items,
            "successful_items": result.successful_items,
            "failed_items": result.failed_items,
            "skipped_items": result.skipped_items,
            "success_rate": success_rate,
            "average_quality_score": avg_quality,
            "total_processing_time": result.total_processing_time,
            "performance_metrics": result.performance_metrics,
            "error_summary": result.error_summary[:10],  # Limit errors shown
            "output_directory": output_dir,
            "improvements": [
                "Safe filename handling for long titles",
                "Memory usage monitoring and batch processing",
                "Timeout protection for large/corrupted PDFs",
                "Progress tracking and resume capability",
                "Enhanced error reporting and recovery"
            ]
        }
        
    finally:
        engine.shutdown()

# World Model Knowledge Interface - Focused on Contradiction Detection
class WorldModelKnowledge:
    """World Model for detecting factual contradictions in candidate answers"""
    
    def __init__(self):
        self.factual_assertions = {}     # fact_key -> assertion_data mapping
        self.scientific_constants = {}   # constant_name -> value mapping
        self.definitions = {}            # concept -> definition mapping
        self.domain_facts = {}           # domain -> facts mapping
        self.quality_threshold = 0.7     # Minimum quality for World Model facts
        
        # Contradiction detection patterns
        self.contradiction_patterns = {
            # Negation patterns
            'negation': [
                (r'(\w+) is not (\w+)', r'(\w+) is (\w+)'),
                (r'(\w+) cannot (\w+)', r'(\w+) can (\w+)'),
                (r'(\w+) never (\w+)', r'(\w+) always (\w+)'),
                (r'(\w+) does not (\w+)', r'(\w+) does (\w+)')
            ],
            # Quantitative contradictions  
            'quantitative': [
                (r'(\w+) is (\d+\.?\d*)', r'different_numerical_value'),
                (r'(\w+) equals (\d+\.?\d*)', r'different_numerical_value'),
                (r'(\w+) measures (\d+\.?\d*)', r'different_numerical_value')
            ],
            # Categorical contradictions
            'categorical': [
                (r'(\w+) is a type of (\w+)', r'(\w+) is not a type of (\w+)'),
                (r'(\w+) belongs to (\w+)', r'(\w+) does not belong to (\w+)')
            ]
        }
        
    def load_from_processed_content(self, processed_content_list: List[ProcessedContent]):
        """Load World Model knowledge from processed ZIM content for contradiction detection"""
        
        logger.info("Building World Model contradiction detector from ZIM content")
        
        for content in processed_content_list:
            if content.metadata.content_format == ContentFormat.ZIM and content.processing_status == ProcessingStatus.COMPLETED:
                self._extract_factual_assertions(content)
                
        logger.info(f"World Model loaded: {len(self.factual_assertions)} facts, "
                   f"{len(self.scientific_constants)} constants, "
                   f"{len(self.definitions)} definitions")
    
    def _extract_factual_assertions(self, content: ProcessedContent):
        """Extract factual assertions from ZIM content for contradiction detection"""
        
        structured_data = content.structured_content
        domain = self._detect_domain(structured_data.get('title', ''))
        
        if domain not in self.domain_facts:
            self.domain_facts[domain] = []
        
        # Process articles to extract facts
        articles = structured_data.get('world_model_articles', [])
        facts_extracted = 0
        
        # DEBUG: Show structured_data keys to understand the content
        logger.info(f"DEBUG: Structured data keys: {list(structured_data.keys())}")
        logger.info(f"DEBUG: Processing {len(articles)} articles from {domain} domain")
        
        # Additional DEBUG: Check if articles exist under different key
        if len(articles) == 0:
            logger.info(f"DEBUG: No 'world_model_articles' found. Checking other keys...")
            for key, value in structured_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    logger.info(f"DEBUG: Found list data under key '{key}' with {len(value)} items")
        
        for article in articles:
            if article.get('word_count', 0) < 50:  # Skip short articles
                continue
                
            title = article['title']
            content_text = article['content']
            
            # DEBUG: Log sample content to understand what we're working with
            logger.info(f"DEBUG: Processing article '{title}' with {article.get('word_count', 0)} words")
            logger.info(f"DEBUG: Content preview: {content_text[:300]}...")
            
            # Extract different types of factual assertions
            article_facts = []
            
            # 1. Extract scientific constants and measurements
            constants = self._extract_scientific_constants(title, content_text)
            if constants:
                logger.info(f"DEBUG: Found {len(constants)} scientific constants in '{title}': {[c['subject'] for c in constants]}")
            article_facts.extend(constants)
            
            # 2. Extract definitional statements  
            definitions = self._extract_definitions(title, content_text)
            if definitions:
                logger.info(f"DEBUG: Found {len(definitions)} definitions in '{title}': {[d['subject'] for d in definitions]}")
            article_facts.extend(definitions)
            
            # 3. Extract categorical relationships
            categories = self._extract_categorical_facts(title, content_text)
            if categories:
                logger.info(f"DEBUG: Found {len(categories)} categorical facts in '{title}': {[c['subject'] for c in categories]}")
            article_facts.extend(categories)
            
            # 4. Extract quantitative facts
            quantities = self._extract_quantitative_facts(title, content_text)
            if quantities:
                logger.info(f"DEBUG: Found {len(quantities)} quantitative facts in '{title}': {[q['subject'] for q in quantities]}")
            article_facts.extend(quantities)
            
            # Store facts with metadata
            for fact in article_facts:
                fact_key = self._generate_fact_key(fact)
                
                self.factual_assertions[fact_key] = {
                    **fact,
                    'source_article': title,
                    'source_domain': domain,
                    'confidence': 0.9,  # High confidence for Wikipedia facts
                    'source_archive': structured_data.get('title', '')
                }
                
                # Also store in specialized indexes
                if fact['type'] == 'scientific_constant':
                    self.scientific_constants[fact['subject']] = fact['object']
                elif fact['type'] == 'definition':
                    self.definitions[fact['subject']] = fact['object']
            
            self.domain_facts[domain].extend(article_facts)
            facts_extracted += len(article_facts)
        
        logger.info(f"Extracted {facts_extracted} factual assertions from {domain} domain")
    
    def _extract_scientific_constants(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Extract scientific constants and measurements"""
        
        facts = []
        content_lower = content.lower()
        
        # Common scientific constants patterns
        constant_patterns = [
            # Speed of light
            (r'speed of light.*?(\d+,?\d*,?\d*)\s*(m/s|meters per second|km/s)', 
             'speed_of_light', 'physics'),
            # Pi
            (r'pi.*?(\d+\.\d+)', 'pi', 'mathematics'),
            # Gravity
            (r'gravitational.*?acceleration.*?(\d+\.\d+)\s*(m/s)', 
             'gravitational_acceleration', 'physics'),
            # Planck constant
            (r'planck.*?constant.*?(\d+\.\d+e-?\d+)', 'planck_constant', 'physics'),
            # Avogadro number
            (r'avogadro.*?(\d+\.\d+e\+?\d+)', 'avogadro_number', 'chemistry')
        ]
        
        for pattern, constant_name, domain in constant_patterns:
            import re
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                # Clean up numerical formatting
                clean_value = value.replace(',', '')
                
                facts.append({
                    'type': 'scientific_constant',
                    'subject': constant_name,
                    'predicate': 'equals',
                    'object': clean_value,
                    'domain': domain,
                    'extraction_pattern': pattern
                })
        
        return facts
    
    def _extract_definitions(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Extract definitional statements"""
        
        facts = []
        
        # Definition patterns
        definition_patterns = [
            r'([A-Z][a-z]+) is (a|an) ([^.]+)',
            r'([A-Z][a-z]+) refers to ([^.]+)',
            r'([A-Z][a-z]+) means ([^.]+)',
            r'([A-Z][a-z]+) is defined as ([^.]+)'
        ]
        
        import re
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).lower()
                definition = match.groups()[-1].strip()  # Last group
                
                # Filter out very long or short definitions
                if 10 <= len(definition) <= 200:
                    facts.append({
                        'type': 'definition',
                        'subject': subject,
                        'predicate': 'is_defined_as',
                        'object': definition,
                        'domain': self._infer_domain_from_content(definition),
                        'extraction_pattern': pattern
                    })
        
        return facts
    
    def _extract_categorical_facts(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Extract categorical relationships (X is a type of Y)"""
        
        facts = []
        
        # Categorical patterns
        categorical_patterns = [
            r'([A-Z][a-z]+) is a type of ([^.]+)',
            r'([A-Z][a-z]+) is a kind of ([^.]+)',
            r'([A-Z][a-z]+) belongs to ([^.]+)',
            r'([A-Z][a-z]+) is classified as ([^.]+)',
            r'([A-Z][a-z]+) are ([^.]+)'  # Plural form
        ]
        
        import re
        for pattern in categorical_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).lower()
                category = match.group(2).strip()
                
                if 5 <= len(category) <= 100:  # Reasonable category length
                    facts.append({
                        'type': 'categorical',
                        'subject': subject,
                        'predicate': 'is_type_of',
                        'object': category,
                        'domain': self._infer_domain_from_content(category),
                        'extraction_pattern': pattern
                    })
        
        return facts
    
    def _extract_quantitative_facts(self, title: str, content: str) -> List[Dict[str, Any]]:
        """Extract quantitative facts (measurements, dates, numbers)"""
        
        facts = []
        
        # Quantitative patterns
        quantitative_patterns = [
            # Temperatures
            (r'([A-Z][a-z\s]+) (?:is|has|measures) (\d+\.?\d*)\s*(°C|°F|Kelvin|degrees)', 
             'temperature'),
            # Distances and sizes  
            (r'([A-Z][a-z\s]+) (?:is|measures|spans) (\d+\.?\d*)\s*(km|miles|meters|m|cm)', 
             'measurement'),
            # Dates
            (r'([A-Z][a-z\s]+) (?:was|occurred|happened) in (\d{4})', 
             'date'),
            # Weights and masses
            (r'([A-Z][a-z\s]+) (?:weighs|has mass) (\d+\.?\d*)\s*(kg|pounds|grams)', 
             'mass')
        ]
        
        import re
        for pattern, fact_type in quantitative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip().lower()
                value = match.group(2)
                unit = match.group(3) if len(match.groups()) > 2 else ''
                
                facts.append({
                    'type': 'quantitative',
                    'subject': subject,
                    'predicate': fact_type,
                    'object': f"{value} {unit}".strip(),
                    'domain': self._infer_domain_from_content(subject),
                    'extraction_pattern': pattern
                })
        
        return facts
    
    def _generate_fact_key(self, fact: Dict[str, Any]) -> str:
        """Generate unique key for factual assertion"""
        return f"{fact['type']}:{fact['subject']}:{fact['predicate']}"
    
    def _infer_domain_from_content(self, content: str) -> str:
        """Infer domain from content text"""
        
        content_lower = content.lower()
        
        domain_keywords = {
            'physics': ['physics', 'force', 'energy', 'quantum', 'particle', 'wave'],
            'chemistry': ['chemistry', 'chemical', 'molecule', 'atom', 'reaction'],
            'biology': ['biology', 'cell', 'organism', 'species', 'evolution', 'dna'],
            'mathematics': ['mathematics', 'equation', 'theorem', 'formula', 'number'],
            'medicine': ['medicine', 'medical', 'disease', 'treatment', 'patient'],
            'computer_science': ['computer', 'algorithm', 'programming', 'software'],
            'astronomy': ['astronomy', 'planet', 'star', 'galaxy', 'space', 'universe']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return domain
                
        return 'general'
    
    def _detect_domain(self, title: str) -> str:
        """Detect knowledge domain from title"""
        
        title_lower = title.lower()
        domain_patterns = {
            'physics': ['physics', 'physical'],
            'chemistry': ['chemistry', 'chemical'], 
            'biology': ['biology', 'biological', 'molcell'],
            'medicine': ['medicine', 'medical'],
            'mathematics': ['mathematics', 'mathematical', 'math'],
            'astronomy': ['astronomy', 'astronomical'],
            'computer_science': ['computer', 'computing']
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return domain
        
        return 'general_knowledge'
    
    def detect_contradictions(self, candidate_text: str, query_context: str = "") -> Dict[str, Any]:
        """Detect factual contradictions in candidate answer against World Model"""
        
        if not self.factual_assertions:
            return {
                'has_contradictions': False,
                'contradiction_score': 0.0,
                'contradictions_found': [],
                'contradiction_confidence': 0.0,
                'facts_checked': 0
            }
        
        candidate_lower = candidate_text.lower()
        contradictions_found = []
        facts_checked = 0
        
        # Check each type of factual assertion for contradictions
        
        # 1. Check scientific constants
        constant_contradictions = self._check_scientific_constant_contradictions(candidate_lower)
        contradictions_found.extend(constant_contradictions)
        
        # 2. Check definitional contradictions
        definition_contradictions = self._check_definition_contradictions(candidate_lower)
        contradictions_found.extend(definition_contradictions)
        
        # 3. Check categorical contradictions
        categorical_contradictions = self._check_categorical_contradictions(candidate_lower)
        contradictions_found.extend(categorical_contradictions)
        
        # 4. Check quantitative contradictions
        quantitative_contradictions = self._check_quantitative_contradictions(candidate_lower)
        contradictions_found.extend(quantitative_contradictions)
        
        facts_checked = len(self.factual_assertions)
        
        # Calculate contradiction scores
        contradiction_score = min(len(contradictions_found) / max(facts_checked, 1), 1.0)
        has_contradictions = len(contradictions_found) > 0
        
        # Calculate confidence based on the quality of contradictions found
        if contradictions_found:
            avg_confidence = sum(c['confidence'] for c in contradictions_found) / len(contradictions_found)
            contradiction_confidence = avg_confidence
        else:
            contradiction_confidence = 0.0
        
        return {
            'has_contradictions': has_contradictions,
            'contradiction_score': contradiction_score,
            'contradictions_found': contradictions_found,
            'contradiction_confidence': contradiction_confidence,
            'facts_checked': facts_checked,
            'major_contradictions': len([c for c in contradictions_found if c['severity'] == 'major']),
            'minor_contradictions': len([c for c in contradictions_found if c['severity'] == 'minor'])
        }
    
    def _check_scientific_constant_contradictions(self, candidate_text: str) -> List[Dict[str, Any]]:
        """Check for contradictions with scientific constants"""
        
        contradictions = []
        import re
        
        for constant_name, known_value in self.scientific_constants.items():
            # Look for mentions of this constant with different values
            constant_patterns = [
                f"{constant_name.replace('_', ' ')}.*?is.*?(\\d+[.,]?\\d*[.,]?\\d*)",
                f"{constant_name.replace('_', ' ')}.*?equals.*?(\\d+[.,]?\\d*[.,]?\\d*)",
                f"{constant_name.replace('_', ' ')}.*?(\\d+[.,]?\\d*[.,]?\\d*)"
            ]
            
            for pattern in constant_patterns:
                matches = re.finditer(pattern, candidate_text, re.IGNORECASE)
                for match in matches:
                    claimed_value = match.group(1).replace(',', '')
                    known_value_clean = str(known_value).replace(',', '')
                    
                    # Check if values are significantly different
                    try:
                        claimed_num = float(claimed_value)
                        known_num = float(known_value_clean)
                        
                        # Allow for small rounding differences (1% tolerance)
                        if abs(claimed_num - known_num) / known_num > 0.01:
                            contradictions.append({
                                'type': 'scientific_constant',
                                'constant_name': constant_name,
                                'claimed_value': claimed_value,
                                'known_value': known_value,
                                'severity': 'major',
                                'confidence': 0.9,
                                'description': f"Candidate claims {constant_name.replace('_', ' ')} is {claimed_value}, but it's actually {known_value}"
                            })
                    except ValueError:
                        continue  # Skip if we can't parse numbers
        
        return contradictions
    
    def _check_definition_contradictions(self, candidate_text: str) -> List[Dict[str, Any]]:
        """Check for contradictions with established definitions"""
        
        contradictions = []
        import re
        
        for concept, known_definition in self.definitions.items():
            # Look for alternative definitions of the same concept
            definition_patterns = [
                f"{concept} is (a|an) ([^.]+)",
                f"{concept} refers to ([^.]+)",
                f"{concept} means ([^.]+)"
            ]
            
            for pattern in definition_patterns:
                matches = re.finditer(pattern, candidate_text, re.IGNORECASE)
                for match in matches:
                    claimed_definition = match.groups()[-1].strip().lower()
                    known_definition_lower = known_definition.lower()
                    
                    # Simple contradiction check (could be enhanced with embeddings)
                    contradiction_indicators = [
                        ('not', 'is'), ('never', 'always'), ('cannot', 'can'),
                        ('false', 'true'), ('incorrect', 'correct')
                    ]
                    
                    for neg, pos in contradiction_indicators:
                        if ((neg in claimed_definition and pos in known_definition_lower) or
                            (pos in claimed_definition and neg in known_definition_lower)):
                            contradictions.append({
                                'type': 'definition',
                                'concept': concept,
                                'claimed_definition': claimed_definition,
                                'known_definition': known_definition,
                                'severity': 'minor',
                                'confidence': 0.7,
                                'description': f"Candidate definition of '{concept}' contradicts established definition"
                            })
        
        return contradictions
    
    def _check_categorical_contradictions(self, candidate_text: str) -> List[Dict[str, Any]]:
        """Check for contradictions with categorical relationships"""
        
        contradictions = []
        import re
        
        # Check against known categorical facts
        for fact_key, fact_data in self.factual_assertions.items():
            if fact_data['type'] != 'categorical':
                continue
                
            subject = fact_data['subject']
            known_category = fact_data['object']
            
            # Look for contradictory categorical statements (escape regex special chars)
            escaped_subject = re.escape(subject)
            escaped_category = re.escape(known_category)
            contradiction_patterns = [
                f"{escaped_subject} is not (a|an) {escaped_category}",
                f"{escaped_subject} is not .*{escaped_category}",
                f"{escaped_subject} does not belong to {escaped_category}",
                f"{escaped_subject} is (a|an) (?!{escaped_category})([^.]+)"  # Different category
            ]
            
            for pattern in contradiction_patterns:
                matches = re.finditer(pattern, candidate_text, re.IGNORECASE)
                for match in matches:
                    contradictions.append({
                        'type': 'categorical',
                        'subject': subject,
                        'claimed_category': match.groups()[-1] if match.groups() else 'negation',
                        'known_category': known_category,
                        'severity': 'minor',
                        'confidence': 0.8,
                        'description': f"Candidate contradicts categorical relationship: {subject} vs {known_category}"
                    })
        
        return contradictions
    
    def _check_quantitative_contradictions(self, candidate_text: str) -> List[Dict[str, Any]]:
        """Check for contradictions with quantitative facts"""
        
        contradictions = []
        import re
        
        for fact_key, fact_data in self.factual_assertions.items():
            if fact_data['type'] != 'quantitative':
                continue
                
            subject = fact_data['subject']
            known_value = fact_data['object']
            
            # Look for different quantitative claims about the same subject
            quantity_patterns = [
                f"{subject}.*?(\\d+\\.?\\d*)\\s*(\\w+)",
                f"{subject}.*?is.*?(\\d+\\.?\\d*)\\s*(\\w+)",
                f"{subject}.*?measures.*?(\\d+\\.?\\d*)\\s*(\\w+)"
            ]
            
            for pattern in quantity_patterns:
                matches = re.finditer(pattern, candidate_text, re.IGNORECASE)
                for match in matches:
                    claimed_value = f"{match.group(1)} {match.group(2)}".strip()
                    
                    # Simple string comparison (could be enhanced with unit conversion)
                    if claimed_value.lower() != known_value.lower():
                        contradictions.append({
                            'type': 'quantitative',
                            'subject': subject,
                            'claimed_value': claimed_value,
                            'known_value': known_value,
                            'severity': 'minor',
                            'confidence': 0.6,
                            'description': f"Candidate claims {subject} is {claimed_value}, but known value is {known_value}"
                        })
        
        return contradictions
    
    def get_world_model_summary(self) -> Dict[str, Any]:
        """Get summary of World Model knowledge for contradiction detection"""
        
        # Count facts by type
        fact_types = {}
        for fact_data in self.factual_assertions.values():
            fact_type = fact_data['type']
            fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
        
        # Count facts by domain
        domain_counts = {}
        for fact_data in self.factual_assertions.values():
            domain = fact_data.get('source_domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            'total_facts': len(self.factual_assertions),
            'total_concepts': len(self.definitions),  # Add missing key
            'scientific_constants': len(self.scientific_constants),
            'definitions': len(self.definitions),
            'fact_types': fact_types,
            'domains': domain_counts,  # Add missing key
            'domain_distribution': domain_counts,
            'total_domains': len(domain_counts),
            'average_quality': self.quality_threshold,  # Add missing key
            'quality_threshold': self.quality_threshold,
            'contradiction_detection_ready': len(self.factual_assertions) > 0
        }

# World Model Processing Functions
async def process_world_model_zim_files(zim_directory: str, output_dir: Optional[str] = None) -> WorldModelKnowledge:
    """Process ZIM files to build World Model knowledge base"""
    
    logger.info("Processing ZIM files for World Model", directory=zim_directory)
    
    # Find all ZIM files
    zim_files = []
    if os.path.exists(zim_directory):
        for file in os.listdir(zim_directory):
            if file.endswith('.zim'):
                zim_files.append(os.path.join(zim_directory, file))
    
    if not zim_files:
        logger.warning(f"No ZIM files found for World Model: {zim_directory}")
        return WorldModelKnowledge()
    
    logger.info(f"Found {len(zim_files)} ZIM files for processing")
    
    # Create ingestion engine
    engine = UniversalIngestionEngine()
    
    try:
        # Process ZIM files
        result = await engine.ingest_content(
            content_paths=zim_files,
            content_formats=None,  # Auto-detect
            batch_size=1,  # Process ZIM files one at a time due to size
            save_progress=True if output_dir else False,
            output_dir=output_dir
        )
        
        # Build World Model knowledge
        world_model = WorldModelKnowledge()
        world_model.load_from_processed_content(result.processed_content)
        
        summary = world_model.get_world_model_summary()
        
        logger.info(f"World Model processing complete: "
                   f"{summary['total_domains']} domains, "
                   f"{summary['total_facts']} facts, "
                   f"{summary['scientific_constants']} constants")
        
        return world_model
        
    finally:
        engine.shutdown()

# Main interface function for integration
async def universal_content_ingestion(content_paths: List[str],
                                    content_formats: Optional[List[str]] = None,
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Universal content ingestion interface for NWTN integration"""
    
    # Convert string formats to ContentFormat enums if provided
    format_enums = None
    if content_formats:
        format_map = {fmt.value: fmt for fmt in ContentFormat}
        format_enums = [format_map.get(fmt, ContentFormat.TXT) for fmt in content_formats]
    
    # Create ingestion engine
    engine = UniversalIngestionEngine()
    
    try:
        # Perform ingestion
        result = await engine.ingest_content(content_paths, format_enums)
        
        # Calculate aggregate metrics
        success_rate = result.successful_items / max(result.total_items, 1)
        avg_quality = sum(content.metadata.quality_score for content in result.processed_content) / max(len(result.processed_content), 1)
        
        # Convert to dictionary format expected by NWTN systems
        return {
            "conclusion": f"Successfully ingested {result.successful_items}/{result.total_items} content items with {success_rate:.1%} success rate",
            "total_items": result.total_items,
            "successful_items": result.successful_items,
            "failed_items": result.failed_items,
            "success_rate": success_rate,
            "average_quality_score": avg_quality,
            "total_processing_time": result.total_processing_time,
            "performance_metrics": result.performance_metrics,
            "processed_content": result.processed_content,
            "supported_formats": engine.get_supported_formats(),
            "reasoning_chain": [
                f"Processed {result.total_items} content items across multiple formats",
                f"Achieved {success_rate:.1%} success rate with {avg_quality:.2f} average quality",
                f"Processing completed in {result.total_processing_time:.2f} seconds",
                f"Supports {len(engine.get_supported_formats())} different content formats"
            ],
            "quality_score": avg_quality,
            "ingestion_result": result,
            "error_summary": result.error_summary
        }
        
    finally:
        engine.shutdown()

if __name__ == "__main__":
    # Test the universal knowledge ingestion engine
    async def test_universal_ingestion():
        import tempfile
        
        print("Universal Knowledge Ingestion Engine Test:")
        print("=" * 50)
        
        # Create test files
        test_files = []
        
        # Create test TXT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about artificial intelligence and machine learning.")
            test_files.append(f.name)
        
        # Create test JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "title": "Test Data",
                "items": ["item1", "item2", "item3"],
                "metadata": {"version": "1.0", "author": "test"}
            }
            json.dump(test_data, f)
            test_files.append(f.name)
        
        # Create test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,value,category\n")
            f.write("item1,100,A\n")
            f.write("item2,200,B\n")
            f.write("item3,300,A\n")
            test_files.append(f.name)
        
        try:
            # Test universal content ingestion
            result = await universal_content_ingestion(
                content_paths=test_files,
                context={'test_mode': True}
            )
            
            print(f"Ingestion Results:")
            print(f"Total Items: {result['total_items']}")
            print(f"Successful: {result['successful_items']}")
            print(f"Failed: {result['failed_items']}")
            print(f"Success Rate: {result['success_rate']:.1%}")
            print(f"Average Quality: {result['average_quality_score']:.2f}")
            print(f"Processing Time: {result['total_processing_time']:.2f}s")
            
            print(f"\nSupported Formats ({len(result['supported_formats'])}):")
            for fmt in result['supported_formats'][:10]:
                print(f"  - {fmt}")
            if len(result['supported_formats']) > 10:
                print(f"  ... and {len(result['supported_formats']) - 10} more")
            
            print("\nProcessed Content:")
            for i, content in enumerate(result['processed_content'][:3], 1):
                print(f"{i}. {content.metadata.title}")
                print(f"   Format: {content.metadata.content_format.value}")
                print(f"   Quality: {content.metadata.quality_score:.2f}")
                print(f"   Status: {content.processing_status.value}")
            
            if result['error_summary']:
                print(f"\nErrors ({len(result['error_summary'])}):")
                for error in result['error_summary'][:3]:
                    print(f"  - {error}")
            
        finally:
            # Clean up test files
            for file_path in test_files:
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    asyncio.run(test_universal_ingestion())