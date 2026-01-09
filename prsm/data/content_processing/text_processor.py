"""
PRSM Content Text Processing Pipeline

Advanced text preprocessing for optimal embedding generation and storage.
Handles content cleaning, chunking, normalization, and optimization for
different content types (research papers, code, datasets, etc.).
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# Optional dependencies for advanced processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import spacy
    HAS_SPACY = False  # Set to False initially, will be True if model loads
    try:
        nlp = spacy.load("en_core_web_sm")
        HAS_SPACY = True
    except OSError:
        nlp = None
except ImportError:
    HAS_SPACY = False
    nlp = None

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types for specialized processing"""
    TEXT = "text"
    RESEARCH_PAPER = "research_paper"
    CODE = "code"
    DATASET = "dataset"
    ACADEMIC_ABSTRACT = "academic_abstract"
    TECHNICAL_DOCUMENTATION = "technical_documentation"


@dataclass
class ProcessingConfig:
    """Configuration for content processing"""
    max_chunk_size: int = 512  # Max tokens per chunk
    chunk_overlap: int = 50    # Overlap between chunks
    min_chunk_size: int = 100  # Minimum viable chunk size
    remove_stopwords: bool = False  # Keep stopwords for context
    normalize_whitespace: bool = True
    preserve_structure: bool = True  # Keep section headers, etc.
    extract_metadata: bool = True    # Extract titles, authors, etc.
    content_type: ContentType = ContentType.TEXT


@dataclass
class ProcessedChunk:
    """A processed chunk of content ready for embedding"""
    text: str
    chunk_id: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_index: int
    parent_content_id: str


@dataclass
class ProcessedContent:
    """Complete processed content with all chunks and metadata"""
    content_id: str
    original_text: str
    processed_chunks: List[ProcessedChunk]
    extracted_metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]


class ContentTextProcessor:
    """
    Advanced text processor for PRSM content
    
    Features:
    - Content type-specific processing
    - Intelligent chunking with overlap
    - Metadata extraction
    - Text normalization and cleaning
    - Token counting and optimization
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.stemmer = PorterStemmer() if HAS_NLTK else None
        
        # Download required NLTK data if available
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading required NLTK data...")
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data: {e}")
        
        # Set up stop words
        self.stop_words = set()
        if HAS_NLTK and self.config.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Failed to load stopwords: {e}")
    
    def process_content(self, text: str, content_id: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
        """
        Process raw content into optimized chunks for embedding
        
        Args:
            text: Raw content text
            content_id: Unique identifier for the content
            metadata: Additional metadata about the content
            
        Returns:
            ProcessedContent with chunks and extracted metadata
        """
        start_time = time.time()
        metadata = metadata or {}
        
        logger.info(f"Processing content {content_id} (type: {self.config.content_type.value})")
        
        # Step 1: Extract metadata from content
        extracted_metadata = self._extract_metadata(text, metadata)
        
        # Step 2: Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Step 3: Intelligent chunking based on content type
        chunks = self._create_chunks(cleaned_text, content_id)
        
        # Step 4: Post-process chunks
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = self._process_chunk(
                chunk_text, 
                chunk_id=f"{content_id}_chunk_{i:03d}",
                chunk_index=i,
                parent_content_id=content_id,
                parent_metadata=extracted_metadata
            )
            processed_chunks.append(chunk)
        
        # Step 5: Compile processing statistics
        processing_time = time.time() - start_time
        processing_stats = {
            "processing_time": processing_time,
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "chunk_count": len(processed_chunks),
            "total_tokens": sum(chunk.token_count for chunk in processed_chunks),
            "average_chunk_size": sum(chunk.token_count for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0,
            "content_type": self.config.content_type.value,
            "processing_config": {
                "max_chunk_size": self.config.max_chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "preserve_structure": self.config.preserve_structure
            }
        }
        
        return ProcessedContent(
            content_id=content_id,
            original_text=text,
            processed_chunks=processed_chunks,
            extracted_metadata=extracted_metadata,
            processing_stats=processing_stats
        )
    
    def _extract_metadata(self, text: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from content based on content type"""
        metadata = existing_metadata.copy()
        
        if self.config.content_type == ContentType.RESEARCH_PAPER:
            metadata.update(self._extract_paper_metadata(text))
        elif self.config.content_type == ContentType.CODE:
            metadata.update(self._extract_code_metadata(text))
        elif self.config.content_type == ContentType.DATASET:
            metadata.update(self._extract_dataset_metadata(text))
        
        # Universal metadata extraction
        metadata.update(self._extract_universal_metadata(text))
        
        return metadata
    
    def _extract_paper_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata specific to research papers"""
        metadata = {}
        
        # Extract title (usually first line or after specific patterns)
        title_patterns = [
            r'^(.+?)(?:\n|\r\n|\r)',  # First line
            r'(?:Title|TITLE):\s*(.+?)(?:\n|\r\n|\r)',
            r'# (.+?)(?:\n|\r\n|\r)',  # Markdown title
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match and not metadata.get('title'):
                title = match.group(1).strip()
                if len(title) > 10 and len(title) < 200:  # Reasonable title length
                    metadata['title'] = title
                break
        
        # Extract authors
        author_patterns = [
            r'(?:Authors?|BY|By):\s*(.+?)(?:\n|\r\n|\r)',
            r'(?:Authors?|BY|By)\s*\n\s*(.+?)(?:\n|\r\n|\r)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                authors_text = match.group(1).strip()
                # Split authors by common delimiters
                authors = [author.strip() for author in re.split(r'[,;&]|\sand\s', authors_text)]
                metadata['authors'] = [author for author in authors if author and len(author) > 2]
                break
        
        # Extract abstract
        abstract_patterns = [
            r'(?:Abstract|ABSTRACT):\s*(.+?)(?:\n\s*\n|\r\n\s*\r\n)',
            r'(?:Abstract|ABSTRACT)\s*\n\s*(.+?)(?:\n\s*\n|\r\n\s*\r\n)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) > 50:  # Reasonable abstract length
                    metadata['abstract'] = abstract[:1000]  # Limit length
                break
        
        # Extract keywords
        keyword_patterns = [
            r'(?:Keywords?|Key words?):\s*(.+?)(?:\n|\r\n|\r)',
            r'(?:Tags?):\s*(.+?)(?:\n|\r\n|\r)',
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                keywords_text = match.group(1).strip()
                keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text)]
                metadata['keywords'] = [kw for kw in keywords if kw and len(kw) > 1]
                break
        
        return metadata
    
    def _extract_code_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata specific to code content"""
        metadata = {}
        
        # Detect programming language
        language_indicators = {
            'python': [r'def\s+\w+\(', r'import\s+\w+', r'from\s+\w+\s+import', r'if\s+__name__\s*==\s*["\']__main__["\']'],
            'javascript': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*='],
            'java': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'import\s+java\.'],
            'cpp': [r'#include\s*<\w+>', r'int\s+main\s*\(', r'std::', r'namespace\s+\w+'],
            'rust': [r'fn\s+\w+\(', r'use\s+\w+::', r'let\s+mut\s+\w+'],
            'go': [r'func\s+\w+\(', r'package\s+\w+', r'import\s*\('],
        }
        
        detected_languages = []
        for lang, patterns in language_indicators.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text, re.MULTILINE))
            if score > 0:
                detected_languages.append((lang, score))
        
        if detected_languages:
            detected_languages.sort(key=lambda x: x[1], reverse=True)
            metadata['programming_language'] = detected_languages[0][0]
            metadata['language_confidence'] = detected_languages[0][1] / len(language_indicators[detected_languages[0][0]])
        
        # Extract function/class names
        function_patterns = [
            r'def\s+(\w+)\s*\(',  # Python functions
            r'function\s+(\w+)\s*\(',  # JavaScript functions
            r'public\s+(?:static\s+)?(?:\w+\s+)?(\w+)\s*\(',  # Java methods
        ]
        
        functions = []
        for pattern in function_patterns:
            functions.extend(re.findall(pattern, text, re.MULTILINE))
        
        if functions:
            metadata['functions'] = list(set(functions))  # Remove duplicates
        
        return metadata
    
    def _extract_dataset_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata specific to dataset descriptions"""
        metadata = {}
        
        # Extract data format information
        format_indicators = {
            'csv': [r'\.csv', r'comma.separated', r'CSV'],
            'json': [r'\.json', r'JSON', r'\{.*\}'],
            'xml': [r'\.xml', r'XML', r'<\w+>'],
            'parquet': [r'\.parquet', r'Parquet'],
            'hdf5': [r'\.h5', r'\.hdf5', r'HDF5'],
            'netcdf': [r'\.nc', r'NetCDF', r'netCDF'],
        }
        
        detected_formats = []
        for fmt, patterns in format_indicators.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                detected_formats.append(fmt)
        
        if detected_formats:
            metadata['data_formats'] = detected_formats
        
        # Extract size information
        size_patterns = [
            r'(\d+(?:\.\d+)?)\s*(GB|MB|KB|TB|gigabytes?|megabytes?|kilobytes?|terabytes?)',
            r'(\d+(?:,\d+)*)\s*(?:rows?|records?|samples?|observations?)',
            r'(\d+(?:,\d+)*)\s*(?:columns?|features?|variables?)',
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metadata['size_mentions'] = matches
                break
        
        return metadata
    
    def _extract_universal_metadata(self, text: str) -> Dict[str, Any]:
        """Extract universal metadata applicable to all content types"""
        metadata = {}
        
        # Text statistics
        metadata['text_stats'] = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1,
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }
        
        # Language detection (simple heuristic)
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()
        english_score = sum(1 for word in english_indicators if word in words[:100])  # Check first 100 words
        metadata['likely_language'] = 'en' if english_score > 5 else 'unknown'
        
        # Extract URLs and references
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        if urls:
            metadata['urls'] = urls[:10]  # Limit to first 10 URLs
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata['emails'] = emails[:5]  # Limit to first 5 emails
        
        # Content hash for deduplication
        metadata['content_hash'] = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for optimal processing"""
        cleaned = text
        
        if self.config.normalize_whitespace:
            # Normalize whitespace
            cleaned = re.sub(r'\r\n', '\n', cleaned)  # Normalize line endings
            cleaned = re.sub(r'\r', '\n', cleaned)    # Handle old Mac line endings
            cleaned = re.sub(r'\t', ' ', cleaned)     # Convert tabs to spaces
            cleaned = re.sub(r' +', ' ', cleaned)     # Collapse multiple spaces
            
            # Clean up excessive newlines but preserve structure
            if self.config.preserve_structure:
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
            else:
                cleaned = re.sub(r'\n+', ' ', cleaned)  # Replace all newlines with spaces
        
        # Remove or normalize special characters based on content type
        if self.config.content_type == ContentType.CODE:
            # Preserve code structure
            pass
        else:
            # Remove non-printable characters but keep basic punctuation
            cleaned = re.sub(r'[^\x20-\x7E\n]', '', cleaned)
        
        # Remove stopwords if configured
        if self.config.remove_stopwords and self.stop_words:
            words = cleaned.split()
            cleaned = ' '.join(word for word in words if word.lower() not in self.stop_words)
        
        return cleaned.strip()
    
    def _create_chunks(self, text: str, content_id: str) -> List[str]:
        """Create intelligent chunks based on content type and configuration"""
        if self.config.content_type == ContentType.RESEARCH_PAPER:
            return self._chunk_research_paper(text)
        elif self.config.content_type == ContentType.CODE:
            return self._chunk_code(text)
        else:
            return self._chunk_generic_text(text)
    
    def _chunk_research_paper(self, text: str) -> List[str]:
        """Chunk research paper preserving section structure"""
        chunks = []
        
        # Try to split by sections first
        section_patterns = [
            r'\n(?:#{1,3}\s+.+?)(?=\n)',  # Markdown headers
            r'\n(?:\d+\.?\s+[A-Z].+?)(?=\n)',  # Numbered sections
            r'\n(?:[A-Z][A-Z\s]+)(?=\n)',  # ALL CAPS sections
        ]
        
        sections = [text]  # Default to whole text
        for pattern in section_patterns:
            potential_sections = re.split(pattern, text)
            if len(potential_sections) > 1:
                sections = potential_sections
                break
        
        # Process each section
        for section in sections:
            if len(section.strip()) < self.config.min_chunk_size:
                continue
            
            # If section is too large, chunk it further
            if self._estimate_tokens(section) > self.config.max_chunk_size:
                chunks.extend(self._chunk_generic_text(section))
            else:
                chunks.append(section.strip())
        
        return chunks
    
    def _chunk_code(self, text: str) -> List[str]:
        """Chunk code preserving function/class boundaries"""
        chunks = []
        
        # Try to split by functions/classes
        function_patterns = [
            r'\n(?=def\s+\w+)',  # Python functions
            r'\n(?=class\s+\w+)',  # Python classes
            r'\n(?=function\s+\w+)',  # JavaScript functions
            r'\n(?=public\s+(?:static\s+)?(?:class|interface)\s+\w+)',  # Java classes
        ]
        
        sections = [text]
        for pattern in function_patterns:
            potential_sections = re.split(pattern, text)
            if len(potential_sections) > 2:  # Need at least 3 sections to be worth it
                sections = potential_sections
                break
        
        for section in sections:
            if len(section.strip()) < self.config.min_chunk_size:
                continue
            
            if self._estimate_tokens(section) > self.config.max_chunk_size:
                # Split large functions by lines if needed
                lines = section.split('\n')
                current_chunk = []
                current_tokens = 0
                
                for line in lines:
                    line_tokens = self._estimate_tokens(line)
                    if current_tokens + line_tokens > self.config.max_chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_tokens = line_tokens
                    else:
                        current_chunk.append(line)
                        current_tokens += line_tokens
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
            else:
                chunks.append(section.strip())
        
        return chunks
    
    def _chunk_generic_text(self, text: str) -> List[str]:
        """Generic text chunking with overlap"""
        chunks = []
        
        # Use sentence-based chunking if NLTK is available
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = text.split('. ')
        else:
            sentences = text.split('. ')
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence would exceed max size
            if current_tokens + sentence_tokens > self.config.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Add sentences from the end for overlap
                for prev_sentence in reversed(current_chunk):
                    prev_tokens = self._estimate_tokens(prev_sentence)
                    if overlap_tokens + prev_tokens <= self.config.chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _process_chunk(self, text: str, chunk_id: str, chunk_index: int, 
                      parent_content_id: str, parent_metadata: Dict[str, Any]) -> ProcessedChunk:
        """Process individual chunk and create metadata"""
        token_count = self._estimate_tokens(text)
        
        chunk_metadata = {
            'chunk_index': chunk_index,
            'parent_content_id': parent_content_id,
            'token_count': token_count,
            'character_count': len(text),
            'content_type': self.config.content_type.value,
        }
        
        # Add relevant parent metadata
        for key in ['title', 'authors', 'programming_language', 'data_formats']:
            if key in parent_metadata:
                chunk_metadata[key] = parent_metadata[key]
        
        # Add chunk-specific context
        chunk_metadata['chunk_preview'] = text[:100] + "..." if len(text) > 100 else text
        
        return ProcessedChunk(
            text=text,
            chunk_id=chunk_id,
            metadata=chunk_metadata,
            token_count=token_count,
            chunk_index=chunk_index,
            parent_content_id=parent_content_id
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximation for various tokenizers)"""
        # Simple estimation: ~4 characters per token on average
        # This is a reasonable approximation for OpenAI's tokenizers
        return max(1, len(text) // 4)


# Utility functions for easy usage

def create_processor_for_content_type(content_type: ContentType, 
                                    max_chunk_size: int = 512,
                                    chunk_overlap: int = 50) -> ContentTextProcessor:
    """Create a processor optimized for specific content type"""
    config = ProcessingConfig(
        content_type=content_type,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_structure=True,
        extract_metadata=True
    )
    return ContentTextProcessor(config)


def process_research_paper(text: str, content_id: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
    """Quick function to process research paper content"""
    processor = create_processor_for_content_type(ContentType.RESEARCH_PAPER)
    return processor.process_content(text, content_id, metadata)


def process_code_content(text: str, content_id: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
    """Quick function to process code content"""
    processor = create_processor_for_content_type(ContentType.CODE)
    return processor.process_content(text, content_id, metadata)


# Add missing import
import time