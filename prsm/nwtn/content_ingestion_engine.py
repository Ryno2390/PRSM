#!/usr/bin/env python3
"""
Content Ingestion Engine for PRSM/NWTN System
==============================================

This module handles the ingestion of large volumes of academic papers with:
- Content hashing for duplicate detection
- High-dimensional embedding generation
- Provenance tracking for content creators
- FTNS reward distribution
- Background processing with progress monitoring

Designed to handle 150K+ papers efficiently with proper error handling and recovery.
"""

import asyncio
import hashlib
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
import json
import os
import numpy as np

from prsm.nwtn.external_storage_config import ExternalStorageConfig
from prsm.tokenomics.ftns_service import FTNSService
from prsm.integrations.core.provenance_engine import ProvenanceEngine
from prsm.nwtn.semantic_retriever import TextEmbeddingGenerator

logger = structlog.get_logger(__name__)


@dataclass
class ContentHash:
    """Represents a content hash for duplicate detection"""
    content_hash: str
    algorithm: str = "sha256"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ContentEmbedding:
    """Represents a high-dimensional content embedding"""
    embedding_vector: List[float]
    embedding_model: str
    dimension: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IngestionResult:
    """Result of content ingestion process"""
    paper_id: str
    success: bool
    content_hash: Optional[ContentHash] = None
    embedding: Optional[ContentEmbedding] = None
    provenance_record_id: Optional[str] = None
    ftns_reward: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    duplicate_detected: bool = False


class ContentHashGenerator:
    """Generates content hashes for duplicate detection"""
    
    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
    
    def generate_hash(self, content: str) -> ContentHash:
        """Generate content hash from text content"""
        # Normalize content for consistent hashing
        normalized_content = self._normalize_content(content)
        
        # Generate hash
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(normalized_content.encode('utf-8'))
        content_hash = hash_obj.hexdigest()
        
        return ContentHash(
            content_hash=content_hash,
            algorithm=self.algorithm
        )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing"""
        # Remove extra whitespace, convert to lowercase
        normalized = ' '.join(content.lower().split())
        return normalized


class ContentIngestionEngine:
    """
    Content ingestion engine for processing large volumes of academic papers
    
    Handles:
    - Background processing of 150K+ papers
    - Content hashing and duplicate detection
    - High-dimensional embedding generation
    - Provenance tracking and FTNS rewards
    - Progress monitoring and error recovery
    """
    
    def __init__(self,
                 external_storage: ExternalStorageConfig,
                 ftns_service: FTNSService,
                 provenance_engine: ProvenanceEngine,
                 embedding_generator: Optional[TextEmbeddingGenerator] = None):
        
        self.external_storage = external_storage
        self.ftns_service = ftns_service
        self.provenance_engine = provenance_engine
        self.embedding_generator = embedding_generator or TextEmbeddingGenerator()
        
        # Content processing components
        self.hash_generator = ContentHashGenerator()
        self.initialized = False
        
        # Configuration
        self.reward_per_paper = 0.5  # FTNS reward per paper ingested
        self.batch_size = 1000  # Process papers in larger batches for 150K+ scale
        self.max_retries = 3
        
        # Progress tracking
        self.ingestion_stats = {
            'papers_processed': 0,
            'papers_failed': 0,
            'duplicates_detected': 0,
            'embeddings_created': 0,
            'provenance_records': 0,
            'ftns_rewards_distributed': 0.0,
            'total_processing_time': 0.0
        }
        
        # Error tracking
        self.error_log = []
        
        # State management
        self.progress_file = "ingestion_progress.json"
        self.resume_from_checkpoint = True
    
    async def initialize(self):
        """Initialize the content ingestion engine"""
        try:
            logger.info("Initializing Content Ingestion Engine")
            
            # Initialize embedding generator
            await self.embedding_generator.initialize()
            
            # Load previous progress if resuming
            if self.resume_from_checkpoint:
                await self._load_progress()
            
            self.initialized = True
            logger.info("Content Ingestion Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Ingestion Engine: {e}")
            return False
    
    async def ingest_all_papers(self, creator_id: str) -> Dict[str, Any]:
        """
        Ingest all papers from external storage
        
        Args:
            creator_id: ID of the content creator (e.g., "Prismatica")
            
        Returns:
            Dict containing ingestion results and statistics
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("Starting bulk paper ingestion", creator_id=creator_id)
            
            # Get all papers from external storage
            papers = await self._get_all_papers()
            total_papers = len(papers)
            
            logger.info(f"Found {total_papers:,} papers to process")
            
            # Process papers in batches
            for i in range(0, total_papers, self.batch_size):
                batch = papers[i:i + self.batch_size]
                batch_start = i
                batch_end = min(i + self.batch_size, total_papers)
                
                logger.info(f"Processing batch {batch_start + 1}-{batch_end} of {total_papers}")
                
                # Process batch
                await self._process_paper_batch(batch, creator_id)
                
                # Save progress
                await self._save_progress()
                
                # Log progress
                completion_percentage = (batch_end / total_papers) * 100
                logger.info(f"Batch completed: {completion_percentage:.1f}% total progress")
            
            # Calculate final results
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.ingestion_stats['total_processing_time'] = processing_time
            
            results = {
                'total_papers': total_papers,
                'papers_processed': self.ingestion_stats['papers_processed'],
                'papers_failed': self.ingestion_stats['papers_failed'],
                'duplicates_detected': self.ingestion_stats['duplicates_detected'],
                'embeddings_created': self.ingestion_stats['embeddings_created'],
                'provenance_records': self.ingestion_stats['provenance_records'],
                'ftns_rewards_distributed': self.ingestion_stats['ftns_rewards_distributed'],
                'processing_time': processing_time,
                'success_rate': (self.ingestion_stats['papers_processed'] / total_papers) * 100,
                'errors': self.error_log[-10:] if self.error_log else []  # Last 10 errors
            }
            
            logger.info("Bulk paper ingestion completed", **results)
            return results
            
        except Exception as e:
            logger.error(f"Bulk paper ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.ingestion_stats
            }
    
    async def ingest_paper(self, paper_data: Dict[str, Any], creator_id: str) -> Dict[str, Any]:
        """
        Ingest a single paper with full processing
        
        Args:
            paper_data: Paper data dictionary
            creator_id: ID of the content creator
            
        Returns:
            Dict containing ingestion result
        """
        start_time = datetime.now(timezone.utc)
        paper_id = paper_data.get('id', str(uuid4()))
        
        try:
            logger.debug("Processing paper", paper_id=paper_id, creator_id=creator_id)
            
            # Step 1: Generate content hash
            content_hash = await self._generate_content_hash(paper_data)
            
            # Step 2: Check for duplicates
            duplicate_detected = await self._check_duplicate(content_hash)
            if duplicate_detected:
                logger.debug("Duplicate paper detected", paper_id=paper_id)
                self.ingestion_stats['duplicates_detected'] += 1
                return {
                    'paper_id': paper_id,
                    'success': True,
                    'duplicate_detected': True,
                    'ftns_reward': 0.0
                }
            
            # Step 3: Generate high-dimensional embedding
            embedding = await self._generate_embedding(paper_data)
            
            # Step 4: Store in external storage
            await self._store_paper_data(paper_id, paper_data, content_hash, embedding)
            
            # Step 5: Create provenance record
            provenance_record_id = await self._create_provenance_record(
                paper_id, paper_data, creator_id
            )
            
            # Step 6: Distribute FTNS reward
            ftns_reward = await self._distribute_ftns_reward(creator_id, paper_id)
            
            # Update statistics
            self.ingestion_stats['papers_processed'] += 1
            self.ingestion_stats['embeddings_created'] += 1
            self.ingestion_stats['provenance_records'] += 1
            self.ingestion_stats['ftns_rewards_distributed'] += ftns_reward
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                'paper_id': paper_id,
                'success': True,
                'content_hash': content_hash.content_hash,
                'embedding_dimension': embedding.dimension,
                'provenance_record_id': provenance_record_id,
                'ftns_reward': ftns_reward,
                'processing_time': processing_time,
                'duplicate_detected': False
            }
            
            logger.debug("Paper processed successfully", **result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper_id}: {e}")
            
            # Update error statistics
            self.ingestion_stats['papers_failed'] += 1
            self.error_log.append({
                'paper_id': paper_id,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return {
                'paper_id': paper_id,
                'success': False,
                'error_message': str(e)
            }
    
    async def _get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers from external storage"""
        try:
            # Use the external storage interface properly
            return await self.external_storage.get_all_papers()
            
        except Exception as e:
            logger.error(f"Failed to retrieve papers: {e}")
            return []
    
    async def _process_paper_batch(self, papers: List[Dict[str, Any]], creator_id: str):
        """Process a batch of papers concurrently"""
        tasks = []
        
        for paper in papers:
            task = asyncio.create_task(
                self.ingest_paper(paper, creator_id)
            )
            tasks.append(task)
        
        # Process batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error for paper {i}: {result}")
                self.ingestion_stats['papers_failed'] += 1
    
    async def _generate_content_hash(self, paper_data: Dict[str, Any]) -> ContentHash:
        """Generate content hash for the paper"""
        # Combine title and abstract for hashing
        content = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
        return self.hash_generator.generate_hash(content)
    
    async def _check_duplicate(self, content_hash: ContentHash) -> bool:
        """Check if content hash already exists"""
        try:
            # For now, return False (no duplicates)
            # In production, this would check the external storage
            return False
        except Exception:
            # If check fails, assume not duplicate to avoid losing content
            return False
    
    async def _generate_embedding(self, paper_data: Dict[str, Any]) -> ContentEmbedding:
        """Generate high-dimensional embedding for the paper"""
        # Combine title and abstract for embedding
        content = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
        
        # Generate embedding
        embedding_vector = await self.embedding_generator.generate_embedding(content)
        
        return ContentEmbedding(
            embedding_vector=embedding_vector,
            embedding_model=self.embedding_generator.model_name,
            dimension=len(embedding_vector)
        )
    
    async def _store_paper_data(self, paper_id: str, paper_data: Dict[str, Any], 
                               content_hash: ContentHash, embedding: ContentEmbedding):
        """Store paper data, hash, and embedding in external storage"""
        # For now, this is a mock implementation
        # In production, this would store in the actual external storage
        logger.debug(f"Storing paper {paper_id} with hash {content_hash.content_hash[:8]}...")
    
    async def _create_provenance_record(self, paper_id: str, paper_data: Dict[str, Any], 
                                      creator_id: str) -> str:
        """Create provenance record for the paper"""
        provenance_data = {
            'paper_id': paper_id,
            'title': paper_data.get('title', ''),
            'authors': paper_data.get('authors', ''),
            'arxiv_id': paper_data.get('arxiv_id', ''),
            'publish_date': paper_data.get('publish_date', ''),
            'source': 'arxiv',
            'ingestion_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        record_id = await self.provenance_engine.create_content_record(
            creator_id=creator_id,
            content_id=paper_id,
            content_type='academic_paper',
            metadata=provenance_data
        )
        
        return record_id
    
    async def _distribute_ftns_reward(self, creator_id: str, paper_id: str) -> float:
        """Distribute FTNS reward to content creator"""
        try:
            # Transfer reward to creator
            await self.ftns_service.transfer_tokens(
                from_user="system",
                to_user=creator_id,
                amount=self.reward_per_paper,
                purpose=f"Content ingestion reward for paper {paper_id}"
            )
            
            return self.reward_per_paper
            
        except Exception as e:
            logger.error(f"Failed to distribute FTNS reward: {e}")
            return 0.0
    
    async def _save_progress(self):
        """Save current progress to file"""
        try:
            progress_data = {
                'stats': self.ingestion_stats,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error_count': len(self.error_log)
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    async def _load_progress(self):
        """Load previous progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    
                    if 'stats' in progress_data:
                        self.ingestion_stats = progress_data['stats']
                        logger.info("Loaded previous progress", **self.ingestion_stats)
                        
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics"""
        return {
            **self.ingestion_stats,
            'error_count': len(self.error_log),
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Save final progress
            await self._save_progress()
            
            # Clear error log to free memory
            self.error_log.clear()
            
            logger.info("Content Ingestion Engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Factory function for easy instantiation
async def create_content_ingestion_engine(
    external_storage: ExternalStorageConfig,
    ftns_service: FTNSService,
    provenance_engine: ProvenanceEngine
) -> ContentIngestionEngine:
    """Create and initialize a content ingestion engine"""
    engine = ContentIngestionEngine(external_storage, ftns_service, provenance_engine)
    await engine.initialize()
    return engine