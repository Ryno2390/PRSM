"""
Enhanced Meta-Reasoning Engine with Rich Context Aggregation.

This module extends the existing MetaReasoningEngine to include comprehensive
context aggregation and validation, addressing the core problem where 
sophisticated reasoning insights are lost during synthesis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from .reasoning_context_types import (
    RichReasoningContext, EnhancedReasoningResult, ContextValidationResult
)
from .reasoning_context_aggregator import ReasoningContextAggregator
from .context_validation_engine import ContextValidationEngine

# Import the existing meta-reasoning engine (assuming it exists)
try:
    from .meta_reasoning_engine import MetaReasoningEngine
except ImportError:
    # Fallback if the original doesn't exist yet
    class MetaReasoningEngine:
        async def process_query(self, query: str, **kwargs):
            return type('MockResult', (), {
                'parallel_results': {},
                'quality_metrics': {},
                'timing_info': {},
                'summary': 'Mock meta-reasoning result'
            })()


logger = logging.getLogger(__name__)


class EnhancedMetaReasoningEngine(MetaReasoningEngine):
    """
    Extended meta-reasoning engine with comprehensive context aggregation.
    
    This class addresses the core problem where NWTN performs brilliant 
    multi-engine reasoning but produces stilted outputs due to context 
    starvation at the final synthesis stage.
    """
    
    def __init__(self, **kwargs):
        """Initialize the enhanced meta-reasoning engine"""
        super().__init__(**kwargs)
        
        # Initialize context aggregation components
        self.context_aggregator = ReasoningContextAggregator()
        self.context_validator = ContextValidationEngine()
        self.context_persistence = ContextPersistenceManager()
        
        # Configuration for context aggregation
        self.config = {
            'enable_context_aggregation': True,
            'enable_context_validation': True,
            'enable_context_persistence': True,
            'validation_threshold': 0.6,
            'retry_on_low_quality': True,
            'max_retries': 2
        }
        
        logger.info("EnhancedMetaReasoningEngine initialized with context aggregation")
    
    async def process_with_rich_context(self, 
                                      query: str, 
                                      search_corpus: Optional[List[Any]] = None,
                                      user_parameters: Optional[Dict[str, Any]] = None,
                                      **kwargs) -> EnhancedReasoningResult:
        """
        Process query and generate rich reasoning context for enhanced synthesis.
        
        Args:
            query: The user's query
            search_corpus: Optional search corpus used in reasoning
            user_parameters: Optional user parameters for customization
            **kwargs: Additional arguments passed to base meta-reasoning
            
        Returns:
            EnhancedReasoningResult with rich context and validation
        """
        
        logger.info(f"Processing query with rich context: {query[:100]}...")
        processing_start = datetime.now()
        
        try:
            # Step 1: Standard meta-reasoning
            logger.debug("Executing standard meta-reasoning")
            standard_result = await self._execute_standard_reasoning(query, **kwargs)
            
            # Step 2: Context aggregation
            logger.debug("Aggregating rich reasoning context")
            rich_context = await self._aggregate_reasoning_context(
                standard_result, search_corpus, query
            )
            
            # Step 3: Context validation
            logger.debug("Validating context completeness")
            validation_result = await self._validate_context_quality(
                rich_context, standard_result
            )
            
            # Step 4: Quality improvement if needed
            if self.config['retry_on_low_quality'] and validation_result.overall_score < self.config['validation_threshold']:
                logger.info(f"Context quality below threshold ({validation_result.overall_score:.2f}), attempting improvement")
                rich_context = await self._improve_context_quality(
                    rich_context, validation_result, standard_result, search_corpus, query
                )
                # Re-validate after improvement
                validation_result = await self._validate_context_quality(rich_context, standard_result)
            
            # Step 5: Context persistence
            if self.config['enable_context_persistence']:
                context_id = await self._persist_reasoning_context(rich_context, validation_result)
                processing_metadata = {'context_id': context_id}
            else:
                processing_metadata = {}
            
            # Add processing timing
            processing_time = (datetime.now() - processing_start).total_seconds()
            processing_metadata.update({
                'total_processing_time': processing_time,
                'context_aggregation_enabled': True,
                'context_validation_score': validation_result.overall_score
            })
            
            # Create enhanced result
            enhanced_result = EnhancedReasoningResult(
                standard_result=standard_result,
                rich_context=rich_context,
                context_validation=validation_result,
                processing_metadata=processing_metadata
            )
            
            logger.info(f"Enhanced reasoning completed in {processing_time:.2f}s "
                       f"with context quality {validation_result.overall_score:.2f}")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced meta-reasoning: {e}")
            # Fallback to standard reasoning with minimal context
            standard_result = await self._execute_standard_reasoning(query, **kwargs)
            
            return EnhancedReasoningResult(
                standard_result=standard_result,
                rich_context=RichReasoningContext(
                    original_query=query,
                    metadata={'error': str(e), 'fallback_mode': True}
                ),
                context_validation=ContextValidationResult(
                    overall_score=0.0,
                    validation_checks={},
                    recommendations=[f"Processing failed: {str(e)}"],
                    missing_components=["Full context unavailable due to error"],
                    quality_issues=[f"Error: {str(e)}"]
                ),
                processing_metadata={'error': str(e), 'fallback_mode': True}
            )
    
    # ========================================================================
    # Core Processing Methods
    # ========================================================================
    
    async def _execute_standard_reasoning(self, query: str, **kwargs) -> Any:
        """Execute standard meta-reasoning process"""
        
        # Call the parent class's process_query method
        return await super().process_query(query, **kwargs)
    
    async def _aggregate_reasoning_context(self, 
                                         standard_result: Any,
                                         search_corpus: Optional[List[Any]],
                                         query: str) -> RichReasoningContext:
        """Aggregate rich reasoning context from standard result"""
        
        if not self.config['enable_context_aggregation']:
            # Return minimal context if aggregation is disabled
            return RichReasoningContext(
                original_query=query,
                metadata={'context_aggregation_disabled': True}
            )
        
        try:
            rich_context = await self.context_aggregator.aggregate_reasoning_context(
                standard_result, search_corpus, query
            )
            
            logger.debug(f"Context aggregation completed with {len(rich_context.engine_insights)} engine insights")
            return rich_context
            
        except Exception as e:
            logger.error(f"Context aggregation failed: {e}")
            # Return minimal context with error info
            return RichReasoningContext(
                original_query=query,
                metadata={'aggregation_error': str(e), 'partial_context': True}
            )
    
    async def _validate_context_quality(self, 
                                       rich_context: RichReasoningContext,
                                       standard_result: Any) -> ContextValidationResult:
        """Validate the quality of aggregated context"""
        
        if not self.config['enable_context_validation']:
            # Return default validation result if validation is disabled
            return ContextValidationResult(
                overall_score=1.0,
                validation_checks={'validation_disabled': 1.0},
                recommendations=[],
                missing_components=[],
                quality_issues=[]
            )
        
        try:
            validation_result = await self.context_validator.validate_context_completeness(
                rich_context, standard_result
            )
            
            logger.debug(f"Context validation completed with score {validation_result.overall_score:.2f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Context validation failed: {e}")
            return ContextValidationResult(
                overall_score=0.0,
                validation_checks={},
                recommendations=[f"Validation failed: {str(e)}"],
                missing_components=["Validation unavailable"],
                quality_issues=[f"Validation error: {str(e)}"]
            )
    
    async def _improve_context_quality(self,
                                     rich_context: RichReasoningContext,
                                     validation_result: ContextValidationResult,
                                     standard_result: Any,
                                     search_corpus: Optional[List[Any]],
                                     query: str) -> RichReasoningContext:
        """Attempt to improve context quality based on validation results"""
        
        logger.info("Attempting to improve context quality based on validation feedback")
        
        # Analyze validation issues and attempt targeted improvements
        improved_context = rich_context
        
        try:
            # Check for specific improvement opportunities
            validation_checks = validation_result.validation_checks
            
            # 1. Improve engine coverage if needed
            if validation_checks.get('engine_coverage', 0) < 0.5:
                improved_context = await self._enhance_engine_coverage(
                    improved_context, standard_result
                )
            
            # 2. Enhance insight preservation if needed
            if validation_checks.get('insight_preservation', 0) < 0.6:
                improved_context = await self._enhance_insight_preservation(
                    improved_context, standard_result
                )
            
            # 3. Improve analogical richness if needed
            if validation_checks.get('analogical_richness', 0) < 0.4:
                improved_context = await self._enhance_analogical_analysis(
                    improved_context, standard_result
                )
            
            logger.debug("Context improvement attempts completed")
            return improved_context
            
        except Exception as e:
            logger.error(f"Context improvement failed: {e}")
            # Return original context if improvement fails
            return rich_context
    
    async def _persist_reasoning_context(self,
                                       rich_context: RichReasoningContext,
                                       validation_result: ContextValidationResult) -> str:
        """Persist reasoning context for later retrieval"""
        
        try:
            context_id = await self.context_persistence.store_reasoning_context(
                rich_context, validation_result
            )
            
            logger.debug(f"Context persisted with ID: {context_id}")
            return context_id
            
        except Exception as e:
            logger.error(f"Context persistence failed: {e}")
            return f"persistence_failed_{datetime.now().timestamp()}"
    
    # ========================================================================
    # Context Improvement Methods
    # ========================================================================
    
    async def _enhance_engine_coverage(self,
                                     rich_context: RichReasoningContext,
                                     standard_result: Any) -> RichReasoningContext:
        """Enhance engine coverage by extracting additional insights"""
        
        # Try to extract insights from engines that might have been missed
        parallel_results = getattr(standard_result, 'parallel_results', {})
        
        for engine_name, result in parallel_results.items():
            # Check if we missed this engine in original aggregation
            engine_type = self.context_aggregator._map_engine_name_to_type(engine_name)
            
            if engine_type and engine_type not in rich_context.engine_insights:
                try:
                    # Attempt to extract insight using appropriate analyzer
                    if engine_type in self.context_aggregator.engine_analyzers:
                        analyzer = self.context_aggregator.engine_analyzers[engine_type]
                        insight = await analyzer.extract_insight(result)
                        rich_context.engine_insights[engine_type] = insight
                        logger.debug(f"Added missing insight for {engine_name}")
                except Exception as e:
                    logger.warning(f"Failed to add insight for {engine_name}: {e}")
        
        return rich_context
    
    async def _enhance_insight_preservation(self,
                                          rich_context: RichReasoningContext,
                                          standard_result: Any) -> RichReasoningContext:
        """Enhance insight preservation by extracting additional details"""
        
        # Try to extract more detailed insights from existing engine results
        for engine_type, insight in rich_context.engine_insights.items():
            try:
                # Look for additional findings in the original result
                parallel_results = getattr(standard_result, 'parallel_results', {})
                
                for engine_name, result in parallel_results.items():
                    mapped_type = self.context_aggregator._map_engine_name_to_type(engine_name)
                    
                    if mapped_type == engine_type:
                        # Try to extract additional findings
                        additional_findings = self._extract_additional_findings(result)
                        insight.primary_findings.extend(additional_findings)
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to enhance insights for {engine_type}: {e}")
        
        return rich_context
    
    async def _enhance_analogical_analysis(self,
                                         rich_context: RichReasoningContext,
                                         standard_result: Any) -> RichReasoningContext:
        """Enhance analogical analysis by finding additional connections"""
        
        try:
            # Use the analogical context analyzer to find more connections
            analogical_analyzer = self.context_aggregator.engine_analyzers.get(
                'analogical', None
            )
            
            if analogical_analyzer:
                additional_connections = await analogical_analyzer.extract_analogical_connections(
                    standard_result
                )
                
                # Add unique connections that weren't found before
                existing_connections = {
                    (conn.source_domain, conn.target_domain) 
                    for conn in rich_context.analogical_connections
                }
                
                for connection in additional_connections:
                    key = (connection.source_domain, connection.target_domain)
                    if key not in existing_connections:
                        rich_context.analogical_connections.append(connection)
                        logger.debug(f"Added analogical connection: {key}")
                        
        except Exception as e:
            logger.warning(f"Failed to enhance analogical analysis: {e}")
        
        return rich_context
    
    def _extract_additional_findings(self, engine_result: Any) -> List[str]:
        """Extract additional findings from engine result"""
        
        additional_findings = []
        
        # Try various attributes that might contain additional insights
        potential_attributes = ['insights', 'secondary_findings', 'implications', 'observations']
        
        for attr in potential_attributes:
            if hasattr(engine_result, attr):
                value = getattr(engine_result, attr)
                if isinstance(value, list):
                    additional_findings.extend(value)
                elif isinstance(value, str):
                    additional_findings.append(value)
            elif isinstance(engine_result, dict) and attr in engine_result:
                value = engine_result[attr]
                if isinstance(value, list):
                    additional_findings.extend(value)
                elif isinstance(value, str):
                    additional_findings.append(value)
        
        return additional_findings
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    async def get_context_by_id(self, context_id: str) -> Optional[RichReasoningContext]:
        """Retrieve persisted reasoning context by ID"""
        
        try:
            return await self.context_persistence.retrieve_reasoning_context(context_id)
        except Exception as e:
            logger.error(f"Failed to retrieve context {context_id}: {e}")
            return None
    
    def get_context_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about context aggregation performance"""
        
        return {
            'aggregator_initialized': hasattr(self, 'context_aggregator'),
            'validator_initialized': hasattr(self, 'context_validator'),
            'persistence_enabled': self.config['enable_context_persistence'],
            'validation_threshold': self.config['validation_threshold'],
            'retry_enabled': self.config['retry_on_low_quality']
        }
    
    def configure_context_aggregation(self, **config_updates):
        """Update context aggregation configuration"""
        
        self.config.update(config_updates)
        logger.info(f"Context aggregation configuration updated: {config_updates}")


# ============================================================================
# Context Persistence Manager
# ============================================================================

class ContextPersistenceManager:
    """Manages persistence and retrieval of rich reasoning contexts"""
    
    def __init__(self):
        """Initialize the context persistence manager"""
        self.storage_backend = 'file'  # Could be 'database', 'redis', etc.
        self.storage_path = '/tmp/nwtn_contexts'  # Configurable
        
        # Ensure storage directory exists
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"ContextPersistenceManager initialized with {self.storage_backend} backend")
    
    async def store_reasoning_context(self,
                                    rich_context: RichReasoningContext,
                                    validation_result: Optional[ContextValidationResult] = None) -> str:
        """Store rich context and return context ID"""
        
        import json
        import uuid
        
        # Generate unique context ID
        context_id = f"ctx_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
        
        try:
            # Prepare context for serialization
            context_data = {
                'context_id': context_id,
                'rich_context': self._serialize_rich_context(rich_context),
                'validation_result': self._serialize_validation_result(validation_result) if validation_result else None,
                'stored_at': datetime.now().isoformat(),
                'storage_version': '1.0'
            }
            
            # Store to file (could be replaced with database storage)
            storage_file = f"{self.storage_path}/{context_id}.json"
            with open(storage_file, 'w') as f:
                json.dump(context_data, f, indent=2, default=str)
            
            logger.debug(f"Context stored with ID: {context_id}")
            return context_id
            
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            raise
    
    async def retrieve_reasoning_context(self, context_id: str) -> RichReasoningContext:
        """Retrieve rich context by ID"""
        
        import json
        
        try:
            storage_file = f"{self.storage_path}/{context_id}.json"
            
            with open(storage_file, 'r') as f:
                context_data = json.load(f)
            
            # Deserialize rich context
            rich_context = self._deserialize_rich_context(context_data['rich_context'])
            
            logger.debug(f"Context retrieved with ID: {context_id}")
            return rich_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context {context_id}: {e}")
            raise
    
    def _serialize_rich_context(self, rich_context: RichReasoningContext) -> Dict[str, Any]:
        """Serialize rich context to JSON-compatible format"""
        
        # Convert rich context to dictionary
        # This is a simplified serialization - would need more sophisticated handling for complex objects
        return {
            'original_query': rich_context.original_query,
            'processing_timestamp': rich_context.processing_timestamp.isoformat(),
            'num_engine_insights': len(rich_context.engine_insights),
            'num_synthesis_patterns': len(rich_context.synthesis_patterns),
            'num_analogical_connections': len(rich_context.analogical_connections),
            'confidence_analysis_present': rich_context.confidence_analysis is not None,
            'breakthrough_analysis_present': rich_context.breakthrough_analysis is not None,
            'metadata': rich_context.metadata
        }
    
    def _serialize_validation_result(self, validation_result: ContextValidationResult) -> Dict[str, Any]:
        """Serialize validation result to JSON-compatible format"""
        
        return {
            'overall_score': validation_result.overall_score,
            'validation_checks': validation_result.validation_checks,
            'recommendations': validation_result.recommendations,
            'missing_components': validation_result.missing_components,
            'quality_issues': validation_result.quality_issues,
            'timestamp': validation_result.timestamp.isoformat()
        }
    
    def _deserialize_rich_context(self, context_data: Dict[str, Any]) -> RichReasoningContext:
        """Deserialize rich context from JSON-compatible format"""
        
        # This is a simplified deserialization - would need full reconstruction for production
        return RichReasoningContext(
            original_query=context_data['original_query'],
            processing_timestamp=datetime.fromisoformat(context_data['processing_timestamp']),
            metadata=context_data.get('metadata', {})
        )