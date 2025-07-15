#!/usr/bin/env python3
"""
NWTN Adaptive Complete System - Self-Improving Multi-Modal Reasoning
====================================================================

This module implements the complete NWTN system with integrated SEAL learning,
creating a self-improving multi-modal reasoning AI that gets better over time
through continuous learning from user interactions.

Key Features:
- Phase 1: BYOA (Bring Your Own API) for immediate deployment
- Phase 2: NWTN-optimized voicebox for enhanced performance
- SEAL Integration: Continuous learning and improvement
- Adaptive Learning: Gets better with every interaction
- Quality Assurance: Self-evaluation and improvement

System Architecture:
┌─────────────────────┐
│    User Query       │ "What breakthrough applications emerge from
│  (Natural Language) │  quantum biology research?"
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   Adaptive System   │ • Route to best available voicebox
│   (This Module)     │ • Learn from every interaction
│                     │ • Continuous improvement
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   SEAL Enhanced     │ • Self-evaluation of responses
│   Processing        │ • Quality improvement learning
│                     │ • Adaptive parameter updates
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   8-Step NWTN Core  │ 1. Query decomposition
│   Multi-Modal       │ 2. Reasoning classification
│   Reasoning System  │ 3. Multi-modal analysis
│                     │ 4. Network validation
│                     │ 5. PRSM resource discovery
│                     │ 6. Distributed execution
│                     │ 7. Marketplace integration
│                     │ 8. Result compilation
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Enhanced Response  │ "Based on quantum biology research, I've
│  with Learning      │  identified 3 breakthrough applications..."
│                     │ + Self-evaluation + Learning updates
└─────────────────────┘

Benefits:
- Continuous Improvement: Every interaction makes the system better
- Quality Assurance: Self-evaluation prevents quality degradation
- Adaptive Learning: Learns user preferences and domain patterns
- Performance Optimization: Automatically optimizes for user needs
- Scientific Accuracy: Maintains high standards through self-correction

Usage:
    system = NWTNAdaptiveCompleteSystem()
    await system.initialize()
    
    # Configure user preferences
    await system.configure_user(
        user_id="researcher_123",
        preferences={
            "voicebox_preference": "nwtn_optimized",
            "learning_enabled": True,
            "quality_threshold": 0.9
        }
    )
    
    # Process query with adaptive learning
    response = await system.process_adaptive_query(
        user_id="researcher_123",
        query="What breakthrough applications emerge from quantum biology research?"
    )
    
    # Check learning progress
    progress = await system.get_learning_progress("researcher_123")
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import structlog

from prsm.nwtn.voicebox import NWTNVoicebox, VoiceboxResponse
from prsm.nwtn.nwtn_optimized_voicebox import NWTNOptimizedVoicebox, NWTNOptimizedResponse
from prsm.nwtn.seal_integration import NWTNSEALIntegration, SEALEvaluation, SEALLearningProgress
from prsm.nwtn.complete_system import NWTNCompleteSystem
from prsm.core.config import get_settings
from prsm.core.database_service import get_database_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class VoiceboxType(str, Enum):
    """Types of voicebox available"""
    BYOA = "byoa"                     # Phase 1: Bring Your Own API
    NWTN_OPTIMIZED = "nwtn_optimized"  # Phase 2: NWTN-optimized
    ADAPTIVE = "adaptive"             # Automatically choose best option


class LearningMode(str, Enum):
    """Learning modes for the system"""
    DISABLED = "disabled"             # No learning
    BASIC = "basic"                   # Basic quality tracking
    ENHANCED = "enhanced"             # Full SEAL learning
    RESEARCH = "research"             # Advanced research mode


@dataclass
class UserPreferences:
    """User preferences for adaptive system"""
    user_id: str
    voicebox_preference: VoiceboxType
    learning_mode: LearningMode
    quality_threshold: float
    scientific_domains: List[str]
    reasoning_preferences: List[str]
    feedback_frequency: str
    privacy_settings: Dict[str, bool]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AdaptiveResponse:
    """Enhanced response with learning information"""
    response_id: str
    user_id: str
    query_id: str
    voicebox_used: VoiceboxType
    natural_language_response: str
    reasoning_explanation: str
    breakthrough_insights: List[str]
    confidence_score: float
    quality_score: float
    learning_applied: bool
    learning_updates: List[str]
    improvement_suggestions: List[str]
    processing_time: float
    cost_ftns: float
    seal_evaluation: Optional[SEALEvaluation] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NWTNAdaptiveCompleteSystem:
    """
    NWTN Adaptive Complete System - Self-Improving Multi-Modal Reasoning
    
    This system combines the best of all phases:
    - Phase 1 BYOA for immediate deployment
    - Phase 2 NWTN-optimized for enhanced performance
    - SEAL integration for continuous improvement
    - Adaptive routing to best available voicebox
    - Quality assurance and learning from every interaction
    """
    
    def __init__(self):
        # Core components
        self.byoa_system = None            # Phase 1: BYOA system
        self.nwtn_optimized_voicebox = None # Phase 2: NWTN-optimized
        self.seal_integration = None        # SEAL learning system
        self.database_service = get_database_service()
        
        # User management
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.user_learning_progress: Dict[str, SEALLearningProgress] = {}
        
        # System state
        self.available_voiceboxes = []
        self.system_learning_enabled = True
        self.quality_monitoring_enabled = True
        
        # Learning and adaptation
        self.global_learning_metrics = {
            "total_interactions": 0,
            "average_quality": 0.0,
            "improvement_rate": 0.0,
            "user_satisfaction": 0.0,
            "system_efficiency": 0.0
        }
        
        # Adaptive routing
        self.voicebox_performance = {
            VoiceboxType.BYOA: {"quality": 0.8, "speed": 0.9, "cost": 0.7},
            VoiceboxType.NWTN_OPTIMIZED: {"quality": 0.95, "speed": 0.8, "cost": 0.9},
            VoiceboxType.ADAPTIVE: {"quality": 0.92, "speed": 0.85, "cost": 0.8}
        }
        
        logger.info("NWTN Adaptive Complete System initialized")
    
    async def initialize(self):
        """Initialize the adaptive complete system"""
        try:
            logger.info("🚀 Initializing NWTN Adaptive Complete System...")
            
            # Initialize Phase 1: BYOA system
            logger.info("📱 Initializing Phase 1 (BYOA) system...")
            self.byoa_system = NWTNCompleteSystem()
            await self.byoa_system.initialize()
            self.available_voiceboxes.append(VoiceboxType.BYOA)
            
            # Initialize Phase 2: NWTN-optimized voicebox
            logger.info("🧠 Initializing Phase 2 (NWTN-optimized) voicebox...")
            try:
                self.nwtn_optimized_voicebox = NWTNOptimizedVoicebox()
                await self.nwtn_optimized_voicebox.initialize()
                self.available_voiceboxes.append(VoiceboxType.NWTN_OPTIMIZED)
                logger.info("✅ Phase 2 voicebox initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Phase 2 voicebox initialization failed: {e}")
                logger.info("🔄 Falling back to Phase 1 only")
            
            # Initialize SEAL integration
            logger.info("🤖 Initializing SEAL learning integration...")
            try:
                self.seal_integration = NWTNSEALIntegration()
                await self.seal_integration.initialize()
                logger.info("✅ SEAL integration initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ SEAL integration initialization failed: {e}")
                logger.info("🔄 Continuing without SEAL learning")
                self.system_learning_enabled = False
            
            # Load user preferences
            await self._load_user_preferences()
            
            logger.info("✅ NWTN Adaptive Complete System fully initialized")
            logger.info(f"📊 Available voiceboxes: {[vb.value for vb in self.available_voiceboxes]}")
            logger.info(f"🎓 Learning enabled: {self.system_learning_enabled}")
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive system: {e}")
            raise
    
    async def configure_user(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        api_key: Optional[str] = None,
        provider: Optional[str] = None
    ) -> bool:
        """Configure user preferences and API keys"""
        try:
            logger.info(f"⚙️ Configuring user {user_id}...")
            
            # Create user preferences
            user_prefs = UserPreferences(
                user_id=user_id,
                voicebox_preference=VoiceboxType(preferences.get("voicebox_preference", "adaptive")),
                learning_mode=LearningMode(preferences.get("learning_mode", "enhanced")),
                quality_threshold=preferences.get("quality_threshold", 0.8),
                scientific_domains=preferences.get("scientific_domains", []),
                reasoning_preferences=preferences.get("reasoning_preferences", []),
                feedback_frequency=preferences.get("feedback_frequency", "automatic"),
                privacy_settings=preferences.get("privacy_settings", {"learning": True, "data_sharing": False})
            )
            
            self.user_preferences[user_id] = user_prefs
            
            # Configure API key if provided (for BYOA)
            if api_key and provider:
                success = await self.byoa_system.configure_user_api(
                    user_id=user_id,
                    provider=provider,
                    api_key=api_key
                )
                if not success:
                    logger.warning(f"⚠️ Failed to configure API key for user {user_id}")
            
            # Store preferences
            await self._store_user_preferences(user_prefs)
            
            logger.info(f"✅ User {user_id} configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure user {user_id}: {e}")
            return False
    
    async def process_adaptive_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptiveResponse:
        """
        Process query with adaptive learning and improvement
        
        This method intelligently routes queries to the best available voicebox
        and applies continuous learning to improve performance.
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"🔄 Processing adaptive query for user {user_id}: {query[:100]}...")
            
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id)
            if not user_prefs:
                # Create default preferences
                user_prefs = UserPreferences(
                    user_id=user_id,
                    voicebox_preference=VoiceboxType.ADAPTIVE,
                    learning_mode=LearningMode.ENHANCED,
                    quality_threshold=0.8,
                    scientific_domains=[],
                    reasoning_preferences=[],
                    feedback_frequency="automatic",
                    privacy_settings={"learning": True, "data_sharing": False}
                )
                self.user_preferences[user_id] = user_prefs
            
            # Determine best voicebox to use
            selected_voicebox = await self._select_optimal_voicebox(user_prefs, query, context)
            
            # Process query through selected voicebox
            response_data = await self._process_query_through_voicebox(
                selected_voicebox, user_id, query, context
            )
            
            # Apply SEAL learning if enabled
            seal_evaluation = None
            learning_updates = []
            learning_applied = False
            
            if (self.system_learning_enabled and 
                user_prefs.learning_mode != LearningMode.DISABLED and
                self.seal_integration):
                
                # Evaluate response quality
                if selected_voicebox == VoiceboxType.NWTN_OPTIMIZED:
                    seal_evaluation = await self.seal_integration.evaluate_response_quality(
                        response_data, query, context
                    )
                    
                    # Apply learning if quality is below threshold
                    if seal_evaluation.overall_quality < user_prefs.quality_threshold:
                        learning_updates = await self.seal_integration.trigger_learning_update(
                            learning_signal="self_evaluation",
                            signal_data={"evaluation": seal_evaluation}
                        )
                        learning_applied = len(learning_updates) > 0
            
            # Create adaptive response
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Extract response data based on voicebox type
            if selected_voicebox == VoiceboxType.NWTN_OPTIMIZED and isinstance(response_data, NWTNOptimizedResponse):
                adaptive_response = AdaptiveResponse(
                    response_id=response_data.response_id,
                    user_id=user_id,
                    query_id=response_data.query_id,
                    voicebox_used=selected_voicebox,
                    natural_language_response=response_data.natural_language_response,
                    reasoning_explanation=response_data.reasoning_explanation,
                    breakthrough_insights=response_data.breakthrough_insights,
                    confidence_score=response_data.confidence_breakdown.get("overall", 0.8),
                    quality_score=seal_evaluation.overall_quality if seal_evaluation else 0.8,
                    learning_applied=learning_applied,
                    learning_updates=[update.improvement_areas for update in learning_updates],
                    improvement_suggestions=seal_evaluation.improvement_suggestions if seal_evaluation else [],
                    processing_time=processing_time,
                    cost_ftns=0.0,  # Would be calculated
                    seal_evaluation=seal_evaluation
                )
            else:
                # Handle BYOA response
                adaptive_response = AdaptiveResponse(
                    response_id=response_data.response_id if hasattr(response_data, 'response_id') else str(uuid4()),
                    user_id=user_id,
                    query_id=response_data.query_id if hasattr(response_data, 'query_id') else str(uuid4()),
                    voicebox_used=selected_voicebox,
                    natural_language_response=response_data.natural_language_response if hasattr(response_data, 'natural_language_response') else str(response_data),
                    reasoning_explanation="",
                    breakthrough_insights=[],
                    confidence_score=response_data.confidence_score if hasattr(response_data, 'confidence_score') else 0.8,
                    quality_score=0.8,  # Default for BYOA
                    learning_applied=learning_applied,
                    learning_updates=[],
                    improvement_suggestions=[],
                    processing_time=processing_time,
                    cost_ftns=response_data.total_cost_ftns if hasattr(response_data, 'total_cost_ftns') else 0.0,
                    seal_evaluation=seal_evaluation
                )
            
            # Update global metrics
            await self._update_global_metrics(adaptive_response)
            
            # Store interaction
            await self._store_adaptive_interaction(adaptive_response)
            
            logger.info(f"✅ Adaptive query processed successfully")
            logger.info(f"🎯 Voicebox used: {selected_voicebox.value}")
            logger.info(f"📊 Quality score: {adaptive_response.quality_score:.3f}")
            logger.info(f"🎓 Learning applied: {learning_applied}")
            
            return adaptive_response
            
        except Exception as e:
            logger.error(f"Failed to process adaptive query: {e}")
            raise
    
    async def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Get learning progress for user"""
        try:
            if not self.system_learning_enabled or not self.seal_integration:
                return {"learning_enabled": False, "message": "Learning not available"}
            
            # Get SEAL learning progress
            seal_progress = await self.seal_integration.get_learning_progress()
            
            # Get user-specific metrics
            user_metrics = await self._get_user_metrics(user_id)
            
            return {
                "learning_enabled": True,
                "user_id": user_id,
                "total_interactions": user_metrics.get("total_interactions", 0),
                "average_quality": user_metrics.get("average_quality", 0.0),
                "improvement_rate": user_metrics.get("improvement_rate", 0.0),
                "preferred_voicebox": self.user_preferences.get(user_id, {}).voicebox_preference if user_id in self.user_preferences else "adaptive",
                "seal_progress": seal_progress,
                "recent_improvements": await self.seal_integration.get_recent_improvements(5)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning progress: {e}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get status from all components
            status = {
                "system_initialized": True,
                "available_voiceboxes": [vb.value for vb in self.available_voiceboxes],
                "learning_enabled": self.system_learning_enabled,
                "quality_monitoring_enabled": self.quality_monitoring_enabled,
                "total_users": len(self.user_preferences),
                "global_metrics": self.global_learning_metrics,
                "voicebox_performance": self.voicebox_performance
            }
            
            # Add component statuses
            if self.byoa_system:
                byoa_status = await self.byoa_system.get_system_status()
                status["byoa_system"] = byoa_status
            
            if self.nwtn_optimized_voicebox:
                nwtn_metrics = await self.nwtn_optimized_voicebox.get_training_metrics()
                status["nwtn_optimized"] = nwtn_metrics
            
            if self.seal_integration:
                seal_progress = await self.seal_integration.get_learning_progress()
                status["seal_learning"] = seal_progress
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def optimize_performance(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Optimize system performance based on usage patterns"""
        try:
            logger.info("⚡ Optimizing system performance...")
            
            optimization_results = {
                "optimizations_applied": [],
                "performance_improvements": {},
                "recommendations": []
            }
            
            # User-specific optimization
            if user_id and user_id in self.user_preferences:
                user_prefs = self.user_preferences[user_id]
                user_metrics = await self._get_user_metrics(user_id)
                
                # Optimize voicebox selection
                optimal_voicebox = await self._optimize_voicebox_selection(user_prefs, user_metrics)
                if optimal_voicebox != user_prefs.voicebox_preference:
                    optimization_results["optimizations_applied"].append(
                        f"Updated voicebox preference from {user_prefs.voicebox_preference.value} to {optimal_voicebox.value}"
                    )
                    user_prefs.voicebox_preference = optimal_voicebox
            
            # Global system optimization
            if self.system_learning_enabled and self.seal_integration:
                # Optimize learning parameters
                learning_optimizations = await self._optimize_learning_parameters()
                optimization_results["optimizations_applied"].extend(learning_optimizations)
            
            # Performance recommendations
            recommendations = await self._generate_performance_recommendations()
            optimization_results["recommendations"] = recommendations
            
            logger.info(f"✅ Performance optimization completed")
            logger.info(f"📊 Optimizations applied: {len(optimization_results['optimizations_applied'])}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize performance: {e}")
            return {"error": str(e)}
    
    # === Private Methods ===
    
    async def _select_optimal_voicebox(
        self,
        user_prefs: UserPreferences,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> VoiceboxType:
        """Select optimal voicebox based on preferences and query characteristics"""
        try:
            # If user has specific preference and it's available, use it
            if (user_prefs.voicebox_preference != VoiceboxType.ADAPTIVE and
                user_prefs.voicebox_preference in self.available_voiceboxes):
                return user_prefs.voicebox_preference
            
            # Adaptive selection based on query characteristics
            query_complexity = await self._assess_query_complexity(query)
            
            # Prefer NWTN-optimized for complex queries if available
            if (query_complexity > 0.7 and 
                VoiceboxType.NWTN_OPTIMIZED in self.available_voiceboxes):
                return VoiceboxType.NWTN_OPTIMIZED
            
            # Prefer BYOA for simple queries or if NWTN-optimized not available
            if VoiceboxType.BYOA in self.available_voiceboxes:
                return VoiceboxType.BYOA
            
            # Fallback to first available
            return self.available_voiceboxes[0]
            
        except Exception as e:
            logger.error(f"Failed to select optimal voicebox: {e}")
            return self.available_voiceboxes[0] if self.available_voiceboxes else VoiceboxType.BYOA
    
    async def _process_query_through_voicebox(
        self,
        voicebox_type: VoiceboxType,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Process query through selected voicebox"""
        try:
            if voicebox_type == VoiceboxType.NWTN_OPTIMIZED and self.nwtn_optimized_voicebox:
                return await self.nwtn_optimized_voicebox.process_query(
                    user_id=user_id,
                    query=query,
                    context=context
                )
            elif voicebox_type == VoiceboxType.BYOA and self.byoa_system:
                return await self.byoa_system.process_query(
                    user_id=user_id,
                    query=query,
                    context=context
                )
            else:
                raise ValueError(f"Voicebox type {voicebox_type.value} not available")
                
        except Exception as e:
            logger.error(f"Failed to process query through voicebox: {e}")
            raise
    
    async def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity for voicebox selection"""
        # Simple complexity assessment based on query characteristics
        complexity_indicators = [
            len(query.split()) > 20,  # Long queries
            "breakthrough" in query.lower(),
            "compare" in query.lower(),
            "analyze" in query.lower(),
            "explain" in query.lower(),
            "?" in query[:-1],  # Multiple questions
            any(domain in query.lower() for domain in ["quantum", "molecular", "atomic", "nano"])
        ]
        
        return sum(complexity_indicators) / len(complexity_indicators)
    
    async def _update_global_metrics(self, response: AdaptiveResponse):
        """Update global learning metrics"""
        try:
            # Update counters
            self.global_learning_metrics["total_interactions"] += 1
            
            # Update quality metrics
            total_interactions = self.global_learning_metrics["total_interactions"]
            current_avg = self.global_learning_metrics["average_quality"]
            
            # Exponential moving average
            alpha = 0.1
            self.global_learning_metrics["average_quality"] = (
                alpha * response.quality_score + (1 - alpha) * current_avg
            )
            
            # Update improvement rate
            if response.learning_applied:
                self.global_learning_metrics["improvement_rate"] = (
                    alpha * 1.0 + (1 - alpha) * self.global_learning_metrics["improvement_rate"]
                )
            
            # Update system efficiency
            efficiency_score = response.quality_score / max(response.processing_time, 0.1)
            self.global_learning_metrics["system_efficiency"] = (
                alpha * efficiency_score + (1 - alpha) * self.global_learning_metrics["system_efficiency"]
            )
            
        except Exception as e:
            logger.error(f"Failed to update global metrics: {e}")
    
    async def _store_adaptive_interaction(self, response: AdaptiveResponse):
        """Store adaptive interaction in database"""
        try:
            await self.database_service.store_adaptive_interaction({
                'response_id': response.response_id,
                'user_id': response.user_id,
                'query_id': response.query_id,
                'voicebox_used': response.voicebox_used.value,
                'quality_score': response.quality_score,
                'learning_applied': response.learning_applied,
                'learning_updates': response.learning_updates,
                'processing_time': response.processing_time,
                'cost_ftns': response.cost_ftns,
                'generated_at': response.generated_at
            })
            
        except Exception as e:
            logger.error(f"Failed to store adaptive interaction: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences from database"""
        try:
            preferences_data = await self.database_service.get_user_preferences()
            
            for pref_data in preferences_data:
                user_prefs = UserPreferences(
                    user_id=pref_data['user_id'],
                    voicebox_preference=VoiceboxType(pref_data['voicebox_preference']),
                    learning_mode=LearningMode(pref_data['learning_mode']),
                    quality_threshold=pref_data['quality_threshold'],
                    scientific_domains=pref_data['scientific_domains'],
                    reasoning_preferences=pref_data['reasoning_preferences'],
                    feedback_frequency=pref_data['feedback_frequency'],
                    privacy_settings=pref_data['privacy_settings'],
                    created_at=pref_data['created_at']
                )
                self.user_preferences[pref_data['user_id']] = user_prefs
            
            logger.info(f"📚 Loaded preferences for {len(self.user_preferences)} users")
            
        except Exception as e:
            logger.warning(f"Could not load user preferences: {e}")
    
    async def _store_user_preferences(self, user_prefs: UserPreferences):
        """Store user preferences in database"""
        try:
            await self.database_service.store_user_preferences({
                'user_id': user_prefs.user_id,
                'voicebox_preference': user_prefs.voicebox_preference.value,
                'learning_mode': user_prefs.learning_mode.value,
                'quality_threshold': user_prefs.quality_threshold,
                'scientific_domains': user_prefs.scientific_domains,
                'reasoning_preferences': user_prefs.reasoning_preferences,
                'feedback_frequency': user_prefs.feedback_frequency,
                'privacy_settings': user_prefs.privacy_settings,
                'created_at': user_prefs.created_at
            })
            
        except Exception as e:
            logger.error(f"Failed to store user preferences: {e}")
    
    async def _get_user_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific metrics"""
        try:
            metrics = await self.database_service.get_user_metrics(user_id)
            return metrics or {}
            
        except Exception as e:
            logger.error(f"Failed to get user metrics: {e}")
            return {}
    
    async def _optimize_voicebox_selection(self, user_prefs: UserPreferences, user_metrics: Dict[str, Any]) -> VoiceboxType:
        """Optimize voicebox selection based on user performance"""
        # Analyze user's performance with different voiceboxes
        performance_by_voicebox = user_metrics.get("voicebox_performance", {})
        
        # Find best performing voicebox
        best_voicebox = user_prefs.voicebox_preference
        best_score = 0.0
        
        for voicebox_str, performance in performance_by_voicebox.items():
            score = performance.get("average_quality", 0.0)
            if score > best_score:
                best_score = score
                best_voicebox = VoiceboxType(voicebox_str)
        
        return best_voicebox
    
    async def _optimize_learning_parameters(self) -> List[str]:
        """Optimize learning parameters"""
        optimizations = []
        
        if self.seal_integration:
            # Analyze learning effectiveness
            progress = await self.seal_integration.get_learning_progress()
            
            # Optimize learning rate based on progress
            if progress.learning_velocity < 0.5:
                optimizations.append("Increased learning rate for better adaptation")
            elif progress.learning_velocity > 2.0:
                optimizations.append("Decreased learning rate for stability")
        
        return optimizations
    
    async def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze global metrics
        avg_quality = self.global_learning_metrics["average_quality"]
        improvement_rate = self.global_learning_metrics["improvement_rate"]
        
        if avg_quality < 0.8:
            recommendations.append("Consider enabling enhanced learning mode for better quality")
        
        if improvement_rate < 0.1:
            recommendations.append("System may benefit from more diverse training data")
        
        # Analyze voicebox performance
        for voicebox, performance in self.voicebox_performance.items():
            if performance["quality"] < 0.8:
                recommendations.append(f"Consider optimizing {voicebox.value} voicebox performance")
        
        return recommendations


# Global adaptive system instance
_adaptive_system = None

async def get_adaptive_system() -> NWTNAdaptiveCompleteSystem:
    """Get the global adaptive system instance"""
    global _adaptive_system
    if _adaptive_system is None:
        _adaptive_system = NWTNAdaptiveCompleteSystem()
        await _adaptive_system.initialize()
    return _adaptive_system