"""
User Parameter Manager for NWTN - Phase 3 Implementation.

This module implements sophisticated user parameter collection, validation,
and dynamic adaptation based on user preferences and feedback. It provides
the interactive interface for customizing NWTN's synthesis behavior.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from .contextual_synthesizer import UserParameters, ResponseTone


logger = logging.getLogger(__name__)


# ============================================================================
# Parameter Collection and Validation
# ============================================================================

class VerbosityLevel(Enum):
    """Verbosity levels for response length and detail"""
    BRIEF = "BRIEF"
    MODERATE = "MODERATE"
    COMPREHENSIVE = "COMPREHENSIVE"
    EXHAUSTIVE = "EXHAUSTIVE"


class ReasoningMode(Enum):
    """Reasoning modes for synthesis approach"""
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    CREATIVE = "CREATIVE"
    REVOLUTIONARY = "REVOLUTIONARY"


class DepthLevel(Enum):
    """Depth levels for analysis detail"""
    SURFACE = "SURFACE"
    INTERMEDIATE = "INTERMEDIATE"
    DEEP = "DEEP"
    EXHAUSTIVE = "EXHAUSTIVE"


class SynthesisFocus(Enum):
    """Focus areas for synthesis emphasis"""
    INSIGHTS = "INSIGHTS"
    EVIDENCE = "EVIDENCE"
    METHODOLOGY = "METHODOLOGY"
    IMPLICATIONS = "IMPLICATIONS"
    APPLICATIONS = "APPLICATIONS"


class CitationStyle(Enum):
    """Citation presentation styles"""
    MINIMAL = "MINIMAL"
    CONTEXTUAL = "CONTEXTUAL"
    COMPREHENSIVE = "COMPREHENSIVE"
    ACADEMIC = "ACADEMIC"


class UncertaintyHandling(Enum):
    """How to handle uncertainty in responses"""
    HIDE = "HIDE"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    EXPLORE = "EXPLORE"
    EMPHASIZE = "EMPHASIZE"


@dataclass
class ParameterValidationResult:
    """Result of parameter validation"""
    is_valid: bool
    validated_parameters: Optional[UserParameters]
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ParameterCollectionSession:
    """Session for collecting user parameters"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    parameters_collected: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    completion_status: str = "in_progress"  # in_progress, completed, abandoned
    final_parameters: Optional[UserParameters] = None


@dataclass
class UserPreferenceProfile:
    """User's preference profile based on historical interactions"""
    user_id: str
    preferred_verbosity: VerbosityLevel
    preferred_reasoning_mode: ReasoningMode
    preferred_depth: DepthLevel
    preferred_tone: ResponseTone
    preferred_focus: SynthesisFocus
    preferred_citation_style: CitationStyle
    preferred_uncertainty_handling: UncertaintyHandling
    
    # Learning metrics
    satisfaction_scores: List[float] = field(default_factory=list)
    parameter_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    successful_configurations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    profile_created: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0


class UserParameterManager:
    """
    Manages user parameter collection, validation, and adaptation for
    dynamic synthesis customization.
    """
    
    def __init__(self):
        """Initialize the user parameter manager"""
        
        # Parameter collection strategies
        self.collection_strategies = {
            'interactive': self._interactive_parameter_collection,
            'guided': self._guided_parameter_collection,
            'quick': self._quick_parameter_collection,
            'adaptive': self._adaptive_parameter_collection
        }
        
        # Parameter validation rules
        self.validation_rules = {
            'verbosity_depth_consistency': self._validate_verbosity_depth_consistency,
            'mode_focus_alignment': self._validate_mode_focus_alignment,
            'citation_verbosity_alignment': self._validate_citation_verbosity_alignment,
            'uncertainty_mode_consistency': self._validate_uncertainty_mode_consistency
        }
        
        # User preference profiles
        self.user_profiles: Dict[str, UserPreferenceProfile] = {}
        
        # Default parameters
        self.default_parameters = UserParameters(
            verbosity="MODERATE",
            reasoning_mode="BALANCED",
            depth="INTERMEDIATE",
            synthesis_focus="INSIGHTS",
            citation_style="CONTEXTUAL",
            uncertainty_handling="ACKNOWLEDGE",
            tone=ResponseTone.CONVERSATIONAL
        )
        
        logger.info("UserParameterManager initialized with collection strategies")
    
    async def collect_user_parameters(self,
                                    user_id: Optional[str] = None,
                                    collection_strategy: str = "adaptive",
                                    context: Optional[Dict[str, Any]] = None) -> ParameterValidationResult:
        """
        Collect user parameters using specified strategy.
        
        Args:
            user_id: Optional user identifier for personalization
            collection_strategy: Strategy to use for parameter collection
            context: Optional context information (query, domain, etc.)
            
        Returns:
            ParameterValidationResult with validated parameters
        """
        
        logger.info(f"Starting parameter collection with {collection_strategy} strategy")
        
        try:
            # Create collection session
            session = ParameterCollectionSession(
                session_id=f"session_{datetime.now().timestamp()}",
                user_id=user_id,
                start_time=datetime.now()
            )
            
            # Use appropriate collection strategy
            collection_func = self.collection_strategies.get(
                collection_strategy, 
                self._adaptive_parameter_collection
            )
            
            # Collect parameters
            collected_params = await collection_func(session, context)
            
            # Validate parameters
            validation_result = await self._validate_parameters(collected_params, context)
            
            # Update user profile if user_id provided
            if user_id and validation_result.is_valid:
                await self._update_user_profile(user_id, validation_result.validated_parameters)
            
            # Complete session
            session.completion_status = "completed" if validation_result.is_valid else "failed"
            session.final_parameters = validation_result.validated_parameters
            
            logger.info(f"Parameter collection completed: valid={validation_result.is_valid}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in parameter collection: {e}")
            return ParameterValidationResult(
                is_valid=False,
                validated_parameters=None,
                validation_errors=[f"Collection failed: {str(e)}"],
                suggestions=["Please try again with default parameters"]
            )
    
    async def adapt_parameters_from_feedback(self,
                                           current_parameters: UserParameters,
                                           feedback: Dict[str, Any],
                                           user_id: Optional[str] = None) -> UserParameters:
        """
        Adapt parameters based on user feedback.
        
        Args:
            current_parameters: Current parameter configuration
            feedback: User feedback about response quality
            user_id: Optional user identifier
            
        Returns:
            Adapted UserParameters
        """
        
        logger.info("Adapting parameters based on user feedback")
        
        try:
            # Analyze feedback
            feedback_analysis = await self._analyze_feedback(feedback)
            
            # Generate parameter adjustments
            adjustments = await self._generate_parameter_adjustments(
                current_parameters, feedback_analysis
            )
            
            # Apply adjustments
            adapted_parameters = await self._apply_parameter_adjustments(
                current_parameters, adjustments
            )
            
            # Update user profile if available
            if user_id:
                await self._record_parameter_adjustment(user_id, current_parameters, adapted_parameters, feedback)
            
            logger.info(f"Parameters adapted based on feedback: {len(adjustments)} adjustments made")
            
            return adapted_parameters
            
        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")
            return current_parameters  # Return original if adaptation fails
    
    # ========================================================================
    # Parameter Collection Strategies
    # ========================================================================
    
    async def _interactive_parameter_collection(self,
                                              session: ParameterCollectionSession,
                                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Interactive parameter collection with detailed questions"""
        
        print("\n🎯 Welcome to NWTN Parameter Configuration")
        print("=" * 50)
        print("Let's customize your analysis preferences for optimal results.\n")
        
        collected = {}
        
        # 1. Verbosity Collection
        print("📊 RESPONSE VERBOSITY")
        print("How detailed would you like the response to be?")
        print("1. BRIEF - Concise key points (2-3 paragraphs)")
        print("2. MODERATE - Balanced detail (4-6 paragraphs)")
        print("3. COMPREHENSIVE - Detailed analysis (8-12 paragraphs)")
        print("4. EXHAUSTIVE - Complete exploration (15+ paragraphs)")
        
        verbosity_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        verbosity_map = {"1": "BRIEF", "2": "MODERATE", "3": "COMPREHENSIVE", "4": "EXHAUSTIVE"}
        collected['verbosity'] = verbosity_map[verbosity_choice]
        
        # 2. Reasoning Mode Collection
        print("\n🧠 REASONING MODE")
        print("What type of analysis approach do you prefer?")
        print("1. CONSERVATIVE - Established methods, proven approaches")
        print("2. BALANCED - Mix of proven and innovative thinking")
        print("3. CREATIVE - Innovative connections, novel perspectives")
        print("4. REVOLUTIONARY - Breakthrough thinking, paradigm shifts")
        
        mode_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        mode_map = {"1": "CONSERVATIVE", "2": "BALANCED", "3": "CREATIVE", "4": "REVOLUTIONARY"}
        collected['reasoning_mode'] = mode_map[mode_choice]
        
        # 3. Analysis Depth Collection
        print("\n🔍 ANALYSIS DEPTH")
        print("How deep should the reasoning analysis go?")
        print("1. SURFACE - High-level overview")
        print("2. INTERMEDIATE - Moderate detail with key insights")
        print("3. DEEP - Detailed reasoning exploration")
        print("4. EXHAUSTIVE - Complete reasoning transparency")
        
        depth_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        depth_map = {"1": "SURFACE", "2": "INTERMEDIATE", "3": "DEEP", "4": "EXHAUSTIVE"}
        collected['depth'] = depth_map[depth_choice]
        
        # 4. Synthesis Focus Collection
        print("\n🎯 SYNTHESIS FOCUS")
        print("What should the response emphasize?")
        print("1. INSIGHTS - Novel insights and connections")
        print("2. EVIDENCE - Research evidence and validation")
        print("3. METHODOLOGY - How the analysis was conducted")
        print("4. IMPLICATIONS - What the findings mean")
        print("5. APPLICATIONS - Practical applications")
        
        focus_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="1")
        focus_map = {
            "1": "INSIGHTS", "2": "EVIDENCE", "3": "METHODOLOGY", 
            "4": "IMPLICATIONS", "5": "APPLICATIONS"
        }
        collected['synthesis_focus'] = focus_map[focus_choice]
        
        # 5. Citation Style Collection
        print("\n📚 CITATION STYLE")
        print("How should research sources be presented?")
        print("1. MINIMAL - Basic attribution only")
        print("2. CONTEXTUAL - Citations integrated naturally")
        print("3. COMPREHENSIVE - Detailed source information")
        print("4. ACADEMIC - Formal academic citations")
        
        citation_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        citation_map = {"1": "MINIMAL", "2": "CONTEXTUAL", "3": "COMPREHENSIVE", "4": "ACADEMIC"}
        collected['citation_style'] = citation_map[citation_choice]
        
        # 6. Uncertainty Handling Collection
        print("\n❓ UNCERTAINTY HANDLING")
        print("How should uncertainties and limitations be presented?")
        print("1. HIDE - Minimize uncertainty discussion")
        print("2. ACKNOWLEDGE - Brief uncertainty acknowledgment")
        print("3. EXPLORE - Explore uncertainties in detail")
        print("4. EMPHASIZE - Emphasize uncertainties and limitations")
        
        uncertainty_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        uncertainty_map = {"1": "HIDE", "2": "ACKNOWLEDGE", "3": "EXPLORE", "4": "EMPHASIZE"}
        collected['uncertainty_handling'] = uncertainty_map[uncertainty_choice]
        
        # 7. Response Tone Collection
        print("\n🎭 RESPONSE TONE")
        print("What tone should the response have?")
        print("1. ACADEMIC - Formal academic style")
        print("2. CONVERSATIONAL - Natural, engaging style")
        print("3. INNOVATIVE - Creative, forward-thinking style")
        print("4. ANALYTICAL - Systematic, methodical style")
        print("5. EXPLORATORY - Curious, investigative style")
        
        tone_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="2")
        tone_map = {
            "1": ResponseTone.ACADEMIC,
            "2": ResponseTone.CONVERSATIONAL,
            "3": ResponseTone.INNOVATIVE,
            "4": ResponseTone.ANALYTICAL,
            "5": ResponseTone.EXPLORATORY
        }
        collected['tone'] = tone_map[tone_choice]
        
        print("\n✅ Parameter collection completed!")
        print("=" * 50)
        
        return collected
    
    async def _guided_parameter_collection(self,
                                         session: ParameterCollectionSession,
                                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Guided parameter collection with smart defaults"""
        
        print("\n🎯 Quick NWTN Configuration")
        print("=" * 40)
        
        collected = {}
        
        # Smart defaults based on context
        if context:
            query = context.get('query', '').lower()
            
            # Infer preferences from query content
            if any(word in query for word in ['breakthrough', 'revolutionary', 'innovative']):
                collected['reasoning_mode'] = 'REVOLUTIONARY'
                collected['synthesis_focus'] = 'INSIGHTS'
            elif any(word in query for word in ['evidence', 'research', 'study']):
                collected['reasoning_mode'] = 'CONSERVATIVE'
                collected['synthesis_focus'] = 'EVIDENCE'
            else:
                collected['reasoning_mode'] = 'BALANCED'
                collected['synthesis_focus'] = 'INSIGHTS'
        
        # Quick questions with smart defaults
        print("I'll ask just a few quick questions with smart defaults based on your query.")
        
        # Verbosity (most important choice)
        print("\n📊 How detailed should the response be?")
        print("1. Brief overview  2. Moderate detail  3. Comprehensive analysis")
        verbosity_choice = await self._get_user_choice(["1", "2", "3"], default="2")
        verbosity_map = {"1": "BRIEF", "2": "MODERATE", "3": "COMPREHENSIVE"}
        collected['verbosity'] = verbosity_map[verbosity_choice]
        
        # Depth based on verbosity
        if collected['verbosity'] == 'BRIEF':
            collected['depth'] = 'SURFACE'
        elif collected['verbosity'] == 'COMPREHENSIVE':
            collected['depth'] = 'DEEP'
        else:
            collected['depth'] = 'INTERMEDIATE'
        
        # Use defaults for other parameters
        collected.setdefault('citation_style', 'CONTEXTUAL')
        collected.setdefault('uncertainty_handling', 'ACKNOWLEDGE')
        collected.setdefault('tone', ResponseTone.CONVERSATIONAL)
        
        print("✅ Configuration complete with smart defaults!")
        
        return collected
    
    async def _quick_parameter_collection(self,
                                        session: ParameterCollectionSession,
                                        context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Quick parameter collection with minimal user input"""
        
        print("\n⚡ Quick Setup")
        print("Choose your preferred analysis style:")
        print("1. Conservative & Evidence-based")
        print("2. Balanced & Comprehensive") 
        print("3. Creative & Insight-focused")
        print("4. Revolutionary & Breakthrough-oriented")
        
        style_choice = await self._get_user_choice(["1", "2", "3", "4"], default="2")
        
        style_configs = {
            "1": {
                'verbosity': 'MODERATE',
                'reasoning_mode': 'CONSERVATIVE',
                'depth': 'INTERMEDIATE',
                'synthesis_focus': 'EVIDENCE',
                'citation_style': 'COMPREHENSIVE',
                'uncertainty_handling': 'ACKNOWLEDGE',
                'tone': ResponseTone.ACADEMIC
            },
            "2": {
                'verbosity': 'MODERATE',
                'reasoning_mode': 'BALANCED',
                'depth': 'INTERMEDIATE',
                'synthesis_focus': 'INSIGHTS',
                'citation_style': 'CONTEXTUAL',
                'uncertainty_handling': 'ACKNOWLEDGE',
                'tone': ResponseTone.CONVERSATIONAL
            },
            "3": {
                'verbosity': 'COMPREHENSIVE',
                'reasoning_mode': 'CREATIVE',
                'depth': 'DEEP',
                'synthesis_focus': 'INSIGHTS',
                'citation_style': 'CONTEXTUAL',
                'uncertainty_handling': 'EXPLORE',
                'tone': ResponseTone.INNOVATIVE
            },
            "4": {
                'verbosity': 'EXHAUSTIVE',
                'reasoning_mode': 'REVOLUTIONARY',
                'depth': 'EXHAUSTIVE',
                'synthesis_focus': 'IMPLICATIONS',
                'citation_style': 'MINIMAL',
                'uncertainty_handling': 'EMPHASIZE',
                'tone': ResponseTone.EXPLORATORY
            }
        }
        
        print("✅ Quick setup complete!")
        return style_configs[style_choice]
    
    async def _adaptive_parameter_collection(self,
                                           session: ParameterCollectionSession,
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptive parameter collection using user profile and context"""
        
        user_id = session.user_id
        
        # Check if user has a profile
        if user_id and user_id in self.user_profiles:
            print(f"\n👋 Welcome back! Using your preferred settings...")
            profile = self.user_profiles[user_id]
            
            collected = {
                'verbosity': profile.preferred_verbosity.value,
                'reasoning_mode': profile.preferred_reasoning_mode.value,
                'depth': profile.preferred_depth.value,
                'synthesis_focus': profile.preferred_focus.value,
                'citation_style': profile.preferred_citation_style.value,
                'uncertainty_handling': profile.preferred_uncertainty_handling.value,
                'tone': profile.preferred_tone
            }
            
            print("🔧 Want to adjust any settings? (y/n)")
            adjust = await self._get_user_choice(["y", "n"], default="n")
            
            if adjust == "y":
                # Allow selective adjustments
                collected = await self._selective_parameter_adjustment(collected)
            
            return collected
        else:
            # New user - use guided collection
            print("\n👋 Welcome to NWTN! Let me learn your preferences...")
            return await self._guided_parameter_collection(session, context)
    
    # ========================================================================
    # Parameter Validation
    # ========================================================================
    
    async def _validate_parameters(self,
                                 collected_params: Dict[str, Any],
                                 context: Optional[Dict[str, Any]]) -> ParameterValidationResult:
        """Validate collected parameters"""
        
        validation_errors = []
        validation_warnings = []
        suggestions = []
        
        try:
            # Create UserParameters object
            user_params = UserParameters(
                verbosity=collected_params.get('verbosity', 'MODERATE'),
                reasoning_mode=collected_params.get('reasoning_mode', 'BALANCED'),
                depth=collected_params.get('depth', 'INTERMEDIATE'),
                synthesis_focus=collected_params.get('synthesis_focus', 'INSIGHTS'),
                citation_style=collected_params.get('citation_style', 'CONTEXTUAL'),
                uncertainty_handling=collected_params.get('uncertainty_handling', 'ACKNOWLEDGE'),
                tone=collected_params.get('tone', ResponseTone.CONVERSATIONAL)
            )
            
            # Run validation rules
            for rule_name, rule_func in self.validation_rules.items():
                rule_result = await rule_func(user_params, context)
                
                if rule_result['is_valid']:
                    if rule_result.get('warnings'):
                        validation_warnings.extend(rule_result['warnings'])
                else:
                    validation_errors.extend(rule_result.get('errors', []))
                    
                if rule_result.get('suggestions'):
                    suggestions.extend(rule_result['suggestions'])
            
            # Check for critical validation failures
            is_valid = len(validation_errors) == 0
            
            return ParameterValidationResult(
                is_valid=is_valid,
                validated_parameters=user_params if is_valid else None,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ParameterValidationResult(
                is_valid=False,
                validated_parameters=None,
                validation_errors=[f"Parameter validation failed: {str(e)}"],
                suggestions=["Please try again with default parameters"]
            )
    
    async def _validate_verbosity_depth_consistency(self,
                                                  params: UserParameters,
                                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate consistency between verbosity and depth"""
        
        verbosity_depth_map = {
            'BRIEF': ['SURFACE', 'INTERMEDIATE'],
            'MODERATE': ['SURFACE', 'INTERMEDIATE', 'DEEP'],
            'COMPREHENSIVE': ['INTERMEDIATE', 'DEEP', 'EXHAUSTIVE'],
            'EXHAUSTIVE': ['DEEP', 'EXHAUSTIVE']
        }
        
        valid_depths = verbosity_depth_map.get(params.verbosity, [])
        
        if params.depth not in valid_depths:
            return {
                'is_valid': False,
                'errors': [f"Depth '{params.depth}' inconsistent with verbosity '{params.verbosity}'"],
                'suggestions': [f"Consider depth levels: {', '.join(valid_depths)}"]
            }
        
        return {'is_valid': True}
    
    async def _validate_mode_focus_alignment(self,
                                           params: UserParameters,
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate alignment between reasoning mode and synthesis focus"""
        
        mode_focus_recommendations = {
            'CONSERVATIVE': ['EVIDENCE', 'METHODOLOGY'],
            'BALANCED': ['INSIGHTS', 'IMPLICATIONS'],
            'CREATIVE': ['INSIGHTS', 'APPLICATIONS'],
            'REVOLUTIONARY': ['INSIGHTS', 'IMPLICATIONS', 'APPLICATIONS']
        }
        
        recommended = mode_focus_recommendations.get(params.reasoning_mode, [])
        
        if params.synthesis_focus not in recommended:
            return {
                'is_valid': True,  # Warning, not error
                'warnings': [f"Focus '{params.synthesis_focus}' may not align optimally with '{params.reasoning_mode}' mode"],
                'suggestions': [f"Consider focus areas: {', '.join(recommended)}"]
            }
        
        return {'is_valid': True}
    
    async def _validate_citation_verbosity_alignment(self,
                                                   params: UserParameters,
                                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate citation style aligns with verbosity"""
        
        if params.verbosity == 'BRIEF' and params.citation_style == 'ACADEMIC':
            return {
                'is_valid': True,
                'warnings': ["Academic citations may be verbose for brief responses"],
                'suggestions': ["Consider CONTEXTUAL or MINIMAL citation style for brief responses"]
            }
        
        return {'is_valid': True}
    
    async def _validate_uncertainty_mode_consistency(self,
                                                   params: UserParameters,
                                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate uncertainty handling consistency with reasoning mode"""
        
        if params.reasoning_mode == 'CONSERVATIVE' and params.uncertainty_handling == 'HIDE':
            return {
                'is_valid': True,
                'warnings': ["Conservative mode benefits from uncertainty acknowledgment"],
                'suggestions': ["Consider ACKNOWLEDGE or EXPLORE for uncertainty handling"]
            }
        
        return {'is_valid': True}
    
    # ========================================================================
    # User Profile Management
    # ========================================================================
    
    async def _update_user_profile(self, user_id: str, parameters: UserParameters):
        """Update user profile with new parameters"""
        
        if user_id not in self.user_profiles:
            # Create new profile
            self.user_profiles[user_id] = UserPreferenceProfile(
                user_id=user_id,
                preferred_verbosity=VerbosityLevel(parameters.verbosity),
                preferred_reasoning_mode=ReasoningMode(parameters.reasoning_mode),
                preferred_depth=DepthLevel(parameters.depth),
                preferred_tone=parameters.tone,
                preferred_focus=SynthesisFocus(parameters.synthesis_focus),
                preferred_citation_style=CitationStyle(parameters.citation_style),
                preferred_uncertainty_handling=UncertaintyHandling(parameters.uncertainty_handling)
            )
        else:
            # Update existing profile
            profile = self.user_profiles[user_id]
            profile.preferred_verbosity = VerbosityLevel(parameters.verbosity)
            profile.preferred_reasoning_mode = ReasoningMode(parameters.reasoning_mode)
            profile.preferred_depth = DepthLevel(parameters.depth)
            profile.preferred_tone = parameters.tone
            profile.preferred_focus = SynthesisFocus(parameters.synthesis_focus)
            profile.preferred_citation_style = CitationStyle(parameters.citation_style)
            profile.preferred_uncertainty_handling = UncertaintyHandling(parameters.uncertainty_handling)
            profile.last_updated = datetime.now()
            profile.interaction_count += 1
        
        logger.debug(f"Updated user profile for {user_id}")
    
    # ========================================================================
    # Feedback Processing and Adaptation
    # ========================================================================
    
    async def _analyze_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user feedback to understand satisfaction and issues"""
        
        analysis = {
            'satisfaction_score': feedback.get('satisfaction_score', 0.5),
            'issues_identified': [],
            'positive_aspects': [],
            'adjustment_suggestions': []
        }
        
        # Analyze specific feedback dimensions
        if 'too_verbose' in feedback and feedback['too_verbose']:
            analysis['issues_identified'].append('excessive_verbosity')
            analysis['adjustment_suggestions'].append('reduce_verbosity')
        
        if 'too_brief' in feedback and feedback['too_brief']:
            analysis['issues_identified'].append('insufficient_detail')
            analysis['adjustment_suggestions'].append('increase_verbosity')
        
        if 'too_conservative' in feedback and feedback['too_conservative']:
            analysis['issues_identified'].append('insufficient_creativity')
            analysis['adjustment_suggestions'].append('increase_creativity')
        
        if 'too_speculative' in feedback and feedback['too_speculative']:
            analysis['issues_identified'].append('excessive_speculation')
            analysis['adjustment_suggestions'].append('increase_conservatism')
        
        if 'unclear_citations' in feedback and feedback['unclear_citations']:
            analysis['issues_identified'].append('citation_clarity')
            analysis['adjustment_suggestions'].append('improve_citations')
        
        if 'good_insights' in feedback and feedback['good_insights']:
            analysis['positive_aspects'].append('insight_quality')
        
        if 'good_depth' in feedback and feedback['good_depth']:
            analysis['positive_aspects'].append('analysis_depth')
        
        return analysis
    
    async def _generate_parameter_adjustments(self,
                                            current_params: UserParameters,
                                            feedback_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter adjustments based on feedback analysis"""
        
        adjustments = []
        
        for suggestion in feedback_analysis['adjustment_suggestions']:
            if suggestion == 'reduce_verbosity':
                current_verbosity = current_params.verbosity
                if current_verbosity == 'EXHAUSTIVE':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'COMPREHENSIVE'})
                elif current_verbosity == 'COMPREHENSIVE':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'MODERATE'})
                elif current_verbosity == 'MODERATE':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'BRIEF'})
            
            elif suggestion == 'increase_verbosity':
                current_verbosity = current_params.verbosity
                if current_verbosity == 'BRIEF':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'MODERATE'})
                elif current_verbosity == 'MODERATE':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'COMPREHENSIVE'})
                elif current_verbosity == 'COMPREHENSIVE':
                    adjustments.append({'parameter': 'verbosity', 'new_value': 'EXHAUSTIVE'})
            
            elif suggestion == 'increase_creativity':
                current_mode = current_params.reasoning_mode
                if current_mode == 'CONSERVATIVE':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'BALANCED'})
                elif current_mode == 'BALANCED':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'CREATIVE'})
                elif current_mode == 'CREATIVE':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'REVOLUTIONARY'})
            
            elif suggestion == 'increase_conservatism':
                current_mode = current_params.reasoning_mode
                if current_mode == 'REVOLUTIONARY':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'CREATIVE'})
                elif current_mode == 'CREATIVE':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'BALANCED'})
                elif current_mode == 'BALANCED':
                    adjustments.append({'parameter': 'reasoning_mode', 'new_value': 'CONSERVATIVE'})
            
            elif suggestion == 'improve_citations':
                current_style = current_params.citation_style
                if current_style == 'MINIMAL':
                    adjustments.append({'parameter': 'citation_style', 'new_value': 'CONTEXTUAL'})
                elif current_style == 'CONTEXTUAL':
                    adjustments.append({'parameter': 'citation_style', 'new_value': 'COMPREHENSIVE'})
        
        return adjustments
    
    async def _apply_parameter_adjustments(self,
                                         current_params: UserParameters,
                                         adjustments: List[Dict[str, Any]]) -> UserParameters:
        """Apply parameter adjustments to create new parameter set"""
        
        # Create copy of current parameters
        new_params_dict = {
            'verbosity': current_params.verbosity,
            'reasoning_mode': current_params.reasoning_mode,
            'depth': current_params.depth,
            'synthesis_focus': current_params.synthesis_focus,
            'citation_style': current_params.citation_style,
            'uncertainty_handling': current_params.uncertainty_handling,
            'tone': current_params.tone
        }
        
        # Apply adjustments
        for adjustment in adjustments:
            parameter = adjustment['parameter']
            new_value = adjustment['new_value']
            
            if parameter in new_params_dict:
                new_params_dict[parameter] = new_value
        
        # Create new UserParameters object
        return UserParameters(
            verbosity=new_params_dict['verbosity'],
            reasoning_mode=new_params_dict['reasoning_mode'],
            depth=new_params_dict['depth'],
            synthesis_focus=new_params_dict['synthesis_focus'],
            citation_style=new_params_dict['citation_style'],
            uncertainty_handling=new_params_dict['uncertainty_handling'],
            tone=new_params_dict['tone']
        )
    
    async def _record_parameter_adjustment(self,
                                         user_id: str,
                                         old_params: UserParameters,
                                         new_params: UserParameters,
                                         feedback: Dict[str, Any]):
        """Record parameter adjustment in user profile"""
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            adjustment_record = {
                'timestamp': datetime.now().isoformat(),
                'old_parameters': old_params.__dict__,
                'new_parameters': new_params.__dict__,
                'feedback_trigger': feedback,
                'satisfaction_score': feedback.get('satisfaction_score', 0.5)
            }
            
            profile.parameter_adjustments.append(adjustment_record)
            profile.satisfaction_scores.append(feedback.get('satisfaction_score', 0.5))
            
            # Update preferences to new parameters if satisfaction improved
            if feedback.get('satisfaction_score', 0.5) > 0.7:
                profile.successful_configurations.append(new_params.__dict__)
                # Update preferred parameters
                await self._update_user_profile(user_id, new_params)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    async def _get_user_choice(self, options: List[str], default: str) -> str:
        """Get user choice from options (simplified for demo)"""
        
        # In a real implementation, this would handle user input
        # For demo purposes, return default
        print(f"[Simulated choice: {default}]")
        return default
    
    async def _selective_parameter_adjustment(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Allow selective adjustment of specific parameters"""
        
        print("Current settings:")
        for key, value in current_params.items():
            print(f"  {key}: {value}")
        
        print("\nWhich parameter would you like to adjust? (or 'none')")
        # Simplified - in real implementation would allow actual selection
        print("[Simulated: keeping current settings]")
        
        return current_params
    
    def get_user_profile(self, user_id: str) -> Optional[UserPreferenceProfile]:
        """Get user preference profile"""
        return self.user_profiles.get(user_id)
    
    def get_default_parameters(self) -> UserParameters:
        """Get default parameters"""
        return self.default_parameters