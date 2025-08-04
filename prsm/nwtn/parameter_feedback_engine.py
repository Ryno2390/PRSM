"""
Parameter Feedback Engine for NWTN - Phase 3 Implementation.

This module implements real-time parameter adjustment based on user feedback,
enabling continuous improvement of synthesis quality through adaptive learning
and personalized optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

from .contextual_synthesizer import UserParameters, EnhancedResponse
from .user_parameter_manager import UserPreferenceProfile


logger = logging.getLogger(__name__)


# ============================================================================
# Feedback Analysis Data Structures
# ============================================================================

class FeedbackType(Enum):
    """Types of user feedback"""
    SATISFACTION_RATING = "satisfaction_rating"
    VERBOSITY_FEEDBACK = "verbosity_feedback"
    DEPTH_FEEDBACK = "depth_feedback"
    TONE_FEEDBACK = "tone_feedback"
    CITATION_FEEDBACK = "citation_feedback"
    CLARITY_FEEDBACK = "clarity_feedback"
    RELEVANCE_FEEDBACK = "relevance_feedback"
    COMPLETENESS_FEEDBACK = "completeness_feedback"
    ENGAGEMENT_FEEDBACK = "engagement_feedback"


class FeedbackSentiment(Enum):
    """Feedback sentiment classifications"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    intensity: float  # 0.0 to 1.0
    specific_issue: Optional[str] = None
    suggested_adjustment: Optional[str] = None
    confidence: float = 1.0


@dataclass
class FeedbackSession:
    """Complete feedback session"""
    session_id: str
    user_id: Optional[str]
    original_query: str
    original_parameters: UserParameters
    response_received: EnhancedResponse
    feedback_items: List[FeedbackItem] = field(default_factory=list)
    overall_satisfaction: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Derived analysis
    parameter_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    adjustment_confidence: float = 0.0
    learning_insights: List[str] = field(default_factory=list)


@dataclass
class ParameterAdjustment:
    """Specific parameter adjustment recommendation"""
    parameter_name: str
    current_value: Any
    suggested_value: Any
    adjustment_rationale: str
    confidence: float
    expected_improvement: float


@dataclass
class FeedbackAnalysis:
    """Analysis of feedback patterns"""
    dominant_issues: List[str]
    satisfaction_trend: float
    parameter_effectiveness: Dict[str, float]
    adjustment_recommendations: List[ParameterAdjustment]
    learning_rate: float
    user_preference_evolution: Dict[str, Any]


class ParameterFeedbackEngine:
    """
    Analyzes user feedback and generates real-time parameter adjustments
    to continuously improve synthesis quality.
    """
    
    def __init__(self):
        """Initialize the parameter feedback engine"""
        
        # Feedback analysis components
        self.feedback_analyzer = FeedbackAnalyzer()
        self.adjustment_generator = AdjustmentGenerator()
        self.learning_optimizer = LearningOptimizer()
        
        # Feedback patterns and learning
        self.feedback_history: Dict[str, List[FeedbackSession]] = {}
        self.parameter_effectiveness_history: Dict[str, List[float]] = {}
        self.adjustment_success_rates: Dict[str, float] = {}
        
        # Learning configuration
        self.learning_config = {
            'min_feedback_sessions': 3,
            'confidence_threshold': 0.6,
            'adjustment_dampening': 0.7,  # Prevent overcorrection
            'learning_rate_decay': 0.95,
            'feedback_window_hours': 168  # 1 week
        }
        
        logger.info("ParameterFeedbackEngine initialized")
    
    async def analyze_feedback_and_adjust(self,
                                         feedback_session: FeedbackSession,
                                         user_profile: Optional[UserPreferenceProfile] = None) -> Tuple[UserParameters, FeedbackAnalysis]:
        """
        Analyze feedback and generate adjusted parameters.
        
        Args:
            feedback_session: Complete feedback session with user input
            user_profile: Optional user profile for personalized learning
            
        Returns:
            Tuple of (adjusted_parameters, feedback_analysis)
        """
        
        logger.info(f"Analyzing feedback and generating parameter adjustments")
        
        try:
            # Store feedback session
            await self._store_feedback_session(feedback_session)
            
            # Analyze current feedback
            current_analysis = await self.feedback_analyzer.analyze_feedback_session(feedback_session)
            
            # Analyze historical patterns if available
            historical_analysis = await self._analyze_historical_patterns(
                feedback_session.user_id, user_profile
            )
            
            # Generate parameter adjustments
            adjustments = await self.adjustment_generator.generate_adjustments(
                current_analysis, historical_analysis, feedback_session.original_parameters
            )
            
            # Apply learning optimization
            optimized_adjustments = await self.learning_optimizer.optimize_adjustments(
                adjustments, feedback_session, user_profile
            )
            
            # Create adjusted parameters
            adjusted_parameters = await self._apply_adjustments(
                feedback_session.original_parameters, optimized_adjustments
            )
            
            # Create comprehensive feedback analysis
            feedback_analysis = FeedbackAnalysis(
                dominant_issues=current_analysis['dominant_issues'],
                satisfaction_trend=current_analysis['satisfaction_trend'],
                parameter_effectiveness=current_analysis['parameter_effectiveness'],
                adjustment_recommendations=optimized_adjustments,
                learning_rate=self._calculate_learning_rate(feedback_session.user_id),
                user_preference_evolution=historical_analysis.get('preference_evolution', {})
            )
            
            # Update effectiveness tracking
            await self._update_effectiveness_tracking(feedback_session, optimized_adjustments)
            
            logger.info(f"Feedback analysis completed with {len(optimized_adjustments)} adjustments")
            
            return adjusted_parameters, feedback_analysis
            
        except Exception as e:
            logger.error(f"Error in feedback analysis: {e}")
            # Return original parameters if analysis fails
            return feedback_session.original_parameters, FeedbackAnalysis(
                dominant_issues=[f"Analysis error: {str(e)}"],
                satisfaction_trend=0.5,
                parameter_effectiveness={},
                adjustment_recommendations=[],
                learning_rate=0.0,
                user_preference_evolution={}
            )
    
    async def collect_interactive_feedback(self,
                                         response: EnhancedResponse,
                                         original_query: str,
                                         parameters: UserParameters,
                                         user_id: Optional[str] = None) -> FeedbackSession:
        """
        Collect interactive feedback from user about response quality.
        
        Args:
            response: The enhanced response that was generated
            original_query: Original user query
            parameters: Parameters used for generation
            user_id: Optional user identifier
            
        Returns:
            FeedbackSession with collected feedback
        """
        
        print("\n📝 Response Feedback")
        print("=" * 40)
        print("Please provide feedback on the response quality:")
        
        feedback_session = FeedbackSession(
            session_id=f"feedback_{datetime.now().timestamp()}",
            user_id=user_id,
            original_query=original_query,
            original_parameters=parameters,
            response_received=response
        )
        
        # 1. Overall satisfaction
        print("\n⭐ Overall Satisfaction")
        print("How satisfied are you with this response?")
        print("1. Very Dissatisfied  2. Dissatisfied  3. Neutral  4. Satisfied  5. Very Satisfied")
        
        satisfaction_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="3")
        satisfaction_scores = {"1": 0.1, "2": 0.3, "3": 0.5, "4": 0.7, "5": 0.9}
        feedback_session.overall_satisfaction = satisfaction_scores[satisfaction_choice]
        
        # 2. Verbosity feedback
        print("\n📊 Response Length")
        print("Was the response length appropriate?")
        print("1. Too brief  2. Slightly brief  3. Just right  4. Slightly long  5. Too long")
        
        verbosity_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="3")
        verbosity_feedback = await self._interpret_verbosity_feedback(verbosity_choice)
        if verbosity_feedback:
            feedback_session.feedback_items.append(verbosity_feedback)
        
        # 3. Depth feedback
        print("\n🔍 Analysis Depth")
        print("Was the analysis depth appropriate?")
        print("1. Too shallow  2. Slightly shallow  3. Just right  4. Slightly deep  5. Too deep")
        
        depth_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="3")
        depth_feedback = await self._interpret_depth_feedback(depth_choice)
        if depth_feedback:
            feedback_session.feedback_items.append(depth_feedback)
        
        # 4. Clarity feedback
        print("\n💡 Clarity and Understanding")
        print("How clear and understandable was the response?")
        print("1. Very unclear  2. Unclear  3. Okay  4. Clear  5. Very clear")
        
        clarity_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="4")
        clarity_feedback = await self._interpret_clarity_feedback(clarity_choice)
        if clarity_feedback:
            feedback_session.feedback_items.append(clarity_feedback)
        
        # 5. Relevance feedback
        print("\n🎯 Relevance")
        print("How relevant was the response to your question?")
        print("1. Not relevant  2. Somewhat relevant  3. Relevant  4. Highly relevant  5. Perfectly relevant")
        
        relevance_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="4")
        relevance_feedback = await self._interpret_relevance_feedback(relevance_choice)
        if relevance_feedback:
            feedback_session.feedback_items.append(relevance_feedback)
        
        # 6. Tone feedback
        print("\n🎭 Response Tone")
        print("Was the response tone appropriate?")
        print("1. Too formal  2. Slightly formal  3. Just right  4. Slightly casual  5. Too casual")
        
        tone_choice = await self._get_user_choice(["1", "2", "3", "4", "5"], default="3")
        tone_feedback = await self._interpret_tone_feedback(tone_choice)
        if tone_feedback:
            feedback_session.feedback_items.append(tone_feedback)
        
        # 7. Optional specific feedback
        print("\n💬 Additional Comments")
        print("Any specific issues or suggestions? (Enter for skip)")
        # In real implementation, would collect text input
        print("[Simulated: No additional comments]")
        
        print("\n✅ Feedback collection completed!")
        print("=" * 40)
        
        return feedback_session
    
    # ========================================================================
    # Feedback Interpretation Methods
    # ========================================================================
    
    async def _interpret_verbosity_feedback(self, choice: str) -> Optional[FeedbackItem]:
        """Interpret verbosity feedback choice"""
        
        interpretations = {
            "1": FeedbackItem(
                feedback_type=FeedbackType.VERBOSITY_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.8,
                specific_issue="response_too_brief",
                suggested_adjustment="increase_verbosity"
            ),
            "2": FeedbackItem(
                feedback_type=FeedbackType.VERBOSITY_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.4,
                specific_issue="response_slightly_brief",
                suggested_adjustment="slightly_increase_verbosity"
            ),
            "4": FeedbackItem(
                feedback_type=FeedbackType.VERBOSITY_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.4,
                specific_issue="response_slightly_long",
                suggested_adjustment="slightly_decrease_verbosity"
            ),
            "5": FeedbackItem(
                feedback_type=FeedbackType.VERBOSITY_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.8,
                specific_issue="response_too_long",
                suggested_adjustment="decrease_verbosity"
            )
        }
        
        return interpretations.get(choice)
    
    async def _interpret_depth_feedback(self, choice: str) -> Optional[FeedbackItem]:
        """Interpret depth feedback choice"""
        
        interpretations = {
            "1": FeedbackItem(
                feedback_type=FeedbackType.DEPTH_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.8,
                specific_issue="analysis_too_shallow",
                suggested_adjustment="increase_depth"
            ),
            "2": FeedbackItem(
                feedback_type=FeedbackType.DEPTH_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.4,
                specific_issue="analysis_slightly_shallow",
                suggested_adjustment="slightly_increase_depth"
            ),
            "4": FeedbackItem(
                feedback_type=FeedbackType.DEPTH_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.4,
                specific_issue="analysis_slightly_deep",
                suggested_adjustment="slightly_decrease_depth"
            ),
            "5": FeedbackItem(
                feedback_type=FeedbackType.DEPTH_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.8,
                specific_issue="analysis_too_deep",
                suggested_adjustment="decrease_depth"
            )
        }
        
        return interpretations.get(choice)
    
    async def _interpret_clarity_feedback(self, choice: str) -> Optional[FeedbackItem]:
        """Interpret clarity feedback choice"""
        
        if choice in ["1", "2"]:
            return FeedbackItem(
                feedback_type=FeedbackType.CLARITY_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.8 if choice == "1" else 0.5,
                specific_issue="unclear_response",
                suggested_adjustment="improve_clarity"
            )
        
        return None
    
    async def _interpret_relevance_feedback(self, choice: str) -> Optional[FeedbackItem]:
        """Interpret relevance feedback choice"""
        
        if choice in ["1", "2"]:
            return FeedbackItem(
                feedback_type=FeedbackType.RELEVANCE_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.9 if choice == "1" else 0.6,
                specific_issue="irrelevant_response",
                suggested_adjustment="improve_relevance"
            )
        
        return None
    
    async def _interpret_tone_feedback(self, choice: str) -> Optional[FeedbackItem]:
        """Interpret tone feedback choice"""
        
        interpretations = {
            "1": FeedbackItem(
                feedback_type=FeedbackType.TONE_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.6,
                specific_issue="tone_too_formal",
                suggested_adjustment="make_tone_more_casual"
            ),
            "2": FeedbackItem(
                feedback_type=FeedbackType.TONE_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.3,
                specific_issue="tone_slightly_formal",
                suggested_adjustment="slightly_casual_tone"
            ),
            "4": FeedbackItem(
                feedback_type=FeedbackType.TONE_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.3,
                specific_issue="tone_slightly_casual",
                suggested_adjustment="slightly_formal_tone"
            ),
            "5": FeedbackItem(
                feedback_type=FeedbackType.TONE_FEEDBACK,
                sentiment=FeedbackSentiment.NEGATIVE,
                intensity=0.6,
                specific_issue="tone_too_casual",
                suggested_adjustment="make_tone_more_formal"
            )
        }
        
        return interpretations.get(choice)
    
    # ========================================================================
    # Storage and Analysis Methods
    # ========================================================================
    
    async def _store_feedback_session(self, session: FeedbackSession):
        """Store feedback session for historical analysis"""
        
        user_id = session.user_id or "anonymous"
        
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        
        self.feedback_history[user_id].append(session)
        
        # Limit history size
        if len(self.feedback_history[user_id]) > 50:
            self.feedback_history[user_id] = self.feedback_history[user_id][-50:]
        
        logger.debug(f"Stored feedback session for user {user_id}")
    
    async def _analyze_historical_patterns(self,
                                         user_id: Optional[str],
                                         user_profile: Optional[UserPreferenceProfile]) -> Dict[str, Any]:
        """Analyze historical feedback patterns"""
        
        if not user_id or user_id not in self.feedback_history:
            return {}
        
        user_history = self.feedback_history[user_id]
        
        # Filter recent feedback (within feedback window)
        cutoff_time = datetime.now() - timedelta(hours=self.learning_config['feedback_window_hours'])
        recent_feedback = [
            session for session in user_history
            if session.timestamp >= cutoff_time
        ]
        
        if len(recent_feedback) < self.learning_config['min_feedback_sessions']:
            return {}
        
        analysis = {
            'satisfaction_trend': self._calculate_satisfaction_trend(recent_feedback),
            'common_issues': self._identify_common_issues(recent_feedback),
            'parameter_preferences': self._analyze_parameter_preferences(recent_feedback),
            'adjustment_effectiveness': self._analyze_adjustment_effectiveness(recent_feedback),
            'preference_evolution': self._analyze_preference_evolution(recent_feedback, user_profile)
        }
        
        return analysis
    
    def _calculate_satisfaction_trend(self, feedback_sessions: List[FeedbackSession]) -> float:
        """Calculate satisfaction trend over time"""
        
        if len(feedback_sessions) < 2:
            return 0.0
        
        # Calculate weighted trend (recent feedback weighted more)
        weights = np.linspace(0.5, 1.0, len(feedback_sessions))
        satisfactions = [session.overall_satisfaction for session in feedback_sessions]
        
        # Simple linear trend calculation
        x = np.arange(len(satisfactions))
        trend_slope = np.polyfit(x, satisfactions, 1)[0]
        
        return float(trend_slope)
    
    def _identify_common_issues(self, feedback_sessions: List[FeedbackSession]) -> List[str]:
        """Identify common issues across feedback sessions"""
        
        issue_counts = {}
        
        for session in feedback_sessions:
            for feedback_item in session.feedback_items:
                if feedback_item.specific_issue:
                    issue_counts[feedback_item.specific_issue] = issue_counts.get(feedback_item.specific_issue, 0) + 1
        
        # Return issues that appear in multiple sessions
        common_issues = [
            issue for issue, count in issue_counts.items()
            if count >= max(2, len(feedback_sessions) * 0.3)
        ]
        
        return common_issues
    
    def _analyze_parameter_preferences(self, feedback_sessions: List[FeedbackSession]) -> Dict[str, Any]:
        """Analyze parameter preferences from feedback"""
        
        preferences = {
            'preferred_verbosity_direction': 0.0,  # Positive = more verbose
            'preferred_depth_direction': 0.0,      # Positive = more depth
            'tone_preferences': [],
            'satisfaction_by_mode': {}
        }
        
        for session in feedback_sessions:
            # Analyze verbosity preference
            for item in session.feedback_items:
                if item.feedback_type == FeedbackType.VERBOSITY_FEEDBACK:
                    if "increase" in item.suggested_adjustment:
                        preferences['preferred_verbosity_direction'] += item.intensity
                    elif "decrease" in item.suggested_adjustment:
                        preferences['preferred_verbosity_direction'] -= item.intensity
                
                elif item.feedback_type == FeedbackType.DEPTH_FEEDBACK:
                    if "increase" in item.suggested_adjustment:
                        preferences['preferred_depth_direction'] += item.intensity
                    elif "decrease" in item.suggested_adjustment:
                        preferences['preferred_depth_direction'] -= item.intensity
            
            # Track satisfaction by reasoning mode
            mode = session.original_parameters.reasoning_mode
            if mode not in preferences['satisfaction_by_mode']:
                preferences['satisfaction_by_mode'][mode] = []
            preferences['satisfaction_by_mode'][mode].append(session.overall_satisfaction)
        
        return preferences
    
    def _analyze_adjustment_effectiveness(self, feedback_sessions: List[FeedbackSession]) -> Dict[str, float]:
        """Analyze effectiveness of past adjustments"""
        
        effectiveness = {}
        
        # This would compare feedback before and after adjustments
        # For now, return placeholder analysis
        
        for session in feedback_sessions:
            for adjustment in session.parameter_adjustments:
                adj_type = adjustment.get('type', 'unknown')
                if adj_type not in effectiveness:
                    effectiveness[adj_type] = []
                
                # Use satisfaction as effectiveness proxy
                effectiveness[adj_type].append(session.overall_satisfaction)
        
        # Calculate average effectiveness
        avg_effectiveness = {}
        for adj_type, satisfactions in effectiveness.items():
            if satisfactions:
                avg_effectiveness[adj_type] = sum(satisfactions) / len(satisfactions)
        
        return avg_effectiveness
    
    def _analyze_preference_evolution(self,
                                    feedback_sessions: List[FeedbackSession],
                                    user_profile: Optional[UserPreferenceProfile]) -> Dict[str, Any]:
        """Analyze how user preferences have evolved"""
        
        if not user_profile:
            return {}
        
        evolution = {
            'preference_stability': 0.0,
            'trending_preferences': {},
            'learning_confidence': 0.0
        }
        
        # Analyze preference stability over time
        recent_preferences = []
        for session in feedback_sessions[-5:]:  # Last 5 sessions
            prefs = {
                'verbosity': session.original_parameters.verbosity,
                'reasoning_mode': session.original_parameters.reasoning_mode,
                'depth': session.original_parameters.depth
            }
            recent_preferences.append(prefs)
        
        # Calculate stability (how consistent preferences are)
        if len(recent_preferences) > 1:
            stability_score = self._calculate_preference_stability(recent_preferences)
            evolution['preference_stability'] = stability_score
        
        return evolution
    
    def _calculate_preference_stability(self, preferences_list: List[Dict[str, Any]]) -> float:
        """Calculate stability of preferences over time"""
        
        if len(preferences_list) < 2:
            return 1.0
        
        stability_scores = []
        
        for param in ['verbosity', 'reasoning_mode', 'depth']:
            values = [prefs.get(param) for prefs in preferences_list]
            unique_values = len(set(values))
            stability = 1.0 - (unique_values - 1) / len(values)
            stability_scores.append(stability)
        
        return sum(stability_scores) / len(stability_scores)
    
    # ========================================================================
    # Parameter Adjustment Application
    # ========================================================================
    
    async def _apply_adjustments(self,
                               original_parameters: UserParameters,
                               adjustments: List[ParameterAdjustment]) -> UserParameters:
        """Apply parameter adjustments to create new parameter set"""
        
        # Create copy of original parameters
        adjusted_dict = {
            'verbosity': original_parameters.verbosity,
            'reasoning_mode': original_parameters.reasoning_mode,
            'depth': original_parameters.depth,
            'synthesis_focus': original_parameters.synthesis_focus,
            'citation_style': original_parameters.citation_style,
            'uncertainty_handling': original_parameters.uncertainty_handling,
            'tone': original_parameters.tone
        }
        
        # Apply adjustments with dampening to prevent overcorrection
        for adjustment in adjustments:
            if adjustment.confidence >= self.learning_config['confidence_threshold']:
                param_name = adjustment.parameter_name
                if param_name in adjusted_dict:
                    adjusted_dict[param_name] = adjustment.suggested_value
        
        # Create new UserParameters object
        return UserParameters(
            verbosity=adjusted_dict['verbosity'],
            reasoning_mode=adjusted_dict['reasoning_mode'],
            depth=adjusted_dict['depth'],
            synthesis_focus=adjusted_dict['synthesis_focus'],
            citation_style=adjusted_dict['citation_style'],
            uncertainty_handling=adjusted_dict['uncertainty_handling'],
            tone=adjusted_dict['tone']
        )
    
    async def _update_effectiveness_tracking(self,
                                           feedback_session: FeedbackSession,
                                           adjustments: List[ParameterAdjustment]):
        """Update tracking of adjustment effectiveness"""
        
        for adjustment in adjustments:
            adj_key = f"{adjustment.parameter_name}_{adjustment.suggested_value}"
            
            if adj_key not in self.adjustment_success_rates:
                self.adjustment_success_rates[adj_key] = []
            
            # Use satisfaction as success metric
            success_score = feedback_session.overall_satisfaction
            self.adjustment_success_rates[adj_key].append(success_score)
            
            # Keep only recent history
            if len(self.adjustment_success_rates[adj_key]) > 20:
                self.adjustment_success_rates[adj_key] = self.adjustment_success_rates[adj_key][-20:]
    
    def _calculate_learning_rate(self, user_id: Optional[str]) -> float:
        """Calculate learning rate for user"""
        
        if not user_id or user_id not in self.feedback_history:
            return 1.0  # Full learning rate for new users
        
        session_count = len(self.feedback_history[user_id])
        
        # Decrease learning rate as user provides more feedback
        learning_rate = self.learning_config['learning_rate_decay'] ** session_count
        
        return max(learning_rate, 0.1)  # Minimum learning rate
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    async def _get_user_choice(self, options: List[str], default: str) -> str:
        """Get user choice from options (simplified for demo)"""
        
        # In a real implementation, this would handle user input
        # For demo purposes, return default with some variation
        print(f"[Simulated choice: {default}]")
        return default
    
    def get_feedback_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback statistics for analysis"""
        
        if user_id and user_id in self.feedback_history:
            user_sessions = self.feedback_history[user_id]
            return {
                'total_sessions': len(user_sessions),
                'average_satisfaction': sum(s.overall_satisfaction for s in user_sessions) / len(user_sessions),
                'recent_satisfaction': user_sessions[-5:] if len(user_sessions) >= 5 else user_sessions,
                'common_adjustments': self._get_common_adjustments(user_sessions)
            }
        else:
            all_sessions = [session for sessions in self.feedback_history.values() for session in sessions]
            if all_sessions:
                return {
                    'total_sessions': len(all_sessions),
                    'average_satisfaction': sum(s.overall_satisfaction for s in all_sessions) / len(all_sessions),
                    'total_users': len(self.feedback_history)
                }
        
        return {'no_data': True}
    
    def _get_common_adjustments(self, sessions: List[FeedbackSession]) -> List[str]:
        """Get common adjustment types for sessions"""
        
        adjustments = []
        for session in sessions:
            for item in session.feedback_items:
                if item.suggested_adjustment:
                    adjustments.append(item.suggested_adjustment)
        
        # Count and return most common
        adjustment_counts = {}
        for adj in adjustments:
            adjustment_counts[adj] = adjustment_counts.get(adj, 0) + 1
        
        return sorted(adjustment_counts.keys(), key=adjustment_counts.get, reverse=True)[:3]


# ============================================================================
# Specialized Feedback Analysis Components
# ============================================================================

class FeedbackAnalyzer:
    """Analyzes feedback sessions to extract insights"""
    
    async def analyze_feedback_session(self, session: FeedbackSession) -> Dict[str, Any]:
        """Analyze a single feedback session"""
        
        analysis = {
            'dominant_issues': [],
            'satisfaction_score': session.overall_satisfaction,
            'satisfaction_trend': 0.0,  # Would need multiple sessions
            'parameter_effectiveness': {},
            'specific_recommendations': []
        }
        
        # Identify dominant issues
        issue_severity = {}
        for item in session.feedback_items:
            if item.specific_issue:
                severity = item.intensity * (1.0 if item.sentiment in [FeedbackSentiment.NEGATIVE, FeedbackSentiment.VERY_NEGATIVE] else 0.0)
                issue_severity[item.specific_issue] = max(issue_severity.get(item.specific_issue, 0.0), severity)
        
        # Sort by severity
        sorted_issues = sorted(issue_severity.items(), key=lambda x: x[1], reverse=True)
        analysis['dominant_issues'] = [issue for issue, severity in sorted_issues if severity > 0.3]
        
        # Generate recommendations
        for item in session.feedback_items:
            if item.suggested_adjustment and item.intensity > 0.4:
                analysis['specific_recommendations'].append({
                    'adjustment': item.suggested_adjustment,
                    'confidence': item.confidence,
                    'priority': item.intensity
                })
        
        return analysis


class AdjustmentGenerator:
    """Generates parameter adjustments based on feedback analysis"""
    
    async def generate_adjustments(self,
                                 current_analysis: Dict[str, Any],
                                 historical_analysis: Dict[str, Any],
                                 current_parameters: UserParameters) -> List[ParameterAdjustment]:
        """Generate parameter adjustments"""
        
        adjustments = []
        
        # Process specific recommendations
        for rec in current_analysis.get('specific_recommendations', []):
            adjustment = await self._create_adjustment_from_recommendation(
                rec, current_parameters
            )
            if adjustment:
                adjustments.append(adjustment)
        
        # Add historical pattern adjustments
        if historical_analysis:
            historical_adjustments = await self._create_historical_adjustments(
                historical_analysis, current_parameters
            )
            adjustments.extend(historical_adjustments)
        
        return adjustments
    
    async def _create_adjustment_from_recommendation(self,
                                                   recommendation: Dict[str, Any],
                                                   current_parameters: UserParameters) -> Optional[ParameterAdjustment]:
        """Create parameter adjustment from feedback recommendation"""
        
        adjustment_mapping = {
            'increase_verbosity': ('verbosity', self._increase_verbosity),
            'decrease_verbosity': ('verbosity', self._decrease_verbosity),
            'increase_depth': ('depth', self._increase_depth),
            'decrease_depth': ('depth', self._decrease_depth),
            'improve_clarity': ('tone', self._improve_clarity),
            'improve_relevance': ('synthesis_focus', self._improve_relevance)
        }
        
        adj_type = recommendation['adjustment']
        if adj_type in adjustment_mapping:
            param_name, adjustment_func = adjustment_mapping[adj_type]
            new_value = adjustment_func(getattr(current_parameters, param_name))
            
            if new_value != getattr(current_parameters, param_name):
                return ParameterAdjustment(
                    parameter_name=param_name,
                    current_value=getattr(current_parameters, param_name),
                    suggested_value=new_value,
                    adjustment_rationale=f"User feedback: {adj_type}",
                    confidence=recommendation['confidence'],
                    expected_improvement=recommendation['priority']
                )
        
        return None
    
    def _increase_verbosity(self, current: str) -> str:
        mapping = {'BRIEF': 'MODERATE', 'MODERATE': 'COMPREHENSIVE', 'COMPREHENSIVE': 'EXHAUSTIVE'}
        return mapping.get(current, current)
    
    def _decrease_verbosity(self, current: str) -> str:
        mapping = {'EXHAUSTIVE': 'COMPREHENSIVE', 'COMPREHENSIVE': 'MODERATE', 'MODERATE': 'BRIEF'}
        return mapping.get(current, current)
    
    def _increase_depth(self, current: str) -> str:
        mapping = {'SURFACE': 'INTERMEDIATE', 'INTERMEDIATE': 'DEEP', 'DEEP': 'EXHAUSTIVE'}
        return mapping.get(current, current)
    
    def _decrease_depth(self, current: str) -> str:
        mapping = {'EXHAUSTIVE': 'DEEP', 'DEEP': 'INTERMEDIATE', 'INTERMEDIATE': 'SURFACE'}
        return mapping.get(current, current)
    
    def _improve_clarity(self, current) -> Any:
        # Placeholder - would implement tone adjustment for clarity
        return current
    
    def _improve_relevance(self, current: str) -> str:
        # Switch to insights focus for better relevance
        return 'INSIGHTS'
    
    async def _create_historical_adjustments(self,
                                           historical_analysis: Dict[str, Any],
                                           current_parameters: UserParameters) -> List[ParameterAdjustment]:
        """Create adjustments based on historical patterns"""
        
        adjustments = []
        
        # Example: If user consistently prefers more verbosity
        prefs = historical_analysis.get('parameter_preferences', {})
        if prefs.get('preferred_verbosity_direction', 0) > 0.5:
            new_verbosity = self._increase_verbosity(current_parameters.verbosity)
            if new_verbosity != current_parameters.verbosity:
                adjustments.append(ParameterAdjustment(
                    parameter_name='verbosity',
                    current_value=current_parameters.verbosity,
                    suggested_value=new_verbosity,
                    adjustment_rationale='Historical preference for more detailed responses',
                    confidence=0.7,
                    expected_improvement=0.3
                ))
        
        return adjustments


class LearningOptimizer:
    """Optimizes adjustments based on learning algorithms"""
    
    async def optimize_adjustments(self,
                                 adjustments: List[ParameterAdjustment],
                                 feedback_session: FeedbackSession,
                                 user_profile: Optional[UserPreferenceProfile]) -> List[ParameterAdjustment]:
        """Optimize adjustments using learning algorithms"""
        
        optimized = []
        
        for adjustment in adjustments:
            # Apply confidence dampening to prevent overcorrection
            dampened_confidence = adjustment.confidence * 0.8
            
            # Boost confidence if adjustment aligns with user profile
            if user_profile:
                profile_boost = await self._calculate_profile_alignment(adjustment, user_profile)
                dampened_confidence += profile_boost * 0.2
            
            # Only include high-confidence adjustments
            if dampened_confidence >= 0.6:
                optimized_adjustment = ParameterAdjustment(
                    parameter_name=adjustment.parameter_name,
                    current_value=adjustment.current_value,
                    suggested_value=adjustment.suggested_value,
                    adjustment_rationale=adjustment.adjustment_rationale,
                    confidence=dampened_confidence,
                    expected_improvement=adjustment.expected_improvement
                )
                optimized.append(optimized_adjustment)
        
        return optimized
    
    async def _calculate_profile_alignment(self,
                                         adjustment: ParameterAdjustment,
                                         user_profile: UserPreferenceProfile) -> float:
        """Calculate how well adjustment aligns with user profile"""
        
        # Simple alignment check - would be more sophisticated in practice
        if adjustment.parameter_name == 'verbosity':
            if adjustment.suggested_value == user_profile.preferred_verbosity.value:
                return 1.0
            elif abs(self._verbosity_to_numeric(adjustment.suggested_value) - 
                    self._verbosity_to_numeric(user_profile.preferred_verbosity.value)) <= 1:
                return 0.5
        
        return 0.0
    
    def _verbosity_to_numeric(self, verbosity: str) -> int:
        mapping = {'BRIEF': 1, 'MODERATE': 2, 'COMPREHENSIVE': 3, 'EXHAUSTIVE': 4}
        return mapping.get(verbosity, 2)