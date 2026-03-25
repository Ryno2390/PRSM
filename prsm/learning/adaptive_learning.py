"""
Adaptive Learning System
========================

Adaptive learning for continuous improvement.
Accumulates feedback and tracks improvements.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import uuid

logger = logging.getLogger(__name__)


class AdaptiveLearningSystem:
    """System for adaptive learning and improvement."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._feedback_processor = None
        self._improvement_log: List[Dict[str, Any]] = []
        self._session_stats = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "started_at": datetime.now(timezone.utc).isoformat()
        }

    def _get_feedback_processor(self):
        """Lazy initialization of feedback processor."""
        if self._feedback_processor is None:
            try:
                from prsm.learning.feedback_processor import FeedbackProcessor
                self._feedback_processor = FeedbackProcessor(self.config)
            except Exception as e:
                logger.debug(f"Could not initialize FeedbackProcessor: {e}")
        return self._feedback_processor

    async def learn(self, data: Dict[str, Any]) -> bool:
        """Learn from feedback data."""
        self._session_stats["total_feedback"] += 1

        # Track sentiment
        feedback_type = data.get("feedback", data.get("rating", "neutral"))
        if isinstance(feedback_type, str):
            if feedback_type.lower() in ("positive", "good", "helpful"):
                self._session_stats["positive_feedback"] += 1
            elif feedback_type.lower() in ("negative", "bad", "unhelpful"):
                self._session_stats["negative_feedback"] += 1
        elif isinstance(feedback_type, (int, float)):
            if feedback_type >= 4:
                self._session_stats["positive_feedback"] += 1
            elif feedback_type <= 2:
                self._session_stats["negative_feedback"] += 1

        # Record improvement suggestion
        improvement = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "stats": dict(self._session_stats)
        }
        self._improvement_log.append(improvement)

        # Delegate to feedback processor if available
        processor = self._get_feedback_processor()
        if processor:
            try:
                await processor.process(data)
            except Exception as e:
                logger.debug(f"Feedback processing failed: {e}")

        return True

    async def get_improvements(self) -> List[Dict[str, Any]]:
        """Get suggested improvements."""
        improvements = []

        # Analyze feedback patterns
        if self._session_stats["total_feedback"] > 0:
            positive_ratio = self._session_stats["positive_feedback"] / self._session_stats["total_feedback"]

            if positive_ratio < 0.5:
                improvements.append({
                    "type": "quality",
                    "priority": "high",
                    "suggestion": "Review recent responses for quality issues",
                    "confidence": 1.0 - positive_ratio
                })

            if self._session_stats["negative_feedback"] > 5:
                improvements.append({
                    "type": "investigation",
                    "priority": "medium",
                    "suggestion": "Investigate patterns in negative feedback",
                    "count": self._session_stats["negative_feedback"]
                })

        # Add recent improvement entries
        improvements.extend(self._improvement_log[-5:])

        return improvements

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return dict(self._session_stats)
