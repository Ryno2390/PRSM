"""
Feedback Processor
==================

Process and analyze user feedback.
Stores feedback for analysis and improvement.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import uuid

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Process user feedback for learning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._feedback_store: List[Dict[str, Any]] = []
        self._stats = {
            "total_processed": 0,
            "by_type": {},
            "by_rating": {}
        }

    async def process(self, feedback: Dict[str, Any]) -> bool:
        """Process feedback."""
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Normalize feedback data
        processed = {
            "id": feedback_id,
            "timestamp": timestamp,
            "query": feedback.get("query", ""),
            "rating": feedback.get("rating"),
            "feedback": feedback.get("feedback", ""),
            "user_id": feedback.get("user_id", "anonymous"),
            "metadata": feedback.get("metadata", {})
        }

        # Store feedback
        self._feedback_store.append(processed)
        self._stats["total_processed"] += 1

        # Track by type
        feedback_type = feedback.get("type", "general")
        self._stats["by_type"][feedback_type] = self._stats["by_type"].get(feedback_type, 0) + 1

        # Track by rating
        rating = feedback.get("rating")
        if rating is not None:
            rating_key = str(rating)
            self._stats["by_rating"][rating_key] = self._stats["by_rating"].get(rating_key, 0) + 1

        logger.debug(f"Processed feedback {feedback_id}")
        return True

    async def get_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback."""
        return self._feedback_store[-limit:]

    async def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return dict(self._stats)

    async def get_average_rating(self) -> Optional[float]:
        """Get average rating from feedback."""
        ratings = [f["rating"] for f in self._feedback_store if f.get("rating") is not None]
        if ratings:
            return sum(ratings) / len(ratings)
        return None
