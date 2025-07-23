#!/usr/bin/env python3
"""
Review System for Marketplace Ecosystem
======================================

Comprehensive review and rating system for marketplace integrations
with advanced analytics, moderation, and quality metrics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from decimal import Decimal
import json
import asyncio
import hashlib
import re
from collections import defaultdict, Counter


class ReviewRating(Enum):
    """Review rating levels"""
    ONE_STAR = 1
    TWO_STAR = 2
    THREE_STAR = 3
    FOUR_STAR = 4
    FIVE_STAR = 5


class ReviewStatus(Enum):
    """Review processing status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    SPAM = "spam"


class ReviewHelpfulness(Enum):
    """Review helpfulness feedback"""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    SPAM_REPORT = "spam_report"


@dataclass
class ReviewMetrics:
    """Comprehensive review metrics"""
    total_reviews: int = 0
    average_rating: Decimal = Decimal('0.0')
    rating_distribution: Dict[int, int] = None
    recent_reviews_count: int = 0
    helpfulness_ratio: Decimal = Decimal('0.0')
    response_rate: Decimal = Decimal('0.0')
    sentiment_score: Decimal = Decimal('0.0')
    
    def __post_init__(self):
        if self.rating_distribution is None:
            self.rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}


@dataclass
class Review:
    """Individual review data structure"""
    id: str
    integration_id: str
    reviewer_id: str
    rating: ReviewRating
    title: str
    content: str
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    helpful_votes: int = 0
    not_helpful_votes: int = 0
    spam_reports: int = 0
    developer_response: Optional[str] = None
    response_date: Optional[datetime] = None
    verified_purchase: bool = False
    usage_duration: Optional[timedelta] = None
    tags: Set[str] = None
    sentiment_score: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = set()


@dataclass
class ReviewerProfile:
    """Reviewer profile and history"""
    id: str
    username: str
    review_count: int = 0
    helpful_review_ratio: Decimal = Decimal('0.0')
    average_rating_given: Decimal = Decimal('0.0')
    verified_reviewer: bool = False
    expert_categories: Set[str] = None
    reputation_score: int = 0
    join_date: datetime = None
    
    def __post_init__(self):
        if self.expert_categories is None:
            self.expert_categories = set()
        if self.join_date is None:
            self.join_date = datetime.utcnow()


class SentimentAnalyzer:
    """Simple sentiment analysis for reviews"""
    
    def __init__(self):
        self.positive_words = {
            'excellent', 'amazing', 'fantastic', 'wonderful', 'great', 'good',
            'love', 'perfect', 'outstanding', 'brilliant', 'awesome', 'superb',
            'recommended', 'useful', 'helpful', 'easy', 'fast', 'reliable'
        }
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'useless',
            'broken', 'buggy', 'slow', 'difficult', 'confusing', 'unreliable',
            'disappointing', 'frustrating', 'annoying', 'poor', 'worthless'
        }
    
    def analyze_sentiment(self, text: str) -> Decimal:
        """Analyze sentiment of review text (-1.0 to 1.0)"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return Decimal('0.0')
        
        sentiment = (positive_count - negative_count) / total_sentiment_words
        return Decimal(str(round(sentiment, 2)))
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from review text"""
        # Simple key phrase extraction
        phrases = []
        
        # Look for common patterns
        patterns = [
            r'easy to \w+',
            r'hard to \w+',
            r'works \w+',
            r'doesn\'t work',
            r'great for \w+',
            r'perfect for \w+',
            r'not good for \w+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            phrases.extend(matches)
        
        return phrases[:5]  # Return top 5 phrases


class ReviewModerator:
    """Automated review moderation system"""
    
    def __init__(self):
        self.spam_patterns = [
            r'check out my \w+',
            r'visit my website',
            r'click here',
            r'http[s]?://\S+',
            r'www\.\S+',
            r'buy now',
            r'limited time offer'
        ]
        self.profanity_words = {
            # Basic profanity filter - would use comprehensive list in production
            'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron'
        }
    
    def moderate_review(self, review: Review) -> Tuple[ReviewStatus, List[str]]:
        """Moderate review and return status with reasons"""
        flags = []
        
        # Check for spam patterns
        combined_text = f"{review.title} {review.content}".lower()
        for pattern in self.spam_patterns:
            if re.search(pattern, combined_text):
                flags.append(f"Potential spam: {pattern}")
        
        # Check for profanity
        words = re.findall(r'\b\w+\b', combined_text)
        profanity_found = [word for word in words if word in self.profanity_words]
        if profanity_found:
            flags.append(f"Profanity detected: {', '.join(profanity_found)}")
        
        # Check review length
        if len(review.content.strip()) < 10:
            flags.append("Review too short")
        
        # Check for repeated characters
        if re.search(r'(.)\1{4,}', combined_text):
            flags.append("Excessive repeated characters")
        
        # Check rating vs content mismatch
        sentiment_analyzer = SentimentAnalyzer()
        sentiment = sentiment_analyzer.analyze_sentiment(review.content)
        rating_value = review.rating.value
        
        # Significant mismatch between rating and sentiment
        if rating_value >= 4 and sentiment < Decimal('-0.5'):
            flags.append("Rating-sentiment mismatch (positive rating, negative content)")
        elif rating_value <= 2 and sentiment > Decimal('0.5'):
            flags.append("Rating-sentiment mismatch (negative rating, positive content)")
        
        # Determine status
        if len(flags) >= 3:
            return ReviewStatus.REJECTED, flags
        elif len(flags) >= 1:
            return ReviewStatus.FLAGGED, flags
        else:
            return ReviewStatus.APPROVED, []
    
    def check_reviewer_history(self, reviewer_id: str, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze reviewer history for suspicious patterns"""
        reviewer_reviews = [r for r in reviews if r.reviewer_id == reviewer_id]
        
        analysis = {
            'total_reviews': len(reviewer_reviews),
            'suspicious_patterns': [],
            'reputation_score': 0
        }
        
        if len(reviewer_reviews) < 2:
            return analysis
        
        # Check for review bombing (many reviews in short time)
        recent_reviews = [r for r in reviewer_reviews 
                         if r.created_at > datetime.utcnow() - timedelta(days=1)]
        if len(recent_reviews) > 5:
            analysis['suspicious_patterns'].append("Potential review bombing")
        
        # Check for rating consistency (all same rating)
        ratings = [r.rating.value for r in reviewer_reviews]
        if len(set(ratings)) == 1 and len(ratings) > 3:
            analysis['suspicious_patterns'].append("Suspicious rating consistency")
        
        # Check for duplicate content
        contents = [r.content for r in reviewer_reviews]
        if len(set(contents)) < len(contents) * 0.8:
            analysis['suspicious_patterns'].append("Duplicate content detected")
        
        # Calculate reputation score
        helpful_ratio = Decimal('0.0')
        if reviewer_reviews:
            total_helpful = sum(r.helpful_votes for r in reviewer_reviews)
            total_votes = sum(r.helpful_votes + r.not_helpful_votes for r in reviewer_reviews)
            if total_votes > 0:
                helpful_ratio = Decimal(total_helpful) / Decimal(total_votes)
        
        base_score = min(len(reviewer_reviews) * 10, 100)
        helpful_bonus = int(helpful_ratio * 50)
        suspicious_penalty = len(analysis['suspicious_patterns']) * 20
        
        analysis['reputation_score'] = max(0, base_score + helpful_bonus - suspicious_penalty)
        
        return analysis


class ReviewAnalytics:
    """Advanced analytics for review system"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def calculate_integration_metrics(self, integration_id: str, reviews: List[Review]) -> ReviewMetrics:
        """Calculate comprehensive metrics for an integration"""
        integration_reviews = [r for r in reviews if r.integration_id == integration_id]
        
        if not integration_reviews:
            return ReviewMetrics()
        
        # Basic metrics
        total_reviews = len(integration_reviews)
        ratings = [r.rating.value for r in integration_reviews]
        average_rating = Decimal(sum(ratings)) / Decimal(total_reviews)
        
        # Rating distribution
        rating_distribution = Counter(ratings)
        distribution_dict = {i: rating_distribution.get(i, 0) for i in range(1, 6)}
        
        # Recent reviews (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        recent_reviews = [r for r in integration_reviews if r.created_at > cutoff_date]
        
        # Helpfulness ratio
        total_helpful = sum(r.helpful_votes for r in integration_reviews)
        total_votes = sum(r.helpful_votes + r.not_helpful_votes for r in integration_reviews)
        helpfulness_ratio = Decimal('0.0')
        if total_votes > 0:
            helpfulness_ratio = Decimal(total_helpful) / Decimal(total_votes)
        
        # Developer response rate
        reviews_with_response = [r for r in integration_reviews if r.developer_response]
        response_rate = Decimal('0.0')
        if integration_reviews:
            response_rate = Decimal(len(reviews_with_response)) / Decimal(total_reviews)
        
        # Average sentiment score
        sentiment_scores = []
        for review in integration_reviews:
            if review.sentiment_score is not None:
                sentiment_scores.append(review.sentiment_score)
        
        avg_sentiment = Decimal('0.0')
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / Decimal(len(sentiment_scores))
        
        return ReviewMetrics(
            total_reviews=total_reviews,
            average_rating=average_rating,
            rating_distribution=distribution_dict,
            recent_reviews_count=len(recent_reviews),
            helpfulness_ratio=helpfulness_ratio,
            response_rate=response_rate,
            sentiment_score=avg_sentiment
        )
    
    def get_trending_reviews(self, reviews: List[Review], days: int = 7) -> List[Review]:
        """Get trending reviews based on engagement"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_reviews = [r for r in reviews if r.created_at > cutoff_date]
        
        # Score reviews based on engagement
        scored_reviews = []
        for review in recent_reviews:
            engagement_score = (
                review.helpful_votes * 2 +
                review.not_helpful_votes * 0.5 +
                (10 if review.developer_response else 0) +
                (5 if review.verified_purchase else 0)
            )
            scored_reviews.append((review, engagement_score))
        
        # Sort by engagement score
        scored_reviews.sort(key=lambda x: x[1], reverse=True)
        return [review for review, score in scored_reviews[:20]]
    
    def analyze_review_trends(self, integration_id: str, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze review trends over time"""
        integration_reviews = [r for r in reviews if r.integration_id == integration_id]
        integration_reviews.sort(key=lambda r: r.created_at)
        
        # Monthly aggregation
        monthly_data = defaultdict(lambda: {'count': 0, 'ratings': [], 'sentiment': []})
        
        for review in integration_reviews:
            month_key = review.created_at.strftime('%Y-%m')
            monthly_data[month_key]['count'] += 1
            monthly_data[month_key]['ratings'].append(review.rating.value)
            if review.sentiment_score is not None:
                monthly_data[month_key]['sentiment'].append(float(review.sentiment_score))
        
        # Calculate trends
        trends = {}
        for month, data in monthly_data.items():
            avg_rating = sum(data['ratings']) / len(data['ratings']) if data['ratings'] else 0
            avg_sentiment = sum(data['sentiment']) / len(data['sentiment']) if data['sentiment'] else 0
            
            trends[month] = {
                'review_count': data['count'],
                'average_rating': round(avg_rating, 2),
                'average_sentiment': round(avg_sentiment, 2)
            }
        
        return trends
    
    def generate_review_insights(self, integration_id: str, reviews: List[Review]) -> Dict[str, Any]:
        """Generate actionable insights from reviews"""
        integration_reviews = [r for r in reviews if r.integration_id == integration_id]
        
        insights = {
            'common_praise': [],
            'common_complaints': [],
            'improvement_suggestions': [],
            'feature_requests': [],
            'user_segments': {}
        }
        
        # Analyze review content for patterns
        positive_reviews = [r for r in integration_reviews if r.rating.value >= 4]
        negative_reviews = [r for r in integration_reviews if r.rating.value <= 2]
        
        # Extract common themes from positive reviews
        positive_phrases = []
        for review in positive_reviews:
            phrases = self.sentiment_analyzer.extract_key_phrases(review.content)
            positive_phrases.extend(phrases)
        
        if positive_phrases:
            phrase_counts = Counter(positive_phrases)
            insights['common_praise'] = phrase_counts.most_common(5)
        
        # Extract common themes from negative reviews
        negative_phrases = []
        for review in negative_reviews:
            phrases = self.sentiment_analyzer.extract_key_phrases(review.content)
            negative_phrases.extend(phrases)
        
        if negative_phrases:
            phrase_counts = Counter(negative_phrases)
            insights['common_complaints'] = phrase_counts.most_common(5)
        
        # Analyze user segments
        verified_users = [r for r in integration_reviews if r.verified_purchase]
        new_users = [r for r in integration_reviews if r.usage_duration and r.usage_duration < timedelta(days=7)]
        experienced_users = [r for r in integration_reviews if r.usage_duration and r.usage_duration > timedelta(days=30)]
        
        insights['user_segments'] = {
            'verified_users': {
                'count': len(verified_users),
                'avg_rating': sum(r.rating.value for r in verified_users) / len(verified_users) if verified_users else 0
            },
            'new_users': {
                'count': len(new_users),
                'avg_rating': sum(r.rating.value for r in new_users) / len(new_users) if new_users else 0
            },
            'experienced_users': {
                'count': len(experienced_users),
                'avg_rating': sum(r.rating.value for r in experienced_users) / len(experienced_users) if experienced_users else 0
            }
        }
        
        return insights


class ReviewSystem:
    """Main review system coordinator"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./review_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.reviews: Dict[str, Review] = {}
        self.reviewers: Dict[str, ReviewerProfile] = {}
        
        # System components
        self.moderator = ReviewModerator()
        self.analytics = ReviewAnalytics()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load review data from storage"""
        try:
            reviews_file = self.storage_path / "reviews.json"
            if reviews_file.exists():
                with open(reviews_file, 'r') as f:
                    data = json.load(f)
                    for review_data in data:
                        review = Review(**review_data)
                        # Convert datetime strings back to datetime objects
                        review.created_at = datetime.fromisoformat(review_data['created_at'])
                        review.updated_at = datetime.fromisoformat(review_data['updated_at'])
                        if review_data.get('response_date'):
                            review.response_date = datetime.fromisoformat(review_data['response_date'])
                        self.reviews[review.id] = review
            
            reviewers_file = self.storage_path / "reviewers.json"
            if reviewers_file.exists():
                with open(reviewers_file, 'r') as f:
                    data = json.load(f)
                    for reviewer_data in data:
                        reviewer = ReviewerProfile(**reviewer_data)
                        reviewer.join_date = datetime.fromisoformat(reviewer_data['join_date'])
                        self.reviewers[reviewer.id] = reviewer
        
        except Exception as e:
            print(f"Error loading review data: {e}")
    
    def _save_data(self):
        """Save review data to storage"""
        try:
            # Save reviews
            reviews_data = []
            for review in self.reviews.values():
                review_dict = {
                    'id': review.id,
                    'integration_id': review.integration_id,
                    'reviewer_id': review.reviewer_id,
                    'rating': review.rating.value,
                    'title': review.title,
                    'content': review.content,
                    'status': review.status.value,
                    'created_at': review.created_at.isoformat(),
                    'updated_at': review.updated_at.isoformat(),
                    'helpful_votes': review.helpful_votes,
                    'not_helpful_votes': review.not_helpful_votes,
                    'spam_reports': review.spam_reports,
                    'developer_response': review.developer_response,
                    'response_date': review.response_date.isoformat() if review.response_date else None,
                    'verified_purchase': review.verified_purchase,
                    'usage_duration': str(review.usage_duration) if review.usage_duration else None,
                    'tags': list(review.tags),
                    'sentiment_score': float(review.sentiment_score) if review.sentiment_score else None
                }
                reviews_data.append(review_dict)
            
            with open(self.storage_path / "reviews.json", 'w') as f:
                json.dump(reviews_data, f, indent=2)
            
            # Save reviewers
            reviewers_data = []
            for reviewer in self.reviewers.values():
                reviewer_dict = {
                    'id': reviewer.id,
                    'username': reviewer.username,
                    'review_count': reviewer.review_count,
                    'helpful_review_ratio': float(reviewer.helpful_review_ratio),
                    'average_rating_given': float(reviewer.average_rating_given),
                    'verified_reviewer': reviewer.verified_reviewer,
                    'expert_categories': list(reviewer.expert_categories),
                    'reputation_score': reviewer.reputation_score,
                    'join_date': reviewer.join_date.isoformat()
                }
                reviewers_data.append(reviewer_dict)
            
            with open(self.storage_path / "reviewers.json", 'w') as f:
                json.dump(reviewers_data, f, indent=2)
        
        except Exception as e:
            print(f"Error saving review data: {e}")
    
    def submit_review(self, integration_id: str, reviewer_id: str, rating: ReviewRating,
                     title: str, content: str, verified_purchase: bool = False,
                     usage_duration: Optional[timedelta] = None) -> str:
        """Submit a new review"""
        # Generate review ID
        review_id = hashlib.md5(f"{integration_id}_{reviewer_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
        
        # Create review
        review = Review(
            id=review_id,
            integration_id=integration_id,
            reviewer_id=reviewer_id,
            rating=rating,
            title=title,
            content=content,
            verified_purchase=verified_purchase,
            usage_duration=usage_duration
        )
        
        # Analyze sentiment
        review.sentiment_score = self.sentiment_analyzer.analyze_sentiment(content)
        
        # Moderate review
        status, flags = self.moderator.moderate_review(review)
        review.status = status
        
        # Store review
        self.reviews[review_id] = review
        
        # Update reviewer profile
        if reviewer_id not in self.reviewers:
            self.reviewers[reviewer_id] = ReviewerProfile(
                id=reviewer_id,
                username=f"User_{reviewer_id[:8]}"
            )
        
        reviewer = self.reviewers[reviewer_id]
        reviewer.review_count += 1
        
        # Recalculate reviewer stats
        reviewer_reviews = [r for r in self.reviews.values() if r.reviewer_id == reviewer_id]
        if reviewer_reviews:
            total_rating = sum(r.rating.value for r in reviewer_reviews)
            reviewer.average_rating_given = Decimal(total_rating) / Decimal(len(reviewer_reviews))
        
        self._save_data()
        return review_id
    
    def add_developer_response(self, review_id: str, response: str) -> bool:
        """Add developer response to review"""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        review.developer_response = response
        review.response_date = datetime.utcnow()
        review.updated_at = datetime.utcnow()
        
        self._save_data()
        return True
    
    def vote_on_review(self, review_id: str, vote_type: ReviewHelpfulness) -> bool:
        """Vote on review helpfulness"""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        if vote_type == ReviewHelpfulness.HELPFUL:
            review.helpful_votes += 1
        elif vote_type == ReviewHelpfulness.NOT_HELPFUL:
            review.not_helpful_votes += 1
        elif vote_type == ReviewHelpfulness.SPAM_REPORT:
            review.spam_reports += 1
            
            # Auto-flag if too many spam reports
            if review.spam_reports >= 5:
                review.status = ReviewStatus.SPAM
        
        review.updated_at = datetime.utcnow()
        self._save_data()
        return True
    
    def get_integration_reviews(self, integration_id: str, status: Optional[ReviewStatus] = None,
                               limit: Optional[int] = None) -> List[Review]:
        """Get reviews for an integration"""
        reviews = [r for r in self.reviews.values() if r.integration_id == integration_id]
        
        if status:
            reviews = [r for r in reviews if r.status == status]
        
        # Sort by helpfulness and recency
        reviews.sort(key=lambda r: (r.helpful_votes - r.not_helpful_votes, r.created_at), reverse=True)
        
        if limit:
            reviews = reviews[:limit]
        
        return reviews
    
    def get_integration_metrics(self, integration_id: str) -> ReviewMetrics:
        """Get comprehensive metrics for an integration"""
        return self.analytics.calculate_integration_metrics(integration_id, list(self.reviews.values()))
    
    def get_review_insights(self, integration_id: str) -> Dict[str, Any]:
        """Get actionable insights from reviews"""
        return self.analytics.generate_review_insights(integration_id, list(self.reviews.values()))
    
    def get_trending_reviews(self, days: int = 7) -> List[Review]:
        """Get trending reviews"""
        return self.analytics.get_trending_reviews(list(self.reviews.values()), days)
    
    def moderate_reviews(self) -> Dict[str, int]:
        """Run moderation on pending reviews"""
        results = {'approved': 0, 'flagged': 0, 'rejected': 0}
        
        pending_reviews = [r for r in self.reviews.values() if r.status == ReviewStatus.PENDING]
        
        for review in pending_reviews:
            status, flags = self.moderator.moderate_review(review)
            review.status = status
            review.updated_at = datetime.utcnow()
            
            if status == ReviewStatus.APPROVED:
                results['approved'] += 1
            elif status == ReviewStatus.FLAGGED:
                results['flagged'] += 1
            elif status == ReviewStatus.REJECTED:
                results['rejected'] += 1
        
        self._save_data()
        return results
    
    async def bulk_analyze_sentiment(self):
        """Analyze sentiment for all reviews missing sentiment scores"""
        reviews_to_analyze = [r for r in self.reviews.values() if r.sentiment_score is None]
        
        for review in reviews_to_analyze:
            review.sentiment_score = self.sentiment_analyzer.analyze_sentiment(review.content)
            review.updated_at = datetime.utcnow()
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
        
        if reviews_to_analyze:
            self._save_data()
        
        return len(reviews_to_analyze)
    
    def export_reviews(self, integration_id: Optional[str] = None) -> Dict[str, Any]:
        """Export review data for analysis"""
        reviews_to_export = list(self.reviews.values())
        
        if integration_id:
            reviews_to_export = [r for r in reviews_to_export if r.integration_id == integration_id]
        
        export_data = {
            'metadata': {
                'export_date': datetime.utcnow().isoformat(),
                'total_reviews': len(reviews_to_export),
                'integration_id': integration_id
            },
            'reviews': [],
            'summary_metrics': {}
        }
        
        # Export review data
        for review in reviews_to_export:
            review_data = {
                'id': review.id,
                'integration_id': review.integration_id,
                'rating': review.rating.value,
                'title': review.title,
                'content': review.content,
                'status': review.status.value,
                'created_at': review.created_at.isoformat(),
                'helpful_votes': review.helpful_votes,
                'not_helpful_votes': review.not_helpful_votes,
                'verified_purchase': review.verified_purchase,
                'sentiment_score': float(review.sentiment_score) if review.sentiment_score else None,
                'tags': list(review.tags)
            }
            export_data['reviews'].append(review_data)
        
        # Add summary metrics
        if integration_id:
            metrics = self.get_integration_metrics(integration_id)
            export_data['summary_metrics'] = {
                'total_reviews': metrics.total_reviews,
                'average_rating': float(metrics.average_rating),
                'rating_distribution': metrics.rating_distribution,
                'helpfulness_ratio': float(metrics.helpfulness_ratio),
                'response_rate': float(metrics.response_rate),
                'sentiment_score': float(metrics.sentiment_score)
            }
        
        return export_data