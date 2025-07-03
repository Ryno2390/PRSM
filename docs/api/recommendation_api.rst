Recommendation API
==================

The Recommendation API provides sophisticated ML-based recommendations for the PRSM marketplace, delivering personalized content discovery through multi-algorithm fusion, real-time personalization, and comprehensive analytics.

**Key Features:**

* Multi-algorithm recommendation fusion (collaborative, content-based, trending)
* Real-time personalization based on user behavior and preferences
* A/B testing support for algorithm optimization
* Comprehensive recommendation analytics and performance monitoring
* Cold start handling for new users and content
* Diversity optimization to avoid filter bubbles

.. automodule:: prsm.api.recommendation_api
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
-------------

Get Recommendations
~~~~~~~~~~~~~~~~~~~

.. http:get:: /recommendations

   Get personalized marketplace recommendations using intelligent multi-algorithm fusion.

   **Query Parameters:**
   
   * ``resource_type`` (optional) - Filter by resource type (e.g., "model", "dataset", "tutorial")
   * ``current_resource_id`` (optional) - ID of currently viewed resource for context
   * ``search_query`` (optional) - Current search query for contextual recommendations
   * ``limit`` (optional, default: 20) - Maximum number of recommendations (1-100)
   * ``diversity_factor`` (optional, default: 0.3) - Balance between relevance and diversity (0.0-1.0)
   * ``include_reasoning`` (optional, default: true) - Include recommendation reasoning

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "recommendations": [
          {
            "resource_id": "model_gpt4_finance",
            "resource_type": "model",
            "score": 0.92,
            "confidence": 0.85,
            "reasoning": [
              "High relevance to your recent financial modeling queries",
              "Similar users with finance background rated this highly",
              "Trending in the finance category this week"
            ],
            "recommendation_type": "personalized",
            "metadata": {
              "provider": "OpenAI",
              "category": "finance",
              "quality_score": 4.6,
              "download_count": 15420
            }
          }
        ],
        "total_count": 20,
        "execution_time_ms": 156.3,
        "algorithms_used": ["personalized", "content_based", "trending"],
        "personalized": true,
        "metadata": {
          "diversity_factor": 0.3,
          "context_used": true,
          "cold_start": false
        }
      }

   **Algorithm Types:**
   
   * ``personalized`` - Based on user's interaction history and preferences
   * ``content_based`` - Similar to viewed/searched items using content analysis
   * ``collaborative`` - Based on similar users' preferences and behaviors
   * ``trending`` - Currently popular resources with velocity weighting
   * ``business_rules`` - Quality and compliance driven recommendations

Submit Recommendation Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /recommendations/feedback

   Submit feedback on recommendations to improve ML model personalization.

   **Request Body:**

   .. code-block:: json

      {
        "recommendation_id": "rec_abc123",
        "action": "clicked",
        "rating": 4,
        "feedback_text": "Very relevant to my current project"
      }

   **Feedback Actions:**
   
   * ``clicked`` - User clicked on the recommendation
   * ``dismissed`` - User explicitly dismissed the recommendation
   * ``purchased`` - User purchased/downloaded the recommended resource
   * ``rated`` - User rated the recommendation quality (1-5)
   * ``viewed`` - User viewed the recommendation details

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "Feedback received successfully",
        "recommendation_id": "rec_abc123"
      }

Similar Recommendations
~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /recommendations/similar/{resource_id}

   Get recommendations similar to a specific resource using content-based analysis.

   **Path Parameters:**
   
   * ``resource_id`` - ID of the reference resource

   **Query Parameters:**
   
   * ``limit`` (optional, default: 10) - Maximum number of similar resources (1-50)

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "resource_id": "model_bert_nlp",
        "similar_resources": [
          {
            "resource_id": "model_roberta_nlp",
            "resource_type": "model",
            "similarity_score": 0.89,
            "confidence": 0.82,
            "reasoning": [
              "Same NLP domain and transformer architecture",
              "Similar training methodology and datasets",
              "Comparable performance benchmarks"
            ]
          }
        ],
        "count": 10
      }

Trending Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /recommendations/trending

   Get trending recommendations based on recent activity and velocity analysis.

   **Query Parameters:**
   
   * ``resource_type`` (optional) - Filter by resource type
   * ``limit`` (optional, default: 20) - Maximum number of trending resources (1-50)
   * ``time_window`` (optional, default: "7d") - Time window for trending analysis

   **Time Windows:**
   
   * ``1d`` - Last 24 hours
   * ``3d`` - Last 3 days
   * ``7d`` - Last week (default)
   * ``30d`` - Last month

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "trending_resources": [
          {
            "resource_id": "dataset_climate_2025",
            "resource_type": "dataset",
            "trend_score": 0.94,
            "confidence": 0.88,
            "reasoning": [
              "300% increase in downloads this week",
              "High engagement from climate researchers",
              "Featured in recent climate modeling publications"
            ],
            "metadata": {
              "velocity": 2.8,
              "download_acceleration": "300%",
              "category": "climate_science"
            }
          }
        ],
        "count": 20,
        "time_window": "7d",
        "resource_type": null
      }

Analytics & Performance
~~~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /recommendations/analytics

   Get recommendation system analytics and performance metrics. Requires enterprise role or higher.

   **Response:**

   .. code-block:: json

      {
        "total_recommendations_served": 150000,
        "click_through_rate": 0.12,
        "conversion_rate": 0.034,
        "average_rating": 4.2,
        "algorithm_performance": {
          "personalized": {
            "ctr": 0.15,
            "conversion": 0.045,
            "rating": 4.3
          },
          "content_based": {
            "ctr": 0.11,
            "conversion": 0.028,
            "rating": 4.1
          },
          "collaborative": {
            "ctr": 0.13,
            "conversion": 0.038,
            "rating": 4.2
          },
          "trending": {
            "ctr": 0.09,
            "conversion": 0.025,
            "rating": 3.9
          },
          "business_rules": {
            "ctr": 0.08,
            "conversion": 0.022,
            "rating": 4.0
          }
        },
        "user_engagement_metrics": {
          "daily_active_users": 12500,
          "avg_recommendations_per_user": 8.3,
          "user_retention_rate": 0.78,
          "personalization_adoption": 0.85
        }
      }

A/B Testing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /recommendations/ab-test

   Configure A/B testing for recommendation algorithms. Requires admin permissions.

   **Request Body:**

   .. code-block:: json

      {
        "test_name": "Personalization vs Content-Based",
        "algorithm_weights": {
          "control": {"personalized": 0.6, "content_based": 0.4},
          "variant": {"personalized": 0.4, "content_based": 0.6}
        },
        "traffic_split": {"control": 0.5, "variant": 0.5},
        "duration_days": 14,
        "success_metrics": ["ctr", "conversion_rate", "user_satisfaction"],
        "target_audience": {
          "user_segments": ["new_users", "power_users"],
          "min_activity_level": 5
        }
      }

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "A/B test configured successfully",
        "test_id": "test_1672531200",
        "config": {
          "test_name": "Personalization vs Content-Based",
          "start_date": "2025-07-02T10:30:00Z",
          "end_date": "2025-07-16T10:30:00Z",
          "status": "active"
        }
      }

System Health
~~~~~~~~~~~~~

.. http:get:: /recommendations/health

   Health check for recommendation engine with system status and performance metrics.

   **Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "timestamp": "2025-07-02T10:30:00Z",
        "algorithms_available": [
          "personalized",
          "content_based",
          "collaborative",
          "trending",
          "business_rules"
        ],
        "cache_status": {
          "user_profiles": 12500,
          "resource_embeddings": 45000,
          "similarity_matrix": 89000
        },
        "performance_metrics": {
          "avg_response_time_ms": 156.3,
          "cache_hit_rate": 0.87,
          "model_freshness": "2 hours",
          "daily_recommendations": 125000
        },
        "version": "1.0.0"
      }

Authentication & Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most recommendation endpoints support both authenticated and anonymous usage, with enhanced personalization for authenticated users. Analytics and A/B testing endpoints require appropriate permissions.

**Headers for Authenticated Requests:**

.. code-block:: http

   Authorization: Bearer <jwt_token>
   Content-Type: application/json

**Permission Requirements:**

* **Analytics Endpoint** - Enterprise role or higher
* **A/B Testing Configuration** - Admin role required
* **General Recommendations** - No authentication required (anonymous supported)
* **Feedback Submission** - Authentication required

Rate Limiting
~~~~~~~~~~~~~

Recommendation endpoints are subject to rate limiting to ensure fair usage:

* **Anonymous Users** - 100 requests per hour
* **Authenticated Users** - 1000 requests per hour  
* **Enterprise Users** - 10,000 requests per hour
* **A/B Testing Config** - 10 requests per hour

Error Responses
--------------

Recommendation API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "invalid_action",
     "message": "Invalid action. Must be one of: clicked, dismissed, purchased, rated, viewed",
     "code": 400,
     "timestamp": "2025-07-02T10:30:00Z"
   }

**Common Error Codes:**

* ``400`` - Bad Request (invalid parameters, invalid action types)
* ``401`` - Unauthorized (missing token for authenticated endpoints)
* ``403`` - Forbidden (insufficient permissions for analytics/admin endpoints)
* ``404`` - Not Found (resource not found for similar recommendations)
* ``429`` - Rate Limited (too many requests)
* ``500`` - Internal Server Error (recommendation engine temporarily unavailable)

**Fallback Behavior:**

When the recommendation engine is temporarily unavailable, the API returns graceful fallback responses with empty recommendation lists and appropriate error metadata, ensuring system resilience.
