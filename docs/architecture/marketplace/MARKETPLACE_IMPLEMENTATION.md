# PRSM Marketplace Implementation Guide
**Comprehensive Resource Marketplace Architecture and API Documentation**

## Overview

The PRSM marketplace provides a federated platform for trading computational resources, AI services, and data processing capabilities using the FTNS (Fungible Tokens for Node Support). This document consolidates the complete marketplace implementation including API design, database models, and integration patterns.

## Architecture Overview

### Core Components

1. **Real Marketplace Service** - Production marketplace implementation
2. **Expanded Marketplace Service** - Extended functionality for enterprise features  
3. **Database Models** - Comprehensive data layer with resource listings and transactions
4. **API Layer** - RESTful endpoints with real-time capabilities
5. **Integration Layer** - FTNS token system and blockchain integration

### API Consolidation Summary

The marketplace API has been consolidated into a unified structure:

**Core Endpoints:**
- `/marketplace/resources` - Resource management (CRUD operations)
- `/marketplace/transactions` - Transaction processing and history
- `/marketplace/search` - Advanced resource discovery
- `/marketplace/recommendations` - Personalized recommendations
- `/marketplace/analytics` - Usage analytics and insights

**Real-Time Features:**
- WebSocket connections for live updates
- Real-time transaction notifications
- Dynamic pricing updates
- Resource availability changes

## Implementation Status

### ✅ Completed Features

**Resource Management:**
- Resource listing and discovery
- Comprehensive search and filtering
- Category-based organization
- Quality scoring and ratings

**Transaction Processing:**
- FTNS token integration
- Secure payment processing
- Transaction history and receipts
- Automated settlement

**User Experience:**
- Personalized recommendations
- Usage analytics and insights
- Real-time notifications
- Mobile-responsive interface

**Enterprise Features:**
- Bulk resource provisioning
- Enterprise billing and reporting
- API rate limiting and quotas
- Advanced security controls

### 🟡 Integration Points

**Blockchain Integration:**
- FTNS token smart contracts
- On-chain transaction verification
- Decentralized identity management
- Cross-chain compatibility

**External Services:**
- Cloud provider APIs (AWS, GCP, Azure)
- Payment gateway integration
- Identity verification services
- Monitoring and observability

## Database Schema

### Core Tables

```sql
-- Resource listings in marketplace
CREATE TABLE marketplace_resources (
    id UUID PRIMARY KEY,
    provider_id UUID NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    price_per_hour DECIMAL(18,6) NOT NULL,
    currency VARCHAR(10) DEFAULT 'FTNS',
    availability_start TIMESTAMP,
    availability_end TIMESTAMP,
    location VARCHAR(255),
    specifications JSONB,
    quality_score DECIMAL(3,2) DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    total_earnings DECIMAL(18,6) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Transaction records
CREATE TABLE marketplace_transactions (
    id UUID PRIMARY KEY,
    provider_id UUID NOT NULL,
    consumer_id UUID NOT NULL,
    resource_id UUID NOT NULL,
    amount DECIMAL(18,6) NOT NULL,
    currency VARCHAR(10) DEFAULT 'FTNS',
    duration_hours DECIMAL(8,2),
    status VARCHAR(50) NOT NULL,
    transaction_hash VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

### Extended Features

**Recommendation Engine:**
- User preference tracking
- Collaborative filtering algorithms
- Content-based recommendations
- Real-time personalization

**Analytics Dashboard:**
- Usage metrics and trends
- Revenue analytics
- Performance monitoring
- Market insights

## API Reference

### Resource Management

```python
# List resources with filtering
GET /marketplace/resources
Parameters:
  - category: Filter by resource category
  - location: Filter by geographic location
  - price_min/max: Price range filtering
  - availability: Availability date range
  - sort: Sorting criteria (price, rating, availability)

# Create new resource listing
POST /marketplace/resources
Body: {
  "title": "High-Performance GPU Cluster",
  "description": "NVIDIA A100 GPU cluster for AI training",
  "category": "compute",
  "resource_type": "gpu",
  "price_per_hour": 2.50,
  "specifications": {
    "gpu_count": 8,
    "memory_gb": 320,
    "interconnect": "NVLink"
  }
}

# Update resource listing
PUT /marketplace/resources/{resource_id}

# Delete resource listing
DELETE /marketplace/resources/{resource_id}
```

### Transaction Processing

```python
# Create transaction (resource rental)
POST /marketplace/transactions
Body: {
  "resource_id": "uuid",
  "duration_hours": 24.0,
  "payment_method": "ftns_wallet"
}

# Get transaction history
GET /marketplace/transactions
Parameters:
  - user_id: Filter by user
  - status: Filter by transaction status
  - date_range: Time period filtering

# Get transaction details
GET /marketplace/transactions/{transaction_id}
```

### Search and Discovery

```python
# Advanced resource search
POST /marketplace/search
Body: {
  "query": "machine learning GPU",
  "filters": {
    "category": "compute",
    "min_memory": 16,
    "location": "us-west"
  },
  "sort": "relevance",
  "limit": 20
}

# Get personalized recommendations
GET /marketplace/recommendations
Parameters:
  - user_id: Target user for recommendations
  - count: Number of recommendations
  - categories: Preferred categories
```

### Analytics and Insights

```python
# Get marketplace analytics
GET /marketplace/analytics
Parameters:
  - metric: Type of analytics (usage, revenue, trends)
  - time_range: Analysis period
  - aggregation: Aggregation level (hourly, daily, monthly)

# Get provider performance metrics
GET /marketplace/analytics/provider/{provider_id}

# Get resource utilization metrics
GET /marketplace/analytics/resource/{resource_id}
```

## Integration Examples

### FTNS Token Integration

```python
from prsm.tokenomics.ftns_service import FTNSService
from prsm.marketplace.real_marketplace_service import RealMarketplaceService

# Initialize services
ftns = FTNSService()
marketplace = RealMarketplaceService()

# Process marketplace transaction
async def process_resource_rental(user_id, resource_id, duration_hours):
    # Calculate total cost
    resource = await marketplace.get_resource(resource_id)
    total_cost = resource.price_per_hour * duration_hours
    
    # Check user balance
    balance = await ftns.get_balance(user_id)
    if balance < total_cost:
        raise InsufficientFundsError()
    
    # Create transaction
    transaction = await marketplace.create_transaction(
        provider_id=resource.provider_id,
        consumer_id=user_id,
        resource_id=resource_id,
        amount=total_cost,
        duration_hours=duration_hours
    )
    
    # Process payment
    await ftns.transfer(
        from_user=user_id,
        to_user=resource.provider_id,
        amount=total_cost,
        transaction_id=transaction.id
    )
    
    return transaction
```

### Real-Time Updates

```python
# WebSocket integration for live updates
from prsm.api.websocket_auth import WebSocketManager

websocket_manager = WebSocketManager()

# Notify clients of resource availability changes
async def notify_resource_update(resource_id, update_data):
    await websocket_manager.broadcast_to_subscribers(
        channel=f"resource:{resource_id}",
        message={
            "type": "resource_update",
            "resource_id": resource_id,
            "data": update_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Notify transaction status updates
async def notify_transaction_update(transaction_id, status):
    transaction = await marketplace.get_transaction(transaction_id)
    await websocket_manager.send_to_user(
        user_id=transaction.consumer_id,
        message={
            "type": "transaction_update",
            "transaction_id": transaction_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## Performance Optimizations

### Database Optimizations

```sql
-- Indexes for efficient querying
CREATE INDEX idx_marketplace_resources_category ON marketplace_resources(category);
CREATE INDEX idx_marketplace_resources_location ON marketplace_resources(location);
CREATE INDEX idx_marketplace_resources_price ON marketplace_resources(price_per_hour);
CREATE INDEX idx_marketplace_resources_availability ON marketplace_resources(availability_start, availability_end);

-- Transaction history optimization
CREATE INDEX idx_marketplace_transactions_user ON marketplace_transactions(consumer_id, created_at DESC);
CREATE INDEX idx_marketplace_transactions_status ON marketplace_transactions(status);
```

### Caching Strategy

```python
# Redis caching for frequently accessed data
from prsm.core.production_cache import ProductionCacheManager

cache = ProductionCacheManager()

# Cache popular resources
async def get_popular_resources(category=None):
    cache_key = f"popular_resources:{category or 'all'}"
    cached_result = await cache.get(cache_key)
    
    if cached_result:
        return cached_result
    
    # Fetch from database
    resources = await marketplace.get_popular_resources(category)
    
    # Cache for 5 minutes
    await cache.set(cache_key, resources, ttl=300)
    return resources
```

## Security Considerations

### Access Control

```python
# Role-based access control for marketplace operations
from prsm.auth.middleware import require_permission

@require_permission("marketplace:create_resource")
async def create_resource(user_id, resource_data):
    # Validate user can create resources
    if not await marketplace.can_create_resource(user_id):
        raise PermissionDeniedError()
    
    # Validate resource data
    validated_data = await marketplace.validate_resource_data(resource_data)
    
    # Create resource listing
    return await marketplace.create_resource(user_id, validated_data)
```

### Input Validation

```python
# Comprehensive input sanitization
from prsm.security.production_input_sanitization import ProductionInputSanitizer

sanitizer = ProductionInputSanitizer()

async def sanitize_marketplace_input(data):
    """Sanitize all marketplace input data"""
    sanitized = {}
    
    # Sanitize text fields
    if 'title' in data:
        sanitized['title'] = await sanitizer.sanitize_text(data['title'])
    
    if 'description' in data:
        sanitized['description'] = await sanitizer.sanitize_html(data['description'])
    
    # Validate numeric fields
    if 'price_per_hour' in data:
        sanitized['price_per_hour'] = await sanitizer.validate_decimal(
            data['price_per_hour'], 
            min_value=0.0001, 
            max_value=10000.0
        )
    
    return sanitized
```

## Testing and Validation

### Unit Tests

```python
import pytest
from prsm.marketplace.real_marketplace_service import RealMarketplaceService

@pytest.mark.asyncio
async def test_resource_creation():
    marketplace = RealMarketplaceService()
    
    resource_data = {
        "title": "Test GPU",
        "category": "compute",
        "price_per_hour": 1.50
    }
    
    resource = await marketplace.create_resource("user123", resource_data)
    assert resource.title == "Test GPU"
    assert resource.price_per_hour == 1.50

@pytest.mark.asyncio
async def test_transaction_processing():
    marketplace = RealMarketplaceService()
    
    transaction = await marketplace.create_transaction(
        provider_id="provider123",
        consumer_id="consumer456",
        resource_id="resource789",
        amount=25.0,
        duration_hours=10.0
    )
    
    assert transaction.amount == 25.0
    assert transaction.status == "pending"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_end_to_end_marketplace_flow():
    """Test complete marketplace workflow"""
    
    # Create resource
    resource = await create_test_resource()
    
    # Search for resource
    search_results = await marketplace.search_resources(
        query="test",
        filters={"category": "compute"}
    )
    assert resource.id in [r.id for r in search_results]
    
    # Create transaction
    transaction = await marketplace.create_transaction(
        provider_id=resource.provider_id,
        consumer_id="test_consumer",
        resource_id=resource.id,
        amount=resource.price_per_hour * 5,
        duration_hours=5.0
    )
    
    # Verify transaction
    assert transaction.status == "pending"
    retrieved = await marketplace.get_transaction(transaction.id)
    assert retrieved.id == transaction.id
```

## Deployment and Operations

### Production Configuration

```python
# Production marketplace configuration
MARKETPLACE_CONFIG = {
    "max_resources_per_user": 100,
    "transaction_timeout_minutes": 30,
    "search_results_limit": 100,
    "enable_recommendations": True,
    "cache_ttl_seconds": 300,
    "rate_limit_requests_per_minute": 1000
}
```

### Monitoring and Observability

```python
# Marketplace metrics and monitoring
from prsm.monitoring.metrics import MetricsCollector

metrics = MetricsCollector()

# Track marketplace usage
await metrics.increment("marketplace.resource.created")
await metrics.increment("marketplace.transaction.completed")
await metrics.histogram("marketplace.search.response_time", response_time_ms)
await metrics.gauge("marketplace.active_resources", active_count)
```

## Future Enhancements

### Planned Features

1. **Advanced Pricing Models**
   - Dynamic pricing based on demand
   - Auction-based resource allocation
   - Bulk pricing discounts

2. **Enhanced Discovery**
   - AI-powered resource recommendations
   - Semantic search capabilities
   - Resource compatibility matching

3. **Enterprise Integration**
   - Enterprise resource planning integration
   - Advanced billing and cost allocation
   - Compliance and audit reporting

4. **Global Expansion**
   - Multi-currency support
   - Localized payment methods
   - Regional compliance features

---

## Related Documentation

- [FTNS Token System](../../tokenomics/FTNS_OVERVIEW.md)
- [API Reference](../../API_REFERENCE.md)
- [Security Architecture](../../SECURITY_ARCHITECTURE.md)
- [Performance Baselines](../../performance/PERFORMANCE_BASELINES.md)

---

**Document Version:** 1.0.0  
**Last Updated:** July 1, 2025  
**Next Review:** October 1, 2025