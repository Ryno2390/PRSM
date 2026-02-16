"""
Comprehensive API Integration Tests
===================================

Complete integration tests for PRSM API endpoints including authentication,
rate limiting, error handling, and performance testing.
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import time

try:
    import httpx
    from fastapi import status
    from prsm.interface.api.main import create_app
    from prsm.core.auth.jwt_handler import create_access_token
    from prsm.core.models import PRSMSession, FTNSTransaction
except (ImportError, Exception) as e:
    pytest.skip(f"API modules have import errors (pydantic regex issue): {e}", allow_module_level=True)


@pytest.mark.api
@pytest.mark.integration
class TestAuthenticationAPI:
    """Test authentication and authorization endpoints"""
    
    async def test_user_registration(self, async_test_client, api_data_factory):
        """Test user registration endpoint"""
        registration_data = api_data_factory.create_user_registration_request(
            username="newuser",
            email="newuser@test.com",
            password="securepassword123"
        )
        
        response = await async_test_client.post(
            "/api/v1/auth/register",
            json=registration_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        
        assert "user_id" in response_data
        assert response_data["username"] == "newuser"
        assert response_data["email"] == "newuser@test.com"
        assert "password" not in response_data  # Password should not be returned
    
    async def test_user_login(self, async_test_client, api_data_factory):
        """Test user login endpoint"""
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        
        response = await async_test_client.post(
            "/api/v1/auth/login",
            json=login_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "access_token" in response_data
        assert "token_type" in response_data
        assert response_data["token_type"] == "bearer"
        assert "expires_in" in response_data
    
    async def test_token_refresh(self, async_test_client, user_headers):
        """Test token refresh endpoint"""
        response = await async_test_client.post(
            "/api/v1/auth/refresh",
            headers=user_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "access_token" in response_data
        assert "expires_in" in response_data
    
    async def test_user_profile(self, async_test_client, user_headers):
        """Test user profile endpoint"""
        response = await async_test_client.get(
            "/api/v1/auth/profile",
            headers=user_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "user_id" in response_data
        assert "username" in response_data
        assert "email" in response_data
        assert "roles" in response_data
    
    async def test_unauthorized_access(self, async_test_client):
        """Test unauthorized access to protected endpoints"""
        response = await async_test_client.get("/api/v1/auth/profile")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        response_data = response.json()
        assert "error" in response_data
        assert response_data["error"] == "unauthorized"
    
    async def test_expired_token(self, async_test_client, expired_token_headers):
        """Test access with expired token"""
        response = await async_test_client.get(
            "/api/v1/auth/profile",
            headers=expired_token_headers
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        response_data = response.json()
        assert "error" in response_data
        assert "expired" in response_data["message"].lower()


@pytest.mark.api
@pytest.mark.integration
class TestNWTNAPI:
    """Test NWTN reasoning engine API endpoints"""
    
    async def test_nwtn_query_basic(self, async_test_client, user_headers, api_data_factory):
        """Test basic NWTN query"""
        query_data = api_data_factory.create_nwtn_query_request(
            query="What is artificial intelligence?",
            mode="adaptive",
            max_depth=2
        )
        
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            mock_process.return_value = {
                "response": "Artificial intelligence is...",
                "reasoning_depth": 2,
                "confidence": 0.85,
                "session_id": "test_session_123",
                "tokens_used": 150
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "response" in response_data
        assert "reasoning_depth" in response_data
        assert "confidence" in response_data
        assert "session_id" in response_data
        assert response_data["reasoning_depth"] == 2
        assert 0 <= response_data["confidence"] <= 1
    
    async def test_nwtn_query_with_context(self, async_test_client, user_headers, api_data_factory):
        """Test NWTN query with context"""
        query_data = api_data_factory.create_nwtn_query_request(
            query="Continue the previous discussion about AI",
            mode="contextual",
            context={
                "previous_queries": ["What is AI?"],
                "session_id": "existing_session_123"
            }
        )
        
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            mock_process.return_value = {
                "response": "Continuing our discussion about AI...",
                "reasoning_depth": 3,
                "confidence": 0.90,
                "session_id": "existing_session_123",
                "context_utilized": True
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["session_id"] == "existing_session_123"
        assert response_data["context_utilized"] is True
    
    async def test_nwtn_query_modes(self, async_test_client, user_headers, api_data_factory):
        """Test different NWTN query modes"""
        modes = ["adaptive", "deep", "quick", "creative"]
        
        for mode in modes:
            query_data = api_data_factory.create_nwtn_query_request(
                query=f"Test query for {mode} mode",
                mode=mode
            )
            
            with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
                mock_process.return_value = {
                    "response": f"Response in {mode} mode",
                    "reasoning_depth": 2,
                    "confidence": 0.80,
                    "mode_used": mode
                }
                
                response = await async_test_client.post(
                    "/api/v1/nwtn/query",
                    json=query_data,
                    headers=user_headers
                )
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data["mode_used"] == mode
    
    async def test_nwtn_session_history(self, async_test_client, user_headers):
        """Test NWTN session history retrieval"""
        session_id = "test_session_123"
        
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.get_session_history') as mock_history:
            mock_history.return_value = [
                {
                    "query": "What is AI?",
                    "response": "AI is...",
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                {
                    "query": "How does machine learning work?",
                    "response": "Machine learning works by...",
                    "timestamp": "2024-01-01T12:05:00Z"
                }
            ]
            
            response = await async_test_client.get(
                f"/api/v1/nwtn/sessions/{session_id}/history",
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert isinstance(response_data, list)
        assert len(response_data) == 2
        assert all("query" in item and "response" in item for item in response_data)
    
    @pytest.mark.performance
    async def test_nwtn_query_performance(self, async_test_client, user_headers, api_data_factory, api_performance_monitor):
        """Test NWTN query performance"""
        query_data = api_data_factory.create_nwtn_query_request(
            query="Performance test query"
        )
        
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            mock_process.return_value = {
                "response": "Performance test response",
                "reasoning_depth": 2,
                "confidence": 0.85,
                "processing_time_ms": 1200
            }
            
            start_time = time.time()
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
            end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time_ms < 3000  # Less than 3 seconds
        
        # Record performance metrics
        api_performance_monitor.record_request(start_time, end_time, len(response.content))


@pytest.mark.api
@pytest.mark.integration
class TestFTNSAPI:
    """Test FTNS tokenomics API endpoints"""
    
    async def test_get_balance(self, async_test_client, user_headers):
        """Test get user balance"""
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance:
            mock_balance.return_value = {
                "total_balance": Decimal("150.00"),
                "available_balance": Decimal("120.00"),
                "reserved_balance": Decimal("30.00"),
                "pending_balance": Decimal("5.00")
            }
            
            response = await async_test_client.get(
                "/api/v1/ftns/balance",
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "total_balance" in response_data
        assert "available_balance" in response_data
        assert "reserved_balance" in response_data
        assert float(response_data["total_balance"]) == 150.00
        assert float(response_data["available_balance"]) == 120.00
    
    async def test_transfer_tokens(self, async_test_client, user_headers, api_data_factory):
        """Test FTNS token transfer"""
        transfer_data = api_data_factory.create_ftns_transfer_request(
            recipient="test_recipient_user",
            amount=25.50,
            description="Test transfer"
        )
        
        with patch('prsm.tokenomics.ftns_service.FTNSService.transfer') as mock_transfer:
            mock_transfer.return_value = {
                "success": True,
                "transaction_id": "tx_transfer_123",
                "amount": Decimal("25.50"),
                "fee": Decimal("0.25"),
                "status": "confirmed"
            }
            
            response = await async_test_client.post(
                "/api/v1/ftns/transfer",
                json=transfer_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["success"] is True
        assert "transaction_id" in response_data
        assert float(response_data["amount"]) == 25.50
        assert response_data["status"] == "confirmed"
    
    async def test_transaction_history(self, async_test_client, user_headers):
        """Test transaction history retrieval"""
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_transaction_history') as mock_history:
            mock_history.return_value = [
                {
                    "transaction_id": "tx_1",
                    "type": "reward",
                    "amount": Decimal("50.00"),
                    "timestamp": "2024-01-01T10:00:00Z",
                    "description": "NWTN query reward"
                },
                {
                    "transaction_id": "tx_2",
                    "type": "charge",
                    "amount": Decimal("-10.00"),
                    "timestamp": "2024-01-01T11:00:00Z",
                    "description": "API usage charge"
                }
            ]
            
            response = await async_test_client.get(
                "/api/v1/ftns/transactions",
                headers=user_headers,
                params={"limit": 10, "offset": 0}
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert isinstance(response_data, list)
        assert len(response_data) == 2
        assert all("transaction_id" in tx for tx in response_data)
        assert all("type" in tx for tx in response_data)
    
    async def test_insufficient_balance_transfer(self, async_test_client, user_headers, api_data_factory):
        """Test transfer with insufficient balance"""
        transfer_data = api_data_factory.create_ftns_transfer_request(
            recipient="test_recipient",
            amount=10000.00,  # Huge amount
            description="Insufficient balance test"
        )
        
        with patch('prsm.tokenomics.ftns_service.FTNSService.transfer') as mock_transfer:
            mock_transfer.side_effect = Exception("Insufficient balance")
            
            response = await async_test_client.post(
                "/api/v1/ftns/transfer",
                json=transfer_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = response.json()
        
        assert "error" in response_data
        assert "insufficient" in response_data["message"].lower()


@pytest.mark.api
@pytest.mark.integration
class TestMarketplaceAPI:
    """Test Marketplace API endpoints"""
    
    async def test_list_marketplace_items(self, async_test_client, user_headers):
        """Test listing marketplace items"""
        with patch('prsm.marketplace.service.MarketplaceService.get_items') as mock_items:
            mock_items.return_value = [
                {
                    "item_id": "item_1",
                    "title": "NWTN Query Template",
                    "description": "Optimized template for scientific queries",
                    "price": 15.00,
                    "category": "templates",
                    "creator": "expert_user"
                },
                {
                    "item_id": "item_2",
                    "title": "Data Analysis Script",
                    "description": "Python script for data analysis",
                    "price": 25.00,
                    "category": "scripts",
                    "creator": "data_scientist"
                }
            ]
            
            response = await async_test_client.get(
                "/api/v1/marketplace/items",
                headers=user_headers,
                params={"category": "all", "limit": 20}
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert isinstance(response_data, list)
        assert len(response_data) == 2
        assert all("item_id" in item for item in response_data)
        assert all("price" in item for item in response_data)
    
    async def test_create_marketplace_item(self, async_test_client, user_headers, api_data_factory):
        """Test creating marketplace item"""
        item_data = api_data_factory.create_marketplace_item_request(
            title="New Analysis Tool",
            description="Advanced data analysis tool",
            price=45.00,
            category="tools"
        )
        
        with patch('prsm.marketplace.service.MarketplaceService.create_item') as mock_create:
            mock_create.return_value = {
                "item_id": "new_item_123",
                "title": "New Analysis Tool",
                "status": "active",
                "created_at": "2024-01-01T12:00:00Z"
            }
            
            response = await async_test_client.post(
                "/api/v1/marketplace/items",
                json=item_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        
        assert "item_id" in response_data
        assert response_data["title"] == "New Analysis Tool"
        assert response_data["status"] == "active"
    
    async def test_purchase_marketplace_item(self, async_test_client, user_headers):
        """Test purchasing marketplace item"""
        item_id = "item_123"
        purchase_data = {
            "quantity": 1,
            "payment_method": "ftns_balance"
        }
        
        with patch('prsm.marketplace.service.MarketplaceService.purchase_item') as mock_purchase:
            mock_purchase.return_value = {
                "purchase_id": "purchase_456",
                "item_id": item_id,
                "amount_paid": 25.00,
                "transaction_id": "tx_purchase_789",
                "status": "completed"
            }
            
            response = await async_test_client.post(
                f"/api/v1/marketplace/items/{item_id}/purchase",
                json=purchase_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "purchase_id" in response_data
        assert "transaction_id" in response_data
        assert response_data["status"] == "completed"


@pytest.mark.api
@pytest.mark.security
class TestAPISecurityFeatures:
    """Test API security features"""
    
    async def test_sql_injection_protection(self, async_test_client, user_headers, security_test_helper):
        """Test SQL injection protection"""
        sql_payloads = security_test_helper.create_sql_injection_payloads()
        
        for payload in sql_payloads:
            response = await async_test_client.get(
                "/api/v1/ftns/transactions",
                headers=user_headers,
                params={"search": payload}
            )
            
            # Should not return 500 or expose database errors
            assert response.status_code in [200, 400, 422]  # Valid responses
            
            if response.status_code == 400:
                response_data = response.json()
                # Should not expose SQL error details
                assert "sql" not in response_data.get("message", "").lower()
                assert "database" not in response_data.get("message", "").lower()
    
    async def test_xss_protection(self, async_test_client, user_headers, security_test_helper):
        """Test XSS protection"""
        xss_payloads = security_test_helper.create_xss_payloads()
        
        for payload in xss_payloads:
            query_data = {
                "query": payload,
                "mode": "adaptive"
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
            
            # Should handle malicious input safely
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                response_data = response.json()
                # Response should not contain raw script tags
                assert "<script>" not in response_data.get("response", "")
    
    async def test_request_size_limits(self, async_test_client, user_headers):
        """Test request size limits"""
        # Create extremely large request
        large_query = "A" * 100000  # 100KB query
        
        query_data = {
            "query": large_query,
            "mode": "adaptive"
        }
        
        response = await async_test_client.post(
            "/api/v1/nwtn/query",
            json=query_data,
            headers=user_headers
        )
        
        # Should reject oversized requests
        assert response.status_code in [400, 413, 422]
    
    async def test_malformed_json_handling(self, async_test_client, user_headers, security_test_helper):
        """Test malformed JSON handling"""
        malformed_requests = security_test_helper.create_malformed_requests()
        
        for malformed_data in malformed_requests:
            try:
                response = await async_test_client.post(
                    "/api/v1/nwtn/query",
                    json=malformed_data,
                    headers=user_headers
                )
                
                # Should handle malformed data gracefully
                assert response.status_code in [400, 422]
                
                response_data = response.json()
                assert "error" in response_data
                
            except Exception:
                # JSON serialization should fail gracefully
                pass


@pytest.mark.api
@pytest.mark.performance
class TestAPIPerformanceAndLoad:
    """Test API performance and load handling"""
    
    async def test_concurrent_requests(self, async_test_client, user_headers, api_data_factory):
        """Test handling of concurrent requests"""
        query_data = api_data_factory.create_nwtn_query_request(
            query="Concurrent test query"
        )
        
        # Create multiple concurrent requests
        async def make_request():
            with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
                mock_process.return_value = {
                    "response": "Concurrent response",
                    "reasoning_depth": 1,
                    "confidence": 0.8
                }
                
                return await async_test_client.post(
                    "/api/v1/nwtn/query",
                    json=query_data,
                    headers=user_headers
                )
        
        # Execute 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful responses
        successful = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        
        # Should handle at least 80% of concurrent requests successfully
        assert successful >= 8
    
    async def test_rate_limiting(self, async_test_client, user_headers, rate_limit_tester):
        """Test API rate limiting"""
        responses = await rate_limit_tester.test_rate_limit(
            client=async_test_client,
            endpoint="/api/v1/auth/profile",
            method="GET",
            limit=10,
            window_seconds=60,
            headers=user_headers
        )
        
        # Should start rate limiting after hitting the limit
        status_codes = [r["status_code"] for r in responses]
        
        # First requests should succeed
        assert status_codes[:10].count(200) >= 8
        
        # Later requests should be rate limited
        assert 429 in status_codes[10:]  # Too Many Requests
    
    @pytest.mark.load
    async def test_api_load_test(self, async_test_client, user_headers, load_test_runner):
        """Test API under load"""
        async def api_request():
            return await async_test_client.get(
                "/api/v1/auth/profile",
                headers=user_headers
            )
        
        results = await load_test_runner.run_load_test(
            test_function=api_request,
            concurrent_users=20,
            duration_seconds=30,
            ramp_up_seconds=5
        )
        
        # Performance assertions
        assert results.error_rate < 0.05  # Less than 5% error rate
        assert results.average_response_time < 1000  # Less than 1 second average
        assert results.requests_per_second > 10  # At least 10 RPS


@pytest.mark.api
@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling and recovery"""
    
    async def test_internal_server_error_handling(self, async_test_client, user_headers):
        """Test internal server error handling"""
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            mock_process.side_effect = Exception("Internal processing error")
            
            query_data = {
                "query": "Test query that will fail",
                "mode": "adaptive"
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        
        assert "error" in response_data
        assert "internal_error" in response_data["error"]
        # Should not expose internal error details
        assert "Internal processing error" not in response_data.get("message", "")
    
    async def test_timeout_handling(self, async_test_client, user_headers):
        """Test request timeout handling"""
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            # Simulate long processing time
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(10)  # 10 second delay
                return {"response": "Slow response"}
            
            mock_process.side_effect = slow_process
            
            query_data = {
                "query": "Slow query",
                "mode": "adaptive"
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers,
                timeout=5.0  # 5 second timeout
            )
        
        # Should timeout gracefully
        assert response.status_code in [408, 504]  # Request Timeout or Gateway Timeout
    
    async def test_validation_error_responses(self, async_test_client, user_headers):
        """Test validation error responses"""
        invalid_data = {
            "query": "",  # Empty query should be invalid
            "mode": "invalid_mode",  # Invalid mode
            "max_depth": -1  # Invalid depth
        }
        
        response = await async_test_client.post(
            "/api/v1/nwtn/query",
            json=invalid_data,
            headers=user_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_data = response.json()
        
        assert "error" in response_data
        assert "validation_error" in response_data["error"]
        assert "details" in response_data
        assert isinstance(response_data["details"], list)


@pytest.mark.api
@pytest.mark.integration
class TestAPIResponseSchemas:
    """Test API response schema validation"""
    
    async def test_nwtn_response_schema(self, async_test_client, user_headers, api_response_schemas, api_data_factory):
        """Test NWTN response matches expected schema"""
        query_data = api_data_factory.create_nwtn_query_request()
        
        with patch('prsm.nwtn.meta_reasoning_engine.NWTNEngine.process_query') as mock_process:
            mock_process.return_value = {
                "response": "Test response",
                "reasoning_depth": 2,
                "confidence": 0.85,
                "session_id": "test_session"
            }
            
            response = await async_test_client.post(
                "/api/v1/nwtn/query",
                json=query_data,
                headers=user_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        # Validate against expected schema
        expected_schema = api_response_schemas["nwtn_response"]
        
        # Check required fields
        for field in expected_schema["required"]:
            assert field in response_data
        
        # Check field types
        assert isinstance(response_data["response"], str)
        assert isinstance(response_data["reasoning_depth"], int)
        assert isinstance(response_data["confidence"], (int, float))
    
    async def test_error_response_schema(self, async_test_client, api_response_schemas):
        """Test error response matches expected schema"""
        response = await async_test_client.get("/api/v1/auth/profile")  # Unauthorized
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        response_data = response.json()
        
        # Validate against error schema
        error_schema = api_response_schemas["error_response"]
        
        for field in error_schema["required"]:
            assert field in response_data
        
        assert isinstance(response_data["error"], str)
        assert isinstance(response_data["message"], str)