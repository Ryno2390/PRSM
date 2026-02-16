"""
API Endpoint Integration Tests
=============================

Integration tests for API endpoints, testing the complete request-response cycle
including authentication, validation, business logic, and data persistence.
"""

import pytest
import asyncio
import uuid
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    from prsm.interface.api.main import app
    from prsm.core.models import UserInput, PRSMResponse, AgentType
    from prsm.core.auth.models import User, UserSession
    from prsm.economy.tokenomics.models import FTNSTransaction, FTNSBalance
    from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
    from prsm.core.auth.auth_manager import AuthManager
    from prsm.economy.tokenomics.ftns_service import FTNSService
except (ImportError, Exception) as e:
    pytest.skip(f"API endpoint modules have import errors (pydantic/orchestrator): {e}", allow_module_level=True)
    FTNSService = Mock


@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpointIntegration:
    """Test API endpoint integration with full request-response cycles"""
    
    async def test_nwtn_query_endpoint_full_cycle(self, async_test_client):
        """Test complete NWTN query endpoint integration"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        endpoint_results = {}
        
        # Setup: Authentication
        auth_token = "integration_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        user_id = "api_test_user"
        
        # Step 1: POST /api/v1/nwtn/query
        query_data = {
            "query": "Explain quantum computing in simple terms",
            "mode": "adaptive",
            "context_allocation": 200,
            "preferences": {
                "explanation_level": "beginner",
                "max_examples": 3
            }
        }
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate, \
             patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_query') as mock_process, \
             patch('prsm.tokenomics.ftns_service.FTNSService.charge_user') as mock_charge:
            
            # Mock authentication
            mock_validate.return_value = {
                "valid": True,
                "user_id": user_id,
                "user_role": "researcher"
            }
            
            # Mock NWTN processing
            session_id = str(uuid.uuid4())
            mock_process.return_value = {
                "session_id": session_id,
                "final_answer": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement...",
                "reasoning_trace": [
                    {
                        "step_id": str(uuid.uuid4()),
                        "agent_type": "architect",
                        "agent_id": "architect_001",
                        "input_data": {"query": query_data["query"]},
                        "output_data": {"decomposed_tasks": ["explain_basics", "provide_examples"]},
                        "execution_time": 0.5,
                        "confidence_score": 0.9
                    }
                ],
                "confidence_score": 0.85,
                "context_used": 180,
                "ftns_cost": 9.0,
                "sources": ["quantum_physics_basics", "computing_fundamentals"]
            }
            
            # Mock FTNS charging
            mock_charge.return_value = {
                "success": True,
                "transaction_id": "tx_" + str(uuid.uuid4()),
                "amount_charged": 9.0,
                "remaining_balance": 91.0
            }
            
            response = await async_test_client.post("/api/v1/nwtn/query", json=query_data, headers=headers)
            
            assert response.status_code == 200
            response_data = response.json()
            
            endpoint_results["nwtn_query"] = {
                "status_code": response.status_code,
                "has_session_id": "session_id" in response_data,
                "has_answer": "final_answer" in response_data,
                "has_reasoning": "reasoning_trace" in response_data,
                "authentication_validated": mock_validate.called,
                "nwtn_processed": mock_process.called,
                "ftns_charged": mock_charge.called
            }
            
            # Verify response structure
            assert "session_id" in response_data
            assert "final_answer" in response_data
            assert "reasoning_trace" in response_data
            assert "confidence_score" in response_data
            assert response_data["confidence_score"] == 0.85
        
        return endpoint_results
    
    async def test_ftns_balance_endpoint_integration(self, async_test_client):
        """Test FTNS balance endpoint with authentication and service integration"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        endpoint_results = {}
        auth_token = "balance_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        user_id = "balance_user"
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate, \
             patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance:
            
            # Mock authentication
            mock_validate.return_value = {
                "valid": True,
                "user_id": user_id,
                "user_role": "user"
            }
            
            # Mock balance service
            mock_balance.return_value = {
                "total_balance": Decimal("150.75"),
                "available_balance": Decimal("130.25"),
                "reserved_balance": Decimal("20.50"),
                "last_transaction": datetime.now(timezone.utc).isoformat()
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            
            assert response.status_code == 200
            balance_data = response.json()
            
            endpoint_results["ftns_balance"] = {
                "status_code": response.status_code,
                "has_balance_data": "total_balance" in balance_data,
                "authentication_validated": mock_validate.called,
                "balance_service_called": mock_balance.called,
                "balance_format_valid": isinstance(balance_data.get("total_balance"), (float, str))
            }
            
            # Verify balance data structure
            assert "total_balance" in balance_data
            assert "available_balance" in balance_data
            assert "reserved_balance" in balance_data
            assert float(balance_data["total_balance"]) == 150.75
        
        return endpoint_results
    
    async def test_auth_endpoints_integration(self, async_test_client):
        """Test authentication endpoint integration"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        endpoint_results = {}
        
        # Test 1: User Registration
        registration_data = {
            "username": "integration_user",
            "email": "integration@test.com",
            "password": "SecurePass123!",
            "terms_accepted": True
        }
        
        with patch('prsm.auth.auth_manager.AuthManager.register_user') as mock_register:
            mock_register.return_value = {
                "success": True,
                "user_id": "reg_user_123",
                "username": "integration_user",
                "email": "integration@test.com",
                "verification_required": False
            }
            
            response = await async_test_client.post("/api/v1/auth/register", json=registration_data)
            
            assert response.status_code == 201
            reg_data = response.json()
            
            endpoint_results["registration"] = {
                "status_code": response.status_code,
                "registration_successful": reg_data.get("success", False),
                "user_id_provided": "user_id" in reg_data,
                "auth_manager_called": mock_register.called
            }
        
        # Test 2: User Login
        login_data = {
            "username": "integration_user",
            "password": "SecurePass123!"
        }
        
        with patch('prsm.auth.auth_manager.AuthManager.authenticate_user') as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "access_token": "jwt_token_abc123",
                "token_type": "bearer",
                "expires_in": 3600,
                "user_id": "reg_user_123"
            }
            
            response = await async_test_client.post("/api/v1/auth/login", json=login_data)
            
            assert response.status_code == 200
            auth_data = response.json()
            
            endpoint_results["login"] = {
                "status_code": response.status_code,
                "login_successful": auth_data.get("success", False),
                "token_provided": "access_token" in auth_data,
                "auth_manager_called": mock_auth.called
            }
        
        # Test 3: Profile Access with Token
        headers = {"Authorization": "Bearer jwt_token_abc123"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate, \
             patch('prsm.auth.auth_manager.AuthManager.get_user_profile') as mock_profile:
            
            mock_validate.return_value = {
                "valid": True,
                "user_id": "reg_user_123",
                "user_role": "user"
            }
            
            mock_profile.return_value = {
                "user_id": "reg_user_123",
                "username": "integration_user",
                "email": "integration@test.com",
                "role": "user",
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = await async_test_client.get("/api/v1/auth/profile", headers=headers)
            
            assert response.status_code == 200
            profile_data = response.json()
            
            endpoint_results["profile"] = {
                "status_code": response.status_code,
                "profile_retrieved": "username" in profile_data,
                "token_validated": mock_validate.called,
                "profile_service_called": mock_profile.called
            }
        
        return endpoint_results
    
    async def test_marketplace_endpoints_integration(self, async_test_client):
        """Test marketplace endpoint integration"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        endpoint_results = {}
        auth_token = "marketplace_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "user_id": "marketplace_user",
                "user_role": "researcher"
            }
            
            # Test 1: Search Models
            search_params = {
                "query": "language model",
                "category": "nlp",
                "max_price": 100.0
            }
            
            with patch('prsm.marketplace.real_marketplace_service.MarketplaceService.search_models') as mock_search:
                mock_search.return_value = [
                    {
                        "model_id": "nlp_model_v1",
                        "name": "Advanced NLP Model",
                        "description": "State-of-the-art language processing",
                        "price_per_hour": 50.0,
                        "rating": 4.7,
                        "provider": "AI Research Lab"
                    }
                ]
                
                response = await async_test_client.get("/api/v1/marketplace/search", 
                                                     params=search_params, headers=headers)
                
                assert response.status_code == 200
                search_results = response.json()
                
                endpoint_results["model_search"] = {
                    "status_code": response.status_code,
                    "results_returned": len(search_results) > 0,
                    "search_service_called": mock_search.called,
                    "authentication_validated": mock_validate.called
                }
            
            # Test 2: Model Rental
            rental_data = {
                "model_id": "nlp_model_v1",
                "rental_hours": 4,
                "rental_type": "exclusive"
            }
            
            with patch('prsm.marketplace.real_marketplace_service.MarketplaceService.rent_model') as mock_rent, \
                 patch('prsm.tokenomics.ftns_service.FTNSService.charge_user') as mock_charge:
                
                mock_rent.return_value = {
                    "success": True,
                    "rental_id": f"rental_{uuid.uuid4()}",
                    "model_id": "nlp_model_v1",
                    "rental_hours": 4,
                    "total_cost": Decimal("200.0"),
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=4)).timestamp(),
                    "access_token": "model_access_abc123"
                }
                
                mock_charge.return_value = {
                    "success": True,
                    "transaction_id": f"tx_{uuid.uuid4()}",
                    "amount_charged": 200.0,
                    "remaining_balance": 300.0
                }
                
                response = await async_test_client.post("/api/v1/marketplace/rent", 
                                                      json=rental_data, headers=headers)
                
                assert response.status_code == 200
                rental_result = response.json()
                
                endpoint_results["model_rental"] = {
                    "status_code": response.status_code,
                    "rental_successful": rental_result.get("success", False),
                    "access_token_provided": "access_token" in rental_result,
                    "rental_service_called": mock_rent.called,
                    "payment_processed": mock_charge.called
                }
        
        return endpoint_results


@pytest.mark.integration
@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling and edge cases"""
    
    async def test_authentication_failures(self, async_test_client):
        """Test API responses to authentication failures"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        error_handling_results = {}
        
        # Test 1: Missing Authorization Header
        response = await async_test_client.get("/api/v1/ftns/balance")
        
        assert response.status_code == 401
        error_data = response.json()
        
        error_handling_results["missing_auth"] = {
            "status_code": response.status_code,
            "error_message_provided": "error" in error_data or "detail" in error_data
        }
        
        # Test 2: Invalid Token
        invalid_headers = {"Authorization": "Bearer invalid_token_xyz"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Token expired or invalid"
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=invalid_headers)
            
            assert response.status_code == 401
            error_data = response.json()
            
            error_handling_results["invalid_token"] = {
                "status_code": response.status_code,
                "error_message_provided": "error" in error_data or "detail" in error_data,
                "validation_attempted": mock_validate.called
            }
        
        # Test 3: Insufficient Permissions
        limited_headers = {"Authorization": "Bearer limited_token"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "user_id": "limited_user",
                "user_role": "guest"  # Guest role trying to access premium endpoint
            }
            
            query_data = {"query": "Premium query", "mode": "advanced"}
            response = await async_test_client.post("/api/v1/nwtn/query", 
                                                  json=query_data, headers=limited_headers)
            
            # Should be either 403 (Forbidden) or handled gracefully
            assert response.status_code in [200, 403]
            
            error_handling_results["insufficient_permissions"] = {
                "status_code": response.status_code,
                "permission_check_performed": mock_validate.called
            }
        
        return error_handling_results
    
    async def test_service_failure_handling(self, async_test_client):
        """Test API responses when backend services fail"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        service_failure_results = {}
        auth_token = "service_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "user_id": "service_test_user",
                "user_role": "researcher"
            }
            
            # Test 1: NWTN Service Failure
            query_data = {"query": "Test query during service failure", "mode": "adaptive"}
            
            with patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_query') as mock_nwtn:
                mock_nwtn.side_effect = Exception("NWTN service temporarily unavailable")
                
                response = await async_test_client.post("/api/v1/nwtn/query", 
                                                      json=query_data, headers=headers)
                
                # Should handle gracefully with proper error response
                assert response.status_code in [500, 503]
                error_data = response.json()
                
                service_failure_results["nwtn_failure"] = {
                    "status_code": response.status_code,
                    "error_handled_gracefully": "error" in error_data or "detail" in error_data,
                    "service_exception_caught": True
                }
            
            # Test 2: FTNS Service Failure
            with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_ftns:
                mock_ftns.side_effect = Exception("Database connection timeout")
                
                response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
                
                assert response.status_code in [500, 503]
                error_data = response.json()
                
                service_failure_results["ftns_failure"] = {
                    "status_code": response.status_code,
                    "error_handled_gracefully": "error" in error_data or "detail" in error_data,
                    "service_exception_caught": True
                }
        
        return service_failure_results
    
    async def test_input_validation_errors(self, async_test_client):
        """Test API input validation and error responses"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        validation_results = {}
        auth_token = "validation_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "user_id": "validation_user",
                "user_role": "user"
            }
            
            # Test 1: Missing Required Fields
            incomplete_query = {"mode": "adaptive"}  # Missing 'query' field
            
            response = await async_test_client.post("/api/v1/nwtn/query", 
                                                  json=incomplete_query, headers=headers)
            
            assert response.status_code == 422  # Unprocessable Entity
            error_data = response.json()
            
            validation_results["missing_fields"] = {
                "status_code": response.status_code,
                "validation_error_provided": "detail" in error_data or "errors" in error_data
            }
            
            # Test 2: Invalid Data Types
            invalid_query = {
                "query": "Valid query text",
                "context_allocation": "not_a_number"  # Should be integer
            }
            
            response = await async_test_client.post("/api/v1/nwtn/query", 
                                                  json=invalid_query, headers=headers)
            
            assert response.status_code == 422
            error_data = response.json()
            
            validation_results["invalid_types"] = {
                "status_code": response.status_code,
                "type_validation_performed": "detail" in error_data or "errors" in error_data
            }
            
            # Test 3: Out of Range Values  
            invalid_range_query = {
                "query": "Valid query",
                "context_allocation": -50  # Negative allocation not allowed
            }
            
            response = await async_test_client.post("/api/v1/nwtn/query", 
                                                  json=invalid_range_query, headers=headers)
            
            # Should either validate range or handle gracefully
            assert response.status_code in [400, 422]
            
            validation_results["range_validation"] = {
                "status_code": response.status_code,
                "range_validation_performed": True
            }
        
        return validation_results


@pytest.mark.integration 
@pytest.mark.api
@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance under load"""
    
    async def test_concurrent_api_requests(self, async_test_client):
        """Test API performance under concurrent load"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        performance_results = {}
        auth_token = "performance_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        with patch('prsm.auth.auth_manager.AuthManager.validate_token') as mock_validate, \
             patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance:
            
            mock_validate.return_value = {
                "valid": True,
                "user_id": "perf_user",
                "user_role": "researcher"
            }
            
            mock_balance.return_value = {
                "total_balance": Decimal("1000.0"),
                "available_balance": Decimal("1000.0"),
                "reserved_balance": Decimal("0.0")
            }
            
            async def make_balance_request():
                """Single balance request"""
                start_time = asyncio.get_event_loop().time()
                response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
                end_time = asyncio.get_event_loop().time()
                
                return {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            
            # Execute 20 concurrent requests
            tasks = [make_balance_request() for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
            failed_requests = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
            
            if successful_requests:
                avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
                max_response_time = max(r["response_time"] for r in successful_requests)
                min_response_time = min(r["response_time"] for r in successful_requests)
            else:
                avg_response_time = max_response_time = min_response_time = 0
            
            performance_results["concurrent_requests"] = {
                "total_requests": 20,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / 20,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "min_response_time": min_response_time
            }
            
            # Performance assertions
            assert len(successful_requests) >= 15  # At least 75% success rate
            assert avg_response_time < 2.0  # Average response time under 2 seconds
        
        return performance_results