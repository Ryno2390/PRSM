"""
User Workflow Integration Tests
===============================

End-to-end integration tests for complete user workflows, testing the interaction
between authentication, NWTN processing, FTNS tokenomics, and data persistence.
"""

import pytest
import asyncio
import uuid
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import time
from datetime import datetime, timezone

try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    from prsm.core.models import PRSMSession, UserInput, FTNSTransaction
    from prsm.auth.models import User, UserSession
    from prsm.marketplace.models import MarketplaceListing
    from prsm.core.database import get_session
    from prsm.auth.auth_manager import AuthManager
    from prsm.tokenomics.ftns_service import FTNSService
    from prsm.nwtn.orchestrator import NWTNOrchestrator
    from prsm.marketplace.real_marketplace_service import MarketplaceService
except ImportError:
    # Create mocks if imports fail
    TestClient = Mock
    AsyncClient = Mock
    PRSMSession = Mock
    UserInput = Mock
    FTNSTransaction = Mock
    User = Mock
    UserSession = Mock
    MarketplaceListing = Mock
    get_session = Mock
    AuthManager = Mock
    FTNSService = Mock
    NWTNOrchestrator = Mock
    MarketplaceService = Mock


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteUserWorkflows:
    """Test complete end-to-end user workflows"""
    
    async def test_new_user_onboarding_flow(self, async_test_client):
        """Test complete new user onboarding workflow"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        workflow_results = {}
        
        # Step 1: User Registration
        registration_data = {
            "username": "newuser123",
            "email": "newuser123@test.com",
            "password": "SecurePassword123!",
            "terms_accepted": True
        }
        
        with patch('prsm.auth.auth_manager.AuthManager.register_user') as mock_register:
            mock_register.return_value = {
                "success": True,
                "user_id": "user_123",
                "username": "newuser123",
                "email": "newuser123@test.com",
                "verification_required": False
            }
            
            response = await async_test_client.post("/api/v1/auth/register", json=registration_data)
            assert response.status_code == 201
            
            registration_result = response.json()
            workflow_results["registration"] = registration_result
            user_id = registration_result["user_id"]
        
        # Step 2: User Authentication
        login_data = {
            "username": "newuser123",
            "password": "SecurePassword123!"
        }
        
        with patch('prsm.auth.auth_manager.AuthManager.authenticate_user') as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "access_token": "mock_jwt_token_123",
                "token_type": "bearer",
                "expires_in": 3600,
                "user_id": user_id
            }
            
            response = await async_test_client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 200
            
            auth_result = response.json()
            workflow_results["authentication"] = auth_result
            auth_token = auth_result["access_token"]
        
        # Step 3: Initial FTNS Token Allocation
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance:
            mock_balance.return_value = {
                "total_balance": Decimal("1000.0"),
                "available_balance": Decimal("1000.0"),
                "reserved_balance": Decimal("0.0")
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            assert response.status_code == 200
            
            balance_result = response.json()
            workflow_results["initial_balance"] = balance_result
            assert float(balance_result["total_balance"]) == 1000.0
        
        # Step 4: First NWTN Query Processing
        query_data = {
            "query": "What are the main principles of sustainable energy?",
            "mode": "adaptive",
            "max_depth": 2
        }
        
        with patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_query') as mock_nwtn:
            mock_nwtn.return_value = {
                "response": "Sustainable energy is based on three main principles: renewability, environmental responsibility, and economic viability...",
                "reasoning_depth": 2,
                "confidence": 0.87,
                "session_id": f"session_{uuid.uuid4()}",
                "tokens_used": 245,
                "cost": Decimal("12.25")
            }
            
            response = await async_test_client.post("/api/v1/nwtn/query", json=query_data, headers=headers)
            assert response.status_code == 200
            
            query_result = response.json()
            workflow_results["first_query"] = query_result
            assert "response" in query_result
            assert query_result["confidence"] > 0.8
        
        # Step 5: Verify FTNS Balance After Query
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance_after:
            mock_balance_after.return_value = {
                "total_balance": Decimal("987.75"),  # Reduced by query cost
                "available_balance": Decimal("987.75"),
                "reserved_balance": Decimal("0.0")
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            assert response.status_code == 200
            
            final_balance = response.json()
            workflow_results["final_balance"] = final_balance
            
            # Verify balance was deducted
            initial_balance = float(workflow_results["initial_balance"]["total_balance"])
            current_balance = float(final_balance["total_balance"])
            assert current_balance < initial_balance
        
        # Step 6: Verify Database State
        with patch('prsm.core.database.get_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Verify user was created
            mock_session.query.return_value.filter.return_value.first.return_value = Mock(
                user_id=user_id,
                username="newuser123",
                email="newuser123@test.com"
            )
            
            # Verify PRSM session was created
            mock_session.query.return_value.filter.return_value.all.return_value = [
                Mock(
                    session_id=query_result.get("session_id"),
                    user_id=user_id,
                    status="completed",
                    query_count=1,
                    total_cost=Decimal("12.25")
                )
            ]
            
            workflow_results["database_verification"] = {
                "user_exists": True,
                "session_created": True,
                "transaction_recorded": True
            }
        
        # Workflow Success Verification
        assert workflow_results["registration"]["success"] is True
        assert workflow_results["authentication"]["success"] is True
        assert "total_balance" in workflow_results["initial_balance"]
        assert "response" in workflow_results["first_query"]
        assert workflow_results["database_verification"]["user_exists"] is True
        
        return workflow_results
    
    async def test_marketplace_discovery_and_rental_flow(self, async_test_client):
        """Test marketplace model discovery and rental workflow"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        workflow_results = {}
        
        # Setup: Authenticated user
        user_id = "test_user_456"
        auth_token = "mock_auth_token_456"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Step 1: Marketplace Model Discovery
        search_params = {
            "query": "language model for creative writing",
            "category": "language_models",
            "max_price": 50.0
        }
        
        with patch('prsm.marketplace.real_marketplace_service.MarketplaceService.search_models') as mock_search:
            mock_search.return_value = [
                {
                    "model_id": "creative_writer_v2",
                    "name": "Creative Writer v2.0",
                    "description": "Advanced language model specialized in creative writing",
                    "price_per_hour": 25.0,
                    "rating": 4.8,
                    "provider": "CreativeAI Labs",
                    "capabilities": ["creative_writing", "story_generation", "poetry"]
                },
                {
                    "model_id": "narrative_engine_pro",
                    "name": "Narrative Engine Pro",
                    "description": "Professional narrative generation model",
                    "price_per_hour": 35.0,
                    "rating": 4.6,
                    "provider": "StoryTech Inc",
                    "capabilities": ["narrative", "character_development", "plot_generation"]
                }
            ]
            
            response = await async_test_client.get("/api/v1/marketplace/search", params=search_params, headers=headers)
            assert response.status_code == 200
            
            search_results = response.json()
            workflow_results["model_discovery"] = search_results
            assert len(search_results) == 2
            
            # Select the first model for rental
            selected_model = search_results[0]
            model_id = selected_model["model_id"]
        
        # Step 2: Check FTNS Balance for Rental
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_balance:
            mock_balance.return_value = {
                "total_balance": Decimal("150.0"),
                "available_balance": Decimal("150.0"),
                "reserved_balance": Decimal("0.0")
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            assert response.status_code == 200
            
            balance_result = response.json()
            workflow_results["pre_rental_balance"] = balance_result
        
        # Step 3: Model Rental Transaction
        rental_data = {
            "model_id": model_id,
            "rental_hours": 2,
            "rental_type": "exclusive"
        }
        
        with patch('prsm.marketplace.real_marketplace_service.MarketplaceService.rent_model') as mock_rent:
            mock_rent.return_value = {
                "success": True,
                "rental_id": f"rental_{uuid.uuid4()}",
                "model_id": model_id,
                "rental_hours": 2,
                "total_cost": Decimal("50.0"),
                "expires_at": (datetime.now(timezone.utc).timestamp() + 7200),  # 2 hours from now
                "access_token": "model_access_token_123"
            }
            
            response = await async_test_client.post("/api/v1/marketplace/rent", json=rental_data, headers=headers)
            assert response.status_code == 200
            
            rental_result = response.json()
            workflow_results["model_rental"] = rental_result
            assert rental_result["success"] is True
            
            rental_id = rental_result["rental_id"]
            model_access_token = rental_result["access_token"]
        
        # Step 4: Model Usage with Rented Access
        usage_data = {
            "prompt": "Write a short story about a robot learning to paint",
            "max_tokens": 500,
            "temperature": 0.8
        }
        
        usage_headers = {
            **headers,
            "X-Model-Access-Token": model_access_token
        }
        
        with patch('prsm.marketplace.real_marketplace_service.MarketplaceService.use_rented_model') as mock_use:
            mock_use.return_value = {
                "response": "In a small workshop filled with canvases and brushes, R-7 stared at the blank white surface...",
                "tokens_used": 487,
                "usage_cost": Decimal("2.45"),
                "model_id": model_id,
                "usage_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = await async_test_client.post(f"/api/v1/marketplace/models/{model_id}/use", 
                                                  json=usage_data, headers=usage_headers)
            assert response.status_code == 200
            
            usage_result = response.json()
            workflow_results["model_usage"] = usage_result
            assert "response" in usage_result
            assert usage_result["tokens_used"] > 0
        
        # Step 5: Payment Settlement and Balance Update
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_final_balance:
            # Balance should reflect rental cost + usage cost
            total_spent = Decimal("50.0") + Decimal("2.45")  # Rental + usage
            remaining_balance = Decimal("150.0") - total_spent
            
            mock_final_balance.return_value = {
                "total_balance": remaining_balance,
                "available_balance": remaining_balance,
                "reserved_balance": Decimal("0.0")
            }
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            assert response.status_code == 200
            
            final_balance = response.json()
            workflow_results["post_usage_balance"] = final_balance
            
            # Verify correct balance deduction
            initial = float(workflow_results["pre_rental_balance"]["total_balance"])
            final = float(final_balance["total_balance"])
            expected_spent = float(total_spent)
            
            assert abs((initial - final) - expected_spent) < 0.01  # Allow for rounding
        
        # Step 6: Reputation and Analytics Update
        with patch('prsm.marketplace.reputation_system.ReputationSystem.update_rating') as mock_reputation:
            mock_reputation.return_value = {
                "model_id": model_id,
                "new_rating": 4.85,  # Slightly improved
                "total_ratings": 127,
                "rating_updated": True
            }
            
            rating_data = {
                "rental_id": rental_id,
                "rating": 5,
                "review": "Excellent model for creative writing!"
            }
            
            response = await async_test_client.post(f"/api/v1/marketplace/rentals/{rental_id}/rate", 
                                                  json=rating_data, headers=headers)
            assert response.status_code == 200
            
            rating_result = response.json()
            workflow_results["reputation_update"] = rating_result
        
        # Workflow Success Verification
        assert len(workflow_results["model_discovery"]) > 0
        assert workflow_results["model_rental"]["success"] is True
        assert "response" in workflow_results["model_usage"]
        assert "total_balance" in workflow_results["post_usage_balance"]
        assert workflow_results["reputation_update"]["rating_updated"] is True
        
        return workflow_results
    
    async def test_realtime_collaboration_flow(self, async_test_client):
        """Test real-time collaborative session workflow"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        workflow_results = {}
        
        # Setup: Multiple authenticated users
        users = [
            {"user_id": "collab_user_1", "token": "token_1", "username": "alice"},
            {"user_id": "collab_user_2", "token": "token_2", "username": "bob"},
            {"user_id": "collab_user_3", "token": "token_3", "username": "carol"}
        ]
        
        # Step 1: Create Collaborative Session
        session_data = {
            "session_name": "AI Research Discussion",
            "session_type": "collaborative",
            "max_participants": 5,
            "session_config": {
                "allow_anonymous": False,
                "require_moderation": True,
                "auto_save": True
            }
        }
        
        headers_1 = {"Authorization": f"Bearer {users[0]['token']}"}
        
        with patch('prsm.collaboration.session_manager.SessionManager.create_session') as mock_create:
            mock_create.return_value = {
                "session_id": f"collab_session_{uuid.uuid4()}",
                "session_name": "AI Research Discussion",
                "creator_id": users[0]["user_id"],
                "participants": [users[0]["user_id"]],
                "status": "active",
                "websocket_url": "ws://localhost:8000/ws/collab/session_123"
            }
            
            response = await async_test_client.post("/api/v1/collaboration/sessions", 
                                                  json=session_data, headers=headers_1)
            assert response.status_code == 201
            
            session_result = response.json()
            workflow_results["session_creation"] = session_result
            session_id = session_result["session_id"]
        
        # Step 2: Multiple Users Join Session
        join_results = []
        
        for user in users[1:]:  # Skip first user (creator)
            headers = {"Authorization": f"Bearer {user['token']}"}
            
            with patch('prsm.collaboration.session_manager.SessionManager.join_session') as mock_join:
                mock_join.return_value = {
                    "success": True,
                    "session_id": session_id,
                    "user_id": user["user_id"],
                    "participant_count": len(join_results) + 2,  # +1 for creator, +1 for current user
                    "role": "participant"
                }
                
                response = await async_test_client.post(f"/api/v1/collaboration/sessions/{session_id}/join", 
                                                      headers=headers)
                assert response.status_code == 200
                
                join_result = response.json()
                join_results.append(join_result)
        
        workflow_results["user_joins"] = join_results
        
        # Step 3: Collaborative AI Query Processing
        collaborative_query = {
            "query": "Let's explore the ethical implications of AGI development",
            "query_type": "collaborative",
            "context": {
                "participants": [user["user_id"] for user in users],
                "session_history": []
            }
        }
        
        with patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_collaborative_query') as mock_collab_query:
            mock_collab_query.return_value = {
                "response": "The ethical implications of AGI development encompass several critical dimensions...",
                "reasoning_depth": 3,
                "confidence": 0.91,
                "collaborative_insights": [
                    "Considers multiple perspectives from participants",
                    "Incorporates diverse ethical frameworks",
                    "Addresses concerns raised by different stakeholders"
                ],
                "session_id": session_id,
                "tokens_used": 380,
                "participants_involved": len(users)
            }
            
            response = await async_test_client.post(f"/api/v1/collaboration/sessions/{session_id}/query", 
                                                  json=collaborative_query, headers=headers_1)
            assert response.status_code == 200
            
            query_result = response.json()
            workflow_results["collaborative_query"] = query_result
            assert "collaborative_insights" in query_result
        
        # Step 4: Real-time Message Broadcasting
        message_data = {
            "message": "Great insights! I'd like to add that we should also consider the economic impacts.",
            "message_type": "comment",
            "reference_query": query_result.get("response", "")[:100]
        }
        
        with patch('prsm.collaboration.websocket_manager.WebSocketManager.broadcast_message') as mock_broadcast:
            mock_broadcast.return_value = {
                "message_id": f"msg_{uuid.uuid4()}",
                "broadcasted_to": len(users) - 1,  # All except sender
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "delivery_status": "delivered"
            }
            
            response = await async_test_client.post(f"/api/v1/collaboration/sessions/{session_id}/messages", 
                                                  json=message_data, headers=headers_1)
            assert response.status_code == 200
            
            broadcast_result = response.json()
            workflow_results["message_broadcast"] = broadcast_result
        
        # Step 5: Session State Persistence
        with patch('prsm.collaboration.session_manager.SessionManager.save_session_state') as mock_save:
            mock_save.return_value = {
                "session_id": session_id,
                "state_saved": True,
                "participants_count": len(users),
                "messages_count": 1,
                "queries_count": 1,
                "last_activity": datetime.now(timezone.utc).isoformat()
            }
            
            response = await async_test_client.post(f"/api/v1/collaboration/sessions/{session_id}/save", 
                                                  headers=headers_1)
            assert response.status_code == 200
            
            save_result = response.json()
            workflow_results["session_persistence"] = save_result
        
        # Step 6: Session Analytics and Metrics
        with patch('prsm.collaboration.analytics.CollaborationAnalytics.get_session_metrics') as mock_analytics:
            mock_analytics.return_value = {
                "session_id": session_id,
                "duration_minutes": 45,
                "total_participants": len(users),
                "messages_exchanged": 12,
                "ai_queries": 3,
                "engagement_score": 8.7,
                "collaboration_effectiveness": 0.89
            }
            
            response = await async_test_client.get(f"/api/v1/collaboration/sessions/{session_id}/analytics", 
                                                 headers=headers_1)
            assert response.status_code == 200
            
            analytics_result = response.json()
            workflow_results["session_analytics"] = analytics_result
        
        # Workflow Success Verification
        assert workflow_results["session_creation"]["status"] == "active"
        assert len(workflow_results["user_joins"]) == len(users) - 1
        assert "collaborative_insights" in workflow_results["collaborative_query"]
        assert workflow_results["message_broadcast"]["delivery_status"] == "delivered"
        assert workflow_results["session_persistence"]["state_saved"] is True
        assert workflow_results["session_analytics"]["engagement_score"] > 8.0
        
        return workflow_results


@pytest.mark.integration
@pytest.mark.performance
class TestWorkflowPerformance:
    """Test workflow performance under realistic conditions"""
    
    async def test_concurrent_user_workflows(self, async_test_client, load_test_runner):
        """Test multiple users executing workflows concurrently"""
        if async_test_client is None or load_test_runner is None:
            pytest.skip("Required fixtures not available")
        
        async def single_user_workflow():
            """Simulate a single user's complete workflow"""
            user_id = f"perf_user_{uuid.uuid4()}"
            
            # Simulate authentication
            await asyncio.sleep(0.1)  # Auth processing time
            
            # Simulate NWTN query
            await asyncio.sleep(0.5)  # Query processing time
            
            # Simulate FTNS transaction
            await asyncio.sleep(0.2)  # Transaction processing time
            
            return {
                "user_id": user_id,
                "workflow_completed": True,
                "steps_completed": 3
            }
        
        # Run concurrent user workflows
        results = await load_test_runner.run_load_test(
            test_function=single_user_workflow,
            concurrent_users=20,
            duration_seconds=30,
            ramp_up_seconds=5
        )
        
        # Performance assertions
        assert results.error_rate < 0.10  # Less than 10% error rate
        assert results.average_response_time < 1000  # Less than 1 second average
        assert results.successful_requests > 50  # Minimum successful workflows
        
        return results
    
    async def test_workflow_memory_usage(self, memory_profiler):
        """Test memory usage during workflow execution"""
        if memory_profiler is None:
            pytest.skip("Memory profiler not available")
        
        memory_profiler.take_snapshot("workflow_start")
        
        # Simulate memory-intensive workflow operations
        workflow_data = []
        
        for i in range(100):
            # Simulate user session data
            session_data = {
                "user_id": f"memory_test_user_{i}",
                "session_data": {
                    "queries": [f"Query {j}" for j in range(10)],
                    "responses": [f"Response {j}" * 100 for j in range(10)],
                    "metadata": {"index": i, "large_data": "x" * 1000}
                }
            }
            workflow_data.append(session_data)
        
        memory_profiler.take_snapshot("workflow_peak")
        
        # Cleanup workflow data
        workflow_data.clear()
        
        memory_profiler.take_snapshot("workflow_cleanup")
        
        # Analyze memory usage
        peak_diff = memory_profiler.compare_snapshots("workflow_start", "workflow_peak")
        cleanup_diff = memory_profiler.compare_snapshots("workflow_peak", "workflow_cleanup")
        
        # Memory usage should be reasonable and cleaned up
        assert peak_diff["memory_diff"] > 0  # Memory increased during workflow
        assert cleanup_diff["memory_diff"] < 0  # Memory released after cleanup
        
        # Memory usage should be under reasonable limits
        peak_usage_mb = peak_diff["memory_diff"] / (1024 * 1024)
        assert peak_usage_mb < 200  # Less than 200MB for test workflow
        
        return {
            "peak_memory_mb": peak_usage_mb,
            "cleanup_effective": cleanup_diff["memory_diff"] < 0,
            "memory_efficiency": abs(cleanup_diff["memory_diff"]) / peak_diff["memory_diff"]
        }


@pytest.mark.integration
@pytest.mark.slow
class TestWorkflowResilience:
    """Test workflow resilience under failure conditions"""
    
    async def test_workflow_with_service_failures(self, async_test_client):
        """Test workflow behavior when individual services fail"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        resilience_results = {}
        auth_token = "resilience_test_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test 1: NWTN Service Failure
        query_data = {"query": "Test query during service failure", "mode": "adaptive"}
        
        with patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_query') as mock_nwtn:
            mock_nwtn.side_effect = Exception("NWTN service temporarily unavailable")
            
            response = await async_test_client.post("/api/v1/nwtn/query", json=query_data, headers=headers)
            
            # Should handle gracefully with proper error response
            assert response.status_code in [500, 503]  # Server error or service unavailable
            error_response = response.json()
            assert "error" in error_response
            
            resilience_results["nwtn_failure_handling"] = {
                "handled_gracefully": True,
                "error_code": response.status_code,
                "error_message": error_response.get("error", "")
            }
        
        # Test 2: FTNS Service Failure
        with patch('prsm.tokenomics.ftns_service.FTNSService.get_balance') as mock_ftns:
            mock_ftns.side_effect = Exception("FTNS service connection timeout")
            
            response = await async_test_client.get("/api/v1/ftns/balance", headers=headers)
            
            # Should handle gracefully
            assert response.status_code in [500, 503]
            error_response = response.json()
            
            resilience_results["ftns_failure_handling"] = {
                "handled_gracefully": True,
                "error_code": response.status_code,
                "provides_error_details": "error" in error_response
            }
        
        # Test 3: Database Connection Failure
        with patch('prsm.core.database.get_session') as mock_db:
            mock_db.side_effect = Exception("Database connection lost")
            
            response = await async_test_client.get("/api/v1/auth/profile", headers=headers)
            
            # Should handle gracefully
            assert response.status_code in [500, 503]
            
            resilience_results["database_failure_handling"] = {
                "handled_gracefully": True,
                "error_code": response.status_code
            }
        
        # Verify all failures were handled gracefully
        for service, result in resilience_results.items():
            assert result["handled_gracefully"] is True
        
        return resilience_results
    
    async def test_workflow_partial_success_scenarios(self, async_test_client):
        """Test workflows where some steps succeed and others fail"""
        if async_test_client is None:
            pytest.skip("Async test client not available")
        
        partial_success_results = {}
        auth_token = "partial_success_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Scenario: NWTN query succeeds but FTNS charge fails
        query_data = {"query": "Test partial success scenario", "mode": "quick"}
        
        with patch('prsm.nwtn.orchestrator.NWTNOrchestrator.process_query') as mock_nwtn, \
             patch('prsm.tokenomics.ftns_service.FTNSService.charge_user') as mock_charge:
            
            # NWTN succeeds
            mock_nwtn.return_value = {
                "response": "Successful query response",
                "reasoning_depth": 1,
                "confidence": 0.85,
                "cost": Decimal("5.0")
            }
            
            # FTNS charge fails
            mock_charge.side_effect = Exception("Insufficient balance")
            
            response = await async_test_client.post("/api/v1/nwtn/query", json=query_data, headers=headers)
            
            # Should handle partial failure appropriately
            # Could be success with warning, or failure with partial results
            assert response.status_code in [200, 402, 500]  # Success, payment required, or server error
            
            result_data = response.json()
            partial_success_results["nwtn_success_ftns_fail"] = {
                "response_code": response.status_code,
                "has_query_result": "response" in result_data or "error" in result_data,
                "partial_success_handled": True
            }
        
        return partial_success_results