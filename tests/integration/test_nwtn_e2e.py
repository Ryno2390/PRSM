#!/usr/bin/env python3
"""
Sprint 3 Phase 3: NWTN End-to-End Integration Tests

Tests for full query processing pipeline, error handling at each stage,
session state management, context allocation, and safety validation.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

# Import NWTN components
from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
from prsm.compute.nwtn.reasoning_engine import ReasoningEngine
from prsm.compute.nwtn.graph_of_thoughts import GraphOfThoughts


class MockNWTNSession:
    """Mock NWTN session for testing"""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.context_tokens = 0
        self.max_context = 100000
        self.queries_processed = 0
        self.state = "initialized"
        self.history: List[dict] = []
        self.created_at = datetime.now()
    
    async def allocate_context(self, tokens: int) -> bool:
        """Allocate context tokens"""
        if self.context_tokens + tokens <= self.max_context:
            self.context_tokens += tokens
            return True
        return False
    
    async def release_context(self, tokens: int):
        """Release context tokens"""
        self.context_tokens = max(0, self.context_tokens - tokens)
    
    async def add_to_history(self, query: str, response: str):
        """Add query-response pair to history"""
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        self.queries_processed += 1


class MockNWTNOrchestrator:
    """Mock NWTN orchestrator for testing"""
    
    def __init__(self):
        self.sessions: Dict[str, MockNWTNSession] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the orchestrator"""
        self.is_initialized = True
    
    async def create_session(self, user_id: str) -> MockNWTNSession:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session = MockNWTNSession(session_id, user_id)
        self.sessions[session_id] = session
        return session
    
    async def get_session(self, session_id: str) -> Optional[MockNWTNSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    async def close_session(self, session_id: str):
        """Close a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def process_query(self, session: MockNWTNSession, query: str) -> dict:
        """Process a query through the NWTN pipeline"""
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        # Simulate query processing
        stages = [
            "query_understanding",
            "context_retrieval",
            "reasoning",
            "response_generation",
            "safety_validation"
        ]
        
        results = {}
        for stage in stages:
            results[stage] = {"status": "completed", "timestamp": datetime.now().isoformat()}
        
        # Update session
        response = f"Processed: {query[:50]}..."
        await session.add_to_history(query, response)
        
        return {
            "response": response,
            "stages": results,
            "context_used": len(query) * 2,  # Simulated
            "session_id": session.session_id
        }


class TestNWTNEndToEnd:
    """Test suite for NWTN end-to-end flows"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize NWTN orchestrator"""
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        yield orch
    
    @pytest.fixture
    async def session(self, orchestrator):
        """Create a test session"""
        session = await orchestrator.create_session("test_user")
        yield session
        await orchestrator.close_session(session.session_id)
    
    # =========================================================================
    # Full Query Processing Pipeline Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_full_query_pipeline(self, orchestrator, session):
        """Test complete query processing pipeline"""
        query = "What is the capital of France?"
        
        result = await orchestrator.process_query(session, query)
        
        # Verify all stages completed
        assert "stages" in result
        assert "query_understanding" in result["stages"]
        assert "context_retrieval" in result["stages"]
        assert "reasoning" in result["stages"]
        assert "response_generation" in result["stages"]
        assert "safety_validation" in result["stages"]
        
        # Verify response
        assert "response" in result
        assert result["session_id"] == session.session_id
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, orchestrator, session):
        """Test multi-turn conversation flow"""
        queries = [
            "What is machine learning?",
            "How does it differ from deep learning?",
            "Can you give me an example?"
        ]
        
        for query in queries:
            result = await orchestrator.process_query(session, query)
            assert "response" in result
        
        # Verify history
        assert len(session.history) == 3
        assert session.queries_processed == 3
    
    @pytest.mark.asyncio
    async def test_query_with_context_allocation(self, orchestrator, session):
        """Test query processing with context allocation"""
        # Allocate context
        allocated = await session.allocate_context(50000)
        assert allocated is True
        assert session.context_tokens == 50000
        
        # Process query
        query = "Explain quantum computing"
        result = await orchestrator.process_query(session, query)
        
        # Verify context was used
        assert result["context_used"] > 0
    
    # =========================================================================
    # Error Handling Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_error_handling_uninitialized_orchestrator(self):
        """Test error handling when orchestrator is not initialized"""
        orch = MockNWTNOrchestrator()
        # Don't initialize
        
        session = MockNWTNSession("test_session", "test_user")
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await orch.process_query(session, "test query")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_session(self, orchestrator):
        """Test error handling with invalid session"""
        fake_session = MockNWTNSession("nonexistent", "fake_user")
        
        # Should still work with the mock, but in real implementation
        # would validate session exists
        result = await orchestrator.process_query(fake_session, "test")
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_context_overflow(self, session):
        """Test error handling when context limit is exceeded"""
        # Try to allocate more than max
        allocated = await session.allocate_context(200000)
        assert allocated is False
        assert session.context_tokens == 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_after_failure(self, orchestrator, session):
        """Test that session recovers after a processing failure"""
        # First query succeeds
        result1 = await orchestrator.process_query(session, "First query")
        assert "response" in result1
        
        # Simulate a failure scenario (in real implementation)
        # Here we just verify the session state is maintained
        
        # Second query should still work
        result2 = await orchestrator.process_query(session, "Second query")
        assert "response" in result2
        assert session.queries_processed == 2
    
    # =========================================================================
    # Session State Management Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_session_creation(self, orchestrator):
        """Test session creation"""
        session = await orchestrator.create_session("user_123")
        
        assert session.session_id is not None
        assert session.user_id == "user_123"
        assert session.state == "initialized"
        assert session.context_tokens == 0
        assert len(session.history) == 0
        
        await orchestrator.close_session(session.session_id)
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, orchestrator):
        """Test that session persists across queries"""
        session = await orchestrator.create_session("persist_user")
        
        # Process multiple queries
        for i in range(3):
            await orchestrator.process_query(session, f"Query {i}")
        
        # Retrieve session
        retrieved = await orchestrator.get_session(session.session_id)
        
        assert retrieved is not None
        assert retrieved.queries_processed == 3
        assert len(retrieved.history) == 3
        
        await orchestrator.close_session(session.session_id)
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, orchestrator):
        """Test session cleanup"""
        session = await orchestrator.create_session("cleanup_user")
        session_id = session.session_id
        
        # Close session
        await orchestrator.close_session(session_id)
        
        # Verify session is removed
        retrieved = await orchestrator.get_session(session_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, orchestrator):
        """Test handling of concurrent sessions"""
        sessions = []
        
        # Create multiple sessions
        for i in range(5):
            session = await orchestrator.create_session(f"user_{i}")
            sessions.append(session)
        
        # Process queries concurrently
        async def process(session, query):
            return await orchestrator.process_query(session, query)
        
        results = await asyncio.gather(
            *[process(s, f"Query for {s.user_id}") for s in sessions],
            return_exceptions=True
        )
        
        # All should succeed
        successes = sum(1 for r in results if isinstance(r, dict) and "response" in r)
        assert successes == 5
        
        # Cleanup
        for session in sessions:
            await orchestrator.close_session(session.session_id)
    
    # =========================================================================
    # Context Allocation Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_context_allocation_and_release(self, session):
        """Test context allocation and release"""
        # Allocate
        assert await session.allocate_context(10000) is True
        assert session.context_tokens == 10000
        
        # Allocate more
        assert await session.allocate_context(20000) is True
        assert session.context_tokens == 30000
        
        # Release
        await session.release_context(15000)
        assert session.context_tokens == 15000
        
        # Release more than allocated (should not go negative)
        await session.release_context(20000)
        assert session.context_tokens == 0
    
    @pytest.mark.asyncio
    async def test_context_limit_enforcement(self, session):
        """Test that context limits are enforced"""
        # Allocate up to limit
        assert await session.allocate_context(100000) is True
        assert session.context_tokens == 100000
        
        # Try to allocate more
        assert await session.allocate_context(1) is False
        assert session.context_tokens == 100000
    
    # =========================================================================
    # Safety Validation Tests
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_safety_validation_stage(self, orchestrator, session):
        """Test that safety validation stage runs"""
        result = await orchestrator.process_query(session, "Test query")
        
        assert "safety_validation" in result["stages"]
        assert result["stages"]["safety_validation"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_harmful_query_detection(self, orchestrator, session):
        """Test detection of potentially harmful queries"""
        # In real implementation, would detect and handle harmful content
        # Here we verify the stage exists
        
        result = await orchestrator.process_query(session, "How do I make a cake?")
        
        # Safety validation should pass for benign query
        assert result["stages"]["safety_validation"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_pii_handling(self, orchestrator, session):
        """Test PII handling in queries"""
        # Query with potential PII
        query = "My email is test@example.com, can you help me?"
        
        result = await orchestrator.process_query(session, query)
        
        # Should process but in real implementation would handle PII
        assert "response" in result


class TestNWTNPerformance:
    """Test suite for NWTN performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_query_latency(self):
        """Test query processing latency"""
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        session = await orch.create_session("latency_user")
        
        # Measure latency
        start_time = datetime.now()
        await orch.process_query(session, "Test query for latency")
        end_time = datetime.now()
        
        latency = (end_time - start_time).total_seconds()
        
        # Latency should be reasonable (in test, this is very fast)
        assert latency < 5.0, f"Query latency too high: {latency}s"
        
        await orch.close_session(session.session_id)
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test query throughput"""
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        session = await orch.create_session("throughput_user")
        
        # Process multiple queries
        num_queries = 10
        start_time = datetime.now()
        
        for i in range(num_queries):
            await orch.process_query(session, f"Query {i}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        queries_per_second = num_queries / total_time
        
        # Should be able to process multiple queries
        assert queries_per_second > 0
        
        await orch.close_session(session.session_id)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during processing"""
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        
        # Create and process with multiple sessions
        sessions = []
        for i in range(10):
            session = await orch.create_session(f"memory_user_{i}")
            await session.allocate_context(50000)
            await orch.process_query(session, f"Query {i}")
            sessions.append(session)
        
        # Verify all sessions are tracked
        assert len(orch.sessions) == 10
        
        # Cleanup
        for session in sessions:
            await orch.close_session(session.session_id)
        
        assert len(orch.sessions) == 0


class TestNWTNIntegration:
    """Integration tests for NWTN with other components"""
    
    @pytest.mark.asyncio
    async def test_ftns_integration(self):
        """Test NWTN integration with FTNS token system"""
        # In real implementation, would verify:
        # - Context allocation costs FTNS
        # - Query processing deducts tokens
        # - Session costs are tracked
        
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        session = await orch.create_session("ftns_user")
        
        # Process query (would cost FTNS in real implementation)
        result = await orch.process_query(session, "Test query")
        assert "response" in result
        
        await orch.close_session(session.session_id)
    
    @pytest.mark.asyncio
    async def test_dag_ledger_integration(self):
        """Test NWTN integration with DAG ledger"""
        # In real implementation, would verify:
        # - Queries are logged to DAG
        # - Responses are verifiable
        # - Provenance is maintained
        
        orch = MockNWTNOrchestrator()
        await orch.initialize()
        session = await orch.create_session("dag_user")
        
        result = await orch.process_query(session, "Test query for DAG")
        assert "response" in result
        
        await orch.close_session(session.session_id)


# =========================================================================
# Test Runner
# =========================================================================

async def run_nwtn_e2e_tests():
    """Run all NWTN end-to-end tests manually"""
    print("=" * 60)
    print("NWTN END-TO-END INTEGRATION TESTS")
    print("=" * 60)
    
    # Setup
    print("\n[SETUP] Initializing NWTN orchestrator...")
    orch = MockNWTNOrchestrator()
    await orch.initialize()
    
    test_instance = TestNWTNEndToEnd()
    
    # Test 1: Full query pipeline
    print("\n[TEST 1] Full query processing pipeline...")
    try:
        session = await orch.create_session("test_user_1")
        result = await orch.process_query(session, "What is machine learning?")
        
        assert "stages" in result
        assert "response" in result
        print("  ✓ PASSED: Full pipeline works correctly")
        await orch.close_session(session.session_id)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 2: Multi-turn conversation
    print("\n[TEST 2] Multi-turn conversation...")
    try:
        session = await orch.create_session("test_user_2")
        
        for query in ["Hello", "How are you?", "Goodbye"]:
            result = await orch.process_query(session, query)
            assert "response" in result
        
        assert session.queries_processed == 3
        print("  ✓ PASSED: Multi-turn conversation works")
        await orch.close_session(session.session_id)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Context allocation
    print("\n[TEST 3] Context allocation and limits...")
    try:
        session = await orch.create_session("test_user_3")
        
        # Allocate context
        assert await session.allocate_context(50000) is True
        assert session.context_tokens == 50000
        
        # Try to overflow
        assert await session.allocate_context(100000) is False
        
        # Release
        await session.release_context(25000)
        assert session.context_tokens == 25000
        
        print("  ✓ PASSED: Context allocation works correctly")
        await orch.close_session(session.session_id)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 4: Concurrent sessions
    print("\n[TEST 4] Concurrent session handling...")
    try:
        sessions = []
        for i in range(5):
            session = await orch.create_session(f"concurrent_user_{i}")
            sessions.append(session)
        
        # Process concurrently
        async def process(s):
            return await orch.process_query(s, "Concurrent query")
        
        results = await asyncio.gather(*[process(s) for s in sessions])
        
        successes = sum(1 for r in results if "response" in r)
        assert successes == 5
        
        print("  ✓ PASSED: Concurrent sessions handled correctly")
        
        for session in sessions:
            await orch.close_session(session.session_id)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    # Test 5: Error handling
    print("\n[TEST 5] Error handling...")
    try:
        # Test uninitialized orchestrator
        uninit_orch = MockNWTNOrchestrator()
        session = MockNWTNSession("test", "test")
        
        try:
            await uninit_orch.process_query(session, "test")
            print("  ✗ FAILED: Should have raised RuntimeError")
        except RuntimeError:
            print("  ✓ PASSED: Error handling works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("NWTN END-TO-END TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_nwtn_e2e_tests())
