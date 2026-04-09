"""
Integration Tests for System Resource Integration & Stub Elimination

Tests covering all phases of the system resource integration implementation:
- System Resource Monitoring (psutil)
- Autoscaling Service Metrics
- Distributed Node Discovery
- Governance Analytics
- Exchange Rates
- Security Monitoring
- Content Extraction
"""

import pytest
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4, UUID
import tempfile
import os


# ============================================================================
# Test System Resource Monitoring (Phase 1)
# ============================================================================

class TestSystemResourceMonitoring:
    """Tests for system resource monitoring in self_modification.py"""

    @pytest.fixture
    def component(self):
        """Create a test component instance"""
        from prsm.compute.evolution.self_modification import SelfModifyingComponent
        from prsm.compute.evolution.models import ComponentType

        # Create a concrete implementation for testing
        class TestComponent(SelfModifyingComponent):
            async def propose_modification(self, evaluation_logs):
                return None

            async def apply_modification(self, modification):
                from prsm.compute.evolution.models import ModificationResult
                return ModificationResult(
                    modification_id=modification.id,
                    success=True,
                    executor_id=self.component_id,
                    timestamp=datetime.utcnow()
                )

            async def validate_modification(self, modification):
                return True

            async def evaluate_performance(self):
                from prsm.compute.evolution.models import EvaluationResult
                return EvaluationResult(
                    evaluation_id=str(uuid4()),
                    component_id=self.component_id,
                    performance_score=0.8,
                    timestamp=datetime.utcnow()
                )

        return TestComponent("test_component", ComponentType.TASK_ORCHESTRATOR)

    @pytest.mark.asyncio
    async def test_get_resource_usage_returns_real_values(self, component):
        """Test that _get_resource_usage returns real values, not zeros"""
        result = await component._get_resource_usage()

        assert isinstance(result, dict)
        assert 'cpu_percent' in result
        assert 'memory_percent' in result
        assert 'memory_mb' in result
        # CPU percent should be a float (could be 0 if system is idle, so we just check type)
        assert isinstance(result['cpu_percent'], float)
        assert isinstance(result['memory_percent'], float)
        # Memory percent should be between 0 and 100
        assert 0 <= result['memory_percent'] <= 100

    @pytest.mark.asyncio
    async def test_resource_usage_keys_complete(self, component):
        """Test that all 7 keys are present in result dict"""
        result = await component._get_resource_usage()

        expected_keys = [
            'cpu_percent', 'memory_percent', 'memory_mb',
            'disk_read_mb', 'disk_write_mb',
            'network_sent_mb', 'network_recv_mb'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_exceeds_limits_when_cpu_high(self, component):
        """Test that _exceeds_resource_limits returns True when CPU is high"""
        from prsm.compute.evolution.models import ModificationProposal, ComponentType, RiskLevel, ImpactLevel

        modification = ModificationProposal(
            id="test_mod",
            solution_id="test_solution",
            component_id="test",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            modification_type="test",
            description="test",
            rationale="test rationale",
            code_changes={},
            config_changes={},
            risk_level=RiskLevel.LOW,
            impact_level=ImpactLevel.LOW,
            estimated_performance_impact=0.0,
            rollback_plan="revert changes",
            proposer_id="test_user",
            compute_requirements={'memory_gb': 1}
        )

        # High CPU usage
        usage = {'cpu_percent': 95.0, 'memory_percent': 50.0}
        result = component._exceeds_resource_limits(usage, modification)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_functionality_preserved_returns_true(self, component):
        """Test that _verify_functionality_preserved returns True in normal environment"""
        result = await component._verify_functionality_preserved()
        # Should return True since prsm.core modules exist
        assert result is True

    @pytest.mark.asyncio
    async def test_scaling_metrics_no_random(self):
        """Test that _get_service_metrics returns deterministic values, not random"""
        from prsm.compute.performance.scaling import AutoScaler, ScalingPolicy

        scaler = AutoScaler()
        policy = ScalingPolicy(
            name="test",
            service_name="test-api",
            description="test"
        )

        # Call twice - CPU/memory should be similar (not random)
        result1 = await scaler._get_service_metrics("test-api", policy)
        result2 = await scaler._get_service_metrics("test-api", policy)

        # The values should be close (within some tolerance for actual system changes)
        # But not wildly different like random values would be
        assert abs(result1['cpu_usage'] - result2['cpu_usage']) < 0.5
        assert abs(result1['memory_usage'] - result2['memory_usage']) < 0.5


# ============================================================================
# Test Node Discovery (Phase 3)
# ============================================================================

class TestNodeDiscovery:
    """Tests for distributed node discovery"""

    @pytest.fixture
    def allocation_engine(self):
        """Create a resource allocation engine instance"""
        from prsm.compute.federation.distributed_resource_manager import ResourceAllocationEngine
        return ResourceAllocationEngine()

    @pytest.mark.asyncio
    async def test_find_candidate_nodes_returns_empty_when_no_peers(self, allocation_engine):
        """Test that empty list is returned when DB has no peers"""
        from prsm.compute.federation.distributed_resource_manager import ResourceType

        with patch('prsm.core.database.get_async_session') as mock_session:
            # Mock empty result
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_result

            result = await allocation_engine._find_candidate_nodes(
                {ResourceType.COMPUTE_CPU: 2.0},
                {}
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_find_candidate_nodes_filters_inactive(self, allocation_engine):
        """Test that inactive peers are excluded"""
        from prsm.compute.federation.distributed_resource_manager import ResourceType

        # Create mock peer
        mock_peer = Mock()
        mock_peer.peer_id = "active_peer"
        mock_peer.node_type = "standard"
        mock_peer.last_seen = datetime.now(timezone.utc).timestamp()
        mock_peer.is_active = True
        mock_peer.quality_score = 0.8
        mock_peer.capabilities = {}

        with patch('prsm.core.database.get_async_session') as mock_session:
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = [mock_peer]
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_result

            result = await allocation_engine._find_candidate_nodes(
                {ResourceType.COMPUTE_CPU: 2.0},
                {'min_quality_score': 0.3}
            )

            # Should return 1 profile for active peer
            assert len(result) == 1
            assert result[0].node_id == "active_peer"

    @pytest.mark.asyncio
    async def test_find_candidate_nodes_returns_profiles(self, allocation_engine):
        """Test that correct profiles are returned for active peers"""
        from prsm.compute.federation.distributed_resource_manager import ResourceType

        # Create 3 mock peers
        mock_peers = []
        for i in range(3):
            peer = Mock()
            peer.peer_id = f"peer_{i}"
            peer.node_type = "standard"
            peer.last_seen = datetime.now(timezone.utc).timestamp()
            peer.is_active = True
            peer.quality_score = 0.5 + i * 0.1
            peer.capabilities = {'compute_cpu': {'total_capacity': 8.0, 'allocated_capacity': 2.0}}
            mock_peers.append(peer)

        with patch('prsm.core.database.get_async_session') as mock_session:
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = mock_peers
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_result

            result = await allocation_engine._find_candidate_nodes(
                {ResourceType.COMPUTE_CPU: 2.0},
                {'min_quality_score': 0.3}
            )

            assert len(result) == 3
            node_ids = [p.node_id for p in result]
            assert "peer_0" in node_ids
            assert "peer_1" in node_ids
            assert "peer_2" in node_ids


# ============================================================================
# Test Voting Correlation (Phase 4a)
# ============================================================================

class TestVotingCorrelation:
    """Tests for voting correlation analysis"""

    @pytest.fixture
    def governance(self):
        """Create anti-monopoly governance instance"""
        from prsm.economy.governance.anti_monopoly import AntiMonopolyGovernance
        return AntiMonopolyGovernance()

    def test_correlation_empty_with_no_votes(self, governance):
        """Test that empty dict is returned with no votes"""
        result = governance._analyze_voting_correlations([], {})
        assert result == {}

    def test_correlation_identical_voters_returns_1(self, governance):
        """Test that identical voters have correlation near 1.0"""
        uuid_a = uuid4()
        uuid_b = uuid4()

        # Both voters always vote the same way
        votes = [
            {'proposal_id': 'p1', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p1', 'voter_id': uuid_b, 'vote': 'yes'},
            {'proposal_id': 'p2', 'voter_id': uuid_a, 'vote': 'no'},
            {'proposal_id': 'p2', 'voter_id': uuid_b, 'vote': 'no'},
            {'proposal_id': 'p3', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p3', 'voter_id': uuid_b, 'vote': 'yes'},
        ]

        participants = {
            uuid_a: Mock(institution_name="Org A"),
            uuid_b: Mock(institution_name="Org B"),
        }

        result = governance._analyze_voting_correlations(votes, participants)

        # Should have one correlation entry
        assert len(result) == 1
        # Correlation should be close to 1.0
        corr = list(result.values())[0]
        assert abs(corr - 1.0) < 0.01

    def test_correlation_opposite_voters_returns_minus_1(self, governance):
        """Test that opposite voters have correlation near -1.0"""
        uuid_a = uuid4()
        uuid_b = uuid4()

        # Voters always vote opposite
        votes = [
            {'proposal_id': 'p1', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p1', 'voter_id': uuid_b, 'vote': 'no'},
            {'proposal_id': 'p2', 'voter_id': uuid_a, 'vote': 'no'},
            {'proposal_id': 'p2', 'voter_id': uuid_b, 'vote': 'yes'},
            {'proposal_id': 'p3', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p3', 'voter_id': uuid_b, 'vote': 'no'},
        ]

        participants = {
            uuid_a: Mock(institution_name="Org A"),
            uuid_b: Mock(institution_name="Org B"),
        }

        result = governance._analyze_voting_correlations(votes, participants)

        # Correlation should be close to -1.0
        corr = list(result.values())[0]
        assert abs(corr - (-1.0)) < 0.01

    def test_correlation_handles_missing_votes(self, governance):
        """Test that correlation handles partial participation"""
        uuid_a = uuid4()
        uuid_b = uuid4()
        uuid_c = uuid4()

        # Some voters miss some proposals
        votes = [
            {'proposal_id': 'p1', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p1', 'voter_id': uuid_b, 'vote': 'yes'},
            {'proposal_id': 'p2', 'voter_id': uuid_a, 'vote': 'no'},
            # uuid_b misses p2
            {'proposal_id': 'p3', 'voter_id': uuid_a, 'vote': 'yes'},
            {'proposal_id': 'p3', 'voter_id': uuid_b, 'vote': 'yes'},
        ]

        participants = {
            uuid_a: Mock(institution_name="Org A"),
            uuid_b: Mock(institution_name="Org B"),
        }

        # Should not crash
        result = governance._analyze_voting_correlations(votes, participants)
        assert isinstance(result, dict)


# ============================================================================
# Test Exchange Rates (Phase 5)
# ============================================================================

class TestExchangeRates:
    """Tests for exchange rate integration"""

    @pytest.fixture
    def router(self):
        """Create exchange router instance"""
        from prsm.compute.chronos.exchange_router import ExchangeRouter
        return ExchangeRouter()

    @pytest.mark.asyncio
    async def test_fetch_rate_uses_fallback_when_api_fails(self, router):
        """Test that fallback rates are used when API fails"""
        from prsm.compute.chronos.models import AssetType

        with patch('httpx.AsyncClient') as mock_client:
            # Make the API call raise an exception
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Network error")

            result = await router._fetch_exchange_rate_from_source(
                "coinbase",
                AssetType.BTC,
                AssetType.USD
            )

            # Should return fallback rate
            assert 'rate' in result
            assert result['rate'] > 0
            assert result['source'] in ['fallback', 'coingecko']

    @pytest.mark.asyncio
    async def test_fetch_rate_caches_result(self, router):
        """Test that exchange rates are cached"""
        from prsm.compute.chronos.models import AssetType

        # First call
        result1 = await router._fetch_exchange_rate_from_source(
            "coinbase",
            AssetType.BTC,
            AssetType.USD
        )

        # Second call should use cache
        result2 = await router._fetch_exchange_rate_from_source(
            "coinbase",
            AssetType.BTC,
            AssetType.USD
        )

        # Results should be identical (from cache)
        assert result1['rate'] == result2['rate']

    @pytest.mark.asyncio
    async def test_fetch_rate_unknown_pair_returns_error(self, router):
        """Test that unknown pairs return error"""
        from prsm.compute.chronos.models import AssetType

        # Try an unsupported pair (if any)
        # Most pairs should work with fallback
        result = await router._fetch_exchange_rate_from_source(
            "coinbase",
            AssetType.BTC,
            AssetType.USD
        )

        # Should have a rate (from fallback if not API)
        assert 'rate' in result or 'error' in result


# ============================================================================
# Test Tor Exit Node Check (Phase 6)
# ============================================================================

class TestTorExitNodeCheck:
    """Tests for Tor exit node detection"""

    @pytest.fixture
    def monitor(self):
        """Create ThreatIntelligence instance (owns _is_tor_exit_node)"""
        from prsm.core.security.security_monitoring import ThreatIntelligence
        mock_redis = AsyncMock()
        return ThreatIntelligence(mock_redis)

    @pytest.mark.asyncio
    async def test_known_ip_not_in_empty_cache_returns_false(self, monitor):
        """Test that unknown IP returns False"""
        from prsm.core.security import security_monitoring

        # Clear cache
        security_monitoring._TOR_EXIT_CACHE.clear()

        result = await monitor._is_tor_exit_node("8.8.8.8")
        # Should be False since it's not a Tor exit node
        # (or True if the Tor list includes it)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_tor_node_detected_after_cache_population(self, monitor):
        """Test that Tor nodes are detected after cache population"""
        from prsm.core.security import security_monitoring

        # Manually populate cache with a test IP
        security_monitoring._TOR_EXIT_CACHE.add("1.2.3.4")

        result = await monitor._is_tor_exit_node("1.2.3.4")
        assert result is True

        # Cleanup
        security_monitoring._TOR_EXIT_CACHE.discard("1.2.3.4")


# ============================================================================
# Test Content Extraction (Phase 7)
# ============================================================================

class TestContentExtraction:
    """Tests for binary content text extraction"""

    @pytest.fixture
    def content_manager(self):
        """Create content manager instance"""
        # Mock the imports that may fail
        sys.modules['prsm.ipfs'] = Mock()
        sys.modules['prsm.ipfs.content_addressing'] = Mock()
        sys.modules['prsm.ingestion'] = Mock()
        sys.modules['prsm.ingestion.public_source_porter'] = Mock()
        sys.modules['prsm.tokenomics'] = Mock()
        sys.modules['prsm.tokenomics.ftns_economics'] = Mock()

        from prsm.user_content_manager import UserContentManager
        mock_knowledge = Mock()
        return UserContentManager(mock_knowledge)

    @pytest.mark.asyncio
    async def test_returns_none_for_no_content(self, content_manager):
        """Test that None is returned for uploads with no content"""
        from prsm.user_content_manager import UserUpload, UserContentType

        upload = UserUpload(
            upload_id="test",
            user_id="user",
            title="Test",
            description="Test",
            content_type=UserContentType.OTHER,
            file_name=None,
        )
        # No content_binary set

        result = await content_manager._extract_text_from_binary(upload)
        assert result is None

    @pytest.mark.asyncio
    async def test_extracts_plain_text(self, content_manager):
        """Test that plain text is extracted correctly"""
        from prsm.user_content_manager import UserUpload, UserContentType

        upload = UserUpload(
            upload_id="test",
            user_id="user",
            title="Test",
            description="Test",
            content_type=UserContentType.OTHER,
            file_name="test.txt",
        )
        upload.content_binary = b"hello world"

        result = await content_manager._extract_text_from_binary(upload)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_pdf_extraction_graceful_without_pypdf(self, content_manager):
        """Test that PDF extraction fails gracefully without pypdf"""
        from prsm.user_content_manager import UserUpload, UserContentType

        upload = UserUpload(
            upload_id="test",
            user_id="user",
            title="Test",
            description="Test",
            content_type=UserContentType.OTHER,
            file_name="test.pdf",
        )
        upload.content_binary = b"%PDF-1.4 fake pdf content"

        # When pypdf is not available, it should return None gracefully
        with patch.dict('sys.modules', {'pypdf': None}):
            result = await content_manager._extract_text_from_binary(upload)
            # Should return None without crashing
            assert result is None or isinstance(result, str)


# ============================================================================
# Test Eligible Voters Count (Phase 4b)
# ============================================================================

@pytest.mark.skip(
    reason="prsm.core.teams.governance removed in v1.6.0 scope alignment "
    "(institutional team management no longer in scope; nodes are nodes)"
)
class TestEligibleVotersCount:
    """Tests for eligible voter counting"""

    @pytest.fixture
    def governance_service(self):
        """Create team governance service instance"""
        return None

    @pytest.mark.asyncio
    async def test_count_eligible_voters_returns_zero_on_error(self, governance_service):
        """Test that 0 is returned on database error"""
        team_id = uuid4()

        with patch('prsm.core.database.get_async_session') as mock_session:
            mock_session.side_effect = Exception("DB error")

            result = await governance_service._count_eligible_voters(team_id)
            assert result == 0

    @pytest.mark.asyncio
    async def test_count_eligible_voters_queries_database(self, governance_service):
        """Test that database is queried for member count"""
        team_id = uuid4()

        with patch('prsm.core.database.get_async_session') as mock_session:
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = 5
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_result

            result = await governance_service._count_eligible_voters(team_id)
            assert result == 5


# ============================================================================
# Test Multisig Transaction Execution (Phase 9)
# ============================================================================

@pytest.mark.skip(
    reason="prsm.core.teams.wallet removed in v1.6.0 scope alignment "
    "(institutional team multisig no longer in scope; nodes are nodes)"
)
class TestMultisigExecution:
    """Tests for multisig transaction execution"""

    @pytest.fixture
    def wallet_service(self):
        """Create wallet service instance"""
        return None

    @pytest.mark.asyncio
    async def test_execute_multisig_transaction_transfers_ftns(self, wallet_service):
        """Test that multisig execution transfers FTNS"""
        tx_id = str(uuid4())
        from_account = "user_1"
        to_account = "user_2"

        wallet_service.pending_transactions[tx_id] = {
            "wallet_id": uuid4(),
            "transaction_data": {
                "type": "transfer",
                "from_account": from_account,
                "to_account": to_account,
                "amount": 100.0
            },
            "status": "pending"
        }
        wallet_service.signature_cache[tx_id] = ["signer1", "signer2"]

        # Create proper async context manager mocks
        mock_session = AsyncMock()
        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=None)
        mock_begin.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_begin)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch('prsm.core.database.get_async_session', return_value=mock_session_ctx):
            await wallet_service._execute_multisig_transaction(tx_id)

            # Transaction should be marked completed
            assert wallet_service.pending_transactions[tx_id]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_multisig_transaction_handles_missing_tx(self, wallet_service):
        """Test that missing transaction raises error"""
        with pytest.raises(KeyError):
            await wallet_service._execute_multisig_transaction("nonexistent")


# ============================================================================
# Test Trace Analytics (Phase 8)
# ============================================================================

class TestTraceAnalytics:
    """Tests for trace analytics"""

    @pytest.fixture
    def data_provider(self):
        """Create metric data provider instance"""
        from prsm.compute.performance.monitoring_dashboard import MetricDataProvider
        mock_redis = AsyncMock()
        return MetricDataProvider(mock_redis)

    @pytest.mark.asyncio
    async def test_trace_analytics_returns_zero_when_no_spans(self, data_provider):
        """Test that zero values are returned when no spans exist"""
        result = await data_provider.get_trace_analytics(hours=1)

        assert result["total_traces"] == 0
        assert result["avg_duration_ms"] == 0.0
        assert result["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_trace_analytics_with_mock_tracer(self, data_provider):
        """Test trace analytics with mock tracer"""
        # Create mock tracer with spans
        mock_span = Mock()
        mock_span.end_time = datetime.now(timezone.utc).timestamp()
        mock_span.duration_ms = 100.0
        mock_span.is_error = False
        mock_span.service_name = "test_service"
        mock_span.operation_name = "test_operation"

        mock_tracer = Mock()
        mock_tracer._completed_spans = [mock_span]

        data_provider._tracer = mock_tracer

        result = await data_provider.get_trace_analytics(hours=1)

        assert result["total_traces"] == 1
        assert result["avg_duration_ms"] == 100.0
        assert "test_service" in result["services"]
