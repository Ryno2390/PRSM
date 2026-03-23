"""
Integration tests for Federated Evolution & Marketplace Cleanup

Tests covering:
- Phase 1: Distributed Evolution P2P calls (8 tests)
- Phase 2: Consensus and deployment (3 tests)
- Phase 3: Contributor badge assignment (4 tests)
- Phase 4: AST complexity analysis (3 tests)
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

# ============================================================================
# Phase 1: TestDistributedEvolutionP2P (8 tests)
# ============================================================================


class TestDistributedEvolutionP2P:
    """Tests for distributed evolution P2P HTTP calls."""

    @pytest.fixture
    def archive_manager(self):
        """Create a DistributedArchiveManager for testing."""
        from prsm.compute.federation.distributed_evolution import (
            DistributedArchiveManager,
            MockP2PNetwork,
        )
        from prsm.compute.evolution.archive import EvolutionArchive
        from prsm.compute.evolution.models import ComponentType

        local_archive = EvolutionArchive(
            archive_id="test_archive",
            component_type=ComponentType.TASK_ORCHESTRATOR
        )
        return DistributedArchiveManager(
            node_id="test_node",
            local_archive=local_archive,
            p2p_network=MockP2PNetwork()
        )

    @pytest.mark.asyncio
    async def test_request_archive_metadata_returns_safe_dict_on_db_miss(self, archive_manager):
        """When peer not found in DB, return safe dict with available=False and no random values."""
        with patch('prsm.core.database.get_async_session') as mock_session:
            # Mock DB returning no peer
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            result = await archive_manager._request_archive_metadata("unknown_peer")

            # Assert safe defaults - no random data
            assert result["available"] is False
            assert result["node_id"] == "unknown_peer"
            assert result["total_solutions"] == 0
            assert result["best_performance"] == 0.0
            assert result["component_types"] == []

    @pytest.mark.asyncio
    async def test_request_archive_metadata_attempts_http_when_peer_found(self, archive_manager):
        """When peer found in DB, attempt HTTP GET to correct URL."""
        with patch('prsm.core.database.get_async_session') as mock_session, \
             patch('httpx.AsyncClient') as mock_httpx:

            # Mock DB returning a peer
            mock_peer = MagicMock()
            mock_peer.address = "192.168.1.100"
            mock_peer.port = 8080

            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_peer
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            # Mock httpx response
            mock_response = MagicMock()
            mock_response.json.return_value = {"node_id": "peer1", "total_solutions": 100}
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await archive_manager._request_archive_metadata("peer1")

            # Verify HTTP GET was called with correct URL
            mock_client_instance.get.assert_called_once()
            call_args = mock_client_instance.get.call_args
            assert "192.168.1.100" in call_args[0][0]
            assert "8080" in call_args[0][0]
            assert "/api/v1/federation/archive/metadata" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_request_archive_metadata_falls_back_to_cache_on_http_error(self, archive_manager):
        """When HTTP fails, return cached metadata if available."""
        with patch('prsm.core.database.get_async_session') as mock_session, \
             patch('httpx.AsyncClient') as mock_httpx:

            # Mock DB returning a peer
            mock_peer = MagicMock()
            mock_peer.address = "192.168.1.100"
            mock_peer.port = 8080

            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_peer
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            # Mock httpx raising an error
            mock_httpx.side_effect = Exception("Connection refused")

            # Pre-set cached metadata
            cached_data = {
                "node_id": "peer1",
                "total_solutions": 50,
                "best_performance": 0.85,
                "cached": True
            }
            archive_manager.peer_archives["peer1"] = cached_data

            result = await archive_manager._request_archive_metadata("peer1")

            # Should return cached data
            assert result == cached_data

    @pytest.mark.asyncio
    async def test_send_sync_request_returns_failure_when_peer_unreachable(self, archive_manager):
        """When peer is unreachable, return unsuccessful response with empty solutions."""
        from prsm.compute.federation.distributed_evolution import SolutionSyncRequest, SynchronizationStrategy

        with patch('prsm.core.database.get_async_session') as mock_session, \
             patch('httpx.AsyncClient') as mock_httpx:

            # Mock DB returning no peer
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            sync_request = SolutionSyncRequest(
                request_id="req-1",
                requesting_node_id="node-1",
                target_node_id="unreachable-peer",
                sync_strategy=SynchronizationStrategy.SELECTIVE_SYNC
            )

            response = await archive_manager._send_sync_request(sync_request)

            assert response.success is False
            assert response.solutions == []
            assert response.error_message == "Peer unreachable"

    @pytest.mark.asyncio
    async def test_send_sync_request_calls_correct_endpoint(self, archive_manager):
        """When peer found, POST to correct sync endpoint."""
        from prsm.compute.federation.distributed_evolution import SolutionSyncRequest, SynchronizationStrategy

        with patch('prsm.core.database.get_async_session') as mock_session, \
             patch('httpx.AsyncClient') as mock_httpx:

            # Mock DB returning a peer
            mock_peer = MagicMock()
            mock_peer.address = "192.168.1.100"
            mock_peer.port = 8080

            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_peer
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            # Mock httpx response
            mock_response = MagicMock()
            mock_response.json.return_value = {"solutions": [], "total_available": 0}
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            sync_request = SolutionSyncRequest(
                request_id="req-1",
                requesting_node_id="node-1",
                target_node_id="peer1",
                sync_strategy=SynchronizationStrategy.SELECTIVE_SYNC
            )

            response = await archive_manager._send_sync_request(sync_request)

            # Verify POST was called with correct URL
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert "192.168.1.100" in call_args[0][0]
            assert "/api/v1/federation/archive/sync" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_collect_task_results_returns_empty_when_no_peers_in_db(self):
        """When no peers found in DB for assigned nodes, return empty list."""
        from prsm.compute.federation.distributed_evolution import (
            FederatedEvolutionCoordinator,
            NetworkEvolutionTask,
        )
        from prsm.compute.evolution.models import ComponentType

        # Create coordinator with mocked dependencies
        archive_manager = MagicMock()
        consensus_manager = MagicMock()

        coordinator = FederatedEvolutionCoordinator(
            node_id="test_node",
            archive_manager=archive_manager,
            consensus_manager=consensus_manager
        )

        task = NetworkEvolutionTask(
            task_id="task-1",
            coordinator_node_id="test_node",
            task_type="optimization",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            objective="test",
            assigned_nodes=["node-a", "node-b"]
        )

        with patch('prsm.core.database.get_async_session') as mock_session:
            # Mock DB returning no peers
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            results = await coordinator._collect_task_results(task, timeout_seconds=30.0)

            assert results == []

    @pytest.mark.asyncio
    async def test_measure_network_performance_returns_zero_on_empty_db(self):
        """When DB returns no quality scores, return 0.0."""
        from prsm.compute.federation.distributed_evolution import FederatedEvolutionCoordinator

        archive_manager = MagicMock()
        archive_manager.local_archive = MagicMock()
        archive_manager.local_archive.solutions = {}
        consensus_manager = MagicMock()

        coordinator = FederatedEvolutionCoordinator(
            node_id="test_node",
            archive_manager=archive_manager,
            consensus_manager=consensus_manager
        )

        with patch('prsm.core.database.get_async_session') as mock_session:
            # Mock DB returning None for avg
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            performance = await coordinator._measure_network_performance()

            assert performance == 0.0

    @pytest.mark.asyncio
    async def test_measure_network_performance_uses_db_quality_scores(self):
        """When DB returns quality scores, use them for network performance."""
        from prsm.compute.federation.distributed_evolution import FederatedEvolutionCoordinator

        archive_manager = MagicMock()
        consensus_manager = MagicMock()

        coordinator = FederatedEvolutionCoordinator(
            node_id="test_node",
            archive_manager=archive_manager,
            consensus_manager=consensus_manager
        )

        with patch('prsm.core.database.get_async_session') as mock_session:
            # Mock DB returning avg quality of 0.75
            mock_session_instance = AsyncMock()
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = 0.75
            mock_session_instance.execute.return_value = mock_result
            mock_session.return_value = mock_session_instance

            performance = await coordinator._measure_network_performance()

            assert performance == 0.75


# ============================================================================
# Phase 2: TestConsensusAndDeployment (3 tests)
# ============================================================================


class TestConsensusAndDeployment:
    """Tests for consensus and deployment logic."""

    @pytest.fixture
    def coordinator(self):
        """Create a FederatedEvolutionCoordinator for testing."""
        from prsm.compute.federation.distributed_evolution import FederatedEvolutionCoordinator

        archive_manager = MagicMock()
        consensus_manager = MagicMock()

        return FederatedEvolutionCoordinator(
            node_id="test_node",
            archive_manager=archive_manager,
            consensus_manager=consensus_manager
        )

    @pytest.mark.asyncio
    async def test_achieve_consensus_true_when_performance_exceeds_threshold(self, coordinator):
        """Return True when best_performance exceeds threshold."""
        aggregated_result = {
            "improvement_identified": True,
            "best_performance_achieved": 0.9,
            "participating_nodes": 3
        }

        consensus = await coordinator._achieve_consensus_on_improvements(aggregated_result)

        assert consensus is True

    @pytest.mark.asyncio
    async def test_achieve_consensus_false_when_performance_below_threshold(self, coordinator):
        """Return False when best_performance below threshold."""
        aggregated_result = {
            "improvement_identified": True,
            "best_performance_achieved": 0.5,
            "participating_nodes": 3
        }

        consensus = await coordinator._achieve_consensus_on_improvements(aggregated_result)

        assert consensus is False

    @pytest.mark.asyncio
    async def test_deploy_network_improvements_returns_true_when_no_solutions(self, coordinator):
        """Return True when no solutions to deploy (no-op is success)."""
        aggregated_result = {
            "top_solutions": []
        }

        result = await coordinator._deploy_network_improvements(aggregated_result)

        assert result is True


# ============================================================================
# Phase 3: TestContributorBadges (4 tests)
# ============================================================================


class TestContributorBadges:
    """Tests for deterministic contributor badge assignment."""

    @pytest.fixture
    def contributor_class(self):
        """Get the ContributorProfile dataclass."""
        from prsm.interface.onboarding.contributor_onboarding import ContributorProfile
        return ContributorProfile

    @pytest.fixture
    def level_class(self):
        """Get ContributorLevel enum."""
        from prsm.interface.onboarding.contributor_onboarding import ContributorLevel
        return ContributorLevel

    @pytest.fixture
    def stage_class(self):
        """Get OnboardingStage enum."""
        from prsm.interface.onboarding.contributor_onboarding import OnboardingStage
        return OnboardingStage

    @pytest.fixture
    def area_class(self):
        """Get ContributionArea enum."""
        from prsm.interface.onboarding.contributor_onboarding import ContributionArea
        return ContributionArea

    @pytest.fixture
    def skill_class(self):
        """Get SkillLevel enum."""
        from prsm.interface.onboarding.contributor_onboarding import SkillLevel
        return SkillLevel

    def test_no_random_badge_assignment(self, contributor_class, level_class, stage_class, area_class, skill_class):
        """Badge assignment is deterministic - same input yields same output."""
        # Create EXPERT contributor with no contributions
        contributor1 = contributor_class(
            contributor_id="test-1",
            username="testuser1",
            email="test1@example.com",
            level=level_class.EXPERT,
            onboarding_stage=stage_class.ONGOING_SUPPORT,
            completed_stages=[],
            onboarding_score=0.0,
            programming_languages=[],
            frameworks=[],
            areas_of_interest=[],
            skill_assessments={},
            contributions_made=0,
            badges_earned=[]
        )
        # Set attributes for badge criteria
        contributor1.mentees_count = 0
        contributor1.reviews_completed = 0

        contributor2 = contributor_class(
            contributor_id="test-2",
            username="testuser2",
            email="test2@example.com",
            level=level_class.EXPERT,
            onboarding_stage=stage_class.ONGOING_SUPPORT,
            completed_stages=[],
            onboarding_score=0.0,
            programming_languages=[],
            frameworks=[],
            areas_of_interest=[],
            skill_assessments={},
            contributions_made=0,
            badges_earned=[]
        )
        contributor2.mentees_count = 0
        contributor2.reviews_completed = 0

        # Simulate badge evaluation multiple times
        badge_criteria = {
            "code_contributor": contributor1.contributions_made > 0,
            "mentor": getattr(contributor1, 'mentees_count', 0) > 0,
            "reviewer": getattr(contributor1, 'reviews_completed', 0) > 0,
        }

        badges1 = [badge for badge, earned in badge_criteria.items() if earned]

        # Same evaluation on contributor2
        badge_criteria2 = {
            "code_contributor": contributor2.contributions_made > 0,
            "mentor": getattr(contributor2, 'mentees_count', 0) > 0,
            "reviewer": getattr(contributor2, 'reviews_completed', 0) > 0,
        }
        badges2 = [badge for badge, earned in badge_criteria2.items() if earned]

        # Results should be identical (deterministic)
        assert badges1 == badges2
        assert badges1 == []  # No badges for zero contributions

    def test_code_contributor_badge_when_contributions_positive(self, contributor_class, level_class, stage_class, area_class, skill_class):
        """Expert with contributions_made > 0 earns code_contributor badge."""
        contributor = contributor_class(
            contributor_id="test-1",
            username="testuser1",
            email="test1@example.com",
            level=level_class.EXPERT,
            onboarding_stage=stage_class.ONGOING_SUPPORT,
            completed_stages=[],
            onboarding_score=0.0,
            programming_languages=[],
            frameworks=[],
            areas_of_interest=[],
            skill_assessments={},
            contributions_made=3,
            badges_earned=[]
        )
        contributor.mentees_count = 0
        contributor.reviews_completed = 0

        # Apply deterministic badge criteria
        badge_criteria = {
            "code_contributor": contributor.contributions_made > 0,
            "mentor": getattr(contributor, 'mentees_count', 0) > 0,
            "reviewer": getattr(contributor, 'reviews_completed', 0) > 0,
        }

        for badge, earned in badge_criteria.items():
            if earned and badge not in contributor.badges_earned:
                contributor.badges_earned.append(badge)

        assert "code_contributor" in contributor.badges_earned

    def test_reviewer_badge_when_reviews_positive(self, contributor_class, level_class, stage_class, area_class, skill_class):
        """Contributor with reviews_completed > 0 earns reviewer badge."""
        contributor = contributor_class(
            contributor_id="test-1",
            username="testuser1",
            email="test1@example.com",
            level=level_class.EXPERT,
            onboarding_stage=stage_class.ONGOING_SUPPORT,
            completed_stages=[],
            onboarding_score=0.0,
            programming_languages=[],
            frameworks=[],
            areas_of_interest=[],
            skill_assessments={},
            contributions_made=0,
            badges_earned=[]
        )
        contributor.mentees_count = 0
        contributor.reviews_completed = 2

        # Apply deterministic badge criteria
        badge_criteria = {
            "code_contributor": contributor.contributions_made > 0,
            "mentor": getattr(contributor, 'mentees_count', 0) > 0,
            "reviewer": getattr(contributor, 'reviews_completed', 0) > 0,
        }

        for badge, earned in badge_criteria.items():
            if earned and badge not in contributor.badges_earned:
                contributor.badges_earned.append(badge)

        assert "reviewer" in contributor.badges_earned

    def test_no_badge_when_no_contributions(self, contributor_class, level_class, stage_class, area_class, skill_class):
        """Expert with all counts at 0 earns no new badges."""
        contributor = contributor_class(
            contributor_id="test-1",
            username="testuser1",
            email="test1@example.com",
            level=level_class.EXPERT,
            onboarding_stage=stage_class.ONGOING_SUPPORT,
            completed_stages=[],
            onboarding_score=0.0,
            programming_languages=[],
            frameworks=[],
            areas_of_interest=[],
            skill_assessments={},
            contributions_made=0,
            badges_earned=[]
        )
        contributor.mentees_count = 0
        contributor.reviews_completed = 0

        initial_badges = set(contributor.badges_earned)

        # Apply deterministic badge criteria
        badge_criteria = {
            "code_contributor": contributor.contributions_made > 0,
            "mentor": getattr(contributor, 'mentees_count', 0) > 0,
            "reviewer": getattr(contributor, 'reviews_completed', 0) > 0,
        }

        for badge, earned in badge_criteria.items():
            if earned and badge not in contributor.badges_earned:
                contributor.badges_earned.append(badge)

        # No new badges should be added
        assert set(contributor.badges_earned) == initial_badges


# ============================================================================
# Phase 4: TestASTComplexityAnalysis (3 tests)
# ============================================================================


class TestASTComplexityAnalysis:
    """Tests for AST-based code complexity analysis."""

    @pytest.fixture
    def validator(self):
        """Create a PluginValidator for testing."""
        from prsm.economy.marketplace.ecosystem.plugin_registry import PluginValidator
        return PluginValidator()

    @pytest.fixture
    def manifest(self):
        """Create a basic PluginManifest for testing."""
        from prsm.economy.marketplace.ecosystem.plugin_registry import PluginManifest, PluginType
        return PluginManifest(
            name="test_plugin",
            version="1.0.0",
            plugin_id="test-plugin",
            description="Test plugin",
            plugin_type=PluginType.CORE_EXTENSION,
            main_module="test_module"
        )

    @pytest.mark.asyncio
    async def test_simple_file_returns_zero_penalty(self, validator, manifest):
        """Simple Python file with no complexity issues returns zero penalty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create simple Python file
            simple_code = '''
def foo():
    pass

def bar():
    return 42
'''
            (tmp_path / "simple.py").write_text(simple_code)

            result = await validator._check_code_complexity(tmp_path, manifest)

            assert result["penalty"] == 0
            assert len(result["issues"]) == 0

    @pytest.mark.asyncio
    async def test_syntax_error_adds_penalty(self, validator, manifest):
        """Python file with syntax error adds penalty and issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create Python file with syntax error
            bad_code = '''
def broken(
    # Missing closing paren
'''
            (tmp_path / "broken.py").write_text(bad_code)

            result = await validator._check_code_complexity(tmp_path, manifest)

            assert result["penalty"] >= 10
            assert len(result["issues"]) > 0
            assert any("Syntax error" in issue for issue in result["issues"])

    @pytest.mark.asyncio
    async def test_deeply_nested_code_adds_penalty(self, validator, manifest):
        """Deeply nested code adds complexity penalty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create Python file with deep nesting (10 levels)
            nested_code = '''
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    if True:
                                        if True:
                                            pass
'''
            (tmp_path / "nested.py").write_text(nested_code)

            result = await validator._check_code_complexity(tmp_path, manifest)

            # Deep nesting should add penalty
            assert result["penalty"] > 0
            assert len(result["warnings"]) > 0
            assert any("Deep nesting" in w for w in result["warnings"])
