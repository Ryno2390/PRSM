"""
Integration tests for Economics Model & Config Fixes

Tests for:
1. Database config bug fix (database_url property)
2. Agent-based model reproducibility with seeded RNG (requires Mesa <3.0)
3. AgentProfile behavioral parameters

NOTE: Tests requiring PRSMEconomicModel need Mesa <3.0 due to API changes.
Mesa 3.5.1 removed RandomActivation in favor of Schedule, breaking compatibility.
"""

import pytest
import uuid as uuid_lib
import numpy as np
from decimal import Decimal
from typing import Tuple

# Check for Mesa availability and API compatibility
try:
    from mesa.time import RandomActivation
    MESA_COMPATIBLE = True
except ImportError:
    MESA_COMPATIBLE = False


class TestDatabaseConfig:
    """Tests for database_url config bug fix"""

    def test_database_url_with_sqlite_string_type(self):
        """Test database_url works when type is stored as string (use_enum_values=True)"""
        from prsm.core.config.schemas import DatabaseConfig, PRSMConfig

        # Create config with default settings (type will be string due to use_enum_values=True)
        config = PRSMConfig()

        # This should not raise AttributeError
        db_url = config.database_url

        assert db_url is not None
        assert db_url.startswith("sqlite://")

    def test_database_url_with_enum_type(self):
        """Test database_url works when type is explicitly passed as enum"""
        from prsm.core.config.schemas import DatabaseConfig, DatabaseTypeEnum, PRSMConfig

        # Create config with explicit enum type
        config = PRSMConfig()
        config.database.type = DatabaseTypeEnum.SQLITE

        # This should not raise AttributeError
        db_url = config.database_url

        assert db_url is not None
        assert "sqlite" in db_url

    def test_database_url_with_postgresql(self):
        """Test database_url with postgresql configuration"""
        from prsm.core.config.schemas import DatabaseConfig, DatabaseTypeEnum, PRSMConfig

        config = PRSMConfig()
        config.database.type = "postgresql"
        config.database.host = "localhost"
        config.database.port = 5432
        config.database.database = "prsm"

        db_url = config.database_url

        assert db_url == "postgresql://localhost:5432/prsm"

    def test_database_url_handles_enum_value_correctly(self):
        """Test that database_url correctly extracts value from enum"""
        from prsm.core.config.schemas import PRSMConfig

        config = PRSMConfig()

        # The database.type should be accessible and produce a valid URL
        db_url = config.database_url

        # Should contain a valid protocol (not the enum name like 'DatabaseTypeEnum.SQLITE')
        assert "DatabaseTypeEnum" not in db_url
        assert "://" in db_url


@pytest.mark.skipif(not MESA_COMPATIBLE, reason="Mesa API incompatible - requires Mesa <3.0")
class TestAgentBasedModelReproducibility:
    """Tests for agent-based model reproducibility with seeded RNG"""

    @pytest.fixture
    def model_with_seed_42(self):
        """Create a model instance with seed 42"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel
        return PRSMEconomicModel(num_agents=10, seed=42)

    @pytest.fixture
    def model_with_seed_43(self):
        """Create a model instance with seed 43"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel
        return PRSMEconomicModel(num_agents=10, seed=43)

    def test_same_seed_produces_identical_profiles(self):
        """Test that same seed produces identical agent profiles"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel, StakeholderType

        model1 = PRSMEconomicModel(num_agents=10, seed=42)
        model2 = PRSMEconomicModel(num_agents=10, seed=42)

        profile1 = model1._generate_agent_profile(StakeholderType.CONTENT_CREATOR)
        profile2 = model2._generate_agent_profile(StakeholderType.CONTENT_CREATOR)

        assert profile1.initial_balance == profile2.initial_balance
        assert profile1.risk_tolerance == profile2.risk_tolerance
        assert profile1.activity_frequency == profile2.activity_frequency
        assert profile1.quality_threshold == profile2.quality_threshold
        assert profile1.network_connectivity == profile2.network_connectivity

    def test_different_seeds_produce_different_profiles(self):
        """Test that different seeds produce different agent profiles"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel, StakeholderType

        model1 = PRSMEconomicModel(num_agents=10, seed=42)
        model2 = PRSMEconomicModel(num_agents=10, seed=43)

        profile1 = model1._generate_agent_profile(StakeholderType.CONTENT_CREATOR)
        profile2 = model2._generate_agent_profile(StakeholderType.CONTENT_CREATOR)

        # At least one field should differ
        fields_differ = (
            profile1.initial_balance != profile2.initial_balance or
            profile1.risk_tolerance != profile2.risk_tolerance or
            profile1.activity_frequency != profile2.activity_frequency or
            profile1.quality_threshold != profile2.quality_threshold or
            profile1.network_connectivity != profile2.network_connectivity
        )
        assert fields_differ

    def test_rng_attribute_present_on_model(self, model_with_seed_42):
        """Test that rng attribute is present and is a numpy Generator"""
        assert hasattr(model_with_seed_42, 'rng')
        assert isinstance(model_with_seed_42.rng, np.random.Generator)

    def test_no_stdlib_random_import_in_module(self):
        """Test that stdlib random module is not imported in agent_based_model"""
        # Read the file directly and check for 'import random'
        with open('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/economy/economics/agent_based_model.py', 'r') as f:
            content = f.read()

        # Check that 'import random' is not at the module level
        lines = content.split('\n')
        import_random_lines = [line for line in lines if line.strip() == 'import random']

        assert len(import_random_lines) == 0, f"Found 'import random' in file: {import_random_lines}"

    def test_content_quality_weighted_selection_favors_high_quality(self):
        """Test that quality-weighted content selection favors high-quality content"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel

        model = PRSMEconomicModel(num_agents=10, seed=0)

        # Run process_query multiple times and track selections
        high_quality_selections = 0
        low_quality_selections = 0

        for i in range(100):
            # Reset content registry for each iteration
            model.content_registry = {
                0: {'creator_id': 0, 'quality': 0.1, 'usage_count': 0, 'total_revenue': Decimal('0')},
                1: {'creator_id': 1, 'quality': 0.9, 'usage_count': 0, 'total_revenue': Decimal('0')},
            }

            # Call process_query (it will select content based on quality weights)
            quality = model.process_query(0, Decimal('1.0'))

            # Track which was selected
            if quality == 0.9:
                high_quality_selections += 1
            else:
                low_quality_selections += 1

        # High quality content should be selected significantly more often (>60%)
        high_quality_ratio = high_quality_selections / 100
        assert high_quality_ratio > 0.6, f"High quality ratio {high_quality_ratio} should be > 0.6"

    def test_volatile_market_uses_model_rng(self):
        """Test that volatile market uses seeded RNG for reproducibility"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel, MarketCondition

        # Create two models with same seed in VOLATILE market condition
        model1 = PRSMEconomicModel(num_agents=5, seed=42, market_condition=MarketCondition.VOLATILE)
        model2 = PRSMEconomicModel(num_agents=5, seed=42, market_condition=MarketCondition.VOLATILE)

        prices1 = []
        prices2 = []

        # Run 5 steps and collect prices
        for _ in range(5):
            model1.step()
            prices1.append(float(model1.token_price))

            model2.step()
            prices2.append(float(model2.token_price))

        # Price sequences should be identical with same seed
        assert prices1 == prices2, "Price sequences should be identical with same seed"


@pytest.mark.skipif(not MESA_COMPATIBLE, reason="Requires Mesa <3.0 (RandomActivation API)")
class TestAgentProfileParameters:
    """Tests for AgentProfile behavioral parameters - requires Mesa <3.0"""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        from prsm.economy.economics.agent_based_model import PRSMEconomicModel
        return PRSMEconomicModel(num_agents=10, seed=42)

    def test_profile_contains_behavioral_parameters(self, model):
        """Test that AgentProfile contains behavioral parameters"""
        from prsm.economy.economics.agent_based_model import StakeholderType

        profile = model._generate_agent_profile(StakeholderType.CONTENT_CREATOR)

        assert hasattr(profile, 'content_creation_probability')
        assert 0 <= profile.content_creation_probability <= 1

    def test_node_operator_profile_parameters(self, model):
        """Test that NODE_OPERATOR profile has compute reward range"""
        from prsm.economy.economics.agent_based_model import StakeholderType

        profile = model._generate_agent_profile(StakeholderType.NODE_OPERATOR)

        assert hasattr(profile, 'compute_reward_range')
        assert isinstance(profile.compute_reward_range, tuple)
        assert len(profile.compute_reward_range) == 2
        assert profile.compute_reward_range[0] < profile.compute_reward_range[1]

    def test_token_holder_profile_parameters(self, model):
        """Test that TOKEN_HOLDER profile has staking and trading parameters"""
        from prsm.economy.economics.agent_based_model import StakeholderType

        profile = model._generate_agent_profile(StakeholderType.TOKEN_HOLDER)

        assert hasattr(profile, 'staking_probability')
        assert hasattr(profile, 'trading_probability')
        assert 0 <= profile.staking_probability <= 1
        assert 0 <= profile.trading_probability <= 1

    def test_simulation_id_is_valid_uuid(self):
        """Test that simulation_id is a valid UUID"""
        import asyncio
        from prsm.economy.economics.agent_based_model import EconomicSimulationRunner

        async def run_test():
            runner = EconomicSimulationRunner()

            # Mock minimal scenarios for fast test
            runner.scenarios = [
                {"name": "test", "condition": runner.scenarios[0]["condition"], "steps": 1}
            ]

            result = await runner.run_comprehensive_simulation()

            # Verify simulation_id is a valid UUID
            simulation_id = result["simulation_id"]

            # This should not raise an error
            parsed_uuid = uuid_lib.UUID(simulation_id)
            assert str(parsed_uuid) == simulation_id

        asyncio.run(run_test())


class TestAgentBasedModelCodeChanges:
    """Tests that verify code changes without requiring Mesa runtime"""

    def test_agent_profile_has_behavioral_parameters(self):
        """Test that AgentProfile dataclass has the new behavioral parameters"""
        from prsm.economy.economics.agent_based_model import AgentProfile
        from dataclasses import fields

        field_names = [f.name for f in fields(AgentProfile)]

        # Check for all behavioral parameters added
        expected_fields = [
            'content_creation_probability',
            'content_quality_range',
            'creation_cost_range',
            'validation_probability',
            'validation_fee_range',
            'query_probability',
            'query_cost_range',
            'compute_reward_range',
            'operational_cost_range',
            'investment_probability',
            'investment_cost_range',
            'staking_probability',
            'stake_fraction_range',
            'trading_probability',
            'trade_fraction_range',
            'trade_price_change_range',
        ]

        for field in expected_fields:
            assert field in field_names, f"Missing field: {field}"

    def test_uuid_import_present(self):
        """Test that uuid module is imported (for simulation_id)"""
        with open('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/economy/economics/agent_based_model.py', 'r') as f:
            content = f.read()

        assert 'import uuid' in content, "uuid import should be present"

    def test_no_random_module_import(self):
        """Test that random module is NOT imported"""
        with open('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/economy/economics/agent_based_model.py', 'r') as f:
            lines = f.readlines()

        # Check that no line is exactly 'import random'
        for i, line in enumerate(lines):
            assert line.strip() != 'import random', f"Found 'import random' at line {i+1}"

    def test_rng_initialization_in_model(self):
        """Test that PRSMEconomicModel has rng initialization"""
        with open('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/economy/economics/agent_based_model.py', 'r') as f:
            content = f.read()

        assert 'self.rng = np.random.default_rng' in content, "RNG initialization should be present"
        assert 'seed: Optional[int] = None' in content or 'seed: int | None = None' in content, "seed parameter should be present"
