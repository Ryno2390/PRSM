"""
Unit tests for DatabaseConfigLoader.

Tests use mocked SQLAlchemy/asyncpg - no live database required.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json

from prsm.core.config.loaders import (
    DatabaseConfigLoader,
    load_database_config,
    create_composite_loader,
)


class TestDatabaseConfigLoader:
    """Tests for DatabaseConfigLoader class."""

    def test_load_returns_empty_when_table_missing(self):
        """Test that load() returns empty dict when configuration table doesn't exist."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        # Mock SQLAlchemy components
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = []  # Table doesn't exist
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result == {}

    def test_load_parses_string_value(self):
        """Test that load() correctly parses string values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        # Return a row with string value
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("app.name", "PRSM", "string")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result == {"app": {"name": "PRSM"}}

    def test_load_parses_int_value(self):
        """Test that load() correctly parses integer values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("nwtn.max_concurrent_queries", "10", "int")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result == {"nwtn": {"max_concurrent_queries": 10}}
            assert isinstance(result["nwtn"]["max_concurrent_queries"], int)

    def test_load_parses_float_value(self):
        """Test that load() correctly parses float values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("model.temperature", "0.7", "float")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result == {"model": {"temperature": 0.7}}
            assert isinstance(result["model"]["temperature"], float)

    def test_load_parses_bool_value_true(self):
        """Test that load() correctly parses boolean true values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("feature.enabled", "true", "bool"),
            ("feature.enabled_yes", "yes", "bool"),
            ("feature.enabled_1", "1", "bool"),
            ("feature.enabled_on", "on", "bool")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result["feature"]["enabled"] is True
            assert result["feature"]["enabled_yes"] is True
            assert result["feature"]["enabled_1"] is True
            assert result["feature"]["enabled_on"] is True

    def test_load_parses_bool_value_false(self):
        """Test that load() correctly parses boolean false values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("feature.disabled", "false", "bool"),
            ("feature.disabled_no", "no", "bool"),
            ("feature.disabled_0", "0", "bool"),
            ("feature.disabled_off", "off", "bool")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result["feature"]["disabled"] is False
            assert result["feature"]["disabled_no"] is False
            assert result["feature"]["disabled_0"] is False
            assert result["feature"]["disabled_off"] is False

    def test_load_parses_json_value(self):
        """Test that load() correctly parses JSON values."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        json_value = json.dumps({"key": "value", "nested": {"item": 123}})
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("model.config", json_value, "json")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            assert result == {"model": {"config": {"key": "value", "nested": {"item": 123}}}}

    def test_load_handles_bad_int_gracefully(self):
        """Test that load() handles unparseable int values gracefully."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("bad.value", "not_an_int", "int")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load()
            
            # Should fall back to raw string value
            assert result == {"bad": {"value": "not_an_int"}}

    def test_load_with_key_prefix_filters_rows(self):
        """Test that load() filters rows when key_prefix is provided."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("nwtn.max_queries", "100", "int"),
            ("nwtn.timeout", "30", "int")
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_conn
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["configuration"]
        
        with patch('sqlalchemy.create_engine', return_value=mock_engine), \
             patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch('sqlalchemy.text', side_effect=lambda x: x):
            
            result = loader.load(query_params={"key_prefix": "nwtn"})
            
            # Verify the query was called with the prefix filter
            call_args = mock_conn.execute.call_args
            # The text() wrapper returns the SQL string, check for LIKE clause
            sql_str = str(call_args[0][0])
            assert "WHERE key LIKE" in sql_str or "LIKE" in sql_str
            # Check the parameters passed to execute
            # call_args[0][1] is the second positional argument (the params dict)
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            assert params["prefix"] == "nwtn%"
            
            assert result == {"nwtn": {"max_queries": 100, "timeout": 30}}

    def test_set_nested_key_creates_nested_dict(self):
        """Test that _set_nested_key creates proper nested dictionary structure."""
        loader = DatabaseConfigLoader("postgresql://localhost/test", "configuration")
        
        config = {}
        
        # Test single level
        loader._set_nested_key(config, "level1", "value1")
        assert config == {"level1": "value1"}
        
        # Test two levels
        loader._set_nested_key(config, "level1.level2", "value2")
        assert config == {"level1": {"level2": "value2", **config["level1"]}}
        
        # Test three levels
        config = {}
        loader._set_nested_key(config, "a.b.c", "deep_value")
        assert config == {"a": {"b": {"c": "deep_value"}}}

    def test_load_database_config_convenience_function(self):
        """Test the load_database_config convenience function."""
        with patch.object(DatabaseConfigLoader, 'load') as mock_load:
            mock_load.return_value = {"test": {"key": "value"}}
            
            result = load_database_config(
                connection_string="postgresql://localhost/test",
                table_name="configuration",
                key_prefix="test"
            )
            
            assert result == {"test": {"key": "value"}}
            mock_load.assert_called_once_with({"key_prefix": "test"})

    def test_create_composite_loader_includes_database_when_conn_provided(self):
        """Test that create_composite_loader includes database loader when connection string provided."""
        # Without database connection string
        loader = create_composite_loader()
        assert 'database' not in loader.loaders
        assert 'file' in loader.loaders
        assert 'environment' in loader.loaders
        assert 'remote' in loader.loaders
        
        # With database connection string
        loader_with_db = create_composite_loader(
            database_connection_string="postgresql://localhost/test"
        )
        assert 'database' in loader_with_db.loaders
        assert isinstance(loader_with_db.loaders['database'], DatabaseConfigLoader)
        assert 'file' in loader_with_db.loaders
        assert 'environment' in loader_with_db.loaders
        assert 'remote' in loader_with_db.loaders
