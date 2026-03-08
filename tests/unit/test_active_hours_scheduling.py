"""
Tests for Active Hours Scheduling (Phase 4)

Tests the is_active_now() function for time-based node scheduling.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from prsm.node.config import NodeConfig, is_active_now


class TestIsActiveNow:
    """Test cases for the is_active_now scheduling function."""

    def test_no_schedule_configured_always_on(self):
        """Node with no schedule configured should always be active."""
        config = NodeConfig()
        config.active_hours_start = None
        config.active_hours_end = None
        config.active_days = []

        # Should be active regardless of time
        with patch('prsm.node.config.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 3, 0)  # 3 AM
            assert is_active_now(config) is True

            mock_dt.now.return_value = datetime(2024, 1, 1, 15, 0)  # 3 PM
            assert is_active_now(config) is True

    def test_only_start_configured_always_on(self):
        """If only start is configured (no end), node should always be active."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = None
        config.active_days = []

        assert is_active_now(config) is True

    def test_only_end_configured_always_on(self):
        """If only end is configured (no start), node should always be active."""
        config = NodeConfig()
        config.active_hours_start = None
        config.active_hours_end = 17
        config.active_days = []

        assert is_active_now(config) is True

    def test_normal_range_within_hours(self):
        """Test normal range (e.g., 09:00 - 17:00) during active hours."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = 17
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # 10 AM - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 10, 0)
            assert is_active_now(config) is True

            # 4 PM (16:00) - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 16, 0)
            assert is_active_now(config) is True

            # 9 AM exactly - should be active (start is inclusive)
            mock_dt.now.return_value = datetime(2024, 1, 1, 9, 0)
            assert is_active_now(config) is True

    def test_normal_range_outside_hours(self):
        """Test normal range (e.g., 09:00 - 17:00) outside active hours."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = 17
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # 8 AM - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 8, 0)
            assert is_active_now(config) is False

            # 5 PM (17:00) exactly - should be inactive (end is exclusive)
            mock_dt.now.return_value = datetime(2024, 1, 1, 17, 0)
            assert is_active_now(config) is False

            # 6 PM - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 18, 0)
            assert is_active_now(config) is False

            # Midnight - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0)
            assert is_active_now(config) is False

    def test_wrap_around_midnight_within_hours(self):
        """Test wrap-around range (e.g., 22:00 - 06:00) during active hours."""
        config = NodeConfig()
        config.active_hours_start = 22  # 10 PM
        config.active_hours_end = 6     # 6 AM
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # 11 PM (23:00) - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 23, 0)
            assert is_active_now(config) is True

            # Midnight - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0)
            assert is_active_now(config) is True

            # 3 AM - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 3, 0)
            assert is_active_now(config) is True

            # 5 AM - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 5, 0)
            assert is_active_now(config) is True

            # 10 PM (22:00) exactly - should be active (start is inclusive)
            mock_dt.now.return_value = datetime(2024, 1, 1, 22, 0)
            assert is_active_now(config) is True

    def test_wrap_around_midnight_outside_hours(self):
        """Test wrap-around range (e.g., 22:00 - 06:00) outside active hours."""
        config = NodeConfig()
        config.active_hours_start = 22  # 10 PM
        config.active_hours_end = 6     # 6 AM
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # 6 AM exactly - should be inactive (end is exclusive)
            mock_dt.now.return_value = datetime(2024, 1, 1, 6, 0)
            assert is_active_now(config) is False

            # 9 AM - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 9, 0)
            assert is_active_now(config) is False

            # 3 PM - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 15, 0)
            assert is_active_now(config) is False

            # 9 PM (21:00) - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 21, 0)
            assert is_active_now(config) is False

    def test_active_days_current_day_active(self):
        """Test that node is active when current day is in active_days."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = 17
        config.active_days = [0, 1, 2, 3, 4]  # Mon-Fri

        with patch('prsm.node.config.datetime') as mock_dt:
            # Monday 10 AM (weekday=0)
            mock_dt.now.return_value = datetime(2024, 1, 1, 10, 0)  # Jan 1, 2024 is Monday
            assert is_active_now(config) is True

            # Wednesday 2 PM (weekday=2)
            mock_dt.now.return_value = datetime(2024, 1, 3, 14, 0)
            assert is_active_now(config) is True

            # Friday 4 PM (weekday=4)
            mock_dt.now.return_value = datetime(2024, 1, 5, 16, 0)
            assert is_active_now(config) is True

    def test_active_days_current_day_inactive(self):
        """Test that node is inactive when current day is not in active_days."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = 17
        config.active_days = [0, 1, 2, 3, 4]  # Mon-Fri

        with patch('prsm.node.config.datetime') as mock_dt:
            # Saturday 10 AM (weekday=5)
            mock_dt.now.return_value = datetime(2024, 1, 6, 10, 0)  # Jan 6, 2024 is Saturday
            assert is_active_now(config) is False

            # Sunday 2 PM (weekday=6)
            mock_dt.now.return_value = datetime(2024, 1, 7, 14, 0)  # Jan 7, 2024 is Sunday
            assert is_active_now(config) is False

    def test_active_days_empty_every_day(self):
        """Test that empty active_days means every day is active."""
        config = NodeConfig()
        config.active_hours_start = 9
        config.active_hours_end = 17
        config.active_days = []  # Empty = every day

        with patch('prsm.node.config.datetime') as mock_dt:
            # Monday
            mock_dt.now.return_value = datetime(2024, 1, 1, 10, 0)
            assert is_active_now(config) is True

            # Saturday
            mock_dt.now.return_value = datetime(2024, 1, 6, 10, 0)
            assert is_active_now(config) is True

            # Sunday
            mock_dt.now.return_value = datetime(2024, 1, 7, 10, 0)
            assert is_active_now(config) is True

    def test_active_days_weekends_only(self):
        """Test weekend-only schedule."""
        config = NodeConfig()
        config.active_hours_start = 10
        config.active_hours_end = 22
        config.active_days = [5, 6]  # Sat-Sun only

        with patch('prsm.node.config.datetime') as mock_dt:
            # Saturday 2 PM - should be active
            mock_dt.now.return_value = datetime(2024, 1, 6, 14, 0)
            assert is_active_now(config) is True

            # Sunday 2 PM - should be active
            mock_dt.now.return_value = datetime(2024, 1, 7, 14, 0)
            assert is_active_now(config) is True

            # Wednesday 2 PM - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 3, 14, 0)
            assert is_active_now(config) is False

    def test_active_days_with_wrap_around_hours(self):
        """Test active_days combined with wrap-around hours."""
        config = NodeConfig()
        config.active_hours_start = 22  # 10 PM
        config.active_hours_end = 6     # 6 AM
        config.active_days = [0, 1, 2, 3, 4]  # Mon-Fri

        with patch('prsm.node.config.datetime') as mock_dt:
            # Monday 2 AM - should be active (within hours, Monday is active)
            mock_dt.now.return_value = datetime(2024, 1, 1, 2, 0)
            assert is_active_now(config) is True

            # Saturday 2 AM - should be inactive (within hours, but Saturday is not active)
            mock_dt.now.return_value = datetime(2024, 1, 6, 2, 0)
            assert is_active_now(config) is False

            # Friday 11 PM - should be active (within hours, Friday is active)
            mock_dt.now.return_value = datetime(2024, 1, 5, 23, 0)
            assert is_active_now(config) is True

    def test_single_hour_active_window(self):
        """Test a single-hour active window."""
        config = NodeConfig()
        config.active_hours_start = 12
        config.active_hours_end = 13  # Only active from 12:00-12:59
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # 12:00 - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 12, 0)
            assert is_active_now(config) is True

            # 12:59 - should be active
            mock_dt.now.return_value = datetime(2024, 1, 1, 12, 59)
            assert is_active_now(config) is True

            # 13:00 - should be inactive (end is exclusive)
            mock_dt.now.return_value = datetime(2024, 1, 1, 13, 0)
            assert is_active_now(config) is False

            # 11:00 - should be inactive
            mock_dt.now.return_value = datetime(2024, 1, 1, 11, 0)
            assert is_active_now(config) is False

    def test_full_24_hour_coverage(self):
        """Test full 24-hour coverage (0-24)."""
        config = NodeConfig()
        config.active_hours_start = 0
        config.active_hours_end = 24
        config.active_days = []

        with patch('prsm.node.config.datetime') as mock_dt:
            # Midnight
            mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0)
            assert is_active_now(config) is True

            # Noon
            mock_dt.now.return_value = datetime(2024, 1, 1, 12, 0)
            assert is_active_now(config) is True

            # 11 PM (23:00)
            mock_dt.now.return_value = datetime(2024, 1, 1, 23, 0)
            assert is_active_now(config) is True
