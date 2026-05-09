"""Request-ID propagation into log records.

Companion to the X-Request-ID middleware: when a request is
in flight, log records produced by handlers + downstream code
get tagged with the current request_id via a contextvar-backed
log filter. Operators reading logs can grep for a specific
request_id and see exactly the lines emitted while that request
was processing.
"""
from __future__ import annotations

import logging

import pytest

from prsm.node.request_id_logging import (
    REQUEST_ID_VAR,
    RequestIdLogFilter,
    set_request_id,
    clear_request_id,
)


# ──────────────────────────────────────────────────────────────────────
# Contextvar primitives
# ──────────────────────────────────────────────────────────────────────


class TestContextVar:
    def test_default_request_id_is_dash(self):
        """When nothing is set, the contextvar default '-' renders
        in log records as a benign placeholder."""
        clear_request_id()
        assert REQUEST_ID_VAR.get() == "-"

    def test_set_and_clear_round_trip(self):
        token = set_request_id("abc-123")
        try:
            assert REQUEST_ID_VAR.get() == "abc-123"
        finally:
            clear_request_id(token)
        assert REQUEST_ID_VAR.get() == "-"


# ──────────────────────────────────────────────────────────────────────
# Log filter injection
# ──────────────────────────────────────────────────────────────────────


class TestLogFilter:
    def test_filter_injects_request_id_attribute(self):
        clear_request_id()
        token = set_request_id("test-rid-xyz")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO,
                pathname="x.py", lineno=1,
                msg="hello", args=(), exc_info=None,
            )
            filt = RequestIdLogFilter()
            filt.filter(record)
            assert getattr(record, "request_id", None) == "test-rid-xyz"
        finally:
            clear_request_id(token)

    def test_filter_injects_dash_when_no_request_id(self):
        clear_request_id()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="x.py", lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        filt = RequestIdLogFilter()
        filt.filter(record)
        assert record.request_id == "-"

    def test_filter_returns_true_to_pass_record(self):
        """LogFilter.filter() must return truthy to let the record
        through; we never drop records."""
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="x.py", lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        filt = RequestIdLogFilter()
        assert filt.filter(record) is True


# ──────────────────────────────────────────────────────────────────────
# Independent contextvars per coroutine (asyncio isolation)
# ──────────────────────────────────────────────────────────────────────


class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_distinct_coroutines_see_distinct_ids(self):
        """Two coroutines running concurrently see their own
        request_id; one's set doesn't leak into the other's view."""
        import asyncio

        async def task(rid: str, observed: list):
            token = set_request_id(rid)
            try:
                # yield to let the scheduler interleave
                await asyncio.sleep(0)
                observed.append((rid, REQUEST_ID_VAR.get()))
            finally:
                clear_request_id(token)

        observed_a: list = []
        observed_b: list = []
        await asyncio.gather(
            task("aaa", observed_a),
            task("bbb", observed_b),
        )
        assert observed_a == [("aaa", "aaa")]
        assert observed_b == [("bbb", "bbb")]
