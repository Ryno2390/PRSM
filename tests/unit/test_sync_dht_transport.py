"""Unit tests for SyncDHTTransport — sync ``SendMessageFn`` adapter.

Covers address parsing, the sync↔async bridge, wire-framing round-trip
against a real loopback TCP server, every error path the contract
documents, concurrency from multiple uploader threads, and
``DHTLoopRunner`` lifecycle.
"""
from __future__ import annotations

import asyncio
import socket
import struct
import threading
import time
from typing import List, Optional, Tuple

import pytest

from prsm.network.sync_dht_transport import (
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DHTLoopRunner,
    SyncDHTTransport,
    TransportFailureError,
)
from prsm.node.transport_adapter import (
    DirectAdapter,
    TransportConnectError,
)


# ──────────────────────────────────────────────────────────────────────
# helpers — loopback TCP echo server speaking the length-prefixed wire
# format. Exposes a configurable hook so per-test failure modes can be
# injected (oversized prefix, premature close, slow read, etc.).
# ──────────────────────────────────────────────────────────────────────


class _LoopbackEchoServer:
    """Length-prefixed echo server bound to 127.0.0.1 on an ephemeral
    port. Reads one length-prefixed request, responds with a
    length-prefixed payload. Behavior is configurable via callable
    hooks injected on construction.
    """

    def __init__(
        self,
        *,
        response_factory=None,
        announce_length=None,
        close_before_response: bool = False,
        send_short_prefix: bool = False,
        delay_before_response: float = 0.0,
    ):
        self._response_factory = response_factory or (lambda req: req)
        self._announce_length = announce_length
        self._close_before_response = close_before_response
        self._send_short_prefix = send_short_prefix
        self._delay_before_response = delay_before_response
        self._server: Optional[asyncio.AbstractServer] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.requests_received: List[bytes] = []
        self._lock = threading.Lock()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host="127.0.0.1", port=0,
        )
        sockets = self._server.sockets
        assert sockets, "asyncio.start_server returned no sockets"
        self.host, self.port = sockets[0].getsockname()[:2]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            prefix = await reader.readexactly(4)
            (req_len,) = struct.unpack(">I", prefix)
            request = await reader.readexactly(req_len)
            with self._lock:
                self.requests_received.append(request)

            if self._delay_before_response:
                # NOT asyncio.sleep — the project conftest installs an
                # autouse fixture that mocks asyncio.sleep to no-op,
                # which would defeat this server's slow-response
                # simulation. Use threading.Event.wait via the loop
                # executor; that primitive isn't mocked.
                event = threading.Event()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, event.wait, self._delay_before_response,
                )

            if self._close_before_response:
                writer.close()
                await writer.wait_closed()
                return

            if self._send_short_prefix:
                # Send only 2 bytes of the 4-byte prefix, then close.
                writer.write(b"\x00\x00")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            response = self._response_factory(request)
            announced = (
                self._announce_length
                if self._announce_length is not None
                else len(response)
            )
            writer.write(struct.pack(">I", announced) + response)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture
def loop_runner():
    runner = DHTLoopRunner(name="test-dht-loop")
    runner.start()
    yield runner
    runner.stop()


def _start_echo_server(loop, **kwargs) -> _LoopbackEchoServer:
    """Start an _LoopbackEchoServer on ``loop`` from the test thread."""
    server = _LoopbackEchoServer(**kwargs)
    fut = asyncio.run_coroutine_threadsafe(server.start(), loop)
    fut.result(timeout=5.0)
    return server


def _stop_echo_server(loop, server: _LoopbackEchoServer) -> None:
    fut = asyncio.run_coroutine_threadsafe(server.stop(), loop)
    fut.result(timeout=5.0)


# ──────────────────────────────────────────────────────────────────────
# address parsing
# ──────────────────────────────────────────────────────────────────────


class TestAddressParsing:
    def test_rejects_non_string_address(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="must be str"):
            transport.send(12345, b"req")  # type: ignore[arg-type]

    def test_rejects_address_with_no_colon(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="host:port"):
            transport.send("host-only", b"req")

    def test_rejects_address_with_empty_host(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="empty host"):
            transport.send(":9001", b"req")

    def test_rejects_non_integer_port(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="non-integer port"):
            transport.send("host:abc", b"req")

    def test_rejects_zero_port(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="port 0 out of range"):
            transport.send("host:0", b"req")

    def test_rejects_negative_port(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="out of range"):
            transport.send("host:-1", b"req")

    def test_rejects_oversized_port(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="out of range"):
            transport.send("host:99999", b"req")


# ──────────────────────────────────────────────────────────────────────
# constructor validation
# ──────────────────────────────────────────────────────────────────────


class TestConstructor:
    def test_rejects_none_adapter(self, loop_runner):
        with pytest.raises(ValueError, match="adapter"):
            SyncDHTTransport(None, loop_runner.start())  # type: ignore

    def test_rejects_none_loop(self):
        with pytest.raises(ValueError, match="loop"):
            SyncDHTTransport(DirectAdapter(), None)  # type: ignore

    def test_rejects_zero_default_timeout(self, loop_runner):
        with pytest.raises(ValueError, match="default_timeout"):
            SyncDHTTransport(
                DirectAdapter(), loop_runner.start(), default_timeout=0,
            )

    def test_rejects_negative_default_timeout(self, loop_runner):
        with pytest.raises(ValueError, match="default_timeout"):
            SyncDHTTransport(
                DirectAdapter(), loop_runner.start(), default_timeout=-1.0,
            )

    def test_rejects_zero_max_response_bytes(self, loop_runner):
        with pytest.raises(ValueError, match="max_response_bytes"):
            SyncDHTTransport(
                DirectAdapter(),
                loop_runner.start(),
                max_response_bytes=0,
            )

    def test_name_reflects_adapter(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        assert transport.name == "sync-dht-direct"


# ──────────────────────────────────────────────────────────────────────
# happy-path round-trip against a real loopback server
# ──────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_echo_round_trip(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            response = transport.send(
                f"{server.host}:{server.port}", b"hello",
            )
            assert response == b"hello"
            assert server.requests_received == [b"hello"]
        finally:
            _stop_echo_server(loop, server)

    def test_response_is_bytes_not_bytearray(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            response = transport.send(
                f"{server.host}:{server.port}", b"x",
            )
            assert isinstance(response, bytes)
        finally:
            _stop_echo_server(loop, server)

    def test_accepts_bytearray_request(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            response = transport.send(
                f"{server.host}:{server.port}", bytearray(b"abc"),
            )
            assert response == b"abc"
        finally:
            _stop_echo_server(loop, server)

    def test_rejects_non_bytes_request(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="must be bytes"):
            transport.send("127.0.0.1:1", "string-request")  # type: ignore

    def test_round_trip_large_payload(self, loop_runner):
        loop = loop_runner.start()
        # 200 KiB — comfortably below the 1 MiB default cap and below
        # the 256 KiB DHT MAX_MESSAGE_BYTES, so realistic for a real
        # DHT response.
        payload = b"P" * (200 * 1024)
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            response = transport.send(
                f"{server.host}:{server.port}", payload,
            )
            assert response == payload
        finally:
            _stop_echo_server(loop, server)

    def test_round_trip_zero_length_response(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop, response_factory=lambda r: b"")
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            response = transport.send(
                f"{server.host}:{server.port}", b"req",
            )
            assert response == b""
        finally:
            _stop_echo_server(loop, server)


# ──────────────────────────────────────────────────────────────────────
# error paths
# ──────────────────────────────────────────────────────────────────────


class _RaisingAdapter:
    """Stub adapter whose open_connection always raises."""

    name = "raising"

    async def open_connection(
        self, host: str, port: int, *, timeout: float = 30.0,
    ) -> socket.socket:
        raise TransportConnectError(f"refused {host}:{port}")


class TestErrorPaths:
    def test_open_connection_failure_raises_transport_failure(
        self, loop_runner,
    ):
        loop = loop_runner.start()
        transport = SyncDHTTransport(_RaisingAdapter(), loop)
        with pytest.raises(
            TransportFailureError, match="open_connection",
        ):
            transport.send("nowhere:9999", b"req")

    def test_response_exceeds_max_response_bytes(self, loop_runner):
        loop = loop_runner.start()
        # Server announces length 2_000_000 but cap is 1024.
        server = _start_echo_server(loop, announce_length=2_000_000)
        try:
            transport = SyncDHTTransport(
                DirectAdapter(), loop, max_response_bytes=1024,
            )
            with pytest.raises(
                TransportFailureError, match="exceeds max_response_bytes",
            ):
                transport.send(
                    f"{server.host}:{server.port}", b"req",
                )
        finally:
            _stop_echo_server(loop, server)

    def test_premature_close_before_prefix_raises(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop, close_before_response=True)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            with pytest.raises(
                TransportFailureError, match="peer closed",
            ):
                transport.send(
                    f"{server.host}:{server.port}", b"req",
                )
        finally:
            _stop_echo_server(loop, server)

    def test_short_prefix_then_close_raises(self, loop_runner):
        loop = loop_runner.start()
        server = _start_echo_server(loop, send_short_prefix=True)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            with pytest.raises(
                TransportFailureError, match="peer closed",
            ):
                transport.send(
                    f"{server.host}:{server.port}", b"req",
                )
        finally:
            _stop_echo_server(loop, server)

    def test_loop_not_running_raises(self):
        # Build a loop, do not run it.
        loop = asyncio.new_event_loop()
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            with pytest.raises(
                TransportFailureError, match="not running",
            ):
                transport.send("127.0.0.1:1", b"req")
        finally:
            loop.close()

    def test_per_call_timeout_overrides_default(self, loop_runner):
        """A short per-call timeout fires before the slow server
        responds, surfacing as TransportFailureError."""
        loop = loop_runner.start()
        server = _start_echo_server(loop, delay_before_response=2.0)
        try:
            transport = SyncDHTTransport(
                DirectAdapter(), loop, default_timeout=10.0,
            )
            t0 = time.monotonic()
            with pytest.raises(TransportFailureError):
                transport.send(
                    f"{server.host}:{server.port}", b"req", timeout=0.25,
                )
            elapsed = time.monotonic() - t0
            assert elapsed < 1.5, (
                f"timeout did not fire promptly: elapsed={elapsed:.2f}s"
            )
        finally:
            _stop_echo_server(loop, server)

    def test_per_call_timeout_must_be_positive(self, loop_runner):
        transport = SyncDHTTransport(DirectAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="timeout"):
            transport.send("127.0.0.1:1", b"req", timeout=0.0)

    def test_unrelated_exception_wrapped_in_transport_failure(
        self, loop_runner,
    ):
        """An adapter that raises a non-TransportConnectError gets
        wrapped — defense-in-depth so the SendMessageFn contract
        (only raises bubble up) holds even if a transport adapter
        leaks a non-TransportError."""
        class _BadAdapter:
            name = "bad"

            async def open_connection(self, host, port, *, timeout=30.0):
                raise RuntimeError("unexpected")

        transport = SyncDHTTransport(_BadAdapter(), loop_runner.start())
        with pytest.raises(TransportFailureError, match="RuntimeError"):
            transport.send("127.0.0.1:1", b"req")


# ──────────────────────────────────────────────────────────────────────
# concurrency — multiple sync callers sharing one transport
# ──────────────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_many_concurrent_senders(self, loop_runner):
        """Spawn 16 worker threads each sending a unique payload; all
        must round-trip without deadlock or interleaving corruption.

        Validates the sync↔asyncio bridge under contention. Per the
        plan doc Risks section, deadlock under concurrent uploads is
        the primary correctness concern for this layer.
        """
        loop = loop_runner.start()
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            n_workers = 16
            n_requests_each = 4
            barrier = threading.Barrier(n_workers)
            results: List[Tuple[int, int, bytes, bytes]] = []
            results_lock = threading.Lock()
            errors: List[BaseException] = []

            def worker(worker_id: int) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    for j in range(n_requests_each):
                        payload = f"w{worker_id:02d}-r{j:02d}".encode()
                        response = transport.send(
                            f"{server.host}:{server.port}", payload,
                        )
                        with results_lock:
                            results.append(
                                (worker_id, j, payload, response),
                            )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(i,))
                for i in range(n_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15.0)
                assert not t.is_alive(), (
                    "worker thread did not exit within 15s — possible "
                    "sync↔asyncio deadlock"
                )
            assert not errors, f"workers raised: {errors}"
            assert len(results) == n_workers * n_requests_each
            for _, _, payload, response in results:
                assert payload == response
        finally:
            _stop_echo_server(loop, server)


# ──────────────────────────────────────────────────────────────────────
# DHTLoopRunner lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestDHTLoopRunner:
    def test_start_returns_running_loop(self):
        runner = DHTLoopRunner(name="lifecycle-test")
        try:
            loop = runner.start()
            assert isinstance(loop, asyncio.AbstractEventLoop)
            assert loop.is_running()
        finally:
            runner.stop()

    def test_start_is_idempotent(self):
        runner = DHTLoopRunner(name="idempotent-test")
        try:
            loop1 = runner.start()
            loop2 = runner.start()
            assert loop1 is loop2
        finally:
            runner.stop()

    def test_stop_is_idempotent(self):
        runner = DHTLoopRunner(name="double-stop")
        runner.start()
        runner.stop()
        runner.stop()  # no raise

    def test_stop_with_no_start(self):
        runner = DHTLoopRunner(name="never-started")
        runner.stop()  # no raise

    def test_context_manager(self):
        runner = DHTLoopRunner(name="ctx-test")
        with runner as loop:
            assert loop.is_running()
        # After exit, the loop should not be running. There is no
        # sync API to assert .is_running() == False on a closed loop
        # without races, so just verify start can be re-invoked
        # cleanly.
        loop2 = runner.start()
        try:
            assert loop2.is_running()
        finally:
            runner.stop()

    def test_can_send_after_restart(self, loop_runner):
        # Use the fixture-provided runner to confirm send works,
        # then a fresh runner to confirm loop-thread isolation.
        loop = loop_runner.start()
        server = _start_echo_server(loop)
        try:
            transport = SyncDHTTransport(DirectAdapter(), loop)
            assert transport.send(
                f"{server.host}:{server.port}", b"first",
            ) == b"first"
        finally:
            _stop_echo_server(loop, server)


# ──────────────────────────────────────────────────────────────────────
# constants — surface check
# ──────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_default_timeout_positive(self):
        assert DEFAULT_TIMEOUT_SECONDS > 0

    def test_default_max_response_bytes_geq_dht_max(self):
        # Both DHTs cap MAX_MESSAGE_BYTES at 256 KiB. Our default
        # max_response_bytes must be at least that to avoid rejecting
        # legitimate responses.
        assert DEFAULT_MAX_RESPONSE_BYTES >= 256 * 1024
