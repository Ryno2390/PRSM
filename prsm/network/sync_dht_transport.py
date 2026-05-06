"""SyncDHTTransport — sync ``SendMessageFn`` adapter over an async transport.

Wraps a :class:`prsm.node.transport_adapter.TransportAdapter` plus a
running ``asyncio`` loop (typically owned by ``DHTLoopRunner`` in a
dedicated thread) to expose a synchronous

    (address: str, request_bytes: bytes) -> bytes

callable that the sync DHT clients (``ManifestDHTClient`` and
``EmbeddingDHTClient``) call directly from the upload-critical path.

Why sync at the call site
-------------------------
Both DHT clients are sync because the upload critical path is sync.
``run_coroutine_threadsafe`` is the standard sync↔asyncio bridge but
requires a long-running asyncio loop in a different thread than the
caller. ``DHTLoopRunner`` provides that. Tests may also pass any
running loop (e.g. one already running in the test fixture).

Wire framing
------------
4-byte big-endian length prefix, followed by ``request_bytes``. The
peer responds the same way: 4-byte length prefix + response bytes.
Single-shot per connection; matches the existing ManifestDHT /
EmbeddingDHT message-per-RPC semantics. Connection pooling is a
deliberate non-goal for v1 (see plan doc Risks).

Error model
-----------
Any failure (address parse, transport open, write, read, length cap
exceeded, peer-side close before length prefix, loop already stopped)
raises :class:`TransportFailureError`. This matches what the DHT
clients' ``send_message`` callable contract documents: a raise from the
callable indicates a transport-level failure. The DHT clients catch
broadly with ``except Exception`` and re-raise their typed
``TransportFailureError`` to callers, so this module's exception class
is mostly an internal-correctness anchor.

Concurrency
-----------
Designed for concurrent ``send()`` calls from multiple uploader threads
sharing one ``SyncDHTTransport``. Each call schedules an independent
coroutine on the loop, so contention is bounded by socket I/O
parallelism and the loop's executor pool. No per-instance mutable state
is touched on the hot path.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import struct
import threading
from typing import Optional, Tuple

from prsm.node.transport_adapter import TransportAdapter, TransportConnectError

logger = logging.getLogger(__name__)


# 1 MiB ceiling on inbound DHT response. Both ManifestDHT and
# EmbeddingDHT cap MAX_MESSAGE_BYTES at 256 KiB, so 1 MiB leaves slack
# for protocol overhead and future framing fields without exposing
# the listener to memory exhaustion via crafted length prefixes.
DEFAULT_MAX_RESPONSE_BYTES = 1 << 20

# Wall-clock budget for one full RPC: connect + write + read.
DEFAULT_TIMEOUT_SECONDS = 10.0

_LENGTH_PREFIX_BYTES = 4
_LENGTH_PREFIX_FORMAT = ">I"  # big-endian uint32


class TransportFailureError(Exception):
    """Raised by :meth:`SyncDHTTransport.send` on any transport failure.

    Distinct class so internal callers can ``except TransportFailureError``
    without conflating with bugs. The DHT clients' own
    ``TransportFailureError`` types are different classes — they catch
    broadly via ``except Exception`` and re-raise their typed variant.
    """


def _parse_address(address: str) -> Tuple[str, int]:
    """Parse a 'host:port' string into ``(host, port)``.

    IPv6 addresses with embedded colons are not supported in v1 — the
    PRSM peer model uses IPv4 + DNS hostnames. If/when IPv6 is added
    this parser will need to grow bracket handling.
    """
    if not isinstance(address, str):
        raise TransportFailureError(
            f"address must be str, got {type(address).__name__}"
        )
    if ":" not in address:
        raise TransportFailureError(
            f"address must be 'host:port', got {address!r}"
        )
    host, _, port_str = address.rpartition(":")
    if not host:
        raise TransportFailureError(f"empty host in address {address!r}")
    try:
        port = int(port_str)
    except ValueError:
        raise TransportFailureError(
            f"non-integer port in address {address!r}"
        ) from None
    if port <= 0 or port > 65535:
        raise TransportFailureError(
            f"port {port} out of range in address {address!r}"
        )
    return host, port


def _length_prefix(payload: bytes) -> bytes:
    """Pack a 4-byte big-endian length prefix for ``payload``."""
    return struct.pack(_LENGTH_PREFIX_FORMAT, len(payload)) + payload


async def _read_exactly(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, n: int
) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` via the loop. Raises
    :class:`TransportFailureError` on premature EOF or socket errors."""
    if n == 0:
        return b""
    buf = bytearray()
    remaining = n
    while remaining > 0:
        try:
            chunk = await loop.sock_recv(sock, remaining)
        except OSError as exc:
            raise TransportFailureError(
                f"sock_recv failed after {len(buf)}/{n} bytes: {exc}"
            ) from exc
        if not chunk:
            raise TransportFailureError(
                f"peer closed connection after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
        remaining -= len(chunk)
    return bytes(buf)


async def _send_and_receive(
    adapter: TransportAdapter,
    host: str,
    port: int,
    request_bytes: bytes,
    *,
    connect_timeout: float,
    max_response_bytes: int,
) -> bytes:
    """Open a connection, write the framed request, read the framed
    response, close the connection. All of the asyncio-side work for one
    RPC. Designed to be invoked via ``asyncio.run_coroutine_threadsafe``.
    """
    try:
        sock = await adapter.open_connection(
            host, port, timeout=connect_timeout,
        )
    except TransportConnectError as exc:
        raise TransportFailureError(
            f"transport open_connection({host}:{port}) failed: {exc}"
        ) from exc
    # Adapters return a connected socket; ensure non-blocking semantics
    # so the asyncio loop primitives operate correctly. DirectAdapter
    # already calls setblocking(False), but other adapters (SocksAdapter
    # via python-socks) may return blocking sockets — be defensive.
    try:
        sock.setblocking(False)
    except OSError:
        # Already non-blocking, or kernel rejected — non-fatal because
        # the loop primitives will surface the real issue downstream.
        pass

    try:
        loop = asyncio.get_running_loop()
        framed = _length_prefix(request_bytes)
        try:
            await loop.sock_sendall(sock, framed)
        except OSError as exc:
            raise TransportFailureError(
                f"sock_sendall to {host}:{port} failed: {exc}"
            ) from exc

        prefix = await _read_exactly(loop, sock, _LENGTH_PREFIX_BYTES)
        (response_len,) = struct.unpack(_LENGTH_PREFIX_FORMAT, prefix)
        if response_len > max_response_bytes:
            raise TransportFailureError(
                f"response from {host}:{port} announced length "
                f"{response_len} exceeds max_response_bytes "
                f"{max_response_bytes}"
            )
        return await _read_exactly(loop, sock, response_len)
    finally:
        try:
            sock.close()
        except OSError:
            pass


class SyncDHTTransport:
    """Synchronous ``SendMessageFn`` over an async ``TransportAdapter``.

    Construct with a ``TransportAdapter`` (DirectAdapter or
    SocksAdapter, R9 §6.2) and a running asyncio loop in a different
    thread than the caller (typically owned by :class:`DHTLoopRunner`).
    Then :meth:`send` is the sync ``(address, request_bytes) -> bytes``
    callable both DHT clients consume.

    :param adapter: Underlying transport adapter; opens a connected
        socket to ``(host, port)`` for each RPC.
    :param loop: A running asyncio event loop in a thread other than
        the calling thread. Coroutines are scheduled here via
        ``asyncio.run_coroutine_threadsafe``. The transport does NOT
        own the loop's lifecycle — the caller is responsible for
        starting it before any ``send()`` call and stopping it after
        all in-flight calls have completed.
    :param default_timeout: Wall-clock budget per RPC (connect +
        write + read). Passing ``timeout=`` to :meth:`send` overrides
        per-call.
    :param max_response_bytes: Reject any response whose length-prefix
        exceeds this value. Defends against malicious or buggy peers
        announcing oversized responses to exhaust memory.
    """

    def __init__(
        self,
        adapter: TransportAdapter,
        loop: asyncio.AbstractEventLoop,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if adapter is None:
            raise ValueError("adapter must not be None")
        if loop is None:
            raise ValueError("loop must not be None")
        if default_timeout <= 0:
            raise ValueError(
                f"default_timeout must be positive; got {default_timeout}"
            )
        if max_response_bytes <= 0:
            raise ValueError(
                f"max_response_bytes must be positive; "
                f"got {max_response_bytes}"
            )
        self._adapter = adapter
        self._loop = loop
        self._default_timeout = default_timeout
        self._max_response_bytes = max_response_bytes

    @property
    def name(self) -> str:
        """Identifier for logging/metrics; reflects the adapter."""
        adapter_name = getattr(self._adapter, "name", "unknown")
        return f"sync-dht-{adapter_name}"

    def send(
        self,
        address: str,
        request_bytes: bytes,
        *,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Send ``request_bytes`` to ``address`` (host:port); return the
        response payload bytes.

        :raises TransportFailureError: any transport-layer failure
            (address parse, open_connection failure, write failure,
            premature EOF, response-length cap exceeded, timeout,
            loop not running).
        """
        if not isinstance(request_bytes, (bytes, bytearray)):
            raise TransportFailureError(
                f"request_bytes must be bytes, got "
                f"{type(request_bytes).__name__}"
            )
        host, port = _parse_address(address)
        budget = timeout if timeout is not None else self._default_timeout
        if budget <= 0:
            raise TransportFailureError(
                f"timeout must be positive; got {budget}"
            )

        # Schedule the coroutine on the foreign loop. If the loop has
        # been stopped, run_coroutine_threadsafe still queues the
        # coroutine but it will never run; future.result(timeout=) will
        # surface as TimeoutError, which we map to TransportFailureError.
        # We pre-check is_running for a cleaner error.
        if not self._loop.is_running():
            raise TransportFailureError(
                "asyncio loop is not running; cannot send DHT request"
            )

        coro = _send_and_receive(
            self._adapter, host, port, bytes(request_bytes),
            connect_timeout=budget,
            max_response_bytes=self._max_response_bytes,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=budget)
        except TransportFailureError:
            raise
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise TransportFailureError(
                f"DHT request to {address} timed out after {budget}s"
            ) from exc
        except TimeoutError as exc:
            # concurrent.futures.TimeoutError on Py3.11+; future.cancel
            # is best-effort.
            future.cancel()
            raise TransportFailureError(
                f"DHT request to {address} timed out after {budget}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise TransportFailureError(
                f"DHT request to {address} failed: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc


class DHTLoopRunner:
    """Owns a daemon thread running an asyncio loop dedicated to DHT
    transport.

    Creating one and calling :meth:`start` returns a running asyncio
    loop suitable for handing to :class:`SyncDHTTransport`. Calling
    :meth:`stop` cleanly shuts the loop down. The runner is idempotent
    — repeated start/stop calls are safe.

    Lifecycle pattern::

        runner = DHTLoopRunner()
        loop = runner.start()
        try:
            transport = SyncDHTTransport(adapter, loop)
            # ... use transport ...
        finally:
            runner.stop()

    The loop is shared across SyncDHTTransport, DHTListener, and any
    other DHT-side asyncio code at the node level. Run loops are
    relatively expensive in OS-thread overhead; one per node (not one
    per transport instance) is the intended deployment.
    """

    def __init__(self, *, name: str = "prsm-dht-loop") -> None:
        self._name = name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> asyncio.AbstractEventLoop:
        """Start the loop thread; idempotent. Returns the running loop."""
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            self._started.clear()
            self._thread = threading.Thread(
                target=self._run, name=self._name, daemon=True,
            )
            self._thread.start()
        # Wait outside the lock so a concurrent stop() can interrupt.
        if not self._started.wait(timeout=5.0):
            raise RuntimeError(
                f"DHTLoopRunner({self._name!r}) failed to start within 5s"
            )
        assert self._loop is not None
        return self._loop

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the loop and join the thread. Idempotent."""
        with self._lock:
            loop = self._loop
            thread = self._thread
        if loop is None or thread is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "DHTLoopRunner(%r): loop thread did not exit within "
                "%.1fs; leaking the thread (daemon=True so it dies on "
                "interpreter shutdown)",
                self._name, timeout,
            )
        with self._lock:
            self._loop = None
            self._thread = None
            self._started.clear()

    def _run(self) -> None:
        """Thread target — owns the loop. Stops on loop.stop()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_forever()
        finally:
            try:
                # Cancel pending tasks to avoid leaks on shutdown.
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True),
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "DHTLoopRunner(%r): error draining pending tasks",
                    self._name,
                )
            finally:
                try:
                    loop.close()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "DHTLoopRunner(%r): error closing loop",
                        self._name,
                    )

    def __enter__(self) -> asyncio.AbstractEventLoop:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DHTLoopRunner",
    "SyncDHTTransport",
    "TransportFailureError",
]
