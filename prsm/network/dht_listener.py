"""DHTListener — asyncio TCP listener that serves DHT requests via a
:class:`DHTRequestRouter`.

Pairs with :class:`SyncDHTTransport` on the client side. A node that
runs the ManifestDHT and/or EmbeddingDHT instantiates a
``DHTRequestRouter``, wraps it in a ``DHTListener``, and starts the
listener on the same asyncio loop a ``DHTLoopRunner`` provides. Peer
nodes connect via TCP, send a length-prefixed DHT request, and receive
a length-prefixed response.

Wire framing
------------
4-byte big-endian length prefix, then ``request_bytes``. Response
mirrors the framing. Single request per connection, then the server
closes — same shape as :class:`SyncDHTTransport.send`. Connection
pooling is a deliberate non-goal; one-shot RPCs match the existing
ManifestDHT / EmbeddingDHT message-per-RPC semantics and avoid
per-connection state machines.

Backpressure / DoS
------------------
A configurable ``max_request_bytes`` ceiling rejects oversized requests
before allocating buffer for them — defends against memory-exhaustion
attacks via crafted length prefixes. ``max_concurrent_connections``
caps the number of in-flight handlers so a flood of slow / never-
completing connections cannot exhaust the loop's task budget. Both
limits log the rejection at WARNING so operators can detect attacks.

Loop-blocking work
------------------
``router.handle(request_bytes)`` is sync and may perform potentially-
blocking work (disk read inside ManifestDHTServer for manifest fetch,
embedding-DHT signature verification). We run ``handle`` inside
``loop.run_in_executor`` so a slow handler can't stall other peers'
requests on the same loop — addresses the listener-concurrency risk
called out in the plan doc.
"""
from __future__ import annotations

import asyncio
import logging
import struct
from typing import Optional

from prsm.network.dht_router import DHTRequestRouter

logger = logging.getLogger(__name__)


# Defaults match the SyncDHTTransport ceiling so client and server
# share one wire-frame budget. Both DHTs cap MAX_MESSAGE_BYTES at
# 256 KiB; 1 MiB leaves slack for protocol overhead and future framing
# fields without exposing the listener to memory exhaustion via crafted
# length prefixes.
DEFAULT_MAX_REQUEST_BYTES = 1 << 20

# Cap concurrent connections so a peer-side flood can't exhaust the
# loop's task budget. 1024 is high enough for legitimate DHT traffic
# (tens of peers × ~k-bucket RPCs/second per peer) and low enough to
# clip a runaway attacker without OOM'ing the node.
DEFAULT_MAX_CONCURRENT_CONNECTIONS = 1024

# Per-request wall-clock budget for the read-handle-write cycle.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0

_LENGTH_PREFIX_BYTES = 4
_LENGTH_PREFIX_FORMAT = ">I"


class DHTListener:
    """asyncio TCP listener that serves DHT requests via a router.

    :param router: A :class:`DHTRequestRouter` that will dispatch each
        request to the correct DHT server.
    :param port: TCP port to bind. Pass 0 to let the kernel assign one
        — actual port is then available via :attr:`port` after start().
    :param host: Bind address. Defaults to ``"0.0.0.0"`` so the
        listener accepts connections from any interface; tests
        typically pass ``"127.0.0.1"``.
    :param max_request_bytes: Reject any request whose length-prefix
        exceeds this value. Defaults to 1 MiB (matches
        SyncDHTTransport's response cap so client and server share
        one frame budget).
    :param max_concurrent_connections: Cap on in-flight per-connection
        handlers. New connections beyond this limit are accepted then
        immediately closed; the limit is logged at WARNING.
    :param request_timeout: Wall-clock budget for one full request
        (read + dispatch + write). A handler that exceeds this is
        cancelled and the connection closed.
    """

    def __init__(
        self,
        router: DHTRequestRouter,
        port: int,
        *,
        host: str = "0.0.0.0",
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_concurrent_connections: int = (
            DEFAULT_MAX_CONCURRENT_CONNECTIONS
        ),
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if router is None:
            raise ValueError("router must not be None")
        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(
                f"port must be 0..65535, got {port!r}"
            )
        if not isinstance(host, str) or not host:
            raise ValueError(f"host must be non-empty str, got {host!r}")
        if max_request_bytes <= 0:
            raise ValueError(
                f"max_request_bytes must be positive; got "
                f"{max_request_bytes}"
            )
        if max_concurrent_connections <= 0:
            raise ValueError(
                f"max_concurrent_connections must be positive; got "
                f"{max_concurrent_connections}"
            )
        if request_timeout <= 0:
            raise ValueError(
                f"request_timeout must be positive; got {request_timeout}"
            )

        self._router = router
        self._configured_host = host
        self._configured_port = port
        self._max_request_bytes = max_request_bytes
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent_connections = max_concurrent_connections
        self._request_timeout = request_timeout

        self._server: Optional[asyncio.AbstractServer] = None
        self._actual_host: Optional[str] = None
        self._actual_port: Optional[int] = None

    # ── public API ────────────────────────────────────────────────

    @property
    def host(self) -> Optional[str]:
        """The bound host. ``None`` until :meth:`start` has run."""
        return self._actual_host

    @property
    def port(self) -> Optional[int]:
        """The bound port. ``None`` until :meth:`start` has run.

        When ``port=0`` was passed to the constructor, this returns
        the kernel-assigned port after start().
        """
        return self._actual_port

    @property
    def is_running(self) -> bool:
        """True between successful start() and stop()."""
        return self._server is not None and self._server.is_serving()

    async def start(self) -> None:
        """Bind the listener and start serving. Idempotent: a second
        start() while already running is a no-op."""
        if self._server is not None and self._server.is_serving():
            return
        # Construct the semaphore lazily so it's bound to the loop
        # currently running. If the listener is restarted on a
        # different loop the semaphore will be reconstructed.
        self._semaphore = asyncio.Semaphore(
            self._max_concurrent_connections,
        )
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._configured_host,
            port=self._configured_port,
        )
        sockets = self._server.sockets
        if not sockets:
            raise RuntimeError(
                "asyncio.start_server returned no sockets — cannot "
                "determine bound (host, port)"
            )
        sockname = sockets[0].getsockname()
        self._actual_host = sockname[0]
        self._actual_port = sockname[1]
        logger.info(
            "DHTListener bound %s:%d (max_request_bytes=%d, "
            "max_concurrent_connections=%d, request_timeout=%.1fs)",
            self._actual_host, self._actual_port,
            self._max_request_bytes,
            self._max_concurrent_connections,
            self._request_timeout,
        )

    async def stop(self) -> None:
        """Stop the listener; idempotent. Lets in-flight handlers run
        to completion (subject to their per-request timeout)."""
        if self._server is None:
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception:  # noqa: BLE001
            logger.exception(
                "DHTListener: error during server.wait_closed()",
            )
        self._server = None
        self._actual_host = None
        self._actual_port = None
        self._semaphore = None

    # ── per-connection handler ────────────────────────────────────

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """One connection = one length-prefixed request, one length-
        prefixed response, then close."""
        peer = writer.get_extra_info("peername") or "<unknown>"
        sem = self._semaphore
        if sem is None:
            # Listener is shutting down between accept and dispatch.
            await self._close(writer)
            return
        if sem.locked():
            logger.warning(
                "DHTListener: rejecting connection from %s — "
                "max_concurrent_connections=%d reached",
                peer, self._max_concurrent_connections,
            )
            await self._close(writer)
            return
        async with sem:
            try:
                await asyncio.wait_for(
                    self._serve_one_request(reader, writer, peer),
                    timeout=self._request_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "DHTListener: request from %s exceeded "
                    "request_timeout=%.1fs; closing",
                    peer, self._request_timeout,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "DHTListener: unexpected error serving %s",
                    peer,
                )
            finally:
                await self._close(writer)

    async def _serve_one_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer,
    ) -> None:
        """Read one framed request, dispatch via router, write framed
        response. Network errors are logged at DEBUG (peer churn is
        normal); router-level malformed requests get a structured
        error envelope from the router itself, which we forward."""
        try:
            prefix = await reader.readexactly(_LENGTH_PREFIX_BYTES)
        except asyncio.IncompleteReadError:
            logger.debug(
                "DHTListener: peer %s closed before sending length "
                "prefix",
                peer,
            )
            return
        (req_len,) = struct.unpack(_LENGTH_PREFIX_FORMAT, prefix)
        if req_len == 0:
            logger.debug(
                "DHTListener: peer %s sent zero-length request — "
                "router will reject", peer,
            )
        if req_len > self._max_request_bytes:
            logger.warning(
                "DHTListener: peer %s announced req_len=%d > "
                "max_request_bytes=%d; refusing without read",
                peer, req_len, self._max_request_bytes,
            )
            return
        try:
            request_bytes = (
                await reader.readexactly(req_len) if req_len else b""
            )
        except asyncio.IncompleteReadError as exc:
            logger.debug(
                "DHTListener: peer %s sent incomplete payload "
                "(expected %d, got %d): %s",
                peer, req_len, len(exc.partial), exc,
            )
            return

        # Run the sync router.handle() in the executor so a slow
        # downstream server (disk read, signature verify) doesn't stall
        # other peers' requests on the same loop. Router is documented
        # never-raises, but we defend in depth: if the executor returns
        # an exception, we log and close.
        loop = asyncio.get_running_loop()
        try:
            response_bytes = await loop.run_in_executor(
                None, self._router.handle, request_bytes,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "DHTListener: router.handle raised for peer %s — "
                "closing without response",
                peer,
            )
            return

        if not isinstance(response_bytes, (bytes, bytearray)):
            logger.error(
                "DHTListener: router.handle returned non-bytes (%s); "
                "this indicates a regression — closing without response",
                type(response_bytes).__name__,
            )
            return

        framed = (
            struct.pack(
                _LENGTH_PREFIX_FORMAT, len(response_bytes),
            )
            + bytes(response_bytes)
        )
        try:
            writer.write(framed)
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError) as exc:
            logger.debug(
                "DHTListener: peer %s closed before response drain: %s",
                peer, exc,
            )

    @staticmethod
    async def _close(writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            # Closing a closed/broken writer is normal during peer
            # churn; downgrade to debug rather than burning a stack.
            logger.debug(
                "DHTListener: ignoring error during writer.close()",
                exc_info=True,
            )


__all__ = [
    "DEFAULT_MAX_CONCURRENT_CONNECTIONS",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DHTListener",
]
