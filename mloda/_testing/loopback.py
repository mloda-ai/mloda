"""Loopback HTTP stand-in and no-network env preset for plugin tests.

Plugins that send anything over the network (usage telemetry, a license check, a remote sink)
otherwise each grow their own throwaway server and env juggling to keep tests off the real
backend. This module is the shared version: ``LoopbackServer`` records what a plugin sends to a
local stand-in, and ``no_network_env`` builds the env preset that keeps subprocesses offline.
Both are also exposed as pytest fixtures in ``mloda._testing.fixtures``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


@dataclass(frozen=True)
class RecordedRequest:
    """One HTTP request the stand-in received."""

    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        """The request body parsed as JSON."""
        return json.loads(self.body)


class LoopbackServer:
    """A throwaway HTTP server on 127.0.0.1 that records every request it receives.

    Point the plugin under test at ``url`` (for example via its host/base-url setting), let it
    send, then assert on ``requests``. Every request is answered with ``response_status`` and
    ``response_body``. Use as a context manager, or call ``close()``.
    """

    def __init__(self, response_body: bytes = b"{}", response_status: int = 200) -> None:
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        requests = self.requests
        lock = self._lock

        class _Handler(BaseHTTPRequestHandler):
            def _record_and_respond(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with lock:
                    requests.append(RecordedRequest(self.command, self.path, dict(self.headers.items()), body))
                self.send_response(response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            do_GET = _record_and_respond
            do_POST = _record_and_respond
            do_PUT = _record_and_respond
            do_PATCH = _record_and_respond
            do_DELETE = _record_and_respond
            do_HEAD = _record_and_respond
            do_OPTIONS = _record_and_respond

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """The base URL of the stand-in, ``http://127.0.0.1:<port>``."""
        return f"http://127.0.0.1:{self._port}"

    @property
    def bodies(self) -> list[bytes]:
        """The recorded request bodies, in arrival order."""
        return [request.body for request in self.requests]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def no_network_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """An environment that keeps a subprocess off the real network backends.

    Returns a copy of ``base`` (default: ``os.environ``) with the proxy variables removed,
    ``DO_NOT_TRACK=1`` set — the opt-out convention telemetry libraries (PostHog and friends)
    honour — and ``NO_PROXY``/``no_proxy`` extended with the loopback hosts, so a
    ``LoopbackServer`` stand-in stays reachable directly. Pass the result as ``env=`` to
    ``subprocess``; it does not mutate ``base`` or ``os.environ``.
    """
    env = dict(os.environ if base is None else base)
    for var in _PROXY_VARS:
        env.pop(var, None)
    env["DO_NOT_TRACK"] = "1"
    hosts = [host for host in env.get("NO_PROXY", "").split(",") if host]
    hosts += [host for host in _LOOPBACK_HOSTS if host not in hosts]
    env["NO_PROXY"] = env["no_proxy"] = ",".join(hosts)
    return env
