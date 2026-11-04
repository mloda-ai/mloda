"""Tests for the loopback stand-in and no-network env preset (issue #2373)."""

from __future__ import annotations

import os
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from mloda._testing import fixtures as fixtures_module
from mloda._testing.loopback import LoopbackServer, no_network_env


def _post(url: str, body: bytes) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def test_loopback_server_records_request_bodies_and_answers() -> None:
    with LoopbackServer() as server:
        status, payload = _post(f"{server.url}/capture", b'{"event": "install"}')
        assert status == 200
        assert payload == b"{}"
        with urllib.request.urlopen(f"{server.url}/health", timeout=5) as response:
            assert response.status == 200

    assert [request.method for request in server.requests] == ["POST", "GET"]
    assert server.requests[0].path == "/capture"
    assert server.requests[0].json() == {"event": "install"}
    assert server.bodies == [b'{"event": "install"}', b""]


def test_loopback_server_custom_response_and_close() -> None:
    server = LoopbackServer(response_body=b'{"ok": true}', response_status=202)
    status, payload = _post(f"{server.url}/x", b"body")
    assert (status, payload) == (202, b'{"ok": true}')
    url = server.url
    server.close()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url, timeout=5)


def test_no_network_env_clears_proxies_and_sets_do_not_track() -> None:
    base = {"HTTP_PROXY": "http://proxy:8080", "https_proxy": "http://proxy:8080", "PATH": "/bin"}

    env = no_network_env(base)

    assert "HTTP_PROXY" not in env and "https_proxy" not in env
    assert env["DO_NOT_TRACK"] == "1"
    assert "127.0.0.1" in env["NO_PROXY"] and "localhost" in env["no_proxy"]
    assert env["PATH"] == "/bin"
    assert base["HTTP_PROXY"] == "http://proxy:8080"  # the input mapping is not mutated


def test_no_network_env_keeps_existing_no_proxy_hosts() -> None:
    env = no_network_env({"NO_PROXY": "internal.example.com"})

    assert env["NO_PROXY"].split(",")[0] == "internal.example.com"
    assert "127.0.0.1" in env["NO_PROXY"]


def test_fixtures_are_pytest_fixtures() -> None:
    for fixture in (fixtures_module.loopback_server, fixtures_module.no_network_env):
        assert hasattr(fixture, "_fixture_function_marker") or hasattr(fixture, "_pytestfixturefunction")
        assert hasattr(fixture, "__wrapped__")


def test_loopback_fixture_yields_a_working_server() -> None:
    gen = fixtures_module.loopback_server.__wrapped__()
    server = next(gen)
    status, _ = _post(f"{server.url}/via-fixture", b"ping")
    assert status == 200
    assert server.bodies == [b"ping"]
    next(gen, None)


def test_pyproject_declares_testing_pytest11_entry_point() -> None:
    with open(Path(__file__).parents[2] / "pyproject.toml", "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    assert data["project"]["entry-points"]["pytest11"].get("mloda-testing") == "mloda._testing.fixtures"


def test_no_network_fixture_applies_the_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    env = fixtures_module.no_network_env.__wrapped__(monkeypatch)
    assert "HTTP_PROXY" not in os.environ
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert env["DO_NOT_TRACK"] == "1"
