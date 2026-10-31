"""Planner plan tokens: signed, expiring, bound to key, settings and principal, and free of configured paths."""

import base64
import json
import re
from pathlib import Path
from typing import Any

import pytest

from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import is_error
from mloda.interfaces.service.planner import PLAN_TOKEN_PREFIX, Planner
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_interfaces.source_groups import FILE_SOURCE_GROUPS


CLOCK_START = 1_800_000_000.0
KEY = b"k" * 32
REQUEST = {"features": ["customer_id", "income"]}
_URLSAFE = re.compile(r"[A-Za-z0-9_-]+")


class _Clock:
    def __init__(self, now: float = CLOCK_START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    path = tmp_path / "sales.csv"
    path.write_text("customer_id,income,segment\n1,10,a\n2,20,b\n3,30,a\n4,40,b\n5,50,a\n")
    sources = {"sales": SourceConfig(alias="sales", files=(str(path),))}
    return Settings(compute_frameworks=(PythonDictFramework.get_class_name(),), sources=sources, **kwargs)


def _planner(
    settings: Settings,
    *,
    key: bytes | None = KEY,
    principal: str | None = "alice",
    clock: _Clock | None = None,
) -> Planner:
    base = PluginCollector.enabled_feature_groups(set(FILE_SOURCE_GROUPS))
    return Planner(
        settings,
        Catalog.build(settings, base),
        signing_key=key,
        principal=principal,
        plugin_collector=base,
        clock=clock or _Clock(),
    )


def _plan_id(planner: Planner, request: dict[str, Any] = REQUEST) -> str:
    result = planner.explain(request)
    assert not is_error(result), result
    plan_id: str = result["plan_id"]
    return plan_id


def _assert_invalid(result: dict[str, Any]) -> dict[str, Any]:
    assert is_error(result), result
    error: dict[str, Any] = result["error"]
    assert error["code"] == "plan_invalid"
    json.dumps(result)
    return error


def _flip(token: str, index: int) -> str:
    while token[index] == ".":
        index += 1
    replacement = "A" if token[index] != "A" else "B"
    return token[:index] + replacement + token[index + 1 :]


def _decoded_segments(token: str) -> list[bytes]:
    decoded = []
    for segment in token.split("."):
        if _URLSAFE.fullmatch(segment) and len(segment) % 4 != 1:
            decoded.append(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    return decoded


class TestRoundTrip:
    def test_verify_returns_normalized_request_and_digest(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))
        result = planner.explain(REQUEST)
        assert not is_error(result), result

        verified = planner.verify_plan_id(result["plan_id"])

        assert not is_error(verified), verified
        assert set(verified) == {"request", "digest"}
        assert verified["request"] == result["normalized_request"]
        assert isinstance(verified["digest"], str)
        assert verified["digest"]

    def test_token_prefix(self, tmp_path: Path) -> None:
        assert PLAN_TOKEN_PREFIX == "mpl1"  # nosec B105
        assert _plan_id(_planner(_settings(tmp_path))).startswith("mpl1.")

    def test_same_request_same_digest(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))

        first = planner.verify_plan_id(_plan_id(planner))
        second = planner.verify_plan_id(_plan_id(planner))

        assert first["digest"] == second["digest"]

    def test_different_request_different_digest(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))

        one = planner.verify_plan_id(_plan_id(planner, {"features": ["customer_id"]}))
        other = planner.verify_plan_id(_plan_id(planner, {"features": ["customer_id", "income"]}))

        assert one["digest"] != other["digest"]

    def test_default_signing_key_round_trips(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path), key=None)

        verified = planner.verify_plan_id(_plan_id(planner))

        assert not is_error(verified), verified
        assert verified["request"] == REQUEST

    def test_token_does_not_carry_configured_paths(self, tmp_path: Path) -> None:
        token = _plan_id(_planner(_settings(tmp_path)))

        assert str(tmp_path) not in token
        for segment in _decoded_segments(token):
            assert str(tmp_path).encode() not in segment
            assert b"sales.csv" not in segment


class TestTampering:
    def test_flipped_payload_character_is_invalid(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))
        token = _plan_id(planner)

        _assert_invalid(planner.verify_plan_id(_flip(token, len(PLAN_TOKEN_PREFIX) + 3)))

    def test_flipped_signature_character_is_invalid(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))
        token = _plan_id(planner)

        _assert_invalid(planner.verify_plan_id(_flip(token, len(token) - 5)))

    def test_other_prefix_is_invalid(self, tmp_path: Path) -> None:
        planner = _planner(_settings(tmp_path))
        token = _plan_id(planner)

        _assert_invalid(planner.verify_plan_id("mpl2" + token[len(PLAN_TOKEN_PREFIX) :]))

    @pytest.mark.parametrize(
        "token",
        ["", "mpl1", "mpl1.", "mpl1..", "nope", "mpl1.abc.def", "mpl1." + "A" * 40, "mpl1.%%%.###", "a.b.c.d.e"],
    )
    def test_garbage_string_is_invalid(self, tmp_path: Path, token: str) -> None:
        _assert_invalid(_planner(_settings(tmp_path)).verify_plan_id(token))

    @pytest.mark.parametrize("token", [None, 1, 1.5, True, b"mpl1.x", ["mpl1."], {"plan_id": "mpl1."}])
    def test_non_string_is_invalid(self, tmp_path: Path, token: Any) -> None:
        _assert_invalid(_planner(_settings(tmp_path)).verify_plan_id(token))


class TestExpiry:
    def test_valid_until_ttl(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        clock = _Clock()
        planner = _planner(settings, clock=clock)
        token = _plan_id(planner)

        clock.now = CLOCK_START + settings.plan_ttl_seconds - 1

        assert not is_error(planner.verify_plan_id(token))

    def test_expired_token_is_invalid(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        clock = _Clock()
        planner = _planner(settings, clock=clock)
        token = _plan_id(planner)

        clock.now = CLOCK_START + settings.plan_ttl_seconds + 1

        error = _assert_invalid(planner.verify_plan_id(token))
        assert "expired" in error["message"].lower()


class TestBinding:
    def test_other_signing_key_rejects(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        token = _plan_id(_planner(settings))

        _assert_invalid(_planner(settings, key=b"j" * 32).verify_plan_id(token))

    def test_other_settings_reject(self, tmp_path: Path) -> None:
        token = _plan_id(_planner(_settings(tmp_path)))

        _assert_invalid(_planner(_settings(tmp_path, max_rows=1000)).verify_plan_id(token))

    def test_other_principal_rejects(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        token = _plan_id(_planner(settings))

        _assert_invalid(_planner(settings, principal="bob").verify_plan_id(token))

    def test_missing_principal_rejects_a_principal_token(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        token = _plan_id(_planner(settings))

        _assert_invalid(_planner(settings, principal=None).verify_plan_id(token))

    def test_same_key_settings_and_principal_accept(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        token = _plan_id(_planner(settings))

        verified = _planner(settings).verify_plan_id(token)

        assert not is_error(verified), verified
        assert verified["request"] == REQUEST
