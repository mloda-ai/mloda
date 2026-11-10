"""Unit tests for tests/helpers/linear_scaling.py driven by a fake CPU clock."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tests.helpers import linear_scaling
from tests.helpers.linear_scaling import assert_runs_linearly


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    now = [0.0]
    monkeypatch.setattr(linear_scaling, "time", SimpleNamespace(process_time=lambda: now[0]))

    def advance(seconds: float) -> None:
        now[0] += seconds

    return advance


def _build(n: int) -> str:
    return "x" * n


def test_linear_operation_passes_and_returns_large_result(monkeypatch: pytest.MonkeyPatch) -> None:
    advance = _fake_clock(monkeypatch)

    def operation(text: str) -> int:
        advance(len(text) * 1e-5)
        return len(text)

    assert assert_runs_linearly(operation, _build, 4000, "fake op") == 4000


def test_quadratic_operation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    advance = _fake_clock(monkeypatch)

    def operation(text: str) -> str:
        advance(len(text) ** 2 * 1e-8)
        return text

    with pytest.raises(AssertionError, match="operation scales super-linearly on fake op"):
        assert_runs_linearly(operation, _build, 4000, "fake op")


def test_linear_but_slow_operation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    advance = _fake_clock(monkeypatch)

    def operation(text: str) -> str:
        advance(len(text) * 1e-3)
        return text

    with pytest.raises(AssertionError, match="operation is too slow on fake op"):
        assert_runs_linearly(operation, _build, 4000, "fake op")


def test_one_noisy_run_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    advance = _fake_clock(monkeypatch)
    calls = [0]

    def operation(text: str) -> str:
        calls[0] += 1
        noise = 0.5 if calls[0] == 2 else 0.0
        advance(len(text) * 1e-5 + noise)
        return text

    assert assert_runs_linearly(operation, _build, 4000, "fake op") == _build(4000)
