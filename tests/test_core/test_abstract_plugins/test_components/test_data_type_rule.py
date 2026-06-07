"""Tests for the data_type_rule tagged union (RuleResult / RuleOutcome).

These tests define the expected behavior of the module
``mloda.core.abstract_plugins.components.data_type_rule``, which carries the
result of ``FeatureGroup.return_data_type_rule``.
"""

from dataclasses import FrozenInstanceError

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.data_type_rule import (
    Broken,
    Deferred,
    Open,
    RuleResult,
)


def test_variants_are_constructible() -> None:
    assert DataType.INT64 is not None
    assert Open() is not None
    assert Deferred() is not None
    assert Broken(ValueError("x")) is not None


def test_broken_is_frozen() -> None:
    broken = Broken(ValueError("x"))
    with pytest.raises(FrozenInstanceError):
        broken.error = ValueError("y")  # type: ignore[misc]


def test_open_and_deferred_equality() -> None:
    assert Open() == Open()
    assert Deferred() == Deferred()
    open_result: RuleResult = Open()
    deferred_result: RuleResult = Deferred()
    assert open_result != deferred_result


def test_broken_carries_exact_exception() -> None:
    exc = ValueError("x")
    assert Broken(exc).error is exc
