"""Tests for the data_type_rule tagged union (RuleResult / RuleOutcome).

These tests define the expected behavior of the not-yet-implemented module
``mloda.core.abstract_plugins.components.data_type_rule``, which carries the
result of ``FeatureGroup.return_data_type_rule``.
"""

from dataclasses import FrozenInstanceError

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.data_type_rule import (
    Broken,
    Deferred,
    Fixed,
    Open,
    RuleResult,
)


def test_fixed_carries_data_type() -> None:
    assert Fixed(DataType.STRING).data_type == DataType.STRING


def test_variants_are_constructible() -> None:
    assert Fixed(DataType.INT64) is not None
    assert Open() is not None
    assert Deferred() is not None
    assert Broken(ValueError("x")) is not None


def test_fixed_is_frozen() -> None:
    fixed = Fixed(DataType.INT64)
    with pytest.raises(FrozenInstanceError):
        fixed.data_type = DataType.INT32  # type: ignore[misc]


def test_broken_is_frozen() -> None:
    broken = Broken(ValueError("x"))
    with pytest.raises(FrozenInstanceError):
        broken.error = ValueError("y")  # type: ignore[misc]


def test_fixed_value_equality() -> None:
    assert Fixed(DataType.INT64) == Fixed(DataType.INT64)
    assert Fixed(DataType.INT64) != Fixed(DataType.INT32)


def test_open_and_deferred_equality() -> None:
    assert Open() == Open()
    assert Deferred() == Deferred()
    open_result: RuleResult = Open()
    deferred_result: RuleResult = Deferred()
    assert open_result != deferred_result


def test_broken_carries_exact_exception() -> None:
    exc = ValueError("x")
    assert Broken(exc).error is exc
