"""Tests for ``Engine.set_data_type`` reconciliation against the new
``return_data_type_rule`` contract.

The rule now returns a ``RuleResult`` (``Fixed | Open | Deferred``) instead of a
raw ``Optional[DataType]``. ``set_data_type`` keeps returning ``Optional[DataType]``
but reconciles the rule outcome against any declared ``feature.data_type`` and must
never let a raising rule crash planning.

``set_data_type`` uses no instance state, so we bypass the heavy ``Engine.__init__``
via ``Engine.__new__`` and call the method directly.
"""

import logging

import pytest

from mloda.core.abstract_plugins.components.data_type_rule import (
    Deferred,
    Fixed,
    Open,
    RuleResult,
)
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine
from mloda.user import DataType
from mloda.user import Feature

ENGINE_LOGGER = "mloda.core.core.engine"


def _engine() -> Engine:
    """Engine instance without running ``__init__`` (set_data_type uses no self)."""
    return Engine.__new__(Engine)


class _FixedInt64(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> RuleResult:
        return Fixed(DataType.INT64)


class _OpenRule(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> RuleResult:
        return Open()


class _DeferredRule(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> RuleResult:
        return Deferred()


class _RaisesValueError(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> RuleResult:
        raise ValueError("boom-data-shape")


class _RaisesAttributeError(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> RuleResult:
        raise AttributeError("boom-programmer-error")


def test_fixed_with_no_declared_type_returns_fixed_type() -> None:
    feature = Feature("reconcile_fixed_undeclared")
    result = _engine().set_data_type(feature, _FixedInt64)
    assert result == DataType.INT64


def test_fixed_with_matching_declared_type_returns_type() -> None:
    feature = Feature("reconcile_fixed_match", data_type=DataType.INT64)
    result = _engine().set_data_type(feature, _FixedInt64)
    assert result == DataType.INT64


def test_fixed_with_conflicting_declared_type_raises() -> None:
    feature = Feature("reconcile_fixed_mismatch", data_type=DataType.STRING)
    with pytest.raises(ValueError):
        _engine().set_data_type(feature, _FixedInt64)


def test_open_with_declared_type_returns_declared() -> None:
    feature = Feature("reconcile_open_declared", data_type=DataType.STRING)
    result = _engine().set_data_type(feature, _OpenRule)
    assert result == DataType.STRING


def test_open_with_no_declared_type_returns_none() -> None:
    feature = Feature("reconcile_open_undeclared")
    result = _engine().set_data_type(feature, _OpenRule)
    assert result is None


def test_deferred_with_no_declared_type_returns_none() -> None:
    feature = Feature("reconcile_deferred_undeclared")
    result = _engine().set_data_type(feature, _DeferredRule)
    assert result is None


def test_deferred_with_declared_type_returns_declared() -> None:
    feature = Feature("reconcile_deferred_declared", data_type=DataType.STRING)
    result = _engine().set_data_type(feature, _DeferredRule)
    assert result == DataType.STRING


def test_rule_raises_value_error_no_declared_does_not_crash_and_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    feature = Feature("reconcile_raise_value_undeclared")
    with caplog.at_level(logging.DEBUG, logger=ENGINE_LOGGER):
        result = _engine().set_data_type(feature, _RaisesValueError)

    assert result is None

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "expected a DEBUG record for a data-shape rule error without a declared type"
    assert any("reconcile_raise_value_undeclared" in r.getMessage() for r in debug_records)


def test_rule_raises_attribute_error_no_declared_does_not_crash_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    feature = Feature("reconcile_raise_attr_undeclared")
    with caplog.at_level(logging.DEBUG, logger=ENGINE_LOGGER):
        result = _engine().set_data_type(feature, _RaisesAttributeError)

    assert result is None

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected a WARNING record for a non-data-shape rule error"
    assert any("reconcile_raise_attr_undeclared" in r.getMessage() for r in warning_records)


def test_rule_raises_value_error_while_declared_logs_warning_and_returns_declared(
    caplog: pytest.LogCaptureFixture,
) -> None:
    feature = Feature("reconcile_raise_value_declared", data_type=DataType.STRING)
    with caplog.at_level(logging.DEBUG, logger=ENGINE_LOGGER):
        result = _engine().set_data_type(feature, _RaisesValueError)

    assert result == DataType.STRING

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "raise-while-declared must warn even for data-shape errors (possible silent mismatch)"
    assert any("reconcile_raise_value_declared" in r.getMessage() for r in warning_records)
