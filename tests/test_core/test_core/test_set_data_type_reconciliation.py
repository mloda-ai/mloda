"""Tests for ``Engine.set_data_type`` reconciliation against the new
``return_data_type_rule`` contract.

The rule returns a ``DataTypeDeclaration`` (``DataType | None | Deferred``) instead of a
raw ``Optional[DataType]``. ``set_data_type`` keeps returning ``Optional[DataType]``
and reconciles the rule outcome against any declared ``feature.data_type``. The engine
is fail-fast: it does NOT swallow exceptions raised by a rule. A rule that raises must
let the exception propagate so planning fails loudly. ``None`` is the explicit
"no fixed type" signal and ``Deferred`` is the compute-time sentinel.

``set_data_type`` uses no instance state, so we bypass the heavy ``Engine.__init__``
via ``Engine.__new__`` and call the method directly.
"""

import pytest

from mloda.core.abstract_plugins.components.data_type_rule import (
    DataTypeDeclaration,
    Deferred,
)
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine
from mloda.user import DataType
from mloda.user import Feature


def _engine() -> Engine:
    """Engine instance without running ``__init__`` (set_data_type uses no self)."""
    return Engine.__new__(Engine)


class _FixedInt64(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        return DataType.INT64


class _OpenRule(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        return None


class _DeferredRule(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        return Deferred()


class _RaisesValueError(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        raise ValueError("boom-data-shape")


class _RaisesAttributeError(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        raise AttributeError("boom-programmer-error")


class _RaisesKeyError(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataTypeDeclaration:
        raise KeyError("boom-missing-key")


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


def test_rule_raises_value_error_no_declared_propagates() -> None:
    feature = Feature("reconcile_raise_value_undeclared")
    with pytest.raises(ValueError):
        _engine().set_data_type(feature, _RaisesValueError)


def test_rule_raises_attribute_error_no_declared_propagates() -> None:
    feature = Feature("reconcile_raise_attr_undeclared")
    with pytest.raises(AttributeError):
        _engine().set_data_type(feature, _RaisesAttributeError)


def test_rule_raises_while_declared_propagates_raised_exception() -> None:
    # A distinct exception type so the propagated error cannot be confused with
    # the declared/fixed mismatch path (which raises ValueError).
    feature = Feature("reconcile_raise_key_declared", data_type=DataType.STRING)
    with pytest.raises(KeyError):
        _engine().set_data_type(feature, _RaisesKeyError)
