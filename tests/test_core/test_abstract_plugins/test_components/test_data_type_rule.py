"""Tests for the data_type_rule tagged union (DataTypeDeclaration).

These tests define the expected behavior of the module
``mloda.core.abstract_plugins.components.data_type_rule``, which carries the
result of ``FeatureGroup.return_data_type_rule``.
"""

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.data_type_rule import (
    DataTypeDeclaration,
    Deferred,
)


def test_variants_are_constructible() -> None:
    assert DataType.INT64 is not None
    assert Deferred() is not None


def test_none_and_deferred_inequality() -> None:
    assert Deferred() == Deferred()
    none_result: DataTypeDeclaration = None
    deferred_result: DataTypeDeclaration = Deferred()
    assert none_result != deferred_result
