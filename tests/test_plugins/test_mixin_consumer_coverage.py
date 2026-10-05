"""Every concrete mloda_plugins implementation of a covered family must have a consumer of its contract mixin.

Plugins come from PluginLoader, which skips modules missing an optional dependency. Consumers come from listed modules
imported explicitly (no collection-order dependence); one skipped for a missing dependency still counts."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.provider import BaseFilterEngine
from mloda.provider import BaseMaskEngine
from mloda.user import PluginLoader
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import FilterEngineTestMixin
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import MaskEngineTestMixin
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.aggregated_zero_row_test_mixin import (
    AggregatedZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_missing_value_feature_group.missing_value_zero_row_test_mixin import (
    MissingValueZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.time_window_zero_row_test_mixin import (
    TimeWindowZeroRowTestMixin,
)

_EXPERIMENTAL = "tests.test_plugins.feature_group.experimental"
_FRAMEWORKS = "tests.test_plugins.compute_framework.base_implementations"


@dataclass(frozen=True)
class MixinContract:
    """A plugin base, the mixin its implementations must join, and where the consumers live."""

    base: type[Any]
    mixin: type[Any]
    attribute: str
    consumer_modules: tuple[str, ...]


CONTRACTS: tuple[MixinContract, ...] = (
    MixinContract(
        AggregatedFeatureGroup,
        AggregatedZeroRowTestMixin,
        "feature_group_class",
        (
            f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_aggregated_feature_group",
            f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_polars_lazy_aggregated_feature_group",
            f"{_EXPERIMENTAL}.test_base_aggregated_feature_group.test_pyarrow_aggregated_feature_group",
        ),
    ),
    MixinContract(
        TimeWindowFeatureGroup,
        TimeWindowZeroRowTestMixin,
        "feature_group_class",
        (
            f"{_EXPERIMENTAL}.test_time_window_feature_group.test_pandas_time_window_feature_group",
            f"{_EXPERIMENTAL}.test_time_window_feature_group.test_pyarrow_time_window_feature_group",
        ),
    ),
    MixinContract(
        MissingValueFeatureGroup,
        MissingValueZeroRowTestMixin,
        "feature_group_class",
        (
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_pandas_missing_value_feature_group",
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_pyarrow_missing_value_feature_group",
            f"{_EXPERIMENTAL}.test_missing_value_feature_group.test_python_dict_missing_value_feature_group",
        ),
    ),
    MixinContract(
        BaseMaskEngine,
        MaskEngineTestMixin,
        "mask_engine_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_mask_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_mask_engine",
            f"{_FRAMEWORKS}.polars.test_polars_expr_mask_engine",
            f"{_FRAMEWORKS}.polars.test_polars_mask_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_mask_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_mask_engine",
            f"{_FRAMEWORKS}.spark.test_spark_mask_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_mask_engine",
        ),
    ),
    MixinContract(
        BaseFilterEngine,
        FilterEngineTestMixin,
        "filter_engine_class",
        (
            f"{_FRAMEWORKS}.duckdb.test_duckdb_filter_engine",
            f"{_FRAMEWORKS}.iceberg.test_iceberg_filter_engine",
            f"{_FRAMEWORKS}.pandas.test_pandas_filter_engine",
            f"{_FRAMEWORKS}.polars.test_polars_filter_engine",
            f"{_FRAMEWORKS}.pyarrow.test_pyarrow_filter_engine",
            f"{_FRAMEWORKS}.python_dict.test_python_dict_filter_engine",
            f"{_FRAMEWORKS}.spark.test_spark_filter_engine",
            f"{_FRAMEWORKS}.sqlite.test_sqlite_filter_engine",
        ),
    ),
)


def _qualified_name(cls: type[Any]) -> str:
    """Compared by name, not identity: a reloaded module leaves a stale duplicate class behind."""
    return f"{cls.__module__}.{cls.__qualname__}"


def concrete_plugin_classes(base: type[Any]) -> set[str]:
    """Concrete subclasses of base defined under mloda_plugins; test-local subclasses fall out by module."""
    PluginLoader.all()
    return {
        _qualified_name(cls)
        for cls in get_all_subclasses(base)
        if cls.__module__.startswith("mloda_plugins.") and not inspect.isabstract(cls)
    }


def covered_classes(contract: MixinContract) -> set[str]:
    """Classes named by the contract attribute of the mixin consumers defined in the listed modules."""
    modules = {importlib.import_module(name).__name__ for name in contract.consumer_modules}
    return {
        _qualified_name(getattr(consumer, contract.attribute))
        for consumer in get_all_subclasses(contract.mixin)
        if consumer.__module__ in modules and hasattr(consumer, contract.attribute)
    }


# The PluginLoader import costs a few seconds cold, hence the raised timeout.
@pytest.mark.timeout(30)
@pytest.mark.parametrize("contract", CONTRACTS, ids=[contract.mixin.__name__ for contract in CONTRACTS])
def test_every_concrete_implementation_has_a_mixin_consumer(contract: MixinContract) -> None:
    implementations = concrete_plugin_classes(contract.base)
    assert implementations, f"no concrete {contract.base.__name__} found under mloda_plugins: the plugin import broke"
    uncovered = sorted(implementations - covered_classes(contract))
    assert uncovered == [], (
        f"{contract.base.__name__} implementations without a {contract.mixin.__name__} consumer: {uncovered}. "
        f"Add a test class inheriting {contract.mixin.__name__} with {contract.attribute} set to the class, "
        f"in a module listed in its MixinContract in {__name__}."
    )
