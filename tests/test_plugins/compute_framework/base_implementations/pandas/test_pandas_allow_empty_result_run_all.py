"""End-to-end empty-result policy test for pandas.

Consumes the shipped EmptyResultRunAllSuite. Pandas is a guaranteed dependency, so no import guard is needed.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from mloda._testing.compute_framework.run_all_suite import (
    UNDECLARED_COLUMN,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    attribute_read,
    key_read,
)
from mloda._testing.framework_adapter import PandasAdapter
from mloda._testing.source import reclaim
from mloda.provider import DataCreatorFeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda


def _undefined_variable_error() -> type[BaseException]:
    from pandas.errors import UndefinedVariableError

    cause: type[BaseException] = UndefinedVariableError
    return cause


class TestPandasAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on pandas."""

    adapter_class = PandasAdapter


@pytest.fixture
def _none_collector() -> Iterator[None]:
    yield from reclaim.reclaimed_scope()


def _none_result_collector() -> PluginCollector:
    class _NoneFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"empty_result_none_col"}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return None

    return PluginCollector.enabled_feature_groups({reclaim.track(_NoneFeatureGroup)})


@pytest.mark.usefixtures("_none_collector")
def test_empty_result_none_raises_pandas(flight_server: Any) -> None:
    """A FG whose ``calculate_feature`` returns ``None`` raises end-to-end on pandas."""
    feature = Feature(name="empty_result_none_col")

    with pytest.raises(Exception, match=r"Data <class \\?'NoneType\\?'> is not supported by PandasDataFrame"):
        mloda.run_all(
            [feature],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_none_result_collector(),
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )


class TestPandasSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all."""

    adapter_class = PandasAdapter


def _query_read(data: Any) -> Any:
    return data.query(f"{UNDECLARED_COLUMN} > 0")


class TestPandasMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on pandas raise an explanatory error."""

    adapter_class = PandasAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "KeyReadConsumer": (key_read, lambda: KeyError),
            "PandasQueryReadConsumer": (_query_read, _undefined_variable_error),
            "AttributeReadConsumer": (attribute_read, lambda: AttributeError),
        }
