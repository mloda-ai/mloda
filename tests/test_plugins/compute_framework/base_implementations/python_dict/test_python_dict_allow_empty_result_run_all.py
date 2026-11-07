"""End-to-end empty-result policy test for PythonDict.

Consumes the shipped EmptyResultRunAllSuite. A zero-row column-bearing result succeeds on every backend; a
zero-column ``{}`` result raises ``EmptyResultError``. THREADING and MULTIPROCESSING get module tests below.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from mloda._testing.compute_framework.run_all_suite import (
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    key_read,
)
from mloda._testing.framework_adapter import PythonDictAdapter
from mloda._testing.source import reclaim
from mloda.provider import DataCreatorFeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda


class TestPythonDictAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on PythonDict."""

    adapter_class = PythonDictAdapter


@pytest.fixture
def _built_classes() -> Iterator[None]:
    yield from reclaim.reclaimed_scope()


def _zero_row_collector(feature_name: str) -> PluginCollector:
    class _ZeroRowFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {feature_name: []}

    return PluginCollector.enabled_feature_groups({reclaim.track(_ZeroRowFeatureGroup)})


def _row_count(frame: Any) -> int:
    """Tolerant of the wire format: a PyArrow table or a columnar dict."""
    return int(frame.num_rows) if hasattr(frame, "num_rows") else len(next(iter(frame.values()), []))


@pytest.mark.usefixtures("_built_classes")
def test_empty_result_default_succeeds_threading(flight_server: Any) -> None:
    """A schema-bearing zero-row default result survives the THREADING worker channel."""
    result = mloda.run_all(
        [Feature(name="empty_result_default_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_zero_row_collector("empty_result_default_col"),
        parallelization_modes={ParallelizationMode.THREADING},
        flight_server=flight_server,
    )

    assert len(result) == 1
    assert _row_count(result[0]) == 0


class _ZeroRowAllowedFeatureGroup(DataCreatorFeatureGroup):
    """Module level: multiprocessing must pickle the class by path."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_allowed_col"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"empty_result_allowed_col": []}


def test_empty_result_allowed_succeeds_multiprocessing(flight_server: Any) -> None:
    """The empty success path survives the flight-server upload in MULTIPROCESSING."""
    result = mloda.run_all(
        [Feature(name="empty_result_allowed_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=PluginCollector.enabled_feature_groups({_ZeroRowAllowedFeatureGroup}),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert len(result) == 1
    assert _row_count(result[0]) == 0


class EmptyResultNoneAllowedFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` returns ``None`` (state A).

    Module-local on purpose: it pins that a MISSING result is rejected upstream in
    ``transform`` (never legitimized as an empty result).
    """

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_none_allowed_col"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # No result at all -> state A. The opt-in must NOT legitimize this.
        return None


_ENABLED_NONE_ALLOWED = PluginCollector.enabled_feature_groups({EmptyResultNoneAllowedFeatureGroup})


def test_empty_result_none_raises_python_dict(flight_server: Any) -> None:
    """State A live test: a ``None`` result raises on PythonDict.

    A MISSING result is not an empty one. PythonDict's ``transform`` must reject ``None``
    like every schema-bearing framework does (``ValueError: Data type <class 'NoneType'>
    is not supported by PythonDictFramework``) instead of converting it to ``{}`` and
    turning the run into a silent success. ``run_all`` re-wraps the framework error as a
    plain ``Exception`` whose message embeds that text (repr-escaped, so the inner quotes
    around ``NoneType`` carry backslashes), which the ``match`` below pins.
    """
    feature = Feature(name="empty_result_none_allowed_col")

    with pytest.raises(Exception, match=r"Data type <class \\?'NoneType\\?'> is not supported by PythonDictFramework"):
        mloda.run_all(
            [feature],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_NONE_ALLOWED,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )


class TestPythonDictSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all."""

    adapter_class = PythonDictAdapter


class TestPythonDictMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on PythonDict raise an explanatory error."""

    adapter_class = PythonDictAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {"KeyReadConsumer": (key_read, lambda: KeyError)}
