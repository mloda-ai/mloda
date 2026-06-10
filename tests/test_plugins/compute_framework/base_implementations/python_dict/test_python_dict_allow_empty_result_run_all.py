"""End-to-end ``allow_empty_result`` policy test for PythonDict.

Consumes the shared EmptyResultRunAllTestBase. PythonDict is selected by name; no
connection is required. The shared empty-producing FeatureGroups emit a columnar dict
with one empty column. Unlike schema-bearing frameworks, ``PythonDictFramework.transform``
collapses ``{"col": []}`` to ``[]``, dropping the schema: the default (not-allowed) FG
therefore yields a schema-less result (state B) and must RAISE, not succeed.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    _ENABLED_ALLOWED,
    _ENABLED_DEFAULT,
    EmptyResultRunAllTestBase,
    _EmptyResultMatchData,
)
from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import records_from_frame


class TestPythonDictAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on PythonDict."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PythonDictFramework"

    @classmethod
    def default_empty_is_schemaless(cls) -> bool:
        # transform collapses {"col": []} to [], so the default empty result has no schema.
        return True


def test_empty_result_default_raises_threading(flight_server: Any) -> None:
    """The EmptyResultError survives the THREADING worker error channel.

    Mirrors the base's default test (state B raises on PythonDict) but runs with
    ``ParallelizationMode.THREADING`` instead of SYNC, pinning that the framework-raised
    error propagates out of the thread worker. As in the base, run_all wraps the error,
    so the assertion targets the type NAME in the wrapped message.
    """
    with pytest.raises(Exception) as excinfo:
        mloda.run_all(
            [Feature(name="empty_result_default_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_DEFAULT,
            parallelization_modes={ParallelizationMode.THREADING},
            flight_server=flight_server,
        )
    assert EmptyResultError.__name__ in str(excinfo.value)


def test_empty_result_allowed_succeeds_multiprocessing(flight_server: Any) -> None:
    """The allow-empty success path survives the flight-server upload in MULTIPROCESSING.

    Mirrors the base's ``test_empty_result_allowed_succeeds`` (state B with
    ``allow_empty_result()`` True succeeds) but runs with
    ``ParallelizationMode.MULTIPROCESSING``: the zero-row result must round-trip
    through the flight server upload from the process worker and come back as one
    result with zero rows. ``records_from_frame`` keeps the row-count assertion
    tolerant of the wire format (PyArrow table or list of row dicts).
    """
    result = mloda.run_all(
        [Feature(name="empty_result_allowed_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED_ALLOWED,
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert len(result) == 1
    assert len(records_from_frame(result[0])) == 0


class EmptyResultNoneAllowedFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FG whose ``calculate_feature`` returns ``None`` (state A) AND opts into empty results.

    Module-local on purpose: it pins that the ``allow_empty_result`` opt-in covers only
    representational empties (``[]`` / ``{}``), never a missing result.
    """

    @classmethod
    def allow_empty_result(cls) -> bool:
        return True

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_none_allowed_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # No result at all -> state A. The opt-in must NOT legitimize this.
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_none_allowed_col"}


_ENABLED_NONE_ALLOWED = PluginCollector.enabled_feature_groups({EmptyResultNoneAllowedFeatureGroup})


def test_empty_result_none_raises_python_dict_even_when_allowed(flight_server: Any) -> None:
    """State A live test: a ``None`` result raises on PythonDict even with the opt-in.

    ``allow_empty_result()`` legitimizes EMPTY results, not MISSING ones. PythonDict's
    ``transform`` must reject ``None`` like every schema-bearing framework does
    (``ValueError: Data type <class 'NoneType'> is not supported by PythonDictFramework``)
    instead of converting it to ``[]`` and letting the opt-in turn the run into a silent
    success. ``run_all`` re-wraps the framework error as a plain ``Exception`` whose
    message embeds that text (repr-escaped, so the inner quotes around ``NoneType`` carry
    backslashes), which the ``match`` below pins.
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
