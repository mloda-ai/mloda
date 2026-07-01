"""Red-phase regression test for Defect 1 (P2): a malformed (scalar-valued) dict result
must not bypass columnar validation.

Under the columnar model ``PythonDictFramework.expected_data_framework()`` returns ``dict``.
In ``ComputeFramework.run_calculation`` the ``transform`` step (which enforces the columnar
contract: every value is a list, all equal length) is SKIPPED when the ``calculate_feature``
result is already an instance of the expected framework type. A FeatureGroup that returns a
non-columnar dict with SCALAR values, e.g. ``{"feat": 1}``, is therefore stored as native
WITHOUT ever passing through ``transform``. This later corrupts joins / pyarrow upload.

Desired behavior: such a result must be REJECTED with a clear error (``ValueError`` /
``EmptyResultError`` surfaced as an ``Exception`` through the public ``run_all`` path), not
silently accepted. ``transform`` already rejects ``{"a": 1, "b": 2}`` (see
``test_python_dict_columnar_contract.py::TestTransformColumnar.test_dict_with_non_list_values_raises``);
the run must enforce the same contract even when the framework short-circuits ``transform``.

This test is expected to FAIL against the current implementation because ``{"feat": 1}`` is a
``dict`` (the expected framework type), so ``transform`` is skipped and the malformed result is
accepted and returned instead of raising.
"""

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    _EmptyResultMatchData,
)


class _ScalarValuedResultFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup whose ``calculate_feature`` returns a scalar-valued dict.

    ``{"scalar_result_col": 1}`` is NOT columnar: the value is a bare int, not a list. It
    must be rejected the same way ``transform`` rejects ``{"a": 1, "b": 2}``.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"scalar_result_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Scalar value, not a list -> malformed columnar dict. Bypasses transform today.
        return {"scalar_result_col": 1}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"scalar_result_col"}


class _MultiScalarValuedResultFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup returning a multi-column scalar-valued dict ``{"a": 1, "b": 2}``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"multi_scalar_result_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Both values scalar -> malformed columnar dict.
        return {"multi_scalar_result_col": 1, "extra_scalar_col": 2}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"multi_scalar_result_col"}


_ENABLED_SCALAR = PluginCollector.enabled_feature_groups({_ScalarValuedResultFeatureGroup})
_ENABLED_MULTI_SCALAR = PluginCollector.enabled_feature_groups({_MultiScalarValuedResultFeatureGroup})


def test_scalar_valued_dict_result_is_rejected(flight_server: Any) -> None:
    """A single-column scalar-valued dict result must raise through run_all, not be accepted.

    Expected failure reason (Red): the current implementation SKIPS ``transform`` for a dict
    result, so ``{"scalar_result_col": 1}`` is stored and returned as-is. ``run_all`` returns
    one malformed result instead of raising, so ``pytest.raises`` is unsatisfied.
    """
    with pytest.raises(Exception):
        mloda.run_all(
            [Feature(name="scalar_result_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SCALAR,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )


def test_multi_column_scalar_valued_dict_result_is_rejected(flight_server: Any) -> None:
    """A multi-column scalar-valued dict result must raise through run_all, not be accepted.

    Expected failure reason (Red): same short-circuit as above; ``{"a": 1, "b": 2}`` bypasses
    ``transform`` and is accepted, so no exception is raised.
    """
    with pytest.raises(Exception):
        mloda.run_all(
            [Feature(name="multi_scalar_result_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_MULTI_SCALAR,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )
