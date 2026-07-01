"""Contract tests for the generic ``PolicyRunAllTestBase`` (issue #521).

Pins the public API of the generic policy run_all base directly, independently of the
``allow_empty_result`` consumer: the base must drive any FeatureGroup-level policy flag
end-to-end through ``mloda.run_all``. These tests reuse the existing empty-result fixtures
and exercise both the success and raises expectations, in SYNC and in a worker-based
(THREADING) parallelization mode.
"""

from typing import Any

from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda.user import ParallelizationMode

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    _ENABLED_ALLOWED,
    _ENABLED_SCHEMALESS_ALLOWED,
)
from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import (
    PolicyRaises,
    PolicyRunAllTestBase,
    PolicySuccess,
    records_from_frame,
)


class _PythonDictPolicyConformance(PolicyRunAllTestBase):
    """Concrete subclass for python_dict; no ``Test`` prefix so pytest does not collect it."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PythonDictFramework"


def _assert_single_empty(result: list[Any]) -> None:
    assert len(result) == 1
    assert len(records_from_frame(result[0])) == 0


def test_policy_base_drives_success_case(flight_server: Any) -> None:
    """A policy success expectation returns a schema-bearing zero-row result."""
    conformance = _PythonDictPolicyConformance()

    result = conformance.assert_policy_case(
        feature_name="empty_result_allowed_col",
        plugin_collector=_ENABLED_ALLOWED,
        expectation=PolicySuccess(assert_result=_assert_single_empty),
        mode=ParallelizationMode.SYNC,
        flight_server=flight_server,
    )

    assert result is not None
    _assert_single_empty(result)


def test_policy_base_drives_raises_case(flight_server: Any) -> None:
    """A policy raises expectation: a zero-column ``{}`` result is schema-less and raises."""
    conformance = _PythonDictPolicyConformance()

    result = conformance.assert_policy_case(
        feature_name="empty_result_schemaless_col",
        plugin_collector=_ENABLED_SCHEMALESS_ALLOWED,
        expectation=PolicyRaises(match_substring=EmptyResultError.__name__),
        mode=ParallelizationMode.SYNC,
        flight_server=flight_server,
    )

    assert result is None


def test_policy_base_drives_worker_mode(flight_server: Any) -> None:
    """The same success case under THREADING proves the worker-based path is covered."""
    conformance = _PythonDictPolicyConformance()

    result = conformance.assert_policy_case(
        feature_name="empty_result_allowed_col",
        plugin_collector=_ENABLED_ALLOWED,
        expectation=PolicySuccess(assert_result=_assert_single_empty),
        mode=ParallelizationMode.THREADING,
        flight_server=flight_server,
    )

    assert result is not None
    _assert_single_empty(result)
