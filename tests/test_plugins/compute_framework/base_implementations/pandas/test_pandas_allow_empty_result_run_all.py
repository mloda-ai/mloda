"""End-to-end ``allow_empty_result`` policy test for pandas.

Consumes the shared EmptyResultRunAllTestBase. Pandas is a guaranteed dependency,
so no import guard is needed; the subclass only selects the framework by name.
"""

from typing import Any

import pytest

from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import mloda

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
    _ENABLED_NONE,
)


class TestPandasAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on pandas."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PandasDataFrame"


def test_empty_result_none_raises_pandas(flight_server: Any) -> None:
    """State A live test: a FG whose ``calculate_feature`` returns ``None`` raises end-to-end.

    Driven through the pandas compute framework, which needs no connection. Requested as a
    FINAL feature with the default (not-allowed) policy. Pandas rejects the ``None`` result
    upstream (``transform`` raises ``ValueError: Data <NoneType> is not supported``) before
    the empty-result guard runs; ``run_all`` re-wraps the framework error as a plain
    ``Exception``. Kept as a module-level function (not a method on
    ``EmptyResultRunAllTestBase``) so connection-backed frameworks don't inherit and run it
    without a connection.
    """
    feature = Feature(name="empty_result_none_col")

    with pytest.raises(Exception):
        mloda.run_all(
            [feature],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_ENABLED_NONE,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )
