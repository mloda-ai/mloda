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
    _ENABLED_SCHEMALESS_ALLOWED,
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
    upstream (``transform`` raises ``ValueError: Data <class 'NoneType'> is not supported by
    PandasDataFrame``) before the empty-result guard runs; ``run_all`` re-wraps the framework
    error as a plain ``Exception`` whose message embeds that text (repr-escaped, so the inner
    quotes around ``NoneType`` carry backslashes), which the ``match`` below pins. Kept as a
    module-level function (not a method on ``EmptyResultRunAllTestBase``) so connection-backed
    frameworks don't inherit and run it without a connection.
    """
    feature = Feature(name="empty_result_none_col")

    with pytest.raises(Exception, match=r"Data <class \\?'NoneType\\?'> is not supported by PandasDataFrame"):
        mloda.run_all(
            [feature],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_ENABLED_NONE,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
        )


def test_zero_column_allowed_succeeds_pandas(flight_server: Any) -> None:
    """An opted-in FG returning ``{}`` (zero columns) must succeed end-to-end on pandas.

    The output guard accepts the schema-less result because ``allow_empty_result()`` is
    True; result selection must then pass the zero-column frame through unchanged instead
    of trying (and failing) to match feature names against an empty column set. The caller
    receives exactly one result that carries zero columns.
    """
    feature = Feature(name="empty_result_schemaless_col")

    result = mloda.run_all(
        [feature],
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=_ENABLED_SCHEMALESS_ALLOWED,
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
    )

    assert len(result) == 1
    # Tolerant zero-column check: pandas DataFrame and pa.Table both expose .columns.
    assert len(result[0].columns) == 0
