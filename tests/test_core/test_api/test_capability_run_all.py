"""End-to-end regression lock for the compute-framework capability hook (issue #482).

These tests prove the capability hook fires through the REAL ``mlodaAPI.run_all``
path, not just the resolver unit. They use only the always-installed frameworks
``PandasDataFrame`` and ``PythonDictFramework`` so they never skip (which would
break the tox EXPECTED_SKIP_COUNT).

Both tests are expected to PASS against the already-merged capability behaviour;
they lock the current end-to-end contract in place.
"""

from typing import Any, Optional

import pandas as pd

from mloda.provider import FeatureGroup, FeatureSet, DataCreator, BaseInputData
from mloda.user import Feature, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


FEAT = "E2ECapRunAllFeature"


class E2ECapRunAllFeatureGroup(FeatureGroup):
    """Root feature group that rejects PythonDictFramework only.

    Declared module-level so it registers as a discoverable FeatureGroup. It
    produces a single root feature via ``DataCreator`` and declares the operation
    unsupported on ``PythonDictFramework`` so route-around is deterministic.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({FEAT})

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: Any,
        options: Any,
        compute_framework: Any,
    ) -> bool:
        return compute_framework is not PythonDictFramework

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({FEAT: [1, 2, 3]})


class TestCapabilityRunAllEndToEnd:
    """Capability hook fires through the real ``mlodaAPI.run_all`` path."""

    def test_pinned_to_unsupported_framework_raises_capability_error(self) -> None:
        """Pinning the run to the only-rejected framework raises a capability error."""
        try:
            mloda.run_all([Feature(FEAT)], compute_frameworks=["PythonDictFramework"])
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected a ValueError when pinned to the unsupported framework")

        lowered = message.lower()
        assert "unsupported" in lowered, f"Capability error must signal 'unsupported', got: {message}"
        assert "PythonDictFramework" in message, f"Error must name the rejected framework, got: {message}"
        assert "Did you mean" not in message, f"Capability error must skip the fuzzy suggestion path, got: {message}"

    def test_route_around_to_pandas_returns_dataframe(self) -> None:
        """With both frameworks enabled, the run routes around to Pandas and computes."""
        result = mloda.run_all(
            [Feature(FEAT)],
            compute_frameworks=["PandasDataFrame", "PythonDictFramework"],
        )

        assert isinstance(result, list)
        assert len(result) == 1, f"Expected exactly one result frame, got: {result}"

        frame = result[0]
        assert isinstance(frame, pd.DataFrame), f"Route-around must produce a pandas DataFrame, got: {type(frame)}"
        assert FEAT in frame.columns, f"Result must contain column '{FEAT}', got columns: {list(frame.columns)}"
        assert len(frame) == 3, f"Result must have 3 rows, got: {len(frame)}"
