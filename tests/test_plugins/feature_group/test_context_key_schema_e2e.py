"""End-to-end test proving context-key schema validation fires through the public run path.

An opt-in DataCreator FeatureGroup declares a context-key schema. When a feature
is requested with a typo'd context key, running through the public API must raise
a ValueError that mentions the offending key. Validation happens at planning time
(IdentifyFeatureGroupClass), before the worker layer, so in SYNC mode the error
surfaces as an unwrapped ValueError.
"""

from typing import Any, Optional

import pytest
import pyarrow as pa

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, Features, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.core.abstract_plugins.components.options import Options
from tests.test_core.test_tooling import MlodaTestRunner


class OptInContextSchemaE2EFeatureGroup(FeatureGroup):
    """DataCreator FG that opts in to context-key validation with schema {partition_by: str}."""

    @classmethod
    def context_key_schema(cls) -> dict[str, Any] | None:
        return {"partition_by": str}

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({cls.get_class_name(): [1, 2, 3]})


def test_valid_context_key_runs_end_to_end() -> None:
    """Sanity baseline: with a valid context key the feature computes a table cleanly.

    This proves the FG is well-formed so that the typo test below fails for the
    right reason (missing validation), not because of an unrelated runtime error.
    """
    feature_name = "OptInContextSchemaE2EFeatureGroup"

    features = Features(
        [Feature(name=feature_name, options=Options(context={"partition_by": "x"}), initial_requested_data=True)]
    )

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks={PyArrowTable},
        parallelization_modes={ParallelizationMode.SYNC},
    )

    assert result.results
    for res in result.results:
        data = res.to_pydict()
        assert data[feature_name] == [1, 2, 3]


def test_unknown_context_key_raises_through_run_path() -> None:
    """Requesting the feature with a typo'd context key raises a ValueError mentioning the key."""
    feature_name = "OptInContextSchemaE2EFeatureGroup"

    features = Features(
        [Feature(name=feature_name, options=Options(context={"partiton_by": "x"}), initial_requested_data=True)]
    )

    with pytest.raises(ValueError, match="partiton_by"):
        MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes={ParallelizationMode.SYNC},
        )
