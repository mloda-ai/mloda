from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class BaseTestComputeFramework1(ComputeFramework):
    pass


class BaseTestComputeFramework2(ComputeFramework):
    pass


class BaseTestComputeFramework3(ComputeFramework):
    pass


class TestComputeFrameworkValidationErrors:
    def test_validate_expected_framework_includes_expected_type(self) -> None:
        """The error message should contain 'Expected:' and the expected data framework type name.

        Currently the message only says the data type is not supported, without telling the user
        what type was expected. The improved message should include 'Expected:' followed by the
        expected framework type.
        """

        class DictFramework(ComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> type:
                return dict

        fw = DictFramework(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )
        fw.data = "this is a string, not a dict"

        with pytest.raises(ValueError, match=r"Expected type:"):
            fw.validate_expected_framework()


class _ListFramework(ComputeFramework):
    """Native type is ``list``; transform converts a tuple, as a transformer would."""

    @classmethod
    def expected_data_framework(cls) -> type:
        return list

    def transform(self, data: Any, feature_names: Sequence[str]) -> Any:
        return list(data)

    def set_column_names(self) -> None:
        self.column_names = {"a"}


class _PlanListFramework(_ListFramework):
    @classmethod
    def _positional_append_needs_pinned_frame(cls) -> bool:
        return True


class _ForeignResultFeatureGroup(FeatureGroup):
    """Returns a pyarrow table, a type pandas does not hold natively."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"my_feature": [1, 2, 3]})


class _NativeResultFeatureGroup(FeatureGroup):
    """Returns the framework's own native frame."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"my_feature": [1, 2, 3]})


class TestRunCalculationPinsConvertedResult:
    def _run(self, feature_group: type[FeatureGroup]) -> PandasDataFrame:
        framework = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), uuid=uuid4())
        framework.run_calculation(feature_group, FeatureSet([Feature("my_feature")]), location=None)
        return framework

    def test_foreign_result_converted_by_transform_is_pinned(self) -> None:
        framework = self._run(_ForeignResultFeatureGroup)
        assert isinstance(framework.data, pd.DataFrame)
        assert framework._pinned_frame is framework.data

    def test_already_native_result_is_not_pinned(self) -> None:
        framework = self._run(_NativeResultFeatureGroup)
        assert isinstance(framework.data, pd.DataFrame)
        assert framework._pinned_frame is None


class TestRequirePinnedFrameForColumn:
    def _framework(self, cls: type[_ListFramework]) -> _ListFramework:
        framework = cls(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), uuid=uuid4())
        framework.data = [1]
        return framework

    def test_unpinned_plan_frame_raises_naming_feature_and_framework(self) -> None:
        framework = self._framework(_PlanListFramework)
        with pytest.raises(ValueError, match="my_feature") as excinfo:
            framework._require_pinned_frame_for_column("my_feature")
        assert "_PlanListFramework" in str(excinfo.value)

    def test_pinned_plan_frame_does_not_raise(self) -> None:
        framework = self._framework(_PlanListFramework)
        framework._pinned_frame = framework.data
        framework._require_pinned_frame_for_column("my_feature")

    def test_framework_without_the_opt_in_does_not_raise(self) -> None:
        self._framework(_ListFramework)._require_pinned_frame_for_column("my_feature")
