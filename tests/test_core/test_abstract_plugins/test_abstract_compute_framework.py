from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from mloda.provider import ComputeFramework
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


class _ListFeatureGroup:
    result: Any = None

    @classmethod
    def input_data(cls) -> None:
        return None

    @classmethod
    def get_class_name(cls) -> str:
        return "_ListFeatureGroup"

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return cls.result

    @classmethod
    def validate_input_features(cls, data: Any, features: Any) -> None:
        return None

    @classmethod
    def validate_output_features(cls, data: Any, features: Any) -> None:
        return None

    @classmethod
    def final_filters(cls) -> bool | None:
        return None


def _list_features() -> Any:
    features = MagicMock()
    features.filter_engine = None
    features.get_all_names.return_value = ("a",)
    features.get_options_key.return_value = None
    features.get_initial_requested_features.return_value = ()
    features.features = set()
    return features


def _run(framework: ComputeFramework, result: Any) -> None:
    _ListFeatureGroup.result = result
    with patch.object(ComputeFramework, "set_filter_engine", lambda self, features: features):
        with patch.object(ComputeFramework, "set_mask_engine", lambda self, features: features):
            framework.run_calculation(_ListFeatureGroup, _list_features(), None)


class TestRunCalculationPinsConvertedResult:
    def _framework(self) -> _ListFramework:
        return _ListFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), uuid=uuid4())

    def test_foreign_result_converted_by_transform_is_pinned(self) -> None:
        framework = self._framework()
        _run(framework, (1, 2, 3))
        assert framework.data == [1, 2, 3]
        assert framework._pinned_frame is framework.data

    def test_already_native_result_is_not_pinned(self) -> None:
        framework = self._framework()
        _run(framework, [1, 2, 3])
        assert framework.data == [1, 2, 3]
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
