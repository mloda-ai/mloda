from typing import Any

import pytest
from mloda.core.api.prepare.setup_compute_framework import SetupComputeFramework
from mloda.core.api.request import SetupConfigurationError
from mloda.user import Feature
from mloda.user import Features, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.provider import ComputeFramework
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


class TestSetupComputeFramework:
    @pytest.fixture
    def features(self) -> Features:
        return Features([Feature("some_feature")])

    def get_list_compute_frameworks_as_class_name(self) -> list[str]:
        return [subclass.get_class_name() for subclass in get_all_subclasses(ComputeFramework)]

    def test_init_with_user_compute_frameworks(self, features: Features) -> None:
        user_compute_frameworks = ["PyArrowTable", "PandasDataFrame"]
        setup_compute_framework = SetupComputeFramework(user_compute_frameworks, features)
        assert {fw.get_class_name() for fw in setup_compute_framework.compute_frameworks} == set(
            user_compute_frameworks
        )
        assert [fw.get_class_name() for fw in setup_compute_framework.framework_preference] == user_compute_frameworks

    def test_init_without_user_compute_frameworks(self, features: Features) -> None:
        setup_compute_framework = SetupComputeFramework(None, features)
        assert len(setup_compute_framework.compute_frameworks) == len(self.get_list_compute_frameworks_as_class_name())

    def test_validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
        self, features: Features
    ) -> None:
        feature = Feature("some_feature2", compute_framework=self.get_list_compute_frameworks_as_class_name()[0])
        features.collection.append(feature)

        available_compute_frameworks = {ComputeFramework}
        setup_compute_framework = SetupComputeFramework(None, features)

        # negative test
        with pytest.raises(ValueError):
            setup_compute_framework.validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
                features, available_compute_frameworks
            )

        # positive test
        available_compute_frameworks = get_all_subclasses(ComputeFramework)
        setup_compute_framework = SetupComputeFramework(None, features)
        assert setup_compute_framework.compute_frameworks == available_compute_frameworks

    def test_filter_user_set_in_available_sub_classes(self, features: Features) -> None:
        api_request_compute_frameworks = self.get_list_compute_frameworks_as_class_name()[1:]
        sub_classes = get_all_subclasses(ComputeFramework)

        setup_compute_framework = SetupComputeFramework(None, features)

        filtered_compute_frameworks = setup_compute_framework.filter_user_set_in_available_sub_classes(
            api_request_compute_frameworks,
            sub_classes,
        )
        assert len({fw.get_class_name() for fw in filtered_compute_frameworks}) == len(
            set(api_request_compute_frameworks)
        )

    def test_filter_user_set_in_available_sub_classes_empty_available_classes_suggests_loading_plugins(
        self, features: Features
    ) -> None:
        setup_compute_framework = SetupComputeFramework(None, features)

        with pytest.raises(ValueError) as exc_info:
            setup_compute_framework.filter_user_set_in_available_sub_classes(["missing_compute_framework"], set())

        message = str(exc_info.value)
        assert "missing_compute_framework" in message
        assert "available compute frameworks: []" in message
        assert "Did you call PluginLoader.all()?" in message

    def test_filter_user_set_in_available_sub_classes_non_empty_available_classes_keeps_existing_message(
        self, features: Features
    ) -> None:
        setup_compute_framework = SetupComputeFramework(None, features)

        with pytest.raises(ValueError) as exc_info:
            setup_compute_framework.filter_user_set_in_available_sub_classes(
                ["missing_compute_framework"], {ComputeFramework}
            )

        message = str(exc_info.value)
        assert "missing_compute_framework" in message
        assert "['ComputeFramework']" in message
        assert "PluginLoader" not in message

    @pytest.mark.parametrize(
        "bad",
        [{"PandasDataFrame"}, frozenset({"PandasDataFrame"}), ("PandasDataFrame",), "PandasDataFrame", set()],
        ids=["set", "frozenset", "tuple", "str", "empty_set"],
    )
    def test_non_list_raises_with_ordered_list_hint(self, features: Features, bad: Any) -> None:
        with pytest.raises(ValueError, match="ordered list") as exc_info:
            SetupComputeFramework(bad, features)

        assert '["PolarsDataFrame", "PandasDataFrame"]' in str(exc_info.value)

    @pytest.mark.parametrize("empty", [None, []], ids=["none", "empty_list"])
    def test_none_and_empty_list_keep_all_frameworks_without_preference(
        self, features: Features, empty: list[str] | None
    ) -> None:
        setup_compute_framework = SetupComputeFramework(empty, features)

        assert setup_compute_framework.compute_frameworks == get_all_subclasses(ComputeFramework)
        assert setup_compute_framework.framework_preference == ()

    def test_unknown_name_raises_even_when_another_entry_matches(self, features: Features) -> None:
        with pytest.raises(ValueError, match="NoSuchFramework"):
            SetupComputeFramework(["PandasDataFrame", "NoSuchFramework"], features)

    def test_framework_preference_keeps_list_order(self, features: Features) -> None:
        forward = SetupComputeFramework(["PyArrowTable", "PandasDataFrame"], features)
        backward = SetupComputeFramework(["PandasDataFrame", "PyArrowTable"], features)

        assert forward.framework_preference == (PyArrowTable, PandasDataFrame)
        assert backward.framework_preference == (PandasDataFrame, PyArrowTable)

    def test_framework_preference_dedups_by_class(self, features: Features) -> None:
        setup_compute_framework = SetupComputeFramework(["PandasDataFrame", PandasDataFrame], features)

        assert setup_compute_framework.framework_preference == (PandasDataFrame,)

    def test_set_through_mloda_api_raises_setup_configuration_error(self) -> None:
        with pytest.raises(SetupConfigurationError, match="ordered list"):
            mloda.prepare(["some_feature"], compute_frameworks={"PandasDataFrame"})  # type: ignore[arg-type]
