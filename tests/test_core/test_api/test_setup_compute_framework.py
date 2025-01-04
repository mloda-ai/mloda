import pytest
from mloda_core.api.prepare.setup_compute_framework import SetupComputeFramework
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


class TestSetupComputeFramework:
    @pytest.fixture
    def features(self) -> Features:
        return Features([Feature("some_feature")])

    def get_list_compute_frameworks_as_class_name(self) -> list[str]:
        return [subclass.get_class_name() for subclass in get_all_subclasses(ComputeFrameWork)]

    def test_init_with_user_compute_frameworks(self, features: Features) -> None:
        user_compute_frameworks = self.get_list_compute_frameworks_as_class_name()[:2]
        setup_compute_framework = SetupComputeFramework(user_compute_frameworks, features)
        assert len({fw.get_class_name() for fw in setup_compute_framework.compute_frameworks}) in (1, 2)

    def test_init_without_user_compute_frameworks(self, features: Features) -> None:
        setup_compute_framework = SetupComputeFramework(None, features)
        assert len(setup_compute_framework.compute_frameworks) == len(self.get_list_compute_frameworks_as_class_name())

    def test_validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
        self, features: Features
    ) -> None:
        feature = Feature("some_feature2", compute_framework=self.get_list_compute_frameworks_as_class_name()[0])
        features.collection.append(feature)

        available_compute_frameworks = {ComputeFrameWork}
        setup_compute_framework = SetupComputeFramework(None, features)

        # negative test
        with pytest.raises(ValueError):
            setup_compute_framework.validate_if_at_least_one_feature_compute_framework_is_in_available_compute_framework(
                features, available_compute_frameworks
            )

        # positive test
        available_compute_frameworks = get_all_subclasses(ComputeFrameWork)
        setup_compute_framework = SetupComputeFramework(None, features)
        assert setup_compute_framework.compute_frameworks == available_compute_frameworks

    def test_filter_user_set_in_available_sub_classes(self, features: Features) -> None:
        api_request_compute_frameworks = set(self.get_list_compute_frameworks_as_class_name()[1:])
        sub_classes = get_all_subclasses(ComputeFrameWork)

        setup_compute_framework = SetupComputeFramework(None, features)

        filtered_compute_frameworks = setup_compute_framework.filter_user_set_in_available_sub_classes(
            api_request_compute_frameworks,  # type: ignore
            sub_classes,
        )
        assert len({fw.get_class_name() for fw in filtered_compute_frameworks}) == len(api_request_compute_frameworks)
