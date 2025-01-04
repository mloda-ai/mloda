from typing import Any, Optional, Set, Type, Union
from uuid import UUID, uuid4
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.cfw_manager import CfwManager
from mloda_core.core.step.abstract_step import Step
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class FeatureGroupStep(Step):
    def __init__(
        self,
        feature_group: Type[AbstractFeatureGroup],
        features: FeatureSet,
        required_uuids: Set[UUID],
        compute_framework: Type[ComputeFrameWork],
        children_if_root: set[UUID] = set(),
        api_input_data: Union[BaseApiData, bool] = False,
    ) -> None:
        self.feature_group = feature_group
        self.features = features
        self.required_uuids = required_uuids
        self.compute_framework = compute_framework
        self.children_if_root = frozenset(children_if_root.union({f.uuid for f in features.features}))
        self.api_input_data = api_input_data

        self.uuid = uuid4()

        self.step_is_done = False

        self.need_to_upload = False

        # Currently, also used for joinsteps without tfs. This might be a bug.
        self.tfs_ids: Set[UUID] = set()

    def get_uuids(self) -> Set[UUID]:
        return {feature.uuid for feature in self.features.features}

    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFrameWork,
        from_cfw: Optional[ComputeFrameWork] = None,  # Not used in this implementation
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        self.location = cfw_register.get_location()
        if self.api_input_data:
            data = self.get_api_input_data(data, cfw_register)

        data = self.run_calculate_feature(cfw, data)
        self.save_artifact(self.features, cfw_register)

        # return_data_type_rule

        if self.location:
            if self.need_to_upload:
                cfw.upload_finished_data(self.location)
                cfw_register.add_uuid_flyway_datasets(cfw.uuid, set(self.children_if_root))
            return data
        return None

    def run_calculate_feature(self, cfw: ComputeFrameWork, data: Optional[Any] = None) -> Any:
        if self.feature_group.calculate_feature is None:
            raise ValueError("FeatureGroup calculate_feature is not implemented")

        data = cfw.run_calculation(self.feature_group, self.features, self.location, data)
        cfw.validate_expected_framework(self.location)
        return data

    def save_artifact(self, features: FeatureSet, cfw_register: CfwManager) -> None:
        artifact = self.validate_set_artifact_to_save(features)
        if artifact is None:
            return

        artifact_or_meta_information = artifact.save(features, features.save_artifact)
        cfw_register.set_artifact_to_save(features.artifact_to_save, artifact_or_meta_information)  # type: ignore

    def validate_set_artifact_to_save(self, features: FeatureSet) -> None | Type[BaseArtifact]:
        if features.artifact_to_save is None:
            return None

        if features.save_artifact is None:
            raise ValueError(
                f"No artifact to save although it was requested. {self.feature_group} {features.artifact_to_save}."
            )

        artifact = self.feature_group.artifact()
        if artifact is None:
            raise ValueError(
                f"Artifact is not implemented for {self.feature_group}, but requested {features.artifact_to_save}"
            )
        return artifact

    def get_api_input_data(self, data: Any, cfw_register: CfwManager) -> Any:
        if data is not None:
            raise ValueError(f"Data is not None, but api_input_data is not False. {self.feature_group}.")

        if not isinstance(self.api_input_data, BaseApiData):
            raise ValueError(f"Api input data is not a tuple. {self.feature_group}.")

        api_data = cfw_register.get_api_data_by_name(self.api_input_data.get_api_input_name())

        data = self.api_input_data.get_data_by_using_api_data(api_data)

        if data is None:
            raise ValueError(
                f"Data is None: {self.feature_group} although we have an api_input_data {self.api_input_data.get_api_input_name()}."
            )

        return data

    def add_value_to_children_if_root(self, value: UUID) -> None:
        self.children_if_root = self.children_if_root | frozenset([value])
