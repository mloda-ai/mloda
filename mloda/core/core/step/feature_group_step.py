from typing import Any
from uuid import UUID, uuid4
from mloda.core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.core.abstract_plugins.components.input_data.api_data_scope import ApiDataRef, api_data_scope
from mloda.core.abstract_plugins.components.credential_access import readable_credentials
from mloda.core.abstract_plugins.components.credential_grant import credential_read_mark, credentials_read_since
from mloda.core.abstract_plugins.components.failed_rows import collect_failed_rows
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.core.step.abstract_step import Step
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class FeatureGroupStep(Step):
    source_connection_handle: str | None = None
    credential_handles: tuple[str, ...] = ()
    reads_join_result: bool = True
    join_tokens: frozenset[UUID] = frozenset()
    own_frame: bool = False

    def __init__(
        self,
        feature_group: type[FeatureGroup],
        features: FeatureSet,
        required_uuids: set[UUID],
        compute_framework: type[ComputeFramework],
        children_if_root: set[UUID] | None = None,
    ) -> None:
        if children_if_root is None:
            children_if_root = set()
        self.feature_group = feature_group
        self.features = features
        self.required_uuids = required_uuids
        self.compute_framework = compute_framework
        self.children_if_root = frozenset(children_if_root.union({f.uuid for f in features.features}))

        self.uuid = uuid4()
        features.step_uuid = self.uuid

        self.need_to_upload = False

        self.tfs_ids: set[UUID] = set()

    def freeze(self) -> None:
        self.features.freeze()
        super().freeze()

    def get_uuids(self) -> set[UUID]:
        return {feature.uuid for feature in self.features.features}

    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFramework,
        from_cfw: ComputeFramework | UUID | None = None,  # source frame of an own-frame step
        data: Any | None = None,
    ) -> Any | None:
        location = cfw_register.get_location()
        binding = cfw_register.get_artifact_binding(self.uuid) if self.feature_group.artifact() is not None else None
        parameters = None
        if self.features.runtime_parameters:
            run_parameters = cfw_register.get_run_parameters()
            parameters = {k: run_parameters[k] for k in self.features.runtime_parameters if k in run_parameters}
        features = self.features.for_run(binding, parameters)
        if self.own_frame and data is None:
            data = self._source_data(cfw, from_cfw, location)

        match = features.input_data_match
        access = None if match is None else match[1].access
        with collect_failed_rows(self.feature_group.get_class_name()) as collector:
            if isinstance(access, ApiDataRef) and is_nominal_subclass(self.feature_group, ApiDataFG):
                key = access.key
                with api_data_scope(lambda: cfw_register.get_api_data_by_name(key), key):
                    data = self.run_calculate_feature(cfw, features, location, data)
            else:
                data = self.run_calculate_feature(cfw, features, location, data)
        if collector.entries:
            cfw_register.add_failed_rows(collector.entries)
        saver_mark = credential_read_mark()
        try:
            with readable_credentials():
                self.save_artifact(features, cfw_register)
        finally:
            calculate_context = getattr(features, "_calculate_context", None)
            if calculate_context is not None:
                merged = sorted({*calculate_context.credentials_read, *credentials_read_since(saver_mark)})
                object.__setattr__(calculate_context, "credentials_read", tuple(merged))

        # return_data_type_rule

        if location:
            if self.need_to_upload:
                if not isinstance(cfw.data, str):
                    native = cfw.data
                    cfw.upload_finished_data(location)
                    if cfw.mode is not ParallelizationMode.MULTIPROCESSING:
                        # a same-process consumer or the result collector reads cfw.data next; keep it native
                        cfw.set_data(native)
                cfw_register.add_uuid_flyway_datasets(cfw.uuid, set(self.children_if_root))
            return data
        return None

    def _source_data(
        self, cfw: ComputeFramework, from_cfw: ComputeFramework | UUID | None, location: str | None
    ) -> Any:
        """The input frame of an own-frame step: in-process data, or a worker-owned source's flight table."""
        if isinstance(from_cfw, ComputeFramework):
            return from_cfw.get_data()
        if isinstance(from_cfw, UUID) and location:
            return cfw.download_frame(location, from_cfw)
        raise ValueError(internal_invariant_error("an own-frame step has no source frame to read", f"step={self.uuid}"))

    def run_calculate_feature(
        self, cfw: ComputeFramework, features: FeatureSet, location: str | None, data: Any | None = None
    ) -> Any:
        if self.feature_group.calculate_feature is None:
            raise ValueError("FeatureGroup calculate_feature is not implemented")

        data = cfw.run_calculation(self.feature_group, features, location, data)
        cfw.validate_expected_framework(location)
        return data

    def save_artifact(self, features: FeatureSet, cfw_register: CfwManager) -> None:
        artifact = self.validate_set_artifact_to_save(features)
        if artifact is None:
            return

        artifact_or_meta_information = artifact.save(features, features.save_artifact)

        # Handle multiple artifacts case (when artifact.save returns a dict of artifacts)
        if isinstance(artifact_or_meta_information, dict) and isinstance(features.save_artifact, dict):
            # This is a multiple artifacts case - save each artifact individually
            for artifact_key, artifact_path in artifact_or_meta_information.items():
                cfw_register.set_artifact_to_save(artifact_key, artifact_path)
        else:
            # Single artifact case (legacy behavior)
            assert features.artifact_to_save is not None  # validated in validate_set_artifact_to_save
            cfw_register.set_artifact_to_save(features.artifact_to_save, artifact_or_meta_information)

    def validate_set_artifact_to_save(self, features: FeatureSet) -> None | type[BaseArtifact]:
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

    def get_parallelization_mode(self) -> set[ParallelizationMode]:
        return self.compute_framework.supported_parallelization_modes()

    def add_join_token(self, token: UUID) -> None:
        self.children_if_root = self.children_if_root | frozenset([token])
        self.join_tokens = self.join_tokens | frozenset([token])
