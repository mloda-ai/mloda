from __future__ import annotations
from typing import Generator, List, Optional, Set, Union
from uuid import UUID
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class Features:
    """A class to create the features collection and do basic validation.

    Parent uuid is internal logic for now. Don t give this parameter.
    """

    def __init__(
        self,
        features: List[Union[Feature, str]],
        child_options: Optional[Options] = None,
        child_uuid: Optional[UUID] = None,
    ) -> None:
        self.collection: List[Feature] = []
        self.child_uuid: Optional[UUID] = child_uuid

        self.parent_uuids: set[UUID] = set()

        if child_options is None:
            child_options = Options({})

        self.check_for_duplicate_string_features(features)
        self.build_feature_collection(features, child_options, child_uuid)

    def build_feature_collection(
        self, features: List[Union[Feature, str]], child_options: Options, child_uuid: Optional[UUID] = None
    ) -> None:
        for feature in features:
            if child_options.group == {} and child_options.context == {}:
                child_options = Options({})

            feature = Feature(name=feature, options=child_options) if isinstance(feature, str) else feature
            if child_uuid:
                self.parent_uuids.add(feature.uuid)
                self.child_uuid = child_uuid
                feature.child_options = child_options
                self.merge_options(feature.options, child_options)

            self.check_duplicate_feature(feature)
            self.collection.append(feature)

    def merge_options(self, feature_options: Options, child_options: Options) -> None:
        """
        Merge child_options into feature_options, respecting protected keys.

        Protected keys allow parent and child features to maintain different values
        without conflicts. This is essential for feature chaining where each level
        needs its own configuration for certain parameters.

        Protected keys are determined dynamically by reading:
        - in_features (always protected)
        - Keys listed in feature_options.get(mloda_feature_chainer_parser_key)

        Args:
            feature_options: Parent feature's options (will be updated)
            child_options: Child feature's options to merge in

        Raises:
            ValueError: If non-protected keys have conflicting values
        """
        # Get protected keys dynamically from the feature options
        protected_keys = {DefaultOptionKeys.in_features}
        if feature_options.get(DefaultOptionKeys.mloda_feature_chainer_parser_key):
            protected_keys.update(feature_options.get(DefaultOptionKeys.mloda_feature_chainer_parser_key))

        # Check for conflicts in non-protected keys only
        for key_child, value_child in child_options.items():
            for key_parent, value_parent in feature_options.items():
                if key_child == key_parent:
                    # Skip protected keys - they're allowed to differ
                    if key_parent in protected_keys:
                        continue

                    # Non-protected keys must match
                    if value_child != value_parent:
                        raise ValueError(
                            f"Duplicate key '{key_child}' found with conflicting values. "
                            f"Parent has '{value_parent}', child has '{value_child}'. "
                            f"Protected keys that can differ: {protected_keys}"
                        )

        # Merge child options into parent, excluding protected keys
        # update_with_protected_keys will read mloda_feature_chainer_parser_key dynamically
        feature_options.update_with_protected_keys(child_options)

    def check_duplicate_feature(self, feature: Feature) -> None:
        if feature in self.collection:
            raise ValueError(f"Duplicate feature setup: {feature.name}")

    def __iter__(self) -> Generator[Feature, None, None]:
        yield from self.collection

    def check_for_duplicate_string_features(self, features: List[Union[Feature, str]]) -> None:
        check_set: Set[str] = set()
        for feature in features:
            if isinstance(feature, str):
                if feature in check_set:
                    raise ValueError(f"You are adding same feature as string twice: {feature}")
                check_set.add(feature)
