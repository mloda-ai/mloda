from __future__ import annotations
from typing import Generator, Optional
from uuid import UUID
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.validators.options_validator import OptionsValidator


class Features:
    """A class to create the features collection and do basic validation.

    Parent uuid is internal logic for now. Don t give this parameter.
    """

    def __init__(
        self,
        features: list[Feature | str],
        child_options: Optional[Options] = None,
        child_uuid: Optional[UUID] = None,
        parent_domain: Optional[str] = None,
    ) -> None:
        self.collection: list[Feature] = []
        self.child_uuid: Optional[UUID] = child_uuid
        self.parent_domain: Optional[str] = parent_domain

        self.parent_uuids: set[UUID] = set()

        if child_options is None:
            child_options = Options({})

        self.check_for_duplicate_string_features(features)
        self.build_feature_collection(features, child_options, child_uuid)

    def build_feature_collection(
        self, features: list[Feature | str], child_options: Options, child_uuid: Optional[UUID] = None
    ) -> None:
        for feature in features:
            if child_options.group == {} and child_options.context == {}:
                child_options = Options({})

            if isinstance(feature, str):
                feature = Feature(name=feature, options=child_options, domain=self.parent_domain)
            else:
                if feature.domain is None and self.parent_domain is not None:
                    feature.domain = Domain(self.parent_domain)
            if child_uuid:
                self.parent_uuids.add(feature.uuid)
                self.child_uuid = child_uuid
                feature.child_options = child_options
                self.merge_options(feature, child_options)

            self.check_duplicate_feature(feature)
            self.collection.append(feature)

    def merge_options(self, feature: Feature, child_options: Options) -> None:
        """
        Forward allowlisted consumer (child_options) options onto the input feature.

        Allowlist semantics (issue #579): by default NOTHING from the consumer is
        merged onto the input feature. Forwarding is opt-in:
        - feature.forward_group: consumer GROUP keys copied onto the input feature
          (a set/frozenset of keys, or True for all keys except in_features)
        - feature.inherit_context_keys: consumer CONTEXT keys pulled onto the input feature
        - child_options.propagate_context_keys: consumer-side push of context keys

        Raises:
            ValueError: If a forwarded/inherited/propagated key already exists on the
                input feature with a different value
        """
        self._forward_group_options(feature, child_options)
        self._inherit_context_options(feature, child_options)
        self._push_propagated_context_options(feature.options, child_options)

    def _forward_group_options(self, feature: Feature, child_options: Options) -> None:
        forward_group = feature.forward_group
        if not forward_group:
            return

        if isinstance(forward_group, (set, frozenset)):
            keys = [key for key in forward_group if key in child_options.group]
        else:
            keys = [key for key in child_options.group if key != DefaultOptionKeys.in_features]

        OptionsValidator.validate_no_group_context_conflicts(set(keys), set(feature.options.context.keys()))

        for key in keys:
            value = child_options.group[key]
            if key in feature.options.group and feature.options.group[key] != value:
                raise ValueError(
                    f"Forwarded group key '{key}' conflict: consumer='{value}', "
                    f"input feature='{feature.options.group[key]}'"
                )
            feature.options.group[key] = value

    def _inherit_context_options(self, feature: Feature, child_options: Options) -> None:
        if not feature.inherit_context_keys:
            return

        keys = [key for key in feature.inherit_context_keys if key in child_options.context]

        OptionsValidator.validate_no_context_group_conflicts(set(keys), set(feature.options.group.keys()))

        for key in keys:
            value = child_options.context[key]
            if key in feature.options.context and feature.options.context[key] != value:
                raise ValueError(
                    f"Inherited context key '{key}' conflict: consumer='{value}', "
                    f"input feature='{feature.options.context[key]}'"
                )
            feature.options.context[key] = value

    def _push_propagated_context_options(self, feature_options: Options, child_options: Options) -> None:
        if not child_options.propagate_context_keys:
            return

        propagating = {
            key: value
            for key, value in child_options.context.items()
            if key in child_options.propagate_context_keys and key != DefaultOptionKeys.in_features
        }

        OptionsValidator.validate_no_context_group_conflicts(set(propagating.keys()), set(feature_options.group.keys()))

        for key, value in propagating.items():
            if key in feature_options.context and feature_options.context[key] != value:
                raise ValueError(
                    f"Context key '{key}' conflict: parent='{value}', child='{feature_options.context[key]}'"
                )

        feature_options.context.update(propagating)

    def check_duplicate_feature(self, feature: Feature) -> None:
        if feature in self.collection:
            raise ValueError(f"Duplicate feature setup: {feature.name}")

    def __iter__(self) -> Generator[Feature, None, None]:
        yield from self.collection

    def check_for_duplicate_string_features(self, features: list[Feature | str]) -> None:
        check_set: set[str] = set()
        for feature in features:
            if isinstance(feature, str):
                if feature in check_set:
                    raise ValueError(f"You are adding same feature as string twice: {feature}")
                check_set.add(feature)
