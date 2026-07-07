from __future__ import annotations
from typing import Generator, Optional
from uuid import UUID
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options


class Features:
    """A class to create the features collection and do basic validation.

    Input features inherit ALL consumer group options by default. Feature.forward_group=False
    isolates the input feature, a frozenset allowlist restricts the copy to the listed keys,
    and Feature.forward_group_exclude carves out single keys. Context flows only via the
    consumer-side Options.propagate_context_keys push or the Feature.inherit_context_keys pull.

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
                feature = Feature(name=feature, options=Options({}), domain=self.parent_domain)
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
        Apply forward-by-default inheritance of the consumer's options onto the input feature.

        Delegates to Options.inherit_from: the input feature inherits ALL consumer group
        keys by default; feature.forward_group=False isolates it, an allowlist restricts
        the copy, and feature.forward_group_exclude carves out single keys. Consumer
        context keys flow only when listed in feature.inherit_context_keys or pushed via
        the consumer's propagate_context_keys. Conflict detection lives in inherit_from.

        Args:
            feature: The input feature whose options are updated in place
            child_options: The consumer feature's options (never mutated)

        Raises:
            ValueError: If an inherited key conflicts with an existing value
        """
        feature.options.inherit_from(
            child_options,
            forward_group=feature.forward_group,
            forward_group_exclude=feature.forward_group_exclude,
            inherit_context_keys=feature.inherit_context_keys,
            owner=str(feature.name),
        )

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
