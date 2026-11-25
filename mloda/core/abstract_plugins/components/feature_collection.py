from __future__ import annotations
from collections.abc import Sequence
from typing import Generator
from uuid import UUID, uuid4
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options


def consume_seed(requested: dict[int, tuple[Feature, Feature]], feature: Feature) -> Feature | None:
    """Return the plan object for an unconsumed seed and drop every key pointing at it."""
    seed = requested.get(id(feature))
    if seed is None:
        return None
    source, plan_object = seed
    for key in [k for k, (_, p) in requested.items() if p is plan_object]:
        del requested[key]
    if source is not plan_object:
        plan_object.plan_source = source
    return plan_object


class Features:
    """A class to create the features collection and do basic validation.

    Input features inherit all consumer group options by default, except keys whose spec sets
    forward_to_inputs=False (consumer_local_keys). Feature.forward_group=False
    isolates the input feature, a frozenset allowlist restricts the copy to the listed keys,
    and Feature.forward_group_exclude carves out single keys. Context flows only via the
    consumer-side Options.propagate_context_keys or cascade_context_keys push or the
    Feature.inherit_context_keys pull.

    Parent uuid is internal logic for now. Don t give this parameter.
    """

    def __init__(
        self,
        features: Sequence[Feature | str],
        child_options: Options | None = None,
        child_uuid: UUID | None = None,
        parent_domain: str | None = None,
        consumer_local_keys: frozenset[str] = frozenset(),
        requested: dict[int, tuple[Feature, Feature]] | None = None,
    ) -> None:
        if isinstance(features, str):
            raise ValueError(
                "features must be a list or tuple of names or Feature objects; wrap a single name in a list"
            )

        self.collection: list[Feature] = []
        self.child_uuid: UUID | None = child_uuid
        self.parent_domain: str | None = parent_domain
        self.consumer_local_keys: frozenset[str] = consumer_local_keys
        # Unconsumed seeds by id: (clean source, plan request object). The first consumer plans into the plan object.
        self.requested: dict[int, tuple[Feature, Feature]] = {} if requested is None else requested

        self.parent_uuids: set[UUID] = set()

        if child_options is None:
            child_options = Options({})

        self.check_for_duplicate_string_features(features)
        self.build_feature_collection(features, child_options, child_uuid)

    def build_feature_collection(
        self, features: Sequence[Feature | str], child_options: Options, child_uuid: UUID | None = None
    ) -> None:
        for feature in features:
            if child_options.group == {} and child_options.context == {}:
                child_options = Options({})

            if isinstance(feature, str):
                feature = Feature(name=feature, options=Options({}), domain=self.parent_domain)
            else:
                if child_uuid:
                    seeded = consume_seed(self.requested, feature)
                    if seeded is not None:
                        feature = seeded
                    else:
                        # Each later consumer plans its own copy, from the clean source when known.
                        held = feature
                        feature = (held.plan_source or held).copy_for_plan()
                        feature.uuid = uuid4()
                        # A group may set merge directives on the object it returns; the clean source lacks them.
                        feature.forward_group = held.forward_group
                        feature.forward_group_exclude = held.forward_group_exclude
                        feature.inherit_context_keys = held.inherit_context_keys
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
        Apply forward-by-default inheritance onto the input feature and record the forwarded
        group-key set on it. Conflict detection lives in Options.inherit_from; child_options
        is never mutated.
        """
        feature.forwarded_group_keys = feature._inherit_consumer_options(
            feature.options, child_options, self.consumer_local_keys
        )

    def check_duplicate_feature(self, feature: Feature) -> None:
        if feature in self.collection:
            raise ValueError(f"Duplicate feature setup: {feature.name}")

    def __iter__(self) -> Generator[Feature, None, None]:
        yield from self.collection

    def check_for_duplicate_string_features(self, features: Sequence[Feature | str]) -> None:
        check_set: set[str] = set()
        for feature in features:
            if isinstance(feature, str):
                if feature in check_set:
                    raise ValueError(f"You are adding same feature as string twice: {feature}")
                check_set.add(feature)
