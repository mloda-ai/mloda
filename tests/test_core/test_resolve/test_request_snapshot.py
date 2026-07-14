"""Failing tests for issue #722 Stage 2: the immutable request state API.

Defines the contract of ``mloda.core.resolve.request``:
``ResolutionRequestSnapshot.from_feature`` captures what per-feature matching
consumes (name, domain, scope, framework pin, option key sets, dependency path)
at build time, carries ``Options`` as an opaque hook input that the resolver
never mutates, and ``to_payload()`` redacts option values, the ``Options``
object, the ``DataAccessCollection``, and link internals.

RED phase: ``mloda.core.resolve`` does not exist yet, so this module fails at
collection with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.resolve.request import ResolutionRequestSnapshot


class CfwRequestPin722D(ComputeFramework):
    """Uniquely named framework, pinnable by name from a Feature."""


class _RequestProbeBase722D(FeatureGroup):
    """Shared probe base: never matches anything itself."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False


class ProbeLinkLeft722D(_RequestProbeBase722D):
    """Anchors the left side of the request link; never matches a name."""


class ProbeLinkRight722D(_RequestProbeBase722D):
    """Anchors the right side of the request link; never matches a name."""


def _probe_link() -> Link:
    return Link.inner(
        JoinSpec(ProbeLinkLeft722D, "probe722d_left_id"),
        JoinSpec(ProbeLinkRight722D, "probe722d_right_id"),
    )


def test_from_feature_captures_request_fields() -> None:
    """from_feature captures name, domain, scope, framework pin, and option key sets."""
    feature = Feature(
        "probe722d_request_full",
        Options(group={"group_key_722d": "gval"}, context={"context_key_722d": "cval"}),
        domain="domain_722d",
        compute_framework="CfwRequestPin722D",
        feature_group="ProbeScopeTarget722D",
    )

    snapshot = ResolutionRequestSnapshot.from_feature(feature)

    assert snapshot.feature_name == "probe722d_request_full"
    assert snapshot.domain == "domain_722d"
    assert snapshot.feature_group_scope == "ProbeScopeTarget722D"
    assert snapshot.framework_pin == "CfwRequestPin722D"
    assert snapshot.pinned_frameworks == (CfwRequestPin722D,)
    assert snapshot.group_option_keys == frozenset({"group_key_722d"})
    assert snapshot.context_option_keys == frozenset({"context_key_722d"})
    assert snapshot.inherited_group_keys == frozenset()
    assert snapshot.dependency_path == ()
    assert snapshot.links == ()
    assert snapshot.data_access_collection is None


def test_from_feature_captures_inherited_group_keys() -> None:
    """Keys forwarded by Options.inherit_from (the Features.merge_options mechanism) are captured."""
    consumer = Options(group={"forwarded_722d": "shared-value"})
    child = Options()
    forwarded = child.inherit_from(consumer)
    assert forwarded == frozenset({"forwarded_722d"})  # premise guard
    assert child.inherited_group_keys == frozenset({"forwarded_722d"})  # premise guard

    feature = Feature("probe722d_inherited", child)
    snapshot = ResolutionRequestSnapshot.from_feature(feature)

    assert snapshot.inherited_group_keys == frozenset({"forwarded_722d"})
    assert snapshot.group_option_keys == frozenset({"forwarded_722d"})


def test_snapshot_is_isolated_from_later_feature_mutation() -> None:
    """Captured fields do not change when the feature or its options mutate afterwards."""
    options = Options(group={"initial_key_722d": 1})
    feature = Feature("probe722d_mutation", options)

    snapshot = ResolutionRequestSnapshot.from_feature(feature)

    feature.name = FeatureName("probe722d_mutation_renamed")
    options.add_to_group("late_key_722d", 2)

    assert snapshot.feature_name == "probe722d_mutation"
    assert snapshot.group_option_keys == frozenset({"initial_key_722d"})


def test_snapshot_is_frozen() -> None:
    """Snapshot fields cannot be reassigned."""
    feature = Feature("probe722d_frozen")
    snapshot = ResolutionRequestSnapshot.from_feature(feature)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snapshot, "feature_name", "probe722d_forged")


def test_payload_redacts_option_values_and_data_access() -> None:
    """to_payload carries option KEYS but never values, the collection, or link internals."""
    feature = Feature("probe722d_redaction", Options(group={"password_722d": "hunter2-722d"}))  # nosec B105
    collection = DataAccessCollection(files={"probe722d_secret_file.csv"})

    snapshot = ResolutionRequestSnapshot.from_feature(feature, links={_probe_link()}, data_access_collection=collection)
    assert snapshot.data_access_collection is collection
    assert len(snapshot.links) == 1

    payload = snapshot.to_payload()
    dumped = json.dumps(payload)

    assert "password_722d" in dumped
    assert "hunter2-722d" not in dumped
    assert "data_access_collection" not in payload
    assert isinstance(payload["links"], int)
    assert payload["links"] == 1


def test_dependency_path_round_trips_into_payload() -> None:
    """The dependency path is captured as a tuple and rendered as a list in the payload."""
    feature = Feature("probe722d_dependency")

    snapshot = ResolutionRequestSnapshot.from_feature(feature, dependency_path=("root_722d", "child_722d"))

    assert snapshot.dependency_path == ("root_722d", "child_722d")
    assert snapshot.to_payload()["dependency_path"] == ["root_722d", "child_722d"]


def test_options_are_carried_as_the_same_object() -> None:
    """snapshot.options IS feature.options: an opaque hook input the resolver never mutates."""
    feature = Feature("probe722d_opaque", Options(group={"opaque_key_722d": "opaque-value-722d"}))

    snapshot = ResolutionRequestSnapshot.from_feature(feature)

    assert snapshot.options is feature.options


def test_payload_option_key_lists_are_sorted() -> None:
    """Payload option key lists are deterministic: sorted, under stable payload keys."""
    feature = Feature(
        "probe722d_sorted",
        Options(
            group={"b_group_722d": 1, "a_group_722d": 2},
            context={"d_context_722d": 3, "c_context_722d": 4},
        ),
    )

    payload = ResolutionRequestSnapshot.from_feature(feature).to_payload()

    assert payload["group_option_keys"] == ["a_group_722d", "b_group_722d"]
    assert payload["context_option_keys"] == ["c_context_722d", "d_context_722d"]
    expected_keys = {
        "feature_name",
        "domain",
        "feature_group_scope",
        "framework_pin",
        "group_option_keys",
        "context_option_keys",
        "inherited_group_keys",
        "dependency_path",
        "links",
    }
    assert expected_keys <= set(payload.keys())
