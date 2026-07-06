"""Tests pinning that a bare STRING input feature starts clean (issue #579, gap 1).

``Features.build_feature_collection`` has a recursion path (invoked whenever
``child_uuid`` is set, i.e. when constructing a feature group's declared input
features) that historically wrapped a bare string input feature directly in the
consumer's ``child_options``:

    feature = Feature(name=feature, options=child_options, domain=self.parent_domain)

That construction hands the WHOLE consumer options object (group and context)
to the input feature before the allowlist-based merge (``forward_group`` /
``inherit_context_keys``) ever runs, so a plain string input feature inherits
everything unconditionally. A bare string must behave exactly like
``Feature(name)``: clean by default, with forwarding governed only by the
allowlist opt-ins.
"""

from __future__ import annotations

from uuid import uuid4

from mloda.provider import DefaultOptionKeys
from mloda.user import Feature
from mloda.user import Features
from mloda.user import Options


def test_bare_string_input_feature_does_not_inherit_consumer_group() -> None:
    """A bare string input feature must not inherit the consumer's group options."""
    consumer_options = Options(group={"top_k": 9, DefaultOptionKeys.in_features: "upstream"})

    features = Features(["upstream"], child_options=consumer_options, child_uuid=uuid4())

    resolved = features.collection[0]
    assert "top_k" not in resolved.options.group


def test_feature_object_input_feature_matches_bare_string_behavior() -> None:
    """The Feature(name) form of the same input feature must match the bare string form.

    Both forms go through the same recursion path (``child_uuid`` set) and must
    end up with an equally clean ``options.group`` (no leaked "top_k").
    """
    consumer_options = Options(group={"top_k": 9, DefaultOptionKeys.in_features: "upstream"})

    features = Features([Feature("upstream")], child_options=consumer_options, child_uuid=uuid4())

    resolved = features.collection[0]
    assert "top_k" not in resolved.options.group


def test_top_level_string_features_without_child_uuid_still_construct() -> None:
    """Top-level string features (no child_uuid, no recursion) still build without error."""
    features = Features(["income", "age"])

    assert len(features.collection) == 2
    assert features.collection[0].name == Feature("income").name
    assert features.collection[1].name == Feature("age").name
