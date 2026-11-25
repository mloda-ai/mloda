from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.options import Options


def test_same_feature_name_in_two_feature_groups_is_rejected() -> None:
    first = Feature(name="my_feature", options=Options({}), feature_group="group_a")
    second = Feature(name="my_feature", options=Options({}), feature_group="group_b")

    with pytest.raises(ValueError, match="Duplicate feature setup: my_feature"):
        Features([first, second])


def test_same_feature_name_in_same_feature_group_is_rejected() -> None:
    first = Feature(name="my_feature", options=Options({}), feature_group="group_a")
    second = Feature(name="my_feature", options=Options({}), feature_group="group_a")

    with pytest.raises(ValueError, match="Duplicate feature setup: my_feature"):
        Features([first, second])


def test_one_object_returned_twice_by_one_consumer_is_rejected() -> None:
    f = Feature(name="my_feature", options=Options({}))

    with pytest.raises(ValueError, match="Duplicate feature setup: my_feature"):
        Features([f, f], child_options=Options({}), child_uuid=uuid4())


def test_distinct_feature_names_in_different_feature_groups_are_allowed() -> None:
    first = Feature(name="feature_a", options=Options({}), feature_group="group_a")
    second = Feature(name="feature_b", options=Options({}), feature_group="group_b")

    collection = Features([first, second])

    assert [feature.name for feature in collection] == ["feature_a", "feature_b"]


def _seeded_pair() -> tuple[Feature, Feature]:
    plan_object = Feature(name="seed_once_feature", options=Options(group={"g": 1}, context={"k": "v"}))
    return plan_object.copy_for_plan(), plan_object


@pytest.mark.parametrize("key", ["plan_object", "source"])
def test_first_build_with_a_seed_plans_into_the_plan_object(key: str) -> None:
    source, plan_object = _seeded_pair()
    key_object = plan_object if key == "plan_object" else source
    requested: Any = {id(key_object): (source, plan_object)}

    collection = Features([key_object], child_options=Options({}), child_uuid=uuid4(), requested=requested, sources={})

    assert collection.collection[0] is plan_object


def test_second_build_reaching_a_consumed_seed_gets_a_copy_of_the_clean_source() -> None:
    source, plan_object = _seeded_pair()
    requested: Any = {id(plan_object): (source, plan_object)}
    sources: dict[int, Feature] = {}
    Features([plan_object], child_options=Options({}), child_uuid=uuid4(), requested=requested, sources=sources)
    plan_object.initial_requested_data = True
    plan_object.options.add_to_context("later", "x")

    second = Features(
        [plan_object], child_options=Options({}), child_uuid=uuid4(), requested=requested, sources=sources
    )

    copy_made = second.collection[0]
    assert copy_made is not plan_object
    assert copy_made.uuid != plan_object.uuid
    assert copy_made.options.get("k") == "v"
    assert copy_made.options.get("later") is None
    assert copy_made.initial_requested_data is False


def test_consuming_a_seed_moves_every_key_pointing_at_the_plan_object_into_sources() -> None:
    source, plan_object = _seeded_pair()
    requested: Any = {id(source): (source, plan_object), id(plan_object): (source, plan_object)}
    sources: dict[int, Feature] = {}

    Features([plan_object], child_options=Options({}), child_uuid=uuid4(), requested=requested, sources=sources)

    assert requested == {}
    assert sources == {id(source): source, id(plan_object): source}
    assert not hasattr(plan_object, "plan_source")
