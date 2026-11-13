from __future__ import annotations

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


def test_distinct_feature_names_in_different_feature_groups_are_allowed() -> None:
    first = Feature(name="feature_a", options=Options({}), feature_group="group_a")
    second = Feature(name="feature_b", options=Options({}), feature_group="group_b")

    collection = Features([first, second])

    assert [feature.name for feature in collection] == ["feature_a", "feature_b"]
