import pytest
from mloda_core.abstract_plugins.components.feature_name import FeatureName

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.options import Options


def test_features_init() -> None:
    feature1 = Feature(name="Feature1", options={"option1": 1})
    feature2 = Feature(name="Feature2", options={"option2": 2})
    collection = Features([feature1, feature2, "Feature3_str"])
    assert len(collection.collection) == 3
    assert collection.collection[0] == feature1
    assert collection.collection[1] == feature2
    assert collection.collection[2].name == FeatureName("Feature3_str")
    assert collection.collection[2].options == Options({})
    assert collection.collection[2].child_options is None


def test_features_duplication_check() -> None:
    feature1 = Feature(name="Feature1", options={"option1": 1})
    feature2 = Feature(name="Feature2", options={"option2": 2})
    duplicate_feature = Feature(name="Feature1", options={"option1": 1})
    with pytest.raises(ValueError):
        Features([feature1, feature2, duplicate_feature])

    feature3 = Feature(name="Feature2", options={"option2": 3})
    Features([feature2, feature3])
