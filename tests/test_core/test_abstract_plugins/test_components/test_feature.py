import pytest
from mloda.provider import ComputeFramework
from mloda.user import Feature
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401


def test_feature_equals() -> None:
    feature1 = Feature(name="Feature1", options={"option1": 1})
    feature2 = Feature(name="Feature1", options={"option1": 1})
    feature3 = Feature(name="Feature2", options={"option2": 2})
    assert feature1 == feature2
    assert feature1 != feature3


def test_feature_set_compute_framework() -> None:
    feature = Feature(name="Feature1")

    # Test if frameworks are not found
    with pytest.raises(ValueError):
        feature._set_compute_framework(None, "ComputeFrameworkNotExists")
    with pytest.raises(ValueError):
        feature._set_compute_framework("ComputeFrameworkNotExists", None)

    # Test when neither compute_framework nor compute_framework_options are set
    assert feature._set_compute_framework(None, None) is None

    # Test valid cases
    valid_fw_subclases = get_all_subclasses(ComputeFramework)
    result1 = feature._set_compute_framework(next(iter(valid_fw_subclases)).get_class_name(), None)
    result2 = feature._set_compute_framework(None, next(iter(valid_fw_subclases)).get_class_name())
    assert result1 == result2 is not None


def test_feature_set_domain() -> None:
    feature = Feature(name="Feature1")

    # Test when domain is set
    result = feature._set_domain("example_domain", None)
    assert result.name == "example_domain"  # type: ignore

    # Test when domain_options is set
    result = feature._set_domain(None, "example_domain_options")
    assert result.name == "example_domain_options"  # type: ignore

    # Test when neither domain nor domain_options are set
    assert feature._set_domain(None, None) is None


def test_feature_data_type_from_string() -> None:
    for name in ("INT32", "FLOAT", "STRING", "BOOLEAN"):
        feature = Feature("f", data_type=name)
        assert feature.data_type == DataType(name)


def test_feature_data_type_from_enum() -> None:
    feature = Feature("f", data_type=DataType.INT32)
    assert feature.data_type == DataType.INT32


def test_feature_data_type_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        Feature("f", data_type="NOT_A_TYPE")


def test_feature_data_type_none_default() -> None:
    feature = Feature("f")
    assert feature.data_type is None


def test_feature_data_type_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        Feature("f", data_type=42)  # type: ignore[arg-type]


def test_feature_data_type_string_equals_enum() -> None:
    f1 = Feature("f", data_type="INT32")
    f2 = Feature("f", data_type=DataType.INT32)
    assert f1 == f2
    assert hash(f1) == hash(f2)
