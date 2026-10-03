import copy
import enum
import pickle  # nosec B403
from typing import Any

import pytest
from mloda.provider import ComputeFramework
from mloda.user import Feature
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.options import Options
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


def test_feature_set_domain_accepts_already_built_domain() -> None:
    """Passing a Domain instance for domain must not re-wrap it."""
    feature = Feature(name="Feature1")

    result = feature._set_domain(Domain("example_domain"), None)
    assert result.name == "example_domain"  # type: ignore


def test_feature_set_domain_options_accepts_already_built_domain() -> None:
    """Passing a Domain instance for domain_options must not re-wrap it."""
    feature = Feature(name="Feature1")

    result = feature._set_domain(None, Domain("example_domain_options"))
    assert result.name == "example_domain_options"  # type: ignore


def test_feature_domain_param_matches_domain_str_param() -> None:
    """Feature(domain=Domain(...)) must resolve identically to Feature(domain="...")."""
    assert Feature("x", domain=Domain("m")) == Feature("x", domain="m")


def test_feature_domain_options_matches_domain_str_param() -> None:
    """Feature(options={"domain": Domain(...)}) must resolve identically to Feature(domain="...")."""
    assert Feature("x", options={"domain": Domain("m")}) == Feature("x", domain="m")


def test_feature_construction_does_not_mutate_a_shared_options_object() -> None:
    """A caller-owned Options instance reused across features must keep its "domain" key."""
    shared = Options({"domain": "sales"})

    first = Feature("a", shared)
    second = Feature("b", shared)

    assert shared.group.get("domain") == "sales"
    assert first.domain.name == "sales"  # type: ignore
    assert second.domain.name == "sales"  # type: ignore


def test_feature_domain_param_does_not_double_wrap() -> None:
    """feature.domain.name must be the plain str, not a nested Domain instance."""
    feature = Feature("x", domain=Domain("m"))
    assert isinstance(feature.domain.name, str)  # type: ignore
    assert feature.domain.name == "m"  # type: ignore


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


# required_declarations: a resolver-read constraint, excluded from identity


class _ReqUnit(str):
    """A str subclass a caller might pass as a required value."""


class _ReqLevel(enum.IntEnum):
    """An int-subclass enum a caller might pass as a required value."""

    THREE = 3


TYPED_HELPERS = [
    "str_of",
    "int32_of",
    "int64_of",
    "float_of",
    "double_of",
    "boolean_of",
    "binary_of",
    "date_of",
    "timestamp_millis_of",
    "timestamp_micros_of",
    "decimal_of",
]


def test_required_declarations_defaults_to_none() -> None:
    assert Feature("subject_token").required_declarations is None


def test_required_declarations_stored_as_dict() -> None:
    feature = Feature("subject_token", required_declarations={"scale": None, "unit": "m"})
    assert feature.required_declarations == {"scale": None, "unit": "m"}
    assert type(feature.required_declarations) is dict


def test_required_declarations_empty_mapping_is_none() -> None:
    assert Feature("subject_token", required_declarations={}).required_declarations is None


def test_required_declarations_none_is_none() -> None:
    assert Feature("subject_token", required_declarations=None).required_declarations is None


def test_required_declarations_value_is_copied() -> None:
    required: dict[str, str | int | float | bool | None] = {"scale": None}
    feature = Feature("subject_token", required_declarations=required)
    required["unit"] = "m"
    assert feature.required_declarations == {"scale": None}


def test_required_declarations_excluded_from_equality() -> None:
    assert Feature("subject_token", required_declarations={"scale": None}) == Feature("subject_token")


def test_required_declarations_excluded_from_hash() -> None:
    required = Feature("subject_token", required_declarations={"scale": None})
    plain = Feature("subject_token")
    assert hash(required) == hash(plain)
    assert len({required, plain}) == 1


def test_required_declarations_excluded_from_similarity_hash() -> None:
    required = Feature("subject_token", required_declarations={"scale": None})
    plain = Feature("subject_token")
    assert required.similarity_hash() == plain.similarity_hash()
    assert required.base_similarity_hash() == plain.base_similarity_hash()


def test_required_declarations_kept_by_copy() -> None:
    feature = Feature("subject_token", required_declarations={"scale": None})
    assert copy.copy(feature).required_declarations == {"scale": None}


def test_required_declarations_kept_by_deepcopy() -> None:
    feature = Feature("subject_token", required_declarations={"scale": None})
    assert copy.deepcopy(feature).required_declarations == {"scale": None}


def test_required_declarations_kept_by_pickle_round_trip() -> None:
    feature = Feature("subject_token", required_declarations={"scale": None, "unit": "m"})
    assert pickle.loads(pickle.dumps(feature)).required_declarations == {"scale": None, "unit": "m"}  # nosec B301


def test_required_declarations_non_mapping_raises_typeerror() -> None:
    assert Feature("subject_token", required_declarations={"scale": None}).required_declarations
    with pytest.raises(TypeError):
        Feature("subject_token", required_declarations=["scale"])  # type: ignore[arg-type]


def test_required_declarations_non_str_key_raises_typeerror() -> None:
    assert Feature("subject_token", required_declarations={"scale": None}).required_declarations
    with pytest.raises(TypeError):
        Feature("subject_token", required_declarations={1: None})  # type: ignore[dict-item]


@pytest.mark.parametrize("bad", [["a"], {"a": 1}], ids=["list", "dict"])
def test_required_declarations_non_scalar_value_raises_typeerror(bad: object) -> None:
    assert Feature("subject_token", required_declarations={"scale": None}).required_declarations
    with pytest.raises(TypeError):
        Feature("subject_token", required_declarations={"scale": bad})  # type: ignore[dict-item]


def test_required_declarations_scalar_subclasses_are_normalized() -> None:
    feature = Feature(
        "subject_token",
        required_declarations={"unit": _ReqUnit("m"), "level": _ReqLevel.THREE, "flag": True},
    )
    required = feature.required_declarations
    assert required == {"unit": "m", "level": 3, "flag": True}
    assert required is not None
    assert type(required["unit"]) is str
    assert type(required["level"]) is int
    assert type(required["flag"]) is bool


def test_required_declarations_positional_callers_unaffected() -> None:
    """Passing every earlier parameter positionally leaves required_declarations None."""
    feature = Feature("subject_token", {"k": "v"}, None, None, None, False, None, None, None, None, None, frozenset())
    assert feature.required_declarations is None
    assert feature.options.get("k") == "v"


def test_not_typed_stores_required_declarations() -> None:
    feature = Feature.not_typed("subject_token", required_declarations={"scale": None, "unit": "m"})
    assert feature.required_declarations == {"scale": None, "unit": "m"}
    assert type(feature.required_declarations) is dict


def test_not_typed_empty_required_declarations_is_none() -> None:
    assert Feature.not_typed("subject_token", required_declarations={}).required_declarations is None


def test_not_typed_required_declarations_are_validated() -> None:
    assert Feature.not_typed("subject_token", required_declarations={"scale": None}).required_declarations
    with pytest.raises(TypeError):
        Feature.not_typed("subject_token", required_declarations={"scale": [1]})  # type: ignore[dict-item]


@pytest.mark.parametrize("helper", TYPED_HELPERS)
def test_typed_helpers_store_required_declarations(helper: str) -> None:
    feature = getattr(Feature, helper)("subject_token", required_declarations={"scale": None})
    assert feature.required_declarations == {"scale": None}


def test_typed_helper_required_declarations_are_validated() -> None:
    assert Feature.double_of("subject_token", required_declarations={"scale": None}).required_declarations
    with pytest.raises(TypeError):
        Feature.double_of("subject_token", required_declarations=["scale"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Feature.double_of("subject_token", required_declarations={1: None})  # type: ignore[dict-item]


class FeatureMatchReaderA(BaseInputData):
    """Reader stub for input_data_match identity tests."""


class FeatureMatchReaderB(BaseInputData):
    """Second reader stub, same name and options but another source kind."""


def _matched(reader: type[BaseInputData], access: Any) -> Feature:
    feature = Feature("feature_match_col", options={"opt": 1})
    feature.input_data_match = (reader, access)
    return feature


def test_input_data_match_defaults_to_none() -> None:
    assert Feature("feature_match_col").input_data_match is None


def test_features_matched_to_different_sources_are_not_equal() -> None:
    assert _matched(FeatureMatchReaderA, {"alpha": 1}) != _matched(FeatureMatchReaderA, {"beta": 1})
    assert _matched(FeatureMatchReaderA, {"alpha": 1}) != _matched(FeatureMatchReaderB, {"alpha": 1})
    assert _matched(FeatureMatchReaderA, {"alpha": 1}) != Feature("feature_match_col", options={"opt": 1})


def test_features_differing_only_in_a_secret_value_are_equal() -> None:
    one = _matched(FeatureMatchReaderA, {"user": "u", "password": "secret-one"})  # nosec B105
    two = _matched(FeatureMatchReaderA, {"user": "u", "password": "secret-two"})  # nosec B105
    assert one == two
    assert hash(one) == hash(two)


def test_hash_differs_between_sources() -> None:
    assert hash(_matched(FeatureMatchReaderA, {"alpha": 1})) != hash(_matched(FeatureMatchReaderA, {"beta": 1}))


def test_similarity_hash_includes_the_full_pair() -> None:
    one = _matched(FeatureMatchReaderA, "secret-one")
    two = _matched(FeatureMatchReaderA, "secret-two")
    assert one == two
    assert one.similarity_hash() != two.similarity_hash()
    assert one.base_similarity_hash() != two.base_similarity_hash()


def test_similarity_hash_is_stable_for_the_same_pair() -> None:
    one = _matched(FeatureMatchReaderA, "same")
    two = _matched(FeatureMatchReaderA, "same")
    assert one.similarity_hash() == two.similarity_hash()
    assert one.base_similarity_hash() == two.base_similarity_hash()
