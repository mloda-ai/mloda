"""Tests for FeatureName as a str subclass."""

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


class TestFeatureNameIsStr:
    def test_isinstance_str(self) -> None:
        assert isinstance(FeatureName("x"), str)

    def test_direct_string_operations(self) -> None:
        fn = FeatureName("income")
        assert fn.upper() == "INCOME"
        assert fn.startswith("inc")
        assert len(fn) == 6

    def test_identity_for_existing_feature_name(self) -> None:
        fn = FeatureName("x")
        assert FeatureName(fn) is fn


class TestFeatureNameRepr:
    def test_repr_format(self) -> None:
        assert repr(FeatureName("income")) == "FeatureName('income')"

    def test_str_returns_value(self) -> None:
        assert str(FeatureName("income")) == "income"


class TestFeatureNameEquality:
    def test_eq_with_str(self) -> None:
        assert FeatureName("x") == "x"

    def test_eq_str_reverse(self) -> None:
        assert "x" == FeatureName("x")

    def test_eq_with_feature_name(self) -> None:
        assert FeatureName("x") == FeatureName("x")


class TestFeatureNameHash:
    def test_hash_matches_str(self) -> None:
        assert hash(FeatureName("x")) == hash("x")

    def test_usable_as_dict_key(self) -> None:
        d: dict[str, int] = {FeatureName("k"): 1}
        assert d["k"] == 1

    def test_usable_in_set(self) -> None:
        s = {FeatureName("a"), "a"}
        assert len(s) == 1


class TestFeatureNameContains:
    def test_substring_check(self) -> None:
        fn = FeatureName("mean__income")
        assert "income" in fn
        assert "age" not in fn


class TestFeatureNameWithEnum:
    def test_enum_value_extracted(self) -> None:
        fn = FeatureName(DefaultOptionKeys.reference_time)
        assert fn == "reference_time"
        assert fn == DefaultOptionKeys.reference_time.value

    def test_enum_repr(self) -> None:
        fn = FeatureName(DefaultOptionKeys.reference_time)
        assert repr(fn) == "FeatureName('reference_time')"

    def test_enum_value_extracted_for_time_travel(self) -> None:
        fn = FeatureName(DefaultOptionKeys.time_travel)
        assert fn == "time_travel"
        assert fn == DefaultOptionKeys.time_travel.value


class TestFeatureNameAccess:
    def test_feature_dot_name_is_str(self) -> None:
        f = Feature("income")
        assert isinstance(f.name, str)
        assert f.name == "income"

    def test_no_double_name_needed(self) -> None:
        f = Feature("income")
        name: str = f.name
        assert name == "income"

    def test_get_name_removed(self) -> None:
        assert not hasattr(Feature, "get_name")
        assert not hasattr(FeatureName, "get_name")
