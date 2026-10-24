import copy
from copy import deepcopy
from itertools import islice
from typing import Any

import pytest
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.provider import SourceFeatureGroup
from mloda.user import Credential, Feature, Options
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToySourceBase,
    ToySourceFG,
)

IN_FEATURES = DefaultOptionKeys.in_features.value


class _RaisesOnDeepcopy:
    """Value whose deep-copy raises, mimicking the Python 3.14 uuid.UUID KeyError('int')."""

    def __deepcopy__(self, memo: dict[int, object]) -> "_RaisesOnDeepcopy":
        raise KeyError("int")


class _HoldsUncopyable:
    """Deep-copies via __reduce_ex__, so copy memoizes an empty instance before copying state fails."""

    def __init__(self, nested: "_HoldsUncopyable | None" = None) -> None:
        self.nested = nested
        self.handle = _RaisesOnDeepcopy()


class TestOptions:
    """Test suite for the new Options class with group/context separation."""

    def test_legacy_initialization(self) -> None:
        """Legacy initialization should move all data to group."""
        options = Options({"key1": "value1", "key2": "value2"})
        assert options.group == {"key1": "value1", "key2": "value2"}
        assert options.context == {}

    def test_new_initialization(self) -> None:
        """Test new group/context initialization."""
        options = Options(
            group={"data_source": "prod", "environment": "staging"},
            context={"aggregation_type": "sum", "debug_mode": True},
        )

        assert options.group == {"data_source": "prod", "environment": "staging"}
        assert options.context == {"aggregation_type": "sum", "debug_mode": True}

    def test_unknown_parameter_error(self) -> None:
        """Test that using unknown parameters raises TypeError."""
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            Options(data={"key": "value"})  # type: ignore

    def test_duplicate_keys_validation(self) -> None:
        """Test that duplicate keys in group and context raise error."""
        with pytest.raises(ValueError, match="Keys cannot exist in both group and context"):
            Options(group={"shared_key": "value1"}, context={"shared_key": "value2"})

    def test_add_to_group(self) -> None:
        """Test adding parameters to group."""
        options = Options()
        options.add_to_group("data_source", "prod")
        options.add_to_group("environment", "staging")

        assert options.group == {"data_source": "prod", "environment": "staging"}
        assert options.context == {}

    def test_add_to_context(self) -> None:
        """Test adding parameters to context."""
        options = Options()
        options.add_to_context("aggregation_type", "sum")
        options.add_to_context("debug_mode", True)

        assert options.group == {}
        assert options.context == {"aggregation_type": "sum", "debug_mode": True}

    def test_add_duplicate_key_error(self) -> None:
        """Test that adding duplicate keys raises error."""
        options = Options()
        options.add_to_group("key1", "value1")

        # Try to add same key to group
        with pytest.raises(ValueError, match="Key key1 already exists in group options"):
            options.add_to_group("key1", "value2")

        # Try to add same key to context
        with pytest.raises(ValueError, match="Key key1 already exists in group options"):
            options.add_to_context("key1", "value2")

    def test_add_cross_contamination_error(self) -> None:
        """Test that adding key that exists in other section raises error."""
        options = Options()
        options.add_to_group("shared_key", "group_value")

        with pytest.raises(ValueError, match="Key shared_key already exists in group options"):
            options.add_to_context("shared_key", "context_value")

    def test_equality_based_on_group_only(self) -> None:
        """Test that equality is based only on group parameters."""
        options1 = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})
        options2 = Options(
            group={"data_source": "prod"},
            context={"aggregation_type": "avg"},  # Different context
        )
        options3 = Options(
            group={"data_source": "staging"},  # Different group
            context={"aggregation_type": "sum"},
        )

        # Same group, different context -> equal
        assert options1 == options2

        # Different group, same context -> not equal
        assert options1 != options3
        assert options2 != options3

    def test_hash_based_on_group_only(self) -> None:
        """Test that hash is based only on group parameters."""
        options1 = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})
        options2 = Options(
            group={"data_source": "prod"},
            context={"aggregation_type": "avg"},  # Different context
        )
        options3 = Options(
            group={"data_source": "staging"},  # Different group
            context={"aggregation_type": "sum"},
        )

        # Same group, different context -> same hash
        assert hash(options1) == hash(options2)

        # Different group -> different hash
        assert hash(options1) != hash(options3)

    def test_get_legacy_method(self) -> None:
        """Test legacy get method searches both group and context."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})

        # Should find in group first
        assert options.get("group_key") == "group_value"

        # Should find in context if not in group
        assert options.get("context_key") == "context_value"

        # Should return None if not found
        assert options.get("nonexistent_key") is None

    def test_str_representation(self) -> None:
        """Test string representation shows both group and context."""
        options = Options(group={"data_source": "prod"}, context={"aggregation_type": "sum"})

        expected = "Options(group={'data_source': 'prod'}, context={'aggregation_type': 'sum'})"
        assert str(options) == expected

    def test_backward_compatibility_with_feature_class(self) -> None:
        """Test that Options work correctly with Feature class equality."""
        from mloda.user import Feature

        # Features with same group options should be equal
        feature1 = Feature("test_feature", Options(group={"data_source": "prod"}))
        feature2 = Feature("test_feature", Options(group={"data_source": "prod"}))

        assert feature1 == feature2

        # Features with different group options should not be equal
        feature3 = Feature("test_feature", Options(group={"data_source": "staging"}))

        assert feature1 != feature3

    def test_getitem_group_key(self) -> None:
        """options["key"] returns same value as options.get("key") for group key."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})
        assert options["group_key"] == options.get("group_key")

    def test_getitem_context_key(self) -> None:
        """options["key"] returns same value as options.get("key") for context key."""
        options = Options(group={"group_key": "group_value"}, context={"context_key": "context_value"})
        assert options["context_key"] == options.get("context_key")

    def test_getitem_missing_key_returns_none(self) -> None:
        """options["missing"] returns None, consistent with get()."""
        options = Options(group={"key": "value"})
        assert options["missing"] is None

    def test_migration_scenario(self) -> None:
        """Test typical migration scenario: all options start in group."""
        # Current usage pattern (all options in group during migration)
        options = Options(
            group={
                "data_source": "prod",
                "aggregation_type": "sum",
                "in_features": "sales_data",
                "debug_mode": True,
            }
        )

        # Verify all options are in group (maintaining current behavior)
        assert len(options.group) == 4
        assert len(options.context) == 0

        # Legacy get method should still work
        assert options.get("data_source") == "prod"
        assert options.get("aggregation_type") == "sum"

        # Future optimization: move some parameters to context
        optimized_options = Options(
            group={"data_source": "prod"},  # Only isolation-requiring parameters
            context={  # Metadata parameters
                "aggregation_type": "sum",
                "in_features": "sales_data",
                "debug_mode": True,
            },
        )

        # These should NOT be equal (different group parameters)
        assert options != optimized_options

        # But legacy get should still work for both
        assert optimized_options.get("data_source") == "prod"
        assert optimized_options.get("aggregation_type") == "sum"

    @pytest.mark.parametrize(
        "options, expected",
        [(Options(group={"g": 1}, context={"c": 2}), ["g", "c"]), (Options(), [])],
    )
    def test_iter_yields_keys(self, options: Options, expected: list[str]) -> None:
        assert list(islice(options, 5)) == expected
        assert set(islice(options, 5)) == set(expected)


_KEY_ARGUMENTS = ["propagate_context_keys", "cascade_context_keys"]


def _options_with_keys(argument: str, context: dict[str, Any], keys: frozenset[str]) -> Options:
    if argument == "cascade_context_keys":
        return Options(context=context, cascade_context_keys=keys)
    return Options(context=context, propagate_context_keys=keys)


class TestPropagateContextKeys:
    """
    Test suite for the propagate_context_keys attribute on Options.

    WHY THIS EXISTS:
    When features are chained (parent -> child), context keys are normally
    scoped to the feature that defined them. propagate_context_keys explicitly
    marks which context keys should flow through to downstream features in
    the chain. Only keys listed here will be propagated; all others stay local.
    """

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_default_empty(self, argument: str) -> None:
        """Default key set should be an empty frozenset."""
        options = Options()
        assert getattr(options, argument) == frozenset()

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_stored(self, argument: str) -> None:
        """The key set should store the provided frozenset."""
        options = _options_with_keys(argument, {"k": "v"}, frozenset({"k"}))
        assert getattr(options, argument) == frozenset({"k"})
        assert isinstance(getattr(options, argument), frozenset)

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_validates_subset(self, argument: str) -> None:
        """The key set must be a subset of context keys, else ValueError naming the argument."""
        with pytest.raises(ValueError, match=argument):
            _options_with_keys(argument, {"a": 1}, frozenset({"b"}))

    def test_propagate_error_text_does_not_mention_cascade(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Options(context={"a": 1}, propagate_context_keys=frozenset({"b"}))
        assert "cascade_context_keys" not in str(excinfo.value)

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_not_in_hash(self, argument: str) -> None:
        """The key set should not affect hash (same group+context, different key set)."""
        o1 = _options_with_keys(argument, {"k": "v"}, frozenset({"k"}))
        o2 = Options(context={"k": "v"})
        assert hash(o1) == hash(o2)

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_not_in_eq(self, argument: str) -> None:
        """The key set should not affect equality (same group+context, different key set)."""
        o1 = _options_with_keys(argument, {"k": "v"}, frozenset({"k"}))
        o2 = Options(context={"k": "v"})
        assert o1 == o2

    @pytest.mark.parametrize("argument", _KEY_ARGUMENTS)
    def test_propagate_context_keys_deepcopy(self, argument: str) -> None:
        """deepcopy should preserve the key set."""
        o = _options_with_keys(argument, {"k": "v"}, frozenset({"k"}))
        o2 = deepcopy(o)
        assert getattr(o2, argument) == frozenset({"k"})

    def test_str_shows_cascade_context_keys_only_when_non_empty(self) -> None:
        assert "cascade_context_keys=" not in str(Options(context={"k": "v"}))
        assert "cascade_context_keys=" in str(Options(context={"k": "v"}, cascade_context_keys=frozenset({"k"})))

    def test_deepcopy_falls_back_to_reference_when_value_deepcopy_raises(self) -> None:
        """A value whose deep-copy raises ANY exception must be shared by reference, and
        copy.deepcopy(Options(...)) must never propagate that exception (Python 3.14 uuid.UUID
        raises KeyError('int'), which _safe_deepcopy must swallow per-value)."""
        sentinel = _RaisesOnDeepcopy()
        opts = Options(group={"g": sentinel, "d": {"n": 1}}, context={"c": sentinel})

        copied = copy.deepcopy(opts)

        # Un-deep-copyable values fall back to being shared by reference.
        assert copied.group["g"] is sentinel
        assert copied.context["c"] is sentinel
        # Sibling values in the same dict are still independently deep-copied (per-value fallback).
        assert copied.group["d"] is not opts.group["d"]
        assert copied.group["d"] == {"n": 1}

    def test_value_whose_deepcopy_fails_partway_stays_the_original_across_features(self) -> None:
        """A failed deepcopy (e.g. a SparkSession) must not leave a half-built copy in the shared memo
        for a later reference to it, or to an object nested in it, to pick up."""
        inner = _HoldsUncopyable()
        outer = _HoldsUncopyable(nested=inner)
        sibling = {"n": 1}
        features = [
            Feature("a", options={"k": outer, "s": sibling}),
            Feature("b", options={"k": outer}),
            Feature("c", options={"k": inner}),
        ]

        copied = deepcopy(features)

        assert copied[0].options.get("k") is outer
        assert copied[1].options.get("k") is outer
        assert copied[2].options.get("k") is inner
        assert copied[0].options.get("s") is not sibling
        assert copied[0].options.get("s") == sibling


class TestContextPropagationIntegration:
    """Integration tests for context propagation through merge_options."""

    def test_merge_options_propagates_specified_context_keys(self) -> None:
        """Test that merge_options propagates only specified context keys."""
        from mloda.core.abstract_plugins.components.feature_collection import Features
        from mloda.user import Feature

        dependency_options = Options(
            context={"session_id": "abc", "algo": "sum"},
            propagate_context_keys=frozenset({"session_id"}),
        )

        feature = Feature("input_feature")

        fc = Features([])
        fc.merge_options(feature, dependency_options)

        assert feature.options.context["session_id"] == "abc"
        assert "algo" not in feature.options.context


class TestOptionsNestedStructureHashing:
    """
    Test suite for Options hashing with nested structures.

    WHY THIS EXISTS:
    Options.group may contain nested dicts, lists, or sets for complex configuration.
    The _deep_hashable helper recursively converts these to hashable equivalents,
    enabling proper hashing and use in sets/dicts for feature group resolution.
    """

    def test_hash_with_simple_values_baseline(self) -> None:
        """
        Baseline test: Simple hashable values should work.

        This test should PASS with the current implementation.
        """
        options = Options(group={"key": "value", "number": 42})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_nested_dict_in_group(self) -> None:
        """
        Test hashing Options with nested dict in group.

        The current implementation fails with TypeError: unhashable type: 'dict'
        because frozenset cannot hash dict values.
        """
        options = Options(group={"nested": {"key": "value"}})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_list_in_group(self) -> None:
        """
        Test hashing Options with list in group.

        The current implementation fails with TypeError: unhashable type: 'list'
        because frozenset cannot hash list values.
        """
        options = Options(group={"list": [1, 2, 3]})
        result = hash(options)
        assert isinstance(result, int)

    def test_hash_with_set_in_group(self) -> None:
        """
        Test hashing Options with set in group.

        The current implementation fails with TypeError: unhashable type: 'set'
        because frozenset cannot hash set values.
        """
        options = Options(group={"set_value": {1, 2, 3}})
        result = hash(options)
        assert isinstance(result, int)

    def test_equal_options_with_nested_dict_produce_equal_hashes(self) -> None:
        """
        Test that equal Options with nested structures produce equal hashes.

        Two Options with identical nested dicts should have the same hash.
        """
        options1 = Options(group={"nested": {"key": "value", "other": 123}})
        options2 = Options(group={"nested": {"key": "value", "other": 123}})

        assert options1 == options2
        assert hash(options1) == hash(options2)

    def test_equal_options_with_nested_list_produce_equal_hashes(self) -> None:
        """
        Test that equal Options with nested lists produce equal hashes.

        Two Options with identical lists should have the same hash.
        """
        options1 = Options(group={"items": [1, 2, 3]})
        options2 = Options(group={"items": [1, 2, 3]})

        assert options1 == options2
        assert hash(options1) == hash(options2)

    def test_deeply_nested_structure_hashes_successfully(self) -> None:
        """
        Test hashing Options with deeply nested structures.

        Complex nested structures combining dicts and lists should be hashable.
        """
        options = Options(
            group={
                "config": {
                    "nested": {"deep": {"value": 42}},
                    "list_of_dicts": [{"a": 1}, {"b": 2}],
                }
            }
        )
        result = hash(options)
        assert isinstance(result, int)


class TestOptionsGetDefault:
    """Options.get accepts a dict-style default returned only for absent keys (#767)."""

    def test_absent_key_returns_call_site_default(self) -> None:
        """An absent key returns the call-site default."""
        assert Options(group={"a": 1}).get("missing", "d") == "d"

    def test_single_arg_absent_key_still_returns_none(self) -> None:
        """Single-argument get on an absent key still returns None (unchanged)."""
        assert Options(group={"a": 1}).get("missing") is None

    def test_present_group_key_ignores_default(self) -> None:
        """A present group key returns its stored value; the default is ignored."""
        assert Options(group={"a": 1}).get("a", "d") == 1

    def test_present_context_key_ignores_default(self) -> None:
        """A present context key returns its stored value; the default is ignored."""
        assert Options(context={"a": 1}).get("a", "d") == 1

    @pytest.mark.parametrize("falsy", [0, "", False])
    def test_present_falsy_group_value_returned_not_default(self, falsy: Any) -> None:
        """A present falsy group value (0/""/False) is returned as-is, never the default."""
        result = Options(group={"a": falsy}).get("a", "sentinel_default")
        assert result == falsy
        assert type(result) is type(falsy)
        assert result != "sentinel_default"

    @pytest.mark.parametrize("falsy", [0, "", False])
    def test_present_falsy_context_value_returned_not_default(self, falsy: Any) -> None:
        """A present falsy context value (0/""/False) is returned as-is, never the default."""
        result = Options(context={"a": falsy}).get("a", "sentinel_default")
        assert result == falsy
        assert type(result) is type(falsy)
        assert result != "sentinel_default"

    def test_present_none_group_value_returns_none_not_default(self) -> None:
        """A group key stored as None returns None (presence-based), never the default."""
        assert Options(group={"a": None}).get("a", "d") is None

    def test_present_none_context_value_returns_none_not_default(self) -> None:
        """A context key stored as None returns None (presence-based), never the default."""
        assert Options(context={"a": None}).get("a", "d") is None

    def test_default_passed_positionally(self) -> None:
        """The default is accepted as a positional second argument."""
        assert Options().get("m", "d") == "d"

    def test_default_passed_by_keyword(self) -> None:
        """The default is accepted as a keyword argument."""
        assert Options().get("m", default="d") == "d"


class TestOptionsGetInFeaturesFalsyValue:
    """get_in_features distinguishes an absent in_features key from a present falsy one (#942)."""

    def _message(self, options: Options) -> str:
        with pytest.raises(ValueError) as excinfo:
            options.get_in_features()
        return str(excinfo.value)

    def test_absent_key_names_the_key(self) -> None:
        """An absent key still raises ValueError naming the key, not the enum repr."""
        message = self._message(Options())
        assert f"'{IN_FEATURES}'" in message, message
        assert "DefaultOptionKeys" not in message, message

    @pytest.mark.parametrize("falsy", ["", 0, {}], ids=["empty_str", "zero_int", "empty_dict"])
    def test_present_falsy_value_still_raises_value_error(self, falsy: Any) -> None:
        """A present but falsy value keeps raising ValueError, not another error type."""
        with pytest.raises(ValueError):
            Options(context={IN_FEATURES: falsy}).get_in_features()

    @pytest.mark.parametrize("falsy", ["", 0, {}], ids=["empty_str", "zero_int", "empty_dict"])
    def test_present_falsy_value_message_names_the_key(self, falsy: Any) -> None:
        """The falsy-value message names the key readably, so the user knows which option is at fault."""
        message = self._message(Options(context={IN_FEATURES: falsy}))
        assert f"'{IN_FEATURES}'" in message, message
        assert "DefaultOptionKeys" not in message, message

    @pytest.mark.parametrize("falsy", ["", 0, {}], ids=["empty_str", "zero_int", "empty_dict"])
    def test_present_falsy_value_message_names_the_type_not_the_value(self, falsy: Any) -> None:
        """The falsy-value message describes the value by type and never prints its repr."""
        message = self._message(Options(context={IN_FEATURES: falsy}))
        assert type(falsy).__name__ in message, message
        assert f"is set to {falsy!r}" not in message, message

    @pytest.mark.parametrize("falsy", ["", 0, {}], ids=["empty_str", "zero_int", "empty_dict"])
    def test_present_falsy_value_message_differs_from_absent_key_message(self, falsy: Any) -> None:
        """A set key must not be reported with the same wording as a key that is not set at all."""
        absent = self._message(Options())
        present = self._message(Options(context={IN_FEATURES: falsy}))
        assert present != absent, present

    def test_falsy_value_in_group_is_described_by_type_not_value(self) -> None:
        """The distinction holds for a group key too, not only a context key."""
        message = self._message(Options(group={IN_FEATURES: ""}))
        assert f"'{IN_FEATURES}'" in message, message
        assert "DefaultOptionKeys" not in message, message
        assert "is set to a falsy str" in message, message
        assert f"is set to {''!r}" not in message, message


class TestOptionsGetInFeaturesUnresolvableTruthyValue:
    """get_in_features names the offending value for a present, truthy but unresolvable in_features (#942)."""

    def _message(self, value: Any) -> str:
        with pytest.raises(TypeError) as excinfo:
            Options(context={IN_FEATURES: value}).get_in_features()
        return str(excinfo.value)

    @pytest.mark.parametrize("value", [424242, {"token": "sk-secret-9"}], ids=["int", "dict"])  # nosec B105
    def test_unsupported_type_message_names_the_type_not_the_value(self, value: Any) -> None:
        """The unsupported-type message describes the value by type and never prints it."""
        message = self._message(value)
        assert type(value).__name__ in message, message
        assert repr(value) not in message, message
        assert "424242" not in message, message
        assert "sk-secret-9" not in message, message

    @pytest.mark.parametrize("value", [5, {"a": 1}], ids=["int", "dict"])
    def test_unsupported_type_message_still_shows_the_type(self, value: Any) -> None:
        """The unsupported-type message keeps naming the type alongside the value."""
        message = self._message(value)
        assert type(value).__name__ in message, message


class TestOptionsGetInFeaturesOrdered:
    """get_in_features returns an ordered tuple: declared order for sequences, name order for sets."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (["a", "b", "c"], ("a", "b", "c")),
            (["c", "b", "a"], ("c", "b", "a")),
            (("b", "a"), ("b", "a")),
            (["a", "a"], ("a", "a")),
            ("a", ("a",)),
            ("b, a", ("b", "a")),
            (Feature("a"), ("a",)),
            ([Feature("b"), Feature("a")], ("b", "a")),
            ({"b", "c", "a"}, ("a", "b", "c")),
            (frozenset({"b", "c", "a"}), ("a", "b", "c")),
        ],
        ids=[
            "list",
            "reversed_list",
            "tuple",
            "duplicates",
            "single_str",
            "comma_str",
            "feature",
            "feature_list",
            "set",
            "frozenset",
        ],
    )
    def test_returns_ordered_tuple_of_features(self, value: Any, expected: tuple[str, ...]) -> None:
        result = Options(context={IN_FEATURES: value}).get_in_features()
        assert type(result) is tuple
        assert tuple(f.name for f in result) == expected


class TestOptionsWithCredentialValues:
    """Options group holding Credential values compares/hashes by value and never leaks it via str()."""

    def test_options_with_equal_credentials_are_equal_and_hash_alike(self) -> None:
        options1 = Options(group={"credential_option_key": Credential(sqlite="/x.db")})
        options2 = Options(group={"credential_option_key": Credential(sqlite="/x.db")})
        assert options1 == options2
        assert hash(options1) == hash(options2)

    def test_options_with_different_credentials_are_not_equal(self) -> None:
        options1 = Options(group={"credential_option_key": Credential(sqlite="/x.db")})
        options2 = Options(group={"credential_option_key": Credential(sqlite="/y.db")})
        assert options1 != options2

    def test_str_never_contains_the_secret_value(self) -> None:
        options = Options(group={"credential_option_key": Credential(sqlite="/secret/path/analytics.db")})
        assert "/secret/path/analytics.db" not in str(options)

    def test_str_redacts_a_reader_tuple_in_reserved_key_context(self) -> None:
        """Only the reserved reader tuple is masked; a plain non-reader dict stays readable."""
        options = Options(
            context={RESERVED_READER_OPTION_KEY: (SqliteFG, {"sqlite": "/raw/reader_tuple_marker.db"})},
        )
        text = str(options)
        assert "/raw/reader_tuple_marker.db" not in text
        assert RESERVED_READER_OPTION_KEY in text
        assert "sqlite" in text

        plain = Options(group={"cfg": {"a": "b"}})
        assert "'a': 'b'" in str(plain)

        toy = Options(group={ToySourceFG.get_class_name(): {"toy_col": "/raw/toy_pointer_marker.db"}})
        assert "/raw/toy_pointer_marker.db" not in str(toy)
        assert "toy_col" in str(toy)
        toy_context = Options(context={ToySourceFG.get_class_name(): {"toy_col": "/raw/toy_pointer_marker.db"}})
        assert "/raw/toy_pointer_marker.db" not in str(toy_context)
        url = Options(group={ToySourceFG.get_class_name(): "https://user:toy-url-secret@api.example/v1"})
        assert "toy-url-secret" not in str(url)
        assert "api.example" in str(url)

    @pytest.mark.parametrize("where", ["group", "context"])
    def test_str_masks_a_credential_key_in_a_nested_mapping(self, where: str) -> None:
        nested = {"cfg": {"password": "hunter2", "host": "h"}}  # nosec B105
        options = Options(group=nested) if where == "group" else Options(context=nested)
        text = str(options)
        assert "hunter2" not in text
        assert "'password': '***'" in text
        assert "'host': 'h'" in text

    @pytest.mark.parametrize("where", ["group", "context"])
    def test_str_masks_a_top_level_credential_key(self, where: str) -> None:
        options = Options(group={"api_key": "hunter2"}) if where == "group" else Options(context={"api_key": "hunter2"})
        text = str(options)
        assert "hunter2" not in text
        assert "'api_key': '***'" in text

    @pytest.mark.parametrize("where", ["group", "context"])
    @pytest.mark.parametrize("container", [list, tuple])
    def test_str_masks_a_credential_key_in_a_mapping_inside_a_sequence(
        self, where: str, container: type[list[Any]] | type[tuple[Any, ...]]
    ) -> None:
        nested = {"sources": container([{"password": "hunter2", "host": "h"}])}  # nosec B105
        options = Options(group=nested) if where == "group" else Options(context=nested)
        text = str(options)
        assert "hunter2" not in text
        assert "'password': '***'" in text
        assert "'host': 'h'" in text

    def test_str_keeps_a_non_secret_top_level_key_readable(self) -> None:
        assert "'region': 'eu'" in str(Options(group={"region": "eu"}))

    def test_masking_is_display_only_the_raw_value_stays_in_stable_text(self) -> None:
        from mloda.core.prepare.stable_text import stable_text

        options = Options(group={"api_key": "hunter2", "cfg": {"password": "hunter2"}})  # nosec B105
        assert "hunter2" in stable_text(options.group)
        assert options.group["cfg"]["password"] == "hunter2"

    def test_str_of_a_cyclic_nested_mapping_does_not_raise(self) -> None:
        inner: dict[str, Any] = {"password": "hunter2"}  # nosec B105
        inner["self"] = inner
        text = str(Options(context={"cfg": inner}))
        assert "hunter2" not in text

    def test_str_redacts_a_reader_tuple_dsn_string_in_reserved_key_context(self) -> None:
        """The reserved reader tuple also masks a DSN string second element."""
        options = Options(
            context={RESERVED_READER_OPTION_KEY: (SqliteFG, "postgresql://dbuser:cred_marker_q7@dbhost/db")},
        )
        text = str(options)
        assert "cred_marker_q7" not in text
        assert RESERVED_READER_OPTION_KEY in text


class TestOptionsClassKeyNormalization:
    def test_source_group_class_key_normalizes_to_its_class_name_not_its_alias(self) -> None:
        aliased: Any = type(
            "_ToyfmtAliasKeyFG", (ToySourceBase,), {"data_access_name": classmethod(lambda cls: "toyfmt_k")}
        )

        options = Options({aliased: {"x": 1}})

        assert "_ToyfmtAliasKeyFG" in options.group
        assert "toyfmt_k" not in options.group
        del aliased

    def test_the_abstract_source_group_base_class_is_a_valid_key(self) -> None:
        key: Any = SourceFeatureGroup
        options = Options({key: {"x": 1}})

        assert "SourceFeatureGroup" in options.group
