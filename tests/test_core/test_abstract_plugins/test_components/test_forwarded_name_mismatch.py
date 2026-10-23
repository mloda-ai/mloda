"""
Failing tests pinning the forwarded-name-mismatch check contract.

FeatureChainParserMixin.match_feature_group_criteria, when the feature matches
this group via string parsing (a name-parsed operation value exists and maps to
a PROPERTY_MAPPING key K), must additionally check: if the feature's options
carry K, AND K is in options.inherited_group_keys (it arrived via consumer
forwarding, not set by the author), AND str(option value) differs from the
name-parsed value, then raise ValueError. The message contains the feature
name, the key K, both values, and the remedy string "forward_group_exclude",
plus a mention of the env var MLODA_ALLOW_FORWARDED_NAME_MISMATCH. If that env
var is set to "1" or "true", the check logs ONE logging WARNING (same content)
instead of raising and matching proceeds normally.

No behavior change when: values are equal, the feature is config-based (no
string parse), or K is absent from the options. An author-declared K that
contradicts the name aborts too (TestDeclaredNameMismatch).

Also covers the context path: inherited_context_keys must be checked like
inherited_group_keys, with remedy text naming
inherit_context_keys/propagate_context_keys instead of forward_group_exclude.

All fixture names carry a "namemis579" (group path) or "ctxmis579" (context
path) marker so they cannot collide with other tests in the global plugin
registry.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import PropertySpec
from mloda.user import Options


OPERATION_KEY = "operation_namemis579"
STRING_FEATURE_NAME = "sales__sum_namemis579"


class _NameMismatchChainedGroup(FeatureChainParserMixin):
    """Minimal chainer subclass following the existing mixin test fixture style.

    The operation key is group-categorized on purpose (context=False): only group
    options flow through consumer forwarding.
    """

    PREFIX_PATTERN = r".*__(sum|max)_namemis579$"
    MIN_IN_FEATURES = 0
    PROPERTY_MAPPING = {
        OPERATION_KEY: PropertySpec(
            "Operation of the namemis579 fixture",
            allowed_values={
                "sum": "Sum of the in feature (namemis579 fixture)",
                "max": "Maximum of the in feature (namemis579 fixture)",
            },
            context=False,
            strict_validation=True,
        )
    }


class _OpenValueGroup(FeatureChainParserMixin):
    """Named capture with no allowed_values, so a long declared value can reach the contradiction message."""

    PREFIX_PATTERN = r".*__(?P<open_key_lv>\w+)_openlv$"
    MIN_IN_FEATURES = 0
    PROPERTY_MAPPING = {"open_key_lv": PropertySpec("Open-valued key", context=False)}


def _inherited_child_options(consumer_group: dict[str, Any]) -> Options:
    """Build child options exactly like the engine does: inherit_from the consumer."""
    child_options = Options()
    child_options.inherit_from(Options(group=consumer_group))
    return child_options


CONTEXT_KEY = "operation_ctxmis579"
STRING_FEATURE_NAME_CTX = "sales__sum_ctxmis579"


class _ContextMismatchChainedGroup(FeatureChainParserMixin):
    """Context-path counterpart of ``_NameMismatchChainedGroup``; context=True, reachable only via context forward."""

    PREFIX_PATTERN = r".*__(sum|max)_ctxmis579$"
    PROPERTY_MAPPING = {
        CONTEXT_KEY: PropertySpec(
            "Operation of the ctxmis579 fixture",
            allowed_values={
                "sum": "Sum of the in feature (ctxmis579 fixture)",
                "max": "Maximum of the in feature (ctxmis579 fixture)",
            },
            context=True,
            strict_validation=True,
        )
    }


def _context_pull_child_options(consumer_context: dict[str, Any]) -> Options:
    """Child-side pull: child.inherit_from(consumer, inherit_context_keys=...)."""
    child_options = Options()
    child_options.inherit_from(
        Options(context=consumer_context), inherit_context_keys=frozenset(consumer_context.keys())
    )
    return child_options


def _context_propagate_child_options(consumer_context: dict[str, Any]) -> Options:
    """Consumer-side push: consumer.propagate_context_keys flows into the child on inherit_from."""
    child_options = Options()
    consumer = Options(context=consumer_context, propagate_context_keys=frozenset(consumer_context.keys()))
    child_options.inherit_from(consumer)
    return child_options


class TestFixtureSanity:
    def test_string_feature_matches_with_empty_options(self) -> None:
        """Precondition: the fixture group claims the chained name via string parsing."""
        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, Options()) is True


class TestForwardedNameMismatch:
    def test_inherited_differing_value_raises(self) -> None:
        """An inherited option value differing from the name-parsed value must raise."""
        child_options = _inherited_child_options({OPERATION_KEY: "max"})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        message = str(exc_info.value)
        assert STRING_FEATURE_NAME in message
        assert OPERATION_KEY in message
        assert "sum" in message
        assert "max" in message
        assert "forward_group_exclude" in message
        assert "MLODA_ALLOW_FORWARDED_NAME_MISMATCH" in message

    def test_inherited_equal_value_matches(self) -> None:
        """An inherited option value equal to the name-parsed value matches silently."""
        child_options = _inherited_child_options({OPERATION_KEY: "sum"})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_author_set_differing_value_raises(self) -> None:
        """An author-set option value (not inherited) contradicting the name aborts the match."""
        child_options = Options(group={OPERATION_KEY: "max"})
        assert child_options.inherited_group_keys == frozenset()  # precondition: nothing inherited

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        message = str(exc_info.value)
        assert STRING_FEATURE_NAME in message
        assert OPERATION_KEY in message
        assert "sum" in message
        assert "max" in message
        assert "remove" in message

    def test_message_does_not_carry_a_long_raw_option_value(self) -> None:
        long_value = "m" * 200

        with pytest.raises(ValueError) as exc_info:
            _OpenValueGroup.match_feature_group_criteria(
                "sales__short_openlv", Options(group={"open_key_lv": long_value})
            )

        assert long_value not in str(exc_info.value)

    def test_env_var_downgrades_to_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 the check warns once and matching proceeds."""
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        with caplog.at_level(logging.WARNING):
            result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True
        records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and OPERATION_KEY in record.getMessage()
        ]
        assert len(records) == 1, (
            f"Expected exactly one WARNING mentioning '{OPERATION_KEY}', got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "sum" in message
        assert "max" in message

    def test_config_based_feature_unaffected(self) -> None:
        """A config-based feature (no string parse) keeps today's matching, no raise."""
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        result = _NameMismatchChainedGroup.match_feature_group_criteria("namemis579_config_feature", child_options)

        assert result is True

    def test_absent_key_unaffected(self) -> None:
        """A string feature whose options lack the key (but carry another inherited key) matches."""
        child_options = _inherited_child_options({"other_key_namemis579": "x"})
        assert child_options.inherited_group_keys == frozenset({"other_key_namemis579"})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True


class TestForwardedSingletonUnpack:
    """Issue #764: a forwarded SINGLETON collection whose sole element equals the name-parsed
    value must match, because _unpack_property_value treats ['sum'] and 'sum' as equivalent
    everywhere else. A genuine mismatch, and a non-singleton collection, must still raise.
    """

    def test_forwarded_singleton_list_equal_value_matches(self) -> None:
        """A forwarded one-element list equal to the name value unpacks to it and matches."""
        child_options = _inherited_child_options({OPERATION_KEY: ["sum"]})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_forwarded_singleton_tuple_equal_value_matches(self) -> None:
        """A forwarded one-element tuple equal to the name value unpacks to it and matches."""
        child_options = _inherited_child_options({OPERATION_KEY: ("sum",)})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_forwarded_singleton_set_equal_value_matches(self) -> None:
        """A forwarded one-element set equal to the name value unpacks to it and matches."""
        child_options = _inherited_child_options({OPERATION_KEY: {"sum"}})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert result is True

    def test_forwarded_singleton_list_differing_value_still_raises(self) -> None:
        """A forwarded one-element list contradicting the name value stays a genuine mismatch."""
        child_options = _inherited_child_options({OPERATION_KEY: ["max"]})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        message = str(exc_info.value)
        assert "max" in message
        assert "forward_group_exclude" in message

    def test_forwarded_multi_element_list_still_raises(self) -> None:
        """A multi-element forwarded list is not a singleton and contradicts the scalar name value."""
        child_options = _inherited_child_options({OPERATION_KEY: ["sum", "max"]})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert "forward_group_exclude" in str(exc_info.value)

    def test_forwarded_empty_collection_still_raises(self) -> None:
        """An empty forwarded collection unpacks to length 0, not a singleton, so it still raises."""
        child_options = _inherited_child_options({OPERATION_KEY: []})
        assert child_options.inherited_group_keys == frozenset({OPERATION_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, child_options)

        assert "forward_group_exclude" in str(exc_info.value)


class TestForwardedContextNameMismatch:
    """The check must also catch a value forwarded via the context path (inherited_context_keys), not just group."""

    def test_context_pull_differing_value_raises(self) -> None:
        """A context-pulled value contradicting the name-parsed value must raise."""
        child_options = _context_pull_child_options({CONTEXT_KEY: "max"})
        assert child_options.inherited_context_keys == frozenset({CONTEXT_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        message = str(exc_info.value)
        assert STRING_FEATURE_NAME_CTX in message
        assert CONTEXT_KEY in message
        assert "sum" in message
        assert "max" in message
        assert "MLODA_ALLOW_FORWARDED_NAME_MISMATCH" in message
        assert "forward_group_exclude" not in message
        assert "inherit_context_keys" in message
        assert "propagate_context_keys" in message

    def test_context_propagate_differing_value_raises(self) -> None:
        """A context-pushed value contradicting the name-parsed value must also raise."""
        child_options = _context_propagate_child_options({CONTEXT_KEY: "max"})
        assert child_options.inherited_context_keys == frozenset({CONTEXT_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        message = str(exc_info.value)
        assert STRING_FEATURE_NAME_CTX in message
        assert CONTEXT_KEY in message
        assert "sum" in message
        assert "max" in message
        assert "MLODA_ALLOW_FORWARDED_NAME_MISMATCH" in message
        assert "forward_group_exclude" not in message
        assert "inherit_context_keys" in message
        assert "propagate_context_keys" in message

    def test_context_cascade_two_hops_differing_value_names_cascade(self) -> None:
        """A value delivered through a two-hop cascade that contradicts the name names cascade_context_keys."""
        consumer = Options(context={CONTEXT_KEY: "max"}, cascade_context_keys=frozenset({CONTEXT_KEY}))
        middle = Options()
        middle.inherit_from(consumer)
        child_options = Options()
        child_options.inherit_from(middle)
        assert child_options.inherited_context_keys == frozenset({CONTEXT_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        message = str(exc_info.value)
        assert CONTEXT_KEY in message
        assert "cascade_context_keys" in message

    def test_pull_and_one_hop_push_messages_do_not_name_cascade(self) -> None:
        for child_options in (
            _context_pull_child_options({CONTEXT_KEY: "max"}),
            _context_propagate_child_options({CONTEXT_KEY: "max"}),
        ):
            with pytest.raises(ValueError) as exc_info:
                _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

            assert "cascade_context_keys" not in str(exc_info.value)

    def test_context_pull_equal_value_matches(self) -> None:
        """A pulled context value equal to the name-parsed value matches silently."""
        child_options = _context_pull_child_options({CONTEXT_KEY: "sum"})
        assert child_options.inherited_context_keys == frozenset({CONTEXT_KEY})  # precondition

        result = _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        assert result is True

    def test_env_var_downgrades_context_mismatch_to_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 a context-path mismatch warns once, no raise."""
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        child_options = _context_pull_child_options({CONTEXT_KEY: "max"})

        with caplog.at_level(logging.WARNING):
            result = _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        assert result is True
        records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and CONTEXT_KEY in record.getMessage()
        ]
        assert len(records) == 1, (
            f"Expected exactly one WARNING mentioning '{CONTEXT_KEY}', got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert "sum" in message
        assert "max" in message

    def test_author_set_context_value_raises(self) -> None:
        """An author-set context value (not inherited) contradicting the name aborts the match."""
        child_options = Options(context={CONTEXT_KEY: "max"})
        assert child_options.inherited_context_keys == frozenset()  # precondition: nothing inherited

        with pytest.raises(ValueError) as exc_info:
            _ContextMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME_CTX, child_options)

        message = str(exc_info.value)
        assert CONTEXT_KEY in message
        assert "sum" in message
        assert "max" in message


class TestDeclaredNameMismatch:
    """A declared (own, non-inherited) option must agree with the value the name binds."""

    def test_equal_value_matches(self) -> None:
        options = Options(group={OPERATION_KEY: "sum"})

        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options) is True

    def test_singleton_list_equal_value_matches(self) -> None:
        options = Options(group={OPERATION_KEY: ["sum"]})

        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options) is True

    def test_singleton_list_differing_value_raises(self) -> None:
        options = Options(group={OPERATION_KEY: ["max"]})

        with pytest.raises(ValueError):
            _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options)

    def test_rebuild_added_value_is_not_declared(self) -> None:
        """A value added by Options.rebuild (a materialized default) is not own, so it cannot contradict."""
        options = Options().rebuild(group={OPERATION_KEY: "max"}, context={})
        assert not options.is_own(OPERATION_KEY)  # precondition

        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options) is True

    def test_key_added_after_lock_own_keys_is_not_declared(self) -> None:
        options = Options()
        options.lock_own_keys()
        options.add_to_group(OPERATION_KEY, "max")
        assert not options.is_own(OPERATION_KEY)  # precondition

        assert _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options) is True

    def test_env_var_downgrades_declared_mismatch_to_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        options = Options(group={OPERATION_KEY: "max"})

        with caplog.at_level(logging.WARNING):
            result = _NameMismatchChainedGroup.match_feature_group_criteria(STRING_FEATURE_NAME, options)

        assert result is True
        records = [r for r in caplog.records if r.levelno == logging.WARNING and OPERATION_KEY in r.getMessage()]
        assert len(records) == 1
        assert "sum" in records[0].getMessage()
        assert "max" in records[0].getMessage()


FLOAT_KEY = "operation_cvmis"
FLOAT_FEATURE_NAME = "sales__fl_1.0_cvmis"


def _float_converter_group() -> type[FeatureChainParserMixin]:
    """Group key bound from the name through a float name_converter (captured `1.0` binds `1.0`)."""
    namespace = {
        "__module__": __name__,
        "PREFIX_PATTERN": rf".*__fl_(?P<{FLOAT_KEY}>[0-9.]+)_cvmis$",
        "MIN_IN_FEATURES": 0,
        "PROPERTY_MAPPING": {FLOAT_KEY: PropertySpec("Float key", context=False, name_converter=float)},
    }
    return type("FloatConverterCvmisGroup", (FeatureChainParserMixin,), namespace)


RAW_KEY = "operation_rawcv"


def _single_item_list(text: str) -> list[int]:
    return [int(text)]


def _split_digits(text: str) -> list[int]:
    return [int(char) for char in text]


def _raw_group(converter: Any) -> type[FeatureChainParserMixin]:
    """Group key bound from `<src>__raw_<digits>_rawcv` through ``converter``."""
    namespace = {
        "__module__": __name__,
        "PREFIX_PATTERN": rf".*__raw_(?P<{RAW_KEY}>[0-9]+)_rawcv$",
        "MIN_IN_FEATURES": 0,
        "PROPERTY_MAPPING": {RAW_KEY: PropertySpec("Raw key", context=False, name_converter=converter)},
    }
    return type("RawConverterRawcvGroup", (FeatureChainParserMixin,), namespace)


class TestNameConverterAgreement:
    """A declared or forwarded typed value agrees with a converted bound value; a contradiction still aborts."""

    def test_declared_typed_value_equal_to_the_bound_value_matches(self) -> None:
        options = Options(group={FLOAT_KEY: 1})

        assert _float_converter_group().match_feature_group_criteria(FLOAT_FEATURE_NAME, options) is True

    def test_declared_text_equal_to_the_bound_value_matches(self) -> None:
        options = Options(group={FLOAT_KEY: "1.0"})

        assert _float_converter_group().match_feature_group_criteria(FLOAT_FEATURE_NAME, options) is True

    def test_declared_contradicting_value_raises(self) -> None:
        options = Options(group={FLOAT_KEY: 2})

        with pytest.raises(ValueError):
            _float_converter_group().match_feature_group_criteria(FLOAT_FEATURE_NAME, options)

    def test_forwarded_typed_value_equal_to_the_bound_value_matches(self) -> None:
        child_options = _inherited_child_options({FLOAT_KEY: 1})

        assert _float_converter_group().match_feature_group_criteria(FLOAT_FEATURE_NAME, child_options) is True

    def test_forwarded_contradicting_value_raises(self) -> None:
        child_options = _inherited_child_options({FLOAT_KEY: 2})

        with pytest.raises(ValueError, match="forward_group_exclude"):
            _float_converter_group().match_feature_group_criteria(FLOAT_FEATURE_NAME, child_options)

    def test_declared_text_equal_to_the_raw_capture_matches(self) -> None:
        options = Options(group={FLOAT_KEY: "1"})

        assert _float_converter_group().match_feature_group_criteria("sales__fl_1_cvmis", options) is True

    def test_forwarded_text_equal_to_the_raw_capture_matches(self) -> None:
        child_options = _inherited_child_options({FLOAT_KEY: "1"})

        assert _float_converter_group().match_feature_group_criteria("sales__fl_1_cvmis", child_options) is True

    def test_declared_zero_padded_text_equal_to_the_raw_capture_matches(self) -> None:
        options = Options(group={RAW_KEY: "05"})

        assert _raw_group(int).match_feature_group_criteria("sales__raw_05_rawcv", options) is True

    def test_contradiction_message_quotes_the_raw_capture(self) -> None:
        options = Options(group={RAW_KEY: "06"})

        with pytest.raises(ValueError, match="'05'"):
            _raw_group(int).match_feature_group_criteria("sales__raw_05_rawcv", options)

    @pytest.mark.parametrize(
        ("converter", "name", "declared"),
        [
            (_single_item_list, "sales__raw_1_rawcv", [1]),
            (_split_digits, "sales__raw_12_rawcv", [1, 2]),
        ],
        ids=["single_element", "multi_element"],
    )
    def test_container_converter_with_an_agreeing_declared_list_matches(
        self, converter: Any, name: str, declared: list[int]
    ) -> None:
        options = Options(group={RAW_KEY: declared})

        assert _raw_group(converter).match_feature_group_criteria(name, options) is True

    @pytest.mark.parametrize(
        ("converter", "name", "declared"),
        [
            (_single_item_list, "sales__raw_1_rawcv", [2]),
            (_split_digits, "sales__raw_12_rawcv", [1, 3]),
            (_split_digits, "sales__raw_12_rawcv", [1]),
        ],
        ids=["single_element", "multi_element", "length_differs"],
    )
    def test_container_converter_with_a_contradicting_declared_list_raises(
        self, converter: Any, name: str, declared: list[int]
    ) -> None:
        options = Options(group={RAW_KEY: declared})

        with pytest.raises(ValueError):
            _raw_group(converter).match_feature_group_criteria(name, options)


class TestHandParsedBindings:
    def test_forwarded_contradicting_binding_raises(self) -> None:
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        with pytest.raises(ValueError, match="forward_group_exclude"):
            _NameMismatchChainedGroup.validate_hand_parsed_bindings(
                STRING_FEATURE_NAME, {OPERATION_KEY: "sum"}, child_options
            )

    def test_declared_contradicting_binding_raises(self) -> None:
        options = Options(group={OPERATION_KEY: "max"})

        with pytest.raises(ValueError, match=OPERATION_KEY):
            _NameMismatchChainedGroup.validate_hand_parsed_bindings(
                STRING_FEATURE_NAME, {OPERATION_KEY: "sum"}, options
            )

    def test_agreeing_values_pass(self) -> None:
        for options in (Options(group={OPERATION_KEY: "sum"}), _inherited_child_options({OPERATION_KEY: "sum"})):
            _NameMismatchChainedGroup.validate_hand_parsed_bindings(
                STRING_FEATURE_NAME, {OPERATION_KEY: "sum"}, options
            )

    def test_typed_declared_value_agrees_with_string_binding(self) -> None:
        options = Options(group={RAW_KEY: 5})

        _raw_group(int).validate_hand_parsed_bindings("sales__raw_5_rawcv", {RAW_KEY: "5"}, options)

    def test_env_var_downgrades_to_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        child_options = _inherited_child_options({OPERATION_KEY: "max"})

        with caplog.at_level(logging.WARNING):
            _NameMismatchChainedGroup.validate_hand_parsed_bindings(
                STRING_FEATURE_NAME, {OPERATION_KEY: "sum"}, child_options
            )

        records = [r for r in caplog.records if r.levelno == logging.WARNING and OPERATION_KEY in r.getMessage()]
        assert len(records) == 1
