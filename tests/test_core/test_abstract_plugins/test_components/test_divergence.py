"""Divergence record: construction rules, definition-time checks on groups and Implementations, stamping."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mloda.core.abstract_plugins.components import divergence as divergence_module
from mloda.core.abstract_plugins.components.divergence import (
    MAX_DOCS_LENGTH,
    MAX_REASON_LENGTH,
    Divergence,
    DivergenceKind,
    DivergenceSource,
    divergence_problems,
    stamped,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import ANY_OUTPUT_PART
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import Cause, OperationFeatureGroup, SubtypeDeclaration, property_spec
from mloda.user import Options
from mloda.user.pyarrow import PyArrowTable
from mloda.user.python_dict import PythonDictFramework

KEY = "divtest_kind"
V = DivergenceKind.VALUE
B = Cause.BACKEND


def _record(**kwargs: Any) -> Divergence:
    base: dict[str, Any] = {"kind": V, "cause": B, "reason": "skips NaN"}
    return Divergence(**{**base, **kwargs})


def _make_group(**attrs: Any) -> type[OperationFeatureGroup]:
    body: dict[str, Any] = {
        "SUBTYPES": SubtypeDeclaration(key=KEY),
        "PROPERTY_MAPPING": {
            KEY: property_spec("Kind.", strict=True, allowed_values={"sum": "Sum", "mean": "Mean"}),
        },
        **attrs,
    }
    return type("DivTestGroup", (OperationFeatureGroup,), body)


def _make_impl(group: type[Any], framework: type[Any] = PythonDictFramework, **attrs: Any) -> type[Implementation[Any]]:
    body: dict[str, Any] = {
        "feature_group": group,
        "compute_framework": framework,
        "calculate": classmethod(lambda cls, group, data, features, framework: data),
        **attrs,
    }
    return type("DivTestImpl", (Implementation,), body)


def _feature_set(*features: Feature) -> FeatureSet:
    return FeatureSet(features)


def _feature(name: str = "divtest_x", kind: str | None = "sum", part: str | None = None) -> Feature:
    feature = Feature(name, Options({KEY: kind} if kind else {}))
    feature.record_part(part)
    return feature


class TestConstruction:
    def test_minimal_record_has_defaults_and_no_stamp(self) -> None:
        record = _record()
        assert record.subtypes == frozenset()
        assert record.outputs == frozenset()
        assert record.variant is None
        assert record.docs is None
        assert record.source is None
        assert record.framework is None

    def test_record_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _record().reason = "other"  # type: ignore[misc]

    def test_iterables_are_stored_as_frozensets(self) -> None:
        record = _record(subtypes=["sum", "mean"])
        assert record.subtypes == frozenset({"sum", "mean"})
        assert isinstance(record.subtypes, frozenset)
        assert _record(outputs=("a",)).outputs == frozenset({"a"})

    def test_arguments_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            Divergence(V, B)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "value"},
            {"cause": "backend"},
            {"subtypes": "sum"},
            {"outputs": "a"},
            {"subtypes": [1]},
            {"outputs": [None]},
            {"reason": 3},
            {"docs": 3, "cause": Cause.OTHER},
            {"variant": 3, "kind": DivergenceKind.VARIANT},
            {"subtypes": {"sum": "skips NaN"}},
            {"outputs": {"a": "skips NaN"}},
        ],
        ids=[
            "kind",
            "cause",
            "bare_subtypes",
            "bare_outputs",
            "int_subtype",
            "none_output",
            "int_reason",
            "docs",
            "variant",
            "mapping_subtypes",
            "mapping_outputs",
        ],
    )
    def test_wrong_types_raise_type_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(TypeError):
            _record(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"subtypes": [""]},
            {"outputs": [""]},
            {"subtypes": ["sum"], "outputs": ["a"]},
            {"reason": ""},
            {"reason": "   "},
            {"reason": "x" * (MAX_REASON_LENGTH + 1)},
            {"cause": Cause.OTHER},
            {"cause": Cause.OTHER, "docs": ""},
            {"kind": DivergenceKind.VARIANT},
            {"kind": DivergenceKind.VARIANT, "variant": ""},
            {"variant": "fast"},
            {"docs": "x" * (MAX_DOCS_LENGTH + 1), "cause": Cause.OTHER},
            {"reason": "   ", "cause": Cause.NOT_IMPLEMENTED},
            {"reason": "a\tb"},
            {"reason": "a\x00b"},
            {"reason": "a\x1bb"},
            {"kind": DivergenceKind.VARIANT, "variant": "a\tb"},
            {"kind": DivergenceKind.VARIANT, "variant": "a\x00b"},
            {"kind": DivergenceKind.VARIANT, "variant": "a\x1bb"},
            {"cause": Cause.OTHER, "docs": "a\tb"},
            {"cause": Cause.OTHER, "docs": "a\x00b"},
            {"cause": Cause.OTHER, "docs": "a\x1bb"},
            {"kind": DivergenceKind.VARIANT, "variant": "x" * (MAX_REASON_LENGTH + 1)},
        ],
        ids=[
            "empty_subtype",
            "empty_output",
            "subtypes_and_outputs",
            "empty_reason",
            "whitespace_reason",
            "reason_too_long",
            "other_without_docs",
            "other_with_empty_docs",
            "variant_kind_without_variant",
            "variant_kind_empty_variant",
            "variant_on_non_variant_kind",
            "docs_too_long",
            "whitespace_reason_not_implemented",
            "reason_tab",
            "reason_nul",
            "reason_esc",
            "variant_tab",
            "variant_nul",
            "variant_esc",
            "docs_tab",
            "docs_nul",
            "docs_esc",
            "variant_too_long",
        ],
    )
    def test_wrong_values_raise_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _record(**kwargs)

    @pytest.mark.parametrize("char", ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "])
    @pytest.mark.parametrize("field", ["reason", "variant", "docs"])
    def test_every_line_break_is_rejected(self, field: str, char: str) -> None:
        kwargs: dict[str, Any] = {"kind": DivergenceKind.VARIANT, "variant": "fast", "cause": Cause.OTHER, "docs": "d"}
        kwargs[field] = f"one{char}two"
        with pytest.raises(ValueError):
            _record(**kwargs)

    @pytest.mark.parametrize("field", ["reason", "variant", "docs"])
    def test_a_trailing_line_break_is_rejected(self, field: str) -> None:
        kwargs: dict[str, Any] = {"kind": DivergenceKind.VARIANT, "variant": "fast", "cause": Cause.OTHER, "docs": "d"}
        kwargs[field] = "one\n"
        with pytest.raises(ValueError):
            _record(**kwargs)

    def test_docs_up_to_its_own_limit_are_accepted(self) -> None:
        assert _record(cause=Cause.OTHER, docs="x" * MAX_DOCS_LENGTH).docs == "x" * MAX_DOCS_LENGTH

    def test_a_reason_just_over_its_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _record(reason="x" * 201)

    def test_a_reason_of_exactly_the_limit_is_accepted(self) -> None:
        assert _record(reason="x" * MAX_REASON_LENGTH).reason == "x" * MAX_REASON_LENGTH

    def test_not_implemented_needs_no_reason(self) -> None:
        assert _record(cause=Cause.NOT_IMPLEMENTED, reason="").reason == ""

    def test_other_with_docs_is_accepted(self) -> None:
        assert _record(cause=Cause.OTHER, docs="see the guide").docs == "see the guide"

    def test_variant_kind_with_variant_is_accepted(self) -> None:
        assert _record(kind=DivergenceKind.VARIANT, variant="fast").variant == "fast"

    def test_unknown_keyword_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            _record(bogus=1)

    @pytest.mark.parametrize("name", ["source", "framework"])
    def test_authors_cannot_pass_the_stamp_fields(self, name: str) -> None:
        with pytest.raises(TypeError):
            _record(**{name: "x"})


class TestDescribe:
    def test_describe_is_the_reason(self) -> None:
        assert _record(reason="skips NaN").describe() == "skips NaN"

    def test_reasonless_not_implemented_describes_itself(self) -> None:
        assert _record(cause=Cause.NOT_IMPLEMENTED, reason="").describe() == "not implemented"

    def test_not_implemented_with_a_reason_keeps_it(self) -> None:
        assert _record(cause=Cause.NOT_IMPLEMENTED, reason="no median").describe() == "no median"


class TestAppliesTo:
    def test_whole_plugin_record_applies_always(self) -> None:
        record = _record()
        assert record.applies_to("sum", None)
        assert record.applies_to(None, None)
        assert record.applies_to("mean", frozenset({"a"}))

    def test_subtypes_record_applies_to_its_subtypes_only(self) -> None:
        record = _record(subtypes={"sum"})
        assert record.applies_to("sum", None)
        assert not record.applies_to("mean", None)
        assert not record.applies_to(None, None)

    def test_outputs_record_applies_to_a_whole_feature_request(self) -> None:
        assert _record(outputs={"a"}).applies_to("sum", None)

    def test_outputs_record_applies_when_a_requested_part_is_covered(self) -> None:
        record = _record(outputs={"a"})
        assert record.applies_to("sum", frozenset({"a"}))
        assert record.applies_to("sum", frozenset({"a", "b"}))
        assert not record.applies_to("sum", frozenset({"b"}))


class TestStamped:
    def test_stamped_copies_with_source_and_framework(self) -> None:
        record = _record()
        copy = stamped(record, DivergenceSource.IMPLEMENTATION, "PythonDictFramework")
        assert copy.source is DivergenceSource.IMPLEMENTATION
        assert copy.framework == "PythonDictFramework"
        assert record.source is None
        assert record.framework is None

    def test_stamped_keeps_the_declared_fields(self) -> None:
        record = _record(subtypes={"sum"}, reason="r")
        copy = stamped(record, DivergenceSource.FEATURE_GROUP, "F")
        assert (copy.kind, copy.cause, copy.subtypes, copy.reason) == (V, B, frozenset({"sum"}), "r")


class TestStampedScrub:
    URL = "postgres://admin:hunter2@db.example.com/prod"

    def _raw(self) -> Divergence:
        return _record(
            kind=DivergenceKind.VARIANT,
            variant=f"via {self.URL}",
            cause=Cause.OTHER,
            docs=f"see {self.URL}",
            reason=f"backend at {self.URL}",
        )

    def test_a_stamped_copy_masks_credentials_in_reason_variant_and_docs(self) -> None:
        copy = stamped(self._raw(), DivergenceSource.IMPLEMENTATION, "F")
        assert copy.variant is not None and copy.docs is not None
        for text in (copy.reason, copy.variant, copy.docs):
            assert "hunter2" not in text
            assert "db.example.com" in text

    def test_the_raw_declaration_stays_as_written(self) -> None:
        raw = self._raw()
        stamped(raw, DivergenceSource.IMPLEMENTATION, "F")
        assert "hunter2" in raw.reason
        assert "hunter2" in (raw.variant or "")
        assert "hunter2" in (raw.docs or "")
        assert "hunter2" in repr(raw)

    def test_absent_variant_and_docs_stay_none(self) -> None:
        copy = stamped(_record(), DivergenceSource.FEATURE_GROUP, None)
        assert copy.variant is None
        assert copy.docs is None

    def test_a_credential_free_record_is_unchanged(self) -> None:
        copy = stamped(_record(reason="skips NaN"), DivergenceSource.FEATURE_GROUP, None)
        assert copy.reason == "skips NaN"

    def test_every_record_a_group_or_implementation_stamps_is_scrubbed(self) -> None:
        raw = _record(reason=f"at {self.URL}")
        group = _make_group(DIVERGENCES=(raw,))
        impl = _make_impl(group, divergences=(raw,))
        assert "hunter2" not in group.divergence_records()[0].reason
        assert "hunter2" not in impl.divergence_records()[0].reason


class TestDivergenceProblems:
    UNIVERSE = frozenset({"sum", "mean"})

    def test_conforming_records_have_no_problems(self) -> None:
        records = (_record(subtypes={"sum"}), _record(outputs={"a"}))
        assert divergence_problems("Owner", records, self.UNIVERSE, ("a", "b")) == []

    def test_empty_tuple_has_no_problems(self) -> None:
        assert divergence_problems("Owner", (), frozenset(), None) == []

    @pytest.mark.parametrize("records", [[], None, "x", 3, {"a": 1}, {_record()}])
    def test_a_non_tuple_container_is_reported_without_raising(self, records: object) -> None:
        problems = divergence_problems("Owner", records, self.UNIVERSE, None)
        assert problems
        assert all("Owner" in p for p in problems)

    @pytest.mark.parametrize("entry", ["sum", None, 3, {"kind": V}])
    def test_a_non_record_entry_is_reported_without_raising(self, entry: object) -> None:
        problems = divergence_problems("Owner", (entry,), self.UNIVERSE, None)
        assert problems
        assert all("Owner" in p for p in problems)

    def test_a_subtype_outside_the_universe_is_reported(self) -> None:
        problems = divergence_problems("Owner", (_record(subtypes={"bogus"}),), self.UNIVERSE, None)
        assert any("outside the subtype universe" in p and "bogus" in p for p in problems)

    def test_any_subtypes_record_fails_on_an_empty_universe(self) -> None:
        problems = divergence_problems("Owner", (_record(subtypes={"sum"}),), frozenset(), None)
        assert any("outside the subtype universe" in p for p in problems)

    def test_an_output_outside_the_declared_parts_is_reported(self) -> None:
        problems = divergence_problems("Owner", (_record(outputs={"c"}),), self.UNIVERSE, ("a", "b"))
        assert any("c" in p and "OUTPUT_PARTS" in p for p in problems)

    def test_outputs_are_not_checked_when_the_group_declares_no_parts(self) -> None:
        assert divergence_problems("Owner", (_record(outputs={"c"}),), self.UNIVERSE, None) == []

    def test_a_smuggled_empty_reason_is_still_reported(self) -> None:
        record = _record(subtypes={"sum"})
        object.__setattr__(record, "reason", "")
        assert any("no reason" in p for p in divergence_problems("Owner", (record,), self.UNIVERSE, None))

    def test_a_smuggled_multiline_reason_is_still_reported(self) -> None:
        record = _record()
        object.__setattr__(record, "reason", "a\nb")
        assert divergence_problems("Owner", (record,), self.UNIVERSE, None)

    def test_a_smuggled_other_without_docs_is_still_reported(self) -> None:
        record = _record()
        object.__setattr__(record, "cause", Cause.OTHER)
        assert divergence_problems("Owner", (record,), self.UNIVERSE, None)


class TestFeatureGroupDeclaration:
    def test_default_is_an_empty_tuple(self) -> None:
        assert FeatureGroup.DIVERGENCES == ()

    def test_a_valid_declaration_is_kept(self) -> None:
        record = _record(subtypes={"sum"})
        assert _make_group(DIVERGENCES=(record,)).DIVERGENCES == (record,)

    def test_a_subtype_outside_the_universe_fails_at_definition(self) -> None:
        with pytest.raises(ValueError, match="outside the subtype universe"):
            _make_group(DIVERGENCES=(_record(subtypes={"bogus"}),))

    def test_subtypes_on_a_group_without_subtypes_fail_at_definition(self) -> None:
        with pytest.raises(ValueError):
            type("DivTestPlain", (FeatureGroup,), {"DIVERGENCES": (_record(subtypes={"sum"}),)})

    def test_an_output_outside_output_parts_fails_at_definition(self) -> None:
        with pytest.raises(ValueError):
            _make_group(OUTPUT_PARTS=("a",), DIVERGENCES=(_record(outputs={"b"}),))

    def test_whole_plugin_and_matching_outputs_are_accepted(self) -> None:
        group = _make_group(OUTPUT_PARTS=("a",), DIVERGENCES=(_record(), _record(outputs={"a"})))
        assert len(group.DIVERGENCES) == 2

    def test_a_non_tuple_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_group(DIVERGENCES=[_record()])

    def test_a_non_record_entry_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_group(DIVERGENCES=("skips NaN",))

    def test_the_implementation_spelling_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_group(divergences=(_record(),))

    def test_empty_declaration_does_not_populate_the_universe_cache(self) -> None:
        group = _make_group(DIVERGENCES=())
        assert "_subtype_universe_cache" not in group.__dict__

    def test_whole_plugin_declaration_does_not_populate_the_universe_cache(self) -> None:
        group = _make_group(DIVERGENCES=(_record(),))
        assert "_subtype_universe_cache" not in group.__dict__

    def test_records_are_stamped_with_the_feature_group_source(self) -> None:
        group = _make_group(DIVERGENCES=(_record(subtypes={"sum"}),))
        (record,) = group.divergence_records()
        assert record.source is DivergenceSource.FEATURE_GROUP
        assert record.framework is None
        assert record.subtypes == frozenset({"sum"})

    def test_no_records_reads_as_an_empty_tuple(self) -> None:
        assert _make_group().divergence_records() == ()


class TestContractVersionDeclaration:
    def test_default_is_none(self) -> None:
        assert FeatureGroup.CONTRACT_VERSION is None
        assert _make_group().CONTRACT_VERSION is None

    @pytest.mark.parametrize("value", [1, 2, 10])
    def test_a_positive_int_is_kept(self, value: int) -> None:
        assert _make_group(CONTRACT_VERSION=value).CONTRACT_VERSION == value

    def test_a_subclass_inherits_the_version(self) -> None:
        group = _make_group(CONTRACT_VERSION=2)
        child: Any = type("DivTestChild", (group,), {})
        assert child.CONTRACT_VERSION == 2

    @pytest.mark.parametrize("value", [True, False, "1", 1.0, [1]], ids=["true", "false", "str", "float", "list"])
    def test_a_non_int_raises_type_error(self, value: Any) -> None:
        with pytest.raises(TypeError):
            _make_group(CONTRACT_VERSION=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_an_int_below_one_raises_value_error(self, value: int) -> None:
        with pytest.raises(ValueError):
            _make_group(CONTRACT_VERSION=value)

    def test_the_lowercase_spelling_fails_at_definition(self) -> None:
        with pytest.raises(TypeError, match="CONTRACT_VERSION"):
            _make_group(contract_version=1)

    @pytest.mark.parametrize("name", ["contract_version", "CONTRACT_VERSION"])
    def test_an_implementation_cannot_declare_it(self, name: str) -> None:
        with pytest.raises(TypeError, match="contract"):
            _make_impl(_make_group(), **{name: 1})  # type: ignore[arg-type]


class TestImplementationDeclaration:
    def test_default_is_an_empty_tuple(self) -> None:
        assert Implementation.divergences == ()

    def test_a_valid_declaration_is_kept(self) -> None:
        record = _record(subtypes={"sum"})
        assert _make_impl(_make_group(), divergences=(record,)).divergences == (record,)

    def test_a_subtype_outside_the_universe_fails_at_definition(self) -> None:
        with pytest.raises(ValueError, match="outside the subtype universe"):
            _make_impl(_make_group(), divergences=(_record(subtypes={"bogus"}),))

    def test_subtypes_on_a_group_without_subtypes_fail_at_definition(self) -> None:
        plain = type("DivTestPlainOp", (OperationFeatureGroup,), {})
        with pytest.raises(ValueError):
            _make_impl(plain, divergences=(_record(subtypes={"sum"}),))

    def test_an_output_outside_output_parts_fails_at_definition(self) -> None:
        group = _make_group(OUTPUT_PARTS=("a",))
        with pytest.raises(ValueError):
            _make_impl(group, divergences=(_record(outputs={"b"}),))

    def test_a_non_tuple_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_impl(_make_group(), divergences=[_record()])

    def test_a_non_record_entry_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_impl(_make_group(), divergences=({"sum": "skips NaN"},))

    def test_the_feature_group_spelling_fails_at_definition(self) -> None:
        with pytest.raises(TypeError):
            _make_impl(_make_group(), DIVERGENCES=(_record(),))

    def test_the_legacy_divergent_spelling_fails_at_definition(self) -> None:
        with pytest.raises(TypeError, match="divergences"):
            _make_impl(_make_group(), divergent={"sum": "skips NaN"})

    def test_an_any_output_part_group_accepts_any_output_name(self) -> None:
        group = _make_group(OUTPUT_PARTS=(ANY_OUTPUT_PART,))
        impl = _make_impl(group, divergences=(_record(outputs={"col1"}),))
        assert impl.divergences[0].outputs == frozenset({"col1"})

    def test_an_any_output_part_group_accepts_any_output_name_in_its_own_divergences(self) -> None:
        group = _make_group(OUTPUT_PARTS=(ANY_OUTPUT_PART,), DIVERGENCES=(_record(outputs={"col1"}),))
        assert group.DIVERGENCES[0].outputs == frozenset({"col1"})

    def test_the_any_output_part_constant_mirrors_the_parser(self) -> None:
        assert divergence_module.ANY_OUTPUT_PART == ANY_OUTPUT_PART

    def test_the_removed_divergent_mapping_is_not_an_attribute(self) -> None:
        assert not hasattr(Implementation, "divergent")
        assert not hasattr(Implementation, "divergence")

    def test_records_are_stamped_with_implementation_and_framework(self) -> None:
        impl = _make_impl(_make_group(), divergences=(_record(subtypes={"sum"}),))
        (record,) = impl.divergence_records()
        assert record.source is DivergenceSource.IMPLEMENTATION
        assert record.framework == PythonDictFramework.get_class_name()

    def test_an_inherited_record_takes_the_subclass_framework(self) -> None:
        group = _make_group()
        parent = _make_impl(group, divergences=(_record(subtypes={"sum"}),))
        child: Any = type("DivTestChildImpl", (parent,), {"compute_framework": PyArrowTable})
        (inherited,) = child.divergence_records()
        assert inherited.framework == PyArrowTable.get_class_name()
        assert inherited.framework != PythonDictFramework.get_class_name()
        (own,) = parent.divergence_records()
        assert own.framework == PythonDictFramework.get_class_name()


class TestDivergencesFor:
    def test_returns_the_stamped_records_applying_to_the_feature(self) -> None:
        group = _make_group()
        whole = _record(kind=DivergenceKind.TYPE, reason="floats")
        impl = _make_impl(group, divergences=(_record(subtypes={"sum"}), _record(subtypes={"mean"}), whole))
        found = impl.divergences_for(group, "divtest_x", Options({KEY: "sum"}))
        assert [r.reason for r in found] == ["skips NaN", "floats"]
        assert all(r.source is DivergenceSource.IMPLEMENTATION for r in found)

    def test_gap_records_are_excluded(self) -> None:
        group = _make_group()
        gap = _record(kind=DivergenceKind.GAP, subtypes={"sum"})
        impl = _make_impl(group, divergences=(gap,))
        assert impl.divergences_for(group, "divtest_x", Options({KEY: "sum"})) == ()

    def test_an_explicit_part_selects_outputs_records(self) -> None:
        group = _make_group(OUTPUT_PARTS=("a", "b"))
        impl = _make_impl(group, divergences=(_record(outputs={"a"}),))
        options = Options({KEY: "sum"})
        assert len(impl.divergences_for(group, "divtest_x", options, parts=frozenset({"a"}))) == 1
        assert impl.divergences_for(group, "divtest_x", options, parts=frozenset({"b"})) == ()
        assert len(impl.divergences_for(group, "divtest_x", options)) == 1


class TestStepDivergences:
    """The step-level union of group and serving-implementation records over a FeatureSet."""

    @staticmethod
    def _call(group: Any, impl: Any, features: Any) -> tuple[Divergence, ...]:
        from mloda.core.abstract_plugins.implementation import step_divergences

        return tuple(step_divergences(group, impl, features))

    def test_group_and_implementation_records_are_joined(self) -> None:
        group = _make_group(DIVERGENCES=(_record(kind=DivergenceKind.TYPE, reason="floats"),))
        impl = _make_impl(group, divergences=(_record(subtypes={"sum"}),))

        found = self._call(group, impl, _feature_set(_feature()))

        assert [(r.source, r.reason) for r in found] == [
            (DivergenceSource.FEATURE_GROUP, "floats"),
            (DivergenceSource.IMPLEMENTATION, "skips NaN"),
        ]
        assert found[1].framework == PythonDictFramework.get_class_name()

    def test_gap_records_are_included(self) -> None:
        group = _make_group(DIVERGENCES=(_record(kind=DivergenceKind.GAP, subtypes={"sum"}, reason="g"),))
        impl = _make_impl(group, divergences=(_record(kind=DivergenceKind.GAP, subtypes={"sum"}, reason="i"),))

        found = self._call(group, impl, _feature_set(_feature()))

        assert sorted(r.reason for r in found) == ["g", "i"]

    def test_records_for_other_subtypes_are_filtered_out(self) -> None:
        group = _make_group(DIVERGENCES=(_record(subtypes={"mean"}, reason="g-mean"),))
        impl = _make_impl(group, divergences=(_record(subtypes={"mean"}, reason="i-mean"),))

        assert self._call(group, impl, _feature_set(_feature(kind="sum"))) == ()
        assert len(self._call(group, impl, _feature_set(_feature(kind="mean")))) == 2

    def test_the_union_spans_every_feature_of_the_set(self) -> None:
        group = _make_group()
        impl = _make_impl(
            group, divergences=(_record(subtypes={"sum"}, reason="s"), _record(subtypes={"mean"}, reason="m"))
        )

        found = self._call(group, impl, _feature_set(_feature("divtest_a", "sum"), _feature("divtest_b", "mean")))

        assert sorted(r.reason for r in found) == ["m", "s"]

    def test_parts_only_features_select_outputs_records(self) -> None:
        group = _make_group(OUTPUT_PARTS=("a", "b"), DIVERGENCES=(_record(outputs={"a"}, reason="a-only"),))

        assert len(self._call(group, None, _feature_set(_feature(part="a")))) == 1
        assert self._call(group, None, _feature_set(_feature(part="b"))) == ()
        assert len(self._call(group, None, _feature_set(_feature()))) == 1

    def test_a_plain_group_without_an_implementation_yields_its_whole_plugin_records(self) -> None:
        plain = type("DivTestPlainStep", (FeatureGroup,), {"DIVERGENCES": (_record(reason="plain"),)})

        found = self._call(plain, None, _feature_set(_feature("divtest_plain", None)))

        assert [(r.source, r.framework, r.reason) for r in found] == [(DivergenceSource.FEATURE_GROUP, None, "plain")]

    def test_a_plain_group_without_records_yields_nothing(self) -> None:
        plain = type("DivTestPlainEmpty", (FeatureGroup,), {})
        assert self._call(plain, None, _feature_set(_feature("divtest_plain", None))) == ()

    def test_duplicates_across_features_are_collapsed(self) -> None:
        group = _make_group(DIVERGENCES=(_record(reason="once"),))
        features = _feature_set(_feature("divtest_a"), _feature("divtest_b"))

        assert len(self._call(group, None, features)) == 1

    def test_the_order_is_deterministic(self) -> None:
        group = _make_group(
            DIVERGENCES=(
                _record(kind=DivergenceKind.TYPE, reason="z"),
                _record(reason="b"),
                _record(reason="a"),
            )
        )
        impl = _make_impl(group, divergences=(_record(reason="i"),))
        features = _feature_set(_feature())

        first = self._call(group, impl, features)
        assert first == self._call(group, impl, features)
        assert [r.source for r in first] == [DivergenceSource.FEATURE_GROUP] * 3 + [DivergenceSource.IMPLEMENTATION]
        assert [r.reason for r in first[:3]] == ["a", "b", "z"]

    def test_ties_on_everything_but_variant_or_docs_order_by_variant_then_docs(self) -> None:
        from mloda.core.abstract_plugins.components.divergence import divergence_order

        variants = [_record(kind=DivergenceKind.VARIANT, variant=v, reason="same") for v in ("b", "a", "c")]
        docs = [_record(cause=Cause.OTHER, docs=d, reason="same") for d in ("y", "x")]
        group = _make_group(DIVERGENCES=(*variants, *docs))

        found = self._call(group, None, _feature_set(_feature()))

        assert [r.variant for r in found if r.kind is DivergenceKind.VARIANT] == ["a", "b", "c"]
        assert [r.docs for r in found if r.cause is Cause.OTHER] == ["x", "y"]
        assert list(found) == sorted(found, key=divergence_order)
        assert divergence_order(stamped(variants[1], DivergenceSource.FEATURE_GROUP, None)) < divergence_order(
            stamped(variants[0], DivergenceSource.FEATURE_GROUP, None)
        )
