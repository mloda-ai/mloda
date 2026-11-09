"""check_subclass_scope: a well-scoped subclass passes, each broken pair fails only its own sub-check."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.subclass_scope import assert_subclass_scope, check_subclass_scope
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.divergence import Cause, Divergence, DivergenceKind
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.user.python_dict import PythonDictFramework

Pair = tuple[type[FeatureGroup], type[FeatureGroup]]
SELECTING = Options(group={"variant": "x"})
DATA = TypedTestData({"sc_src": (DataType.INT64, [1, 2, 3])})
CHANGED = Divergence(kind=DivergenceKind.VALUE, cause=Cause.BACKEND, reason="values changed")
CHANGED_TYPE = Divergence(kind=DivergenceKind.TYPE, cause=Cause.BACKEND, reason="type changed")
CHANGED_PART = Divergence(
    kind=DivergenceKind.VALUE, cause=Cause.BACKEND, outputs=frozenset({"p"}), reason="part changed"
)
GAP = Divergence(kind=DivergenceKind.GAP, cause=Cause.NOT_IMPLEMENTED)
CHANGED_VARIANT = Divergence(kind=DivergenceKind.VARIANT, cause=Cause.BACKEND, variant="v", reason="variant changed")
ALLOWING = Options(group={"variant": "x", str(DefaultOptionKeys.allow_variants): "v"})
Calc = Callable[[Any, Any, Any], Any]


def _calc(main: int | None, part: int | None = None) -> Calc:
    """A calculate function writing sc_a = sc_src * main and sc_a~p = sc_src * part; None skips the column."""

    def calc(cls: Any, data: Any, features: Any) -> Any:
        if main is not None:
            data["sc_a"] = [v * main for v in data["sc_src"]]
        if part is not None:
            data["sc_a~p"] = [v * part for v in data["sc_src"]]
        return data

    return calc


def make_parent(
    *,
    base: type[FeatureGroup] = FeatureGroup,
    calculates: bool = True,
    parts: bool = False,
    divergences: tuple[Divergence, ...] = (),
    variants: frozenset[str] = frozenset(),
) -> type[FeatureGroup]:
    class ScopeParent(base):  # type: ignore[valid-type, misc]
        DIVERGENCES = divergences
        if variants:
            VARIANTS = variants
        if parts:
            OUTPUT_PARTS = ("p",)

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"sc_a"}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("sc_src")}

        if calculates:
            calculate_feature = classmethod(_calc(2, 3 if parts else None))

    return ScopeParent


def _gated_subclass(
    parent: type[FeatureGroup],
    required_key: str,
    calc: Calc | None = None,
    divergences: tuple[Divergence, ...] = (),
) -> type[FeatureGroup]:
    class GatedChild(parent):  # type: ignore[valid-type, misc]
        if divergences:
            DIVERGENCES = divergences
        if calc is not None:
            calculate_feature = classmethod(calc)

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if required_key not in options.group and required_key not in options.context:
                return False
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)  # type: ignore[no-any-return]

    return GatedChild


def make_well_scoped() -> Pair:
    parent = make_parent()
    return parent, _gated_subclass(parent, "variant")


def make_changed(calc: Calc | None = None, divergences: tuple[Divergence, ...] = (), *, parts: bool = False) -> Pair:
    parent = make_parent(parts=parts)
    return parent, _gated_subclass(parent, "variant", calc or _calc(5), divergences)


def make_changed_with_inherited_record() -> Pair:
    parent = make_parent(divergences=(CHANGED,))
    return parent, _gated_subclass(parent, "variant", _calc(5))


def make_changed_with_variant_record() -> Pair:
    parent = make_parent(variants=frozenset({"v"}))
    return parent, _gated_subclass(parent, "variant", _calc(5), (CHANGED_VARIANT,))


def make_broken_run() -> Pair:
    def calc(cls: Any, data: Any, features: Any) -> Any:
        raise RuntimeError("boom")

    return make_changed(calc)


def make_ungated() -> Pair:
    parent = make_parent()

    class UngatedChild(parent):  # type: ignore[valid-type, misc]
        pass

    return parent, UngatedChild


def make_gate_rejects_own_option() -> Pair:
    parent = make_parent()
    return parent, _gated_subclass(parent, "other")


def make_unrelated() -> Pair:
    return make_parent(), make_well_scoped()[1]


def make_nameless_pair() -> Pair:
    class NamelessParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "nl_a"

    class NamelessChild(NamelessParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "nl_a" and "variant" in options.group

    return NamelessParent, NamelessChild


def make_keyed_pair() -> Pair:
    class KeyedParent(FeatureGroup):
        PROPERTY_MAPPING = {"variant": PropertySpec("Variant.", allowed_values=("x", "y"), strict_validation=True)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"kp_a"}

    class KeyedChild(KeyedParent):
        pass

    return KeyedParent, KeyedChild


def make_request_pair() -> Pair:
    class RequestParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "rq_a"

    class RequestChild(RequestParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "rq_a" and "variant" in options.group

    return RequestParent, RequestChild


def make_optional_option_stealer() -> Pair:
    class OptParent(FeatureGroup):
        PROPERTY_MAPPING = {"opt": PropertySpec("Optional.", allowed_values=("a", "b"), default=None)}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {"op_a"}

    class OptStealer(OptParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            if not {"opt", "variant"} & {*options.group, *options.context}:
                return False
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    return OptParent, OptStealer


def make_parent_rejecting_selecting_option() -> Pair:
    class StrictParent(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "sp_a" and "variant" not in options.group

    class StrictChild(StrictParent):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Options, data_access_collection: Any = None
        ) -> bool:
            return str(feature_name) == "sp_a" and "variant" in options.group

    return StrictParent, StrictChild


def make_unloaded_framework_pair() -> tuple[type[FeatureGroup], type[FeatureGroup], type[ComputeFramework]]:
    class UnloadedFramework(ComputeFramework):
        pass

    parent = make_parent()

    class UnloadedChild(parent):  # type: ignore[valid-type, misc]
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {UnloadedFramework}

    return parent, UnloadedChild, UnloadedFramework


@pytest.mark.parametrize("data", [None, DATA])
def test_well_scoped_subclass_passes(data: TypedTestData | None) -> None:
    parent, sub = make_well_scoped()
    result = check_subclass_scope(sub, parent, SELECTING, input_data=data)
    assert result.check == "subclass_scope"
    assert result.subject == sub.__name__
    assert result.failures == ()


@pytest.mark.parametrize(
    ("make", "prefix", "data", "phrase"),
    [
        (make_ungated, "steals:", None, ""),
        (make_gate_rejects_own_option, "misses:", None, ""),
        (make_ungated, "steals:", DATA, ""),
        (make_changed, "values:", DATA, "differs"),
        (functools.partial(make_changed, divergences=(GAP,)), "values:", DATA, "differs"),
        (make_changed_with_inherited_record, "values:", DATA, "differs"),
        (
            functools.partial(make_changed, _calc(5, 3), (CHANGED_PART,), parts=True),
            "values:",
            DATA,
            "differs",
        ),
        (functools.partial(make_changed, _calc(2), parts=True), "values:", DATA, "missing"),
        (make_changed_with_variant_record, "values:", DATA, "differs"),
        (make_broken_run, "values:", DATA, "run raised"),
    ],
)
def test_broken_pair_fails_only_its_sub_check(
    make: Callable[[], Pair], prefix: str, data: TypedTestData | None, phrase: str
) -> None:
    parent, sub = make()
    result = check_subclass_scope(sub, parent, SELECTING, input_data=data)
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures
    assert phrase in " ".join(result.failures), result.failures


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(functools.partial(make_changed, divergences=(CHANGED,)), id="value"),
        pytest.param(functools.partial(make_changed, divergences=(CHANGED_TYPE,)), id="type"),
        pytest.param(functools.partial(make_changed, _calc(2, 9), (CHANGED_PART,), parts=True), id="part"),
    ],
)
def test_declared_divergence_excuses_changed_column(make: Callable[[], Pair]) -> None:
    parent, sub = make()
    assert check_subclass_scope(sub, parent, SELECTING, input_data=DATA).failures == ()


def test_allowed_variant_record_excuses_changed_column() -> None:
    parent, sub = make_changed_with_variant_record()
    assert check_subclass_scope(sub, parent, ALLOWING, input_data=DATA).failures == ()


def test_misses_failure_skips_values_with_note() -> None:
    parent, sub = make_gate_rejects_own_option()
    result = check_subclass_scope(sub, parent, SELECTING, input_data=DATA)
    assert all(f.startswith("misses:") for f in result.failures), result.failures
    assert result.failures
    assert any(n.startswith("values: not compared") for n in result.notes), result.notes


def test_duplicate_request_names_fail_once() -> None:
    parent, sub = make_changed()
    result = check_subclass_scope(sub, parent, SELECTING, requests=["sc_a", "sc_a"], input_data=DATA)
    values = [f for f in result.failures if f.startswith("values:") and "sc_a" in f]
    assert len(values) == 1, result.failures


def test_no_input_data_notes_values_not_compared() -> None:
    parent, sub = make_changed()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert result.failures == ()
    assert any(n.startswith("values:") and "no input_data" in n for n in result.notes), result.notes


def test_parent_without_calculate_feature_notes_and_skips_values() -> None:
    parent = make_parent(calculates=False)
    sub = _gated_subclass(parent, "variant", _calc(5))
    result = check_subclass_scope(sub, parent, SELECTING, input_data=DATA)
    assert result.failures == ()
    assert any(n.startswith("values:") for n in result.notes), result.notes


@pytest.mark.parametrize(("record", "passes"), [(True, True), (False, False)])
def test_operation_group_values_use_serving_implementations(
    make_implementation: Callable[..., type[Implementation[Any]]], record: bool, passes: bool
) -> None:
    parent = make_parent(base=OperationFeatureGroup, calculates=False)
    sub = _gated_subclass(parent, "variant")
    make_implementation(parent, PythonDictFramework, _calc(2))
    sub_impl = make_implementation(sub, PythonDictFramework, _calc(5))
    if record:
        setattr(sub_impl, "divergences", (CHANGED,))
    result = check_subclass_scope(sub, parent, SELECTING, input_data=DATA)
    assert result.passed is passes, result.failures
    assert all(f.startswith("values:") for f in result.failures), result.failures


def test_not_a_subclass_is_a_setup_failure() -> None:
    parent, sub = make_unrelated()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("setup:") for f in result.failures), result.failures


def test_selecting_key_already_in_base_options_is_a_setup_failure() -> None:
    parent, sub = make_keyed_pair()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("setup:") for f in result.failures), result.failures


def test_parent_without_names_needs_explicit_requests() -> None:
    parent, sub = make_nameless_pair()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("names:") for f in result.failures), result.failures


def test_explicit_requests_pass_for_nameless_parent() -> None:
    parent, sub = make_request_pair()
    assert check_subclass_scope(sub, parent, SELECTING, requests=["rq_a"]).failures == ()


def test_partial_steal_by_optional_parent_option_fails_only_steals() -> None:
    parent, sub = make_optional_option_stealer()
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert all(f.startswith("steals:") for f in result.failures), result.failures


def test_subclass_serving_request_alone_when_parent_rejects_selecting_option_passes() -> None:
    parent, sub = make_parent_rejecting_selecting_option()
    assert check_subclass_scope(sub, parent, SELECTING, requests=["sp_a"]).failures == ()


def test_unloaded_compute_framework_is_a_setup_failure_naming_the_class(monkeypatch: pytest.MonkeyPatch) -> None:
    parent, sub, dummy = make_unloaded_framework_pair()
    loaded = PreFilterPlugins.get_cfw_subclasses()
    monkeypatch.setattr(
        PreFilterPlugins, "get_cfw_subclasses", staticmethod(lambda: {c for c in loaded if c is not dummy})
    )
    result = check_subclass_scope(sub, parent, SELECTING)
    assert not result.passed
    assert any(f.startswith("setup:") and sub.__name__ in f for f in result.failures), result.failures


def test_parent_rejecting_its_own_request_is_a_setup_failure() -> None:
    parent, sub = make_request_pair()
    result = check_subclass_scope(sub, parent, SELECTING, requests=["not_rq_a"])
    assert not result.passed
    assert any(f.startswith("setup:") for f in result.failures), result.failures


def test_selecting_option_in_context_passes_for_gated_subclass() -> None:
    parent, sub = make_well_scoped()
    assert check_subclass_scope(sub, parent, Options(context={"variant": "x"})).failures == ()


def test_explicitly_empty_requests_is_a_names_failure_without_pass_requests_hint() -> None:
    parent, sub = make_well_scoped()
    result = check_subclass_scope(sub, parent, SELECTING, requests=[])
    assert not result.passed
    assert all(f.startswith("names:") for f in result.failures), result.failures
    assert all("pass requests=" not in f for f in result.failures), result.failures


def test_assert_passes_silently_for_well_scoped_subclass() -> None:
    parent, sub = make_well_scoped()
    assert_subclass_scope(sub, parent, SELECTING)


def test_assert_raises_pytest_failure_for_ungated_subclass() -> None:
    parent, sub = make_ungated()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_subclass_scope(sub, parent, SELECTING)
    assert "steals:" in str(excinfo.value)


def test_assert_forwards_input_data_and_raises_on_undeclared_value_change() -> None:
    parent, sub = make_changed()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_subclass_scope(sub, parent, SELECTING, input_data=DATA)
    assert "values:" in str(excinfo.value)
