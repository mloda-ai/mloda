"""check_operation_group_shape: a conforming operation group passes, each broken one fails only its own check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.operation_group_shape import assert_operation_group_shape, check_operation_group_shape
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import OperationFeatureGroup, SubtypeDeclaration, property_spec
from mloda.user.python_dict import PythonDictFramework

KEY = "opshape_kind"
SHAPE_B_KEY = "opshape_b_kind"
Factory = Callable[[], tuple[type[OperationFeatureGroup], dict[str, Any]]]


def _make_group_a() -> type[OperationFeatureGroup]:
    class OpShapeA(OperationFeatureGroup):
        SUBTYPES = SubtypeDeclaration(key=KEY, parametric_families={"rolling": "Rolling window"})
        PROPERTY_MAPPING = {
            KEY: property_spec("Kind.", strict=True, allowed_values={"sum": "Sum", "median": "Median", "mean": "Mean"}),
        }

    return OpShapeA


def _make_impl(group: type[OperationFeatureGroup], **attrs: Any) -> type[Implementation[Any]]:
    body: dict[str, Any] = {
        "feature_group": group,
        "compute_framework": PythonDictFramework,
        "calculate": classmethod(lambda cls, group, data, features, framework: data),
        **attrs,
    }
    return type("OpShapeImpl", (Implementation,), body)


def make_conforming() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    impl = _make_impl(
        group,
        supported=frozenset({"sum", "median", "rolling"}),
        divergent={"sum": "skips NaN"},
    )
    return group, {"impl": impl}


def make_supported_outside_universe() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    return group, {"impl": _make_impl(group, supported=frozenset({"sum", "bogus"}))}


def make_divergent_outside_universe() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    return group, {"impl": _make_impl(group, divergent={"bogus": "reason"})}


def make_empty_reason() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    return group, {"impl": _make_impl(group, divergent={"sum": ""})}


def make_one_sided_alias() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    impl = _make_impl(group, divergent={"sum": "skips NaN"})
    return group, {"impl": impl, "alias_pairs": (("sum", "mean"),)}


def make_alias_outside_universe() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    return group, {"impl": _make_impl(group), "alias_pairs": (("sum", "bogus"),)}


def make_unsupported_not_declining() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    # canonical_subtype returning None lets an unsupported subtype slip past the supported gate.
    impl = _make_impl(
        group,
        supported=frozenset({"sum"}),
        canonical_subtype=classmethod(lambda cls, group, feature_name, options: None),
    )
    return group, {"impl": impl}


def make_probe_never_accepted() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    group = _make_group_a()
    impl = _make_impl(group, accepts=classmethod(lambda cls, feature_name, options, *, part=None: False))
    return group, {"impl": impl}


def make_no_implementation() -> tuple[type[OperationFeatureGroup], dict[str, Any]]:
    return _make_group_a(), {}


def make_shape_b(
    supported: frozenset[str] | None = frozenset({"up"}),
) -> tuple[type[OperationFeatureGroup], Callable[[str], tuple[str, Options]], type[Any]]:
    class OpShapeB(OperationFeatureGroup):
        SUBTYPES = SubtypeDeclaration(
            universe={"up", "down"},
            resolver=lambda name, options: options.get(SHAPE_B_KEY),
        )

    impl = _make_impl(OpShapeB, supported=supported)

    def probe(subtype: str) -> tuple[str, Options]:
        return f"opshape_b_{subtype}", Options(group={SHAPE_B_KEY: subtype})

    return OpShapeB, probe, impl


@pytest.mark.parametrize("alias_pairs", [(), (("median", "mean"),)])
def test_conforming_group_passes(alias_pairs: tuple[tuple[str, str], ...]) -> None:
    group, _keep = make_conforming()
    result = check_operation_group_shape(group, alias_pairs=alias_pairs)
    assert result.check == "operation_group_shape"
    assert result.subject == "OpShapeA"
    assert result.failures == ()


def test_existing_toy_operation_group_passes() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_operation_group import OpToyFG

    assert check_operation_group_shape(OpToyFG).failures == ()


def test_shape_b_group_passes_with_probe() -> None:
    group, probe, _impl = make_shape_b()
    assert check_operation_group_shape(group, probe=probe).failures == ()


def test_shape_b_group_without_probe_fails_only_declines() -> None:
    group, _probe, _impl = make_shape_b()
    result = check_operation_group_shape(group)
    assert not result.passed
    assert all(f.startswith("declines:") for f in result.failures), result.failures
    assert any("cannot probe" in f for f in result.failures), result.failures


@pytest.mark.parametrize(
    ("make", "prefix", "message"),
    [
        (make_supported_outside_universe, "supported:", "outside the subtype universe"),
        (make_divergent_outside_universe, "divergent:", "outside the subtype universe"),
        (make_empty_reason, "divergent:", "no reason"),
        (make_one_sided_alias, "aliases:", "diverges for only one"),
        (make_alias_outside_universe, "aliases:", "outside the subtype universe"),
        (make_unsupported_not_declining, "declines:", "does not decline"),
        (make_probe_never_accepted, "declines:", "probe not accepted"),
        (make_no_implementation, "implementations:", "no loaded Implementation"),
    ],
)
def test_broken_group_fails_only_its_check(make: Factory, prefix: str, message: str) -> None:
    group, parts = make()
    kwargs = {k: v for k, v in parts.items() if k != "impl"}
    result = check_operation_group_shape(group, **kwargs)
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures
    assert any(message in f for f in result.failures), result.failures


def test_assert_passes_silently_for_conforming_group() -> None:
    group, _keep = make_conforming()
    assert_operation_group_shape(group)


def test_assert_raises_pytest_failure_for_broken_group() -> None:
    group, _keep = make_no_implementation()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_operation_group_shape(group)
    assert "implementations:" in str(excinfo.value)


def test_plain_operation_group_without_subtypes_passes() -> None:
    class OpShapePlain(OperationFeatureGroup):
        pass

    _keep = _make_impl(OpShapePlain)
    assert check_operation_group_shape(OpShapePlain).failures == ()


def test_shape_b_group_with_unrestricted_implementations_passes_without_probe() -> None:
    group, _probe, _impl = make_shape_b(supported=None)
    assert check_operation_group_shape(group).failures == ()


def test_unrestricted_supported_passes() -> None:
    group = _make_group_a()
    _keep = _make_impl(group, supported=None)
    assert check_operation_group_shape(group).failures == ()


def test_empty_supported_passes_when_every_subtype_declines() -> None:
    group = _make_group_a()
    _keep = _make_impl(group, supported=frozenset())
    assert check_operation_group_shape(group).failures == ()


def test_empty_supported_fails_when_a_subtype_is_still_accepted() -> None:
    group = _make_group_a()
    _keep = _make_impl(group, supported=frozenset(), canonical_subtype=classmethod(lambda cls, g, name, options: None))
    result = check_operation_group_shape(group)
    assert any(f.startswith("declines:") and "does not decline" in f for f in result.failures), result.failures


def test_unavailable_framework_is_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    group, _parts = make_unsupported_not_declining()
    assert not check_operation_group_shape(group).passed
    monkeypatch.setattr(PythonDictFramework, "is_available", classmethod(lambda cls: False))
    assert check_operation_group_shape(group).failures == ()


def test_decline_for_a_reason_other_than_the_supported_gate_fails() -> None:
    group = _make_group_a()
    # The supported gate never sees the subtype (canonical None); accepts declines it for another reason.
    _keep = _make_impl(
        group,
        supported=frozenset({"sum"}),
        canonical_subtype=classmethod(lambda cls, g, name, options: None),
        accepts=classmethod(lambda cls, name, options, *, part=None: name == "conformance_sum"),
    )
    result = check_operation_group_shape(group)
    assert not result.passed
    assert any(f.startswith("declines:") for f in result.failures), result.failures


def test_alias_outside_universe_is_reported_once_across_implementations() -> None:
    pytest.importorskip("pandas")
    from mloda.user.pandas import PandasDataFrame

    group = _make_group_a()
    _first = _make_impl(group)
    _second = _make_impl(group, compute_framework=PandasDataFrame)
    result = check_operation_group_shape(group, alias_pairs=(("sum", "bogus"),))
    assert len([f for f in result.failures if f.startswith("aliases:")]) == 1, result.failures


def test_alias_outside_universe_is_reported_without_implementation() -> None:
    group, _parts = make_no_implementation()
    result = check_operation_group_shape(group, alias_pairs=(("sum", "bogus"),))
    assert any(f.startswith("aliases:") for f in result.failures), result.failures
