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


def make_shape_b() -> tuple[type[OperationFeatureGroup], Callable[[str], tuple[str, Options]], type[Any]]:
    class OpShapeB(OperationFeatureGroup):
        SUBTYPES = SubtypeDeclaration(
            universe={"up", "down"},
            resolver=lambda name, options: options.get(SHAPE_B_KEY),
        )

    impl = _make_impl(OpShapeB, supported=frozenset({"up"}))

    def probe(subtype: str) -> tuple[str, Options]:
        return f"opshape_b_{subtype}", Options(group={SHAPE_B_KEY: subtype})

    return OpShapeB, probe, impl


def test_conforming_group_passes() -> None:
    group, _keep = make_conforming()
    result = check_operation_group_shape(group)
    assert result.check == "operation_group_shape"
    assert result.subject == "OpShapeA"
    assert result.failures == ()


def test_conforming_group_with_consistent_alias_passes() -> None:
    group, _keep = make_conforming()
    assert check_operation_group_shape(group, alias_pairs=(("median", "mean"),)).passed


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
    ("make", "prefix"),
    [
        (make_supported_outside_universe, "supported:"),
        (make_divergent_outside_universe, "divergent:"),
        (make_empty_reason, "divergent:"),
        (make_one_sided_alias, "aliases:"),
        (make_alias_outside_universe, "aliases:"),
        (make_unsupported_not_declining, "declines:"),
        (make_probe_never_accepted, "declines:"),
        (make_no_implementation, "implementations:"),
    ],
)
def test_broken_group_fails_only_its_check(make: Factory, prefix: str) -> None:
    group, parts = make()
    kwargs = {k: v for k, v in parts.items() if k != "impl"}
    result = check_operation_group_shape(group, **kwargs)
    assert not result.passed
    assert all(f.startswith(prefix) for f in result.failures), result.failures


def test_assert_passes_silently_for_conforming_group() -> None:
    group, _keep = make_conforming()
    assert_operation_group_shape(group)


def test_assert_raises_pytest_failure_for_broken_group() -> None:
    group, _keep = make_no_implementation()
    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_operation_group_shape(group)
    assert "implementations:" in str(excinfo.value)
