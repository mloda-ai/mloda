"""ConformanceSuite hooks are validated at class creation: near-miss names on a test class are errors."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda._testing.result import CheckResult
from mloda._testing.declarations import Cause
from mloda._testing.suite import ConformanceSuite, HookError, SwitchedOffError, assert_passed, check_hooks
from mloda.user.python_dict import PythonDictFramework


class _Plugin:
    pass


class _Base(ConformanceSuite, suite_base=True):
    match_names: ClassVar[tuple[str, ...]] = ()

    def input_data(self) -> list[str]:
        return []


def _hint(name: str, hook: str) -> str:
    return f"'{name}' is not a hook; did you mean '{hook}'? Rename it if it is not meant as one."


def test_misspelled_attribute_hook_raises_at_class_creation() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugin = _Plugin
            match_name = ("x",)

    message = str(excinfo.value)
    assert "_Bad" in message
    assert _hint("match_name", "match_names") in message
    assert "_testing" not in message


def test_misspelled_inherited_plugin_hook_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn = _Plugin

    assert _hint("plugn", "plugin") in str(excinfo.value)


def test_misspelled_method_hook_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            def input_dat(self) -> list[str]:
                return []

    assert _hint("input_dat", "input_data") in str(excinfo.value)


def test_annotation_only_typo_raises() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn: ClassVar[type[Any]]

    assert _hint("plugn", "plugin") in str(excinfo.value)


def test_multiple_failures_are_joined() -> None:
    with pytest.raises(HookError) as excinfo:

        class _Bad(_Base):
            plugn = _Plugin
            match_name = ("x",)

    assert str(excinfo.value) == "_Bad: " + _hint("match_name", "match_names") + "; " + _hint("plugn", "plugin")


def test_correct_subclass_with_unrelated_public_names_passes() -> None:
    class _Good(_Base):
        pytestmark = [pytest.mark.filterwarnings("ignore")]
        plugin = _Plugin
        match_names = ("x",)
        DATA = {"a": [1]}

        def setup_method(self) -> None:
            pass

        def input_data(self) -> list[str]:
            return ["x"]

        def build_rows(self) -> list[int]:
            return [1]

        match_values = ("x",)

        def _input_dat(self) -> None:
            pass

        def test_input_data(self) -> None:
            pass

    assert check_hooks(_Good) == CheckResult(check="hooks", subject="_Good", failures=())


def test_check_hooks_on_class_mutated_after_creation_returns_exact_failure() -> None:
    class _Good(_Base):
        plugin = _Plugin

    setattr(_Good, "match_name", ("x",))

    result = check_hooks(_Good)

    assert result == CheckResult(
        check="hooks",
        subject="_Good",
        failures=(_hint("match_name", "match_names"),),
    )


def test_check_hooks_on_root_suite_passes() -> None:
    assert check_hooks(ConformanceSuite) == CheckResult(check="hooks", subject="ConformanceSuite", failures=())


def test_second_level_suite_base_hook_becomes_known() -> None:
    class _Second(_Base, suite_base=True):
        extra_hook: ClassVar[int] = 0

    class _Good(_Second):
        extra_hook = 1

    assert check_hooks(_Good).failures == ()

    with pytest.raises(HookError) as excinfo:

        class _Bad(_Second):
            extra_hok = 1

    assert _hint("extra_hok", "extra_hook") in str(excinfo.value)


def test_suite_base_is_not_inherited() -> None:
    class _Plain(_Base):
        extra_setting = 1

    class _Child(_Plain):
        extra_settin = 2

    assert check_hooks(_Child).failures == ()


def test_plain_mixin_overriding_a_hook_is_allowed() -> None:
    class _Mixin:
        match_names: ClassVar[tuple[str, ...]] = ("x",)
        match_name: ClassVar[tuple[str, ...]] = ("y",)

        def input_data(self) -> list[str]:
            return ["x"]

    class _Good(_Mixin, _Base):
        plugin = _Plugin

    assert check_hooks(_Good).failures == ()


def _raises_name_error(parent: type) -> bool:
    # dont_inherit keeps this file's future import out of the probe source.
    code = compile("class Probe(Parent):\n    peer: Undefined\n", "<suite>", "exec", dont_inherit=True)
    try:
        exec(code, {"Parent": parent})  # nosec B102
    except NameError:
        return True
    return False


def test_unresolvable_annotation_behaves_like_a_plain_class() -> None:
    assert _raises_name_error(_Base) == _raises_name_error(object)


def test_default_compute_framework_is_python_dict() -> None:
    assert ConformanceSuite.compute_framework is PythonDictFramework


def _contract_base() -> type[ConformanceSuite]:
    class _Contract(ConformanceSuite, suite_base=True):
        def test_alpha(self) -> None:
            pass

        def test_beta(self) -> None:
            pass

    return _Contract


def _host_with(kind: str, declared: dict[str, Cause] | None) -> type[ConformanceSuite]:
    base = _contract_base()
    declared = declared or {}
    if kind == "redefined":

        class _Redefined(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            switched_off = declared

            def test_alpha(self) -> None:
                pass

        return _Redefined
    if kind == "blanked":

        class _Blanked(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            switched_off = declared
            test_alpha = None

        return _Blanked

    class _Mixin:
        def test_alpha(self) -> None:
            pass

    class _Mixed(_Mixin, base):  # type: ignore[valid-type, misc]
        plugin = _Plugin
        switched_off = declared

    return _Mixed


@pytest.mark.parametrize("kind", ["redefined", "blanked", "mixin"])
def test_undeclared_override_of_contract_test_raises(kind: str) -> None:
    with pytest.raises(SwitchedOffError) as excinfo:
        _host_with(kind, None)

    assert "test_alpha" in str(excinfo.value)
    assert "test_beta" not in str(excinfo.value)


@pytest.mark.parametrize("kind", ["redefined", "blanked", "mixin"])
def test_declared_override_of_contract_test_passes(kind: str) -> None:
    host = _host_with(kind, {"test_alpha": Cause.NOT_IMPLEMENTED})

    assert host.switched_off == {"test_alpha": Cause.NOT_IMPLEMENTED}


MARKS = [pytest.mark.skip(reason="x"), pytest.mark.skipif(True, reason="x"), pytest.mark.xfail(reason="x")]


@pytest.mark.parametrize("declare_all", [False, True])
@pytest.mark.parametrize("mark", MARKS)
def test_class_body_mark_requires_every_contract_test_declared(mark: Any, declare_all: bool) -> None:
    base = _contract_base()
    declared = {"test_alpha": Cause.BACKEND}
    if declare_all:
        declared["test_beta"] = Cause.BACKEND

    def build() -> type[ConformanceSuite]:
        class _Marked(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            pytestmark = [mark]
            switched_off = declared

        return _Marked

    if declare_all:
        assert build().switched_off == declared
        return
    with pytest.raises(SwitchedOffError) as excinfo:
        build()

    message = str(excinfo.value)
    assert "test_beta" in message
    assert "test_alpha" not in message


@pytest.mark.parametrize(
    ("declared", "mentioned"),
    [({"test_alpa": Cause.OTHER}, ["test_alpa", "test_alpha"]), ({"test_alpha": "backend"}, ["test_alpha"])],
)
def test_invalid_switched_off_entry_raises(declared: Any, mentioned: list[str]) -> None:
    base = _contract_base()

    with pytest.raises(SwitchedOffError) as excinfo:

        class _Bad(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            switched_off = declared

    assert all(name in str(excinfo.value) for name in mentioned)


def test_switched_off_set_raises_switched_off_error() -> None:
    base = _contract_base()

    with pytest.raises(SwitchedOffError):

        class _Bad(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            switched_off = {"test_alpha"}


def test_child_host_replacing_switched_off_must_merge_parent_mapping() -> None:
    base = _contract_base()

    class _A(base):  # type: ignore[valid-type, misc]
        plugin = _Plugin
        switched_off = {"test_alpha": Cause.OTHER}

        def test_alpha(self) -> None:
            pass

    with pytest.raises(SwitchedOffError) as excinfo:

        class _B(_A):
            switched_off = {"test_beta": Cause.OTHER}

    message = str(excinfo.value)
    assert "test_alpha" in message
    assert "merg" in message


def test_switched_off_typo_is_a_hook_error() -> None:
    base = _contract_base()

    with pytest.raises(HookError) as excinfo:

        class _Bad(base):  # type: ignore[valid-type, misc]
            plugin = _Plugin
            switched_of = {"test_alpha": Cause.OTHER}

    assert _hint("switched_of", "switched_off") in str(excinfo.value)


def test_switched_off_defaults_to_empty() -> None:
    assert dict(ConformanceSuite.switched_off) == {}


def test_suite_base_may_redefine_parent_tests() -> None:
    base = _contract_base()

    class _Second(base, suite_base=True):  # type: ignore[valid-type, misc, call-arg]
        def test_alpha(self) -> None:
            pass

    class _Host(_Second):
        plugin = _Plugin

    assert _Host.plugin is _Plugin


@pytest.mark.parametrize("mark", MARKS)
def test_suite_base_with_skip_mark_is_rejected(mark: Any) -> None:
    with pytest.raises(SwitchedOffError):

        class _Bad(ConformanceSuite, suite_base=True):
            pytestmark = [mark]

            def test_alpha(self) -> None:
                pass


def _source_mixin() -> type:
    from mloda._testing.source.source_suite import SourceFeatureGroupTestMixin

    return SourceFeatureGroupTestMixin


def _column_discovery_mixin() -> type:
    from mloda._testing.source.column_discovery_suite import ColumnDiscoveryContractTestMixin

    return ColumnDiscoveryContractTestMixin


@pytest.mark.parametrize(
    ("get_mixin", "contract_test", "near_miss", "hook"),
    [
        (_source_mixin, "test_claims_its_own_source", "feature_group_clas", "feature_group_class"),
        (_column_discovery_mixin, "test_session_route_equals_the_group_call", "group_clss", "group_cls"),
    ],
)
def test_source_mixins_are_checked_conformance_suites(
    get_mixin: Any, contract_test: str, near_miss: str, hook: str
) -> None:
    mixin = get_mixin()

    with pytest.raises(SwitchedOffError) as switched:
        type("_Blanked", (mixin,), {contract_test: None})

    assert contract_test in str(switched.value)

    with pytest.raises(HookError) as hooked:
        type("_Typo", (mixin,), {near_miss: _Plugin})

    assert _hint(near_miss, hook) in str(hooked.value)


def test_assert_passed_accepts_passing_result() -> None:
    assert_passed(CheckResult(check="c", subject="s"))


def test_assert_passed_fails_with_the_failures() -> None:
    result = CheckResult(check="c", subject="s", failures=("first problem", "second problem"))

    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_passed(result)

    message = str(excinfo.value)
    assert "first problem" in message and "second problem" in message
