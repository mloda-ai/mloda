"""ConformanceSuite hooks are validated at class creation: near-miss names on a test class are errors."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, ClassVar

import pytest

import mloda._testing
from mloda._testing.result import CheckResult
from mloda._testing.declarations import Cause
from mloda._testing.suite import ConformanceSuite, HookError, SwitchedOffError, assert_passed, check_hooks
from mloda.user.python_dict import PythonDictFramework


pytest_plugins = ["pytester"]


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


def _read_file_mixin() -> type:
    from mloda._testing.source.read_file_suite import ReadFileFGTestMixin

    return ReadFileFGTestMixin


def _read_db_mixin() -> type:
    from mloda._testing.source.read_db_suite import ReadDBFGTestMixin

    return ReadDBFGTestMixin


def _read_document_mixin() -> type:
    from mloda._testing.source.read_document_suite import ReadDocumentFGTestMixin

    return ReadDocumentFGTestMixin


@pytest.mark.parametrize(
    ("get_mixin", "contract_test", "near_miss", "hook"),
    [
        (_source_mixin, "test_claims_its_own_source", "feature_group_clas", "feature_group_class"),
        (_column_discovery_mixin, "test_session_route_equals_the_group_call", "group_clss", "group_cls"),
        (_read_file_mixin, "test_file_claims_via_a_folder_entry", "write_fiel", "write_file"),
        (_read_db_mixin, "test_db_handle_names_one_credential", "credential_kee", "credential_key"),
        (_read_document_mixin, "test_doc_owns_exactly_the_expected_suffixes", "expected_sufixes", "expected_suffixes"),
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


def test_no_testing_module_imports_a_plugin_feature_group() -> None:
    offenders = []
    for path in Path(mloda._testing.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.startswith("mloda_plugins.feature_group")]

    assert offenders == []


def test_assert_passed_accepts_passing_result() -> None:
    assert_passed(CheckResult(check="c", subject="s"))


def test_assert_passed_fails_with_the_failures() -> None:
    result = CheckResult(check="c", subject="s", failures=("first problem", "second problem"))

    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_passed(result)

    message = str(excinfo.value)
    assert "first problem" in message and "second problem" in message


_WIRING_MODULE = """
from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.framework_adapter import PythonDictAdapter
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class NoFilterFramework(ComputeFramework):
    pass


class NoFilterAdapter(PythonDictAdapter):
    def compute_framework(self):
        return NoFilterFramework


class TestNoFilter(FilterEngineSuite):
    adapter_class = NoFilterAdapter
"""

_LIFECYCLE_MODULE = """
from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.framework_adapter import PythonDictAdapter

INSTANCES = []
SEEN = []


class RecordingAdapter(PythonDictAdapter):
    def __init__(self):
        self.opened = 0
        self.closed = False
        INSTANCES.append(self)

    def open(self):
        self.opened += 1

    def close(self):
        self.closed = True


class TestRecording(FilterEngineSuite):
    adapter_class = RecordingAdapter

    def test_first_adapter_is_open(self, adapter):
        assert adapter.opened == 1 and not adapter.closed
        SEEN.append(adapter)

    def test_second_adapter_is_a_new_open_one(self, adapter):
        assert adapter.opened == 1 and not adapter.closed
        assert all(adapter is not earlier for earlier in SEEN)
        SEEN.append(adapter)


def test_every_adapter_was_closed_and_each_was_opened_once():
    assert len(SEEN) == 2
    assert len(INSTANCES) > 2
    assert [a for a in INSTANCES if not a.closed] == []
    assert [a for a in INSTANCES if a.opened != 1] == []
"""


def test_compute_framework_host_without_adapter_class_fails_at_class_creation() -> None:
    from mloda._testing.compute_framework.filter_suite import FilterEngineSuite

    with pytest.raises(HookError) as excinfo:

        class _NoAdapter(FilterEngineSuite):
            pass

    assert "_NoAdapter" in str(excinfo.value)
    assert "adapter_class" in str(excinfo.value)


def test_compute_framework_host_missing_a_required_hook_fails_naming_it() -> None:
    from mloda._testing.compute_framework.host import ComputeFrameworkSuite
    from mloda._testing.framework_adapter import PythonDictAdapter

    class _NeedsHook(ComputeFrameworkSuite, suite_base=True):
        aspect = "filter"
        required_hooks = ("evaluate_rows",)

        def evaluate_rows(self) -> list[int]:
            raise NotImplementedError

    with pytest.raises(HookError) as excinfo:

        class _Missing(_NeedsHook):
            adapter_class = PythonDictAdapter

    assert "_Missing" in str(excinfo.value)
    assert "evaluate_rows" in str(excinfo.value)

    class _Provided(_NeedsHook):
        adapter_class = PythonDictAdapter

        def evaluate_rows(self) -> list[int]:
            return [1]

    assert _Provided.adapter_class is PythonDictAdapter


_RELATION_WIRING_MODULE = """
from mloda._testing.compute_framework.relation_suite import SqlRelationWindowSuite
from mloda._testing.framework_adapter import PythonDictAdapter


class TestNoRelation(SqlRelationWindowSuite):
    adapter_class = PythonDictAdapter
"""


_MERGE_WIRING_MODULE = """
from mloda._testing.compute_framework.merge_suite import MergeEngineSuite
from mloda._testing.framework_adapter import PythonDictAdapter
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class NoMergeFramework(ComputeFramework):
    @classmethod
    def merge_engine(cls):
        raise NotImplementedError


class NoMergeAdapter(PythonDictAdapter):
    def compute_framework(self):
        return NoMergeFramework


class TestNoMerge(MergeEngineSuite):
    adapter_class = NoMergeAdapter
"""


_ASOF_WIRING_MODULE = """
from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite
from mloda._testing.framework_adapter import PythonDictAdapter
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import BaseMergeEngine


class NoAsofFramework(ComputeFramework):
    @classmethod
    def merge_engine(cls):
        return BaseMergeEngine


class NoAsofAdapter(PythonDictAdapter):
    def compute_framework(self):
        return NoAsofFramework


class TestNoAsof(AsofMergeEngineSuite):
    adapter_class = NoAsofAdapter
"""


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("module", "select", "framework", "aspect"),
    [
        (_WIRING_MODULE, [], "NoFilterFramework", "filter"),
        (_RELATION_WIRING_MODULE, ["-k", "test_from_dict_basic or test_len"], "PythonDictFramework", "sql_relation"),
        (_MERGE_WIRING_MODULE, [], "NoMergeFramework", "merge"),
        (_ASOF_WIRING_MODULE, [], "NoAsofFramework", "asof merge"),
    ],
    ids=["filter", "relation", "merge", "asof"],
)
def test_framework_without_the_aspect_fails_with_the_wiring_message(
    pytester: pytest.Pytester, module: str, select: list[str], framework: str, aspect: str
) -> None:
    pytester.makepyfile(test_wiring=module)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", *select)

    outcomes = result.parseoutcomes()
    assert outcomes.get("errors", 0) > 0
    assert outcomes.get("passed", 0) == 0
    failed_lines = [line for line in result.stdout.str().splitlines() if line.startswith("E ") and "Failed:" in line]
    assert failed_lines
    assert all(framework in line and aspect in line for line in failed_lines)


def test_adapter_is_fresh_per_test_and_closed_afterwards(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_lifecycle=_LIFECYCLE_MODULE)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly")

    assert result.parseoutcomes().get("failed", 0) == 0
    assert result.ret == 0


def test_make_merge_link_uses_feature_group_itself_and_carries_the_settings() -> None:
    from mloda._testing.compute_framework.merge_suite import make_merge_link
    from mloda.core.abstract_plugins.components.index.index import Index
    from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinType
    from mloda.provider import FeatureGroup

    left, right = Index(("k", "a")), Index(("k", "b"))
    config = AsOfJoinConfig("lt", "rt", direction="forward")

    link = make_merge_link(JoinType.ASOF, left, right, config)

    assert link.left_feature_group is FeatureGroup
    assert link.right_feature_group is FeatureGroup
    assert link.jointype == JoinType.ASOF
    assert link.left_index == left
    assert link.right_index == right
    assert link.asof_config is config
    assert make_merge_link(JoinType.INNER, left, right).asof_config is None


def test_unsupported_cases_are_validated_at_class_creation() -> None:
    from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite
    from mloda._testing.framework_adapter import PandasAdapter

    with pytest.raises(SwitchedOffError, match="'nope' is not a case"):

        class _UnknownCase(DataTypeValidatorSuite):
            adapter_class = PandasAdapter
            unsupported_cases = {"nope": (Cause.BACKEND, "reason")}

    with pytest.raises(SwitchedOffError, match=r"must be \(Cause, reason\)"):

        class _BadCause(DataTypeValidatorSuite):
            adapter_class = PandasAdapter
            unsupported_cases = {"int8_col": ("backend", "reason")}  # type: ignore[dict-item]
