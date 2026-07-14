"""Engine-parity tests for resolve_feature() (issue #693, round 2).

The engine's no-match error now points scoped failures at
``resolve_feature(name, options=..., feature_group=...)``. That only holds up if the debug tool
reports the SAME outcome as the engine (IdentifyFeatureGroupClass). These tests pin the four
places where it does not:

1. Subclass filtering. The engine only drops a parent when parent and child support the SAME
   compute frameworks (identify_feature_group.py filter_subclasses). resolve_feature drops the
   parent unconditionally, so a parent/child pair with UNEQUAL framework sets fails in the engine
   ("Multiple feature groups found") while resolve_feature happily returns the child.
2. Abstract bases. The engine skips abstract feature groups (inspect.isabstract) and reports an
   abstract-only error. resolve_feature has no such filter and returns the abstract class.
3. The redefinition-conflict early return ignores the scope: out-of-scope classes leak into
   candidates and the scope callout is missing. The conflict itself must still be reported (the
   engine dedups the whole class universe before scoping, so it raises on an unrelated conflict too).
4. The ambiguity error renders bare class names, so two same-named classes from different modules
   read as ['X', 'X'] exactly when the user scoped BY NAME to disambiguate them.

Probe feature groups and feature names are prefixed 'Scoped693B' / 'scoped_probe_693b_' because every
FeatureGroup subclass in the process is globally visible to matching under pytest-xdist.
"""

from abc import abstractmethod
from collections.abc import Iterator
import linecache
import sys
import textwrap
from typing import Any, Optional, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_api.scoped693b_dup_name_module import (
    Scoped693BDupNameProbe as OtherModuleDupNameProbe,
)


UNEQUAL_FRAMEWORKS_FEATURE = "scoped_probe_693b_unequal_frameworks"
EQUAL_FRAMEWORKS_FEATURE = "scoped_probe_693b_equal_frameworks"
ABSTRACT_ONLY_FEATURE = "scoped_probe_693b_abstract_only"
ABSTRACT_WITH_CONCRETE_FEATURE = "scoped_probe_693b_abstract_with_concrete"
CONFLICT_FEATURE = "scoped_probe_693b_conflict"
DUP_NAME_FEATURE = "scoped_probe_693b_dup_name"

BOTH_FRAMEWORKS: set[type[ComputeFramework]] = {PandasDataFrame, PythonDictFramework}


def _callout(scope_name: str) -> str:
    """The scope callout the engine's scope_callout renders, verbatim."""
    return f"Scoped to feature group: '{scope_name}'."


# --- 1. Unequal framework sets: the engine keeps the parent, resolve_feature drops it ---------


class Scoped693BWideParent(FeatureGroup):
    """Concrete parent declaring TWO frameworks; matches the unequal-frameworks probe name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == UNEQUAL_FRAMEWORKS_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693BNarrowChild(Scoped693BWideParent):
    """Child of the wide parent, narrowed to ONE framework, so the two support sets differ."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}


class Scoped693BEqualParent(FeatureGroup):
    """Concrete parent whose child does not narrow the frameworks: the support sets are EQUAL."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == EQUAL_FRAMEWORKS_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693BEqualChild(Scoped693BEqualParent):
    """Child inheriting the parent's frameworks; the engine drops the parent for this pair."""


# --- 2. Abstract bases ------------------------------------------------------------------------


class Scoped693BAbstractOnlyBase(FeatureGroup):
    """Abstract base (unimplemented abstract hook) that is the ONLY match for its probe name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == ABSTRACT_ONLY_FEATURE

    @classmethod
    @abstractmethod
    def _scoped693b_abstract_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693BAbstractWithConcreteBase(FeatureGroup):
    """Abstract base that DOES have a concrete subclass matching the same probe name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == ABSTRACT_WITH_CONCRETE_FEATURE

    @classmethod
    @abstractmethod
    def _scoped693b_concrete_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693BConcreteMember(Scoped693BAbstractWithConcreteBase):
    """Concrete member of the abstract family; implements the hook, inherits the frameworks."""

    @classmethod
    def _scoped693b_concrete_hook(cls, data: Any) -> Any:
        return data


# --- 3. Redefinition conflict + scope ---------------------------------------------------------


class Scoped693BConflictScopeTarget(FeatureGroup):
    """In-scope feature group for the conflict tests; matches the conflict probe name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == CONFLICT_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


# --- 4. Two same-named classes in different modules --------------------------------------------


class Scoped693BDupNameProbe(FeatureGroup):
    """Same class name as the probe in scoped693b_dup_name_module, different module."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DUP_NAME_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


# --- Jupyter-style redefinition helpers (mirroring tests/test_core/test_prepare/test_feature_group_dedup.py) ---

# Strong refs, as IPython's Out[N] history keeps them: the stale class stays in FeatureGroup.__subclasses__().
_REF_STORE: list[Any] = []


def _exec_fg_in_main(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into __main__, registering its source in linecache."""
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    code_obj = compile(src, filename, "exec")
    exec(code_obj, main_mod.__dict__)  # nosec B102
    return cast(type[FeatureGroup], main_mod.__dict__[class_name])


def _conflicting_pair(label: str) -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Define the same class twice in __main__ with DIFFERENT source: a redefinition conflict.

    Both versions match CONFLICT_FEATURE (via feature_names_supported) and neither is inside the
    Scoped693BConflictScopeTarget scope, which is what makes them the out-of-scope leak under test.

    ``label`` keeps the (module, qualname) dedup key unique per test: the exec'd classes stay alive
    in FeatureGroup.__subclasses__() for the rest of the process, so a shared key would grow the
    conflict group from test to test.
    """
    class_name = f"Scoped693BRedefinedProbe_{label}"
    body = f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_

class {class_name}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{CONFLICT_FEATURE}"}}
"""
    body_v2 = body + "    def _scoped693b_extra(self):\n        return 693\n"

    v1 = _exec_fg_in_main(class_name, body, f"scoped693b-conflict-{label}-v1")
    v2 = _exec_fg_in_main(class_name, body_v2, f"scoped693b-conflict-{label}-v2")
    _REF_STORE.extend([v1, v2])
    return v1, v2


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


@pytest.fixture(autouse=True)
def cleanup_main_module_attrs() -> Iterator[None]:
    """Unbind exec'd classes from __main__ after each test, so the conflict does not outlive it."""
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    for key in set(main_mod.__dict__.keys()) - snapshot:
        main_mod.__dict__.pop(key, None)


def _run_all_scoped(feature: Feature, enabled: set[type[FeatureGroup]]) -> None:
    """Run the engine for one feature with only the probe feature groups enabled."""
    list(
        mloda.run_all(
            [feature],
            compute_frameworks=BOTH_FRAMEWORKS,
            plugin_collector=PluginCollector.enabled_feature_groups(enabled),
        )
    )


class TestUnequalFrameworkSubclassFilterParity:
    """The engine keeps a parent whose framework set differs from its child's; so must the debug tool."""

    def test_engine_reports_multiple_feature_groups_for_the_unequal_pair(self) -> None:
        """Premise: parent {Pandas, PythonDict} and child {Pandas} both survive engine filtering."""
        with pytest.raises(ValueError) as exc_info:
            _run_all_scoped(
                Feature(UNEQUAL_FRAMEWORKS_FEATURE),
                {Scoped693BWideParent, Scoped693BNarrowChild},
            )

        assert "Multiple feature groups found" in str(exc_info.value)

    def test_engine_reports_multiple_feature_groups_when_scoped_to_the_parent(self) -> None:
        """Premise: scoping to the parent keeps both (the child is inside the parent's scope)."""
        with pytest.raises(ValueError) as exc_info:
            _run_all_scoped(
                Feature(UNEQUAL_FRAMEWORKS_FEATURE, feature_group=Scoped693BWideParent),
                {Scoped693BWideParent, Scoped693BNarrowChild},
            )

        engine_message = str(exc_info.value)
        assert "Multiple feature groups found" in engine_message
        assert _callout("Scoped693BWideParent") in engine_message

    def test_resolve_feature_unscoped_reports_the_same_ambiguity(self) -> None:
        """resolve_feature must not silently narrow to the child when the framework sets differ."""
        result = resolve_feature(UNEQUAL_FRAMEWORKS_FEATURE)

        assert result.feature_group is None, (
            "resolve_feature must mirror the engine: the parent's framework set differs from the "
            f"child's, so neither wins; got {result.feature_group}"
        )
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error

    def test_resolve_feature_scoped_reports_the_same_ambiguity_with_callout(self) -> None:
        """Scoped to the parent, resolve_feature must reproduce the engine's ambiguity failure."""
        result = resolve_feature(UNEQUAL_FRAMEWORKS_FEATURE, feature_group=Scoped693BWideParent)

        assert result.feature_group is None, (
            f"a scoped request that fails in the engine must fail here too; got {result.feature_group}"
        )
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error
        assert _callout("Scoped693BWideParent") in result.error

    def test_engine_prefers_the_child_when_framework_sets_are_equal(self) -> None:
        """Premise for the regression guard: equal framework sets, so the engine drops the parent."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            Scoped693BEqualParent: set(BOTH_FRAMEWORKS),
            Scoped693BEqualChild: set(BOTH_FRAMEWORKS),
        }

        identifier = IdentifyFeatureGroupClass(
            feature=Feature(EQUAL_FRAMEWORKS_FEATURE),
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
        resolved, _frameworks = identifier.get()

        assert resolved is Scoped693BEqualChild

    def test_resolve_feature_still_prefers_the_child_when_framework_sets_are_equal(self) -> None:
        """Regression guard: equal framework sets keep today's subclass-preferring behavior."""
        result = resolve_feature(EQUAL_FRAMEWORKS_FEATURE)

        assert result.feature_group is Scoped693BEqualChild
        assert result.error is None

    def test_scoped_resolve_still_prefers_the_child_when_framework_sets_are_equal(self) -> None:
        """Regression guard: the same holds when scoped to the parent of an equal-set pair."""
        result = resolve_feature(EQUAL_FRAMEWORKS_FEATURE, feature_group=Scoped693BEqualParent)

        assert result.feature_group is Scoped693BEqualChild
        assert result.error is None


class TestAbstractBaseScopeParity:
    """An abstract base cannot be instantiated: the engine never lets it win, nor may resolve_feature."""

    def test_engine_rejects_the_abstract_only_scope(self) -> None:
        """Premise: scoping to an abstract-only base fails in the engine with an abstract callout."""
        with pytest.raises(ValueError) as exc_info:
            _run_all_scoped(
                Feature(ABSTRACT_ONLY_FEATURE, feature_group=Scoped693BAbstractOnlyBase),
                {Scoped693BAbstractOnlyBase},
            )

        engine_message = str(exc_info.value)
        assert "abstract" in engine_message.lower()
        assert _callout("Scoped693BAbstractOnlyBase") in engine_message

    def test_resolve_feature_unscoped_does_not_resolve_an_abstract_base(self) -> None:
        """An abstract base is not a usable answer: report it, do not return it."""
        result = resolve_feature(ABSTRACT_ONLY_FEATURE)

        assert result.feature_group is None, (
            f"an abstract base can never be instantiated, so it must not resolve; got {result.feature_group}"
        )
        assert result.error is not None
        assert "abstract" in result.error.lower(), (
            f"the error must say only abstract base(s) matched, got: {result.error}"
        )

    def test_resolve_feature_scoped_to_an_abstract_base_reports_it_with_the_callout(self) -> None:
        """The scoped failure carries both the abstract reason and the scope callout."""
        result = resolve_feature(ABSTRACT_ONLY_FEATURE, feature_group=Scoped693BAbstractOnlyBase)

        assert result.feature_group is None
        assert result.error is not None
        assert "abstract" in result.error.lower(), (
            f"the error must say only abstract base(s) matched, got: {result.error}"
        )
        assert _callout("Scoped693BAbstractOnlyBase") in result.error

    def test_abstract_base_with_a_concrete_subclass_still_resolves_to_the_subclass(self) -> None:
        """Guard: when a concrete subclass also matches, it wins and there is no error."""
        result = resolve_feature(ABSTRACT_WITH_CONCRETE_FEATURE)

        assert result.feature_group is Scoped693BConcreteMember
        assert result.error is None

    def test_scoped_abstract_base_with_a_concrete_subclass_resolves_to_the_subclass(self) -> None:
        """Guard: scoping to the abstract family resolves to its concrete member, as before."""
        result = resolve_feature(ABSTRACT_WITH_CONCRETE_FEATURE, feature_group=Scoped693BAbstractWithConcreteBase)

        assert result.feature_group is Scoped693BConcreteMember
        assert result.error is None


class TestRedefinitionConflictRespectsTheScope:
    """The conflict is still reported when scoped, but candidates and the error become scope-aware."""

    def test_unscoped_conflict_reports_the_conflicting_classes(self) -> None:
        """Premise: the redefinition conflict surfaces as an error and both versions are candidates."""
        v1, v2 = _conflicting_pair("unscoped")

        result = resolve_feature(CONFLICT_FEATURE)

        assert result.feature_group is None
        assert result.error is not None
        assert "redefined" in result.error
        assert v1 in result.candidates
        assert v2 in result.candidates

    def test_scoped_conflict_still_reports_the_redefinition(self) -> None:
        """The engine dedups the whole class universe before scoping, so the conflict still fires."""
        _conflicting_pair("still_reported")

        result = resolve_feature(CONFLICT_FEATURE, feature_group=Scoped693BConflictScopeTarget)

        assert result.feature_group is None
        assert result.error is not None
        assert "redefined" in result.error, f"an out-of-scope conflict must still be reported, got: {result.error}"

    def test_scoped_conflict_error_carries_the_scope_callout(self) -> None:
        """The conflict error must be legible as a SCOPED failure, like every other scoped error."""
        _conflicting_pair("callout")

        result = resolve_feature(CONFLICT_FEATURE, feature_group=Scoped693BConflictScopeTarget)

        assert result.error is not None
        assert _callout("Scoped693BConflictScopeTarget") in result.error, (
            f"the scoped conflict error must carry the scope callout, got: {result.error}"
        )

    def test_scoped_conflict_candidates_exclude_out_of_scope_classes(self) -> None:
        """candidates is 'matching candidates INSIDE the scope', so the conflicting classes drop out."""
        v1, v2 = _conflicting_pair("candidates")

        result = resolve_feature(CONFLICT_FEATURE, feature_group=Scoped693BConflictScopeTarget)

        assert v1 not in result.candidates, (
            f"an out-of-scope conflicting class must not leak into candidates, got: {result.candidates}"
        )
        assert v2 not in result.candidates, (
            f"an out-of-scope conflicting class must not leak into candidates, got: {result.candidates}"
        )


class TestSameNamedClassesAreDisambiguated:
    """Scoping BY NAME is how a user disambiguates; the resulting error must distinguish the twins."""

    def test_engine_ambiguity_error_distinguishes_the_two_modules(self) -> None:
        """Premise: the engine renders name + module (format_feature_group_class)."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            Scoped693BDupNameProbe: {PandasDataFrame},
            OtherModuleDupNameProbe: {PandasDataFrame},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=Feature(DUP_NAME_FEATURE, feature_group="Scoped693BDupNameProbe"),
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        engine_message = str(exc_info.value)
        assert format_feature_group_class(Scoped693BDupNameProbe) in engine_message
        assert format_feature_group_class(OtherModuleDupNameProbe) in engine_message

    def test_resolve_feature_ambiguity_error_distinguishes_the_two_modules(self) -> None:
        """['Scoped693BDupNameProbe', 'Scoped693BDupNameProbe'] is useless: render the modules too."""
        result = resolve_feature(DUP_NAME_FEATURE, feature_group="Scoped693BDupNameProbe")

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error
        assert format_feature_group_class(Scoped693BDupNameProbe) in result.error, (
            f"the ambiguity error must identify the class by name AND module, got: {result.error}"
        )
        assert format_feature_group_class(OtherModuleDupNameProbe) in result.error, (
            f"the ambiguity error must identify the class by name AND module, got: {result.error}"
        )

    def test_resolve_feature_ambiguity_keeps_both_twins_as_candidates(self) -> None:
        """Both same-named classes are genuine candidates inside the name scope."""
        result = resolve_feature(DUP_NAME_FEATURE, feature_group="Scoped693BDupNameProbe")

        assert Scoped693BDupNameProbe in result.candidates
        assert OtherModuleDupNameProbe in result.candidates
