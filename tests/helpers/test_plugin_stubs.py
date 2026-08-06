"""Contract for the shared FeatureGroup stub helper (tests/helpers/plugin_stubs.py)."""

from __future__ import annotations

import gc
import inspect
import weakref
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import _safe_version
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from tests.helpers.plugin_stubs import (
    CountingStubFeatureGroup,
    HookCounter,
    StubFeatureGroup,
    StubHookError,
    make_fg,
    make_raising_fg,
)


HELPER_MODULE = "tests.helpers.plugin_stubs"

# Feature names the stubs match on; the plugin_stub_ prefix keeps them unique to this module.
ALPHA = "plugin_stub_alpha_feature"
BETA = "plugin_stub_beta_feature"
UNMATCHED = "plugin_stub_unmatched_feature"

DOMAIN = "plugin_stub_domain"
PARENT_DOMAIN = "plugin_stub_parent_domain"
OVERRIDE_DOMAIN = "plugin_stub_override_domain"

DOC = "A stub minted for the plugin-stub contract."

# Every make_fg call below needs its own class name: the duplicate-name guard is per (module, name)
# and outlives the class it minted.
DUPLICATE_NAME = "PluginStubDuplicateFG"

FRAMEWORKS: set[type[ComputeFramework]] = {PythonDictFramework, PyArrowTable}

FRAMEWORK_NAME = PythonDictFramework.get_class_name()
OTHER_FRAMEWORK_NAME = PyArrowTable.get_class_name()

INDEX = Index(("plugin_stub_index_column",))

# Every provider-overridable hook a counting stub tallies, in the order _call_every_counted_hook calls them.
COUNTED_HOOKS = (
    "match_feature_group_criteria",
    "get_domain",
    "compute_framework_rule",
    "supports_compute_framework",
    "index_columns",
    "supports_index",
    "feature_names_supported",
    "prefix",
)

# The make_fg keywords only a CountingStubFeatureGroup base reads.
COUNTING_KEYWORDS = ("counter", "supported_frameworks", "index_columns", "supports_index", "raising_hooks")

COUNTED_CLASS = "PluginStubTalliedClassName"
COUNTED_HOOK = "get_domain"
OTHER_HOOK = "prefix"


def _matches(cls: type[FeatureGroup], feature_name: str) -> bool:
    return cls.match_feature_group_criteria(feature_name, Options())


def _as_stub(cls: type[FeatureGroup]) -> type[StubFeatureGroup]:
    """Narrow the factory's declared return type, so the stub ClassVars are readable."""
    assert issubclass(cls, StubFeatureGroup), f"make_fg must mint a StubFeatureGroup subclass, got {cls.__mro__}"
    return cls


def _mint_and_drop(name: str) -> weakref.ref[type[FeatureGroup]]:
    """Hand back only a weak reference: this frame keeps no strong one."""
    return weakref.ref(make_fg(name))


def _collect_until_dead(reference: weakref.ref[type[FeatureGroup]]) -> None:
    for generation in (0, 1, 2):
        gc.collect(generation)
        if reference() is None:
            return


def _ordinary_spelling_verdicts() -> tuple[bool, bool]:
    """Verdicts of a stub written as an ordinary subclass; the class never leaves this frame."""

    class OrdinarySpellingStubFG(StubFeatureGroup):
        MATCHED_NAMES = frozenset({ALPHA})

    return _matches(OrdinarySpellingStubFG, ALPHA), _matches(OrdinarySpellingStubFG, UNMATCHED)


def _as_counting(cls: type[FeatureGroup]) -> type[CountingStubFeatureGroup]:
    """Narrow the factory's declared return type, so the counting ClassVars are readable."""
    assert issubclass(cls, CountingStubFeatureGroup), f"expected a counting stub, got {cls.__mro__}"
    return cls


def _call_every_counted_hook(cls: type[FeatureGroup]) -> None:
    """Call each counted hook exactly once, in COUNTED_HOOKS order."""
    cls.match_feature_group_criteria(ALPHA, Options())
    cls.get_domain()
    cls.compute_framework_rule()
    cls.supports_compute_framework(ALPHA, Options(), PythonDictFramework)
    cls.index_columns()
    cls.supports_index(INDEX)
    cls.feature_names_supported()
    cls.prefix()


def _counting_keyword_value(keyword: str) -> Any:
    """One representative value per counting-only make_fg keyword, built on demand."""
    values: dict[str, Any] = {
        "counter": HookCounter(),
        "supported_frameworks": FRAMEWORK_NAME,
        "index_columns": [INDEX],
        "supports_index": True,
        "raising_hooks": COUNTED_HOOK,
    }
    return values[keyword]


def _mint_raising_and_drop(name: str) -> weakref.ref[type[FeatureGroup]]:
    """Hand back only a weak reference: this frame keeps no strong one."""
    return weakref.ref(make_raising_fg(name, COUNTED_HOOK))


ARMED_STUBS: list[type[CountingStubFeatureGroup]] = []


def _armed(cls: type[FeatureGroup]) -> type[CountingStubFeatureGroup]:
    """Track a stub with an armed hook, so the fixture can disarm it after the test."""
    stub = _as_counting(cls)
    ARMED_STUBS.append(stub)
    return stub


@pytest.fixture(autouse=True)
def disarm_stubs_after_each_test() -> Iterator[None]:
    """An armed stub that outlives its test stays globally visible, so disarm every one a test built."""
    yield
    for stub in ARMED_STUBS:
        stub.ARMED = False
    ARMED_STUBS.clear()


class TestMintedIdentity:
    """A minted class carries the caller's identity, never the helper module's."""

    def test_the_name_is_a_required_parameter(self) -> None:
        """An implicit name would let two unrelated tests collide in the global subclass registry."""
        name_parameter = inspect.signature(make_fg).parameters["name"]
        assert name_parameter.default is inspect.Parameter.empty

    def test_the_minted_class_carries_the_given_name(self) -> None:
        name = "PluginStubIdentityFG"
        cls = make_fg(name)
        assert cls.__name__ == name
        assert cls.__qualname__ == name
        assert cls.get_class_name() == name

    def test_the_minted_class_belongs_to_the_calling_module(self) -> None:
        """(module, qualname) dedup and _is_live_in_module must read a stub like a hand-written class."""
        cls = make_fg("PluginStubCallingModuleFG")
        assert cls.__module__ == __name__
        assert cls.__module__ != HELPER_MODULE


class TestRegistrySafety:
    """Stubs are global FeatureGroup subclasses, so the factory must neither collide nor retain."""

    def test_a_repeated_name_in_the_same_module_raises(self) -> None:
        """A duplicate (module, name) is exactly the cross-contamination the isolation guard exists for."""
        make_fg(DUPLICATE_NAME)
        with pytest.raises(ValueError, match=DUPLICATE_NAME):
            make_fg(DUPLICATE_NAME)

    def test_the_factory_keeps_no_strong_reference(self) -> None:
        """A cached class object would leak every stub in the suite past its own test."""
        reference = _mint_and_drop("PluginStubWeakrefFG")
        _collect_until_dead(reference)
        assert reference() is None, "make_fg must not retain the classes it mints"

    def test_distinct_calls_mint_distinct_classes(self) -> None:
        first = make_fg("PluginStubDistinctFirstFG")
        second = make_fg("PluginStubDistinctSecondFG")
        assert first is not second
        assert not issubclass(first, second)
        assert not issubclass(second, first)


class TestMatching:
    """matches= is the whole matcher: a stub answers True for those names and for nothing else."""

    def test_the_default_stub_matches_nothing(self) -> None:
        """Deliberately unlike FeatureGroup's default matcher, which matches the class name."""
        name = "PluginStubDefaultMatchFG"
        cls = make_fg(name)
        assert _matches(cls, name) is False

    def test_a_string_is_one_name_not_one_per_character(self) -> None:
        cls = make_fg("PluginStubStringMatchFG", matches=ALPHA)
        assert _matches(cls, ALPHA) is True
        assert _matches(cls, ALPHA[0]) is False
        assert _matches(cls, UNMATCHED) is False

    def test_an_iterable_matches_each_of_its_names(self) -> None:
        cls = make_fg("PluginStubIterableMatchFG", matches={ALPHA, BETA})
        assert _as_stub(cls).MATCHED_NAMES == frozenset({ALPHA, BETA})
        assert _matches(cls, ALPHA) is True
        assert _matches(cls, BETA) is True
        assert _matches(cls, UNMATCHED) is False

    def test_the_ordinary_subclass_spelling_matches_identically(self) -> None:
        """Setting MATCHED_NAMES in a class body and passing matches= are one mechanism."""
        verdicts = _ordinary_spelling_verdicts()
        factory = make_fg("PluginStubFactoryTwinFG", matches=ALPHA)
        assert verdicts == (_matches(factory, ALPHA), _matches(factory, UNMATCHED))
        assert verdicts == (True, False)


class TestDeclaredAttributes:
    """Each keyword lands on the FeatureGroup accessor it stands for."""

    def test_no_domain_means_the_default_domain(self) -> None:
        cls = make_fg("PluginStubDefaultDomainFG")
        assert cls.get_domain() == Domain.get_default_domain()

    def test_a_named_domain_becomes_that_domain(self) -> None:
        cls = make_fg("PluginStubNamedDomainFG", domain=DOMAIN)
        assert cls.get_domain() == Domain(DOMAIN)

    def test_no_frameworks_means_no_compute_framework_rule(self) -> None:
        cls = make_fg("PluginStubDefaultFrameworksFG")
        assert cls.compute_framework_rule() is None

    def test_frameworks_become_the_compute_framework_rule(self) -> None:
        cls = make_fg("PluginStubNamedFrameworksFG", frameworks=FRAMEWORKS)
        assert cls.compute_framework_rule() == FRAMEWORKS

    def test_frameworks_are_copied_not_aliased(self) -> None:
        """Mutating the passed set afterwards must not rewrite the minted class's rule."""
        passed: set[type[ComputeFramework]] = {PythonDictFramework}
        cls = make_fg("PluginStubFrameworksCopyFG", frameworks=passed)
        passed.add(PyArrowTable)
        assert cls.compute_framework_rule() == {PythonDictFramework}

    def test_no_supported_names_means_the_empty_set(self) -> None:
        cls = make_fg("PluginStubDefaultSupportedFG")
        assert cls.feature_names_supported() == set()

    def test_supported_names_become_the_supported_feature_names(self) -> None:
        cls = make_fg("PluginStubNamedSupportedFG", supported_names=(ALPHA, BETA))
        assert cls.feature_names_supported() == {ALPHA, BETA}

    def test_doc_becomes_the_class_description(self) -> None:
        cls = make_fg("PluginStubDocFG", doc=DOC)
        assert cls.__doc__ == DOC
        assert cls.description() == DOC


class TestInstances:
    """A stub is a root feature, and instantiable unless it was minted abstract."""

    def test_input_features_returns_none_on_an_instance(self) -> None:
        """None is the root-feature protocol signal that FeatureGroup.is_root reads."""
        instance = make_fg("PluginStubRootFG")()
        assert instance.input_features(Options(), FeatureName(ALPHA)) is None
        assert instance.is_root(Options(), ALPHA) is True

    def test_an_abstract_stub_cannot_be_instantiated(self) -> None:
        cls = make_fg("PluginStubAbstractFG", abstract=True)
        assert inspect.isabstract(cls) is True
        with pytest.raises(TypeError):
            cls()

    def test_a_concrete_stub_is_instantiable(self) -> None:
        cls = make_fg("PluginStubConcreteFG", abstract=False)
        assert inspect.isabstract(cls) is False
        assert isinstance(cls(), cls)


class TestBaseParameter:
    """base= mints a subclass, so a family of stubs can share one set of ClassVars."""

    def test_a_base_stub_is_subclassed_and_its_class_vars_inherited(self) -> None:
        parent = make_fg("PluginStubParentFG", matches=ALPHA, domain=PARENT_DOMAIN, supported_names=(BETA,))
        child = make_fg("PluginStubChildFG", base=parent)
        assert issubclass(child, parent)
        assert _matches(child, ALPHA) is True
        assert child.get_domain() == Domain(PARENT_DOMAIN)
        assert child.feature_names_supported() == {BETA}

    def test_an_explicit_keyword_overrides_the_inherited_class_var(self) -> None:
        parent = make_fg("PluginStubOverrideParentFG", matches=ALPHA, domain=OVERRIDE_DOMAIN)
        child = make_fg("PluginStubOverrideChildFG", base=parent, matches=BETA)
        assert _matches(child, BETA) is True
        assert _matches(child, ALPHA) is False
        assert child.get_domain() == Domain(OVERRIDE_DOMAIN)

    def test_a_base_outside_the_stub_hierarchy_raises(self) -> None:
        """A plain base reads none of the stub ClassVars, so every keyword is silently dropped."""

        class PluginStubPlainFG(FeatureGroup):
            pass

        with pytest.raises(ValueError, match="StubFeatureGroup") as raised:
            make_fg("PluginStubPlainChildFG", base=PluginStubPlainFG, matches=ALPHA)  # type: ignore[arg-type]
        assert "PluginStubPlainFG" in str(raised.value)


class TestEmptyIsAValueNotAnOmission:
    """An omitted keyword inherits from base=; an explicitly empty one declares the value empty."""

    def test_an_omitted_matches_inherits_the_base_names(self) -> None:
        parent = make_fg("PluginStubInheritMatchesParentFG", matches=ALPHA)
        child = make_fg("PluginStubInheritMatchesChildFG", base=parent)
        assert _matches(child, ALPHA) is True

    def test_an_explicitly_empty_matches_clears_the_inherited_names(self) -> None:
        parent = make_fg("PluginStubEmptyMatchesParentFG", matches=(ALPHA, BETA))
        child = make_fg("PluginStubEmptyMatchesChildFG", base=parent, matches=frozenset())
        assert _as_stub(child).MATCHED_NAMES == frozenset()
        assert _matches(child, ALPHA) is False
        assert _matches(child, BETA) is False


class TestStubFeatureGroupBase:
    """The base class alone is inert: it matches nothing and answers the FeatureGroup defaults."""

    def test_it_matches_nothing(self) -> None:
        """Its own name included, so importing the helper adds no matcher to the suite."""
        assert _matches(StubFeatureGroup, StubFeatureGroup.get_class_name()) is False
        assert _matches(StubFeatureGroup, ALPHA) is False

    def test_it_answers_the_feature_group_defaults(self) -> None:
        assert StubFeatureGroup.get_domain() == Domain.get_default_domain()
        assert StubFeatureGroup.compute_framework_rule() is None
        assert StubFeatureGroup.feature_names_supported() == set()

    def test_its_class_vars_are_empty(self) -> None:
        assert StubFeatureGroup.MATCHED_NAMES == frozenset()
        assert StubFeatureGroup.DOMAIN_NAME is None
        assert StubFeatureGroup.FRAMEWORK_RULE is None
        assert StubFeatureGroup.SUPPORTED_NAMES == frozenset()


class TestNoRetrievableSource:
    """A minted class has no source: a stub that needs a real version() must be written as a class statement."""

    def test_source_and_version_are_unavailable(self) -> None:
        cls = make_fg("PluginStubNoSourceFG")
        with pytest.raises(OSError):
            inspect.getsource(cls)
        with pytest.raises(OSError):
            cls.version()

    def test_the_docs_version_accessor_degrades_instead_of_raising(self) -> None:
        """get_feature_group_docs reads the version field through _safe_version, so a stub degrades, not raises."""
        cls = make_fg("PluginStubDocsVersionFG")
        assert _safe_version(cls) == "unavailable"


class TestStubHookError:
    """The single error an armed stub hook raises."""

    def test_it_is_a_runtime_error(self) -> None:
        """Callers catch it the way they would catch a real plugin's failure."""
        assert issubclass(StubHookError, RuntimeError)


class TestHookCounter:
    """The tally the consuming test module owns."""

    def test_a_fresh_counter_is_empty(self) -> None:
        counter = HookCounter()
        assert counter.calls == {}
        assert counter.pairs == {}

    def test_record_counts_one_call_per_class_and_hook(self) -> None:
        counter = HookCounter()
        counter.record(COUNTED_CLASS, COUNTED_HOOK)
        counter.record(COUNTED_CLASS, COUNTED_HOOK)
        counter.record(COUNTED_CLASS, OTHER_HOOK)
        assert counter.calls == {f"{COUNTED_CLASS}.{COUNTED_HOOK}": 2, f"{COUNTED_CLASS}.{OTHER_HOOK}": 1}
        assert counter.pairs == {}

    def test_record_pair_counts_one_call_per_candidate_and_framework(self) -> None:
        counter = HookCounter()
        counter.record_pair(COUNTED_CLASS, FRAMEWORK_NAME)
        counter.record_pair(COUNTED_CLASS, FRAMEWORK_NAME)
        counter.record_pair(COUNTED_CLASS, OTHER_FRAMEWORK_NAME)
        assert counter.pairs == {(COUNTED_CLASS, FRAMEWORK_NAME): 2, (COUNTED_CLASS, OTHER_FRAMEWORK_NAME): 1}
        assert counter.calls == {}

    def test_clear_empties_both_tallies(self) -> None:
        counter = HookCounter()
        counter.record(COUNTED_CLASS, COUNTED_HOOK)
        counter.record_pair(COUNTED_CLASS, FRAMEWORK_NAME)
        counter.clear()
        assert counter.calls == {}
        assert counter.pairs == {}

    def test_clear_empties_the_dicts_in_place(self) -> None:
        """A caller binds counter.calls once at import, so a rebinding clear would leave it reading a stale dict."""
        counter = HookCounter()
        calls = counter.calls
        pairs = counter.pairs
        counter.record(COUNTED_CLASS, COUNTED_HOOK)
        counter.record_pair(COUNTED_CLASS, FRAMEWORK_NAME)
        counter.clear()
        assert calls is counter.calls
        assert pairs is counter.pairs
        assert calls == {}
        assert pairs == {}

    def test_two_counters_do_not_share_state(self) -> None:
        first = HookCounter()
        second = HookCounter()
        first.record(COUNTED_CLASS, COUNTED_HOOK)
        first.record_pair(COUNTED_CLASS, FRAMEWORK_NAME)
        assert second.calls == {}
        assert second.pairs == {}


class TestCountingStubBase:
    """The counting base alone is inert: it matches nothing, counts nowhere and answers the defaults."""

    def test_it_is_a_stub_feature_group(self) -> None:
        assert issubclass(CountingStubFeatureGroup, StubFeatureGroup)

    def test_it_matches_nothing(self) -> None:
        """Its own name included, so importing the helper adds no matcher to the suite."""
        counting = CountingStubFeatureGroup
        assert _matches(counting, counting.get_class_name()) is False
        assert _matches(counting, ALPHA) is False

    def test_its_class_vars_are_the_inert_defaults(self) -> None:
        counting = CountingStubFeatureGroup
        assert counting.COUNTER is None
        assert counting.SUPPORTED_FRAMEWORKS is None
        assert counting.INDEX_COLUMNS is None
        assert counting.SUPPORTS_INDEX_RESULT is None
        assert counting.RAISING_HOOKS == frozenset()
        assert counting.ARMED is True

    def test_an_unset_counter_counts_nowhere_and_still_answers(self) -> None:
        """Other tests' registry sweeps call these hooks on this globally visible class, so it must never raise."""
        counting = CountingStubFeatureGroup
        _call_every_counted_hook(counting)
        assert counting.get_domain() == Domain.get_default_domain()
        assert counting.compute_framework_rule() is None
        assert counting.feature_names_supported() == set()
        assert counting.index_columns() is None
        assert counting.supports_index(INDEX) is None
        assert counting.supports_compute_framework(ALPHA, Options(), PythonDictFramework) is True
        assert counting.prefix() == f"{counting.get_class_name()}_"


class TestCountedHookCalls:
    """A counting stub tallies every hook it answers, keyed by its own class name."""

    def test_every_counted_hook_lands_under_its_class_and_name(self) -> None:
        counter = HookCounter()
        name = "PluginStubCountedHooksFG"
        cls = make_fg(name, base=CountingStubFeatureGroup, counter=counter)
        _call_every_counted_hook(cls)
        assert counter.calls == {f"{name}.{hook}": 1 for hook in COUNTED_HOOKS}

    def test_a_repeated_call_increments_the_same_key(self) -> None:
        counter = HookCounter()
        name = "PluginStubCountedRepeatFG"
        cls = make_fg(name, base=CountingStubFeatureGroup, counter=counter)
        cls.get_domain()
        cls.get_domain()
        assert counter.calls == {f"{name}.get_domain": 2}

    def test_the_answers_are_the_stub_defaults(self) -> None:
        """Counting changes what is observed, never what is answered."""
        counter = HookCounter()
        name = "PluginStubCountedDefaultsFG"
        cls = make_fg(name, base=CountingStubFeatureGroup, counter=counter, matches=ALPHA)
        assert _matches(cls, ALPHA) is True
        assert _matches(cls, UNMATCHED) is False
        assert cls.get_domain() == Domain.get_default_domain()
        assert cls.compute_framework_rule() is None
        assert cls.feature_names_supported() == set()
        assert cls.index_columns() is None
        assert cls.supports_index(INDEX) is None
        assert cls.prefix() == f"{name}_"

    def test_the_existing_keywords_still_declare_their_answers(self) -> None:
        counter = HookCounter()
        name = "PluginStubCountedDeclaredFG"
        cls = make_fg(
            name,
            base=CountingStubFeatureGroup,
            counter=counter,
            domain=DOMAIN,
            frameworks=FRAMEWORKS,
            supported_names=BETA,
        )
        assert cls.get_domain() == Domain(DOMAIN)
        assert cls.compute_framework_rule() == FRAMEWORKS
        assert cls.feature_names_supported() == {BETA}
        assert counter.calls == {
            f"{name}.{hook}": 1 for hook in ("get_domain", "compute_framework_rule", "feature_names_supported")
        }


class TestCountedCapabilityHook:
    """supports_compute_framework answers the declared framework names and records the pair it was asked about."""

    def test_no_declared_frameworks_supports_every_framework(self) -> None:
        cls = make_fg("PluginStubOpenFrameworksFG", base=CountingStubFeatureGroup)
        assert cls.supports_compute_framework(ALPHA, Options(), PythonDictFramework) is True
        assert cls.supports_compute_framework(ALPHA, Options(), PyArrowTable) is True

    def test_declared_frameworks_gate_by_framework_class_name(self) -> None:
        cls = make_fg(
            "PluginStubGatedFrameworksFG",
            base=CountingStubFeatureGroup,
            supported_frameworks=FRAMEWORK_NAME,
        )
        assert cls.supports_compute_framework(ALPHA, Options(), PythonDictFramework) is True
        assert cls.supports_compute_framework(ALPHA, Options(), PyArrowTable) is False

    def test_it_records_both_the_hook_call_and_the_pair(self) -> None:
        """A hook-name tally cannot see one pair being asked twice, which is what the pair tally is for."""
        counter = HookCounter()
        name = "PluginStubPairCountFG"
        cls = make_fg(name, base=CountingStubFeatureGroup, counter=counter)
        cls.supports_compute_framework(ALPHA, Options(), PythonDictFramework)
        cls.supports_compute_framework(ALPHA, Options(), PythonDictFramework)
        cls.supports_compute_framework(ALPHA, Options(), PyArrowTable)
        assert counter.calls == {f"{name}.supports_compute_framework": 3}
        assert counter.pairs == {(name, FRAMEWORK_NAME): 2, (name, OTHER_FRAMEWORK_NAME): 1}


class TestArmedHooks:
    """A hook named in RAISING_HOOKS raises while ARMED, and answers once disarmed."""

    def test_an_armed_hook_raises_a_stub_hook_error_naming_the_hook(self) -> None:
        cls = _armed(make_fg("PluginStubArmedDomainFG", base=CountingStubFeatureGroup, raising_hooks=COUNTED_HOOK))
        with pytest.raises(StubHookError, match=COUNTED_HOOK):
            cls.get_domain()

    def test_the_call_is_counted_before_it_raises(self) -> None:
        """Callers assert on the call count of the very hook that raised."""
        counter = HookCounter()
        name = "PluginStubArmedCountedFG"
        cls = _armed(make_fg(name, base=CountingStubFeatureGroup, counter=counter, raising_hooks=COUNTED_HOOK))
        with pytest.raises(StubHookError):
            cls.get_domain()
        assert counter.calls == {f"{name}.{COUNTED_HOOK}": 1}

    def test_a_hook_that_is_not_listed_is_unaffected(self) -> None:
        counter = HookCounter()
        name = "PluginStubArmedOtherHookFG"
        cls = _armed(make_fg(name, base=CountingStubFeatureGroup, counter=counter, raising_hooks=COUNTED_HOOK))
        assert cls.prefix() == f"{name}_"
        assert counter.calls == {f"{name}.{OTHER_HOOK}": 1}

    def test_every_listed_hook_raises(self) -> None:
        cls = _armed(
            make_fg(
                "PluginStubArmedPairFG",
                base=CountingStubFeatureGroup,
                raising_hooks=(COUNTED_HOOK, OTHER_HOOK),
            )
        )
        with pytest.raises(StubHookError, match=COUNTED_HOOK):
            cls.get_domain()
        with pytest.raises(StubHookError, match=OTHER_HOOK):
            cls.prefix()

    def test_a_disarmed_hook_answers_normally(self) -> None:
        """Disarming is what makes a stub that outlives its own test inert."""
        counter = HookCounter()
        name = "PluginStubDisarmedFG"
        cls = _armed(
            make_fg(
                name,
                base=CountingStubFeatureGroup,
                counter=counter,
                supported_names=BETA,
                raising_hooks="feature_names_supported",
            )
        )
        with pytest.raises(StubHookError):
            cls.feature_names_supported()
        cls.ARMED = False
        assert cls.feature_names_supported() == {BETA}
        assert counter.calls == {f"{name}.feature_names_supported": 2}


class TestCounterReadAtCallTime:
    """A stub reads COUNTER off the ClassVar when the hook runs, not when the class is minted."""

    def test_a_child_counts_into_the_counter_declared_on_its_base(self) -> None:
        counter = HookCounter()
        parent = make_fg("PluginStubInheritedCounterParentFG", base=CountingStubFeatureGroup, counter=counter)
        child_name = "PluginStubInheritedCounterChildFG"
        child = make_fg(child_name, base=parent)
        child.get_domain()
        assert counter.calls == {f"{child_name}.get_domain": 1}

    def test_reassigning_the_bases_counter_redirects_its_children(self) -> None:
        first = HookCounter()
        second = HookCounter()
        parent = _as_counting(
            make_fg("PluginStubRedirectCounterParentFG", base=CountingStubFeatureGroup, counter=first)
        )
        child_name = "PluginStubRedirectCounterChildFG"
        child = make_fg(child_name, base=parent)
        child.get_domain()
        parent.COUNTER = second
        child.get_domain()
        assert first.calls == {f"{child_name}.get_domain": 1}
        assert second.calls == {f"{child_name}.get_domain": 1}


class TestCountingKeywords:
    """Each counting keyword lands on its ClassVar, and only a counting base may receive one."""

    def test_counter_lands_on_the_counter_class_var(self) -> None:
        counter = HookCounter()
        cls = _as_counting(make_fg("PluginStubCounterKeywordFG", base=CountingStubFeatureGroup, counter=counter))
        assert cls.COUNTER is counter

    def test_a_supported_frameworks_string_is_one_name_not_one_per_character(self) -> None:
        cls = _as_counting(
            make_fg(
                "PluginStubFrameworkStringFG",
                base=CountingStubFeatureGroup,
                supported_frameworks=FRAMEWORK_NAME,
            )
        )
        assert cls.SUPPORTED_FRAMEWORKS == frozenset({FRAMEWORK_NAME})

    def test_supported_frameworks_accepts_an_iterable(self) -> None:
        cls = _as_counting(
            make_fg(
                "PluginStubFrameworkIterableFG",
                base=CountingStubFeatureGroup,
                supported_frameworks=(FRAMEWORK_NAME, OTHER_FRAMEWORK_NAME),
            )
        )
        assert cls.SUPPORTED_FRAMEWORKS == frozenset({FRAMEWORK_NAME, OTHER_FRAMEWORK_NAME})

    def test_index_columns_lands_on_the_index_columns_hook(self) -> None:
        cls = make_fg("PluginStubIndexColumnsFG", base=CountingStubFeatureGroup, index_columns=[INDEX])
        assert _as_counting(cls).INDEX_COLUMNS == [INDEX]
        assert cls.index_columns() == [INDEX]

    def test_supports_index_lands_on_the_supports_index_hook(self) -> None:
        cls = make_fg("PluginStubSupportsIndexFG", base=CountingStubFeatureGroup, supports_index=True)
        assert _as_counting(cls).SUPPORTS_INDEX_RESULT is True
        assert cls.supports_index(INDEX) is True

    def test_a_raising_hooks_string_is_one_name_not_one_per_character(self) -> None:
        cls = _armed(
            make_fg(
                "PluginStubRaisingHooksStringFG",
                base=CountingStubFeatureGroup,
                raising_hooks=COUNTED_HOOK,
            )
        )
        assert cls.RAISING_HOOKS == frozenset({COUNTED_HOOK})

    def test_raising_hooks_accepts_an_iterable(self) -> None:
        cls = _armed(
            make_fg(
                "PluginStubRaisingHooksIterableFG",
                base=CountingStubFeatureGroup,
                raising_hooks=(COUNTED_HOOK, OTHER_HOOK),
            )
        )
        assert cls.RAISING_HOOKS == frozenset({COUNTED_HOOK, OTHER_HOOK})

    def test_a_counting_base_without_any_counting_keyword_still_mints(self) -> None:
        name = "PluginStubPlainCountingBaseFG"
        cls = _as_counting(make_fg(name, base=CountingStubFeatureGroup))
        assert cls.COUNTER is None
        assert _matches(cls, name) is False
        assert cls.prefix() == f"{name}_"

    @pytest.mark.parametrize("keyword", COUNTING_KEYWORDS)
    def test_a_non_counting_base_raises_for_every_counting_keyword(self, keyword: str) -> None:
        """A plain stub base reads none of the counting ClassVars, so the keyword would be silently dropped."""
        name = f"PluginStubNonCountingBase{keyword.title().replace('_', '')}FG"
        with pytest.raises(ValueError, match="CountingStubFeatureGroup") as raised:
            make_fg(name, base=StubFeatureGroup, **{keyword: _counting_keyword_value(keyword)})
        assert keyword in str(raised.value)


class TestCountingKeywordsInherit:
    """An omitted counting keyword inherits from base=; an explicitly empty one declares the value empty."""

    def test_every_omitted_counting_keyword_inherits_the_base_declaration(self) -> None:
        counter = HookCounter()
        parent = _armed(
            make_fg(
                "PluginStubInheritCountingParentFG",
                base=CountingStubFeatureGroup,
                counter=counter,
                supported_frameworks=FRAMEWORK_NAME,
                index_columns=[INDEX],
                supports_index=True,
                raising_hooks=COUNTED_HOOK,
            )
        )
        child = _armed(make_fg("PluginStubInheritCountingChildFG", base=parent))
        assert child.COUNTER is counter
        assert child.SUPPORTED_FRAMEWORKS == frozenset({FRAMEWORK_NAME})
        assert child.INDEX_COLUMNS == [INDEX]
        assert child.SUPPORTS_INDEX_RESULT is True
        assert child.RAISING_HOOKS == frozenset({COUNTED_HOOK})

    def test_an_explicitly_empty_supported_frameworks_supports_no_framework(self) -> None:
        parent = make_fg(
            "PluginStubEmptyFrameworksParentFG",
            base=CountingStubFeatureGroup,
            supported_frameworks=FRAMEWORK_NAME,
        )
        child = _as_counting(make_fg("PluginStubEmptyFrameworksChildFG", base=parent, supported_frameworks=frozenset()))
        assert child.SUPPORTED_FRAMEWORKS == frozenset()
        assert child.supports_compute_framework(ALPHA, Options(), PythonDictFramework) is False

    def test_an_explicitly_empty_raising_hooks_disarms_the_inherited_hook(self) -> None:
        parent = _armed(
            make_fg(
                "PluginStubEmptyRaisingParentFG",
                base=CountingStubFeatureGroup,
                raising_hooks=COUNTED_HOOK,
            )
        )
        child = make_fg("PluginStubEmptyRaisingChildFG", base=parent, raising_hooks=frozenset())
        assert child.get_domain() == Domain.get_default_domain()
        with pytest.raises(StubHookError):
            parent.get_domain()

    def test_an_explicitly_false_supports_index_shadows_the_inherited_true(self) -> None:
        parent = make_fg("PluginStubFalseSupportsIndexParentFG", base=CountingStubFeatureGroup, supports_index=True)
        child = make_fg("PluginStubFalseSupportsIndexChildFG", base=parent, supports_index=False)
        assert child.supports_index(INDEX) is False
        assert parent.supports_index(INDEX) is True


class TestTransientRaisingDouble:
    """make_raising_fg stands in for a broken plugin the caller reaps: it claims no name and leaves nothing behind."""

    def test_it_mints_a_subclass_of_the_given_base_under_the_given_name(self) -> None:
        name = "PluginStubTransientNameFG"
        cls = make_raising_fg(name, COUNTED_HOOK)
        assert issubclass(cls, FeatureGroup)
        assert cls.__name__ == name
        assert cls.get_class_name() == name

    def test_a_stub_base_is_honoured(self) -> None:
        """A caller that needs an inert double passes a stub base, which matches nothing."""
        name = "PluginStubTransientStubBaseFG"
        cls = make_raising_fg(name, COUNTED_HOOK, base=StubFeatureGroup)
        assert issubclass(cls, StubFeatureGroup)
        assert _matches(cls, name) is False

    def test_the_minted_class_belongs_to_the_calling_module(self) -> None:
        cls = make_raising_fg("PluginStubTransientModuleFG", COUNTED_HOOK)
        assert cls.__module__ == __name__
        assert cls.__module__ != HELPER_MODULE

    def test_the_named_hook_raises_a_plain_runtime_error(self) -> None:
        """Plain RuntimeError, not StubHookError: it stands in for an arbitrary broken plugin."""
        cls = make_raising_fg("PluginStubTransientRaiseFG", COUNTED_HOOK)
        with pytest.raises(RuntimeError) as raised:
            cls.get_domain()
        assert not isinstance(raised.value, StubHookError)

    def test_the_hook_raises_whatever_arguments_it_is_called_with(self) -> None:
        cls = make_raising_fg("PluginStubTransientArgsFG", "supports_index")
        with pytest.raises(RuntimeError):
            cls.supports_index(INDEX)

    def test_the_same_name_can_be_minted_twice(self) -> None:
        """No (module, name) claim: the caller reaps its class per test instead of binding it at module level."""
        name = "PluginStubTransientTwiceFG"
        first = make_raising_fg(name, COUNTED_HOOK)
        second = make_raising_fg(name, COUNTED_HOOK)
        assert first is not second
        assert first.get_class_name() == name
        assert second.get_class_name() == name

    def test_the_helper_keeps_no_strong_reference(self) -> None:
        reference = _mint_raising_and_drop("PluginStubTransientWeakrefFG")
        _collect_until_dead(reference)
        assert reference() is None, "make_raising_fg must not retain the classes it mints"

    def test_doc_becomes_the_class_description(self) -> None:
        cls = make_raising_fg("PluginStubTransientDocFG", COUNTED_HOOK, doc=DOC)
        assert cls.__doc__ == DOC
        assert cls.description() == DOC

    def test_an_omitted_doc_leaves_the_docstring_unset(self) -> None:
        """A None __doc__ is what makes the docstring-less degrade path reachable for callers."""
        name = "PluginStubTransientNoDocFG"
        cls = make_raising_fg(name, COUNTED_HOOK)
        assert cls.__doc__ is None
        assert cls.description() == name
