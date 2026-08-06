"""Contract for the shared FeatureGroup stub helper (tests/helpers/plugin_stubs.py)."""

from __future__ import annotations

import gc
import inspect
import weakref

import pytest

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from tests.helpers.plugin_stubs import StubFeatureGroup, make_fg


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
