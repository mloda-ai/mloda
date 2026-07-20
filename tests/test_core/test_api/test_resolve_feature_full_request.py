"""Failing tests (TDD Red, issue #756) for resolve_feature expressing the full engine-matcher request.

Target contract for resolve_feature (to be implemented in the Green phase):

    def resolve_feature(
        feature: str | Feature,
        *,
        options: Optional[Options] = None,
        plugin_collector: Optional[PluginCollector] = None,
        feature_group: str | type[FeatureGroup] | None = None,
        links: Optional[set[Link]] = None,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> ResolvedFeature: ...

The full request the engine seam ``IdentifyFeatureGroupClass.evaluate`` consumes is
(feature, accessible_plugins, links, data_access_collection). Today resolve_feature hardcodes
``links=None`` and ``data_access_collection=None`` and cannot carry a domain or a compute-framework
pin, so it cannot express that full request. These tests pin the missing pieces:

  1. a ``Feature`` object accepted directly as the first argument;
  2. passing ``options``/``feature_group`` alongside a ``Feature`` is misuse -> ``TypeError``;
  3. a domain carried on the ``Feature`` gates resolution via ``_filter_feature_group_by_domain``;
  4. a compute-framework pin on the ``Feature`` flows to ``_filter_feature_group_by_framework``;
  5. ``links`` are threaded to ``_filter_feature_group_by_links`` (paired engine/debug parity);
  6. a ``data_access_collection`` is threaded so reader/input-data groups resolve (paired parity);
  7. the never-raising contract survives the new params.

Against the current signature ``resolve_feature(feature_name: str, *, options, plugin_collector,
feature_group)`` every test here FAILS for the right reason: the str-only first argument does not
accept a ``Feature``, and ``links`` / ``data_access_collection`` are not parameters at all.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from mloda.user import PluginCollector, PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


PLAIN_FEATURE = "PlainResolve756Feature"
DOMAIN_FEATURE = "DomainResolve756Feature"
FRAMEWORK_FEATURE = "FrameworkPinResolve756Feature"
LINK_FEATURE = "LinkResolve756Feature"
DATA_ACCESS_FEATURE = "DataAccessResolve756Feature"
RAISING_DAC_FEATURE = "RaisingDacResolve756Feature"

DOMAIN_A = "resolve756_domain_a"
DOMAIN_B = "resolve756_domain_b"
LINK_INDEX_COLUMN = "resolve756_link_idx"
UNSUPPORTED_INDEX_COLUMN = "resolve756_unsupported_idx"
EXPECTED_FILE_HANDLE = "resolve756_source.csv"


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


class PlainResolve756FeatureGroup(FeatureGroup):
    """A plain, unambiguously resolvable fixture matching its own feature name."""

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
        return str(feature_name) == PLAIN_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class DomainAResolve756FeatureGroup(FeatureGroup):
    """Matches DOMAIN_FEATURE and lives in DOMAIN_A."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_A)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class DomainBResolve756FeatureGroup(FeatureGroup):
    """Matches the SAME DOMAIN_FEATURE name but lives in DOMAIN_B, so only a domain pin can disambiguate."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_B)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class FrameworkPinResolve756FeatureGroup(FeatureGroup):
    """Declares two available frameworks so a compute-framework pin can select one."""

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
        return str(feature_name) == FRAMEWORK_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class LinkResolve756FeatureGroup(FeatureGroup):
    """Declares a real Index so the seam's link filter actually validates a supplied Link set."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index((LINK_INDEX_COLUMN,))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == LINK_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class DataAccessResolve756FeatureGroup(FeatureGroup):
    """Only matches when a DataAccessCollection carrying the expected file handle is supplied."""

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
        if str(feature_name) != DATA_ACCESS_FEATURE:
            return False
        if data_access_collection is None:
            return False
        return EXPECTED_FILE_HANDLE in set(data_access_collection.files.values())

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RaisingDacResolve756FeatureGroup(FeatureGroup):
    """Reader-style fixture whose match hook raises once a DataAccessCollection is supplied."""

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
        if str(feature_name) != RAISING_DAC_FEATURE:
            return False
        if data_access_collection is not None:
            raise RuntimeError("resolve756 reader-style match hook blew up on a data access collection")
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _supported_link() -> Link:
    """A Link whose indexes the LinkResolve756 group supports (derived from its own index_columns)."""
    return Link.inner_on(LinkResolve756FeatureGroup, LinkResolve756FeatureGroup)


def _unsupported_link() -> Link:
    """A Link whose indexes the LinkResolve756 group does NOT support."""
    spec = JoinSpec(LinkResolve756FeatureGroup, UNSUPPORTED_INDEX_COLUMN)
    return Link.inner(spec, spec)


class TestResolveFeatureAcceptsFeatureObject:
    """resolve_feature accepts a Feature object as its first argument (issue #756)."""

    def test_feature_object_resolves_like_the_string_form(self) -> None:
        """resolve_feature(Feature("Name")) resolves identically to resolve_feature("Name")."""
        collector = PluginCollector.enabled_feature_groups({PlainResolve756FeatureGroup})

        from_string = resolve_feature(PLAIN_FEATURE, plugin_collector=collector)
        from_feature = resolve_feature(Feature(PLAIN_FEATURE), plugin_collector=collector)

        assert from_string.feature_group is PlainResolve756FeatureGroup
        assert from_feature.feature_group is PlainResolve756FeatureGroup
        assert from_feature.feature_name == PLAIN_FEATURE
        assert from_feature.error is None
        assert from_feature.feature_group is from_string.feature_group


class TestResolveFeatureFeatureObjectMisuse:
    """Passing options/feature_group alongside a Feature is programmer misuse: single source of truth."""

    def test_feature_object_with_options_raises_type_error(self) -> None:
        """resolve_feature(Feature(...), options=...) must raise TypeError, not silently ignore the Feature."""
        resolve_feature_untyped: Any = resolve_feature

        with pytest.raises(TypeError):
            resolve_feature_untyped(Feature(PLAIN_FEATURE), options=Options())

    def test_feature_object_with_feature_group_raises_type_error(self) -> None:
        """resolve_feature(Feature(...), feature_group=...) must raise TypeError for the same reason."""
        resolve_feature_untyped: Any = resolve_feature

        with pytest.raises(TypeError):
            resolve_feature_untyped(Feature(PLAIN_FEATURE), feature_group="PlainResolve756FeatureGroup")


class TestResolveFeatureDomainViaFeature:
    """A domain carried on the Feature gates resolution through _filter_feature_group_by_domain (#756)."""

    def test_unscoped_name_is_ambiguous_across_domains(self) -> None:
        """Premise: without a domain pin both same-named groups match, so resolution is ambiguous.

        This proves the only differentiator between the two fixtures is their domain, hence the domain
        filter is what any successful pin must be gating on.
        """
        collector = PluginCollector.enabled_feature_groups(
            {DomainAResolve756FeatureGroup, DomainBResolve756FeatureGroup}
        )

        result = resolve_feature(Feature(DOMAIN_FEATURE), plugin_collector=collector)

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple feature groups found" in result.error

    def test_matching_domain_pin_resolves_exactly_one(self) -> None:
        """A Feature pinned to DOMAIN_A disambiguates to the DOMAIN_A group only."""
        collector = PluginCollector.enabled_feature_groups(
            {DomainAResolve756FeatureGroup, DomainBResolve756FeatureGroup}
        )

        result = resolve_feature(Feature(DOMAIN_FEATURE, domain=DOMAIN_A), plugin_collector=collector)

        assert result.feature_group is DomainAResolve756FeatureGroup
        assert result.error is None
        assert DomainBResolve756FeatureGroup not in result.candidates

    def test_non_matching_domain_pin_fails_closed(self) -> None:
        """A Feature pinned to a domain no group carries fails to resolve."""
        collector = PluginCollector.enabled_feature_groups(
            {DomainAResolve756FeatureGroup, DomainBResolve756FeatureGroup}
        )

        result = resolve_feature(
            Feature(DOMAIN_FEATURE, domain="resolve756_nonexistent_domain"), plugin_collector=collector
        )

        assert result.feature_group is None
        assert result.error is not None
        # When a Feature is passed, feature_name is str(feature.name); this fails today because the
        # str-only first arg does not accept a Feature and cannot carry the domain pin.
        assert result.feature_name == DOMAIN_FEATURE


class TestResolveFeatureFrameworkPinViaFeature:
    """A compute-framework pin on the Feature flows to _filter_feature_group_by_framework (#756)."""

    def test_matching_framework_pin_resolves_and_reports_it(self) -> None:
        """A Feature pinned to PandasDataFrame resolves and reports PandasDataFrame as supported."""
        collector = PluginCollector.enabled_feature_groups({FrameworkPinResolve756FeatureGroup})

        result = resolve_feature(
            Feature(FRAMEWORK_FEATURE, compute_framework="PandasDataFrame"), plugin_collector=collector
        )

        assert result.feature_group is FrameworkPinResolve756FeatureGroup
        assert result.error is None
        assert "PandasDataFrame" in result.supported_compute_frameworks

    def test_undeclared_framework_pin_fails_closed(self) -> None:
        """A Feature pinned to a framework the group does not declare fails to resolve."""
        collector = PluginCollector.enabled_feature_groups({FrameworkPinResolve756FeatureGroup})

        result = resolve_feature(
            Feature(FRAMEWORK_FEATURE, compute_framework="SqliteFramework"), plugin_collector=collector
        )

        assert result.feature_group is None
        assert result.error is not None
        # When a Feature is passed, feature_name is str(feature.name); this fails today because the
        # str-only first arg does not accept a Feature and cannot carry the framework pin.
        assert result.feature_name == FRAMEWORK_FEATURE

    def test_framework_pin_matches_engine_seam(self) -> None:
        """The framework pin flows to the seam: engine and resolve_feature agree on match and no-match."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            FrameworkPinResolve756FeatureGroup: {PandasDataFrame, PythonDictFramework}
        }

        # Declared pin: the engine identifies the group, and resolve_feature resolves it.
        pinned_declared = Feature(FRAMEWORK_FEATURE, compute_framework="PandasDataFrame")
        engine_declared = evaluate_or_raise(
            feature=pinned_declared,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
        assert next(iter(engine_declared.identified)) is FrameworkPinResolve756FeatureGroup

        collector = PluginCollector.enabled_feature_groups({FrameworkPinResolve756FeatureGroup})
        debug_declared = resolve_feature(
            Feature(FRAMEWORK_FEATURE, compute_framework="PandasDataFrame"), plugin_collector=collector
        )
        assert debug_declared.feature_group is FrameworkPinResolve756FeatureGroup

        # Undeclared pin: the engine refuses (ValueError), and resolve_feature fails closed.
        pinned_undeclared = Feature(FRAMEWORK_FEATURE, compute_framework="SqliteFramework")
        with pytest.raises(ValueError):
            evaluate_or_raise(
                feature=pinned_undeclared,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        debug_undeclared = resolve_feature(
            Feature(FRAMEWORK_FEATURE, compute_framework="SqliteFramework"), plugin_collector=collector
        )
        assert debug_undeclared.feature_group is None


class TestResolveFeatureLinksThreaded:
    """links are threaded to the seam's _filter_feature_group_by_links (issue #756)."""

    def test_no_links_is_the_resolving_baseline(self) -> None:
        """With links=None the link filter cannot reject, so the group resolves."""
        collector = PluginCollector.enabled_feature_groups({LinkResolve756FeatureGroup})

        result = resolve_feature(LINK_FEATURE, plugin_collector=collector, links=None)

        assert result.feature_group is LinkResolve756FeatureGroup
        assert result.error is None

    def test_supported_link_resolves(self) -> None:
        """A links set whose index the group supports passes the link filter."""
        collector = PluginCollector.enabled_feature_groups({LinkResolve756FeatureGroup})

        result = resolve_feature(LINK_FEATURE, plugin_collector=collector, links={_supported_link()})

        assert result.feature_group is LinkResolve756FeatureGroup
        assert result.error is None

    def test_unsupported_link_is_rejected(self) -> None:
        """A links set whose indexes the group does not support is rejected by the link filter."""
        collector = PluginCollector.enabled_feature_groups({LinkResolve756FeatureGroup})

        result = resolve_feature(LINK_FEATURE, plugin_collector=collector, links={_unsupported_link()})

        assert result.feature_group is None
        assert result.error is not None

    def test_links_match_engine_seam(self) -> None:
        """Paired parity: engine and resolve_feature agree for both supported and unsupported link sets."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {LinkResolve756FeatureGroup: {PandasDataFrame}}
        collector = PluginCollector.enabled_feature_groups({LinkResolve756FeatureGroup})

        supported = {_supported_link()}
        engine_supported = evaluate_or_raise(
            feature=Feature(LINK_FEATURE),
            accessible_plugins=accessible_plugins,
            links=supported,
            data_access_collection=None,
        )
        assert next(iter(engine_supported.identified)) is LinkResolve756FeatureGroup
        assert resolve_feature(LINK_FEATURE, plugin_collector=collector, links=supported).feature_group is (
            LinkResolve756FeatureGroup
        )

        unsupported = {_unsupported_link()}
        with pytest.raises(ValueError):
            evaluate_or_raise(
                feature=Feature(LINK_FEATURE),
                accessible_plugins=accessible_plugins,
                links=unsupported,
                data_access_collection=None,
            )
        assert resolve_feature(LINK_FEATURE, plugin_collector=collector, links=unsupported).feature_group is None


class TestResolveFeatureDataAccessCollectionThreaded:
    """data_access_collection is threaded so reader / input_data groups can resolve (issue #756)."""

    def test_collection_flips_no_match_into_a_resolution(self) -> None:
        """Without a collection the gated group never matches; supplying one lets it resolve.

        The no-collection premise passes today, but threading the DataAccessCollection into the seam
        (the headline DoD item) does not exist yet, so the with-collection resolution fails.
        """
        collector = PluginCollector.enabled_feature_groups({DataAccessResolve756FeatureGroup})

        # Premise: gated group does not match when no collection is supplied.
        without = resolve_feature(DATA_ACCESS_FEATURE, plugin_collector=collector)
        assert without.feature_group is None
        assert without.error is not None

        # DoD: supplying the collection threads it to the seam and the gated group resolves.
        dac = DataAccessCollection(files={EXPECTED_FILE_HANDLE})
        with_collection = resolve_feature(DATA_ACCESS_FEATURE, plugin_collector=collector, data_access_collection=dac)
        assert with_collection.feature_group is DataAccessResolve756FeatureGroup
        assert with_collection.error is None

    def test_data_access_collection_matches_engine_seam(self) -> None:
        """Paired parity: the engine and resolve_feature agree with and without the collection."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {DataAccessResolve756FeatureGroup: {PandasDataFrame}}
        collector = PluginCollector.enabled_feature_groups({DataAccessResolve756FeatureGroup})
        dac = DataAccessCollection(files={EXPECTED_FILE_HANDLE})

        engine_with = evaluate_or_raise(
            feature=Feature(DATA_ACCESS_FEATURE),
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=dac,
        )
        assert next(iter(engine_with.identified)) is DataAccessResolve756FeatureGroup
        assert (
            resolve_feature(DATA_ACCESS_FEATURE, plugin_collector=collector, data_access_collection=dac).feature_group
            is DataAccessResolve756FeatureGroup
        )

        with pytest.raises(ValueError):
            evaluate_or_raise(
                feature=Feature(DATA_ACCESS_FEATURE),
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )
        assert resolve_feature(DATA_ACCESS_FEATURE, plugin_collector=collector).feature_group is None


class TestResolveFeatureNeverRaisesWithNewParams:
    """The never-raising contract survives the new links/data_access_collection params (issue #756)."""

    def test_raising_match_hook_with_collection_surfaces_as_error(self) -> None:
        """A reader-style match hook that raises on a collection must surface as ResolvedFeature.error."""
        collector = PluginCollector.enabled_feature_groups({RaisingDacResolve756FeatureGroup})
        dac = DataAccessCollection(files={EXPECTED_FILE_HANDLE})

        result = resolve_feature(RAISING_DAC_FEATURE, plugin_collector=collector, data_access_collection=dac)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
