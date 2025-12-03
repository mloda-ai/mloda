"""
Tests for polymorphic link resolution in mloda.

Tests that Links defined with base classes correctly resolve to concrete subclasses
during the data joining phase. Covers:
- Exact class matching (concrete + concrete)
- Symmetric polymorphic matching (base + base -> concrete + concrete)
- Asymmetric polymorphic matching (base + external concrete)
"""

from typing import Any, List, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import JoinSpec, Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature


# =============================================================================
# Base Classes
# =============================================================================
class BaseFeatureGroupA(AbstractFeatureGroup):
    """Base class A - provides index_columns()."""

    FEATURE_NAME = "feature_a"
    ROW_INDEX = "_idx"

    @classmethod
    def index_columns(cls) -> List[Index]:
        return [Index((cls.ROW_INDEX,))]


class BaseFeatureGroupB(AbstractFeatureGroup):
    """Base class B - provides index_columns()."""

    FEATURE_NAME = "feature_b"
    ROW_INDEX = "_idx"

    @classmethod
    def index_columns(cls) -> List[Index]:
        return [Index((cls.ROW_INDEX,))]


# =============================================================================
# Concrete Implementations
# =============================================================================
class ConcreteFeatureGroupA(BaseFeatureGroupA):
    """Concrete implementation of A."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.FEATURE_NAME})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.ROW_INDEX: [0], cls.FEATURE_NAME: ["value_a"]}


class ConcreteFeatureGroupB(BaseFeatureGroupB):
    """Concrete implementation of B."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.FEATURE_NAME})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.ROW_INDEX: [0], cls.FEATURE_NAME: ["value_b"]}


# =============================================================================
# Assembler that joins A and B
# =============================================================================
class AssemblerWithConcreteLinks(AbstractFeatureGroup):
    """Assembler using CONCRETE classes in links - THIS WORKS."""

    FEATURE_NAME = "assembled_concrete"

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Union[FeatureName, str], options: Any, data_access_collection: Any = None
    ) -> bool:
        name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name
        return name == cls.FEATURE_NAME

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        idx = Index((BaseFeatureGroupA.ROW_INDEX,))

        # CONCRETE classes in link
        link = Link.inner(JoinSpec(ConcreteFeatureGroupA, idx), JoinSpec(ConcreteFeatureGroupB, idx))

        return {
            Feature(name=BaseFeatureGroupA.FEATURE_NAME),
            Feature(name=BaseFeatureGroupB.FEATURE_NAME, link=link),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Extract from joined data
        row = data[0] if isinstance(data, list) else data
        a = row.get(BaseFeatureGroupA.FEATURE_NAME, "MISSING_A")
        b = row.get(BaseFeatureGroupB.FEATURE_NAME, "MISSING_B")
        return {cls.FEATURE_NAME: [f"a={a}, b={b}"]}


class AssemblerWithPolymorphicLinks(AbstractFeatureGroup):
    """Assembler using BASE classes in links - THIS FAILS."""

    FEATURE_NAME = "assembled_polymorphic"

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Union[FeatureName, str], options: Any, data_access_collection: Any = None
    ) -> bool:
        name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name
        return name == cls.FEATURE_NAME

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        idx = Index((BaseFeatureGroupA.ROW_INDEX,))

        # BASE classes in link - polymorphic matching should resolve to concrete
        link = Link.inner(JoinSpec(BaseFeatureGroupA, idx), JoinSpec(BaseFeatureGroupB, idx))

        return {
            Feature(name=BaseFeatureGroupA.FEATURE_NAME),
            Feature(name=BaseFeatureGroupB.FEATURE_NAME, link=link),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Extract from joined data
        row = data[0] if isinstance(data, list) else data
        a = row.get(BaseFeatureGroupA.FEATURE_NAME, "MISSING_A")
        b = row.get(BaseFeatureGroupB.FEATURE_NAME, "MISSING_B")
        return {cls.FEATURE_NAME: [f"a={a}, b={b}"]}


# =============================================================================
# Tests
# =============================================================================
class TestPolymorphicLinkResolution:
    """Tests for symmetric polymorphic link resolution (base + base -> concrete + concrete)."""

    def test_link_matches_works(self) -> None:
        """Link.matches() correctly handles polymorphic matching."""
        idx = Index((BaseFeatureGroupA.ROW_INDEX,))

        # Link defined with BASE classes
        link = Link.inner(JoinSpec(BaseFeatureGroupA, idx), JoinSpec(BaseFeatureGroupB, idx))

        # Should match CONCRETE subclasses - THIS WORKS
        assert link.matches(ConcreteFeatureGroupA, ConcreteFeatureGroupB) is True

    def test_concrete_links_work(self) -> None:
        """Assembler with concrete classes in links works correctly."""
        feature = Feature(name=AssemblerWithConcreteLinks.FEATURE_NAME)

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {AssemblerWithConcreteLinks, ConcreteFeatureGroupA, ConcreteFeatureGroupB}
            ),
        )

        result = results[0][0][AssemblerWithConcreteLinks.FEATURE_NAME]
        assert result == "a=value_a, b=value_b", f"Got: {result}"

    def test_polymorphic_links_resolve_correctly(self) -> None:
        """Assembler with base classes in links resolves to concrete classes."""
        feature = Feature(name=AssemblerWithPolymorphicLinks.FEATURE_NAME)

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {AssemblerWithPolymorphicLinks, ConcreteFeatureGroupA, ConcreteFeatureGroupB}
            ),
        )

        result = results[0][0][AssemblerWithPolymorphicLinks.FEATURE_NAME]
        assert result == "a=value_a, b=value_b", f"Got: {result}"


# =============================================================================
# Asymmetric Case: Base Class + External Concrete Class
# =============================================================================
class AssemblerWithMixedLink(AbstractFeatureGroup):
    """Assembler using base class + external concrete class (asymmetric polymorphic matching)."""

    FEATURE_NAME = "assembled_mixed"

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Union[FeatureName, str], options: Any, data_access_collection: Any = None
    ) -> bool:
        name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name
        return name == cls.FEATURE_NAME

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        idx = Index((BaseFeatureGroupA.ROW_INDEX,))

        # Base class + External concrete class (ApiInputDataFeature has no subclass)
        link = Link.inner(
            JoinSpec(BaseFeatureGroupA, idx),  # Base class -> resolves to ConcreteFeatureGroupA
            JoinSpec(ApiInputDataFeature, idx),  # External concrete class (no subclass)
        )

        return {
            Feature(name=BaseFeatureGroupA.FEATURE_NAME),
            Feature(name="user_query", link=link),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Extract from joined data
        row = data[0] if isinstance(data, list) else data
        a = row.get(BaseFeatureGroupA.FEATURE_NAME, "MISSING_A")
        query = row.get("user_query", "MISSING_QUERY")
        return {cls.FEATURE_NAME: [f"a={a}, query={query}"]}


class TestAsymmetricPolymorphicLinkResolution:
    """Tests for asymmetric polymorphic link resolution (base + external concrete)."""

    def test_mixed_polymorphic_link_resolves_correctly(self) -> None:
        """Assembler with base class + external concrete class in link resolves correctly."""
        feature = Feature(name=AssemblerWithMixedLink.FEATURE_NAME)

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            api_data={"UserQuery": {"_idx": [0], "user_query": ["test query"]}},
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {AssemblerWithMixedLink, ConcreteFeatureGroupA, ApiInputDataFeature}
            ),
        )

        result = results[0][0][AssemblerWithMixedLink.FEATURE_NAME]
        assert result == "a=value_a, query=test query", f"Got: {result}"
