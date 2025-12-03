"""
Minimal reproduction of polymorphic link bug in mloda.

The polymorphic link feature (commit cdcec67) allows defining links with base classes
that match concrete subclasses. Link.matches() works correctly, but the actual data
joining in calculate_feature does not include all columns.

Expected: All input features should be joined into the data dict
Actual: Only some features appear depending on which side uses base classes
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
class TestPolymorphicLinkBug:
    """Demonstrates the polymorphic link bug."""

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

    def test_polymorphic_links_fail(self) -> None:
        """Assembler with base classes in links fails - BUG."""
        feature = Feature(name=AssemblerWithPolymorphicLinks.FEATURE_NAME)

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {AssemblerWithPolymorphicLinks, ConcreteFeatureGroupA, ConcreteFeatureGroupB}
            ),
        )

        result = results[0][0][AssemblerWithPolymorphicLinks.FEATURE_NAME]

        # EXPECTED: "a=value_a, b=value_b" (both features joined)
        # ACTUAL: "a=MISSING_A, b=value_b" (only B is in the data)
        assert result == "a=value_a, b=value_b", f"BUG: Got '{result}' - feature_a missing from joined data"
