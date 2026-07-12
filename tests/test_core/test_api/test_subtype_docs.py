"""Tests for the subtype enumeration API (issue #639).

Contract under test:

9.  ``FeatureGroupInfo`` gains ``subtype_key``, ``subtypes`` (sorted),
    ``subtype_support`` (framework class name -> sorted subtypes) and
    ``subtype_error``. All are defaulted, so existing construction keeps working,
    and ``get_feature_group_docs()`` populates them.
10. ``ResolvedFeature`` gains ``subtype``, populated by ``resolve_feature()``
    from the resolved feature group's ``resolve_subtype(feature_name, Options())``.
11. An ABSTRACT base documents its subtype universe but an EMPTY support matrix.
12. A misdeclared capability is SURFACED, not swallowed: ``subtype_support`` stays
    empty and ``subtype_error`` carries the rejection message. Healthy groups in the
    same call keep ``subtype_error is None``.
"""

from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.api.plugin_docs import get_feature_group_docs, resolve_feature
from mloda.core.api.plugin_info import FeatureGroupInfo, ResolvedFeature
from mloda.provider import FeatureChainParserMixin, FeatureGroup, property_spec
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


DOCS_SUBTYPE_FEATURE = "revenue__median_docstat"

AGGREGATION_SUBTYPES = ["avg", "count", "max", "mean", "median", "min", "std", "sum", "var"]


class DocsSubtypeFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Subtype-carrying family with a bounded framework definition for the docs API."""

    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_docstat$"

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is PythonDictFramework:
            return frozenset({"sum"})
        return cls.subtype_universe()


class DocsNoSubtypeFeatureGroup(FeatureGroup):
    """Family without a subtype dimension: the new fields stay empty."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == cls.get_class_name()

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class DocsAbstractSubtypeFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Abstract base: documents the universe, but declares no per-framework support."""

    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_docsabstract$"

    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value"},
        ),
    }

    @classmethod
    @abstractmethod
    def _perform_stat(cls, data: Any, stat_type: str) -> Any:
        """Framework-specific implementation, supplied by the concrete subclass."""


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


def _info_for(name: str) -> FeatureGroupInfo:
    results = [info for info in get_feature_group_docs(name=name) if info.name == name]
    assert len(results) == 1, f"Expected exactly one FeatureGroupInfo for '{name}', got {[r.name for r in results]}"
    return results[0]


class TestFeatureGroupInfoSubtypeFields:
    """Contract 9: FeatureGroupInfo carries the subtype universe and support matrix."""

    def test_fields_default_so_existing_construction_keeps_working(self) -> None:
        info = FeatureGroupInfo(
            name="test_group",
            description="A test feature group",
            version="1.0.0",
            module="mloda_plugins.test_module",
            compute_frameworks=["pandas"],
            supported_feature_names={"feature1"},
            prefix="test_",
        )

        assert info.subtype_key is None
        assert info.subtypes == []
        assert info.subtype_support == {}
        assert info.subtype_error is None

    def test_docs_populate_subtype_key_and_sorted_subtypes(self) -> None:
        info = _info_for("DocsSubtypeFeatureGroup")

        assert info.subtype_key == "stat_type"
        assert info.subtypes == ["median", "sum"], "subtypes must be sorted"

    def test_docs_populate_the_subtype_support_matrix(self) -> None:
        info = _info_for("DocsSubtypeFeatureGroup")

        assert info.subtype_support == {
            "PandasDataFrame": ["median", "sum"],
            "PythonDictFramework": ["sum"],
        }
        assert info.subtype_error is None

    def test_fields_stay_empty_without_a_subtype_dimension(self) -> None:
        info = _info_for("DocsNoSubtypeFeatureGroup")

        assert info.subtype_key is None
        assert info.subtypes == []
        assert info.subtype_support == {}
        assert info.subtype_error is None


class TestAbstractBaseDocs:
    """Contract 11: an abstract base documents the universe, not a fabricated support matrix."""

    def test_abstract_test_base_reports_subtypes_without_support(self) -> None:
        info = _info_for("DocsAbstractSubtypeFeatureGroup")

        assert info.subtypes == ["median", "sum"]
        assert info.subtype_support == {}, "An abstract base must not claim support on every framework"
        assert info.subtype_error is None

    def test_abstract_aggregated_base_reports_subtypes_without_support(self) -> None:
        info = _info_for("AggregatedFeatureGroup")

        assert info.subtypes == AGGREGATION_SUBTYPES
        assert info.subtype_support == {}, (
            "AggregatedFeatureGroup is abstract: frameworks without an aggregation "
            "implementation must not be documented as supporting every aggregation type"
        )
        assert info.subtype_error is None

    def test_concrete_aggregated_subclass_documents_its_framework(self) -> None:
        info = _info_for("PandasAggregatedFeatureGroup")

        assert info.subtype_support == {"PandasDataFrame": AGGREGATION_SUBTYPES}
        assert info.subtype_error is None


class TestMisdeclaredCapabilityIsSurfaced:
    """Contract 12: a misdeclared capability is reported via subtype_error, not swallowed."""

    def test_misdeclared_group_is_documented_with_an_error(self) -> None:
        class DocsMisdeclaredFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            """Declares a capability outside its own universe."""

            SUBTYPE_KEY = "stat_type"
            PREFIX_PATTERN = r".*__([\w]+)_docsbroken$"

            PROPERTY_MAPPING = {
                "stat_type": property_spec(
                    "statistic to compute",
                    strict=True,
                    allowed_values={"sum": "Sum of values", "median": "Median value"},
                ),
            }

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
                return {PandasDataFrame}

            @classmethod
            def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
                return frozenset({"sum", "bogus_stat"})

            @classmethod
            def _perform_stat(cls, data: Any, stat_type: str) -> Any:
                return None

        infos = {info.name: info for info in get_feature_group_docs(name="Docs")}

        broken = infos["DocsMisdeclaredFeatureGroup"]
        assert broken.subtypes == ["median", "sum"], "The universe is still reported"
        assert broken.subtype_support == {}, "A misdeclared matrix degrades to empty"
        assert broken.subtype_error is not None, (
            "A misdeclared capability must be distinguishable from a legitimately unsupported group"
        )
        assert "bogus_stat" in broken.subtype_error, (
            f"The error must name the offending subtype, but got: {broken.subtype_error}"
        )

        healthy = infos["DocsSubtypeFeatureGroup"]
        assert healthy.subtype_error is None, "Other groups in the same call must document cleanly"
        assert healthy.subtype_support == {
            "PandasDataFrame": ["median", "sum"],
            "PythonDictFramework": ["sum"],
        }

    def test_unrelated_plugins_still_document_cleanly(self) -> None:
        class DocsOtherMisdeclaredFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            """Second misdeclared group: it must not poison unrelated documentation."""

            SUBTYPE_KEY = "stat_type"
            PREFIX_PATTERN = r".*__([\w]+)_docsbroken2$"

            PROPERTY_MAPPING = {
                "stat_type": property_spec(
                    "statistic to compute",
                    strict=True,
                    allowed_values={"sum": "Sum of values"},
                ),
            }

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
                return {PandasDataFrame}

            @classmethod
            def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
                return frozenset({"also_bogus"})

        infos = get_feature_group_docs(name="Aggregated")

        assert infos, "The aggregated family must still be documented"
        for info in infos:
            assert info.subtype_error is None, f"{info.name} must document cleanly: {info.subtype_error}"

        concrete = {info.name: info for info in infos}["PandasAggregatedFeatureGroup"]
        assert concrete.subtype_support == {"PandasDataFrame": AGGREGATION_SUBTYPES}


class TestResolvedFeatureSubtype:
    """Contract 10: ResolvedFeature.subtype comes from the resolved feature group."""

    def test_field_defaults_to_none(self) -> None:
        result = ResolvedFeature("n", None, [], None)

        assert result.subtype is None

    def test_resolve_feature_populates_the_subtype(self) -> None:
        result = resolve_feature(DOCS_SUBTYPE_FEATURE)

        assert result.feature_group is DocsSubtypeFeatureGroup, f"Unexpected resolution: {result.error}"
        assert result.subtype == "median"

    def test_subtype_is_none_for_a_feature_group_without_a_subtype_dimension(self) -> None:
        result = resolve_feature(DocsNoSubtypeFeatureGroup.get_class_name())

        assert result.feature_group is DocsNoSubtypeFeatureGroup, f"Unexpected resolution: {result.error}"
        assert result.subtype is None

    def test_subtype_is_none_when_nothing_resolves(self) -> None:
        result = resolve_feature("CompletelyUnknownSubtypeFeature12345XYZ")

        assert result.feature_group is None
        assert result.subtype is None
