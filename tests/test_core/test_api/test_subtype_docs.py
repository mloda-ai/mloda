"""Failing tests pinning the Phase 2 subtype documentation contract (issue #639)."""

from abc import abstractmethod
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureChainParserMixin, FeatureGroup, property_spec
from mloda.steward import FeatureGroupInfo, ResolvedFeature, get_feature_group_docs, resolve_feature
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


WINDOW_KEY = "sudoc_window_function"
WINDOW_UNIVERSE = frozenset({"median", "sum", "lag"})
WINDOW_DICT_SUPPORTED = frozenset({"sum", "lag"})

RANK_KEY = "suranked_rank_type"
RANK_NAMED = frozenset({"dense", "ordinal"})
RANK_FAMILIES = {"ntile": "N-tile bucketing", "top": "Top-N selection"}
RANK_UNIVERSE = RANK_NAMED | frozenset(RANK_FAMILIES)

ABSTRACT_KEY = "sudocabs_window_function"
ABSTRACT_UNIVERSE = frozenset({"median", "sum"})

OVERREACH_KEY = "sudocbad_kind"
OVERREACH_UNIVERSE = frozenset({"fast", "slow"})
OVERREACH_BOGUS = "sudocbad_bogus"

OPTION_KEY = "sudoc_op_kind"
OPTION_FEATURE = "sudoc_option_feature"
PLAIN_FEATURE = "sudoc_plain_feature"


class SuDocWindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Shape A family with a per-framework supported_subtypes override."""

    SUBTYPE_KEY = WINDOW_KEY
    PREFIX_PATTERN = r".*__([\w]+)_sudoc$"
    PROPERTY_MAPPING = {
        WINDOW_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum", "lag": "Lag"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is PythonDictFramework:
            return WINDOW_DICT_SUPPORTED
        return WINDOW_UNIVERSE


class SuRankedDocFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Rank-like family: named subtypes plus parametric families, 'ntile' unsupported on one framework."""

    SUBTYPE_KEY = RANK_KEY
    PARAMETRIC_SUBTYPE_FAMILIES = RANK_FAMILIES
    PREFIX_PATTERN = r".*__([\w]+)_suranked$"
    PROPERTY_MAPPING = {
        RANK_KEY: property_spec(
            "Rank subtype.",
            strict=True,
            allowed_values={"dense": "Dense ranking", "ordinal": "Ordinal ranking"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        if compute_framework is PythonDictFramework:
            return RANK_UNIVERSE - {"ntile"}
        return RANK_UNIVERSE


class SuDocAbstractWindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Abstract base declaring a subtype universe; support belongs to concrete members."""

    SUBTYPE_KEY = ABSTRACT_KEY
    PREFIX_PATTERN = r".*__([\w]+)_sudocabs$"
    PROPERTY_MAPPING = {
        ABSTRACT_KEY: property_spec(
            "Abstract window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @abstractmethod
    def _sudoc_backend_marker(self) -> None: ...


class SuDocOverreachFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Misdeclared family: declares support for a subtype outside its universe."""

    SUBTYPE_KEY = OVERREACH_KEY
    PREFIX_PATTERN = r".*__([\w]+)_sudocbad$"
    PROPERTY_MAPPING = {
        OVERREACH_KEY: property_spec(
            "Misdeclared kind.",
            strict=True,
            allowed_values={"fast": "Fast", "slow": "Slow"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        return OVERREACH_UNIVERSE | {OVERREACH_BOGUS}


class SuDocOptionFeatureGroup(FeatureGroup):
    """Fixed-name family whose subtype resolves from options only."""

    SUBTYPE_KEY = OPTION_KEY
    PROPERTY_MAPPING = {
        OPTION_KEY: property_spec(
            "Operation kind.",
            strict=True,
            allowed_values={"sum": "Sum", "median": "Median"},
        ),
    }

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
        return str(feature_name) == OPTION_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SuDocPlainFeatureGroup(FeatureGroup):
    """Family without a subtype dimension."""

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


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


def _doc_for(name: str) -> FeatureGroupInfo:
    """Fetch the single FeatureGroupInfo whose name matches exactly."""
    exact = [doc for doc in get_feature_group_docs(name=name) if doc.name == name]
    assert len(exact) == 1, f"expected exactly one doc for {name}, got {[doc.name for doc in exact]}"
    return exact[0]


class TestContractAFeatureGroupInfoFields:
    """Contract A: FeatureGroupInfo gains defaulted subtype fields."""

    def test_positional_construction_defaults_new_fields(self) -> None:
        info = FeatureGroupInfo("n", "d", "v", "m", [], set(), "n_")
        assert info.subtype_key is None
        assert info.subtypes == []
        assert info.parametric_subtypes == []
        assert info.subtype_support == {}
        assert info.subtype_error is None

    def test_keyword_construction_accepts_new_fields(self) -> None:
        info = FeatureGroupInfo(
            "n",
            "d",
            "v",
            "m",
            [],
            set(),
            "n_",
            subtype_key="k",
            subtypes=["a", "b"],
            parametric_subtypes=["b"],
            subtype_support={"Fw": ["a", "b"]},
            subtype_error="boom",
        )
        assert info.subtype_key == "k"
        assert info.subtypes == ["a", "b"]
        assert info.parametric_subtypes == ["b"]
        assert info.subtype_support == {"Fw": ["a", "b"]}
        assert info.subtype_error == "boom"

    def test_mutable_defaults_are_not_shared(self) -> None:
        first = FeatureGroupInfo("n1", "d", "v", "m", [], set(), "n1_")
        second = FeatureGroupInfo("n2", "d", "v", "m", [], set(), "n2_")
        first.subtypes.append("x")
        first.parametric_subtypes.append("x")
        first.subtype_support["Fw"] = ["x"]
        assert second.subtypes == []
        assert second.parametric_subtypes == []
        assert second.subtype_support == {}


class TestContractBFeatureGroupDocs:
    """Contract B: get_feature_group_docs populates the subtype fields."""

    def test_shape_a_family_reports_key_universe_and_support(self) -> None:
        doc = _doc_for("SuDocWindowFeatureGroup")
        assert doc.subtype_key == WINDOW_KEY
        assert doc.subtypes == sorted(WINDOW_UNIVERSE)
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {
            "PandasDataFrame": sorted(WINDOW_UNIVERSE),
            "PythonDictFramework": sorted(WINDOW_DICT_SUPPORTED),
        }
        assert doc.subtype_error is None

    def test_parametric_families_are_enumerable_without_probing(self) -> None:
        doc = _doc_for("SuRankedDocFeatureGroup")
        assert doc.subtype_key == RANK_KEY
        assert doc.subtypes == sorted(RANK_UNIVERSE)
        assert doc.parametric_subtypes == sorted(RANK_FAMILIES)
        assert doc.subtype_support == {
            "PandasDataFrame": sorted(RANK_UNIVERSE),
            "PythonDictFramework": sorted(RANK_UNIVERSE - {"ntile"}),
        }
        assert doc.subtype_error is None

    def test_family_without_subtype_dimension_keeps_defaults(self) -> None:
        doc = _doc_for("SuDocPlainFeatureGroup")
        assert doc.subtype_key is None
        assert doc.subtypes == []
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {}
        assert doc.subtype_error is None

    def test_abstract_base_reports_universe_without_support(self) -> None:
        doc = _doc_for("SuDocAbstractWindowFeatureGroup")
        assert doc.subtype_key == ABSTRACT_KEY
        assert doc.subtypes == sorted(ABSTRACT_UNIVERSE)
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {}
        assert doc.subtype_error is None

    def test_misdeclared_family_is_surfaced_not_swallowed(self) -> None:
        docs = get_feature_group_docs()
        by_name = {doc.name: doc for doc in docs}

        bad = by_name["SuDocOverreachFeatureGroup"]
        assert bad.subtypes == sorted(OVERREACH_UNIVERSE)
        assert bad.subtype_support == {}
        assert bad.subtype_error is not None
        assert OVERREACH_BOGUS in bad.subtype_error

        healthy = by_name["SuDocWindowFeatureGroup"]
        assert healthy.subtype_error is None


class TestContractCResolvedFeatureSubtype:
    """Contract C: ResolvedFeature carries the resolved subtype and its parametric family."""

    def test_direct_construction_defaults_to_none(self) -> None:
        result = ResolvedFeature("n", None, [], None)
        assert result.subtype is None
        assert result.subtype_family is None

    def test_string_path_resolves_subtype(self) -> None:
        result = resolve_feature("value__median_sudoc")
        assert result.feature_group is SuDocWindowFeatureGroup
        assert result.subtype == "median"
        assert result.subtype_family is None

    def test_parametric_path_resolves_subtype_and_family(self) -> None:
        result = resolve_feature("value__ntile_2_suranked")
        assert result.feature_group is SuRankedDocFeatureGroup
        assert result.subtype == "ntile_2"
        assert result.subtype_family == "ntile"

    def test_options_path_resolves_subtype(self) -> None:
        result = resolve_feature(OPTION_FEATURE, options=Options(context={OPTION_KEY: "sum"}))
        assert result.feature_group is SuDocOptionFeatureGroup
        assert result.subtype == "sum"
        assert result.subtype_family is None

    def test_family_without_subtype_dimension_gives_none(self) -> None:
        result = resolve_feature(PLAIN_FEATURE)
        assert result.feature_group is SuDocPlainFeatureGroup
        assert result.subtype is None
        assert result.subtype_family is None

    def test_nothing_resolves_gives_none(self) -> None:
        result = resolve_feature(OPTION_FEATURE)
        assert result.feature_group is SuDocOptionFeatureGroup
        assert result.subtype is None
        assert result.subtype_family is None
