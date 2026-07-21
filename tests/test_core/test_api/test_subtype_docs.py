"""Pinning tests for the cycle-2 subtype persona surfaces of issue #639 (docs and resolve_feature)."""

from abc import abstractmethod
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureChainParserMixin, FeatureGroup, SubtypeDeclaration, property_spec
from mloda.steward import FeatureGroupInfo, ResolvedFeature, get_feature_group_docs, resolve_feature
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


WINDOW_KEY = "sbdd_window_function"
WINDOW_UNIVERSE = frozenset({"median", "sum", "lag"})
WINDOW_DICT_SUPPORTED = frozenset({"sum", "lag"})

RANK_KEY = "sbdd_rank_type"
RANK_NAMED = frozenset({"dense", "ordinal"})
RANK_FAMILIES = {"ntile": "N-tile bucketing", "top": "Top-N selection"}
RANK_UNIVERSE = RANK_NAMED | frozenset(RANK_FAMILIES)

ABSTRACT_KEY = "sbddabs_window_function"
ABSTRACT_UNIVERSE = frozenset({"median", "sum"})

HOOKED_KEY = "sbddhook_kind"
HOOKED_UNIVERSE = frozenset({"median", "sum"})

OPTION_KEY = "sbdd_op_kind"
OPTION_FEATURE = "sbdd_option_feature"
PLAIN_FEATURE = "sbdd_plain_feature"

R4_KEY = "sbddr4_kind"

R5_FEATURE = "sbdd_resolver_feature"


def _sbdd_raising_resolver(feature_name: str, options: Options) -> Optional[str]:
    raise RuntimeError("sbdd resolver exploded")


class SubDeclDocWindowFG(FeatureChainParserMixin, FeatureGroup):
    """Shape A family with a per-framework supported narrowing in the declaration."""

    SUBTYPES = SubtypeDeclaration(
        key=WINDOW_KEY,
        supported={PythonDictFramework.get_class_name(): WINDOW_DICT_SUPPORTED},
    )
    PREFIX_PATTERN = r".*__([\w]+)_sbddwin$"
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


class SubDeclDocRankFG(FeatureChainParserMixin, FeatureGroup):
    """Rank-like family: named subtypes plus parametric families, 'ntile' unsupported on one framework."""

    SUBTYPES = SubtypeDeclaration(
        key=RANK_KEY,
        parametric_families=RANK_FAMILIES,
        supported={PythonDictFramework.get_class_name(): RANK_UNIVERSE - {"ntile"}},
    )
    PREFIX_PATTERN = r".*__([\w]+)_sbddrank$"
    PROPERTY_MAPPING = {
        RANK_KEY: property_spec(
            "Rank subtype.",
            strict=True,
            allowed_values={"dense": "Dense ranking", "ordinal": "Ordinal ranking"},
            deferred_binding=True,  # parametric subtypes (ntile_2) are resolved by SUBTYPES, not name capture
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}


class SubDeclDocAbstractWindowFG(FeatureChainParserMixin, FeatureGroup):
    """Abstract base declaring a subtype universe; support belongs to concrete members."""

    SUBTYPES = SubtypeDeclaration(key=ABSTRACT_KEY)
    PREFIX_PATTERN = r".*__([\w]+)_sbddabs$"
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
    def _sbdd_backend_marker(self) -> None: ...


class SubDeclDocHookOverrideFG(FeatureChainParserMixin, FeatureGroup):
    """Family with a subtype universe AND a hand-written supports_compute_framework hook."""

    SUBTYPES = SubtypeDeclaration(key=HOOKED_KEY)
    PREFIX_PATTERN = r".*__([\w]+)_sbddhook$"
    PROPERTY_MAPPING = {
        HOOKED_KEY: property_spec(
            "Operation kind gated by a hand-written hook.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return True


class SubDeclDocOptionFG(FeatureGroup):
    """Fixed-name family whose subtype resolves from options only."""

    SUBTYPES = SubtypeDeclaration(key=OPTION_KEY)
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


class SubDeclDocPlainFG(FeatureGroup):
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


class SubDeclDocR4RejectingFG(FeatureChainParserMixin, FeatureGroup):
    """Family whose single declared framework rejects everything except 'sum' via the declaration."""

    SUBTYPES = SubtypeDeclaration(
        key=R4_KEY,
        parametric_families={"ntile": "N-tile bucketing"},
        supported={PythonDictFramework.get_class_name(): frozenset({"sum"})},
    )
    PREFIX_PATTERN = r".*__([\w]+)_sbddr4$"
    PROPERTY_MAPPING = {
        R4_KEY: property_spec(
            "Operation kind rejected almost everywhere.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum"},
            deferred_binding=True,  # parametric subtypes (ntile_3) are resolved by SUBTYPES, not name capture
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}


class SubDeclDocRaisingResolverFG(FeatureGroup):
    """Shape B family whose resolver raises when called; resolution must degrade, not raise."""

    SUBTYPES = SubtypeDeclaration(universe={"median", "sum"}, resolver=_sbdd_raising_resolver)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == R5_FEATURE

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


class TestSubDeclDocContractAFeatureGroupInfoFields:
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


class TestSubDeclDocContractBFeatureGroupDocs:
    """Contract B: get_feature_group_docs populates the subtype fields from SUBTYPES."""

    def test_keyed_family_reports_key_universe_and_support(self) -> None:
        doc = _doc_for("SubDeclDocWindowFG")
        assert doc.subtype_key == WINDOW_KEY
        assert doc.subtypes == sorted(WINDOW_UNIVERSE)
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {
            "PandasDataFrame": sorted(WINDOW_UNIVERSE),
            "PythonDictFramework": sorted(WINDOW_DICT_SUPPORTED),
        }
        assert doc.subtype_error is None

    def test_parametric_families_are_enumerable_without_probing(self) -> None:
        doc = _doc_for("SubDeclDocRankFG")
        assert doc.subtype_key == RANK_KEY
        assert doc.subtypes == sorted(RANK_UNIVERSE)
        assert doc.parametric_subtypes == sorted(RANK_FAMILIES)
        assert doc.subtype_support == {
            "PandasDataFrame": sorted(RANK_UNIVERSE),
            "PythonDictFramework": sorted(RANK_UNIVERSE - {"ntile"}),
        }
        assert doc.subtype_error is None

    def test_family_without_subtype_dimension_keeps_defaults(self) -> None:
        doc = _doc_for("SubDeclDocPlainFG")
        assert doc.subtype_key is None
        assert doc.subtypes == []
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {}
        assert doc.subtype_error is None

    def test_abstract_base_reports_universe_without_support(self) -> None:
        doc = _doc_for("SubDeclDocAbstractWindowFG")
        assert doc.subtype_key == ABSTRACT_KEY
        assert doc.subtypes == sorted(ABSTRACT_UNIVERSE)
        assert doc.parametric_subtypes == []
        assert doc.subtype_support == {}
        assert doc.subtype_error is None

    def test_raising_matrix_is_surfaced_not_swallowed(self) -> None:
        bad = _doc_for("SubDeclDocHookOverrideFG")
        assert bad.subtypes == sorted(HOOKED_UNIVERSE)
        assert bad.subtype_support == {}
        assert bad.subtype_error is not None
        assert "supports_compute_framework" in bad.subtype_error

        healthy = _doc_for("SubDeclDocWindowFG")
        assert healthy.subtype_error is None


class TestSubDeclDocContractCResolvedFeatureSubtype:
    """Contract C: ResolvedFeature carries the resolved subtype and its parametric family."""

    def test_direct_construction_defaults_to_none(self) -> None:
        result = ResolvedFeature("n", None, [], None)
        assert result.subtype is None
        assert result.subtype_family is None

    def test_string_path_resolves_subtype(self) -> None:
        result = resolve_feature("value__median_sbddwin")
        assert result.feature_group is SubDeclDocWindowFG
        assert result.subtype == "median"
        assert result.subtype_family is None

    def test_parametric_path_resolves_subtype_and_family(self) -> None:
        result = resolve_feature("value__ntile_2_sbddrank")
        assert result.feature_group is SubDeclDocRankFG
        assert result.subtype == "ntile_2"
        assert result.subtype_family == "ntile"

    def test_options_path_resolves_subtype(self) -> None:
        result = resolve_feature(OPTION_FEATURE, options=Options(context={OPTION_KEY: "sum"}))
        assert result.feature_group is SubDeclDocOptionFG
        assert result.subtype == "sum"
        assert result.subtype_family is None

    def test_family_without_subtype_dimension_gives_none(self) -> None:
        result = resolve_feature(PLAIN_FEATURE)
        assert result.feature_group is SubDeclDocPlainFG
        assert result.subtype is None
        assert result.subtype_family is None

    def test_nothing_resolves_gives_none(self) -> None:
        result = resolve_feature(OPTION_FEATURE)
        assert result.feature_group is SubDeclDocOptionFG
        assert result.subtype is None
        assert result.subtype_family is None


class TestSubDeclDocContractDUnsupportedEverywhere:
    """Contract D: the unsupported-everywhere failure surfaces the rejected frameworks via the engine's
    error text, not via structured split fields or a resolved subtype.

    After #755 resolve_feature delegates the all-rejected failure to the engine, whose error names the
    frameworks. The delegated failure path carries no structured capability split and no resolved subtype.
    """

    def test_unsupported_everywhere_surfaces_engine_error(self) -> None:
        result = resolve_feature("value__median_sbddr4")
        assert result.feature_group is None
        assert result.candidates == [SubDeclDocR4RejectingFG]
        assert result.error is not None
        # Engine wording names the frameworks and the capability hook via the near-miss line.
        assert "Feature group(s) eliminated while matching 'value__median_sbddr4':" in result.error
        assert "SubDeclDocR4RejectingFG (compute framework): supports_compute_framework rejected" in result.error
        assert "PythonDictFramework" in result.error
        # The delegated failure path returns the ResolvedFeature defaults: no structured split, no subtype.
        assert result.supported_compute_frameworks == []
        assert result.unsupported_compute_frameworks == []
        assert result.subtype is None
        assert result.subtype_family is None

    def test_unsupported_everywhere_parametric_instance_surfaces_engine_error(self) -> None:
        result = resolve_feature("value__ntile_3_sbddr4")
        assert result.feature_group is None
        assert result.error is not None
        assert "SubDeclDocR4RejectingFG (compute framework): supports_compute_framework rejected" in result.error
        assert "PythonDictFramework" in result.error
        # No structured subtype on the delegated failure path.
        assert result.subtype is None
        assert result.subtype_family is None


class TestSubDeclDocContractERaisingResolverDegrades:
    """Contract E: a raising shape-B resolver never propagates out of resolve_feature."""

    def test_raising_resolver_degrades_to_none_subtype(self) -> None:
        result = resolve_feature(R5_FEATURE)
        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is SubDeclDocRaisingResolverFG
        assert result.error is None
        assert result.subtype is None
        assert result.subtype_family is None

    def test_raising_resolver_degrades_capability_split_open(self) -> None:
        result = resolve_feature(R5_FEATURE)
        assert result.feature_group is SubDeclDocRaisingResolverFG
        assert result.supported_compute_frameworks == ["PythonDictFramework"]
        assert result.unsupported_compute_frameworks == []


class TestSubDeclDocContractFHookOverrideSurfacedInDocs:
    """Contract F: docs surface a hook-overriding family as subtype_error instead of a fabricated matrix."""

    def test_hook_override_reported_as_subtype_error(self) -> None:
        hooked = _doc_for("SubDeclDocHookOverrideFG")
        assert hooked.subtype_key == HOOKED_KEY
        assert hooked.subtypes == sorted(HOOKED_UNIVERSE)
        assert hooked.parametric_subtypes == []
        assert hooked.subtype_support == {}
        assert hooked.subtype_error is not None
        assert "supports_compute_framework" in hooked.subtype_error
