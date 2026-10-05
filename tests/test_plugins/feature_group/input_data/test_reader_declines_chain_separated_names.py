"""Unvalidated suffix file readers must decline chain/column-separated names.

Database and document groups: their contract mixins.

The reader and feature groups here become global subclasses discovered process-wide, so every
name carries a "chaindecline" marker to stay inert for other tests under pytest-xdist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import (
    BaseInputData,
    CHAIN_SEPARATOR,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    INPUT_DATA_STAGE,
    PropertySpec,
)
from mloda.user import DataAccessCollection, Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.helpers.suffix_file_reader import SuffixFileReader
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group


CHAINDECLINE_FILE_SUFFIX = ".chaindeclinecsv"
CHAINDECLINE_PLAIN_FEATURE = "chaindecline_plain_column"
CHAINDECLINE_CHAIN_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased_chaindeclinechain"


class ChainDeclineUnvalidatedReader(SuffixFileReader):
    """Reader owning CHAINDECLINE_FILE_SUFFIX; never overrides get_column_names."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_FILE_SUFFIX,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_PLAIN_FEATURE: [1]}


class ChainDeclineRootFG(FeatureGroup):
    """Root group fronting ChainDeclineUnvalidatedReader; matches whatever the reader accepts."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ChainDeclineUnvalidatedReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ChainDeclineChainedFG(FeatureChainParserMixin, FeatureGroup):
    """Chain-shaped group: <source>__<op>_chaindeclinechain; default forward_group input_features."""

    PREFIX_PATTERN = r".*__([\w]+)_chaindeclinechain$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation applied to the source values",
            allowed_values={"rebased": "Rebases the source values"},
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec("Source features", context=True),
    }


class ChainDeclineNamedRootFG(FeatureGroup):
    """Root group fronting ChainDeclineUnvalidatedReader that ALSO names the chain-shaped feature explicitly."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ChainDeclineUnvalidatedReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {CHAINDECLINE_CHAIN_FEATURE}


class TestChainedFeatureResolvesToSingleFeatureGroup:
    """A chain-shaped name resolves to exactly the chained group."""

    def _accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return {ChainDeclineRootFG: {PandasDataFrame}, ChainDeclineChainedFG: {PandasDataFrame}}

    def test_plain_root_name_still_resolves_to_root_group_only(self) -> None:
        feature = Feature(
            name=CHAINDECLINE_PLAIN_FEATURE,
            options={ChainDeclineUnvalidatedReader.__name__: f"dummy{CHAINDECLINE_FILE_SUFFIX}"},
        )

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None)

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineRootFG: {PandasDataFrame}}

    def test_chain_shaped_name_resolves_to_chained_group_only(self) -> None:
        """The over-permissive reader declines the chain-shaped name, so only the chain pattern matches."""
        feature = Feature(
            name=CHAINDECLINE_CHAIN_FEATURE,
            options={ChainDeclineUnvalidatedReader.__name__: f"dummy{CHAINDECLINE_FILE_SUFFIX}"},
        )

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None)

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineChainedFG: {PandasDataFrame}}
        assert ChainDeclineRootFG not in result.identified


class TestOwnedContentDeclineGatesExplicitNameRule:
    """A root group that names a chain-shaped feature explicitly is still gated by its reader's owned decline."""

    def test_explicit_chain_shaped_name_still_declines_via_the_owned_reader_veto(self) -> None:
        feature = Feature(
            name=CHAINDECLINE_CHAIN_FEATURE,
            options={ChainDeclineUnvalidatedReader.__name__: f"dummy{CHAINDECLINE_FILE_SUFFIX}"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {ChainDeclineNamedRootFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None)

        assert result.identified == {}
        elimination = result.eliminations.get(ChainDeclineNamedRootFG)
        assert elimination is not None
        assert elimination.stage == INPUT_DATA_STAGE
        assert ChainDeclineUnvalidatedReader.get_class_name() in elimination.reason
        assert CHAINDECLINE_CHAIN_FEATURE in elimination.reason


class TestChainedFeatureResolvesToSingleFeatureGroupOverDocuments:
    """TextFG declares only its three names, so a chain-shaped name never reaches it."""

    def _accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return {load_document_group("text_fg", "TextFG"): {PandasDataFrame}, ChainDeclineChainedFG: {PandasDataFrame}}

    def _dac(self, tmp_path: Path) -> DataAccessCollection:
        path = tmp_path / "doc.txt"
        path.write_text("chaindecline")
        return DataAccessCollection(files={str(path)})

    def test_the_group_name_still_resolves_to_the_document_group_only(self, tmp_path: Path) -> None:
        result = IdentifyFeatureGroupClass.evaluate(
            Feature(name="TextFG"), self._accessible_plugins(), None, self._dac(tmp_path)
        )

        assert result.failure_kind is None
        assert set(result.identified) == {load_document_group("text_fg", "TextFG")}

    def test_a_chain_shaped_name_resolves_to_the_chained_group_only(self, tmp_path: Path) -> None:
        feature = Feature(name="TextFG__rebased_chaindeclinechain")

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None, self._dac(tmp_path))

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineChainedFG: {PandasDataFrame}}
