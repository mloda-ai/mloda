"""A chain-shaped name over a document group resolves to the chained group only.

The feature groups here become global subclasses discovered process-wide, so every name carries a
"chaindecline" marker to stay inert for other tests under pytest-xdist.
"""

from __future__ import annotations

from pathlib import Path

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    PropertySpec,
)
from mloda.user import DataAccessCollection, Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group


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
