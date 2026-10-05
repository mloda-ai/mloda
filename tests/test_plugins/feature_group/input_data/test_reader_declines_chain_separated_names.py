"""Unvalidated readers (a suffix file reader family) must decline chain/column-separated names.

Database and document groups: their contract mixins.

The reader and feature groups here become global subclasses discovered process-wide, so every
name carries a "chaindecline" marker to stay inert for other tests under pytest-xdist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import (
    BaseInputData,
    CHAIN_SEPARATOR,
    COLUMN_SEPARATOR,
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
CHAINDECLINE_MULTI_OUTPUT_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{COLUMN_SEPARATOR}0"

CHAINDECLINE_ABORT_PLAIN_FEATURE = "chaindecline_abort_plain_column"

CHAINDECLINE_GENERIC_SUFFIX = ".chaindeclinegeneric"


class ChainDeclineUnvalidatedReader(SuffixFileReader):
    """Final reader owning CHAINDECLINE_FILE_SUFFIX; never overrides get_column_names."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_FILE_SUFFIX,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_PLAIN_FEATURE: [1]}


class ChainDeclineGenericPinReader(BaseInputData):
    """Never a final reader; routes a pinned request through BaseInputData._resolve_pinned_file only."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_GENERIC_SUFFIX,)

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, DataAccessCollection) and cls._pin_applies(data_access, feature_names):
            return cls._resolve_pinned_file(data_access, feature_names)
        return None


class ChainDeclineGenericAbortReader(ChainDeclineGenericPinReader):
    """Never a final reader; validate_columns raises a marked NotImplementedError."""

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        raise escalate_match_abort(NotImplementedError("chaindecline generic marked abort"))


class ChainDeclineGenericTypeErrorReader(ChainDeclineGenericPinReader):
    """Never a final reader; validate_columns raises a plain TypeError."""

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        raise TypeError("chaindecline generic code defect")


class ChainDeclineGenericAbortSuffixReader(ChainDeclineGenericPinReader):
    """Never a final reader; suffix() raises a marked NotImplementedError."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise escalate_match_abort(NotImplementedError("chaindecline generic suffix abort"))


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


class TestFirstSeparatorName:
    """BaseInputData._first_separator_name finds the first chain/column-separated name, else None."""

    @pytest.mark.parametrize(
        ("feature_names", "expected"),
        [
            ([CHAINDECLINE_PLAIN_FEATURE, CHAINDECLINE_CHAIN_FEATURE], CHAINDECLINE_CHAIN_FEATURE),
            ([CHAINDECLINE_PLAIN_FEATURE, CHAINDECLINE_MULTI_OUTPUT_FEATURE], CHAINDECLINE_MULTI_OUTPUT_FEATURE),
            ([CHAINDECLINE_PLAIN_FEATURE, "chaindecline_other_column"], None),
            ([], None),
            ([CHAINDECLINE_CHAIN_FEATURE, CHAINDECLINE_MULTI_OUTPUT_FEATURE], CHAINDECLINE_CHAIN_FEATURE),
            ([CHAINDECLINE_MULTI_OUTPUT_FEATURE, CHAINDECLINE_CHAIN_FEATURE], CHAINDECLINE_MULTI_OUTPUT_FEATURE),
        ],
        ids=["chain", "column", "plain_names", "empty", "chain_before_column", "column_before_chain"],
    )
    def test_returns_first_offending_name_or_none(self, feature_names: list[str], expected: str | None) -> None:
        assert BaseInputData._first_separator_name(feature_names) == expected


class TestBaseInputDataPinnedResolutionPropagation:
    """BaseInputData's pinned resolution exempts separator names and never contains marked aborts or defects."""

    def test_pinned_chain_shaped_name_resolves_via_the_pin(self) -> None:
        path = f"pinned{CHAINDECLINE_GENERIC_SUFFIX}"
        dac = DataAccessCollection(
            files={"chaindecline_generic_handle": path},
            column_to_file={CHAINDECLINE_CHAIN_FEATURE: "chaindecline_generic_handle"},
        )

        assert ChainDeclineGenericPinReader._resolve_pinned_file(dac, [CHAINDECLINE_CHAIN_FEATURE]) == path
        matched = ChainDeclineGenericPinReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())
        assert matched == path

    def test_marked_abort_from_validate_columns_reraises_on_the_pinned_path(self) -> None:
        dac = DataAccessCollection(
            files={"chaindecline_generic_abort_handle": f"pinned{CHAINDECLINE_GENERIC_SUFFIX}"},
            column_to_file={CHAINDECLINE_ABORT_PLAIN_FEATURE: "chaindecline_generic_abort_handle"},
        )

        with pytest.raises(NotImplementedError) as excinfo:
            ChainDeclineGenericAbortReader._resolve_pinned_file(dac, [CHAINDECLINE_ABORT_PLAIN_FEATURE])

        assert is_match_abort(excinfo.value)

    def test_type_error_from_validate_columns_reraises_on_the_pinned_path(self) -> None:
        dac = DataAccessCollection(
            files={"chaindecline_generic_type_handle": f"pinned{CHAINDECLINE_GENERIC_SUFFIX}"},
            column_to_file={CHAINDECLINE_ABORT_PLAIN_FEATURE: "chaindecline_generic_type_handle"},
        )

        with pytest.raises(TypeError):
            ChainDeclineGenericTypeErrorReader._resolve_pinned_file(dac, [CHAINDECLINE_ABORT_PLAIN_FEATURE])

    def test_marked_abort_from_suffix_reraises_instead_of_reading_as_no_suffix(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            ChainDeclineGenericAbortSuffixReader._matches_suffix("any.file")

        assert is_match_abort(excinfo.value)


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
