"""Unvalidated readers must decline chain/column-separated names instead of assuming they own them.

The reader and feature groups here become global subclasses discovered process-wide, so every
name carries a "chaindecline" marker to stay inert for other tests under pytest-xdist.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
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
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader


CHAINDECLINE_FILE_SUFFIX = ".chaindeclinecsv"
CHAINDECLINE_PLAIN_FEATURE = "chaindecline_plain_column"
CHAINDECLINE_CHAIN_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased_chaindeclinechain"
CHAINDECLINE_MULTI_OUTPUT_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{COLUMN_SEPARATOR}0"

CHAINDECLINE_DB_MARKER_KEY = "chaindecline_db_marker"
CHAINDECLINE_DB_ACCESS: dict[str, Any] = {CHAINDECLINE_DB_MARKER_KEY: True}
CHAINDECLINE_DB_PLAIN_FEATURE = "chaindecline_db_plain_column"
CHAINDECLINE_DB_CHAIN_FEATURE = f"{CHAINDECLINE_DB_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased"
CHAINDECLINE_DB_MULTI_OUTPUT_FEATURE = f"{CHAINDECLINE_DB_PLAIN_FEATURE}{COLUMN_SEPARATOR}0"

CHAINDECLINE_DOC_FILE_SUFFIX = ".chaindeclinedoccsv"
CHAINDECLINE_DOC_PLAIN_FEATURE = "chaindecline_doc_plain_column"
CHAINDECLINE_DOC_CHAIN_FEATURE = f"{CHAINDECLINE_DOC_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased"

CHAINDECLINE_SELF_FILE_SUFFIX = ".chaindeclineselfcsv"
CHAINDECLINE_SELF_COLUMN_FEATURE = "chaindecline_self_column"


class ChainDeclineUnvalidatedReader(ReadFile):
    """Final reader owning CHAINDECLINE_FILE_SUFFIX; never overrides get_column_names."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_FILE_SUFFIX,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_PLAIN_FEATURE: [1]}


class ChainDeclineDocumentedOverrideReader(ReadFile):
    """Final reader whose validate_columns override matches the documented super-delegating shape."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_DOC_FILE_SUFFIX,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_DOC_PLAIN_FEATURE: [1]}

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        return super().validate_columns(file_name, feature_names)


class ChainDeclineSelfValidatingReader(ReadFile):
    """Final reader overriding validate_columns without overriding get_column_names."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_SELF_FILE_SUFFIX,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_SELF_COLUMN_FEATURE: [1]}

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        return True


class ChainDeclineUnvalidatedDbReader(ReadDB):
    """Accepts only the marker credentials; never overrides check_feature_in_data_access."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return credentials.get(CHAINDECLINE_DB_MARKER_KEY) is True

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_DB_PLAIN_FEATURE: [1]}


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


class ChainDeclineDbRootFG(FeatureGroup):
    """Root group fronting ChainDeclineUnvalidatedDbReader; matches whatever credentials it accepts."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ChainDeclineUnvalidatedDbReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ChainDeclineDbChainedFG(FeatureChainParserMixin, FeatureGroup):
    """Chain-shaped db group: chaindecline_db_plain_column__<op>; default forward_group input_features."""

    PREFIX_PATTERN = r"chaindecline_db_plain_column__([\w]+)$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation applied to the source values",
            allowed_values={"rebased": "Rebases the source values"},
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec("Source features", context=True),
    }


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one direct matcher call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


class TestDocumentedOverrideReader:
    """A validate_columns override matching the documented super-delegating shape still gates the general path."""

    def test_chain_shaped_pin_resolves(self) -> None:
        path = f"pinned{CHAINDECLINE_DOC_FILE_SUFFIX}"
        dac = DataAccessCollection(
            files={"chaindecline_doc_handle": path},
            column_to_file={CHAINDECLINE_DOC_CHAIN_FEATURE: "chaindecline_doc_handle"},
        )

        resolved = ChainDeclineDocumentedOverrideReader._resolve_pinned_file(dac, [CHAINDECLINE_DOC_CHAIN_FEATURE])
        assert resolved == path

        matched = ChainDeclineDocumentedOverrideReader.match_subclass_data_access(
            dac, [CHAINDECLINE_DOC_CHAIN_FEATURE], Options()
        )
        assert matched == path

    def test_chain_shaped_name_declines_on_the_general_path(self, rejection_window: dict[str, MatchRejection]) -> None:
        result = ChainDeclineDocumentedOverrideReader.match_read_file_data_access(
            [f"dummy{CHAINDECLINE_DOC_FILE_SUFFIX}"], [CHAINDECLINE_DOC_CHAIN_FEATURE]
        )

        assert result is None
        assert list(rejection_window) == [ChainDeclineDocumentedOverrideReader.get_class_name()]
        stored = rejection_window[ChainDeclineDocumentedOverrideReader.get_class_name()]
        assert "get_column_names" in stored.reason


class TestValidateColumnsOverrideStaysGuarded:
    """A validate_columns override without get_column_names does not exempt the general matching path."""

    def test_validate_columns_override_alone_still_declines_a_chain_shaped_name(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        file_path = f"dummy{CHAINDECLINE_SELF_FILE_SUFFIX}"

        matched = ChainDeclineSelfValidatingReader.match_read_file_data_access(
            [file_path], [f"{CHAINDECLINE_SELF_COLUMN_FEATURE}{CHAIN_SEPARATOR}rebased"]
        )

        assert matched is None
        assert list(rejection_window) == [ChainDeclineSelfValidatingReader.get_class_name()]


class TestReadFileDeclinesChainSeparatedNames:
    """The general matching path must decline CHAIN_SEPARATOR/COLUMN_SEPARATOR names when unvalidated."""

    def test_chain_separated_name_declines_and_records(self, rejection_window: dict[str, MatchRejection]) -> None:
        file_path = f"dummy{CHAINDECLINE_FILE_SUFFIX}"

        result = ChainDeclineUnvalidatedReader.match_read_file_data_access([file_path], [CHAINDECLINE_CHAIN_FEATURE])

        assert result is None
        assert list(rejection_window) == [ChainDeclineUnvalidatedReader.get_class_name()]
        stored = rejection_window[ChainDeclineUnvalidatedReader.get_class_name()]
        assert stored.stage == INPUT_DATA_STAGE
        assert ChainDeclineUnvalidatedReader.get_class_name() in stored.reason
        assert CHAINDECLINE_CHAIN_FEATURE in stored.reason
        assert "get_column_names" in stored.reason

    def test_column_separated_name_declines_and_records(self, rejection_window: dict[str, MatchRejection]) -> None:
        file_path = f"dummy{CHAINDECLINE_FILE_SUFFIX}"

        result = ChainDeclineUnvalidatedReader.match_read_file_data_access(
            [file_path], [CHAINDECLINE_MULTI_OUTPUT_FEATURE]
        )

        assert result is None
        assert list(rejection_window) == [ChainDeclineUnvalidatedReader.get_class_name()]

    def test_plain_name_still_assumes_columns_and_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        file_path = f"dummy{CHAINDECLINE_FILE_SUFFIX}"

        result = ChainDeclineUnvalidatedReader.match_read_file_data_access([file_path], [CHAINDECLINE_PLAIN_FEATURE])

        assert result == file_path
        assert rejection_window == {}


class TestPinnedNameExemptsSeparatorGuard:
    """An explicit column_to_file pin IS the confirmation the separator guard demands; it must not be declined."""

    def test_validate_columns_never_applies_the_separator_guard(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A literal column named "price__scaled" is an ordinary pyarrow column name. Now that
        FeatherReader overrides get_column_names, validate_columns finds it for real, so True
        reflects the column genuinely existing, not a NotImplementedError being swallowed."""
        table = pa.Table.from_pydict({"price__scaled": [1, 2, 3]})
        file_path = str(tmp_path / "pinned.feather")
        with pa.OSFile(file_path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

        result = FeatherReader.validate_columns(file_path, ["price__scaled"])

        assert result is True
        assert rejection_window == {}

    def test_pinned_chain_shaped_name_resolves_via_the_pin(self) -> None:
        """An explicit pin for a chain-shaped name resolves instead of being declined."""
        path = f"pinned{CHAINDECLINE_FILE_SUFFIX}"
        dac = DataAccessCollection(
            files={"chaindecline_pin_handle": path},
            column_to_file={CHAINDECLINE_CHAIN_FEATURE: "chaindecline_pin_handle"},
        )

        resolved = ChainDeclineUnvalidatedReader._resolve_pinned_file(dac, [CHAINDECLINE_CHAIN_FEATURE])
        assert resolved == path

        matched = ChainDeclineUnvalidatedReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())
        assert matched == path

    def test_non_pinned_chain_shaped_name_still_declines_via_the_general_path(self) -> None:
        """The pin exemption is specific to a genuinely pinned name, not a blanket bypass of the reader."""
        path = f"other{CHAINDECLINE_FILE_SUFFIX}"
        dac = DataAccessCollection(
            files={path},
            column_to_file={CHAINDECLINE_PLAIN_FEATURE: path},
        )

        matched = ChainDeclineUnvalidatedReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())

        assert matched is None


class TestReadDbDeclinesChainSeparatedNames:
    """match_read_db_data_access must decline CHAIN_SEPARATOR/COLUMN_SEPARATOR names when unvalidated."""

    def test_chain_separated_name_declines_and_records(self, rejection_window: dict[str, MatchRejection]) -> None:
        matched = ChainDeclineUnvalidatedDbReader.match_read_db_data_access(
            [CHAINDECLINE_DB_ACCESS], [CHAINDECLINE_DB_CHAIN_FEATURE]
        )

        assert matched is None
        assert list(rejection_window) == [ChainDeclineUnvalidatedDbReader.get_class_name()]
        stored = rejection_window[ChainDeclineUnvalidatedDbReader.get_class_name()]
        assert stored.stage == INPUT_DATA_STAGE
        assert ChainDeclineUnvalidatedDbReader.get_class_name() in stored.reason
        assert CHAINDECLINE_DB_CHAIN_FEATURE in stored.reason

    def test_column_separated_name_declines_and_records(self, rejection_window: dict[str, MatchRejection]) -> None:
        matched = ChainDeclineUnvalidatedDbReader.match_read_db_data_access(
            [CHAINDECLINE_DB_ACCESS], [CHAINDECLINE_DB_MULTI_OUTPUT_FEATURE]
        )

        assert matched is None
        assert list(rejection_window) == [ChainDeclineUnvalidatedDbReader.get_class_name()]

    def test_plain_name_still_matches_and_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        """The credentials dict is returned unchanged, silently."""
        matched = ChainDeclineUnvalidatedDbReader.match_read_db_data_access(
            [CHAINDECLINE_DB_ACCESS], [CHAINDECLINE_DB_PLAIN_FEATURE]
        )

        assert matched is CHAINDECLINE_DB_ACCESS
        assert rejection_window == {}


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


class TestChainedDbFeatureResolvesToSingleFeatureGroup:
    """A chain-shaped db name resolves to exactly the chained group."""

    def _accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return {ChainDeclineDbRootFG: {PandasDataFrame}, ChainDeclineDbChainedFG: {PandasDataFrame}}

    def test_plain_db_name_resolves_to_root_group_only(self) -> None:
        feature = Feature(
            name=CHAINDECLINE_DB_PLAIN_FEATURE,
            options={ChainDeclineUnvalidatedDbReader.__name__: CHAINDECLINE_DB_ACCESS},
        )

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None)

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineDbRootFG: {PandasDataFrame}}

    def test_chain_shaped_db_name_resolves_to_chained_group_only(self) -> None:
        feature = Feature(
            name=CHAINDECLINE_DB_CHAIN_FEATURE,
            options={ChainDeclineUnvalidatedDbReader.__name__: CHAINDECLINE_DB_ACCESS},
        )

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None)

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineDbChainedFG: {PandasDataFrame}}
        assert ChainDeclineDbRootFG not in result.identified
