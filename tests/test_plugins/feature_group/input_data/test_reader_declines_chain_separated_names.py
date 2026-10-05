"""Unvalidated readers (a suffix file reader family, ReadDB, ReadDocument) must decline chain/column-separated names.

The reader and feature groups here become global subclasses discovered process-wide, so every
name carries a "chaindecline" marker to stay inert for other tests under pytest-xdist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
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
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature
from tests.helpers.suffix_file_reader import SuffixFileReader


CHAINDECLINE_FILE_SUFFIX = ".chaindeclinecsv"
CHAINDECLINE_PLAIN_FEATURE = "chaindecline_plain_column"
CHAINDECLINE_CHAIN_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased_chaindeclinechain"
CHAINDECLINE_MULTI_OUTPUT_FEATURE = f"{CHAINDECLINE_PLAIN_FEATURE}{COLUMN_SEPARATOR}0"

CHAINDECLINE_DB_MARKER_KEY = "chaindecline_db_marker"
CHAINDECLINE_DB_ACCESS: dict[str, Any] = {CHAINDECLINE_DB_MARKER_KEY: True}
CHAINDECLINE_DB_PLAIN_FEATURE = "chaindecline_db_plain_column"
CHAINDECLINE_DB_CHAIN_FEATURE = f"{CHAINDECLINE_DB_PLAIN_FEATURE}{CHAIN_SEPARATOR}rebased"
CHAINDECLINE_DB_MULTI_OUTPUT_FEATURE = f"{CHAINDECLINE_DB_PLAIN_FEATURE}{COLUMN_SEPARATOR}0"


CHAINDECLINE_ABORT_PLAIN_FEATURE = "chaindecline_abort_plain_column"

CHAINDECLINE_DB_ABORT_MARKER_KEY = "chaindecline_db_abort_marker"
CHAINDECLINE_DB_ABORT_ACCESS: dict[str, Any] = {CHAINDECLINE_DB_ABORT_MARKER_KEY: True}
CHAINDECLINE_DB_ABORT_PLAIN_FEATURE = "chaindecline_db_abort_plain_column"

CHAINDECLINE_GENERIC_SUFFIX = ".chaindeclinegeneric"
CHAINDECLINE_READDOC_SUFFIX = ".chaindeclinereaddoc"
CHAINDECLINE_READDOC_UNOWNED_SUFFIX = ".chaindeclinereaddocunowned"


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


class ChainDeclineDocReader(ReadDocument):
    """Final document reader owning CHAINDECLINE_READDOC_SUFFIX."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (CHAINDECLINE_READDOC_SUFFIX,)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return {}


class ChainDeclineUnvalidatedDbReader(ReadDB):
    """Accepts only the marker credentials; never overrides check_feature_in_data_access."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return credentials.get(CHAINDECLINE_DB_MARKER_KEY) is True

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_DB_PLAIN_FEATURE: [1]}


class ChainDeclineAbortCredentialsDbReader(ReadDB):
    """Accepts only the marker credentials; is_valid_credentials then raises a marked NotImplementedError."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        if credentials.get(CHAINDECLINE_DB_ABORT_MARKER_KEY) is not True:
            return False
        raise escalate_match_abort(NotImplementedError("chaindecline db marked abort"))

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_DB_ABORT_PLAIN_FEATURE: [1]}


class ChainDeclineAbortFeatureDbReader(ReadDB):
    """Accepts only the marker credentials; check_feature_in_data_access then raises a marked NotImplementedError."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return credentials.get(CHAINDECLINE_DB_ABORT_MARKER_KEY) is True

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        raise escalate_match_abort(NotImplementedError("chaindecline db marked abort"))

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CHAINDECLINE_DB_ABORT_PLAIN_FEATURE: [1]}


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


def _document_dac(route: str, tmp_path: Path, file_name: str) -> tuple[DataAccessCollection, str]:
    """A collection exposing one real file directly (route 'files') or through its folder (route 'folders')."""
    path = tmp_path / file_name
    path.write_text("chaindecline")
    if route == "files":
        return DataAccessCollection(files={str(path)}), str(path)
    return DataAccessCollection(folders={str(tmp_path)}), str(path)


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


class TestReadDbMarkedAbortsPropagate:
    """A marked NotImplementedError from ReadDB's credential/feature hooks must escape matching, not decline
    silently: _credentials_predicate and match_read_db_data_access's own is_valid_credentials/
    check_feature_in_data_access calls are independent call sites, each needing its own guard."""

    def test_marked_abort_from_is_valid_credentials_reraises_via_match_read_db_data_access(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            ChainDeclineAbortCredentialsDbReader.match_read_db_data_access(
                [CHAINDECLINE_DB_ABORT_ACCESS], [CHAINDECLINE_DB_ABORT_PLAIN_FEATURE]
            )

        assert is_match_abort(excinfo.value)

    def test_marked_abort_from_check_feature_in_data_access_reraises_via_match_read_db_data_access(self) -> None:
        """Needs BOTH the check_feature_in_data_access handler and the outer is_valid_credentials handler fixed:
        the inner handler's re-raise falls straight into the outer try's own except clause."""
        with pytest.raises(NotImplementedError) as excinfo:
            ChainDeclineAbortFeatureDbReader.match_read_db_data_access(
                [CHAINDECLINE_DB_ABORT_ACCESS], [CHAINDECLINE_DB_ABORT_PLAIN_FEATURE]
            )

        assert is_match_abort(excinfo.value)

    def test_marked_abort_from_is_valid_credentials_reraises_via_credentials_predicate(self) -> None:
        """_credentials_predicate is a separate caller of is_valid_credentials, reached through
        DataAccessCollection.resolve's predicate parameter before match_read_db_data_access ever runs."""
        dac = DataAccessCollection(credentials={"chaindecline_abort_handle": CHAINDECLINE_DB_ABORT_ACCESS})

        with pytest.raises(NotImplementedError) as excinfo:
            ChainDeclineAbortCredentialsDbReader.match_subclass_data_access(
                dac, [CHAINDECLINE_DB_ABORT_PLAIN_FEATURE], Options()
            )

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


class TestReadDocumentDeclinesChainSeparatedNames:
    """Only the DataAccessCollection route declines; the str route does not."""

    @pytest.mark.parametrize("route", ["files", "folders"])
    @pytest.mark.parametrize(
        "feature_name",
        [
            CHAINDECLINE_CHAIN_FEATURE,
            CHAINDECLINE_MULTI_OUTPUT_FEATURE,  # the '~N' form only reaches a reader by direct call
        ],
    )
    def test_separator_name_declines_and_records(
        self, route: str, feature_name: str, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        dac, path = _document_dac(route, tmp_path, f"doc{CHAINDECLINE_READDOC_SUFFIX}")

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [feature_name], Options())

        assert result is None
        assert list(rejection_window) == [ChainDeclineDocReader.get_class_name()]
        stored = rejection_window[ChainDeclineDocReader.get_class_name()]
        assert stored.stage == INPUT_DATA_STAGE
        assert ChainDeclineDocReader.get_class_name() in stored.reason
        assert feature_name in stored.reason
        assert path in stored.reason

    @pytest.mark.parametrize("feature_name", [CHAINDECLINE_CHAIN_FEATURE, CHAINDECLINE_MULTI_OUTPUT_FEATURE])
    def test_hint_at_owned_file_declines_a_separator_name_instead_of_raising(
        self, feature_name: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The hint pre-check and resolve's predicate must agree, else resolve raises a predicate mismatch."""
        dac = DataAccessCollection(files={"notes": f"a{CHAINDECLINE_READDOC_SUFFIX}"})
        options = Options(context={"data_access_handle": "notes"})

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [feature_name], options)

        assert result is None
        assert list(rejection_window) == [ChainDeclineDocReader.get_class_name()]
        assert feature_name in rejection_window[ChainDeclineDocReader.get_class_name()].reason

    def test_hint_at_owned_file_still_matches_a_plain_name_and_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        path = f"a{CHAINDECLINE_READDOC_SUFFIX}"
        dac = DataAccessCollection(files={"notes": path})
        options = Options(context={"data_access_handle": "notes"})

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_PLAIN_FEATURE], options)

        assert result == path
        assert rejection_window == {}

    @pytest.mark.parametrize("route", ["files", "folders"])
    def test_plain_name_still_matches_and_records_nothing(
        self, route: str, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        dac, path = _document_dac(route, tmp_path, f"doc{CHAINDECLINE_READDOC_SUFFIX}")

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_PLAIN_FEATURE], Options())

        assert result == path
        assert rejection_window == {}

    @pytest.mark.parametrize("route", ["files", "folders"])
    def test_unowned_suffix_stays_silent_for_a_separator_name(
        self, route: str, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        dac, _ = _document_dac(route, tmp_path, f"doc{CHAINDECLINE_READDOC_UNOWNED_SUFFIX}")

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())

        assert result is None
        assert rejection_window == {}

    def test_two_matching_files_decline_instead_of_raising_the_resolve_ambiguity(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        paths = {str(tmp_path / f"{stem}{CHAINDECLINE_READDOC_SUFFIX}") for stem in ("first", "second")}
        for path in paths:
            Path(path).write_text("chaindecline")
        dac = DataAccessCollection(files=paths)

        result = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())

        assert result is None
        assert list(rejection_window) == [ChainDeclineDocReader.get_class_name()]

    def test_str_route_still_matches_a_separator_shaped_basename(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """File basenames like '__main__.py' are legitimate feature names on the str route, so it never declines."""
        path = f"x{CHAINDECLINE_READDOC_SUFFIX}"

        result = ChainDeclineDocReader.match_subclass_data_access(path, ["__main__"], Options({}))

        assert result == path
        assert rejection_window == {}


class TestReadDocumentPinnedNameExemptsSeparatorGuard:
    def test_pinned_chain_shaped_name_resolves_via_the_pin(self, rejection_window: dict[str, MatchRejection]) -> None:
        path = f"pinned{CHAINDECLINE_READDOC_SUFFIX}"
        dac = DataAccessCollection(
            files={"chaindecline_readdoc_pin_handle": path},
            column_to_file={CHAINDECLINE_CHAIN_FEATURE: "chaindecline_readdoc_pin_handle"},
        )

        matched = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())

        assert matched == path
        assert rejection_window == {}

    def test_non_pinned_chain_shaped_name_still_declines_via_the_general_path(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        path = f"other{CHAINDECLINE_READDOC_SUFFIX}"
        dac = DataAccessCollection(
            files={"chaindecline_readdoc_other_handle": path},
            column_to_file={CHAINDECLINE_PLAIN_FEATURE: "chaindecline_readdoc_other_handle"},
        )

        matched = ChainDeclineDocReader.match_subclass_data_access(dac, [CHAINDECLINE_CHAIN_FEATURE], Options())

        assert matched is None
        assert list(rejection_window) == [ChainDeclineDocReader.get_class_name()]


class TestChainedFeatureResolvesToSingleFeatureGroupOverDocuments:
    def _accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return {ReadDocumentFeature: {PandasDataFrame}, ChainDeclineChainedFG: {PandasDataFrame}}

    def test_plain_name_still_resolves_to_document_group_only(self, tmp_path: Path) -> None:
        dac, _ = _document_dac("files", tmp_path, f"doc{CHAINDECLINE_READDOC_SUFFIX}")
        feature = Feature(name=CHAINDECLINE_PLAIN_FEATURE)

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None, dac)

        assert result.failure_kind is None
        assert result.identified == {ReadDocumentFeature: {PandasDataFrame}}

    def test_chain_shaped_name_resolves_to_chained_group_only(self, tmp_path: Path) -> None:
        dac, _ = _document_dac("files", tmp_path, f"doc{CHAINDECLINE_READDOC_SUFFIX}")
        feature = Feature(name=CHAINDECLINE_CHAIN_FEATURE)

        result = IdentifyFeatureGroupClass.evaluate(feature, self._accessible_plugins(), None, dac)

        assert result.failure_kind is None
        assert result.identified == {ChainDeclineChainedFG: {PandasDataFrame}}
        assert ReadDocumentFeature not in result.identified
