"""Readers decline with an attributable reason (issue #727, cycles 2 and 3).

A reader that ESTABLISHED OWNERSHIP of an input (suffix match for files, valid credentials for
databases) but whose content rule fails must record an attributable reason via
``record_match_rejection``; a plain non-match (wrong suffix, unrecognized or invalid credentials)
records nothing, and so does a successful match.

Cycle 3: a reader records a ``MatchRejection`` stamped ``stage="input_data"``, the engine harvests
it into an ``input_data`` elimination and ``render_resolution_failure`` labels the near-miss line
``(input data)``, no longer ``(option value)``. A stage string the engine does not know falls back
to ``value_rejection`` without crashing the run.

All names carry a ``rej727`` marker and the file suffixes are globally unique: the readers and the
feature group defined here become global subclasses discovered process-wide, so under pytest-xdist
they must stay inert for every other test's inputs. The ReadDB doubles are additionally never final
readers (no load_data or hook overrides), so reader discovery never returns them at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.match_rejection import (
    record_match_rejection,
)
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.provider import BaseInputData, ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_file import ReadFile


FEATURE_NAME_REJ727 = "rej727_column"
BOGUS_STAGE_FEATURE_REJ727 = "rej727_bogus_stage_column"
BOGUS_STAGE_REASON_REJ727 = "custom decline"
FILE_SUFFIX_REJ727 = ".rej727csv"
NIE_SUFFIX_REJ727 = ".rej727nie"
DB_MARKER_KEY_REJ727 = "rej727db_marker"
DB_ACCESS_REJ727: dict[str, Any] = {DB_MARKER_KEY_REJ727: True}


class Rej727ReaderFamily(ReadFile):
    """Family base of this module's readers; it overrides nothing, so it never classifies as final."""


class Rej727CsvHeaderReader(Rej727ReaderFamily):
    """Final reader owning the unique .rej727csv suffix; introspects the comma-separated header line."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (FILE_SUFFIX_REJ727,)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        with open(file_name, encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {FEATURE_NAME_REJ727: [1]}


class Rej727NoIntrospectionReader(Rej727ReaderFamily):
    """Final reader owning the unique .rej727nie suffix; get_column_names stays NotImplementedError."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (NIE_SUFFIX_REJ727,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {FEATURE_NAME_REJ727: [1]}


class Rej727DecliningDbReader(ReadDB):
    """Recognizes only the marker credentials, then declines every feature at the content rule."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return credentials.get(DB_MARKER_KEY_REJ727) is True

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return False


class Rej727AgnosticDbReader(ReadDB):
    """Accepts the marker credentials, soft-declines unknown ones; cannot introspect features."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        if credentials.get(DB_MARKER_KEY_REJ727) is not True:
            raise NotImplementedError
        return True


class Rej727FileFG(FeatureGroup):
    """Root feature group matching ONLY via its reader family: no name rule claims rej727_column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Rej727ReaderFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class Rej727BogusStageFG(FeatureGroup):
    """Declines its single gated feature with an unknown custom stage; inert for every other name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != BOGUS_STAGE_FEATURE_REJ727:
            return False
        record_match_rejection(cls.get_class_name(), BOGUS_STAGE_REASON_REJ727, stage="rej727_bogus_stage")
        return False


class Rej727NetSpend(FeatureGroup):
    """Requested consumer of rej727_line_value; inert for every other name."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("rej727_line_value")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rej727_net_spend"}


class Rej727LineValue(FeatureGroup):
    """Middle consumer whose input is the unresolvable rej727_column; inert for every other name."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(FEATURE_NAME_REJ727)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"rej727_line_value"}


def _write_csv(path: Path, header: str) -> str:
    """Write a two-line comma-separated file and return its path as a string."""
    path.write_text(f"{header}\n1,2\n", encoding="utf-8")
    return str(path)


class TestReadFileMatchRejections:
    """Unit level: the suffix match establishes ownership, validate_columns judges the content."""

    def test_missing_column_after_suffix_match_records_attributable_reason(
        self, tmp_path: Path, rejection_window: dict[str, Any]
    ) -> None:
        """Right suffix but the header lacks the feature: one input_data rejection naming reader, file, feature."""
        file_path = _write_csv(tmp_path / f"data{FILE_SUFFIX_REJ727}", "other_a,other_b")

        matched = Rej727CsvHeaderReader.match_subclass_data_access(file_path, [FEATURE_NAME_REJ727], Options())

        assert matched is None
        assert list(rejection_window) == [Rej727CsvHeaderReader.get_class_name()]
        stored = rejection_window[Rej727CsvHeaderReader.get_class_name()]
        assert stored.stage == "input_data"
        assert Rej727CsvHeaderReader.get_class_name() in stored.reason
        assert file_path in stored.reason
        assert FEATURE_NAME_REJ727 in stored.reason

    def test_wrong_suffix_is_a_plain_non_match_and_records_nothing(
        self, tmp_path: Path, rejection_window: dict[str, Any]
    ) -> None:
        """Wrong suffix: ownership was never established, so even a missing column records nothing."""
        file_path = _write_csv(tmp_path / "data.txt", "other_a,other_b")

        matched = Rej727CsvHeaderReader.match_subclass_data_access(file_path, [FEATURE_NAME_REJ727], Options())

        assert matched is None
        assert rejection_window == {}

    def test_successful_match_records_nothing(self, tmp_path: Path, rejection_window: dict[str, Any]) -> None:
        """Right suffix and the header contains the feature: the path matches and nothing is recorded."""
        file_path = _write_csv(tmp_path / f"data{FILE_SUFFIX_REJ727}", f"{FEATURE_NAME_REJ727},other_b")

        matched = Rej727CsvHeaderReader.match_subclass_data_access(file_path, [FEATURE_NAME_REJ727], Options())

        assert matched == file_path
        assert rejection_window == {}

    def test_not_implemented_get_column_names_assumes_columns_and_records_nothing(
        self, tmp_path: Path, rejection_window: dict[str, Any]
    ) -> None:
        """A reader without column introspection keeps assuming the columns are there, silently."""
        file_path = _write_csv(tmp_path / f"data{NIE_SUFFIX_REJ727}", "other_a,other_b")

        assert Rej727NoIntrospectionReader.validate_columns(file_path, [FEATURE_NAME_REJ727]) is True
        assert rejection_window == {}


class TestPinnedFileMatchRejection:
    """The pinned-file path (_resolve_pinned_file) also judges content only after ownership is established."""

    def test_pinned_file_missing_column_records_attributable_reason(
        self, tmp_path: Path, rejection_window: dict[str, Any]
    ) -> None:
        """A pin routes the feature to one suffix-owned file; that file lacking the column is an input_data decline."""
        file_path = _write_csv(tmp_path / f"pinned{FILE_SUFFIX_REJ727}", "other_a,other_b")
        dac = DataAccessCollection(
            files={"rej727_pin_handle": file_path},
            column_to_file={FEATURE_NAME_REJ727: "rej727_pin_handle"},
        )

        matched = Rej727CsvHeaderReader.match_subclass_data_access(dac, [FEATURE_NAME_REJ727], Options())

        assert matched is None
        assert list(rejection_window) == [Rej727CsvHeaderReader.get_class_name()]
        stored = rejection_window[Rej727CsvHeaderReader.get_class_name()]
        assert stored.stage == "input_data"
        assert Rej727CsvHeaderReader.get_class_name() in stored.reason
        assert file_path in stored.reason
        assert FEATURE_NAME_REJ727 in stored.reason


class TestReadDbMatchRejections:
    """Unit level: valid credentials establish ownership, check_feature_in_data_access is the content rule."""

    def test_declined_feature_with_valid_credentials_records_attributable_reason(
        self, rejection_window: dict[str, Any]
    ) -> None:
        """Ownership established, feature declined: one input_data rejection naming reader and feature."""
        matched = Rej727DecliningDbReader.match_read_db_data_access([DB_ACCESS_REJ727], [FEATURE_NAME_REJ727])

        assert matched is None
        assert list(rejection_window) == [Rej727DecliningDbReader.get_class_name()]
        stored = rejection_window[Rej727DecliningDbReader.get_class_name()]
        assert stored.stage == "input_data"
        assert Rej727DecliningDbReader.get_class_name() in stored.reason
        assert FEATURE_NAME_REJ727 in stored.reason

    def test_not_implemented_feature_check_matches_and_records_nothing(self, rejection_window: dict[str, Any]) -> None:
        """A reader that cannot introspect features keeps matching on credentials alone, silently."""
        matched = Rej727AgnosticDbReader.match_read_db_data_access([DB_ACCESS_REJ727], [FEATURE_NAME_REJ727])

        assert matched is DB_ACCESS_REJ727
        assert rejection_window == {}

    def test_not_implemented_credentials_check_records_nothing(self, rejection_window: dict[str, Any]) -> None:
        """NotImplementedError from is_valid_credentials is a soft no-match: no ownership, no recording."""
        matched = Rej727AgnosticDbReader.match_read_db_data_access([{"rej727_unrelated": 1}], [FEATURE_NAME_REJ727])

        assert matched is None
        assert rejection_window == {}

    def test_invalid_credentials_record_nothing(self, rejection_window: dict[str, Any]) -> None:
        """is_valid_credentials returning False is a plain non-match: no ownership, no recording."""
        matched = Rej727DecliningDbReader.match_read_db_data_access([{"rej727_unrelated": 1}], [FEATURE_NAME_REJ727])

        assert matched is None
        assert rejection_window == {}


CLAIM1753_PREFIX = "rej1753_"
CLAIM1753_KEY = "rej1753db_marker"
CLAIM1753_HANDLE = "rej1753_handle"
CLAIM1753_OWN_NAME = "rej1753_own_column"
CLAIM1753_FOREIGN_NAME = "other1753_column"


class Rej1753ClaimingDbReader(ReadDB):
    """Final reader whose credential slot can be malformed; claims only rej1753_ feature names."""

    probe_calls: list[Any] = []

    @classmethod
    def claims_feature_name(cls, feature_name: str) -> bool:
        return feature_name.startswith(CLAIM1753_PREFIX)

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        if CLAIM1753_KEY not in credentials:
            return False
        cls.probe_calls.append(credentials)
        if credentials[CLAIM1753_KEY] is not True:
            record_match_rejection(cls.get_class_name(), "malformed rej1753 credential slot", stage="input_data")
            return False
        return True

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return True

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {CLAIM1753_OWN_NAME: [1]}


@pytest.fixture()
def claim1753_calls() -> Iterator[list[Any]]:
    Rej1753ClaimingDbReader.probe_calls.clear()
    yield Rej1753ClaimingDbReader.probe_calls
    Rej1753ClaimingDbReader.probe_calls.clear()


def _malformed_dac() -> DataAccessCollection:
    return DataAccessCollection(credentials={CLAIM1753_HANDLE: {CLAIM1753_KEY: "malformed"}})


class TestReadDbClaimsFeatureName:
    """An unclaimed feature name is a silent non-match before any credential probe."""

    def test_default_claims_every_feature_name(self) -> None:
        assert ReadDB.claims_feature_name("anything") is True

    @pytest.mark.parametrize(
        ("data_access", "options"),
        [
            pytest.param(_malformed_dac(), Options(), id="collection"),
            pytest.param(_malformed_dac(), Options({"data_access_handle": CLAIM1753_HANDLE}), id="handle_hint"),
            pytest.param({CLAIM1753_KEY: "malformed"}, Options(), id="plain_dict"),
        ],
    )
    def test_unclaimed_name_skips_credential_probe(
        self, rejection_window: dict[str, Any], claim1753_calls: list[Any], data_access: Any, options: Options
    ) -> None:
        matched = Rej1753ClaimingDbReader.match_subclass_data_access(data_access, [CLAIM1753_FOREIGN_NAME], options)

        assert matched is None
        assert claim1753_calls == []
        assert rejection_window == {}

    def test_claimed_name_still_probes_and_records_malformed_slot(
        self, rejection_window: dict[str, Any], claim1753_calls: list[Any]
    ) -> None:
        matched = Rej1753ClaimingDbReader.match_subclass_data_access(_malformed_dac(), [CLAIM1753_OWN_NAME], Options())

        assert matched is None
        assert len(claim1753_calls) >= 1
        assert list(rejection_window) == [Rej1753ClaimingDbReader.get_class_name()]

    def test_claimed_name_with_valid_credentials_matches(
        self, rejection_window: dict[str, Any], claim1753_calls: list[Any]
    ) -> None:
        dac = DataAccessCollection(credentials={CLAIM1753_HANDLE: {CLAIM1753_KEY: True}})

        matched = Rej1753ClaimingDbReader.match_subclass_data_access(dac, [CLAIM1753_OWN_NAME], Options())

        assert matched is not None
        assert len(claim1753_calls) >= 1
        assert rejection_window == {}


class TestEngineHarvestsReaderRejection:
    """Engine integration, deliberately WITHOUT a window fixture: the engine owns the per-candidate window."""

    def test_wrong_header_becomes_an_input_data_elimination_and_renders(self, tmp_path: Path) -> None:
        """The reader's recorded decline surfaces as an input_data elimination and an '(input data)' near-miss line."""
        file_path = _write_csv(tmp_path / f"engine{FILE_SUFFIX_REJ727}", "other_a,other_b")
        dac = DataAccessCollection(files={file_path})
        feature = Feature(name=FEATURE_NAME_REJ727)
        accessible_plugins: FeatureGroupEnvironmentMapping = {Rej727FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, dac)

        assert result.identified == {}
        elimination = result.eliminations.get(Rej727FileFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert Rej727CsvHeaderReader.get_class_name() in elimination.reason
        assert file_path in elimination.reason

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - {Rej727FileFG.__name__} (input data): {elimination.reason}" in message

    def test_nested_input_names_its_chain_and_drops_the_did_you_mean_hint(self, tmp_path: Path) -> None:
        """An unresolvable input two levels down names the chain and skips the hint beside an input_data decline."""
        file_path = _write_csv(tmp_path / f"nested{FILE_SUFFIX_REJ727}", "other_a,other_b")

        with pytest.raises(FeatureResolutionError) as excinfo:
            mloda.prepare(
                ["rej727_net_spend"],
                compute_frameworks={PandasDataFrame},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {Rej727FileFG, Rej727NetSpend, Rej727LineValue}
                ),
                data_access_collection=DataAccessCollection(files={file_path}),
            )

        lines = str(excinfo.value).split("\n")
        assert lines[0] == (
            f"No feature groups found for feature name: '{FEATURE_NAME_REJ727}' "
            "(needed by rej727_net_spend -> rej727_line_value)."
        )
        assert any(line.startswith(f"  - {Rej727FileFG.__name__} (input data):") for line in lines)
        assert not any(line.startswith("Did you mean") for line in lines)

    def test_matching_header_identifies_and_pins_the_reader_pair(self, tmp_path: Path) -> None:
        """A content-passing file identifies the group and stores the (reader class, path) pair, unchanged."""
        file_path = _write_csv(tmp_path / f"engine_ok{FILE_SUFFIX_REJ727}", f"{FEATURE_NAME_REJ727},other_b")
        dac = DataAccessCollection(files={file_path})
        feature = Feature(name=FEATURE_NAME_REJ727)
        accessible_plugins: FeatureGroupEnvironmentMapping = {Rej727FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, dac)

        assert Rej727FileFG in result.identified
        assert result.eliminations == {}
        assert feature.input_data_match == (Rej727CsvHeaderReader, file_path)


class TestEngineFallsBackOnUnknownStage:
    """The neutral channel does not validate stages; the engine maps unknown ones to value_rejection."""

    def test_unknown_recorded_stage_falls_back_to_value_rejection(self) -> None:
        """A custom stage string neither crashes the run nor leaks into the elimination stage."""
        feature = Feature(name=BOGUS_STAGE_FEATURE_REJ727)
        accessible_plugins: FeatureGroupEnvironmentMapping = {Rej727BogusStageFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Rej727BogusStageFG)
        assert elimination is not None
        assert elimination.stage == "value_rejection"
        assert elimination.reason == BOGUS_STAGE_REASON_REJ727
