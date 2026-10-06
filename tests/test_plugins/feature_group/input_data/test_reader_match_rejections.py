"""Readers decline with an attributable reason (issue #727, cycles 2 and 3).

A reader that ESTABLISHED OWNERSHIP of an input (suffix match for files) but whose content rule fails must
record an attributable reason via ``record_match_rejection``; a plain non-match (wrong suffix) records nothing,
and so does a successful match. Databases: the database contract mixin.

Cycle 3: a reader records a ``MatchRejection`` stamped ``stage="input_data"``, the engine harvests
it into an ``input_data`` elimination and ``render_resolution_failure`` labels the near-miss line
``(input data)``, no longer ``(option value)``. A stage string the engine does not know falls back
to ``value_rejection`` without crashing the run.

All names carry a ``rej727`` marker and the file suffixes are globally unique: the readers and the
feature group defined here become global subclasses discovered process-wide, so under pytest-xdist
they must stay inert for every other test's inputs.
"""

from __future__ import annotations

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
from tests.helpers.suffix_file_reader import SuffixFileReader


FEATURE_NAME_REJ727 = "rej727_column"
BOGUS_STAGE_FEATURE_REJ727 = "rej727_bogus_stage_column"
BOGUS_STAGE_REASON_REJ727 = "custom decline"
FILE_SUFFIX_REJ727 = ".rej727csv"
NIE_SUFFIX_REJ727 = ".rej727nie"


class Rej727CsvHeaderReader(SuffixFileReader):
    """Reader owning the unique .rej727csv suffix; introspects the comma-separated header line."""

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


class Rej727NoIntrospectionReader(SuffixFileReader):
    """Reader owning the unique .rej727nie suffix; get_column_names stays NotImplementedError."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (NIE_SUFFIX_REJ727,)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {FEATURE_NAME_REJ727: [1]}


class Rej727FileFG(FeatureGroup):
    """Root feature group matching ONLY via its reader: no name rule claims rej727_column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Rej727CsvHeaderReader()

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
                compute_frameworks=[PandasDataFrame],
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
