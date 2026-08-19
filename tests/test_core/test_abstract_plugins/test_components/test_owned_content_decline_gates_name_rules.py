"""ONLY an owned CONTENT decline gates the name rules; unrecorded non-matches and pin-then-match keep resolving."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.property_spec import is_no_default
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    MATCH_REJECTION_REASONS,
    MatchRejection,
    match_rejection_owners,
    record_match_rejection,
    restamp_match_rejections_since,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_file import ReadFile


MODULE_SUFFIX_MARKERS = ("vg961", "vg1006")
"""Markers a module-level file reader's suffixes must carry, so none of them can fire on a foreign path."""

VG961_FILE_FEATURE = "vg961_file_column"
VG961_FILE_SUFFIX = ".vg961csv"

VG961_DB_FEATURE = "vg961_db_column"
VG961_DB_MARKER = "vg961_db_marker"

VG1006_FILE_FEATURE = "vg1006_file_column"
VG1006_FILE_SUFFIX = ".vg1006csv"
VG1006_ALIAS_NAME = "vg1006_alias_access"

VG1006_UNIT_OWNER = "vg1006_unit_owner"
VG1006_UNIT_OTHER_OWNER = "vg1006_unit_other_owner"
VG1006_UNIT_REASON = "vg1006 unit reason"

VG1006_FOREIGN_OWNER = "vg1006_foreign_owner"
VG1006_FOREIGN_REASON = "vg1006 foreign reason"


class Vg961FileFamily(ReadFile):
    """Family base of the file shape; it overrides nothing, so it never classifies as final."""


class Vg961CsvReader(Vg961FileFamily):
    """Final reader owning the unique .vg961csv suffix; introspects the comma-separated header line."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (VG961_FILE_SUFFIX,)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        with open(file_name, encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG961_FILE_FEATURE: [1]}


class Vg961FileFG(FeatureGroup):
    """Root FG whose name rule claims vg961_file_column while its addressed reader declines on content."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg961FileFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG961_FILE_FEATURE}


class Vg961DbFamily(ReadDB):
    """Family base of the db shape; it overrides nothing, so it never classifies as final."""


class Vg961DbReader(Vg961DbFamily):
    """Final db reader accepting only vg961-marked credentials, then declining its feature on content."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG961_DB_FEATURE: [1]}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return VG961_DB_MARKER in credentials

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return False


class Vg961DbFG(FeatureGroup):
    """Root FG whose name rule claims vg961_db_column while its addressed db reader declines the feature."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg961DbFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG961_DB_FEATURE}


class Vg1006AliasFamily(ReadFile):
    """Family base of the aliased file shape; it overrides nothing, so it never classifies as final."""


class Vg1006AliasReader(Vg1006AliasFamily):
    """Final reader addressed by an alias name, so it records its content decline under that name."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (VG1006_FILE_SUFFIX,)

    @classmethod
    def data_access_name(cls) -> str:
        return VG1006_ALIAS_NAME

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        with open(file_name, encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        columns = cls.get_column_names(file_name)
        missing = [feature for feature in feature_names if feature not in columns]
        if not missing:
            return True
        record_match_rejection(
            cls.data_access_name(),
            f"{cls.data_access_name()} matched the suffix of {file_name} but it lacks the column(s): "
            f"{', '.join(missing)}",
            stage=INPUT_DATA_STAGE,
        )
        return False

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG1006_FILE_FEATURE: [1]}


class Vg1006AliasFG(FeatureGroup):
    """Root FG whose name rule claims vg1006_file_column while its aliased reader declines on content."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg1006AliasFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG1006_FILE_FEATURE}


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


class TestProbeScopedRestamp:
    """The snapshot and delta restamp promote only what one probe recorded."""

    def test_owners_outside_a_window_are_empty_and_open_none(self) -> None:
        """Without an open window the snapshot is empty and no window is opened."""
        assert MATCH_REJECTION_REASONS.get() is None
        assert match_rejection_owners() == frozenset()
        assert MATCH_REJECTION_REASONS.get() is None

    def test_owners_inside_a_window_are_the_recorded_names(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Inside a window the snapshot holds every recorded owner name."""
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)
        record_match_rejection(VG1006_UNIT_OTHER_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)

        assert match_rejection_owners() == frozenset({VG1006_UNIT_OWNER, VG1006_UNIT_OTHER_OWNER})

    def test_no_active_window_is_a_no_op(self) -> None:
        """Without an open window the restamp neither raises nor opens one."""
        assert MATCH_REJECTION_REASONS.get() is None
        restamp_match_rejections_since(frozenset(), INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)
        assert MATCH_REJECTION_REASONS.get() is None

    def test_an_owner_in_the_snapshot_keeps_its_stage(self, rejection_window: dict[str, MatchRejection]) -> None:
        """A recording that predates the snapshot is not part of the probe's delta."""
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)
        known_owners = match_rejection_owners()
        restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)

        assert rejection_window[VG1006_UNIT_OWNER].stage == INPUT_DATA_STAGE

    def test_a_delta_recording_is_restamped_with_the_reason_preserved(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """An owner recorded after the snapshot at the from_stage keeps its reason and takes the to_stage."""
        known_owners = match_rejection_owners()
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)
        restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)

        rejection = rejection_window[VG1006_UNIT_OWNER]
        assert rejection.reason == VG1006_UNIT_REASON
        assert rejection.stage == INPUT_DATA_OWNED_STAGE

    def test_a_delta_recording_with_a_different_stage_keeps_its_stage(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The from_stage comparison is exact: a default-stage delta recording is not an input_data one."""
        known_owners = match_rejection_owners()
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON)
        restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)

        assert rejection_window[VG1006_UNIT_OWNER].stage == "value_rejection"

    def test_only_the_recordings_after_the_snapshot_are_restamped(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """One window, two same-stage recordings: only the one the probe added is promoted."""
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)
        known_owners = match_rejection_owners()
        record_match_rejection(VG1006_UNIT_OTHER_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_STAGE)
        restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)

        assert rejection_window[VG1006_UNIT_OWNER].stage == INPUT_DATA_STAGE
        assert rejection_window[VG1006_UNIT_OTHER_OWNER].stage == INPUT_DATA_OWNED_STAGE


class TestProbeScopedRestampAtTheCallSite:
    """feature_scope_data_access snapshots the window before its probe, so it promotes only that probe's delta."""

    def test_a_recording_predating_the_probe_keeps_its_stage_while_the_probe_decline_is_owned(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """One window, one foreign recording seeded first: only the addressed reader's own decline is promoted."""
        path = tmp_path / f"data{VG1006_FILE_SUFFIX}"
        path.write_text("vg1006_other_a,vg1006_other_b\n1,2\n", encoding="utf-8")
        record_match_rejection(VG1006_FOREIGN_OWNER, VG1006_FOREIGN_REASON, stage=INPUT_DATA_STAGE)
        options = Options({VG1006_ALIAS_NAME: str(path)})

        matched = Vg1006AliasFamily.feature_scope_data_access(options, VG1006_FILE_FEATURE)

        assert matched is False
        assert rejection_window[VG1006_FOREIGN_OWNER].stage == INPUT_DATA_STAGE
        assert rejection_window[VG1006_ALIAS_NAME].stage == INPUT_DATA_OWNED_STAGE


class TestOwnedContentDeclineGatesNameRules:
    """Engine level, deliberately WITHOUT a window fixture: the engine owns the per-candidate window."""

    def test_an_owned_missing_column_decline_gates_the_name_rule(self, tmp_path: Path) -> None:
        """The addressed reader's suffix-owned file lacks the column: eliminated, not recovered by name."""
        path = tmp_path / f"data{VG961_FILE_SUFFIX}"
        path.write_text("vg961_other_a,vg961_other_b\n1,2\n", encoding="utf-8")
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: str(path)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg961FileFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert Vg961CsvReader.get_class_name() in elimination.reason
        assert "lacks the column" in elimination.reason

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - {Vg961FileFG.__name__} (input data): {elimination.reason}" in message

    def test_an_owned_db_feature_decline_gates_the_name_rule(self) -> None:
        """The addressed db reader accepts the credentials but declines the feature: eliminated too."""
        feature = Feature(name=VG961_DB_FEATURE, options={Vg961DbReader.__name__: {VG961_DB_MARKER: "vg961"}})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961DbFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg961DbFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert "declined" in elimination.reason
        assert VG961_DB_FEATURE in elimination.reason


class TestOwnedShapesThatMustNotGate:
    """The owned shapes without an eligible recording, or with a later match, must keep resolving."""

    def test_an_owned_plain_non_match_without_a_recording_does_not_gate(self, tmp_path: Path) -> None:
        """A wrong-suffix path never establishes ownership of the file, so the name rule still recovers."""
        path = tmp_path / "data.vg961other"
        path.write_text("vg961_other_a,vg961_other_b\n1,2\n", encoding="utf-8")
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: str(path)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg961FileFG in result.identified
        assert "BaseInputData" not in feature.options

    def test_an_owned_decline_on_one_file_with_a_match_on_another_still_pins_the_pair(self, tmp_path: Path) -> None:
        """The pinned file declines with a recording, then the resolve fallback matches the other file."""
        path_a = tmp_path / f"a{VG961_FILE_SUFFIX}"
        path_a.write_text("vg961_other\n1\n", encoding="utf-8")
        path_b = tmp_path / f"b{VG961_FILE_SUFFIX}"
        path_b.write_text(f"{VG961_FILE_FEATURE}\n1\n", encoding="utf-8")
        dac = DataAccessCollection(
            files={"vg961_a": str(path_a), "vg961_b": str(path_b)},
            column_to_file={VG961_FILE_FEATURE: "vg961_a"},
        )
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: dac})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg961FileFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (Vg961CsvReader, str(path_b))

    def test_an_owned_decline_then_a_global_match_still_pins_the_pair(self, tmp_path: Path) -> None:
        """The addressed file declines with a recording, then the global collection matches the other file."""
        path_a = tmp_path / f"a{VG961_FILE_SUFFIX}"
        path_a.write_text("vg961_other\n1\n", encoding="utf-8")
        path_b = tmp_path / f"b{VG961_FILE_SUFFIX}"
        path_b.write_text(f"{VG961_FILE_FEATURE}\n1\n", encoding="utf-8")
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: str(path_a)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(
            feature, accessible_plugins, None, DataAccessCollection(files={str(path_b)})
        )

        assert Vg961FileFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (Vg961CsvReader, str(path_b))


class TestAliasedDataAccessNameOwnership:
    """Ownership keys on data_access_name(), so a decline recorded under that alias must gate the name rules too."""

    def test_an_owned_decline_under_an_aliased_name_gates_the_name_rule(self, tmp_path: Path) -> None:
        """The reader addressed by its alias owns the suffix but lacks the column: eliminated, not recovered."""
        path = tmp_path / f"data{VG1006_FILE_SUFFIX}"
        path.write_text("vg1006_other_a,vg1006_other_b\n1,2\n", encoding="utf-8")
        feature = Feature(name=VG1006_FILE_FEATURE, options={VG1006_ALIAS_NAME: str(path)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1006AliasFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg1006AliasFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert VG1006_ALIAS_NAME in elimination.reason
        assert "lacks the column" in elimination.reason

    def test_an_aliased_decline_on_one_file_with_a_match_on_another_still_pins_the_pair(self, tmp_path: Path) -> None:
        """The pinned file declines with a recording, then the resolve fallback matches the other file."""
        path_a = tmp_path / f"a{VG1006_FILE_SUFFIX}"
        path_a.write_text("vg1006_other\n1\n", encoding="utf-8")
        path_b = tmp_path / f"b{VG1006_FILE_SUFFIX}"
        path_b.write_text(f"{VG1006_FILE_FEATURE}\n1\n", encoding="utf-8")
        dac = DataAccessCollection(
            files={"vg1006_a": str(path_a), "vg1006_b": str(path_b)},
            column_to_file={VG1006_FILE_FEATURE: "vg1006_a"},
        )
        feature = Feature(name=VG1006_FILE_FEATURE, options={VG1006_ALIAS_NAME: dac})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1006AliasFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg1006AliasFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (Vg1006AliasReader, str(path_b))


class TestReaderClassKeyNormalization:
    """A class-object option key must normalize like Options normalizes it, through data_access_name()."""

    def test_a_class_key_of_an_aliasing_reader_normalizes_to_its_alias(self) -> None:
        """The helper agrees with the Options normalization, so a class key still addresses the aliased reader."""
        assert BaseInputData.deal_with_base_input_data_name_as_cls_or_str(Vg1006AliasReader) == VG1006_ALIAS_NAME

    def test_a_class_key_of_a_non_aliasing_reader_stays_its_class_name(self) -> None:
        """A reader that does not override data_access_name() is unaffected by the normalization."""
        assert BaseInputData.deal_with_base_input_data_name_as_cls_or_str(Vg961CsvReader) == "Vg961CsvReader"


class TestModuleLeakPolicy:
    """The module's marker-based leak policy, machine-checked over every module-level final reader."""

    def test_module_level_readers_cannot_fire_on_foreign_options(self) -> None:
        """Every final reader owns marker-carrying suffixes or requires the module-unique credentials marker."""
        module_level = [
            cls for cls in get_all_subclasses(BaseInputData) if cls.__module__ == __name__ and cls.is_final_reader()
        ]

        assert module_level, "expected this module's final readers to be reachable through __subclasses__()"
        for cls in module_level:
            if issubclass(cls, ReadFile):
                assert all(any(marker in s for marker in MODULE_SUFFIX_MARKERS) for s in cls.suffix()), (
                    f"{cls.__name__} must own only suffixes carrying one of {MODULE_SUFFIX_MARKERS}"
                )
            else:
                assert issubclass(cls, ReadDB), f"{cls.__name__} must be a module-marked file or db reader"
                assert cls.is_valid_credentials({"vg961_foreign": "x"}) is False, (
                    f"{cls.__name__} must stay inert on foreign credentials"
                )
                assert cls.is_valid_credentials({VG961_DB_MARKER: "vg961"}) is True, (
                    f"{cls.__name__} must require its module-unique credentials marker"
                )
            for key, spec in cls.reader_option_specs().items():
                if spec.framework_set:
                    continue
                assert not (is_no_default(spec.default) and spec.required_when is None), (
                    f"{cls.__name__}.READER_OPTIONS['{key}'] would fire on every foreign probe"
                )
