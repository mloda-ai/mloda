"""ONLY an owned CONTENT decline gates the name rules; unrecorded non-matches and pin-then-match keep resolving."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import is_no_default
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    MatchRejection,
    record_match_rejection,
    restamp_match_rejection,
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


VG961_FILE_FEATURE = "vg961_file_column"
VG961_FILE_SUFFIX = ".vg961csv"

VG961_DB_FEATURE = "vg961_db_column"
VG961_DB_MARKER = "vg961_db_marker"

VG961_UNIT_OWNER = "vg961_unit_owner"
VG961_UNIT_REASON = "vg961 unit reason"


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


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


class TestRestampMatchRejection:
    """The helper replaces one owner's recording stage in place."""

    def test_no_active_window_is_a_no_op(self) -> None:
        """Without an open window the restamp neither raises nor opens one."""
        assert MATCH_REJECTION_REASONS.get() is None
        restamp_match_rejection(VG961_UNIT_OWNER, "input_data", "input_data_owned")
        assert MATCH_REJECTION_REASONS.get() is None

    def test_an_owner_without_a_recording_leaves_the_window_empty(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """An owner nothing was recorded for stays absent from the window."""
        restamp_match_rejection(VG961_UNIT_OWNER, "input_data", "input_data_owned")

        assert rejection_window == {}

    def test_a_recording_with_a_different_stage_keeps_its_stage(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The from_stage comparison is exact: a default-stage recording is not an input_data one."""
        record_match_rejection(VG961_UNIT_OWNER, VG961_UNIT_REASON)
        restamp_match_rejection(VG961_UNIT_OWNER, "input_data", "input_data_owned")

        assert rejection_window[VG961_UNIT_OWNER].stage == "value_rejection"

    def test_a_matching_owner_and_stage_is_restamped_with_the_reason_preserved(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A matching owner and from_stage replaces the recording with the same reason and the to_stage."""
        record_match_rejection(VG961_UNIT_OWNER, VG961_UNIT_REASON, stage="input_data")
        restamp_match_rejection(VG961_UNIT_OWNER, "input_data", "input_data_owned")

        rejection = rejection_window[VG961_UNIT_OWNER]
        assert rejection.reason == VG961_UNIT_REASON
        assert rejection.stage == "input_data_owned"


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


class TestModuleLeakPolicy:
    """The module's leak policy, machine-checked over every module-level final reader."""

    def test_module_level_readers_cannot_fire_on_foreign_options(self) -> None:
        """Every final reader owns vg961-marked suffixes or requires the vg961 credentials marker, never absence."""
        module_level = [
            cls for cls in get_all_subclasses(BaseInputData) if cls.__module__ == __name__ and cls.is_final_reader()
        ]

        assert module_level, "expected this module's final readers to be reachable through __subclasses__()"
        for cls in module_level:
            if issubclass(cls, ReadFile):
                assert all("vg961" in s for s in cls.suffix()), f"{cls.__name__} must own only vg961-marked suffixes"
            else:
                assert issubclass(cls, ReadDB), f"{cls.__name__} must be a vg961 file or db reader"
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
