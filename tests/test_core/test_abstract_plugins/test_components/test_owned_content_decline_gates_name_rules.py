"""ONLY an owned CONTENT decline gates the name rules, pinned or not; unrecorded non-matches keep resolving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.property_spec import is_no_default
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    MATCH_REJECTION_REASONS,
    MatchRejection,
    drop_match_rejections_since,
    match_rejection_owners,
    record_match_rejection,
    restamp_match_rejections_since,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.helpers.suffix_file_reader import SuffixFileReader


MODULE_SUFFIX_MARKERS = ("vg961", "vg1006", "vg1454")
"""Markers a module-level file reader's suffixes must carry, so none of them can fire on a foreign path."""

VG961_FILE_FEATURE = "vg961_file_column"
VG961_FILE_SUFFIX = ".vg961csv"

VG961_DB_MARKER = "vg961_db_marker"

VG1006_FILE_FEATURE = "vg1006_file_column"
VG1006_FILE_SUFFIX = ".vg1006csv"
VG1006_ALIAS_NAME = "vg1006_alias_access"

VG1006_UNIT_OWNER = "vg1006_unit_owner"
VG1006_UNIT_OTHER_OWNER = "vg1006_unit_other_owner"
VG1006_UNIT_REASON = "vg1006 unit reason"

VG1006_FOREIGN_OWNER = "vg1006_foreign_owner"
VG1006_FOREIGN_REASON = "vg1006 foreign reason"

VG1454_FILE_FEATURE = "vg1454_file_column"
VG1454_FILE_SUFFIX = ".vg1454csv"


class _CredentialFamily(BaseInputData):
    """Credential-matching reader base built directly on BaseInputData, inert without valid credentials."""

    @classmethod
    def wrap_feature_scoped_access(cls, data_access: Any) -> Any:
        if isinstance(data_access, dict) and not isinstance(data_access, RegisteredCredential):
            return RegisteredCredential(data_access)
        return data_access

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return False

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return True

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Any = None) -> Any:
        if isinstance(data_access, DataAccessCollection):
            candidates = list(data_access.credentials.values())
        elif isinstance(data_access, dict):
            candidates = [data_access]
        else:
            candidates = []
        for credentials in candidates:
            if not cls.is_valid_credentials(credentials):
                continue
            if cls.check_feature_in_data_access(feature_names[0], credentials):
                return credentials
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()} accepted the credentials but declined the feature '{feature_names[0]}'",
                stage=INPUT_DATA_STAGE,
            )
        return None


class Vg961CsvReader(SuffixFileReader):
    """Reader owning the unique .vg961csv suffix; introspects the comma-separated header line."""

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
        return Vg961CsvReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG961_FILE_FEATURE}


class Vg1006AliasReader(SuffixFileReader):
    """Reader addressed by an alias name, so it records its content decline under that name."""

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
        return Vg1006AliasReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG1006_FILE_FEATURE}


class Vg1454CsvReader(SuffixFileReader):
    """Reader owning the unique .vg1454csv suffix; introspects the comma-separated header line."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (VG1454_FILE_SUFFIX,)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        with open(file_name, encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG1454_FILE_FEATURE: [1]}


class Vg1454FileFG(FeatureGroup):
    """Root FG whose name rule claims vg1454_file_column while its addressed reader declines on content."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg1454CsvReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG1454_FILE_FEATURE}


VG1756_FEATURE = "vg1756_x1"


class Vg1756PinnedReader(_CredentialFamily):
    """Db reader accepting only its unique credentials, then declining every feature."""

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return isinstance(credentials, dict) and ("vg1756_pinned" in credentials or VG961_DB_MARKER in credentials)

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return False

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return [{VG1756_FEATURE: "from Vg1756PinnedReader"}]


class Vg1756SiblingReader(_CredentialFamily):
    """Db reader accepting only its unique credentials and only the unique feature."""

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return isinstance(credentials, dict) and ("vg1756_sibling" in credentials or VG961_DB_MARKER in credentials)

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return feature_name == VG1756_FEATURE

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return [{VG1756_FEATURE: "from Vg1756SiblingReader"}]


class Vg1756DbFG(FeatureGroup):
    """Root FG returning the accepting db reader, claiming vg1756_x1 by name."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg1756SiblingReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG1756_FEATURE}


class Vg1756PinnedDbFG(FeatureGroup):
    """Root FG returning the db reader that declines every feature, claiming vg1756_x1 by name."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg1756PinnedReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG1756_FEATURE}


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

    def test_drop_without_a_window_is_a_no_op(self) -> None:
        assert MATCH_REJECTION_REASONS.get() is None
        drop_match_rejections_since(frozenset())
        assert MATCH_REJECTION_REASONS.get() is None

    def test_drop_keeps_owners_in_the_snapshot(self, rejection_window: dict[str, MatchRejection]) -> None:
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_OWNED_STAGE)
        drop_match_rejections_since(match_rejection_owners())

        assert set(rejection_window) == {VG1006_UNIT_OWNER}

    def test_drop_removes_owners_recorded_after_the_snapshot(self, rejection_window: dict[str, MatchRejection]) -> None:
        record_match_rejection(VG1006_UNIT_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_OWNED_STAGE)
        known_owners = match_rejection_owners()
        record_match_rejection(VG1006_UNIT_OTHER_OWNER, VG1006_UNIT_REASON, stage=INPUT_DATA_OWNED_STAGE)
        drop_match_rejections_since(known_owners)

        assert set(rejection_window) == {VG1006_UNIT_OWNER}


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

        matched = Vg1006AliasReader.feature_scope_data_access(options, VG1006_FILE_FEATURE)

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


class TestUnownedPinGatesTheNameRule:
    """Engine level: every feature here is bare, so matching routes through global_scope_data_access ->
    match_data_access, never the by-name feature_scope_data_access."""

    def test_a_pin_no_registered_reader_owns_gates_the_name_rule(self, tmp_path: Path) -> None:
        """No reader anywhere owns the pinned suffix: eliminated, not recovered by the name rule."""
        path = tmp_path / "data.vg1454nobodyowns"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        dac = DataAccessCollection(files={"vg1454_h": str(path)}, column_to_file={VG1454_FILE_FEATURE: "vg1454_h"})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1454FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(Feature(name=VG1454_FILE_FEATURE), accessible_plugins, None, dac)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg1454FileFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert str(path) in elimination.reason
        assert "no registered reader" in elimination.reason

    def test_a_pin_owned_and_valid_still_binds_normally(self, tmp_path: Path) -> None:
        """The pinned file is owned and valid: the loop's own match wins, the post-loop check never fires."""
        path = tmp_path / f"data{VG1454_FILE_SUFFIX}"
        path.write_text(f"{VG1454_FILE_FEATURE}\n1\n", encoding="utf-8")
        dac = DataAccessCollection(files={"vg1454_h": str(path)}, column_to_file={VG1454_FILE_FEATURE: "vg1454_h"})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1454FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(Feature(name=VG1454_FILE_FEATURE), accessible_plugins, None, dac)

        assert Vg1454FileFG in result.identified
        assert result.eliminations == {}


class TestUnownedPinKeyDoesNotCollideWithTheCandidatesOwnKey:
    """Bug 3: MATCH_REJECTION_REASONS.setdefault means _record_unowned_pin's key must not collide with a
    key cls's own natural rejection machinery may already have written earlier in the same probe window."""

    def test_a_pre_seeded_rejection_under_the_candidates_own_key_does_not_swallow_the_unowned_pin_reason(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A plain-stage rejection already recorded under Vg1454CsvReader's own key, before the post-loop
        check runs (mirroring cls's own _reader_options_admit machinery having written there earlier in
        the same window), must not silently absorb the distinct "unowned pin" gating rejection
        _record_unowned_pin tries to record next under that same base key."""
        path = tmp_path / "data.vg1454nobodyowns"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        dac = DataAccessCollection(
            files={"vg1454_h": str(path)}, column_to_file={"vg1454_no_owner_feature": "vg1454_h"}
        )
        record_match_rejection(Vg1454CsvReader.data_access_name(), "unrelated earlier reason", stage=INPUT_DATA_STAGE)

        Vg1454CsvReader.match_data_access(["vg1454_no_owner_feature"], dac, options=Options({}))

        assert len(rejection_window) > 1
        assert rejection_window[Vg1454CsvReader.data_access_name()].reason == "unrelated earlier reason"
        assert any("no registered reader" in r.reason for r in rejection_window.values())


class TestOwnedShapesThatMustKeepResolving:
    """Owned shapes that bind normally must keep resolving."""

    def test_a_pin_owned_and_valid_still_binds_for_a_db_sibling(self) -> None:
        """A pinned db reader that accepts the credentials and the feature still binds."""
        feature = Feature(name=VG1756_FEATURE, options={Vg1756SiblingReader.__name__: {"vg1756_sibling": {}}})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1756DbFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg1756DbFG in result.identified
        assert result.eliminations == {}


class TestPinnedReaderDoesNotFallBackToGlobalRoute:
    """A pinned reader that declines must not fall back to the collection-wide route."""

    def test_a_pinned_non_match_without_a_recording_is_reported_and_gates(self, tmp_path: Path) -> None:
        """A wrong-suffix pinned path records no reason itself, so an owned rejection naming the reader is added."""
        path = tmp_path / "data.vg961other"
        path.write_text("vg961_other_a,vg961_other_b\n1,2\n", encoding="utf-8")
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: str(path)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg961FileFG not in result.identified
        elimination = result.eliminations.get(Vg961FileFG)
        assert elimination is not None
        assert Vg961CsvReader.get_class_name() in elimination.reason
        assert "BaseInputData" not in feature.options

    def test_a_pinned_db_reader_rejecting_the_credentials_without_a_recording_is_reported(self) -> None:
        """No recording and no match: the owned reason names the reader and never echoes the credentials."""
        credential_value = "vg1756_credential_value"
        feature = Feature(
            name=VG1756_FEATURE, options={Vg1756PinnedReader.__name__: {"vg1756_other": credential_value}}
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg1756PinnedDbFG: {PandasDataFrame}}
        dac = DataAccessCollection(credentials=[{"vg1756_sibling": {}}])

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, dac)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg1756PinnedDbFG)
        assert elimination is not None
        assert Vg1756PinnedReader.get_class_name() in elimination.reason
        assert credential_value not in elimination.reason
        assert "vg1756_other" not in elimination.reason

    def test_an_owned_decline_then_a_global_match_is_not_recovered(self, tmp_path: Path) -> None:
        """The pinned file declines with a recording; the other file in the collection must not rescue it."""
        path_a = tmp_path / f"a{VG961_FILE_SUFFIX}"
        path_a.write_text("vg961_other\n1\n", encoding="utf-8")
        path_b = tmp_path / f"b{VG961_FILE_SUFFIX}"
        path_b.write_text(f"{VG961_FILE_FEATURE}\n1\n", encoding="utf-8")
        feature = Feature(name=VG961_FILE_FEATURE, options={Vg961CsvReader.__name__: str(path_a)})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg961FileFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(
            feature, accessible_plugins, None, DataAccessCollection(files={str(path_b)})
        )

        assert Vg961FileFG not in result.identified
        elimination = result.eliminations.get(Vg961FileFG)
        assert elimination is not None
        assert Vg961CsvReader.get_class_name() in elimination.reason
        assert "lacks the column" in elimination.reason

    def test_run_all_raises_when_the_pinned_reader_declines(self) -> None:
        """End to end, the pinned decline fails resolution."""
        feature = Feature(VG1756_FEATURE, options={Vg1756PinnedReader.__name__: {"vg1756_pinned": {}}})

        with pytest.raises(FeatureResolutionError, match="Vg1756PinnedReader"):
            mloda.run_all(
                [feature],
                compute_frameworks=[PythonDictFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({Vg1756PinnedDbFG}),
            )


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


class TestReaderClassKeyNormalization:
    """A class-object option key must normalize like Options normalizes it, through data_access_name()."""

    def test_a_class_key_of_an_aliasing_reader_normalizes_to_its_alias(self) -> None:
        """The helper agrees with the Options normalization, so a class key still addresses the aliased reader."""
        assert BaseInputData.deal_with_base_input_data_name_as_cls_or_str(Vg1006AliasReader) == VG1006_ALIAS_NAME

    def test_a_class_key_of_a_non_aliasing_reader_stays_its_class_name(self) -> None:
        """A reader that does not override data_access_name() is unaffected by the normalization."""
        assert BaseInputData.deal_with_base_input_data_name_as_cls_or_str(Vg961CsvReader) == "Vg961CsvReader"


class TestModuleLeakPolicy:
    """The module's marker-based leak policy, machine-checked over every module-level reader."""

    def test_module_level_readers_cannot_fire_on_foreign_options(self) -> None:
        """Every reader owns marker-carrying suffixes or requires the module-unique credentials marker."""
        module_level = [
            cls for cls in get_all_subclasses(BaseInputData) if cls.__module__ == __name__ and "load_data" in vars(cls)
        ]

        assert module_level, "expected this module's readers to be reachable through __subclasses__()"
        for cls in module_level:
            if issubclass(cls, SuffixFileReader):
                assert all(any(marker in s for marker in MODULE_SUFFIX_MARKERS) for s in cls.suffix()), (
                    f"{cls.__name__} must own only suffixes carrying one of {MODULE_SUFFIX_MARKERS}"
                )
            elif issubclass(cls, _CredentialFamily):
                assert cls.is_valid_credentials({"vg961_foreign": "x"}) is False, (
                    f"{cls.__name__} must stay inert on foreign credentials"
                )
                assert cls.is_valid_credentials({VG961_DB_MARKER: "vg961"}) is True, (
                    f"{cls.__name__} must require its module-unique credentials marker"
                )
            else:
                assert cls.match_subclass_data_access({"vg961_foreign": "x"}, ["any"], Options()) is None, (
                    f"{cls.__name__} must stay inert on foreign credentials"
                )
            for key, spec in cls.reader_option_specs().items():
                if spec.framework_set:
                    continue
                assert not (is_no_default(spec.default) and spec.required_when is None), (
                    f"{cls.__name__}.READER_OPTIONS['{key}'] would fire on every foreign probe"
                )
