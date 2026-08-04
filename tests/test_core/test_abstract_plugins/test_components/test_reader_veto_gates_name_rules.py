"""Pins issue #954, corrected: ONLY an OWNED reader veto (the user addressed the reader via its
``data_access_name()`` key and its READER_OPTIONS declined) gates the name-based OR rules of
``match_feature_group_criteria``; unowned global-probe declines, the MatchData rule, sibling probing,
input-data-free candidates and valid values keep resolving. Leaked vg954 fixtures stay foreign-inert."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    MatchRejection,
    has_match_rejection,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.input_data.read_file import ReadFile


VG954_FEATURE_NAME = "vg954_column"
VG954_PLAIN_FEATURE_NAME = "vg954_plain_column"
VG954_SIBLING_FEATURE_NAME = "vg954_sibling_column"
VG954_REQUIRED_FEATURE_NAME = "vg954_required_column"

VG954_GATE_ACCESS = "vg954_gate_access"
VG954_FORMAT_KEY = "vg954_format"

VG954_VETOED_ACCESS = "vg954_vetoed_access"
VG954_CLEAN_ACCESS = "vg954_clean_access"
VG954_SIBLING_KEY = "vg954_sibling_mode"

VG954_REQUIRED_ACCESS = "vg954_required_access"
VG954_REQUIRED_KEY = "vg954_required"

VG954_GLOBAL_FEATURE_NAME = "vg954_global_column"
VG954_GLOBAL_SUFFIX = ".vg954gcsv"

VG954_UNOWNED_FEATURE_NAME = "vg954_unowned_column"
VG954_UNOWNED_ACCESS = "vg954_unowned_access"
VG954_UNOWNED_HANDLE = "vg954_unowned_handle"
VG954_UNOWNED_KEY = "vg954_unowned_mode"

VG954_MATCHDATA_FEATURE_NAME = "vg954_matchdata_column"

VG954_UNIT_OWNER = "vg954_unit_owner"
VG954_UNIT_REASON = "vg954 unit reason"


class _Vg954MarkedReader(BaseInputData):
    """Family base: a child matches ONLY its own module-unique access string or dac folder handle."""

    VG954_ACCESS: ClassVar[str] = ""
    VG954_HANDLE: ClassVar[str] = ""

    @classmethod
    def match_subclass_data_access(
        cls, data_access: Any, feature_names: list[str], options: Options | None = None
    ) -> Any:
        if not cls.VG954_ACCESS:
            return None
        if isinstance(data_access, str) and data_access == cls.VG954_ACCESS:
            return data_access
        if isinstance(data_access, DataAccessCollection) and cls.VG954_HANDLE in data_access.folders:
            return cls.VG954_ACCESS
        return None


class Vg954GateFamily(_Vg954MarkedReader):
    """Scopes the probes of the gated-shape tests to exactly one final reader."""


class Vg954GateReader(Vg954GateFamily):
    """Final reader whose strict key vetoes the gated-shape bogus value; the default keeps it foreign-inert."""

    VG954_ACCESS = VG954_GATE_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        VG954_FORMAT_KEY: PropertySpec(
            "Strict membership key: it fires only on a PRESENT out-of-space value.",
            allowed_values=("vg954_ok",),
            strict_validation=True,
            default="vg954_ok",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG954_FEATURE_NAME: [1]}


class Vg954GatedFG(FeatureGroup):
    """Root FG whose name rule accepts ONLY vg954_column; the recorded OWNED veto must gate that rule."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg954GateFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG954_FEATURE_NAME}


class Vg954SiblingFamily(_Vg954MarkedReader):
    """Two-reader family: one spec-vetoed sibling plus one clean sibling, both addressed by name."""


class Vg954VetoedSiblingReader(Vg954SiblingFamily):
    """Final sibling whose strict key rejects the sibling test's supplied value."""

    VG954_ACCESS = VG954_VETOED_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        VG954_SIBLING_KEY: PropertySpec(
            "Strict key of the vetoed sibling; module-level safe because it declares a default.",
            allowed_values=("vg954_ok",),
            strict_validation=True,
            default="vg954_ok",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG954_SIBLING_FEATURE_NAME: [1]}


class Vg954CleanSiblingReader(Vg954SiblingFamily):
    """Final clean sibling; it must keep matching while its sibling is vetoed."""

    VG954_ACCESS = VG954_CLEAN_ACCESS

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG954_SIBLING_FEATURE_NAME: [1]}


class Vg954SiblingFG(FeatureGroup):
    """Root FG matching ONLY via its sibling reader family; unique names keep it inert elsewhere."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg954SiblingFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class Vg954PlainFG(FeatureGroup):
    """FG without input data: only its name rule claims vg954_plain_column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG954_PLAIN_FEATURE_NAME}


class Vg954GlobalFamily(ReadFile):
    """Family base of the unowned-content shape; it overrides nothing, so it never classifies as final."""


class Vg954GlobalCsvReader(Vg954GlobalFamily):
    """Final reader owning the unique .vg954gcsv suffix; introspects the comma-separated header line."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (VG954_GLOBAL_SUFFIX,)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        with open(file_name, encoding="utf-8") as handle:
            return handle.readline().strip().split(",")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG954_GLOBAL_FEATURE_NAME: [1]}


class Vg954GlobalDeclineFG(FeatureGroup):
    """Root FG whose name rule claims vg954_global_column while its reader family declines only globally."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg954GlobalFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG954_GLOBAL_FEATURE_NAME}


class Vg954UnownedFamily(_Vg954MarkedReader):
    """Scopes the unowned-strict-shape global probe to exactly one final reader."""


class Vg954UnownedReader(Vg954UnownedFamily):
    """Final reader claiming the unique dac folder handle; here its strict key vetoes only on the global probe."""

    VG954_ACCESS = VG954_UNOWNED_ACCESS
    VG954_HANDLE = VG954_UNOWNED_HANDLE

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        VG954_UNOWNED_KEY: PropertySpec(
            "Strict membership key with a default, so it is not absence-firing and stays module-level safe.",
            allowed_values=("vg954_ok",),
            strict_validation=True,
            default="vg954_ok",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {VG954_UNOWNED_FEATURE_NAME: [1]}


class Vg954UnownedFG(FeatureGroup):
    """Root FG whose name rule claims vg954_unowned_column; the UNOWNED veto must not gate it."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg954UnownedFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {VG954_UNOWNED_FEATURE_NAME}


class Vg954MatchDataFG(FeatureGroup, MatchData):
    """Root MatchData FG claiming ONLY vg954_matchdata_column through its matches hook, never by name."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return Vg954GateFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def matches(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return feature_name == VG954_MATCHDATA_FEATURE_NAME


def _vg954_required_setup(tag: str) -> tuple[type[BaseInputData], type[FeatureGroup]]:
    """(reader, feature group) with an unconditionally required key, built test-locally per the leak policy;
    the per-test tag keys data_access_name() so a traceback-kept stale twin is never the addressed reader."""

    class Vg954LocalRequiredFamily(_Vg954MarkedReader):
        """Scopes the solo probe to the one required-key reader."""

    class Vg954LocalRequiredReader(Vg954LocalRequiredFamily):
        VG954_ACCESS = VG954_REQUIRED_ACCESS

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            VG954_REQUIRED_KEY: PropertySpec("Unconditionally required: declares no default, no required_when."),
        }

        @classmethod
        def data_access_name(cls) -> str:
            return f"Vg954LocalRequiredReader_{tag}"

        @classmethod
        def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
            return {VG954_REQUIRED_FEATURE_NAME: [1]}

    class Vg954LocalRequiredFG(FeatureGroup):
        """Root FG whose name rule accepts ONLY vg954_required_column."""

        @classmethod
        def input_data(cls) -> BaseInputData | None:
            return Vg954LocalRequiredFamily()

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {VG954_REQUIRED_FEATURE_NAME}

    return Vg954LocalRequiredReader, Vg954LocalRequiredFG


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


class TestHasMatchRejection:
    """The helper reports whether the ACTIVE window holds a rejection with the exact stage."""

    def test_no_active_window_reports_false(self) -> None:
        """Without an open window nothing is recorded, so the helper answers False."""
        assert MATCH_REJECTION_REASONS.get() is None
        assert has_match_rejection("input_data") is False

    def test_a_window_holding_only_a_foreign_stage_reports_false(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The stage comparison is exact: a value_rejection recording is not an input_data one."""
        record_match_rejection(VG954_UNIT_OWNER, VG954_UNIT_REASON)

        assert has_match_rejection("input_data") is False

    def test_a_window_holding_an_input_data_rejection_reports_true(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """An input_data recording in the active window flips the helper to True."""
        record_match_rejection(VG954_UNIT_OWNER, VG954_UNIT_REASON, stage="input_data")

        assert has_match_rejection("input_data") is True


class TestReaderVetoGatesNameRules:
    """Engine level, deliberately WITHOUT a window fixture: the engine owns the per-candidate window."""

    def test_a_strict_value_veto_gates_the_name_rule(self) -> None:
        """The recorded OWNED strict-value veto must eliminate the candidate; the name rule must not recover it."""
        feature = Feature(
            name=VG954_FEATURE_NAME,
            options={Vg954GateReader.__name__: VG954_GATE_ACCESS, VG954_FORMAT_KEY: "vg954_bogus"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954GatedFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(Vg954GatedFG)
        assert elimination is not None
        assert elimination.stage == "input_data"
        assert Vg954GateReader.get_class_name() in elimination.reason
        assert VG954_FORMAT_KEY in elimination.reason
        assert "vg954_bogus" in elimination.reason

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - {Vg954GatedFG.__name__} (input data): {elimination.reason}" in message

    def test_an_absent_required_key_veto_gates_the_name_rule(self) -> None:
        """The recorded absence veto on the ownership path must eliminate the candidate too."""
        reader, feature_group = _vg954_required_setup("gate_absent")
        feature = Feature(
            name=VG954_REQUIRED_FEATURE_NAME,
            options={reader.data_access_name(): VG954_REQUIRED_ACCESS},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {feature_group: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        identified_names = sorted(fg.__name__ for fg in result.identified)
        elimination = result.eliminations.get(feature_group)
        stage = None if elimination is None else elimination.stage
        reason = "" if elimination is None else elimination.reason
        # A failing assert's traceback must not pin the test-local classes, or the conftest leak guard errors too.
        del reader, feature_group, accessible_plugins, result, elimination
        assert identified_names == []
        assert stage == "input_data"
        assert VG954_REQUIRED_KEY in reason


class TestUngatedShapesKeepResolving:
    """The shapes the gate must NOT touch: unowned vetoes, the MatchData rule, siblings, gate-free candidates."""

    def test_a_vetoed_sibling_still_lets_the_clean_sibling_match(self) -> None:
        """One vetoed sibling must not gate the family: the clean sibling wins and pins the pair."""
        feature = Feature(
            name=VG954_SIBLING_FEATURE_NAME,
            options={
                Vg954VetoedSiblingReader.__name__: VG954_VETOED_ACCESS,
                Vg954CleanSiblingReader.__name__: VG954_CLEAN_ACCESS,
                VG954_SIBLING_KEY: "vg954_bogus",
            },
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954SiblingFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg954SiblingFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (Vg954CleanSiblingReader, VG954_CLEAN_ACCESS)

    def test_an_unowned_global_content_decline_does_not_gate_the_name_rule(self, tmp_path: Path) -> None:
        """A content decline on the global probe, without the user addressing the reader, must not gate."""
        path = tmp_path / f"data{VG954_GLOBAL_SUFFIX}"
        path.write_text("other_a,other_b\n1,2\n", encoding="utf-8")
        dac = DataAccessCollection(files={str(path)})
        feature = Feature(name=VG954_GLOBAL_FEATURE_NAME)
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954GlobalDeclineFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, dac)

        assert Vg954GlobalDeclineFG in result.identified
        assert "BaseInputData" not in feature.options

    def test_an_unowned_strict_value_rejection_does_not_gate_the_name_rule(self) -> None:
        """A strict present-value rejection on the global probe, without ownership, must not gate."""
        feature = Feature(name=VG954_UNOWNED_FEATURE_NAME, options={VG954_UNOWNED_KEY: "vg954_bogus"})
        dac = DataAccessCollection(folders={VG954_UNOWNED_HANDLE: "/vg954/nowhere"})
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954UnownedFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, dac)

        assert Vg954UnownedFG in result.identified

    def test_an_owned_veto_leaves_the_match_data_rule_deciding(self) -> None:
        """An owned reader veto gates the name rules only; the MatchData rule must still identify."""
        feature = Feature(
            name=VG954_MATCHDATA_FEATURE_NAME,
            options={Vg954GateReader.__name__: VG954_GATE_ACCESS, VG954_FORMAT_KEY: "vg954_bogus"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954MatchDataFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg954MatchDataFG in result.identified

    def test_a_feature_group_without_input_data_keeps_resolving_by_name(self) -> None:
        """A sibling candidate's owned veto records in its own window and never gates a gate-free candidate."""
        feature = Feature(
            name=VG954_PLAIN_FEATURE_NAME,
            options={Vg954GateReader.__name__: VG954_GATE_ACCESS, VG954_FORMAT_KEY: "vg954_bogus"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            Vg954GatedFG: {PandasDataFrame},
            Vg954PlainFG: {PandasDataFrame},
        }

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        gated_elimination = result.eliminations.get(Vg954GatedFG)
        assert gated_elimination is not None
        assert gated_elimination.stage == "input_data"
        assert Vg954PlainFG in result.identified
        assert Vg954PlainFG not in result.eliminations

    def test_a_valid_strict_value_still_identifies_and_pins_the_pair(self) -> None:
        """Control: the allowed value keeps the gated-shape candidate identified with its reader pair."""
        feature = Feature(
            name=VG954_FEATURE_NAME,
            options={Vg954GateReader.__name__: VG954_GATE_ACCESS, VG954_FORMAT_KEY: "vg954_ok"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {Vg954GatedFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert Vg954GatedFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (Vg954GateReader, VG954_GATE_ACCESS)


class TestModuleLeakPolicy:
    """The module's leak policy, machine-checked over every module-level final reader."""

    def test_module_level_readers_cannot_fire_on_foreign_options(self) -> None:
        """Every module-level final reader carries its marker and no absence-firing spec; Vg954Local* are exempt."""
        module_level = [
            cls
            for cls in get_all_subclasses(BaseInputData)
            if cls.__module__ == __name__ and cls.is_final_reader() and "Local" not in cls.__name__
        ]

        assert module_level, "expected this module's final readers to be reachable through __subclasses__()"
        for cls in module_level:
            if issubclass(cls, ReadFile):
                assert all("vg954" in s for s in cls.suffix()), f"{cls.__name__} must own only vg954-marked suffixes"
            else:
                assert getattr(cls, "VG954_ACCESS", ""), f"{cls.__name__} must declare its module-unique access marker"
            for key, spec in cls.reader_option_specs().items():
                if spec.framework_set:
                    continue
                assert not (is_no_default(spec.default) and spec.required_when is None), (
                    f"{cls.__name__}.READER_OPTIONS['{key}'] would fire on every foreign probe"
                )
