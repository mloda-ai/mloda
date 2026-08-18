"""Pins selection-time enforcement of ``READER_OPTIONS`` specs (issue #949, cycle 2): per-candidate
vetoes before the probe; absence vetoes record only on the feature-scope (ownership) path.
Leak policy: leaked final readers are foreign-inert (TestModuleLeakPolicy pins it); absence-firing ones are test-local.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import (
    PropertySpec,
    is_no_default,
    is_positive_int,
)
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    MATCH_REJECTION_REASONS,
    MatchRejection,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


ROE_FEATURE_NAME = "roe_enforced_column"

ROE_STRICT_ACCESS = "roe_strict_access"
ROE_STRICT_HANDLE = "roe_strict_handle"
ROE_FORMAT_KEY = "roe_format"
ROE_CHAR_KEY = "roe_char_mode"

ROE_SCALAR_ACCESS = "roe_scalar_access"
ROE_SCALAR_KEY = "roe_scalar"

ROE_VALIDATOR_ACCESS = "roe_validator_access"
ROE_COUNT_KEY = "roe_count"
ROE_TOUCHY_KEY = "roe_touchy"
ROE_BOOM_VALUE = "roe_boom"

ROE_LENIENT_ACCESS = "roe_lenient_access"
ROE_LAX_KEY = "roe_lax"
ROE_DEFAULTED_KEY = "roe_defaulted"
ROE_NONE_DEFAULT_KEY = "roe_none_defaulted"

ROE_COND_ACCESS = "roe_conditional_access"
ROE_COND_HANDLE = "roe_conditional_handle"
ROE_COND_KEY = "roe_conditional"
ROE_TRIGGER_KEY = "roe_trigger"

ROE_FUSSY_ACCESS = "roe_fussy_access"
ROE_FUSSY_KEY = "roe_fussy"

ROE_PROBE_ACCESS = "roe_probe_access"
ROE_PROBE_KEY = "roe_probe"

ROE_PAIR_HANDLE = "roe_pair_handle"
ROE_PAIR_KEY = "roe_pair"
ROE_PAIR_VETOED_ACCESS = "roe_pair_vetoed_access"
ROE_PAIR_CLEAN_ACCESS = "roe_pair_clean_access"

ROE_REQUIRED_ACCESS = "roe_required_access"
ROE_REQUIRED_HANDLE = "roe_required_handle"
ROE_REQUIRED_KEY = "roe_required"
ROE_NULLABLE_ACCESS = "roe_nullable_access"
ROE_NULLABLE_KEY = "roe_nullable"

ROE_ENGINE_FEATURE = "roe_engine_column"
ROE_ENGINE_ACCESS = "roe_engine_access"
ROE_ENGINE_KEY = "roe_engine_format"


def _roe_trigger_required(options: Any) -> bool:
    """Conditionally-required predicate keyed to the module-unique trigger, so it is foreign-inert."""
    return options is not None and options.get(ROE_TRIGGER_KEY) is True


def _roe_raising_predicate(options: Any) -> bool:
    """A required_when predicate that cannot judge: the candidate must become a silent non-match."""
    raise RuntimeError("roe required_when predicate crash")


def _roe_raising_validator(value: Any) -> Any:
    """Accepts everything except the module-unique trigger value, on which it raises instead of judging."""
    if value == ROE_BOOM_VALUE:
        raise RuntimeError("roe element_validator crash")
    return True


class _RoeMarkedReader(BaseInputData):
    """Family base: a child matches ONLY its own module-unique access string or dac folder handle."""

    ROE_ACCESS: ClassVar[str] = ""
    ROE_HANDLE: ClassVar[str] = ""

    @classmethod
    def match_subclass_data_access(
        cls, data_access: Any, feature_names: list[str], options: Options | None = None
    ) -> Any:
        if not cls.ROE_ACCESS:
            return None
        if isinstance(data_access, str) and data_access == cls.ROE_ACCESS:
            return data_access
        if isinstance(data_access, DataAccessCollection) and cls.ROE_HANDLE in data_access.folders:
            return cls.ROE_ACCESS
        return None


class RoeStrictFamily(_RoeMarkedReader):
    """Scopes the global probes of the strict-values reader to exactly that one candidate."""


class RoeStrictValuesReader(RoeStrictFamily):
    """Final reader with two strict membership keys; both fire only when their unique key is present."""

    ROE_ACCESS = ROE_STRICT_ACCESS
    ROE_HANDLE = ROE_STRICT_HANDLE

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_FORMAT_KEY: PropertySpec(
            "Strict membership over a Mapping value space, so an unhashable value is a clean rejection.",
            allowed_values={"roe_csv": "comma separated", "roe_parquet": "columnar"},
            strict_validation=True,
            default="roe_csv",
        ),
        ROE_CHAR_KEY: PropertySpec(
            "Strict membership over single characters, pinning that a str value is one scalar.",
            allowed_values=("x", "y", "z"),
            strict_validation=True,
            default="x",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeScalarOnlyReader(_RoeMarkedReader):
    """Final reader whose key rejects a list/tuple/set/frozenset value outright, never unpacked (#1154)."""

    ROE_ACCESS = ROE_SCALAR_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_SCALAR_KEY: PropertySpec(
            "Scalar-only strict key: a container value is rejected before any element-wise check.",
            element_validator=is_positive_int,
            strict_validation=True,
            scalar_only=True,
            default=3,
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeValidatorReader(_RoeMarkedReader):
    """Final reader with validator-backed strict keys: one replacing membership, one raising."""

    ROE_ACCESS = ROE_VALIDATOR_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_COUNT_KEY: PropertySpec(
            "The declared element_validator REPLACES the also-declared membership space.",
            allowed_values=("roe_member",),
            element_validator=is_positive_int,
            strict_validation=True,
            default=3,
        ),
        ROE_TOUCHY_KEY: PropertySpec(
            "A validator that raises on the trigger value; the raise must stay contained.",
            element_validator=_roe_raising_validator,
            strict_validation=True,
            default=None,
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeLenientReader(_RoeMarkedReader):
    """Final reader whose keys never reject: non-strict values and declared defaults (None included)."""

    ROE_ACCESS = ROE_LENIENT_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_LAX_KEY: PropertySpec(
            "Non-strict despite a declared value space; a present value is never validated.",
            allowed_values=("roe_only",),
            default="roe_only",
        ),
        ROE_DEFAULTED_KEY: PropertySpec("A declared default keeps the absent key optional.", default="roe_fallback"),
        ROE_NONE_DEFAULT_KEY: PropertySpec("A declared None default is optional too.", default=None),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeCondFamily(_RoeMarkedReader):
    """Scopes the global probes of the conditional reader to exactly that one candidate."""


class RoeConditionalReader(RoeCondFamily):
    """Final reader whose key is required only when the module-unique trigger option is supplied."""

    ROE_ACCESS = ROE_COND_ACCESS
    ROE_HANDLE = ROE_COND_HANDLE

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_COND_KEY: PropertySpec("Required iff the trigger key is supplied.", required_when=_roe_trigger_required),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeRaisingPredicateReader(_RoeMarkedReader):
    """Final reader whose required_when predicate always raises: a silent non-match, never a crash."""

    ROE_ACCESS = ROE_FUSSY_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_FUSSY_KEY: PropertySpec("Requiredness undecidable by design.", required_when=_roe_raising_predicate),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeProbeCountingReader(_RoeMarkedReader):
    """Final reader instrumenting its probe, pinning that the veto pre-empts match_subclass_data_access."""

    ROE_ACCESS = ROE_PROBE_ACCESS

    roe_calls: ClassVar[list[Any]] = []

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_PROBE_KEY: PropertySpec(
            "Strict key of the instrumented reader.",
            allowed_values=("roe_ok",),
            strict_validation=True,
            default="roe_ok",
        ),
    }

    @classmethod
    def match_subclass_data_access(
        cls, data_access: Any, feature_names: list[str], options: Options | None = None
    ) -> Any:
        cls.roe_calls.append(data_access)
        return super().match_subclass_data_access(data_access, feature_names, options)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoePairFamily(_RoeMarkedReader):
    """Two-candidate family: a spec-vetoed reader plus a clean sibling on the SAME dac handle."""


class RoePairVetoedFamily(RoePairFamily):
    """Narrower family whose only final reader is the vetoed one, for a deterministic solo probe."""


class RoePairVetoedReader(RoePairVetoedFamily):
    """Final reader whose strict key rejects the pair test's supplied value."""

    ROE_ACCESS = ROE_PAIR_VETOED_ACCESS
    ROE_HANDLE = ROE_PAIR_HANDLE

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_PAIR_KEY: PropertySpec(
            "Strict key shared by the pair scenario.",
            allowed_values=("roe_ok",),
            strict_validation=True,
            default="roe_ok",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoePairCleanReader(RoePairFamily):
    """Final sibling declaring no extra specs; it must win once the vetoed sibling is a non-match."""

    ROE_ACCESS = ROE_PAIR_CLEAN_ACCESS
    ROE_HANDLE = ROE_PAIR_HANDLE

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_FEATURE_NAME: [1]}


class RoeEngineFamily(_RoeMarkedReader):
    """Reader family the engine feature group binds; only its one final child is probed."""


class RoeEngineReader(RoeEngineFamily):
    """Final reader behind the engine integration tests."""

    ROE_ACCESS = ROE_ENGINE_ACCESS

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        ROE_ENGINE_KEY: PropertySpec(
            "Strict key whose veto must surface as an input_data elimination.",
            allowed_values=("roe_ok",),
            strict_validation=True,
            default="roe_ok",
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {ROE_ENGINE_FEATURE: [1]}


class RoeEnforcementFG(FeatureGroup):
    """Root feature group matching ONLY via its reader family; unique names keep it inert elsewhere."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return RoeEngineFamily()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _roe_required_family(tag: str) -> tuple[type[BaseInputData], type[BaseInputData]]:
    """(family, reader) with an unconditionally required key, built test-locally per the leak policy;
    the per-test tag keys data_access_name() so a traceback-kept stale twin is never the addressed reader."""

    class RoeLocalRequiredFamily(_RoeMarkedReader):
        """Scopes the solo global probe to the one required-key reader."""

    class RoeLocalRequiredReader(RoeLocalRequiredFamily):
        ROE_ACCESS = ROE_REQUIRED_ACCESS
        ROE_HANDLE = ROE_REQUIRED_HANDLE

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            ROE_REQUIRED_KEY: PropertySpec("Unconditionally required: declares no default, no required_when."),
        }

        @classmethod
        def data_access_name(cls) -> str:
            return f"RoeLocalRequiredReader_{tag}"

        @classmethod
        def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
            return {ROE_FEATURE_NAME: [1]}

    return RoeLocalRequiredFamily, RoeLocalRequiredReader


def _roe_nullable_reader(tag: str) -> type[BaseInputData]:
    """A required key opting into explicit None, built test-locally per the leak policy."""

    class RoeLocalNullableReader(_RoeMarkedReader):
        ROE_ACCESS = ROE_NULLABLE_ACCESS

        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            ROE_NULLABLE_KEY: PropertySpec(
                "Required, but an explicit None counts as present.", allow_explicit_none=True
            ),
        }

        @classmethod
        def data_access_name(cls) -> str:
            return f"RoeLocalNullableReader_{tag}"

        @classmethod
        def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
            return {ROE_FEATURE_NAME: [1]}

    return RoeLocalNullableReader


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one selection call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


@pytest.fixture()
def collect_after() -> Iterator[None]:
    """Reclaim test-local RoeLocal* readers out of __subclasses__ before the next test on this worker."""
    yield
    gc.collect()


class TestFeatureScopeStrictValues:
    """Strict value validation on the feature-scope path, for PRESENT keys only."""

    def test_bad_membership_value_is_a_non_match_and_records_the_reason(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A strict-rejected value vetoes the addressed reader and records class, key and value."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: "roe_bogus"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        assert "BaseInputData" not in options
        owner = RoeStrictValuesReader.get_class_name()
        assert list(rejection_window) == [owner]
        stored = rejection_window[owner]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert owner in stored.reason
        assert ROE_FORMAT_KEY in stored.reason
        assert "roe_bogus" in stored.reason

    def test_valid_membership_value_still_matches_and_pins_the_pair(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: a valid value matches, writes the reader pair, and records nothing."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: "roe_parquet"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert options.get("BaseInputData") == (RoeStrictValuesReader, ROE_STRICT_ACCESS)
        assert rejection_window == {}

    @pytest.mark.parametrize(
        "container",
        [
            ["roe_csv", "roe_bogus"],
            ("roe_csv", "roe_bogus"),
            {"roe_csv", "roe_bogus"},
            frozenset({"roe_csv", "roe_bogus"}),
        ],
        ids=["list", "tuple", "set", "frozenset"],
    )
    def test_container_values_validate_element_wise(
        self, container: Any, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """One bad element in any sequence container rejects the whole value."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: container})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeStrictValuesReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_FORMAT_KEY in stored.reason

    def test_all_valid_container_elements_still_match(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: a container of only in-space elements passes element-wise validation."""
        options = Options(
            {RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: ("roe_csv", "roe_parquet")}
        )

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}

    def test_a_str_value_is_one_scalar_not_a_char_sequence(self, rejection_window: dict[str, MatchRejection]) -> None:
        """The value "xyz" over allowed {"x","y","z"} must reject: char-wise iteration would accept it."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_CHAR_KEY: "xyz"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeStrictValuesReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_CHAR_KEY in stored.reason
        assert "xyz" in stored.reason

    def test_a_dict_value_is_one_composite_value_and_rejects_cleanly(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """An unhashable composite can never be a member of the Mapping space: rejection, not TypeError."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: {"roe_csv": 1}})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeStrictValuesReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_FORMAT_KEY in stored.reason

    def test_explicit_none_on_a_flagless_strict_spec_reads_absent(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: without allow_explicit_none a supplied None is absent, so nothing is validated."""
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: None})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}

    def test_non_strict_present_values_are_never_validated(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: a value outside a NON-strict spec's declared space keeps matching byte-identically."""
        options = Options({RoeLenientReader.__name__: ROE_LENIENT_ACCESS, ROE_LAX_KEY: "roe_bogus"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert options.get("BaseInputData") == (RoeLenientReader, ROE_LENIENT_ACCESS)
        assert rejection_window == {}

    def test_the_framework_written_pair_is_never_validated(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: the framework tuple under "BaseInputData" is exempt on both paths."""
        pair = (RoeStrictValuesReader, ROE_STRICT_ACCESS)
        options = Options(group={"BaseInputData": pair, RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS})

        assert BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME) is True
        assert options.get("BaseInputData") == pair
        assert rejection_window == {}

        global_options = Options(group={"BaseInputData": pair})
        dac = DataAccessCollection(folders={ROE_STRICT_HANDLE: "/roe/nowhere"})
        assert RoeStrictFamily.match_data_access([ROE_FEATURE_NAME], dac, options=global_options) == (
            RoeStrictValuesReader,
            ROE_STRICT_ACCESS,
        )
        assert rejection_window == {}


class TestScalarOnlyRejectsCollectionsOutright:
    """``scalar_only=True`` rejects a list/tuple/set/frozenset value BEFORE any element-wise check (#1154)."""

    @pytest.mark.parametrize(
        "container",
        [
            [3, 5],
            (3, 5),
            {3, 5},
            frozenset({3, 5}),
        ],
        ids=["list", "tuple", "set", "frozenset"],
    )
    def test_every_element_valid_container_is_still_rejected_outright(
        self, container: Any, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Every element individually passes the validator, yet scalar_only rejects the shape itself."""
        options = Options({RoeScalarOnlyReader.__name__: ROE_SCALAR_ACCESS, ROE_SCALAR_KEY: container})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeScalarOnlyReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_SCALAR_KEY in stored.reason
        assert "scalar_only" in stored.reason

    def test_a_valid_scalar_still_matches(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: a plain scalar value the validator accepts still matches."""
        options = Options({RoeScalarOnlyReader.__name__: ROE_SCALAR_ACCESS, ROE_SCALAR_KEY: 5})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}

    def test_an_invalid_scalar_is_still_rejected_by_the_ordinary_path(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: scalar_only changes nothing for a plain scalar value the validator rejects."""
        options = Options({RoeScalarOnlyReader.__name__: ROE_SCALAR_ACCESS, ROE_SCALAR_KEY: -1})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeScalarOnlyReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_SCALAR_KEY in stored.reason


class TestElementValidator:
    """A declared element_validator replaces membership; its raise stays contained."""

    def test_element_validator_replaces_membership_for_an_accepting_verdict(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: a value outside allowed_values passes because only the validator decides."""
        options = Options({RoeValidatorReader.__name__: ROE_VALIDATOR_ACCESS, ROE_COUNT_KEY: 7})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert options.get("BaseInputData") == (RoeValidatorReader, ROE_VALIDATOR_ACCESS)
        assert rejection_window == {}

    def test_element_validator_replaces_membership_for_a_rejecting_verdict(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A member of allowed_values still rejects when the validator says no."""
        options = Options({RoeValidatorReader.__name__: ROE_VALIDATOR_ACCESS, ROE_COUNT_KEY: "roe_member"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeValidatorReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert RoeValidatorReader.get_class_name() in stored.reason
        assert ROE_COUNT_KEY in stored.reason
        assert "roe_member" in stored.reason

    def test_a_raising_element_validator_rejects_the_value_without_escaping(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A validator that raises cannot judge the value, so the value is rejected, not the run."""
        options = Options({RoeValidatorReader.__name__: ROE_VALIDATOR_ACCESS, ROE_TOUCHY_KEY: ROE_BOOM_VALUE})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[RoeValidatorReader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_TOUCHY_KEY in stored.reason


class TestRequiredness:
    """Requiredness for ABSENT keys on the feature-scope path, where the addressed reader RECORDS."""

    def test_conditionally_required_key_absent_with_trigger_is_rejected(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A truthy required_when over an absent key records a rejection naming class and key."""
        options = Options({RoeConditionalReader.__name__: ROE_COND_ACCESS, ROE_TRIGGER_KEY: True})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        assert "BaseInputData" not in options
        owner = RoeConditionalReader.get_class_name()
        assert list(rejection_window) == [owner]
        stored = rejection_window[owner]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert owner in stored.reason
        assert ROE_COND_KEY in stored.reason

    def test_conditionally_required_key_absent_without_trigger_stays_optional(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: a falsy predicate makes the absent key simply optional."""
        options = Options({RoeConditionalReader.__name__: ROE_COND_ACCESS})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert options.get("BaseInputData") == (RoeConditionalReader, ROE_COND_ACCESS)
        assert rejection_window == {}

    def test_conditionally_required_key_present_satisfies_the_requirement(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: requiredness gates ABSENT keys only; a present key satisfies it."""
        options = Options(
            {RoeConditionalReader.__name__: ROE_COND_ACCESS, ROE_TRIGGER_KEY: True, ROE_COND_KEY: "roe_supplied"}
        )

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}

    def test_a_raising_required_when_predicate_is_a_silent_non_match(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A predicate that raises makes the reader a non-match WITHOUT a recorded rejection."""
        options = Options({RoeRaisingPredicateReader.__name__: ROE_FUSSY_ACCESS})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        assert "BaseInputData" not in options
        assert rejection_window == {}

    def test_unconditionally_required_key_absent_is_rejected(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """A NO_DEFAULT spec without required_when makes absence a RECORDED rejection: ownership is established."""
        _, reader = _roe_required_family("absent")
        options = Options({reader.data_access_name(): ROE_REQUIRED_ACCESS})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        owner = reader.get_class_name()
        assert list(rejection_window) == [owner]
        stored = rejection_window[owner]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert owner in stored.reason
        assert ROE_REQUIRED_KEY in stored.reason

    def test_unconditionally_required_key_present_matches(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """Control: supplying the required key restores the match and the pair write."""
        _, reader = _roe_required_family("present")
        options = Options({reader.data_access_name(): ROE_REQUIRED_ACCESS, ROE_REQUIRED_KEY: "roe_supplied"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert options.get("BaseInputData") == (reader, ROE_REQUIRED_ACCESS)
        assert rejection_window == {}

    def test_an_opted_in_explicit_none_satisfies_requiredness(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """Control: on an allow_explicit_none spec a supplied None counts as present."""
        reader = _roe_nullable_reader("supplied")
        options = Options({reader.data_access_name(): ROE_NULLABLE_ACCESS, ROE_NULLABLE_KEY: None})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}

    def test_an_opted_in_required_key_absent_is_still_rejected(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """A truly absent opted-in key still fails requiredness."""
        reader = _roe_nullable_reader("absent")
        options = Options({reader.data_access_name(): ROE_NULLABLE_ACCESS})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        stored = rejection_window[reader.get_class_name()]
        assert stored.stage == INPUT_DATA_OWNED_STAGE
        assert ROE_NULLABLE_KEY in stored.reason

    def test_declared_defaults_keep_absent_keys_optional(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: declared defaults, a None default included, never reject an absent key."""
        options = Options({RoeLenientReader.__name__: ROE_LENIENT_ACCESS})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert rejection_window == {}


class TestGlobalProbeEnforcement:
    """The same VETOES run on match_data_access (options may be None); only present-value rejections record."""

    def test_bad_strict_value_rejects_the_candidate_on_the_global_probe(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A strict-rejected PRESENT value vetoes AND records on the global probe too."""
        options = Options({ROE_FORMAT_KEY: "roe_bogus"})
        dac = DataAccessCollection(folders={ROE_STRICT_HANDLE: "/roe/nowhere"})

        result = RoeStrictFamily.match_data_access([ROE_FEATURE_NAME], dac, options=options)

        assert result == (None, None)
        owner = RoeStrictValuesReader.get_class_name()
        assert list(rejection_window) == [owner]
        stored = rejection_window[owner]
        assert stored.stage == INPUT_DATA_STAGE
        assert owner in stored.reason
        assert ROE_FORMAT_KEY in stored.reason
        assert "roe_bogus" in stored.reason

    def test_none_options_read_every_key_absent(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """options=None reads as all-absent: the unconditional key still vetoes, silently."""
        family, _ = _roe_required_family("none_options")
        dac = DataAccessCollection(folders={ROE_REQUIRED_HANDLE: "/roe/nowhere"})

        result = family.match_data_access([ROE_FEATURE_NAME], dac, options=None)

        assert result == (None, None)
        assert rejection_window == {}

    def test_an_absent_required_key_is_a_silent_veto_on_the_global_probe(
        self, rejection_window: dict[str, MatchRejection], collect_after: None
    ) -> None:
        """Supplied Options missing the required key: the candidate vetoes, the window stays empty."""
        family, _ = _roe_required_family("global_absent")
        dac = DataAccessCollection(folders={ROE_REQUIRED_HANDLE: "/roe/nowhere"})

        result = family.match_data_access([ROE_FEATURE_NAME], dac, options=Options())

        assert result == (None, None)
        assert rejection_window == {}

    def test_a_firing_required_when_is_a_silent_veto_on_the_global_probe(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """A firing required_when over an absent key is the same silent veto as NO_DEFAULT absence."""
        options = Options({ROE_TRIGGER_KEY: True})
        dac = DataAccessCollection(folders={ROE_COND_HANDLE: "/roe/nowhere"})

        result = RoeCondFamily.match_data_access([ROE_FEATURE_NAME], dac, options=options)

        assert result == (None, None)
        assert rejection_window == {}

    def test_a_dormant_required_when_still_matches_on_the_global_probe(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: without the trigger the conditional candidate claims the collection cleanly."""
        dac = DataAccessCollection(folders={ROE_COND_HANDLE: "/roe/nowhere"})

        result = RoeCondFamily.match_data_access([ROE_FEATURE_NAME], dac, options=Options())

        assert result == (RoeConditionalReader, ROE_COND_ACCESS)
        assert rejection_window == {}

    def test_a_vetoed_candidate_leaves_sibling_candidates_probing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The veto is per candidate: probed solo it is a plain non-match, and its clean sibling wins."""
        options = Options({ROE_PAIR_KEY: "roe_bogus"})
        dac = DataAccessCollection(folders={ROE_PAIR_HANDLE: "/roe/nowhere"})

        solo = RoePairVetoedFamily.match_data_access([ROE_FEATURE_NAME], dac, options=options)
        assert solo == (None, None)
        stored = rejection_window[RoePairVetoedReader.get_class_name()]
        assert stored.stage == INPUT_DATA_STAGE

        paired = RoePairFamily.match_data_access([ROE_FEATURE_NAME], dac, options=options)
        assert paired == (RoePairCleanReader, ROE_PAIR_CLEAN_ACCESS)
        assert set(rejection_window) <= {RoePairVetoedReader.get_class_name()}

    def test_valid_options_still_match_on_the_global_probe(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: a valid strict value keeps the global probe matching, recording nothing."""
        options = Options({ROE_FORMAT_KEY: "roe_csv"})
        dac = DataAccessCollection(folders={ROE_STRICT_HANDLE: "/roe/nowhere"})

        result = RoeStrictFamily.match_data_access([ROE_FEATURE_NAME], dac, options=options)

        assert result == (RoeStrictValuesReader, ROE_STRICT_ACCESS)
        assert rejection_window == {}

    def test_none_options_with_declared_defaults_still_match(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Control: all-absent keys with declared defaults pass the checks even for options=None."""
        dac = DataAccessCollection(folders={ROE_STRICT_HANDLE: "/roe/nowhere"})

        result = RoeStrictFamily.match_data_access([ROE_FEATURE_NAME], dac, options=None)

        assert result == (RoeStrictValuesReader, ROE_STRICT_ACCESS)
        assert rejection_window == {}


class TestEnforcementPrecedesTheProbe:
    """The spec check runs BEFORE the candidate's match_subclass_data_access."""

    def test_the_veto_preempts_match_subclass_data_access(self, rejection_window: dict[str, MatchRejection]) -> None:
        """A vetoed candidate's own probe is never called, even though it would have matched."""
        RoeProbeCountingReader.roe_calls.clear()
        options = Options({RoeProbeCountingReader.__name__: ROE_PROBE_ACCESS, ROE_PROBE_KEY: "roe_bogus"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert RoeProbeCountingReader.roe_calls == []
        assert matched is False
        assert list(rejection_window) == [RoeProbeCountingReader.get_class_name()]

    def test_a_passing_check_hands_over_to_match_subclass_data_access(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Control: a passing check runs the probe exactly once and the match stands."""
        RoeProbeCountingReader.roe_calls.clear()
        options = Options({RoeProbeCountingReader.__name__: ROE_PROBE_ACCESS, ROE_PROBE_KEY: "roe_ok"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is True
        assert RoeProbeCountingReader.roe_calls == [ROE_PROBE_ACCESS]
        assert rejection_window == {}


class TestOutsideAnActiveWindow:
    """The rejection channel is a no-op without an open window; enforcement itself still applies."""

    def test_enforcement_records_nothing_and_raises_nothing_without_a_window(self) -> None:
        """No window open: the veto still makes a non-match, and nothing is recorded or raised."""
        assert MATCH_REJECTION_REASONS.get() is None
        options = Options({RoeStrictValuesReader.__name__: ROE_STRICT_ACCESS, ROE_FORMAT_KEY: "roe_bogus"})

        matched = BaseInputData.feature_scope_data_access(options, ROE_FEATURE_NAME)

        assert matched is False
        assert MATCH_REJECTION_REASONS.get() is None


class TestEngineIntegration:
    """Deliberately WITHOUT a window fixture: the engine owns the per-candidate window."""

    def test_a_spec_veto_surfaces_as_an_input_data_elimination(self) -> None:
        """The declared-spec rejection harvests exactly like a hand-rolled input_data decline (PR #948)."""
        feature = Feature(
            name=ROE_ENGINE_FEATURE,
            options={RoeEngineReader.__name__: ROE_ENGINE_ACCESS, ROE_ENGINE_KEY: "roe_bogus"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {RoeEnforcementFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert result.identified == {}
        elimination = result.eliminations.get(RoeEnforcementFG)
        assert elimination is not None
        assert elimination.stage == INPUT_DATA_STAGE
        assert RoeEngineReader.get_class_name() in elimination.reason
        assert ROE_ENGINE_KEY in elimination.reason
        assert "roe_bogus" in elimination.reason

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - {RoeEnforcementFG.__name__} (input data): {elimination.reason}" in message

    def test_a_valid_value_identifies_and_pins_the_reader_pair(self) -> None:
        """Control: a valid value identifies the group and stores the (reader class, access) pair."""
        feature = Feature(
            name=ROE_ENGINE_FEATURE,
            options={RoeEngineReader.__name__: ROE_ENGINE_ACCESS, ROE_ENGINE_KEY: "roe_ok"},
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {RoeEnforcementFG: {PandasDataFrame}}

        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

        assert RoeEnforcementFG in result.identified
        assert result.eliminations == {}
        assert feature.options.get("BaseInputData") == (RoeEngineReader, ROE_ENGINE_ACCESS)


class TestModuleLeakPolicy:
    """The module's leak policy, machine-checked over every module-level final reader."""

    def test_module_level_readers_cannot_fire_on_foreign_options(self) -> None:
        """Every module-level final reader carries its marker and no absence-firing spec; RoeLocal* are exempt."""
        module_level = [
            cls
            for cls in get_all_subclasses(BaseInputData)
            if cls.__module__ == __name__ and cls.is_final_reader() and "Local" not in cls.__name__
        ]

        assert module_level, "expected this module's final readers to be reachable through __subclasses__()"
        for cls in module_level:
            assert getattr(cls, "ROE_ACCESS", ""), f"{cls.__name__} must declare its module-unique access marker"
            for key, spec in cls.reader_option_specs().items():
                if spec.framework_set:
                    continue
                assert not (is_no_default(spec.default) and spec.required_when is None), (
                    f"{cls.__name__}.READER_OPTIONS['{key}'] would fire on every foreign probe"
                )
