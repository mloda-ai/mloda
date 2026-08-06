"""Every gate GlobalFilter.identify_matched_filters closes records WHY, as a stage plus a reason, and the
unmatched-filter warning names each filter's nearest miss from those facts. Probe classes live inside factory
functions and are dropped before any assert runs, so a failing assert never pins a throwaway FeatureGroup and
trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    record_match_rejection,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import _STAGE_LABELS, _render_near_miss_block
from mloda.core.prepare.resolution_types import Elimination, EliminationStage, EvaluationResult
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, SingleFilter
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


GF_LOGGER_NAME = "mloda.core.filter.global_filter"

HOST_FEATURE = "fer_host_feat"  # the resolved feature the filters are matched against
FILTER_FEATURE = "fer_filter_feat"  # the declared filter feature every probe is asked about

PLAIN_CLASS_NAME = "FerPlainFG"
MATCHER_ERROR_CLASS_NAME = "FerMatcherErrorFG"
STAGE_DECLINE_CLASS_NAME = "FerStageDeclineFG"
GROUP_DOMAIN_CLASS_NAME = "FerGroupDomainFG"
CAPABILITY_REJECT_CLASS_NAME = "FerCapabilityRejectFG"
SCOPE_TIE_A_CLASS_NAME = "FerScopeTieAFG"
SCOPE_TIE_B_CLASS_NAME = "FerScopeTieBFG"

MISSING_SCOPE = "FerNoSuchScope"  # a scope string naming no accessible class
SCOPE_REASON = "outside the requested feature group scope"
UNMATCHED_PHRASE = "matched no feature group"
NEAREST_MISS_PHRASE = "Nearest miss: "
FALSY_REPORT_FRAGMENT = "falsy non-bool"
BARE_MESSAGE = f"Filter feature '{FILTER_FEATURE}' matched no feature group."

RUNTIME_MESSAGE = "fer_runtime_boom"
RUNTIME_TYPE_NAME = "RuntimeError"
DEFECT_MESSAGE = "fer_defect_after_decline"
DECLINE_REASON = "fer_decline_reason"
UNKNOWN_STAGE = "fer_unknown_stage_hint"  # a free-form hint no engine stage knows

GROUP_DOMAIN = "fer_group_domain"  # declared by the group-domain probe
FEATURE_DOMAIN = "fer_feature_domain"  # carried by the resolved host feature
OTHER_DOMAIN = "fer_other_domain"  # declared by the filter feature; matches neither

MATCHER_ERROR_STAGE: EliminationStage = "matcher_error"
VALUE_REJECTION_STAGE: EliminationStage = "value_rejection"
INPUT_DATA_ELIMINATION_STAGE: EliminationStage = "input_data"
DOMAIN_STAGE: EliminationStage = "domain"
SCOPE_STAGE: EliminationStage = "scope"
CAPABILITY_STAGE: EliminationStage = "capability"
FRAMEWORK_PIN_STAGE: EliminationStage = "framework_pin"

# The canonical seam's own wording over the one framework the filter would ride.
CAPABILITY_REASON = f"supports_compute_framework rejected {[PythonDictFramework.__name__]}"

# (recorded free-form hint, the elimination stage it maps onto).
STAGE_HINT_TABLE: tuple[tuple[str, EliminationStage], ...] = (
    (INPUT_DATA_STAGE, INPUT_DATA_ELIMINATION_STAGE),
    (INPUT_DATA_OWNED_STAGE, INPUT_DATA_ELIMINATION_STAGE),
    (VALUE_REJECTION_STAGE, VALUE_REJECTION_STAGE),
    (UNKNOWN_STAGE, VALUE_REJECTION_STAGE),
)
STAGE_HINT_IDS = [hint for hint, _ in STAGE_HINT_TABLE]

T = TypeVar("T")

_Factory = Callable[[], type[FeatureGroup]]


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> tuple[str, ...]:
    """Formatted messages GlobalFilter logged at exactly that level."""
    records = [record for record in caplog.records if record.name == GF_LOGGER_NAME and record.levelno == level]
    return tuple(record.getMessage() for record in records)


def _stage_reason(stage: str) -> str:
    """The reason text the stage-recording probe stores for one recorded hint."""
    return f"fer_stage_reason_{stage}"


def _fact_of(recorded: Any) -> tuple[str, str]:
    """(stage, reason) of one recorded fact. Any, because the ledger's value type is what this suite pins."""
    return str(recorded.stage), str(recorded.reason)


def _ledger_rows(global_filter: GlobalFilter) -> tuple[tuple[str, str, str, str], ...]:
    """(group class name, filter feature name, stage, reason) per recorded fact, sorted. Holds no class."""
    rows: list[tuple[str, str, str, str]] = []
    for key, recorded in global_filter.dropped_filters.items():
        stage, reason = _fact_of(recorded)
        rows.append((key[0].get_class_name(), key[1], stage, reason))
    return tuple(sorted(rows))


def _filter_feature(
    domain: str | None = None,
    scope: str | None = None,
    pin: type[ComputeFramework] | None = None,
) -> Feature:
    """The module's filter feature, declared with the given domain, scope and single framework pin."""
    feature = Feature(FILTER_FEATURE, domain=domain, feature_group=scope)
    if pin is not None:
        feature.compute_frameworks = {pin}
    return feature


def _host_feature(domain: str | None = None, pin: type[ComputeFramework] | None = None) -> Feature:
    """The resolved feature the filters are matched against."""
    feature = Feature(HOST_FEATURE, domain=domain)
    if pin is not None:
        feature.compute_frameworks = {pin}
    return feature


def _plain_host() -> Feature:
    """The host carrying neither a domain nor a pin."""
    return _host_feature()


def _make_plain_fg() -> type[FeatureGroup]:
    """A throwaway group matching both module names through the default criteria."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class FerPlainFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

    return FerPlainFG


def _make_scope_tie_a_fg() -> type[FeatureGroup]:
    """A plain matcher whose class name sorts before its twin's."""
    gc.collect()

    class FerScopeTieAFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

    return FerScopeTieAFG


def _make_scope_tie_b_fg() -> type[FeatureGroup]:
    """A plain matcher whose class name sorts after its twin's."""
    gc.collect()

    class FerScopeTieBFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

    return FerScopeTieBFG


def _make_matcher_error_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook raises a plain RuntimeError for the filter feature."""
    gc.collect()

    class FerMatcherErrorFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                raise RuntimeError(RUNTIME_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return FerMatcherErrorFG


def _make_stage_decline_fg(stage: str) -> type[FeatureGroup]:
    """A throwaway group whose hook records under the caller's stage hint and declines the filter feature."""
    gc.collect()

    class FerStageDeclineFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, _stage_reason(stage), stage=stage)
                return False
            return str(feature_name) in cls.feature_names_supported()

    return FerStageDeclineFG


def _make_group_domain_fg() -> type[FeatureGroup]:
    """A plain matcher declaring a non-default group domain."""
    gc.collect()

    class FerGroupDomainFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def get_domain(cls) -> Domain:
            return Domain(GROUP_DOMAIN)

    return FerGroupDomainFG


def _make_capability_reject_fg() -> type[FeatureGroup]:
    """A plain matcher whose capability hook rejects every framework."""
    gc.collect()

    class FerCapabilityRejectFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def supports_compute_framework(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            compute_framework: type[ComputeFramework],
        ) -> bool:
            return False

    return FerCapabilityRejectFG


def _make_plain_decline_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook returns a literal False for the filter feature, recording nothing."""
    gc.collect()

    class FerPlainDeclineFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                return False
            return str(feature_name) in cls.feature_names_supported()

    return FerPlainDeclineFG


def _make_falsy_decline_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook returns None for the filter feature, recording nothing."""
    gc.collect()

    class FerFalsyDeclineFG(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> Any:  # Any, not bool: the falsy non-bool return is the case under test.
            if str(feature_name) == FILTER_FEATURE:
                return None
            return str(feature_name) in cls.feature_names_supported()

    return FerFalsyDeclineFG


def _make_decline_then_defect_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook declines with a recorded reason once, then raises on the next ask."""
    gc.collect()

    class FerDeclineThenDefectFG(FeatureGroup):
        declined: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                if not cls.declined:
                    cls.declined = True
                    record_match_rejection(cls.__name__, DECLINE_REASON)
                    return False
                raise RuntimeError(DEFECT_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return FerDeclineThenDefectFG


def _make_defect_then_match_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook raises once, then matches the filter feature on every later ask."""
    gc.collect()

    class FerDefectThenMatchFG(FeatureGroup):
        raised: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE and not cls.raised:
                cls.raised = True
                raise RuntimeError(RUNTIME_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return FerDefectThenMatchFG


def _make_match_then_decline_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook matches the filter feature once, then declines with a recorded reason."""
    gc.collect()

    class FerMatchThenDeclineFG(FeatureGroup):
        matched: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                if not cls.matched:
                    cls.matched = True
                    return True
                record_match_rejection(cls.__name__, DECLINE_REASON)
                return False
            return str(feature_name) in cls.feature_names_supported()

    return FerMatchThenDeclineFG


@dataclass(frozen=True)
class _LedgerSnapshot:
    """Plain-data readout of one matching pass. Holds no class, no filter and no recorded fact object."""

    escaped: str | None
    names: tuple[str, ...]
    fact_types: tuple[str, ...]
    # Set when reading a stored fact raised, which is how a bare-string ledger reports itself.
    ledger_error: str | None
    rows: tuple[tuple[str, str, str, str], ...]
    warnings: tuple[str, ...]
    debugs: tuple[str, ...]
    unmatched: tuple[str, ...]


def _drive_matching(
    makes: Sequence[_Factory],
    caplog: pytest.LogCaptureFixture,
    filter_feature: Feature | str = FILTER_FEATURE,
    make_host: Callable[[], Feature] = _plain_host,
    calls: int = 1,
    warn_unmatched: bool = False,
) -> _LedgerSnapshot:
    """Match one declared filter against every probe, `calls` times each, on ONE fresh GlobalFilter."""
    caplog.clear()
    groups = [make() for make in makes]
    global_filter = GlobalFilter()
    global_filter.add_filter(filter_feature, FilterType.EQUAL, {"value": 1})
    matched: set[SingleFilter] | None = None
    names: tuple[str, ...] = ()
    escaped: str | None = None
    try:
        with caplog.at_level(logging.DEBUG, logger=GF_LOGGER_NAME):
            for _ in range(calls):
                # By index: no loop local may keep a probe class alive for a failing assert's traceback.
                for index in range(len(groups)):
                    matched, call_escaped = _capture(
                        partial(global_filter.identify_matched_filters, groups[index], make_host(), None)
                    )
                    escaped = escaped or call_escaped
                    names = (*names, *sorted(single.name for single in matched or ()))
            if warn_unmatched:
                global_filter.warn_on_unmatched_filters()
        rows, ledger_error = _capture(partial(_ledger_rows, global_filter))
        warnings = _messages(caplog, logging.WARNING)
        return _LedgerSnapshot(
            escaped=escaped,
            names=tuple(sorted(names)),
            fact_types=tuple(sorted(type(fact).__name__ for fact in global_filter.dropped_filters.values())),
            ledger_error=ledger_error,
            rows=rows or (),
            warnings=warnings,
            debugs=_messages(caplog, logging.DEBUG),
            unmatched=tuple(message for message in warnings if UNMATCHED_PHRASE in message),
        )
    finally:
        groups.clear()
        del groups, global_filter, matched
        gc.collect()


@dataclass(frozen=True)
class _CanonicalSnapshot:
    """Plain-data readout of one evaluate() pass. Holds no class and no Elimination object."""

    escaped: str | None
    identified: tuple[str, ...]
    eliminations: tuple[tuple[str, str, str], ...]


def _canonical_snapshot(result: EvaluationResult | None, escaped: str | None) -> _CanonicalSnapshot:
    """Fold one evaluate() outcome to plain data."""
    if result is None:
        return _CanonicalSnapshot(escaped=escaped, identified=(), eliminations=())
    return _CanonicalSnapshot(
        escaped=escaped,
        identified=tuple(sorted(g.get_class_name() for g in result.identified)),
        eliminations=tuple(
            sorted((g.get_class_name(), str(e.stage), str(e.reason)) for g, e in result.eliminations.items())
        ),
    )


def _drive_canonical(make: _Factory) -> _CanonicalSnapshot:
    """Evaluate the filter feature against the probe alone and fold the eliminations to plain tuples."""
    fg = make()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    result = None
    try:
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, Feature(FILTER_FEATURE), plugins, None))
        snapshot = _canonical_snapshot(result, escaped)
        del result
        result = None
        return snapshot
    finally:
        del fg, plugins, result
        gc.collect()


def _drive_near_miss_block() -> tuple[str | None, str]:
    """(the rendered block, the same block rebuilt from the shared bullet) over two eliminated candidates."""
    from mloda.core.prepare.resolution_failure_renderer import near_miss_text

    # Built before the classes exist, so a raising helper leaves no throwaway class behind.
    expected = "\n".join(
        (
            f"Feature group(s) eliminated while matching '{FILTER_FEATURE}':",
            f"  - {near_miss_text(SCOPE_TIE_A_CLASS_NAME, VALUE_REJECTION_STAGE, DECLINE_REASON)}",
            f"  - {near_miss_text(SCOPE_TIE_B_CLASS_NAME, SCOPE_STAGE, SCOPE_REASON)}",
        )
    )
    fg_a = _make_scope_tie_a_fg()
    fg_b = _make_scope_tie_b_fg()
    result = None
    try:
        # B first: the block owns the ordering, so insertion order must not decide it.
        eliminations = {
            fg_b: Elimination(stage=SCOPE_STAGE, reason=SCOPE_REASON),
            fg_a: Elimination(stage=VALUE_REJECTION_STAGE, reason=DECLINE_REASON),
        }
        result = EvaluationResult(identified={}, eliminations=eliminations)
        rendered = _render_near_miss_block(result, Feature(FILTER_FEATURE))
        del eliminations, result
        result = None
        return rendered, expected
    finally:
        del fg_a, fg_b, result
        gc.collect()


class TestTheSharedStageMapper:
    """One mapper projects a recorded free-form stage hint onto its elimination stage for both seams."""

    @pytest.mark.parametrize(("hint", "expected"), STAGE_HINT_TABLE, ids=STAGE_HINT_IDS)
    def test_the_mapper_table(self, hint: str, expected: EliminationStage) -> None:
        from mloda.core.prepare.resolution_types import rejection_elimination_stage

        assert rejection_elimination_stage(hint) == expected, f"'{hint}' must map onto '{expected}'"

    @pytest.mark.parametrize(("hint", "expected"), STAGE_HINT_TABLE, ids=STAGE_HINT_IDS)
    def test_the_canonical_seam_records_the_stage_the_mapper_names(self, hint: str, expected: EliminationStage) -> None:
        from mloda.core.prepare.resolution_types import rejection_elimination_stage

        snapshot = _drive_canonical(partial(_make_stage_decline_fg, hint))

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.eliminations == ((STAGE_DECLINE_CLASS_NAME, expected, _stage_reason(hint)),), (
            f"exactly one elimination at '{expected}', got: {snapshot.eliminations}"
        )
        assert rejection_elimination_stage(hint) == expected, "the canonical seam must map through the shared mapper"


class TestTheSharedNearMissBullet:
    """One helper renders a near-miss bullet; the canonical block must stay byte-identical to it."""

    def test_the_bullet_names_the_candidate_its_stage_label_and_its_reason(self) -> None:
        from mloda.core.prepare.resolution_failure_renderer import near_miss_text

        rendered = near_miss_text(PLAIN_CLASS_NAME, SCOPE_STAGE, SCOPE_REASON)

        assert rendered == f"{PLAIN_CLASS_NAME} ({_STAGE_LABELS[SCOPE_STAGE]}): {SCOPE_REASON}", (
            f"the bullet must read '<Candidate> (<stage label>): <reason>', got: {rendered}"
        )

    def test_the_near_miss_block_is_the_shared_bullet_per_candidate(self) -> None:
        """The block's output must not move: it is the shared bullet, indented, one candidate per line."""
        rendered, expected = _drive_near_miss_block()

        assert rendered == expected, f"the block must render the shared bullet, got: {rendered!r} vs {expected!r}"


class TestEachGateRecordsItsOwnFact:
    """One fact per (feature group, filter feature): the gate that closed and why it closed."""

    def test_the_ledger_stores_elimination_facts(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bare reason string cannot say which gate closed, so the ledger holds the stage-plus-reason fact."""
        snapshot = _drive_matching([_make_matcher_error_fg], caplog)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.fact_types == (Elimination.__name__,), (
            f"the ledger must store {Elimination.__name__} facts, got: {snapshot.fact_types}"
        )
        assert snapshot.ledger_error is None, f"a stored fact must carry a stage and a reason: {snapshot.ledger_error}"

    def test_a_matcher_defect_records_matcher_error(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_matcher_error_fg], caplog)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert len(snapshot.rows) == 1, f"exactly one fact, got: {snapshot.rows}"
        name, filter_name, stage, reason = snapshot.rows[0]
        assert (name, filter_name) == (MATCHER_ERROR_CLASS_NAME, FILTER_FEATURE), f"wrong key: {name}, {filter_name}"
        assert stage == MATCHER_ERROR_STAGE, f"a contained raise is a matcher defect, got stage: {stage}"
        assert RUNTIME_TYPE_NAME in reason, f"the reason must name the exception type: {reason}"
        assert RUNTIME_MESSAGE in reason, f"the reason must carry the raise message: {reason}"

    @pytest.mark.parametrize(("hint", "expected"), STAGE_HINT_TABLE, ids=STAGE_HINT_IDS)
    def test_a_typed_decline_records_its_mapped_stage(
        self, hint: str, expected: EliminationStage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The recorded hint decides the stage through the shared mapper; the reason text is unchanged."""
        snapshot = _drive_matching([partial(_make_stage_decline_fg, hint)], caplog)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.rows == ((STAGE_DECLINE_CLASS_NAME, FILTER_FEATURE, expected, _stage_reason(hint)),), (
            f"a decline recorded under '{hint}' must land at '{expected}', got: {snapshot.rows}"
        )
        assert snapshot.warnings == (), f"a typed decline is a verdict, not a defect, got: {snapshot.warnings}"

    @pytest.mark.parametrize(
        ("make", "host_domain", "compared"),
        [
            pytest.param(_make_group_domain_fg, None, GROUP_DOMAIN, id="against_the_group_domain"),
            pytest.param(_make_plain_fg, FEATURE_DOMAIN, FEATURE_DOMAIN, id="against_the_feature_domain"),
        ],
    )
    def test_the_domain_gate_names_both_domains(
        self, make: _Factory, host_domain: str | None, compared: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        snapshot = _drive_matching(
            [make],
            caplog,
            filter_feature=_filter_feature(domain=OTHER_DOMAIN),
            make_host=partial(_host_feature, domain=host_domain),
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.names == (), f"a diverging domain must attach no filter, got: {snapshot.names}"
        assert len(snapshot.rows) == 1, f"exactly one fact, got: {snapshot.rows}"
        _, _, stage, reason = snapshot.rows[0]
        assert stage == DOMAIN_STAGE, f"the domain gate owns this drop, got stage: {stage}"
        assert OTHER_DOMAIN in reason, f"the reason must name the filter feature's declared domain: {reason}"
        assert compared in reason, f"the reason must name the domain it was compared against: {reason}"

    def test_the_scope_gate_records_the_canonical_seams_own_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """Byte-identical to the string the canonical seam records at its own scope gate."""
        snapshot = _drive_matching([_make_plain_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE))

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.rows == ((PLAIN_CLASS_NAME, FILTER_FEATURE, SCOPE_STAGE, SCOPE_REASON),), (
            f"exactly one scope fact carrying the shared string, got: {snapshot.rows}"
        )

    def test_the_capability_gate_mirrors_the_canonical_seams_wording(self, caplog: pytest.LogCaptureFixture) -> None:
        """The rejected frameworks are the ones the filter would ride, named as the canonical seam names them."""
        snapshot = _drive_matching(
            [_make_capability_reject_fg], caplog, make_host=partial(_host_feature, pin=PythonDictFramework)
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.names == (), f"a hook rejecting every framework must attach no filter, got: {snapshot.names}"
        assert snapshot.rows == (
            (CAPABILITY_REJECT_CLASS_NAME, FILTER_FEATURE, CAPABILITY_STAGE, CAPABILITY_REASON),
        ), f"exactly one capability fact carrying the shared wording, got: {snapshot.rows}"

    def test_the_framework_pin_gate_names_the_pin_and_the_resolved_framework(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        snapshot = _drive_matching(
            [_make_plain_fg],
            caplog,
            filter_feature=_filter_feature(pin=PandasDataFrame),
            make_host=partial(_host_feature, pin=PythonDictFramework),
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.names == (), f"a diverging pin must attach no filter, got: {snapshot.names}"
        assert len(snapshot.rows) == 1, f"exactly one fact, got: {snapshot.rows}"
        _, _, stage, reason = snapshot.rows[0]
        assert stage == FRAMEWORK_PIN_STAGE, f"the pin gate owns this drop, got stage: {stage}"
        assert PandasDataFrame.__name__ in reason, f"the reason must name the filter's pinned framework: {reason}"
        assert PythonDictFramework.__name__ in reason, f"the reason must name the feature's own framework: {reason}"


class TestAPlainNonMatchIsNotAFact:
    """Only a near-miss is a fact: saying no is the matcher's judgment and records nothing."""

    def test_a_literal_false_records_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_plain_decline_fg], caplog)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.rows == (), f"a plain non-match is not a near-miss, got: {snapshot.rows}"
        assert snapshot.warnings == (), f"saying no correctly must not warn, got: {snapshot.warnings}"

    def test_a_falsy_non_bool_records_nothing_but_is_still_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_falsy_decline_fg], caplog)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.rows == (), f"a falsy non-bool is a non-match, never a fact, got: {snapshot.rows}"
        assert len(snapshot.warnings) == 1, f"the detached filter must still be reported, got: {snapshot.warnings}"
        assert FALSY_REPORT_FRAGMENT in snapshot.warnings[0], (
            f"the report must call the return a falsy non-bool: {snapshot.warnings[0]}"
        )


class TestPrecedenceAmongFacts:
    """A defect outranks a stored decline; every other stage is first-one-wins and never displaces a defect."""

    def test_a_defect_outranks_a_stored_decline(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_decline_then_defect_fg], caplog, calls=2)

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert len(snapshot.rows) == 1, f"exactly one fact for the key, got: {snapshot.rows}"
        _, _, stage, reason = snapshot.rows[0]
        assert stage == MATCHER_ERROR_STAGE, f"the defect must take the key from the decline, got stage: {stage}"
        assert DEFECT_MESSAGE in reason, f"the reason must carry the defect message: {reason}"
        assert DECLINE_REASON not in reason, f"the defect outranks the recorded decline: {reason}"
        assert len(snapshot.warnings) == 1, f"the defect must warn exactly once, got: {snapshot.warnings}"

    def test_a_later_gate_never_displaces_a_stored_defect(self, caplog: pytest.LogCaptureFixture) -> None:
        """The second pass clears criteria and loses at scope, which must not overwrite the recorded defect."""
        snapshot = _drive_matching(
            [_make_defect_then_match_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE), calls=2
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.names == (), f"the scope still detaches the filter, got: {snapshot.names}"
        assert len(snapshot.rows) == 1, f"exactly one fact for the key, got: {snapshot.rows}"
        _, _, stage, reason = snapshot.rows[0]
        assert stage == MATCHER_ERROR_STAGE, f"a later gate must not displace the defect, got stage: {stage}"
        assert RUNTIME_MESSAGE in reason, f"the defect's reason must survive the later gate: {reason}"

    def test_the_first_near_miss_wins_among_non_defects(self, caplog: pytest.LogCaptureFixture) -> None:
        """Scope loses the first pass, a typed decline the second: the first fact recorded keeps the key."""
        snapshot = _drive_matching(
            [_make_match_then_decline_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE), calls=2
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert len(snapshot.rows) == 1, f"exactly one fact for the key, got: {snapshot.rows}"
        _, _, stage, reason = snapshot.rows[0]
        assert stage == SCOPE_STAGE, f"the first near-miss keeps the key, got stage: {stage}"
        assert reason == SCOPE_REASON, f"the later decline must not rewrite the reason: {reason}"


class TestCriteriaRunsBeforeTheScopeGate:
    """Gate order is observable in the recorded stage: criteria decides before the scope gate is asked."""

    def test_a_raising_matcher_records_matcher_error_even_when_the_scope_would_exclude_it_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        snapshot = _drive_matching(
            [_make_matcher_error_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE)
        )

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert len(snapshot.rows) == 1, f"exactly one fact, got: {snapshot.rows}"
        assert snapshot.rows[0][2] == MATCHER_ERROR_STAGE, (
            f"criteria runs before the scope gate, got stage: {snapshot.rows[0][2]}"
        )

    def test_a_probe_that_clears_criteria_records_scope_without_hook_noise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        snapshot = _drive_matching([_make_plain_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE))

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.ledger_error is None, f"the stored fact must be readable: {snapshot.ledger_error}"
        assert snapshot.rows == ((PLAIN_CLASS_NAME, FILTER_FEATURE, SCOPE_STAGE, SCOPE_REASON),), (
            f"a cleared criteria gate must leave the scope fact alone in the ledger, got: {snapshot.rows}"
        )
        assert snapshot.warnings == (), f"a scope drop is no defect and must not warn, got: {snapshot.warnings}"


class TestTheUnmatchedWarningNamesTheNearestMiss:
    """warn_on_unmatched_filters projects the captured facts: the deepest gate a filter reached."""

    def test_without_a_captured_fact_the_message_is_the_bare_sentence(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_plain_decline_fg], caplog, warn_unmatched=True)

        assert snapshot.rows == (), f"a plain non-match captures nothing, got: {snapshot.rows}"
        assert snapshot.unmatched == (BARE_MESSAGE,), f"the message must stay unchanged, got: {snapshot.unmatched}"

    def test_a_captured_fact_appends_the_shared_near_miss_bullet(self, caplog: pytest.LogCaptureFixture) -> None:
        from mloda.core.prepare.resolution_failure_renderer import near_miss_text

        snapshot = _drive_matching(
            [_make_plain_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE), warn_unmatched=True
        )

        expected = f"{BARE_MESSAGE} {NEAREST_MISS_PHRASE}{near_miss_text(PLAIN_CLASS_NAME, SCOPE_STAGE, SCOPE_REASON)}"
        assert snapshot.unmatched == (expected,), f"the suffix must be the shared bullet, got: {snapshot.unmatched}"

    def test_the_label_comes_from_the_live_stage_table(self, caplog: pytest.LogCaptureFixture) -> None:
        """Never a second spelling of the label: the renderer's own table owns it."""
        snapshot = _drive_matching(
            [_make_plain_fg], caplog, filter_feature=_filter_feature(scope=MISSING_SCOPE), warn_unmatched=True
        )

        assert len(snapshot.unmatched) == 1, f"exactly one unmatched warning, got: {snapshot.unmatched}"
        assert f"({_STAGE_LABELS[SCOPE_STAGE]}):" in snapshot.unmatched[0], (
            f"the label must come from the live table, got: {snapshot.unmatched[0]}"
        )

    def test_the_deepest_gate_wins_between_two_groups(self, caplog: pytest.LogCaptureFixture) -> None:
        """One group loses at criteria, the other survives to the pin gate: the pin is the nearer miss."""
        snapshot = _drive_matching(
            [partial(_make_stage_decline_fg, VALUE_REJECTION_STAGE), _make_plain_fg],
            caplog,
            filter_feature=_filter_feature(pin=PandasDataFrame),
            make_host=partial(_host_feature, pin=PythonDictFramework),
            warn_unmatched=True,
        )

        assert len(snapshot.unmatched) == 1, f"exactly one unmatched warning, got: {snapshot.unmatched}"
        message = snapshot.unmatched[0]
        assert message.startswith(
            f"{BARE_MESSAGE} {NEAREST_MISS_PHRASE}{PLAIN_CLASS_NAME} ({_STAGE_LABELS[FRAMEWORK_PIN_STAGE]}):"
        ), f"the deepest gate must own the nearest miss, got: {message}"
        assert STAGE_DECLINE_CLASS_NAME not in message, f"the shallower miss must not be named: {message}"

    def test_a_defect_loses_to_a_declining_sibling(self, caplog: pytest.LogCaptureFixture) -> None:
        """A crash says nothing about how far the filter got, so a real decline is the nearer miss."""
        from mloda.core.prepare.resolution_failure_renderer import near_miss_text

        snapshot = _drive_matching(
            [_make_matcher_error_fg, partial(_make_stage_decline_fg, VALUE_REJECTION_STAGE)],
            caplog,
            warn_unmatched=True,
        )

        expected_bullet = near_miss_text(
            STAGE_DECLINE_CLASS_NAME, VALUE_REJECTION_STAGE, _stage_reason(VALUE_REJECTION_STAGE)
        )
        assert snapshot.unmatched == (f"{BARE_MESSAGE} {NEAREST_MISS_PHRASE}{expected_bullet}",), (
            f"the decline must outrank the defect, got: {snapshot.unmatched}"
        )

    def test_an_equal_stage_ties_break_by_class_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two groups losing at the same gate: the pick is the renderer's own candidate order, not insertion order."""
        from mloda.core.prepare.resolution_failure_renderer import near_miss_text

        snapshot = _drive_matching(
            [_make_scope_tie_b_fg, _make_scope_tie_a_fg],
            caplog,
            filter_feature=_filter_feature(scope=MISSING_SCOPE),
            warn_unmatched=True,
        )

        expected_bullet = near_miss_text(SCOPE_TIE_A_CLASS_NAME, SCOPE_STAGE, SCOPE_REASON)
        assert snapshot.unmatched == (f"{BARE_MESSAGE} {NEAREST_MISS_PHRASE}{expected_bullet}",), (
            f"the tie must break by class name, got: {snapshot.unmatched}"
        )

    def test_a_filter_that_matched_warns_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_matching([_make_plain_fg], caplog, warn_unmatched=True)

        assert snapshot.names == (FILTER_FEATURE,), f"the filter must attach, got: {snapshot.names}"
        assert snapshot.unmatched == (), f"an attached filter is not unmatched, got: {snapshot.unmatched}"
