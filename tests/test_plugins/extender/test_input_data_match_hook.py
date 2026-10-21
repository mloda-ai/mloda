"""Tests for ExtenderHook.INPUT_DATA_MATCH: extenders wrap and may refuse a reader's or format group's match probe.

Each test candidate sits behind its own marker access and only accepts it, so it never hijacks matching
in other tests. Tests run once per candidate kind (legacy reader, format group) unless the contract differs.
"""

import logging
from typing import Any, ClassVar, NamedTuple

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.reader_probe import ReaderProbeScope, reader_probe_scope
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.provider import ClaimRoute, FeatureGroup, FeatureSet, FormatFeatureGroup, NamePolicy, SourceMatch
from mloda.user import Feature, FeatureName, GlobalFilter, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
    TOY_KEY,
    ToyFormatBase,
)

_ACCESS = "imdmatch_pinned_access"
_CRED_KEY = "imdmatch_marker"
_COL_A = "imdmatch_col_a"
_COL_B = "imdmatch_col_b"
_COL_C = "imdmatch_col_c"
_COL_D = "imdmatch_col_d"
_COL_F = "imdmatch_col_f"
_COL_G = "imdmatch_col_g"
_COL_P = "imdmatch_col_p"
_FMT_COLUMNS = {c: [1] for c in (_COL_A, _COL_B, _COL_C, _COL_D, _COL_F, _COL_G)}
_PROBE_LOGGER = "mloda.core.abstract_plugins.components.input_data.reader_probe"
_REFUSED = "refused at INPUT_DATA_MATCH"


def _in_scope(data_access: Any) -> bool:
    if isinstance(data_access, DataAccessCollection):
        return any(_CRED_KEY in credential for credential in data_access.credentials.values())
    return bool(data_access == _ACCESS)


class _ImdBaseReader(BaseInputData):
    """Accepts only in-scope access and only the names in `served`; counts probes."""

    served: frozenset[str] = frozenset()
    probes = 0

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if not _in_scope(data_access):
            return None
        cls.probes += 1
        return data_access if set(feature_names) <= cls.served else None


class ImdMatchGateReader(_ImdBaseReader):
    served = frozenset({_COL_A, _COL_F, _COL_G})
    probes = 0

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        return {"classification": "restricted", "level": 3}

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_A: [1], _COL_F: [1], _COL_G: [1]})


class ImdMatchSiblingReader(_ImdBaseReader):
    served = frozenset({_COL_B})
    probes = 0

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_B: [2]})


class ImdMatchBrokenDeclReader(_ImdBaseReader):
    served = frozenset({_COL_C})
    probes = 0

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        raise RuntimeError("imdmatch declared_attributes boom")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_C: [3]})


class ImdMatchRaisingProbeReader(_ImdBaseReader):
    served = frozenset({_COL_D})
    probes = 0

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if _in_scope(data_access) and set(feature_names) <= cls.served:
            cls.probes += 1
            raise ValueError("imdmatch probe boom")
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_D: [4]})


class _ImdBaseGroup(FeatureGroup):
    """Root group serving one reader; `reader` is set per subclass."""

    reader: type[BaseInputData] | None = None

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return None if cls.reader is None else cls.reader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.input_data().load(features)  # type: ignore[union-attr]


class ImdMatchGateGroup(_ImdBaseGroup):
    reader = ImdMatchGateReader


class ImdMatchSiblingGroup(_ImdBaseGroup):
    reader = ImdMatchSiblingReader


class ImdMatchBrokenDeclGroup(_ImdBaseGroup):
    reader = ImdMatchBrokenDeclReader


class ImdMatchRaisingProbeGroup(_ImdBaseGroup):
    reader = ImdMatchRaisingProbeReader


def _fmt_in_scope(cls: type[FormatFeatureGroup], options: Options, data_access_collection: Any) -> bool:
    if options.get(cls.get_class_name()) is not None:
        return True
    return isinstance(data_access_collection, DataAccessCollection) and any(
        _CRED_KEY in credential for credential in data_access_collection.credentials.values()
    )


class _ImdFmtBase(ToyFormatBase):
    """Toy format group serving only `served` columns; reacts only to in-scope sources; counts source touches."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("imdmatch", NamePolicy.CHECKED, True),)
    served: frozenset[str] = frozenset()
    raises = False
    probes = 0
    column_reads = 0

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        if not _fmt_in_scope(cls, options, data_access_collection):
            return []
        cls.probes += 1
        if cls.raises and feature_name in cls.served:
            raise ValueError("imdmatch probe boom")
        return super().find_sources(route, feature_name, options, data_access_collection)

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        cls.column_reads += 1
        return [c for c in match.access if c in cls.served]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return pa.table({c: v for c, v in match.access.items() if c in cls.served})


class ImdMatchGateFormat(_ImdFmtBase):
    served = frozenset({_COL_A, _COL_F, _COL_G})

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        return {"classification": "restricted", "level": 3}


class ImdMatchGateSubFormat(ImdMatchGateFormat):
    pass


class ImdMatchSiblingFormat(_ImdFmtBase):
    served = frozenset({_COL_B})


class ImdMatchBrokenDeclFormat(_ImdFmtBase):
    served = frozenset({_COL_C})

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        raise RuntimeError("imdmatch declared_attributes boom")


class ImdMatchRaisingFindFormat(_ImdFmtBase):
    served = frozenset({_COL_D})
    raises = True


class ImdMatchAaTwinFormat(_ImdFmtBase):
    served = frozenset({_COL_A})


class ImdMatchZzTwinFormat(_ImdFmtBase):
    served = frozenset({_COL_A})


class ImdMatchRefusedThenRaisingFormat(ImdMatchGateFormat):
    """Probes its source (refused by the extender), then raises an ordinary matcher error."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        super().match_feature_group_criteria(feature_name, options, data_access_collection)
        raise RuntimeError("imdmatch matcher boom")


_PROBE_TEXT = "imdmatch own probe rejection"


class ImdMatchRejectingFormat(_ImdFmtBase):
    """Records its own rejection while probing and finds nothing."""

    served = frozenset({_COL_A})

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        record_match_rejection(cls.get_class_name(), _PROBE_TEXT, stage=INPUT_DATA_STAGE)
        return []


class ImdMatchComputingGroup(FeatureGroup):
    """Computing group (it has an input) for the column the pointed gate format serves."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_COL_A}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_A: [6]})


class ImdMatchPlainGroup(FeatureGroup):
    """Plain computing group for one name; no source involved."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_COL_P}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({_COL_P: [5]})


_ALL_READERS = (ImdMatchGateReader, ImdMatchSiblingReader, ImdMatchBrokenDeclReader, ImdMatchRaisingProbeReader)
_ALL_FORMATS = (
    ImdMatchGateFormat,
    ImdMatchGateSubFormat,
    ImdMatchSiblingFormat,
    ImdMatchBrokenDeclFormat,
    ImdMatchRaisingFindFormat,
    ImdMatchAaTwinFormat,
    ImdMatchZzTwinFormat,
    ImdMatchRefusedThenRaisingFormat,
)

_ENABLED = PluginCollector.enabled_feature_groups(
    {ImdMatchGateGroup, ImdMatchSiblingGroup, ImdMatchBrokenDeclGroup, ImdMatchRaisingProbeGroup}
)
_ENABLED_FMT = PluginCollector.enabled_feature_groups(
    {ImdMatchGateFormat, ImdMatchSiblingFormat, ImdMatchBrokenDeclFormat, ImdMatchRaisingFindFormat}
)


class _Kind(NamedTuple):
    gate: Any
    sibling: Any
    broken: Any
    raising: Any
    is_format: bool


_READER_KIND = _Kind(
    ImdMatchGateReader, ImdMatchSiblingReader, ImdMatchBrokenDeclReader, ImdMatchRaisingProbeReader, False
)
_FORMAT_KIND = _Kind(
    ImdMatchGateFormat, ImdMatchSiblingFormat, ImdMatchBrokenDeclFormat, ImdMatchRaisingFindFormat, True
)
_BY_KIND = pytest.mark.parametrize("kind", [_READER_KIND, _FORMAT_KIND], ids=["reader", "format"])


class _ImdRecordingExtender(Extender):
    """Pass-through extender recording the HookContext seen during each probe."""

    def __init__(self) -> None:
        self.priority = 100
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        self.contexts.append(context)
        return func(*args, **kwargs)

    def for_reader(self, reader: type) -> list[HookContext]:
        return [c for c in self.contexts if c.reader_class is reader]


class _ImdRefusingExtender(Extender):
    """Raises PermissionError for the target candidate, passes everything else through."""

    def __init__(
        self, target: type | tuple[type, ...], never_fall_back: bool = False, raise_on_error: bool = True
    ) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.never_fall_back = never_fall_back
        self.name = "imdmatch_refuse"
        self._targets: tuple[type, ...] = target if isinstance(target, tuple) else (target,)
        self.raised: list[Exception] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.reader_class in self._targets:
            refusal = PermissionError(f"no clearance for {context.reader_class.__name__}")
            self.raised.append(refusal)
            raise refusal
        return func(*args, **kwargs)


class _ImdSkippingExtender(_ImdRefusingExtender):
    """Returns a forged probe result (fitting, seen, undeclared) for the target without calling func or raising."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.reader_class in self._targets:
            forged = SourceMatch(source="imdmatch_forged", access=dict(_FMT_COLUMNS))
            return {forged.source: forged}, {}, False
        return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_probe_counters() -> None:
    for candidate in (*_ALL_READERS, *_ALL_FORMATS):
        candidate.probes = 0
    for fmt in _ALL_FORMATS:
        fmt.column_reads = 0


def _global_access(fmt: bool = False) -> DataAccessCollection:
    if fmt:
        return DataAccessCollection(credentials={"imdmatch_handle": {TOY_KEY: dict(_FMT_COLUMNS), _CRED_KEY: 1}})
    return DataAccessCollection(credentials=[{_CRED_KEY: 1}])


def _prepare(
    feature: Feature,
    enabled: PluginCollector,
    data_access_collection: DataAccessCollection | None,
    extender: Extender | None,
    global_filter: GlobalFilter | None = None,
) -> Any:
    return mloda.prepare(
        [feature],
        compute_frameworks=[PyArrowTable],
        plugin_collector=enabled,
        parallelization_modes={ParallelizationMode.SYNC},
        data_access_collection=data_access_collection,
        function_extender={extender} if extender is not None else None,
        global_filter=global_filter,
    )


def _plan(
    scope: str,
    feature_name: str,
    reader: type,
    extender: Extender | None,
    global_filter: GlobalFilter | None = None,
) -> Any:
    fmt = issubclass(reader, FormatFeatureGroup)
    options = {reader.__name__: dict(_FMT_COLUMNS) if fmt else _ACCESS} if scope == "pinned" else None
    return _prepare(
        Feature(feature_name, options=options),
        _ENABLED_FMT if fmt else _ENABLED,
        _global_access(fmt) if scope == "global" else None,
        extender,
        global_filter,
    )


def _refusal_error(kind: _Kind, scope: str) -> tuple[type[Exception], str]:
    """A refused pointed format group aborts the match (a ValueError); everything else fails resolution."""
    if kind.is_format and scope == "pinned":
        return ValueError, _REFUSED
    return FeatureResolutionError, "no clearance"


class _ImdChainedRefusingExtender(_ImdRefusingExtender):
    """Refuses with a PermissionError chained from a LookupError raised inside an except block."""

    def __init__(self, target: type) -> None:
        super().__init__(target, never_fall_back=True)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.reader_class in self._targets:
            try:
                raise KeyError("inner")
            except KeyError:
                try:
                    raise LookupError("lookup failed")
                except LookupError as lookup:
                    raise PermissionError("denied") from lookup
        return func(*args, **kwargs)


class _ImdPostProbeRefusingExtender(_ImdRefusingExtender):
    """Runs the probe first, then raises for the target candidate."""

    def __init__(self, target: type) -> None:
        super().__init__(target, never_fall_back=True)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None and context.reader_class in self._targets:
            raise PermissionError("no clearance")
        return result


class _ImdColumnGate(Extender):
    """Gate raising only when the probed feature names include the filter column."""

    never_fall_back = True

    def __init__(self) -> None:
        self.priority = 100
        self.name = "imdmatch_column_gate"

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and _COL_F in context.feature_names:
            raise PermissionError("no clearance")
        return func(*args, **kwargs)


def _gate(target: type) -> _ImdRefusingExtender:
    return _ImdRefusingExtender(target, never_fall_back=True)


def _breaking(target: type) -> _ImdRefusingExtender:
    return _ImdRefusingExtender(target)


class TestRefusal:
    @_BY_KIND
    @pytest.mark.parametrize("scope", ["pinned", "global"])
    @pytest.mark.parametrize("make_extender", [_gate, _breaking], ids=["gate", "breaking"])
    def test_refused_reader_probe_never_runs_and_planning_fails(
        self, kind: _Kind, scope: str, make_extender: Any
    ) -> None:
        _plan(scope, _COL_A, kind.gate, None)
        assert kind.gate.probes >= 1
        kind.gate.probes = 0
        if kind.is_format:
            kind.gate.column_reads = 0
        error_type, text = _refusal_error(kind, scope)

        extender = make_extender(kind.gate)

        with pytest.raises(error_type) as excinfo:
            _plan(scope, _COL_A, kind.gate, extender)

        assert type(excinfo.value) is error_type
        assert "no clearance" in str(excinfo.value)
        assert text in str(excinfo.value)
        if error_type is FeatureResolutionError:
            assert isinstance(excinfo.value, FeatureResolutionError)
            refusals = excinfo.value.refusals
            assert len(refusals) == 1
            assert type(refusals[0]) is PermissionError
            assert str(refusals[0]) == f"no clearance for {kind.gate.__name__}"
            assert any(refusals[0] is raised for raised in extender.raised)
            assert excinfo.value.__cause__ is refusals[0]
            assert refusals[0].__traceback__ is None
        else:
            assert isinstance(excinfo.value.__cause__, PermissionError)
            assert any(excinfo.value.__cause__ is raised for raised in extender.raised)
        assert kind.gate.probes == 0
        if kind.is_format:
            assert kind.gate.column_reads == 0

    def test_two_refused_candidates_are_both_carried_in_order(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat, ImdMatchAaTwinFormat})
        extender = _ImdRefusingExtender((ImdMatchGateFormat, ImdMatchAaTwinFormat), never_fall_back=True)

        with pytest.raises(FeatureResolutionError) as excinfo:
            _prepare(Feature(_COL_A), enabled, _global_access(True), extender)

        refusals = excinfo.value.refusals
        assert [str(r) for r in refusals] == [
            "no clearance for ImdMatchAaTwinFormat",
            "no clearance for ImdMatchGateFormat",
        ]
        assert all(any(r is raised for raised in extender.raised) for r in refusals)
        assert excinfo.value.__cause__ is refusals[0]
        assert all(r.__traceback__ is None for r in refusals)

    def test_ambiguity_failure_carries_no_refusals(self) -> None:
        enabled = PluginCollector.enabled_feature_groups(
            {ImdMatchGateFormat, ImdMatchAaTwinFormat, ImdMatchZzTwinFormat}
        )
        extender = _gate(ImdMatchGateFormat)

        with pytest.raises(FeatureResolutionError) as excinfo:
            _prepare(Feature(_COL_A), enabled, _global_access(True), extender)

        assert excinfo.value.result.failure_kind == "multiple"
        assert extender.raised
        assert excinfo.value.refusals == ()
        assert excinfo.value.__cause__ is None

    def test_chained_refusal_and_its_chain_are_stripped_of_tracebacks(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat})

        with pytest.raises(FeatureResolutionError) as excinfo:
            _prepare(Feature(_COL_A), enabled, _global_access(True), _ImdChainedRefusingExtender(ImdMatchGateFormat))

        refusal = excinfo.value.refusals[0]
        assert type(refusal) is PermissionError
        assert type(refusal.__cause__) is LookupError
        chain: list[BaseException] = []
        pending: list[BaseException | None] = [refusal]
        while pending:
            current = pending.pop()
            if current is not None and all(current is not seen for seen in chain):
                chain.append(current)
                pending.extend([current.__cause__, current.__context__])
        assert len(chain) >= 2
        assert all(e.__traceback__ is None for e in chain)

    def test_matcher_error_eliminates_the_group_and_drops_its_refusal(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchRefusedThenRaisingFormat})
        extender = _gate(ImdMatchRefusedThenRaisingFormat)

        with pytest.raises(FeatureResolutionError) as excinfo:
            _prepare(Feature(_COL_A), enabled, _global_access(True), extender)

        assert extender.raised
        assert "imdmatch matcher boom" in str(excinfo.value)
        assert excinfo.value.refusals == ()
        assert excinfo.value.__cause__ is None

    @_BY_KIND
    @pytest.mark.parametrize("scope", ["pinned", "global"])
    def test_post_probe_raise_still_refuses_the_reader(self, kind: _Kind, scope: str) -> None:
        error_type, _ = _refusal_error(kind, scope)

        with pytest.raises(error_type) as excinfo:
            _plan(scope, _COL_A, kind.gate, _ImdPostProbeRefusingExtender(kind.gate))

        assert type(excinfo.value) is error_type
        assert kind.gate.probes == 1

    @_BY_KIND
    @pytest.mark.parametrize(
        ("make_extender", "level"), [(_gate, logging.INFO), (_breaking, logging.WARNING)], ids=["gate", "breaking"]
    )
    def test_refusal_is_logged_with_reader_extender_and_reason(
        self, caplog: pytest.LogCaptureFixture, kind: _Kind, make_extender: Any, level: int
    ) -> None:
        extender = make_extender(kind.gate)

        with caplog.at_level(logging.DEBUG):
            _plan("global", _COL_B, kind.gate, extender)

        records = [r for r in caplog.records if r.name == _PROBE_LOGGER]
        assert any(
            kind.gate.__name__ in r.getMessage()
            and type(extender).__name__ in r.getMessage()
            and "no clearance" in r.getMessage()
            and r.levelno == level
            for r in records
        )

    @_BY_KIND
    def test_pinned_gate_refusal_is_named_in_the_planning_error(self, kind: _Kind) -> None:
        with pytest.raises(Exception) as excinfo:  # noqa: B017
            _plan("pinned", _COL_A, kind.gate, _gate(kind.gate))

        assert "no clearance" in str(excinfo.value)

    @_BY_KIND
    def test_refused_reader_leaves_another_reader_able_to_serve(self, kind: _Kind) -> None:
        session = _plan("global", _COL_B, kind.gate, _gate(kind.gate))
        result = session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert result[0].column_names == [_COL_B]
        assert kind.gate.probes == 0

    @_BY_KIND
    def test_warning_only_extender_falls_back_to_running_the_probe(
        self, caplog: pytest.LogCaptureFixture, kind: _Kind
    ) -> None:
        extender = _ImdRefusingExtender(kind.gate, raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            session = _plan("pinned", _COL_A, kind.gate, extender)

        assert session is not None
        assert kind.gate.probes >= 1
        assert any(
            r.levelno == logging.WARNING
            and "_ImdRefusingExtender" in r.getMessage()
            and "no clearance" in r.getMessage()
            for r in caplog.records
        )


class TestContext:
    @_BY_KIND
    @pytest.mark.parametrize("scope", ["pinned", "global"])
    def test_context_carries_reader_declaration_and_planning_identity(self, kind: _Kind, scope: str) -> None:
        extender = _ImdRecordingExtender()

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            _plan(scope, _COL_A, kind.gate, extender)

        contexts = extender.for_reader(kind.gate)
        assert contexts
        if kind.is_format:
            assert len(contexts) == 1
        context = contexts[0]
        assert context.hook is ExtenderHook.INPUT_DATA_MATCH
        assert context.declared_attributes == {"classification": "restricted", "level": 3}
        assert context.feature_names == (_COL_A,)
        assert context.data_access_format == kind.gate.data_access_name()
        assert (context.tenant_id, context.project_id, context.principal) == ("acme", "proj1", "hash123")
        assert context.run_id is None
        assert context.feature_group_class is None
        assert context.feature_group_version is None
        assert context.compute_framework_name is None
        assert context.structure_hash is None
        assert context.step_uuid is None

    @_BY_KIND
    def test_declared_attributes_is_none_when_the_reader_declaration_raises(self, kind: _Kind) -> None:
        extender = _ImdRecordingExtender()

        session = _plan("pinned", _COL_C, kind.broken, extender)

        assert session is not None
        contexts = extender.for_reader(kind.broken)
        assert contexts
        assert all(c.declared_attributes is None for c in contexts)

    @_BY_KIND
    def test_filter_probe_is_dispatched_with_the_filter_column(self, kind: _Kind) -> None:
        extender = _ImdRecordingExtender()
        global_filter = GlobalFilter()
        global_filter.add_filter(_COL_F, "min", {"value": 0})

        _plan("global", _COL_G, kind.gate, extender, global_filter=global_filter)

        probed = [name for c in extender.for_reader(kind.gate) for name in c.feature_names]
        assert _COL_F in probed


class TestScopeAndFilter:
    @_BY_KIND
    def test_refusing_gate_on_a_filter_column_leaves_the_filter_unattached_and_warns(
        self, caplog: pytest.LogCaptureFixture, kind: _Kind
    ) -> None:
        global_filter = GlobalFilter()
        global_filter.add_filter(_COL_F, "min", {"value": 0})

        with caplog.at_level(logging.WARNING):
            session = _plan("global", _COL_G, kind.gate, _ImdColumnGate(), global_filter=global_filter)

        assert session is not None
        assert any(f"Filter feature '{_COL_F}' matched no feature group" in r.getMessage() for r in caplog.records)

    @_BY_KIND
    def test_a_plan_without_its_own_extender_does_not_inherit_an_outer_probe_scope(self, kind: _Kind) -> None:
        extender = _ImdRecordingExtender()

        with reader_probe_scope(ReaderProbeScope(extender, None, None, None, None)):
            _plan("pinned", _COL_A, kind.gate, None)

        assert extender.contexts == []

        with reader_probe_scope(ReaderProbeScope(extender, None, None, None, None)):
            _plan("pinned", _COL_A, kind.gate, extender)

        assert extender.for_reader(kind.gate)


class TestProbeRaise:
    @_BY_KIND
    def test_probe_raise_is_unchanged_by_a_pass_through_extender(self, kind: _Kind) -> None:
        with pytest.raises(Exception) as baseline:  # noqa: B017
            _plan("pinned", _COL_D, kind.raising, None)
        recorder = _ImdRecordingExtender()
        with pytest.raises(Exception) as wrapped:  # noqa: B017
            _plan("pinned", _COL_D, kind.raising, recorder)

        assert recorder.for_reader(kind.raising)
        assert type(wrapped.value) is type(baseline.value)
        assert str(wrapped.value) == str(baseline.value)
        assert "imdmatch probe boom" in str(baseline.value)


class TestFormatGroupHook:
    def test_a_group_excluded_by_a_foreign_pointer_does_not_fire_the_hook(self) -> None:
        extender = _ImdRecordingExtender()
        feature = Feature(_COL_B, options={ImdMatchSiblingFormat.__name__: dict(_FMT_COLUMNS)})

        _prepare(feature, _ENABLED_FMT, _global_access(True), extender)

        assert extender.for_reader(ImdMatchSiblingFormat)
        assert extender.for_reader(ImdMatchGateFormat) == []
        assert ImdMatchGateFormat.probes == 0

    @pytest.mark.parametrize("scope", ["pinned", "global"])
    def test_an_extender_that_returns_without_running_the_probe_declines_the_group(self, scope: str) -> None:
        error_type, _ = _refusal_error(_FORMAT_KIND, scope)

        with pytest.raises(error_type) as excinfo:
            _plan(scope, _COL_A, ImdMatchGateFormat, _ImdSkippingExtender(ImdMatchGateFormat))

        assert type(excinfo.value) is error_type
        assert "the extender did not run the probe" in str(excinfo.value)
        if error_type is FeatureResolutionError:
            assert isinstance(excinfo.value, FeatureResolutionError)
            assert excinfo.value.refusals == ()
            assert excinfo.value.__cause__ is None
        assert ImdMatchGateFormat.probes == 0
        assert ImdMatchGateFormat.column_reads == 0

    def test_a_refused_pointed_group_is_not_rescued_by_a_reader_of_the_same_name(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat, ImdMatchGateGroup})
        feature = Feature(_COL_A, options={ImdMatchGateFormat.__name__: dict(_FMT_COLUMNS)})

        with pytest.raises(ValueError) as excinfo:
            _prepare(feature, enabled, _global_access(False), _gate(ImdMatchGateFormat))

        assert _REFUSED in str(excinfo.value)

    def test_a_refused_subclass_leaves_its_concrete_parent_able_to_serve_the_source(self) -> None:
        """Pins the documented per-class limit: a refusal never reaches the parent."""
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat, ImdMatchGateSubFormat})

        session = _prepare(Feature(_COL_A), enabled, _global_access(True), _gate(ImdMatchGateSubFormat))
        result = session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert result[0].column_names == [_COL_A]
        assert ImdMatchGateSubFormat.probes == 0
        assert ImdMatchGateFormat.probes >= 1

    def test_a_refusing_gate_leaves_an_unrelated_computing_group_planning_and_running(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat, ImdMatchPlainGroup})

        session = _prepare(Feature(_COL_P), enabled, _global_access(True), _gate(ImdMatchGateFormat))
        result = session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert result[0].column_names == [_COL_P]
        assert ImdMatchGateFormat.probes == 0
        assert ImdMatchGateFormat.column_reads == 0

    def test_a_refused_pointed_group_is_masked_by_a_computing_candidate(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchGateFormat, ImdMatchComputingGroup})
        feature = Feature(_COL_A, options={ImdMatchGateFormat.__name__: dict(_FMT_COLUMNS)})

        session = _prepare(feature, enabled, None, _gate(ImdMatchGateFormat))

        assert session is not None
        assert ImdMatchGateFormat.probes == 0

    def test_a_post_probe_refusal_replaces_the_probes_own_rejection_for_an_unpointed_group(self) -> None:
        enabled = PluginCollector.enabled_feature_groups({ImdMatchRejectingFormat})

        with pytest.raises(FeatureResolutionError) as excinfo:
            _prepare(
                Feature(_COL_A),
                enabled,
                _global_access(True),
                _ImdPostProbeRefusingExtender(ImdMatchRejectingFormat),
            )

        assert _REFUSED in str(excinfo.value)
        assert _PROBE_TEXT not in str(excinfo.value)
