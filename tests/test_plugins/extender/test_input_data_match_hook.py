"""Tests for ExtenderHook.INPUT_DATA_MATCH: extenders wrap and may refuse a reader's match probe.

Test readers only accept their own marker access, so they never hijack matching in other tests.
"""

import logging
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.reader_probe import ReaderProbeScope, reader_probe_scope
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.provider import FeatureSet
from mloda.user import Feature, GlobalFilter, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature

_ACCESS = "imdmatch_pinned_access"
_CRED_KEY = "imdmatch_marker"
_COL_A = "imdmatch_col_a"
_COL_B = "imdmatch_col_b"
_COL_C = "imdmatch_col_c"
_COL_D = "imdmatch_col_d"
_COL_F = "imdmatch_col_f"
_COL_G = "imdmatch_col_g"
_PROBE_LOGGER = "mloda.core.abstract_plugins.components.input_data.reader_probe"
_ENABLED = PluginCollector.enabled_feature_groups({ReadFileFeature})


def _in_scope(data_access: Any) -> bool:
    if isinstance(data_access, DataAccessCollection):
        return any(_CRED_KEY in credential for credential in data_access.credentials.values())
    return bool(data_access == _ACCESS)


class _ImdBaseReader(ReadFile):
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
    """Raises PermissionError for the target reader, passes everything else through."""

    def __init__(self, target: type, never_fall_back: bool = False, raise_on_error: bool = True) -> None:
        self.priority = 100
        self.raise_on_error = raise_on_error
        self.never_fall_back = never_fall_back
        self.name = "imdmatch_refuse"
        self._target = target

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.reader_class is self._target:
            raise PermissionError("no clearance")
        return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_probe_counters() -> None:
    for reader in (ImdMatchGateReader, ImdMatchSiblingReader, ImdMatchBrokenDeclReader, ImdMatchRaisingProbeReader):
        reader.probes = 0


def _global_access() -> DataAccessCollection:
    return DataAccessCollection(credentials=[{_CRED_KEY: 1}])


def _plan(
    scope: str,
    feature_name: str,
    reader: type,
    extender: Extender | None,
    global_filter: GlobalFilter | None = None,
) -> Any:
    options = {reader.__name__: _ACCESS} if scope == "pinned" else None
    return mloda.prepare(
        [Feature(feature_name, options=options)],
        compute_frameworks=[PyArrowTable],
        plugin_collector=_ENABLED,
        parallelization_modes={ParallelizationMode.SYNC},
        data_access_collection=_global_access() if scope == "global" else None,
        function_extender={extender} if extender is not None else None,
        global_filter=global_filter,
    )


class _ImdPostProbeRefusingExtender(_ImdRefusingExtender):
    """Runs the probe first, then raises for the target reader."""

    def __init__(self) -> None:
        super().__init__(ImdMatchGateReader, never_fall_back=True)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None and context.reader_class is self._target:
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


def _gate() -> _ImdRefusingExtender:
    return _ImdRefusingExtender(ImdMatchGateReader, never_fall_back=True)


def _breaking() -> _ImdRefusingExtender:
    return _ImdRefusingExtender(ImdMatchGateReader)


class TestRefusal:
    @pytest.mark.parametrize("scope", ["pinned", "global"])
    @pytest.mark.parametrize("make_extender", [_gate, _breaking], ids=["gate", "breaking"])
    def test_refused_reader_probe_never_runs_and_planning_fails(self, scope: str, make_extender: Any) -> None:
        _plan(scope, _COL_A, ImdMatchGateReader, None)
        assert ImdMatchGateReader.probes >= 1
        ImdMatchGateReader.probes = 0

        with pytest.raises(FeatureResolutionError) as excinfo:
            _plan(scope, _COL_A, ImdMatchGateReader, make_extender())

        assert "no clearance" in str(excinfo.value)
        assert ImdMatchGateReader.probes == 0

    @pytest.mark.parametrize("scope", ["pinned", "global"])
    def test_post_probe_raise_still_refuses_the_reader(self, scope: str) -> None:
        with pytest.raises(FeatureResolutionError):
            _plan(scope, _COL_A, ImdMatchGateReader, _ImdPostProbeRefusingExtender())

        assert ImdMatchGateReader.probes == 1

    @pytest.mark.parametrize(
        ("make_extender", "level"), [(_gate, logging.INFO), (_breaking, logging.WARNING)], ids=["gate", "breaking"]
    )
    def test_refusal_is_logged_with_reader_extender_and_reason(
        self, caplog: pytest.LogCaptureFixture, make_extender: Any, level: int
    ) -> None:
        extender = make_extender()

        with caplog.at_level(logging.DEBUG):
            _plan("global", _COL_B, ImdMatchGateReader, extender)

        records = [r for r in caplog.records if r.name == _PROBE_LOGGER]
        assert any(
            "ImdMatchGateReader" in r.getMessage()
            and type(extender).__name__ in r.getMessage()
            and "no clearance" in r.getMessage()
            and r.levelno == level
            for r in records
        )

    def test_pinned_gate_refusal_is_named_in_the_planning_error(self) -> None:
        with pytest.raises(Exception) as excinfo:  # noqa: B017
            _plan("pinned", _COL_A, ImdMatchGateReader, _gate())

        assert "no clearance" in str(excinfo.value)

    def test_refused_reader_leaves_a_sibling_reader_able_to_serve(self) -> None:
        session = _plan("global", _COL_B, ImdMatchGateReader, _gate())
        result = session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert result[0].column_names == [_COL_B]
        assert ImdMatchGateReader.probes == 0

    def test_warning_only_extender_falls_back_to_running_the_probe(self, caplog: pytest.LogCaptureFixture) -> None:
        extender = _ImdRefusingExtender(ImdMatchGateReader, raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            session = _plan("pinned", _COL_A, ImdMatchGateReader, extender)

        assert session is not None
        assert ImdMatchGateReader.probes >= 1


class TestContext:
    def test_context_carries_reader_declaration_and_planning_identity(self) -> None:
        extender = _ImdRecordingExtender()

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            _plan("pinned", _COL_A, ImdMatchGateReader, extender)

        contexts = extender.for_reader(ImdMatchGateReader)
        assert contexts
        context = contexts[0]
        assert context.hook is ExtenderHook.INPUT_DATA_MATCH
        assert context.declared_attributes == {"classification": "restricted", "level": 3}
        assert context.feature_names == (_COL_A,)
        assert context.data_access_format == ImdMatchGateReader.data_access_name()
        assert (context.tenant_id, context.project_id, context.principal) == ("acme", "proj1", "hash123")
        assert context.run_id is None
        assert context.feature_group_class is None
        assert context.compute_framework_name is None

    def test_declared_attributes_is_none_when_the_reader_declaration_raises(self) -> None:
        extender = _ImdRecordingExtender()

        session = _plan("pinned", _COL_C, ImdMatchBrokenDeclReader, extender)

        assert session is not None
        contexts = extender.for_reader(ImdMatchBrokenDeclReader)
        assert contexts
        assert all(c.declared_attributes is None for c in contexts)

    def test_filter_probe_is_dispatched_with_the_filter_column(self) -> None:
        extender = _ImdRecordingExtender()
        global_filter = GlobalFilter()
        global_filter.add_filter(_COL_F, "min", {"value": 0})

        _plan("global", _COL_G, ImdMatchGateReader, extender, global_filter=global_filter)

        probed = [name for c in extender.for_reader(ImdMatchGateReader) for name in c.feature_names]
        assert _COL_F in probed


class TestScopeAndFilter:
    def test_refusing_gate_on_a_filter_column_leaves_the_filter_unattached_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        global_filter = GlobalFilter()
        global_filter.add_filter(_COL_F, "min", {"value": 0})

        with caplog.at_level(logging.WARNING):
            session = _plan("global", _COL_G, ImdMatchGateReader, _ImdColumnGate(), global_filter=global_filter)

        assert session is not None
        assert any(f"Filter feature '{_COL_F}' matched no feature group" in r.getMessage() for r in caplog.records)

    def test_a_plan_without_its_own_extender_does_not_inherit_an_outer_probe_scope(self) -> None:
        extender = _ImdRecordingExtender()

        with reader_probe_scope(ReaderProbeScope(extender, None, None, None, None)):
            _plan("pinned", _COL_A, ImdMatchGateReader, None)

        assert extender.contexts == []


class TestProbeRaise:
    def test_probe_raise_is_unchanged_by_a_pass_through_extender(self) -> None:
        with pytest.raises(Exception) as baseline:  # noqa: B017
            _plan("pinned", _COL_D, ImdMatchRaisingProbeReader, None)
        with pytest.raises(Exception) as wrapped:  # noqa: B017
            _plan("pinned", _COL_D, ImdMatchRaisingProbeReader, _ImdRecordingExtender())

        assert type(wrapped.value) is type(baseline.value)
        assert str(wrapped.value) == str(baseline.value)
        assert "imdmatch probe boom" in str(baseline.value)
