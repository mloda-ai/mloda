"""ComputeFramework's connection resolution seam: ensure_connection()'s resolution order,
open_connection() defaults, __getstate__ dropping the live handle, convert_flight_server_data_back
and run_calculation() resolving through ensure_connection(), and pick_connection_from_dac()
against a ConnectionSpec.
"""

from __future__ import annotations

import inspect
import pickle  # nosec B403
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest

from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource, ConnectionSpec
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, Options, ParallelizationMode


class _OpenedConnection:
    """Picklable sentinel returned by open_connection, carrying the opening spec's params."""

    def __init__(self, params: dict[str, Any] | None) -> None:
        self.params = params

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OpenedConnection) and self.params == other.params


class _PicklableLive:
    """Picklable stand-in for a live connection handle."""


class _ConnRecordingFramework(ComputeFramework):
    """Records every set_framework_connection_object binding, rejects None; open_connection returns a fresh sentinel carrying spec.params, or a default sentinel for None."""

    open_connection_specs: ClassVar[list[ConnectionSpec | None]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bind_calls: list[Any] = []

    def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
        if framework_connection_object is None:
            raise ValueError(f"{type(self).__name__} cannot bind a None connection.")
        self.bind_calls.append(framework_connection_object)
        self.framework_connection_object = framework_connection_object

    @classmethod
    def open_connection(cls, spec: ConnectionSpec | None) -> Any | None:
        cls.open_connection_specs.append(spec)
        if spec is None:
            return _OpenedConnection(None)
        return _OpenedConnection(dict(spec.params))


class _NoOpenFramework(ComputeFramework):
    """set_framework_connection_object records binds; open_connection never has anything to open."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bind_calls: list[Any] = []

    def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
        if framework_connection_object is None:
            raise ValueError(f"{type(self).__name__} cannot bind a None connection.")
        self.bind_calls.append(framework_connection_object)
        self.framework_connection_object = framework_connection_object

    @classmethod
    def open_connection(cls, spec: ConnectionSpec | None) -> Any | None:
        return None


class TestFreshInstanceHasNoConnection:
    def test_connection_source_and_framework_connection_object_are_none(self) -> None:
        fw = _ConnRecordingFramework()
        assert fw.connection_source is None
        assert fw.framework_connection_object is None


class TestEnsureConnectionWithLiveCandidate:
    def test_binds_and_returns_the_live_object(self) -> None:
        fw = _ConnRecordingFramework()
        live = object()
        result = fw.ensure_connection(live)
        assert result is live
        assert fw.framework_connection_object is live
        assert fw.bind_calls == [live]


class TestEnsureConnectionWithSpecCandidate:
    def test_opens_once_and_caches_on_repeat_calls(self) -> None:
        _ConnRecordingFramework.open_connection_specs = []
        fw = _ConnRecordingFramework()
        spec = ConnectionSpec(_ConnRecordingFramework, database="x")

        first = fw.ensure_connection(spec)
        second = fw.ensure_connection(spec)

        assert first is second
        assert isinstance(first, _OpenedConnection)
        assert first.params == {"database": "x"}
        assert _ConnRecordingFramework.open_connection_specs == [spec]


class TestEnsureConnectionIgnoresLaterSpecOnceBound:
    def test_bound_framework_returns_bound_object_and_skips_open(self) -> None:
        _ConnRecordingFramework.open_connection_specs = []
        fw = _ConnRecordingFramework()
        live = object()
        fw.ensure_connection(live)

        spec = ConnectionSpec(_ConnRecordingFramework, database="ignored")
        result = fw.ensure_connection(spec)

        assert result is live
        assert _ConnRecordingFramework.open_connection_specs == []


class TestEnsureConnectionFromConnectionSourceLive:
    def test_binds_the_source_live_object(self) -> None:
        fw = _ConnRecordingFramework()
        live = object()
        fw.connection_source = ConnectionSource(live=live)

        result = fw.ensure_connection()

        assert result is live
        assert fw.framework_connection_object is live


class TestEnsureConnectionFromConnectionSourceSpec:
    def test_opens_the_source_spec(self) -> None:
        _ConnRecordingFramework.open_connection_specs = []
        fw = _ConnRecordingFramework()
        spec = ConnectionSpec(_ConnRecordingFramework, database="y")
        fw.connection_source = ConnectionSource(spec=spec)

        result = fw.ensure_connection()

        assert isinstance(result, _OpenedConnection)
        assert result.params == {"database": "y"}
        assert _ConnRecordingFramework.open_connection_specs == [spec]


class TestEnsureConnectionRaisesWhenLiveDroppedAcrossProcessBoundary:
    def test_raises_with_the_worker_process_guidance_message(self) -> None:
        live = _PicklableLive()
        source = ConnectionSource(live=live)
        restored_source = pickle.loads(pickle.dumps(source))  # nosec B301

        fw = _ConnRecordingFramework()
        fw.connection_source = restored_source

        with pytest.raises(ValueError) as excinfo:
            fw.ensure_connection()

        message = str(excinfo.value)
        assert "cannot cross into this worker process" in message
        assert "ConnectionSpec" in message


class TestEnsureConnectionWithNothingCallsOpenConnectionNone:
    def test_default_sentinel_subclass_binds_the_default(self) -> None:
        _ConnRecordingFramework.open_connection_specs = []
        fw = _ConnRecordingFramework()

        result = fw.ensure_connection()

        assert isinstance(result, _OpenedConnection)
        assert result.params is None
        assert fw.framework_connection_object is result
        assert _ConnRecordingFramework.open_connection_specs == [None]

    def test_returns_none_subclass_stays_unbound(self) -> None:
        fw = _NoOpenFramework()

        result = fw.ensure_connection()

        assert result is None
        assert fw.framework_connection_object is None
        assert fw.bind_calls == []


class TestBaseOpenConnectionReturnsNone:
    def test_none_spec(self) -> None:
        assert ComputeFramework.open_connection(None) is None

    def test_a_spec(self) -> None:
        spec = ConnectionSpec(ComputeFramework, database="z")
        assert ComputeFramework.open_connection(spec) is None


class TestGetstateDropsFrameworkConnectionObject:
    def test_bound_live_handle_is_dropped_other_state_survives(self) -> None:
        fw = _ConnRecordingFramework()
        spec = ConnectionSpec(_ConnRecordingFramework, database="p")
        fw.connection_source = ConnectionSource(spec=spec)
        fw.ensure_connection(object())

        restored = pickle.loads(pickle.dumps(fw))  # nosec B301

        assert restored.framework_connection_object is None
        assert restored.uuid == fw.uuid
        assert restored.connection_source is not None
        assert restored.connection_source.spec == spec


class TestComputeFrameworkGetstateIsFinal:
    def test_getstate_is_marked_final(self) -> None:
        assert getattr(ComputeFramework.__getstate__, "__final__", False) is True


class TestConvertFlightServerDataBackIsAnInstanceMethod:
    def test_is_a_plain_function_not_a_classmethod(self) -> None:
        assert inspect.isfunction(ComputeFramework.convert_flight_server_data_back)
        raw = inspect.getattr_static(ComputeFramework, "convert_flight_server_data_back")
        assert not isinstance(raw, classmethod)


class TestConvertFlightServerDataBackPassesEnsuredConnection:
    def test_transformer_receives_the_ensured_connection_not_none(self) -> None:
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
        from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
            ComputeFrameworkTransformer,
        )

        class _Dummy:
            pass

        class _RecordingTransformer(BaseTransformer):
            recorded_connection: ClassVar[list[Any]] = []

            @classmethod
            def framework(cls) -> Any:
                return _Dummy

            @classmethod
            def other_framework(cls) -> Any:
                return pa.Table

            @classmethod
            def import_fw(cls) -> None:
                return None

            @classmethod
            def import_other_fw(cls) -> None:
                return None

            @classmethod
            def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
                cls.recorded_connection.append(framework_connection_object)
                return _Dummy()

        class _TransformNeedingFramework(ComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> Any:
                return _Dummy

            def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
                self.framework_connection_object = framework_connection_object

        transformer = ComputeFrameworkTransformer()
        transformer.transformer_map = {(pa.Table, _Dummy): _RecordingTransformer}

        fw = _TransformNeedingFramework()
        sentinel_connection = object()
        fw.connection_source = ConnectionSource(live=sentinel_connection)

        data = pa.table({"a": [1]})
        fw.convert_flight_server_data_back(data, transformer)

        assert _RecordingTransformer.recorded_connection == [sentinel_connection]


class _EnsureConnectionRunCalcFramework(ComputeFramework):
    """expected_data_framework differs from calculate_feature's dict output, forcing the transform branch."""

    @classmethod
    def expected_data_framework(cls) -> Any:
        return list

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.keys())


class _EnsureConnectionFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"a": [1, 2]}


class TestRunCalculationResolvesConnectionViaEnsureConnection:
    def test_run_calculation_calls_ensure_connection_with_the_options_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feature_set = FeatureSet([Feature("my_feature")])
        cfw = _EnsureConnectionRunCalcFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        mock_ensure_connection = Mock(return_value=None)
        monkeypatch.setattr(cfw, "ensure_connection", mock_ensure_connection)

        cfw.run_calculation(_EnsureConnectionFeatureGroup, feature_set, location=None)

        expected = feature_set.get_options_key(_EnsureConnectionFeatureGroup.get_class_name())
        mock_ensure_connection.assert_called_once_with(expected)


class _PickLiveMarker:
    """Marker type _PickFramework._connection_matches recognizes as a live connection."""


class _PickFramework(ComputeFramework):
    @classmethod
    def _connection_matches(cls, conn: Any) -> bool:
        return isinstance(conn, _PickLiveMarker)


class _UnrelatedPickFramework(ComputeFramework):
    pass


class TestPickConnectionFromDacResolvesConnectionSpec:
    def test_spec_resolves_for_the_matching_framework(self) -> None:
        spec = ConnectionSpec(_PickFramework, database="x")
        dac = DataAccessCollection(connections={spec})
        assert _PickFramework.pick_connection_from_dac(dac) is spec

    def test_spec_does_not_resolve_for_an_unrelated_framework(self) -> None:
        spec = ConnectionSpec(_PickFramework, database="x")
        dac = DataAccessCollection(connections={spec})
        assert _UnrelatedPickFramework.pick_connection_from_dac(dac) is None


class TestPickConnectionFromDacAmbiguity:
    def test_matching_live_and_matching_spec_raise_ambiguous_resolve(self) -> None:
        live = _PickLiveMarker()
        spec = ConnectionSpec(_PickFramework, database="x")
        dac = DataAccessCollection(connections={"live": live, "spec": spec})

        with pytest.raises(ValueError, match="Ambiguous resolve"):
            _PickFramework.pick_connection_from_dac(dac)

    def test_hint_selects_the_spec_entry(self) -> None:
        live = _PickLiveMarker()
        spec = ConnectionSpec(_PickFramework, database="x")
        dac = DataAccessCollection(connections={"live": live, "spec": spec})
        options = Options(context={"data_access_handle": "spec"})

        result = _PickFramework.pick_connection_from_dac(dac, options=options)

        assert result is spec
